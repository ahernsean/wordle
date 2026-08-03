"""erd_swarm.py — Branch-swarm worker for the ERD_ALL precache.

The precache is built by workers cooperating on one branch at a time.  A
*branch* is a position to solve: the answer words left after a guess+response
(e.g. SALET ----- = 315 words).  To solve a branch we evaluate ~12,972
*candidate* guesses against it and keep the lowest-ERD one.  Each worker
claims one candidate at a time (a *claim*) by index into the policy-canonical
word list, so several workers pour into the same branch at once while sharing
a single running-best ERD as a branch-and-bound bound.

Trust model (see erd_queue.py): a claimed candidate is advisory; only a
done=1 claim is authoritative.  A branch is finalized — its ERD written to
the persistent cache — only once every candidate is done, by whichever worker
observes full coverage.  So a crashed worker's candidate is redone, never
skipped.
"""

from __future__ import annotations

import collections
import faulthandler
import logging
import math
import os
import random
import signal
import sqlite3
import time

import pattern_matrix as pattern_matrix_module
from cache_sqlite import ScoreCache, mem_cache_limit
from wordle_engine import (
    ERD_ALL,
    GAME_GUESSES,
    ResponseCache,
    CANCEL_RECVD,
    SOLVED,
    OVER_DEPTH_BUDGET,
    OVER_ERD_LIMIT,
    NO_INFORMATION_GAINED,
    _ABORT_STATUSES,
    erd_ge,
    cache_all_scores,
    evaluate_candidate,
    estimate_candidate_work,
    load_word_list,
    _cache_reuse,
)
from erd_queue import (ERDQueue, decode_subset, encode_subset,
                       guess_depth_from_spine, disk_stats,
                       DISK_STOP_FRACTION,
                       QUEUE_WAL_HARD_CEILING_BYTES,
                       DEFAULT_SMALL_COUNT, DEFAULT_COUNT_CAP,
                       DEFAULT_REPUBLISH_LIMIT)
from wordle_ui import fmt_pattern

from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    worker_log_path,
)

ANSWER_FILE = DEFAULT_ANSWER_LIST_PATH
WORDS_FILE = DEFAULT_CANDIDATE_LIST_PATH

BEST_REFRESH_SECONDS = 0.25   # how often a worker re-reads the shared bound
HB_SECONDS = 2.0              # liveness heartbeat cadence during a long candidate evaluation
# A worker that hasn't heartbeat within this many seconds is presumed dead, and
# only then are its in-flight candidate claims reclaimed.  Live workers heartbeat
# every HB_SECONDS regardless of how long a single candidate takes (the
# heartbeat fires on every recursive sub-branch call, not just between
# candidates), so a slow-but-alive worker is never reclaimed — only a crashed
# or OOM-killed process whose heartbeat truly stops.  30s = 15 missed
# heartbeats, which is conservative enough for any real process death.
HB_TIMEOUT_SECONDS = 30
CHECKPOINT_SECONDS = 300      # WAL checkpoint interval (5 min)
# Cadence for the per-table WAL traffic log.  Deliberately far shorter than the
# checkpoint interval: at the runaway rates this diagnostic exists to catch, the
# WAL crosses the hard ceiling (and latches the swarm down) in ~100 s — long
# before a 5-minute checkpoint — so the attribution has to be emitted on its own
# fast timer, and from the heartbeat path so it fires mid-evaluation too.
WAL_TRAFFIC_LOG_SECONDS = 30
# Workers space their checkpoint intervals by a per-worker random factor in
# [1-JITTER, 1+JITTER] so six processes spawned within seconds of each other
# don't attempt checkpoints in lockstep every cycle.
CHECKPOINT_JITTER = 0.25
# Poll cadence while honouring the supervisor's checkpoint_pause flag at a
# claim boundary (workers stay off the queue database between polls).
PAUSE_POLL_SECONDS = 2.0
DISK_CHECK_SECONDS = 30       # disk fullness check throttle (workers)
# Worker-side WAL hard-ceiling check throttle.  The check is one getsize()
# call; the throttle keeps it out of per-node and per-iteration hot paths.
WAL_CEILING_CHECK_SECONDS = 5.0
# A branch still status='finalized' this long after finalized_at has a dead
# finalizer (a live one completes the cache write + delete in milliseconds)
# and is reopened by a waiting sibling.
FINALIZE_TAKEOVER_SECONDS = 60
PROGRESS_LOG_SECONDS = 120   # log a mid-candidate progress line this often
RAM_WARN_MB = 1024            # log warning when free RAM drops below this
RAM_CRIT_MB = 512             # force checkpoint when free RAM drops below this

PROMOTE_MIN_SIZE = 60   # cold-model fallback: sub-branches with >= this many
                        # words are promoted cooperatively when the cost model
                        # has no data for their size bucket.
PROMOTED_PRIORITY = 1_000_000  # promoted sub-branches outrank fresh top branches
                               # so freed workers prefer joining in-flight depth.

OVERRUN_K = 4            # a frame spending > K * typical(n) nodes triggers publication
# Absolute wall-clock backstop on a single inline frame, independent of the cost
# model.  When the model is cold (typical(n) is None) the node-proportionate
# overrun check above can't arm, so without this a first-of-its-size tarpit would
# grind on one worker unbounded.  A frame running longer than this hands off its
# remainder regardless of model state.  This value is the *subdivision granularity*
# of a cold fan-out, not just the first worker's wait: every level of the recursive
# cascade fractures only after another full interval of solo grind, so a larger
# value trades slower ramp-to-parallel for less coordination churn.  backstop_telemetry
# records every firing for offline tuning.
COLD_BACKSTOP_SECONDS = 600
MIN_HANDOFF_CANDIDATES = 4   # minimum remaining candidates to bother handing off

# candidate_accuracy logs every non-ERD-pruned claim (the metric-design signal,
# rare in the deep regime) but only 1-in-N ERD-pruned claims (~1 node each,
# redundant) so a multi-day corpus stays bounded.  1 = log all (the
# validation-gate default); raise it (e.g.
# ERD_LOWER_BOUND_PRUNED_SAMPLE_EVERY=100) for a long production run.
ERD_LOWER_BOUND_PRUNED_SAMPLE_EVERY = max(
    1, int(os.environ.get('ERD_LOWER_BOUND_PRUNED_SAMPLE_EVERY', '1')))
MIN_PUBLISH_BRANCH_WORDS = 2  # frames with fewer answer words are base cases, never
                              # worth tracking for overrun (the candidate loop on a
                              # 1-word branch never even runs)

# Adaptive publish threshold (node-equivalents): SAFETY_FACTOR * coordination_time
# / node_time.  Publishing pays only when the handed-off work exceeds the cost of
# coordinating the handoff; both terms are measured live per worker (log-domain
# EMAs, same TAU as the cost model).  The two are a ratio, so their unit cancels —
# they're kept in seconds (what time.time() gives, no false nanosecond precision).
# Until those estimators warm, fall back to PUBLISH_THRESHOLD_BOOTSTRAP.
SAFETY_FACTOR = 8               # dimensionless margin on the coordination break-even
_PUBLISH_EMA_TAU = 86400.0      # half-life (s) for the coordination/node-time EMAs
_PUBLISH_EMA_MIN_WEIGHT = 5     # decayed samples before the adaptive threshold goes live
PUBLISH_THRESHOLD_BOOTSTRAP = 5000  # cold-start prior until the EMAs warm

# Budget at the root — before any guess is played.  A branch's remaining budget
# is ROOT_BUDGET minus its guess_depth (the guesses already played to reach it),
# so a queued position after the opener (guess_depth 1) is solved at ROOT_BUDGET
# - 1.  Depth-limited ERD: a branch unsolvable within its budget is a loss.
ROOT_BUDGET = GAME_GUESSES

# Binary claim packing (issue #67, adaptive_claim_packing.md).  small_count
# and count_cap default to erd_queue's DEFAULT_* (see there for the
# survivor-bundle amortization reasoning behind 8 and the compatibility cap).
#
# bundle_node_cap bounds how much genuinely-heavy search a small bundle's
# strong-splitter head can accumulate before the rest of the bundle is
# handed back for re-packing against a tighter B (§7a cross-candidate
# overrun) — large enough that ordinary multi-thousand-node candidates
# finish without a spurious republish, small enough that a real heavy
# candidate hands off its siblings within seconds rather than stalling a
# whole survivor bundle.
BUNDLE_NODE_CAP = int(os.environ.get('BUNDLE_NODE_CAP', '50000'))
# bundle_wall_cap_seconds bounds the crash-reclaim window
# (HB_TIMEOUT_SECONDS) a dead worker's held survivor bundle would otherwise
# widen.
BUNDLE_WALL_CAP_SECONDS = float(os.environ.get('BUNDLE_WALL_CAP_SECONDS', '60'))
# republish_limit: how many times the cross-candidate mechanism may bounce
# the same candidate before it is evaluated "forced" — exempt from the
# bundle caps so its own within-candidate sub-branch promotion (always
# active) absorbs any real depth instead of the candidate thrashing the
# pool indefinitely.
BUNDLE_REPUBLISH_LIMIT = int(
    os.environ.get('BUNDLE_REPUBLISH_LIMIT', str(DEFAULT_REPUBLISH_LIMIT)))
BUNDLE_SMALL_COUNT = int(
    os.environ.get('BUNDLE_SMALL_COUNT', str(DEFAULT_SMALL_COUNT)))
BUNDLE_COUNT_CAP = int(os.environ.get('BUNDLE_COUNT_CAP', str(DEFAULT_COUNT_CAP)))

logger = logging.getLogger('wordle')


class _LogEMA:
    """Continuous-time log-domain EMA — a streaming geometric mean with half-life
    TAU.  A sample enters as ln(value), so one heavy-tailed outlier shifts the
    estimate far less than it would an arithmetic mean.  value() returns None
    until at least min_weight of (time-decayed) samples have accumulated.
    """

    __slots__ = ('_tau', '_min_weight', '_log_sum', '_weight', '_last')

    def __init__(self, tau=_PUBLISH_EMA_TAU, min_weight=_PUBLISH_EMA_MIN_WEIGHT):
        self._tau = tau
        self._min_weight = min_weight
        self._log_sum = 0.0
        self._weight = 0.0
        self._last = None

    def add(self, value, now=None):
        if value <= 0:
            return
        if now is None:
            now = time.time()
        if self._last is not None:
            decay = math.exp(-max(0.0, now - self._last) / self._tau)
            self._log_sum *= decay
            self._weight *= decay
        self._log_sum += math.log(value)
        self._weight += 1.0
        self._last = now

    def value(self):
        if self._weight < self._min_weight:
            return None
        return math.exp(self._log_sum / self._weight)


class _MidLoopPublisher:
    """Engine-seam object that detects mid-loop overrun and publishes the remainder.

    One instance lives on the worker for its lifetime and is passed down through
    evaluate_candidate / _solve_subset exactly like subbranch_solver.  It holds
    no per-frame mutable state — each active DFS frame creates an independent
    frame-local token via enter() — so a single instance serves every claim.

    When check() fires at frame F:
    - The prefix candidates (evaluated inline in Σk² order) are marked done
      in the candidate_claims table by their all_words index so cooperative
      workers do not redo them.
    - The remainder is driven by cooperative_solve, which claims unclaimed
      slots in natural all_words order.
    - The finished result is returned directly; the engine short-circuits the
      rest of frame F (no re-cache, because cooperative_solve already wrote it).
    """

    def __init__(self, worker):
        self._worker = worker

    def enter(self, branch_words, budget):
        """Called just before the candidate loop of each _solve_subset frame.

        Returns an opaque token
        [nodes_at_entry, predicted, entry_time, branch_words, budget, armed]
        for any non-trivial frame (>= MIN_PUBLISH_BRANCH_WORDS answer words).
        predicted may be None (cold model) — the node-proportionate check then
        can't arm, but the wall-clock backstop in check() still fires off
        entry_time, and record_inline() still warms the model on frame
        completion.

        The token is mutable: check() clears the trailing `armed` flag when a
        publication is declined (unjoinable existing row, cooperative decline,
        WAL ceiling), committing the frame to finish inline.  Without the
        latch, the overrun condition — still true on every subsequent
        iteration — would re-run the whole publish sequence per candidate,
        an O(n²) write storm against a branch that can never be joined.
        """
        n = len(branch_words)
        if n < MIN_PUBLISH_BRANCH_WORDS:
            return None
        predicted = self._worker._typical(n, budget)
        return [self._worker._nodes, predicted, time.time(), branch_words,
                budget, True]

    def check(self, token, candidate_list, last_index,
              best_guess, best_erd, budget):
        """Called every loop iteration (before status-continue checks).

        candidate_list is the frame's full ordered candidate list and last_index
        is the index just evaluated, so remaining_count is derived cheaply and
        the evaluated prefix is sliced only on the rare iteration the overrun
        actually fires (avoiding an O(n²) per-iteration copy).

        Fires on either of two triggers:
        - node-proportionate: the frame has spent > OVERRUN_K * predicted nodes
          since enter() (warm model only — disabled when predicted is None),
          AND at least _publish_threshold() nodes in absolute terms (the
          break-even gate below);
        - wall-clock backstop: the frame has run longer than COLD_BACKSTOP_SECONDS
          since enter() (always armed, the only guard while the model is cold).

        When either fires and enough candidates remain to be worth handing off,
        emits the promotion sentinel, publishes the remainder as a cooperative
        branch, and returns the cooperative result so the engine can short-
        circuit.  Returns None to continue inline — either transiently (the
        trigger may fire again later in the frame) or permanently, by clearing
        the token's armed flag, when the publication was declined and every
        retry would be declined the same way.
        """
        if token is None or not token[5]:
            return None
        nodes_at_entry, predicted, entry_time, branch_words, entry_budget = \
            token[:5]
        delta = self._worker._nodes - nodes_at_entry
        node_overrun = predicted is not None and delta > OVERRUN_K * predicted
        elapsed = time.time() - entry_time
        time_overrun = elapsed > COLD_BACKSTOP_SECONDS
        if not (node_overrun or time_overrun):
            return None
        # Break-even gate: a handoff pays roughly _publish_threshold()
        # node-equivalents of coordination, so the frame must have proven at
        # least that much work in absolute terms before publishing is worth
        # it.  The proportionate trigger alone cannot be trusted for this:
        # once the score cache is warm, most frames complete in a handful of
        # nodes, the model's geometric mean sits near that cache-hit mode, and
        # OVERRUN_K * typical can be single-digit nodes — which would hand off
        # frames whose entire remainder costs less than the coordination to
        # share them.  The wall-clock backstop is exempt: minutes of wall time
        # prove the frame expensive regardless of its node count.
        if not time_overrun and delta < self._worker._publish_threshold():
            return None
        remaining_count = len(candidate_list) - (last_index + 1)
        if remaining_count < MIN_HANDOFF_CANDIDATES:
            return None
        # A worker that must stop writing publishes nothing: publication is
        # pure queue-write traffic.
        if self._worker.cancel() or self._worker._wal_ceiling_tripped():
            token[5] = False
            return None

        n = len(branch_words)
        branch_key = encode_subset(branch_words)

        # A frame with no achieved best whose bound is finite is still riding
        # its entry alpha-beta ceiling (best_erd is only ever lowered from the
        # ceiling by a real SOLVED candidate).  Its prefix candidates were
        # priced out against that ceiling, so the handoff is sound only if the
        # published branch carries the ceiling too — its finalize is then a CUT
        # (never cached) unless someone lands below it.  A user-queued branch
        # must stay exact, so the ceiling is suppressed for it (and the prefix
        # marks are skipped below: their price-outs do not hold in an exact solve).
        frame_ceilinged = (best_guess is None and best_erd is not None
                           and best_erd != float('inf'))
        pending = frame_ceilinged and self._worker.queue.has_pending_row(branch_key)
        branch_ceiling = best_erd if (frame_ceilinged and not pending) else None

        # Create the cooperative branch; idempotent if another worker raced us.
        # First writer for this path, so the composed spine must be supplied here
        # (cooperative_solve's later create_branch is a no-op once this row exists).
        created = self._worker.queue.create_branch(
            branch_key, n, self._worker.n_candidates,
            priority=PROMOTED_PRIORITY,
            source_word=self._worker._top_source_word,
            source_pattern=self._worker._top_source_pattern,
            budget=budget,
            spine=self._worker._promoted_spine(
                self._worker.root_budget - budget),
            root_budget=self._worker.root_budget,
            ceiling=branch_ceiling)

        if created:
            row_budget, row_ceiling = budget, branch_ceiling
        else:
            row = self._worker.queue.get_branch(branch_key)
            row_budget = row['budget'] if row is not None else None
            row_ceiling = row['ceiling'] if row is not None else None

        # Joinability, decided BEFORE any write against the row and mirroring
        # cooperative_solve's own decline rules: the budget must match exactly,
        # and the row's ceiling must be NULL (exact) or >= ours.  A row that
        # fails either test can never serve this frame; publishing to it would
        # pour prefix marks into a branch this frame will not join — and since
        # the overrun condition stays true on every later iteration, it would
        # do so per candidate, forever.  Disarm the token and finish inline.
        ours = best_erd if frame_ceilinged else None
        joinable = row_budget == budget and (
            row_ceiling is None or (ours is not None and erd_ge(row_ceiling, ours, n)))
        if not joinable:
            token[5] = False
            return None

        # Record every wall-clock backstop firing so COLD_BACKSTOP_SECONDS can be
        # tuned offline; the node-proportionate path is the model working as
        # intended and isn't what we're tuning.
        if time_overrun:
            self._worker.queue.add_backstop_telemetry(
                n, entry_budget, int(elapsed * 1000), delta, predicted, remaining_count)
        # This frame is handed off before finishing, so its true cost exceeds the
        # `delta` nodes measured so far: record a right-censored sample (a lower
        # bound).  The online cost model only folds COMPLETED frames, so a partial
        # never pollutes it; this keeps a capped monster honestly visible to an
        # offline survival fit instead of vanishing.
        self._worker.queue.add_cost_sample(
            ERD_ALL, n, delta, 'censored', budget=entry_budget, censored=1)

        # Spine sentinel: mark this frame as handed off in the heartbeat display.
        self._worker._note_depth(entry_budget, -n, None, None)

        # Mark the already-evaluated candidates done by their all_words index so
        # cooperative workers claim only the unevaluated remainder.  The prefix
        # slice is built here — only when an overrun actually fires — not on
        # every loop iteration.
        #
        # Everything the prefix proved, it proved at THIS frame's budget (the
        # row's budget matches — joinability above).  Soundness of the marks
        # then depends on what the prefix was priced against:
        # - an achieved best (best_guess set): sound — the seed below caps the
        #   branch's final value at that best, and every marked candidate is
        #   proven >= it;
        # - nothing (best_erd == inf): the prefix was exhausted or infeasible,
        #   which no bound can manufacture — sound anywhere;
        # - the entry ceiling: sound only on a branch whose own ceiling is <=
        #   ours (its outcomes then never contradict a >= ours price-out).  A
        #   racing creator may have won with a looser or absent ceiling — skip
        #   the marks then and let the branch redo the prefix.
        if frame_ceilinged:
            sound_to_mark = row_ceiling is not None and erd_ge(best_erd, row_ceiling, n)
        else:
            sound_to_mark = True
        if sound_to_mark:
            word_idx = self._worker._word_idx
            done_indices = [word_idx[w] for w in candidate_list[:last_index + 1]
                            if w in word_idx]
            if done_indices:
                self._worker.queue.mark_claims_done(branch_key, done_indices)

        # Seed the cooperative branch's bound only when we have an achieved cost
        # (a None best_guess means no feasible candidate yet — the entry
        # ceiling, if any, rides on the branch's ceiling column instead).
        if best_guess is not None:
            self._worker.queue.update_branch_best(branch_key, best_guess, best_erd)

        result = self._worker.cooperative_solve(
            branch_words, budget,
            ceiling=best_erd if frame_ceilinged else float('inf'))
        if result is None:
            # cooperative_solve re-checked joinability and declined: a racing
            # writer changed the row between our read and its own.  The frame
            # solves inline from here on.
            token[5] = False
        return result

    def record_inline(self, token):
        """Called on the SOLVED return of each completed _solve_subset frame.

        Accumulates node-cost samples in the worker's in-memory buffer keyed by
        sub-branch size, carrying both Σ ln(nodes) and Σ ln²(nodes) so the batch
        flush reaches the cost model's second log-moment faithfully.  The buffer
        is flushed to the DB at each checkpoint so this never touches SQLite
        mid-candidate-loop.
        """
        if token is None:
            return
        nodes_at_entry, _predicted, _entry_time, branch_words, entry_budget = \
            token[:5]
        n = len(branch_words)
        nodes = self._worker._nodes - nodes_at_entry
        if nodes <= 0:
            return
        log_n = math.log(nodes)
        buf = self._worker._cost_model_buffer
        key = (n, entry_budget)
        if key in buf:
            s, sq, c = buf[key]
            buf[key] = (s + log_n, sq + log_n * log_n, c + 1)
        else:
            buf[key] = (log_n, log_n * log_n, 1)


class _BranchWorker:
    """One worker process's state and operations on branches and candidates."""

    def __init__(self, worker_id, cache_path, queue_path, stop_event,
                 root_budget=ROOT_BUDGET, n_workers=1,
                 enable_adaptive_decomposition=True,
                 small_count=BUNDLE_SMALL_COUNT, count_cap=BUNDLE_COUNT_CAP,
                 bundle_node_cap=BUNDLE_NODE_CAP,
                 bundle_wall_cap_seconds=BUNDLE_WALL_CAP_SECONDS,
                 republish_limit=BUNDLE_REPUBLISH_LIMIT):
        self.name = f'worker-{worker_id}'
        self.stop_event = stop_event
        # Process-local stop request, set by this worker's own SIGTERM/SIGINT
        # handler.  Separate from the shared stop_event so terminating one worker
        # (e.g. a recycle) does not signal the whole pool.
        self._stop_requested = False
        self.root_budget = root_budget
        self.n_workers = n_workers
        # Binary claim packing dials (see the BUNDLE_* module constants).
        self.small_count = small_count
        self.count_cap = count_cap
        self.bundle_node_cap = bundle_node_cap
        self.bundle_wall_cap_seconds = bundle_wall_cap_seconds
        self.republish_limit = republish_limit
        # Per-branch best-first order + cost_lower_bound array
        # (_packing_stats), cached for the life of this worker process and
        # evicted when the branch finalizes.
        self._packing_stats_cache = {}
        # The adaptive-decomposition layer — cost model, entry-gate publish
        # threshold, and the mid-loop overrun escape hatch with its wall-clock
        # backstop.
        # When disabled the worker runs the bare claim/evaluate/finalize loop with
        # plain size-based promotion: pure candidate-partition parallelism, which
        # is what the strong-scaling test measures.
        self._adaptive = enable_adaptive_decomposition

        self.all_answers = load_word_list(ANSWER_FILE)
        self.all_words = load_word_list(WORDS_FILE)
        self.n_candidates = len(self.all_words)
        max_entries = mem_cache_limit(n_workers)
        logger.info('%s mem_cache cap: %d entries (~%.0f MB)',
                    self.name, max_entries, max_entries * 250 / 1e6)
        self.score_cache = ScoreCache(cache_path, self.all_answers,
                                      max_mem_entries=max_entries)
        self.rcache = ResponseCache(self.all_answers, self.score_cache)
        self.pattern_matrix = pattern_matrix_module.PatternMatrix.load_or_build(
            cache_path, self.all_words, self.all_answers, self.score_cache)
        self.queue = ERDQueue(queue_path)

        self.started = int(time.time())
        self.claims_done = 0
        self.n_ok = 0
        self.n_cutoff = 0    # cost >= best_erd before full eval (alpha-beta)
        self.n_pruned = 0    # infeasible within budget (depth floor hit)
        self.n_useless = 0
        self._last_hb = 0.0
        self._last_checkpoint = time.time()
        self._checkpoint_interval = CHECKPOINT_SECONDS * random.uniform(
            1 - CHECKPOINT_JITTER, 1 + CHECKPOINT_JITTER)
        self._last_ram_check = time.time()
        self._last_disk_check = time.time()
        self._last_pause_check = 0.0
        self._pause_active = False
        self._last_wal_ceiling_check = 0.0
        self._wal_ceiling_hit = False
        self._cand_max_depth = 0     # deepest guess_depth reached this candidate
        # The candidate under evaluation right now, or None when no work is in
        # flight (idle, or coordinating without a candidate).  A heartbeat
        # reports this; it is not re-supplied per call.
        self._cur_candidate = None
        # Live search probe (transparency): a monotonic node counter plus the
        # active descent spine, so a long candidate evaluation never looks
        # frozen — node count climbs every heartbeat even mid-candidate.
        self._nodes = 0              # candidate evaluations since last claim start
        self._nodes_at_last_hb = 0
        self._cur_depth = 0
        self._spine = {}             # guess_depth -> subset size on the active descent
        self._hb_max_spine = {}      # deepest spine since last heartbeat (→ DB)
        self._log_max_spine = {}     # deepest spine since last progress log (→ log file)
        self._last_progress_log = 0.0
        # Utilisation accounting: cumulative wall time spent inside candidate
        # evaluation (useful work) vs everything else the worker does between
        # evaluations (claiming, waiting for coverage, finalizing, helping a
        # peer).  Logged periodically so coordination overhead is measurable.
        self._eval_seconds = 0.0
        # Seeded to construction time so the first utilisation sample waits a full
        # interval rather than firing immediately against ~0s of elapsed work.
        self._last_util_log = time.time()
        # Previous WAL-traffic snapshot + its time, so the periodic log can
        # report this worker's per-table write rate into the shared queue WAL
        # (which table is the firehose) rather than an unbounded running total.
        self._last_wal_traffic = self.queue.wal_traffic_snapshot()
        self._last_wal_traffic_log = time.time()
        # Attribution for promoted sub-branches: which top-level (opener,pattern)
        # tree the worker is currently descending.
        self._top_source_word = None
        self._top_source_pattern = None
        # Counts ERD-pruned candidate_accuracy claims for 1-in-N down-sampling.
        self._erd_lower_bound_pruned_accuracy_n = 0
        # Absolute root -> branch spine of the branch the worker is currently
        # descending (its claimed branch): the guesses played, space-joined as
        # "GUESS pattern".  Promotion composes a child branch's spine as this base
        # plus the live descent guesses in self._spine.  None until the first claim.
        self._claimed_branch_spine = None
        # In-memory cache of cost-model predictions keyed by sub-branch size.
        # Cleared on any cost-model write so new samples take effect.
        self._typical_cache = {}
        # In-memory buffer for inline node-cost samples:
        # {(n, budget): (sum_log, sum_log_sq, count)}.  Flushed to the DB at each
        # checkpoint to avoid per-frame SQLite writes.
        self._cost_model_buffer: dict[tuple[int, int],
                                      tuple[float, float, int]] = {}
        # Reverse map word → all_words index, built once: the publisher marks the
        # evaluated prefix done by these indices.  Both are needed only by the
        # adaptive layer, so they're skipped (and the publisher is None, which
        # disarms the overrun check) when adaptive decomposition is off.
        if self._adaptive:
            self._word_idx = {w: i for i, w in enumerate(self.all_words)}
            self._mid_loop_publisher = _MidLoopPublisher(self)
        else:
            self._word_idx = None
            self._mid_loop_publisher = None
        # Live coordination/throughput estimators feeding the adaptive publish
        # threshold (node-equivalents); both are outbound telemetry's in-memory
        # twins — the claim_telemetry table is never read back for control.
        self._coord_ema = _LogEMA()
        self._node_time_ema = _LogEMA()
        # Completion time of the previous evaluated claim.  The outbound
        # claim_telemetry coordination figure telescopes from here, so it
        # captures claim acquisition and inter-claim overhead — matching the
        # lifetime eval%/coord% split, not just the in-evaluate_claim window.
        self._last_claim_complete = time.time()

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        # Final unthrottled flush so the last interval's attribution is captured
        # even when a hard-ceiling trip terminates the worker between periodic
        # logs (the ceiling stops the swarm via SIGTERM -> request_stop -> this).
        self._log_wal_traffic(time.time(), force=True)
        self.queue.clear_heartbeat(self.name)
        self.score_cache.checkpoint()
        self.score_cache.close()
        self.queue.checkpoint("PASSIVE")
        self.queue.close()

    def request_stop(self):
        """Ask the worker to finish its current candidate and exit cleanly.

        Called from the process's SIGTERM/SIGINT handler so the normal-exit path
        (run() returns -> close() clears the heartbeat row) runs instead of the
        process dying mid-evaluation."""
        self._stop_requested = True

    def cancel(self):
        return self._stop_requested or (
            self.stop_event is not None and self.stop_event.is_set())

    # -- depth instrumentation ----------------------------------------------

    def _note_depth(self, budget, n, guess=None, pattern=None):
        # Per-position observer: the engine reports its working `budget` at each
        # frame; we key the live descent spine by absolute guess_depth
        # (ROOT_BUDGET - budget) so deeper frames sort larger and the display is
        # an absolute position.  Each entry stores (size, guess, pattern): the
        # sub-branch size and the candidate+response that produced it.  n < 0 is
        # the cooperative-promotion sentinel: preserve the stored guess/pattern,
        # replace size with '•' to mark that the sub-branch was handed to the swarm.
        if budget is None:
            return
        guess_depth = ROOT_BUDGET - budget
        if n < 0:
            prev = self._spine.get(guess_depth, (None, None, None))
            self._spine[guess_depth] = ('•', prev[1], prev[2])
            deeper = [d for d in self._spine if d > guess_depth]
            for d in deeper:
                del self._spine[d]
            if len(self._spine) >= len(self._hb_max_spine):
                self._hb_max_spine = dict(self._spine)
            if len(self._spine) >= len(self._log_max_spine):
                self._log_max_spine = dict(self._spine)
            return
        if guess_depth > self._cand_max_depth:
            self._cand_max_depth = guess_depth
        self._cur_depth = guess_depth
        pattern_str = fmt_pattern(pattern) if isinstance(pattern, int) else pattern
        self._spine[guess_depth] = (n, guess, pattern_str)
        deeper = [d for d in self._spine if d > guess_depth]
        for d in deeper:
            del self._spine[d]
        if len(self._spine) >= len(self._hb_max_spine):
            self._hb_max_spine = dict(self._spine)
        if len(self._spine) >= len(self._log_max_spine):
            self._log_max_spine = dict(self._spine)

    @staticmethod
    def _fmt_spine_entry(entry):
        if not isinstance(entry, tuple):
            return str(entry)
        size, guess, pattern = entry
        if guess and pattern:
            return f'{guess.upper()}:{pattern}/{size}'
        return str(size)

    def _claimed_branch_guess_depth(self):
        """Guess depth of the current claimed branch.

        Entries in _hb_max_spine / _log_max_spine at this depth and shallower
        belong to the outer frame's evaluation path and must be excluded from
        the heartbeat's live-descent string.  _solve_subset fires note_depth
        before calling subbranch_solver, so the entry at the claimed branch
        level is always written by the outer frame and persists into inner
        cooperative_solve sessions until _claimed_branch_spine is updated.
        """
        spine = getattr(self, '_claimed_branch_spine', None)
        return guess_depth_from_spine(spine) if spine else 0

    def _hb_spine_str(self):
        claimed_guess_depth = self._claimed_branch_guess_depth()
        return '→'.join(
            f'{d}:{self._fmt_spine_entry(self._hb_max_spine[d])}'
            for d in sorted(self._hb_max_spine)
            if d > claimed_guess_depth)

    def _log_spine_str(self):
        claimed_guess_depth = self._claimed_branch_guess_depth()
        return '→'.join(
            f'{d}:{self._fmt_spine_entry(self._log_max_spine[d])}'
            for d in sorted(self._log_max_spine)
            if d > claimed_guess_depth)

    @staticmethod
    def _root_spine(source_word, source_pattern):
        """Single-guess spine string for a top-level branch: 'SALET -g-g-'."""
        if not source_word or source_pattern is None:
            return None
        return f'{source_word.upper()} {fmt_pattern(source_pattern)}'

    def _spine_budget(self, spine):
        """Remaining guess budget for a branch reached by the guesses on `spine`:
        ROOT_BUDGET minus its guess_depth (the number of guesses played)."""
        return self.root_budget - guess_depth_from_spine(spine)

    def _branch_budget(self, branch):
        """A branch row's solve budget: its stored `budget`, or — for a legacy
        row (or a dict lacking the column) — derived from its spine.  Accepts
        either a sqlite3.Row or a dict; both support `.keys()` and `[]`."""
        keys = branch.keys()
        stored = branch['budget'] if 'budget' in keys else None
        if stored:
            return stored
        return self._spine_budget(branch['spine'] if 'spine' in keys else None)

    def _promoted_spine(self, max_guess_depth=None):
        """Absolute root -> promoted-branch spine: the claimed branch's base plus
        the live descent guesses (guess_depth-ordered "GUESS pattern" tokens).
        Returns None when the base is unknown, leaving the branch row to fall back
        to the source word.  Sentinel/size-only spine entries (no guess) are skipped.

        max_guess_depth caps the composed spine at the promoted branch's own
        guess depth (root_budget - its budget).  The live descent map keeps
        entries from deeper frames the engine has already unwound out of — a
        candidate loop that recursed to depth d and returned still shows the
        depth-d edge — so without the cap a promotion at a shallower frame
        composes a spine longer than its budget allows, which create_branch's
        budget + guess_depth = root_budget invariant rejects.
        """
        base = getattr(self, '_claimed_branch_spine', None)
        if not base:
            return None
        # Entries at or shallower than the claimed branch's guess depth are
        # never edges of the promoted branch: the base spine already carries
        # the full path to the claimed branch.  They can be stale because the
        # live descent persists across claim boundaries and a claim's top frame
        # is not reported, so only entries strictly below the boundary belong
        # to the current descent.
        edges = base.split()   # flat "GUESS pattern GUESS pattern ..." tokens
        base_guess_depth = guess_depth_from_spine(base)
        for guess_depth in sorted(getattr(self, '_spine', {})):
            if guess_depth <= base_guess_depth:
                continue
            if max_guess_depth is not None and guess_depth > max_guess_depth:
                break
            _size, guess, pattern = self._spine[guess_depth]
            if not (guess and guess != '•' and pattern):
                continue
            guess = guess.upper()
            edges.extend((guess, pattern))
        return ' '.join(edges)

    # -- cost model ---------------------------------------------------------

    def _typical(self, n, budget):
        """Return the cost model's geometric-mean node count for sub-branches of
        size n at remaining-guess `budget`, or None when the model is cold.

        The model is keyed on (size, budget); a cold (size, budget) cell reads
        cold, with no cross-budget fallback.  Results are cached in-memory
        keyed by (n, budget) for the life of the worker; entries are
        invalidated on cooperative finalize so new samples take effect without
        re-querying on every enter() call.
        """
        key = (n, budget)
        if key in self._typical_cache:
            return self._typical_cache[key]
        result = self.queue.get_cost_typical(ERD_ALL, n, budget)
        self._typical_cache[key] = result
        return result

    def _update_cost_model(self, n_words, nodes, budget, wall_millis=None):
        """Update the cost model with a finalized cooperative branch's node count.

        budget keys the (size, budget) cell; wall_millis is the branch's wall
        span, the only per-solve wall figure, recorded on the raw sample.
        """
        self.queue.update_cost_model(ERD_ALL, n_words, nodes, budget=budget)
        self.queue.add_cost_sample(ERD_ALL, n_words, nodes, 'finalize',
                                   budget=budget, wall_millis=wall_millis)
        self._typical_cache.clear()   # bucket changed: drop cached predictions

    def _flush_cost_model_buffer(self):
        """Flush the in-memory inline-sample buffer to the DB and clear it.

        The buffered (Σ ln, Σ ln², count) accumulators are folded straight into
        the cost model, so each sample's magnitude reaches the geometric mean and
        the second log-moment without an exp/int/log round-trip.
        """
        for (n, budget), (sum_log, sum_log_sq, count) in \
                self._cost_model_buffer.items():
            self.queue.update_cost_model_logsums(
                ERD_ALL, n, sum_log, sum_log_sq, float(count), budget=budget)
        if self._cost_model_buffer:
            self._typical_cache.clear()
        self._cost_model_buffer.clear()

    def _publish_threshold(self):
        """Adaptive 'worth-swarming' break-even in node-equivalents:
        SAFETY_FACTOR * coordination_time / node_time (both in seconds, so the
        unit cancels).  Falls back to PUBLISH_THRESHOLD_BOOTSTRAP until both live
        estimators warm.
        """
        coord = self._coord_ema.value()
        node_time = self._node_time_ema.value()
        if coord is None or node_time is None or node_time <= 0:
            return PUBLISH_THRESHOLD_BOOTSTRAP
        return SAFETY_FACTOR * coord / node_time

    # -- RAM check and WAL checkpoint ---------------------------------------

    def _free_ram_mb(self):  # pragma: no cover
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) // 1024
        except OSError:
            return None

    def _maybe_checkpoint(self, force=False):
        # Workers checkpoint the queue PASSIVE-only: passive backfills what it
        # can without taking the writer lock or waiting on readers, so a
        # worker checkpoint can never stall the other workers' writes.  Only
        # the supervisor attempts TRUNCATE, under its quiesce protocol.
        now = time.time()
        if force or now - self._last_checkpoint > self._checkpoint_interval:
            self._flush_cost_model_buffer()
            self.score_cache.checkpoint()
            self.queue.checkpoint("PASSIVE")
            self._last_checkpoint = now
            self._log_wal_traffic(now)

    def _wal_ceiling_tripped(self) -> bool:
        """Worker-side backstop for the queue WAL hard ceiling.

        The supervisor enforces the same ceiling, but a worker that outlives
        the supervisor — or spins in a coordination path that never reaches a
        bundle boundary — must stop writing on its own before the WAL fills
        the disk.  Latches once tripped and requests a stop; throttled to one
        file-size probe per WAL_CEILING_CHECK_SECONDS so it is safe to call
        from hot paths."""
        if self._wal_ceiling_hit:
            return True
        now = time.time()
        if now - self._last_wal_ceiling_check < WAL_CEILING_CHECK_SECONDS:
            return False
        self._last_wal_ceiling_check = now
        wal_bytes = self.queue.wal_size_bytes()
        if wal_bytes >= QUEUE_WAL_HARD_CEILING_BYTES:
            self._wal_ceiling_hit = True
            logger.critical(
                '%s queue WAL %.2f GB breached hard ceiling %.2f GB — '
                'stopping this worker', self.name, wal_bytes / 1e9,
                QUEUE_WAL_HARD_CEILING_BYTES / 1e9)
            self.request_stop()
            return True
        return False

    def _checkpoint_pause_active(self) -> bool:
        """Cached view of the supervisor's checkpoint_pause flag, re-read at
        most once per PAUSE_POLL_SECONDS.  The flag check is itself a queue
        read, so polling it unthrottled from hot paths (heartbeats, the
        mid-evaluation bound refresh) would add exactly the reader overlap
        the quiesce exists to clear.  The cache makes pause transitions
        visible up to PAUSE_POLL_SECONDS late, well inside the supervisor's
        TRUNCATE retry window."""
        now = time.time()
        if now - self._last_pause_check >= PAUSE_POLL_SECONDS:
            self._last_pause_check = now
            self._pause_active = self.queue.checkpoint_paused()
        return self._pause_active

    def _respect_checkpoint_pause(self):
        """At a claim boundary, stay off the queue database while the
        supervisor's checkpoint_pause flag is set (compute in progress is
        unaffected; the flag self-expires, so a dead supervisor cannot wedge
        the swarm).  Once waiting, the flag is polled directly — the sleep
        between polls is the throttle."""
        if not self._checkpoint_pause_active():
            return
        while not self.cancel() and self.queue.checkpoint_paused():
            time.sleep(PAUSE_POLL_SECONDS)
        self._pause_active = False

    def _check_disk(self):
        now = time.time()
        if now - self._last_disk_check < DISK_CHECK_SECONDS:
            return
        self._last_disk_check = now
        used_fraction = disk_stats(self.queue.db_path)["used_fraction"]
        if used_fraction >= DISK_STOP_FRACTION:
            logger.critical(
                '%s disk %.0f%% full (>= %.0f%% stop threshold) — latching '
                'swarm down', self.name, 100 * used_fraction,
                100 * DISK_STOP_FRACTION)
            try:
                self.queue.set_disk_stop(
                    f'{self.name}: disk {100 * used_fraction:.1f}% full')
            except sqlite3.OperationalError as exc:
                # A 100%-full disk can fail this very write; the worker still
                # stops, and run's startup live-fullness check keeps the swarm
                # down even without the latch row.
                logger.critical('%s could not write disk_stop latch: %s',
                                self.name, exc)
            self.request_stop()

    def _check_ram(self):  # pragma: no cover
        now = time.time()
        if now - self._last_ram_check < 30:
            return
        self._last_ram_check = now
        free = self._free_ram_mb()
        if free is None:
            return
        if free < RAM_CRIT_MB:
            logger.warning('%s free RAM critically low (%d MB) — checkpointing',
                           self.name, free)
            self._maybe_checkpoint(force=True)
        elif free < RAM_WARN_MB:
            logger.warning('%s free RAM low (%d MB)', self.name, free)

    # -- heartbeat ----------------------------------------------------------

    def _heartbeat(self, branch_key, n_words, claim_idx, claim_started_at,
                   best_guess, best_erd, force=False,
                   bound_erd=None):
        # Count every invocation (one per node) BEFORE the throttle, so the
        # node counter is exact even though we only write every HB_SECONDS.
        self._nodes += 1
        now = time.time()
        if not force and now - self._last_hb < HB_SECONDS:
            return
        # Per-heartbeat-window WAL backstop: an evaluation deep in the engine
        # passes through here even when it never reaches a bundle boundary,
        # so a runaway writer stops itself instead of relying on the
        # supervisor being alive to stop it.
        self._wal_ceiling_tripped()
        # Emit the WAL traffic attribution on its own fast timer, before the
        # pause-defer below: a runaway that provokes the hard ceiling coincides
        # with the supervisor quiescing, so gating this behind the pause would
        # suppress exactly the diagnostic the ceiling trip needs.
        self._log_wal_traffic(now)
        # Defer unforced heartbeats while the supervisor is quiescing writers
        # for a TRUNCATE checkpoint; the pause window is well inside
        # HB_TIMEOUT_SECONDS, so liveness is never in question.
        if not force and self._checkpoint_pause_active():
            self._last_hb = now
            self._nodes_at_last_hb = self._nodes
            return
        dt = now - self._last_hb
        node_rate = (self._nodes - self._nodes_at_last_hb) / dt if dt > 0 else 0.0
        self._last_hb = now
        self._nodes_at_last_hb = self._nodes
        self.queue.heartbeat(
            self.name, os.getpid(), branch_key, n_words, self.started,
            self.claims_done, claim_idx=claim_idx,
            claim_started_at=claim_started_at,
            cache_hits=self.score_cache.read_hits,
            cache_misses=self.score_cache.read_misses,
            n_cutoff=self.n_cutoff, n_pruned=self.n_pruned, n_ok=self.n_ok,
            best_guess=best_guess, best_erd=best_erd, bound_erd=bound_erd,
            cur_candidate=self._cur_candidate,
            cur_max_depth=self._cand_max_depth,
            cur_nodes=self._nodes, node_rate=node_rate,
            cur_path=self._hb_spine_str())
        self._hb_max_spine = {}
        if self._cur_candidate and now - self._last_progress_log >= PROGRESS_LOG_SECONDS:  # pragma: no cover
            self._last_progress_log = now
            be = f'{best_erd:.4f}' if best_erd is not None else '-'
            bg = (best_guess or '-').upper()
            logger.info('%s claim %d: %s in progress  '
                        'guess_depth=%d path=%s  %.1fM nodes %.0fk/s  best=%s %s',
                        self.name, claim_idx,
                        self._cur_candidate.upper(),
                        self._cand_max_depth, self._log_spine_str(),
                        self._nodes / 1e6, node_rate / 1e3, bg, be)
            self._log_max_spine = {}
            self._cand_max_depth = 0
        if now - self._last_util_log >= PROGRESS_LOG_SECONDS:  # pragma: no cover
            # Fires on coordination heartbeats too (cur_candidate is None), so
            # the split is visible precisely when the worker isn't evaluating.
            self._last_util_log = now
            elapsed = now - self.started
            if elapsed > 0:
                eval_pct = 100.0 * self._eval_seconds / elapsed
                coord_seconds = elapsed - self._eval_seconds
                logger.info('%s utilisation: eval %.0fs (%.0f%%)  '
                            'coord %.0fs (%.0f%%)  over %.0fs',
                            self.name, self._eval_seconds, eval_pct,
                            coord_seconds, 100.0 - eval_pct, elapsed)

    def _log_wal_traffic(self, now, force=False):
        """Log this worker's per-table WAL write/read rate since the last such
        log, largest first.  Names which coordination traffic (bulk-elimination
        sweeps, holes-scan re-reads, claims, updates, deletes) is pouring into
        the shared WAL — the breakdown the WAL file itself does not carry.

        Self-throttled to WAL_TRAFFIC_LOG_SECONDS and called from the heartbeat
        (so it fires even during a single long candidate) so a fast runaway
        still emits attribution before the hard ceiling latches the swarm down.
        force bypasses the throttle for the final flush on shutdown, so the last
        interval's traffic is never lost when the ceiling trip terminates the
        worker between logs."""
        dt = now - self._last_wal_traffic_log
        if not force and dt < WAL_TRAFFIC_LOG_SECONDS:
            return
        _, byts = self.queue.wal_traffic_snapshot()
        _, prev_bytes = self._last_wal_traffic
        self._last_wal_traffic = (None, byts)
        self._last_wal_traffic_log = now
        if dt <= 0:
            return
        deltas = {c: byts[c] - prev_bytes.get(c, 0) for c in byts}
        top = sorted(((c, b) for c, b in deltas.items() if b > 0),
                     key=lambda kv: kv[1], reverse=True)[:4]
        if not top:
            return
        summary = '  '.join(f'{c} {b / 2 ** 20 / dt:.1f} MiB/s' for c, b in top)
        logger.info('%s queue WAL traffic: %s', self.name, summary)

    # -- claim packing --------------------------------------------------------

    def _packing_stats(self, branch_key, words):
        """(order, cost_lower_bound) for `words`, cached per branch for the
        life of this worker process.

        order is the branch's best-first candidate order (a permutation of
        range(n_candidates), Σk² ascending, C2.2); cost_lower_bound is
        candidate_cost_lower_bound indexed by idx (C2.1).  Both come from one
        vectorized candidate_stats pass (pattern_matrix.py §3) over the whole
        guess vocabulary — idx IS the matrix row here, since self.all_words is
        both the ERD_ALL candidate list and the pattern matrix's guess
        vocabulary, in the same order.

        This is a pure function of branch_indices (the pattern matrix, which
        is immutable shared data, plus the branch's words), so every worker
        process computes a bit-identical array independently: the queue DB
        only has to hold the shared cursor and best_erd (claim_next_bundle),
        never this vector (adaptive_claim_packing.md §12).
        """
        cached = self._packing_stats_cache.get(branch_key)
        if cached is not None:
            return cached
        branch_indices = self.pattern_matrix.answer_indices(words)
        stats = self.pattern_matrix.candidate_stats(branch_indices)
        order = sorted(range(self.n_candidates),
                       key=lambda idx: stats.sum_squared_group_sizes[idx])
        result = (order, stats.cost_lower_bound)
        self._packing_stats_cache[branch_key] = result
        return result

    def _claim_bundle(self, branch_key, n_candidates, words):
        """claim_next_bundle for `branch_key`, supplying this worker's
        (cached) packing stats.  Returns (bundle_id, indices, forced) or None
        — see ERDQueue.claim_next_bundle.

        Every claim path (top-level loop, helping, deep solving, focused)
        funnels through here, so this is where a quiescing supervisor's pause
        flag is honoured before touching the queue."""
        self._respect_checkpoint_pause()
        order, cost_lower_bound = self._packing_stats(branch_key, words)
        return self.queue.claim_next_bundle(
            branch_key, self.name, n_candidates, order, cost_lower_bound,
            small_count=self.small_count, count_cap=self.count_cap,
            republish_limit=self.republish_limit)

    # -- evaluate one candidate claim ---------------------------------------

    def evaluate_claim(self, branch_key, words, n_words, idx, budget=None,
                       bundle_id=None, bundle_start_idx=None,
                       bundle_end_idx=None):
        """Evaluate the single candidate self.all_words[idx] against branch_key.

        Folds the result into the branch's shared best and marks the claim
        done=1.  Returns True if the candidate was fully evaluated; False if
        cancelled mid-evaluation (claim left done=0 for reclaim/redo).

        budget is the branch's guess budget (depth-limited ERD): a candidate
        whose strategy can't win within budget is infeasible (and taints the
        branch — see ERDQueue.mark_branch_tainted).

        bundle_id/bundle_start_idx/bundle_end_idx identify the claim_next_bundle
        bundle this candidate belongs to, for claim_telemetry attribution; all
        three are None for a claim taken outside the bundle path.
        """
        candidate = self.all_words[idx]
        self._cur_candidate = candidate
        self._hb_max_spine = {}
        self._log_max_spine = {}
        local_candidate, local_best, branch_ceiling = \
            self.queue.read_branch_best(branch_key)
        local_md = None
        shared_best = local_best
        last_refresh = time.time()
        claim_started = int(time.time())
        t0 = time.time()

        def _bound_provider():
            # Refreshes shared_best from the queue at most every
            # BEST_REFRESH_SECONDS, then returns the tightest known bound as a
            # float (inf when nothing is known yet).  Called by evaluate_candidate
            # at each sub-branch level throughout the full recursion.  The
            # branch ceiling is a permanent bound source: pricing out against
            # it makes the candidate a cut, recorded via mark_branch_cut below.
            nonlocal shared_best, last_refresh
            now = time.time()
            # Skipped while the supervisor quiesces for TRUNCATE: this refresh
            # opens a read snapshot into the WAL every BEST_REFRESH_SECONDS
            # per worker, which is the traffic that blocks truncation.  The
            # cached bound stays valid — bounds only tighten — so the cost is
            # slightly weaker pruning for the pause window.
            if (now - last_refresh > BEST_REFRESH_SECONDS
                    and not self._checkpoint_pause_active()):
                _, new, _ = self.queue.read_branch_best(branch_key)
                if new is not None and (shared_best is None or new < shared_best):
                    shared_best = new
                last_refresh = now
            bests = [b for b in (local_best, shared_best, branch_ceiling)
                     if b is not None]
            return min(bests) if bests else float('inf')

        def _eff_bound():
            v = _bound_provider()
            return None if v == float('inf') else v

        # Throttled, not forced: forcing a heartbeat here writes the DB once per
        # candidate, which floods the WAL (and blocks the checkpoint TRUNCATE by
        # writing through the supervisor's quiesce) when candidates are tiny and
        # evaluate in microseconds.  Liveness and the cur_candidate display come
        # from the throttled per-node heartbeat inside evaluate_candidate.
        self._heartbeat(branch_key, n_words, idx, claim_started,
                        local_candidate, local_best, bound_erd=_eff_bound())

        if self.cancel():
            return False

        self._cand_max_depth = 0
        nodes_before = self._nodes
        cand_t0 = time.time()
        # Work-metric capture for the §10 validation gate: under single-candidate
        # claiming this claim's nodes_delta IS the candidate's true cost, so the
        # observer records the predicted work and the bound it was computed
        # against; the accuracy row is written below once actual_nodes is known.
        metric = {}

        def _metric_observer(group_sizes, has_self, candidate_cost_lower_bound,
                             bound, erd_lower_bound_pruned):
            metric['predicted'] = estimate_candidate_work(
                group_sizes, has_self, n_words, bound, budget, self._typical)
            metric['bound'] = None if bound == float('inf') else bound
            metric['candidate_cost_lower_bound'] = candidate_cost_lower_bound
            metric['erd_lower_bound_pruned'] = erd_lower_bound_pruned
            # Persist the group-size multiset for non-ERD-pruned rows: the
            # sufficient statistic to recompute any candidate work metric
            # offline.  ERD-pruned rows are exactly 0, so their sizes carry no
            # metric-design signal.
            metric['group_sizes'] = (None if erd_lower_bound_pruned else
                                     '-'.join(str(k) for k in group_sizes))

        status, cost, cand_md, budget_tainted = evaluate_candidate(
            words, candidate, self.rcache, self.score_cache,
            n=n_words, best_erd=float('inf'), guesses=self.all_words,
            policy=ERD_ALL, cancel_check=self.cancel,
            note_depth=self._note_depth, budget=budget,
            subbranch_solver=self._subbranch_solver,
            bound_provider=_bound_provider,
            mid_loop_publisher=self._mid_loop_publisher,
            metric_observer=_metric_observer if self._adaptive else None,
            pattern_matrix=self.pattern_matrix,
            heartbeat=lambda: self._heartbeat(
                branch_key, n_words, idx, claim_started,
                local_candidate, local_best, bound_erd=_eff_bound()))
        cand_elapsed = time.time() - cand_t0
        self._eval_seconds += cand_elapsed
        if cand_elapsed > 10:  # pragma: no cover
            logger.warning('%s slow candidate %s (idx=%d): %.1fs  '
                           'status=%s  max_depth=%d', self.name, candidate,
                           idx, cand_elapsed, status, self._cand_max_depth)

        nodes_delta = self._nodes - nodes_before
        if self._adaptive and nodes_delta > 0:
            self.queue.add_nodes_spent(branch_key, nodes_delta)

        if status in _ABORT_STATUSES:  # pragma: no cover
            return False

        # A candidate excluded by the depth cap (anywhere in its subtree)
        # taints the branch: its ERD is only valid at this budget.  Marked
        # for any candidate, winner or not — see the taint rule.
        if budget_tainted:
            self.queue.mark_branch_tainted(branch_key)
        if status == SOLVED:
            self.n_ok += 1
            if local_best is None or cost < local_best:
                local_best, local_candidate, local_md = cost, candidate, cand_md
                self.queue.update_branch_best(branch_key, local_candidate,
                                              local_best, local_md)
                shared_best = local_best
        elif status == OVER_ERD_LIMIT:
            self.n_cutoff += 1
            if branch_ceiling is not None:
                # Priced out on a ceilinged branch.  Only consulted at finalize
                # when best_guess is NULL — where no real best ever existed, so
                # every price-out was against the ceiling and the branch is a
                # cut, not a proven loss.
                self.queue.mark_branch_cut(branch_key)
        elif status == OVER_DEPTH_BUDGET:
            self.n_pruned += 1
        else:  # pragma: no cover
            self.n_useless += 1

        elapsed = time.time() - t0
        self.queue.complete_candidate(branch_key, idx)
        if self._adaptive:
            coord_seconds = max(0.0, elapsed - cand_elapsed)
            # The in-memory estimators feed the adaptive publish threshold and use
            # the narrow in-evaluate_claim coordination window (the ratio is
            # unit-free).  The outbound claim_telemetry row instead telescopes from
            # the previous claim's completion, so its coordination figure also
            # includes claim acquisition (claim_next_bundle's packer transaction,
            # paid once per bundle rather than once per candidate — a candidate
            # evaluated mid-bundle sees this figure collapse to ~0) and any
            # inter-claim overhead; this is the offline-diagnostic span and is not
            # fed back into control.
            now_complete = time.time()
            full_coord_seconds = max(
                0.0, (now_complete - self._last_claim_complete) - cand_elapsed)
            self._last_claim_complete = now_complete
            self._coord_ema.add(coord_seconds)
            if nodes_delta > 0 and cand_elapsed > 0:
                self._node_time_ema.add(cand_elapsed / nodes_delta)
            if metric:
                # Log every non-ERD-pruned claim; down-sample the redundant
                # ERD-pruned mass so a multi-day corpus stays bounded (see
                # ERD_LOWER_BOUND_PRUNED_SAMPLE_EVERY).
                if metric['erd_lower_bound_pruned']:
                    log_it = (self._erd_lower_bound_pruned_accuracy_n
                             % ERD_LOWER_BOUND_PRUNED_SAMPLE_EVERY) == 0
                    self._erd_lower_bound_pruned_accuracy_n += 1
                else:
                    log_it = True
                if log_it:
                    self.queue.add_candidate_accuracy(
                        branch_key, n_words, budget, metric['predicted'],
                        metric['bound'], metric['candidate_cost_lower_bound'],
                        metric['erd_lower_bound_pruned'],
                        nodes_delta, group_sizes=metric['group_sizes'],
                        source_word=self._top_source_word)
            self.queue.add_claim_telemetry(
                n_words, int(full_coord_seconds * 1e3), nodes_delta,
                self.n_workers, branch_key=branch_key,
                spine=self._claimed_branch_spine, worker_id=self.name,
                bundle_id=bundle_id, idx=idx,
                bundle_start_idx=bundle_start_idx, bundle_end_idx=bundle_end_idx)
        self.claims_done += 1
        # Throttled, not forced: see the per-candidate heartbeat above — a forced
        # write here is per-candidate and floods the WAL on fast candidates.
        self._heartbeat(branch_key, n_words, idx, claim_started,
                        local_candidate, local_best, bound_erd=_eff_bound())
        return True

    # -- evaluate a packer-issued bundle of candidate claims -----------------

    def _evaluate_bundle_member(self, branch_key, words, n_words, idx, budget,
                                bundle_id, nodes_at_bundle_start, wall_t0,
                                bundle_start_idx=None, bundle_end_idx=None):
        """evaluate_claim for one bundle member; on cancellation/abort,
        records the bundle as censored.  Returns True to keep going, False
        for the caller to abort evaluate_bundle immediately."""
        if self.cancel():
            self._finish_bundle(branch_key, bundle_id, nodes_at_bundle_start,
                                wall_t0, censored=True)
            return False
        if not self.evaluate_claim(branch_key, words, n_words, idx,
                                   budget=budget, bundle_id=bundle_id,
                                   bundle_start_idx=bundle_start_idx,
                                   bundle_end_idx=bundle_end_idx):
            self._finish_bundle(branch_key, bundle_id, nodes_at_bundle_start,
                                wall_t0, censored=True)
            return False
        return True

    def evaluate_bundle(self, branch_key, words, n_words, bundle_id, indices,
                        forced, budget=None):
        """Evaluate a claim_next_bundle bundle, folding each candidate's
        result into the branch's shared best as it goes (evaluate_claim per
        idx, in the bundle's best-first order).

        Sequential-sibling pruning is preserved exactly as a single-candidate
        sweep: each evaluate_claim call re-reads the branch's shared best,
        already updated by any earlier member of this same bundle
        (adaptive_claim_packing.md §8 invariant 2) — no extra state needs to
        be threaded between iterations for this.

        Tracks cumulative nodes and wall time since the bundle started.  When
        either exceeds this worker's bundle_node_cap / bundle_wall_cap_seconds
        (§7a cross-candidate overrun), the unfinished remainder is republished
        (returned to the unclaimed pool for re-packing) rather than driven to
        completion inline — never re-claimed as `len(remainder)` individual
        claims.  The node cap protects a survivor bundle's strong-splitter
        head from stranding its siblings.

        A candidate in `forced` (its candidate_republish count already hit
        republish_limit, §7's bounded-republish-depth guardrail) is always
        evaluated in this bundle — it is never swept into a republished
        remainder, since republishing it again is exactly the thrash the
        guardrail exists to stop.  Its own cost never counts toward either
        cap for itself OR for later siblings: the cumulative counters are
        re-baselined immediately after it runs, so any real depth in its
        subtree is absorbed by the always-active within-candidate sub-branch
        promotion (mid_loop_publisher) instead of cutting off unrelated
        candidates that happen to follow it in best-first order.

        Returns True if the bundle was fully handled — every candidate either
        evaluated to completion or handed back via republish; False if
        cancelled mid-evaluation, leaving the unfinished remainder's claims
        done=0 for reclaim (same contract as evaluate_claim).
        """
        nodes_at_bundle_start = self._nodes
        wall_t0 = time.time()
        bundle_start_idx = min(indices) if indices else None
        bundle_end_idx = max(indices) if indices else None
        for pos, idx in enumerate(indices):
            if not self._evaluate_bundle_member(
                    branch_key, words, n_words, idx, budget, bundle_id,
                    nodes_at_bundle_start, wall_t0,
                    bundle_start_idx=bundle_start_idx,
                    bundle_end_idx=bundle_end_idx):
                return False
            if idx in forced:
                nodes_at_bundle_start = self._nodes
                wall_t0 = time.time()
                continue
            nodes_delta = self._nodes - nodes_at_bundle_start
            wall_delta = time.time() - wall_t0
            if (nodes_delta > self.bundle_node_cap
                    or wall_delta > self.bundle_wall_cap_seconds):
                remainder = []
                for later_idx in indices[pos + 1:]:
                    if later_idx in forced:
                        if not self._evaluate_bundle_member(
                                branch_key, words, n_words, later_idx, budget,
                                bundle_id, nodes_at_bundle_start, wall_t0,
                                bundle_start_idx=bundle_start_idx,
                                bundle_end_idx=bundle_end_idx):
                            return False
                    else:
                        remainder.append(later_idx)
                if remainder:
                    self.queue.republish_remainder(branch_key, bundle_id, remainder)
                self._finish_bundle(branch_key, bundle_id, nodes_at_bundle_start,
                                    wall_t0, censored=bool(remainder))
                return True
        self._finish_bundle(branch_key, bundle_id, nodes_at_bundle_start,
                            wall_t0, censored=False)
        return True

    def _finish_bundle(self, branch_key, bundle_id, nodes_at_start, wall_t0,
                       censored):
        """Record a completed/censored bundle's actual cost, if this claim
        went through the packer (bundle_id is None for a bare evaluate_claim
        call outside the bundle path).

        wall_t0 is the bundle's own evaluation start (re-baselined past any
        forced member — see evaluate_bundle), so the elapsed time here is
        this bundle's evaluation wall span, not claim-handout coordination
        overhead (that is claim_telemetry's busy_wait_millis, measured in
        claim_next_bundle).
        """
        if bundle_id is None or not self._adaptive:
            return
        nodes = self._nodes - nodes_at_start
        wall_millis = int((time.time() - wall_t0) * 1000)
        self.queue.record_bundle_stats(branch_key, bundle_id, nodes,
                                       wall_millis, censored=censored)

    # -- finalize -----------------------------------------------------------

    def maybe_finalize(self, branch_key, words, n_candidates) -> bool:
        """If every candidate is done, finalize the branch exactly once.

        Returns True when this worker completed the finalize; False when
        candidates remain or a rival holds the finalize (see
        _await_rival_finalize for why the rival may be dead)."""
        if self.queue.branch_done_candidates(branch_key) < n_candidates:
            return False
        if not self.queue.try_finalize_branch(branch_key):  # pragma: no cover
            return False  # another worker won the finalize
        finalize_t0 = time.time()
        meta = self.queue.read_branch_meta(branch_key)
        (best_guess, best_erd, max_depth, tainted, budget,
         ceiling, cut_occurred) = meta
        branch_row = self.queue.get_branch(branch_key)
        nodes_spent = branch_row['nodes_spent'] if branch_row else 0
        created_at = branch_row['created_at'] if branch_row else None
        finalized_at = branch_row['finalized_at'] if branch_row else None
        spine = branch_row['spine'] if branch_row else None
        # Wall span of the branch (upper bound — solves interleave).  The only
        # per-solve wall figure, recorded on the cost sample and the finalize log.
        wall_millis = (None if created_at is None or finalized_at is None
                       else max(0, (finalized_at - created_at) * 1000))
        # Claims drained to finalize, captured before delete_branch drops the rows.
        completed_candidates = self.queue.branch_done_candidates(branch_key)
        bulk_done_candidates = self.queue.branch_bulk_done_candidates(branch_key)
        n_claims = completed_candidates - bulk_done_candidates
        cut = best_guess is None and cut_occurred
        ceiling_proves_loss = (
            cut and budget is not None and ceiling is not None
            and math.isfinite(ceiling) and ceiling > budget
        )
        if best_guess is not None:
            # Exact optimum.  A ceiling (if any) only pruned candidates proven
            # >= a value some candidate beat, so the result is universally
            # valid — cacheable exactly like a ceiling-free solve.
            # Untainted => unconstrained optimum, reusable at any budget >=
            # max_depth (solve_budget NULL).  Tainted => valid only at this
            # budget (solve_budget = budget).  See cache_sqlite reuse rule.
            solve_budget = budget if tainted else None
            self.score_cache.write(branch_key, ERD_ALL, best_guess, best_erd,
                                   max_depth=max_depth, solve_budget=solve_budget)
            cache_all_scores(best_guess, words, self.score_cache, branch_key,
                             cache=self.rcache)
            if self._adaptive and nodes_spent > 0:
                self._update_cost_model(len(words), nodes_spent, budget=budget,
                                        wall_millis=wall_millis)
            # NB: no per-finalize checkpoint — with recursive promotion a worker
            # finalizes thousands of sub-branches; checkpointing each one is
            # ruinous.  WAL is drained by the periodic _maybe_checkpoint instead.
            logger.info('%s finalized branch (%d words) -> %s erd=%.4f '
                        'max_depth=%s budget=%s%s', self.name, len(words),
                        best_guess, best_erd, max_depth, budget,
                        ' TAINTED' if tainted else '')
        elif ceiling_proves_loss:
            # A budget-feasible strategy has ERD <= budget.  Pricing every
            # candidate out at a strictly larger ceiling therefore proves no
            # budget-feasible strategy exists, including for tainted cuts.
            self.score_cache.write_loss(branch_key, ERD_ALL, budget)
            if self._adaptive and nodes_spent > 0:
                self.queue.add_cost_sample(ERD_ALL, len(words), nodes_spent,
                                           'cut', budget=budget, censored=1,
                                           wall_millis=wall_millis)
            logger.warning('%s finalized branch (%d words) as LOSS: ceiling '
                           '%.4f exceeds budget=%s nodes=%d', self.name,
                           len(words), ceiling, budget, nodes_spent)
        elif cut:
            # Every candidate priced out at >= the ceiling and none was proven
            # infeasible: a lower bound only ("true ERD >= ceiling").  That is
            # everything the promoting parent asked, but it is NOT an optimum:
            # never written to the score cache, never a loss.  Delivered to
            # waiters via cut_results (before delete_branch below, so a waiter
            # that sees the branch vanish re-checks the channel and finds it).
            # nodes_spent is right-censored (the exact solve was never run), so
            # it must not fold into the cost model as a completed-solve sample;
            # the raw sample is kept for offline survival analysis.
            self.queue.add_cut_result(branch_key, budget, ceiling,
                                      tainted=tainted)
            if self._adaptive and nodes_spent > 0:
                self.queue.add_cost_sample(ERD_ALL, len(words), nodes_spent,
                                           'cut', budget=budget, censored=1,
                                           wall_millis=wall_millis)
            logger.info('%s finalized branch (%d words) as CUT >= %.4f '
                        'budget=%s nodes=%d', self.name, len(words), ceiling,
                        budget, nodes_spent)
        else:  # pragma: no cover
            # No feasible guess within budget: this branch is a loss.  There is
            # no winning strategy to record in branch_best_by_policy, but the
            # loss itself is persisted so the branch is never re-swept at this
            # (or any smaller) budget.  Sound even under a ceiling: cut_occurred
            # is clear, so every candidate was individually PROVEN infeasible —
            # a proof the ceiling cannot have manufactured.
            if budget is not None:
                self.score_cache.write_loss(branch_key, ERD_ALL, budget)
            logger.warning('%s branch (%d words) UNSOLVABLE within budget %s '
                           '(loss) src=%s', self.name, len(words), budget,
                           branch_key[:25])
        # Persist the branch's timing/cost before delete_branch destroys it, so
        # "how long / how much did branch X cost" stays answerable offline.
        # finalize_bundle_stats aggregates and clears this branch's bundle_stats
        # rows; (None, None, None, None) when it never claimed a bundle (fully
        # solved from reused cache entries).
        # Telemetry failure must not kill the worker or skip the cleanup
        # below: the branch result is already published to the score cache,
        # and without mark_done/delete_branch the branch's claim rows leak
        # and the pending row is never retired.
        # Everything above published the branch's result; that span is the
        # finalize phase of the coordination breakdown, recorded on this
        # branch's own finalize row.
        cache_write_millis = int((time.time() - finalize_t0) * 1000)
        try:
            (n_bundles, max_bundle_nodes, total_bundle_wall_millis,
             censored_units) = self.queue.finalize_bundle_stats(branch_key)
            self.queue.add_branch_finalize_log(
                branch_key, spine, len(words), budget, created_at,
                finalized_at, nodes_spent, n_claims, n_bundles=n_bundles,
                max_bundle_nodes=max_bundle_nodes,
                total_bundle_wall_millis=total_bundle_wall_millis,
                censored_units=censored_units, ceiling=ceiling,
                bulk_done_candidates=bulk_done_candidates,
                best_guess=best_guess, best_erd=best_erd,
                cache_write_millis=cache_write_millis,
                outcome='loss' if ceiling_proves_loss else ('cut' if cut else
                        ('exact' if best_guess is not None else 'loss')))
        except Exception:
            logger.exception(
                '%s finalize telemetry failed for branch %s -- result '
                'already published; continuing to cleanup', self.name,
                branch_key[:25])
        loss = best_guess is None and (not cut or ceiling_proves_loss)
        try:
            if loss:
                self.queue.complete_pending_for_loss(
                    branch_key, budget, self.root_budget)
            elif cut:
                self.queue.requeue_pending(branch_key)
            else:
                self.queue.mark_done(branch_key)
        except Exception:
            logger.exception('%s pending completion failed for branch %s; '
                             'retaining queued work', self.name, branch_key[:25])
            try:
                self.queue.requeue_pending(branch_key)
            except Exception:
                logger.exception('%s could not requeue branch %s', self.name,
                                 branch_key[:25])
        self.queue.delete_branch(branch_key)    # drop transient coordination
        self._packing_stats_cache.pop(branch_key, None)
        # Restart the coordination window past this finalize.  evaluate_claim
        # telescopes coordination_millis from the previous claim's completion,
        # so without this the finalize span would reappear as idle time on the
        # first claim of whatever branch this worker picks up next — a
        # different, unrelated branch.
        self._last_claim_complete = time.time()
        return True

    def _await_rival_finalize(self, branch_key, words, n_words, n_candidates):
        """Every candidate is done but a rival holds the finalize.

        A live rival completes the cache write + delete within milliseconds.
        One killed between winning try_finalize_branch and deleting the row
        leaves the branch 'finalized' forever — try_finalize_branch refuses
        non-'open' rows, so without intervention every waiting sibling spins
        silently at full CPU.  Heartbeat first (this wait must stay visible
        and must not get this worker's parent claims reclaimed), then reopen
        the row once it has been 'finalized' past FINALIZE_TAKEOVER_SECONDS
        and complete the finalize from its intact claims and meta."""
        self._cur_candidate = None      # coordinating, no candidate in flight
        self._heartbeat(branch_key, n_words, None, None, None, None,
                        force=True)
        if self.queue.reclaim_stale_finalize(branch_key,
                                             FINALIZE_TAKEOVER_SECONDS):
            logger.warning('%s reopened branch (%d words): finalizer died '
                           'mid-finalize', self.name, n_words)
            self.maybe_finalize(branch_key, words, n_candidates)
            return
        time.sleep(0.05)

    # -- recursive cooperative solving --------------------------------------

    def _help_other_branch(self, exclude_branch_key: bytes) -> bool:
        """Evaluate one bundle of candidate claims from any open branch other
        than exclude_branch_key.

        Called when the worker is waiting on a dependency branch whose remaining
        candidates are all held by other workers.  Instead of sleeping, the
        worker drains useful work from the queue.  Returns True if a bundle
        was evaluated, False if there was nothing to claim.
        """
        for branch in self.queue.branches_in_progress():
            other_key = bytes(branch['branch_key'])
            if other_key == bytes(exclude_branch_key):
                continue
            n_candidates = branch['n_candidates']
            words = decode_subset(other_key)
            claim = self._claim_bundle(other_key, n_candidates, words)
            if claim is None:
                continue
            bundle_id, indices, forced = claim
            budget = self._branch_budget(branch)
            # Promotions while helping must base off the helped branch's spine.
            saved_spine = self._claimed_branch_spine
            self._claimed_branch_spine = branch['spine'] if 'spine' in branch.keys() \
                else None
            try:
                if self.evaluate_bundle(other_key, words, branch['n_words'],
                                        bundle_id, indices, forced, budget=budget):
                    self.maybe_finalize(other_key, words, n_candidates)
            finally:
                self._claimed_branch_spine = saved_spine
            self._maybe_checkpoint()
            return True
        return False

    def _subbranch_solver(self, words, budget, ceiling=float('inf')):
        """Engine hook: decide whether to solve a sub-branch cooperatively.

        budget <= 2 branches are always solved inline: at budget=2, each
        candidate needs a perfect separator (all response groups singletons),
        which for n >= 3 is near-impossible — candidates return OVER_DEPTH_BUDGET
        instantly, so cooperative overhead is never worthwhile.

        For budget >= 3: when the cost model is warm, promotes sub-branches whose
        predicted node cost exceeds the adaptive publish threshold.  When cold,
        falls back to the PROMOTE_MIN_SIZE size threshold: large cold branches
        promote up front, small ones inline under the mid-loop publisher's
        wall-clock backstop — so a mispredicted small tarpit is bounded by
        COLD_BACKSTOP_SECONDS rather than by a size heuristic that can't see it.

        ceiling is the frame's alpha-beta ceiling; the cooperative solve prunes
        against it exactly as the inline recursion would, so a promoted trap
        family terminates in the cheap >= ceiling proof instead of grinding to
        the exact optimum nobody asked for.  Returns the engine's (status, cost,
        max_depth, floor) tuple — which may be a cut (OVER_ERD_LIMIT, bound,
        None, floor) — or None to inline.
        """
        if budget is None:
            return None
        # budget=2 means each candidate needs a perfect separator (every response
        # group must be a singleton).  For n >= 3 this is near-impossible, so
        # virtually every candidate returns OVER_DEPTH_BUDGET instantly.  The warm
        # cost model handles this correctly (predicted ≈ 0 < threshold → inline),
        # but the cold fallback promotes by size alone.  Guard both paths: these
        # branches are not worth cooperative overhead regardless of model state.
        if budget <= 2:
            return None
        n = len(words)
        if not self._adaptive:
            # Plain size-based promotion: no cost model, no overrun.
            return self.cooperative_solve(words, budget, ceiling) \
                if n >= PROMOTE_MIN_SIZE else None
        predicted = self._typical(n, budget)
        if predicted is None:
            # Cold model: promote large branches by size, inline small ones; the
            # wall-clock backstop bounds a cold inline tarpit.
            return self.cooperative_solve(words, budget, ceiling) \
                if n >= PROMOTE_MIN_SIZE else None
        if predicted < self._publish_threshold():
            return None
        return self.cooperative_solve(words, budget, ceiling)

    def _read_satisfying_cut(self, branch_key, budget, ceiling, n_words):
        """The engine tuple for a recorded cut that satisfies this consumer, or
        None.  Satisfying = proven at a budget >= ours (the bound then holds at
        ours) with a bound >= our ceiling (so it proves what the parent asked).
        A branch can carry several cuts (one per budget/taint class it has
        been proven under, see read_cut_result); the first satisfying one is
        used, whichever budget or taint class it comes from."""
        if ceiling == float('inf'):
            return None
        for cut_bound, cut_budget, cut_tainted in self.queue.read_cut_result(branch_key):
            if budget <= cut_budget and erd_ge(cut_bound, ceiling, n_words):
                # A tainted cut's bound holds only among budget-feasible
                # strategies; the consumer joins the taint so its own result
                # cannot claim the unconstrained optimum.
                return (OVER_ERD_LIMIT, cut_bound, None, cut_tainted)
        return None

    def cooperative_solve(self, words, budget, ceiling=float('inf')):
        """Solve sub-branch `words` at `budget` cooperatively, returning the
        engine's (status, cost, max_depth, floor) tuple.

        Registers the sub-branch as a first-class swarm branch, then *helps*
        solve it — claiming and evaluating candidates alongside any other
        worker that needs it — until it is finalized, then reads the result
        from cache.  Never idle-blocks (the worker that needs it drives it),
        and deadlock-free (a waiting worker holds no claim).

        ceiling is the caller's alpha-beta ceiling (inf = exact required).  A
        finite ceiling is stored on the branch so every helper prunes against
        it; the solve then ends either exact (best found below the ceiling —
        cached and returned as SOLVED) or cut (everything priced out at >=
        ceiling — returned as (OVER_ERD_LIMIT, bound, None, tainted) via the
        cut_results channel unless its ceiling exceeds the budget, which proves
        and caches a loss.  A branch that already exists with
        a TIGHTER ceiling than the caller's cannot serve this caller (its cut
        would prove too little); this method then returns None and the engine
        solves the frame inline under its own ceiling — always correct, just
        unshared.
        """
        # Absolute spine of the branch being promoted, composed from the descent
        # that reached it before this frame's own work overwrites self._spine.
        child_spine = self._promoted_spine(self.root_budget - budget)
        saved_spine = self._claimed_branch_spine
        try:
            branch_key = encode_subset(words)
            n_words = len(words)
            # Already solved by someone? reuse without re-promoting.
            reuse = _cache_reuse(
                self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
            if reuse is not None:
                return (SOLVED, *reuse)
            # Already proven a loss within this budget? don't register a swarm
            # branch just to re-disprove it.
            loss_budget = self.score_cache.read_loss(branch_key, ERD_ALL)
            if loss_budget is not None and budget <= loss_budget:
                return (OVER_DEPTH_BUDGET, float('inf'), None, True)
            # Already cut at a bound this caller's ceiling accepts?  A branch
            # can carry several cuts, one per budget/taint class it has been
            # proven under (see read_cut_result); each holds at any consumer
            # budget <= its own, so any row satisfying that plus our ceiling
            # costs nothing to reuse.  No satisfying row (including the
            # exact-consumer case, ceiling == inf) means someone is about to
            # redo this branch's work — the cost side of the ceiling ledger,
            # logged against the most-useful row on record for the
            # reuse-payback question.
            cuts = self.queue.read_cut_result(branch_key)
            if cuts:
                satisfying = None
                if ceiling != float('inf'):
                    for cut_bound, cut_budget, cut_tainted in cuts:
                        if budget <= cut_budget and erd_ge(cut_bound, ceiling, n_words):
                            satisfying = (cut_bound, cut_tainted)
                            break
                if satisfying is not None:
                    cut_bound, cut_tainted = satisfying
                    return (OVER_ERD_LIMIT, cut_bound, None, cut_tainted)
                top_bound, top_budget, _ = cuts[0]
                self.queue.add_cut_reuse_miss(
                    branch_key, n_words, budget,
                    None if ceiling == float('inf') else ceiling,
                    top_bound, top_budget)

            # A user-queued branch always has an exact-result consumer, so it
            # is never solved under a ceiling: this parent shares the exact
            # solve instead of forcing the queue's copy to be redone.
            branch_ceiling = None
            if ceiling != float('inf') and not self.queue.has_pending_row(branch_key):
                branch_ceiling = ceiling
            created = self.queue.create_branch(
                branch_key, n_words, self.n_candidates,
                priority=PROMOTED_PRIORITY, source_word=self._top_source_word,
                source_pattern=self._top_source_pattern, budget=budget,
                spine=child_spine, root_budget=self.root_budget,
                ceiling=branch_ceiling)
            if not created:
                # Raced or joined an existing branch: its budget and ceiling
                # decide joinability.  The budget must match exactly — every
                # exit of the wait loop assumes the branch's outcome speaks
                # for OUR budget: a cut or loss proven at a smaller budget
                # says nothing at a larger one, and a tainted exact solved at
                # a different budget is not reusable at ours; either way the
                # deleted-branch exit below would misreport a loss.  The
                # ceiling must be NULL (exact) or >= ours: every outcome such
                # a branch can produce satisfies us, while a tighter one
                # (including any finite ceiling when we need exact) could cut
                # having proven less than we need.  Decline both — the engine
                # inlines this frame, always correct, just unshared.
                row = self.queue.get_branch(branch_key)
                if row is not None:
                    if row['budget'] != budget:
                        return None
                    row_ceiling = row['ceiling']
                    ours = None if ceiling == float('inf') else ceiling
                    if row_ceiling is not None and (
                            ours is None or not erd_ge(row_ceiling, ours, n_words)):
                        return None
            # Descents into this branch promote grandchildren relative to its spine.
            self._claimed_branch_spine = child_spine

            while not self.cancel():
                # Finished?  Check before claiming so we never touch a branch that
                # another worker just finalized and deleted.
                reuse = _cache_reuse(
                    self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
                if reuse is not None:
                    return (SOLVED, *reuse)
                loss_budget = self.score_cache.read_loss(
                    branch_key, ERD_ALL, refresh=True)
                if loss_budget is not None and budget <= loss_budget:
                    return (OVER_DEPTH_BUDGET, float('inf'), None, True)
                cut = self._read_satisfying_cut(branch_key, budget, ceiling, n_words)
                if cut is not None:
                    return cut
                if self.queue.get_branch(branch_key) is None:
                    # Finalized + deleted: a cut lands in cut_results and a
                    # ceiling-proven loss lands in the score cache before the
                    # delete, so re-check both before concluding a loss.
                    loss_budget = self.score_cache.read_loss(
                        branch_key, ERD_ALL, refresh=True)
                    if loss_budget is not None and budget <= loss_budget:
                        return (OVER_DEPTH_BUDGET, float('inf'), None, True)
                    cut = self._read_satisfying_cut(branch_key, budget, ceiling, n_words)
                    if cut is not None:
                        return cut
                    break                       # finalized as a loss + deleted
                claim = self._claim_bundle(branch_key, self.n_candidates, words)
                if claim is not None:
                    bundle_id, indices, forced = claim
                    if self.evaluate_bundle(branch_key, words, n_words, bundle_id,
                                            indices, forced, budget=budget):
                        self.maybe_finalize(branch_key, words, self.n_candidates)
                    self._maybe_checkpoint()    # drain WAL during deep solving
                elif self.queue.branch_done_candidates(branch_key) >= self.n_candidates:
                    if not self.maybe_finalize(branch_key, words,
                                               self.n_candidates):
                        self._await_rival_finalize(branch_key, words, n_words,
                                                   self.n_candidates)
                else:  # pragma: no cover
                    # Every candidate is claimed but coverage isn't complete: some
                    # are held by other workers.  Heartbeat first (so THIS worker,
                    # which still holds its own parent claim up the stack, isn't
                    # itself presumed dead while it waits), then free any claim whose
                    # holder has died so we can re-claim it rather than wait forever
                    # — there may be no supervisor in the standalone solve path.
                    self._cur_candidate = None  # coordinating, no candidate in flight
                    self._heartbeat(branch_key, n_words, None, None,
                                    None, None, force=True)
                    self.queue.reclaim_stale_claims(HB_TIMEOUT_SECONDS)
                    if not self._help_other_branch(branch_key):
                        time.sleep(0.05)        # nothing to claim anywhere; let claims land

            if self.cancel():  # pragma: no cover
                return CANCEL_RECVD
            # Finalized as a loss: proven unsolvable within budget (not a cutoff).
            return (OVER_DEPTH_BUDGET, float('inf'), None, True)  # pragma: no cover
        finally:
            self._claimed_branch_spine = saved_spine

    # -- scheduling: claim one candidate from the best available branch ------

    def claim_one(self):
        """Return (branch_row_dict, bundle_id, indices, forced) for the next
        bundle of candidates to work, or None if there is nothing to do right
        now.

        Prefers JOINING an in-progress branch (to finish branches already
        underway, concentrating workers) over PROMOTING a new one from the
        queue.  Promotion claims a pending branch and registers it so others
        can join.  A claimed branch already solved at this budget (e.g.
        synced in from elsewhere) is marked done without being promoted —
        no candidate work is needed.
        """
        for b in self.queue.branches_in_progress():
            branch_key = bytes(b['branch_key'])
            words = decode_subset(branch_key)
            claim = self._claim_bundle(branch_key, b['n_candidates'], words)
            if claim is not None:
                bundle_id, indices, forced = claim
                return dict(b), bundle_id, indices, forced
            if self.queue.branch_done_candidates(branch_key) >= b['n_candidates']:
                self.maybe_finalize(branch_key, words, b['n_candidates'])

        while True:
            claimed = self.queue.claim_next(self.name)
            if claimed is None:
                return None
            root_spine = self._root_spine(claimed['source_word'],
                                          claimed['source_pattern'])
            budget = self._spine_budget(root_spine)
            reuse = _cache_reuse(
                self.score_cache.read_with_depth(claimed['branch_key'], ERD_ALL),
                budget)
            if reuse is None:
                break
            self.queue.mark_done(claimed['branch_key'])

        n_words = claimed['n_words']
        self.queue.create_branch(
            claimed['branch_key'], n_words, self.n_candidates,
            priority=claimed['priority'], source_word=claimed['source_word'],
            source_pattern=claimed['source_pattern'], budget=budget,
            spine=root_spine, root_budget=self.root_budget)
        words = decode_subset(claimed['branch_key'])
        claim = self._claim_bundle(claimed['branch_key'], self.n_candidates, words)
        branch = {
            'branch_key': claimed['branch_key'], 'n_words': n_words,
            'n_candidates': self.n_candidates,
            'source_word': claimed['source_word'],
            'source_pattern': claimed['source_pattern'],
            'spine': root_spine,
            'budget': budget,
        }
        # A bulk-elimination sweep can complete the branch without returning
        # worker work; otherwise another worker grabbed every remaining slot.
        if claim is None:
            if (self.queue.branch_done_candidates(claimed['branch_key'])
                    >= self.n_candidates):
                self.maybe_finalize(claimed['branch_key'], words,
                                    self.n_candidates)
            return None
        bundle_id, indices, forced = claim
        return branch, bundle_id, indices, forced

    # -- main loop ----------------------------------------------------------

    def run(self):  # pragma: no cover
        idle_since = None
        while not self.cancel():
            work = self.claim_one()
            if work is None:
                # Nothing claimable: queue empty or all candidates in flight.
                if idle_since is None:
                    idle_since = time.time()
                self._cur_candidate = None      # idle, no candidate in flight
                self._heartbeat(None, None, None, None,
                                None, None, force=True)
                time.sleep(0.5)
                continue
            idle_since = None
            branch, bundle_id, indices, forced = work
            branch_key = branch['branch_key']
            # Attribute any sub-branches this worker promotes to the tree it is
            # descending (best-effort; for status display).
            self._top_source_word = branch.get('source_word')
            self._top_source_pattern = branch.get('source_pattern')
            # Base spine for deeper promotions: a joined in-progress branch carries
            # its own stored spine; a freshly promoted top branch falls back to root.
            self._claimed_branch_spine = branch.get('spine') or self._root_spine(
                branch.get('source_word'), branch.get('source_pattern'))
            words = decode_subset(branch_key)
            n_candidates = branch['n_candidates']
            if self.cancel():
                break
            completed = self.evaluate_bundle(
                branch_key, words, branch['n_words'], bundle_id, indices, forced,
                budget=self._branch_budget(branch))
            if completed:
                self.maybe_finalize(branch_key, words, n_candidates)
            self._maybe_checkpoint()
            self._check_ram()
            self._check_disk()
            self._respect_checkpoint_pause()

    # -- focused single-branch loop ------------------------------------------

    def solve_branch_focused(self, branch_key):
        """Help solve one already-registered branch to completion: claim and
        evaluate its candidates alongside any sibling workers, finalizing it
        once every candidate is done."""
        branch = self.queue.get_branch(branch_key)
        if branch is None or branch['status'] != 'open':
            return
        self._claimed_branch_spine = branch['spine']
        words = decode_subset(branch_key)
        budget = self._branch_budget(branch)
        n_candidates = branch['n_candidates']
        while not self.cancel():
            # Stop the moment the branch is finalized (and its rows deleted) by
            # any worker: otherwise claim_next_bundle, seeing no claim rows for
            # the now-deleted branch, would re-create them and redo the whole
            # branch from scratch — doubling (or worse) the work for a large
            # branch.
            if self.queue.get_branch(branch_key) is None:
                break
            claim = self._claim_bundle(branch_key, n_candidates, words)
            if claim is None:
                # Every candidate is claimed.  If coverage is complete, finalize
                # and stop.  Otherwise some claims are held by siblings — there is
                # NO supervisor in this path, so free any whose holder has died and
                # retry, rather than abandoning the branch one candidate short of
                # finalizing (which would strand it forever).
                if self.queue.branch_done_candidates(branch_key) >= n_candidates:
                    if self.maybe_finalize(branch_key, words, n_candidates):
                        break
                    self._await_rival_finalize(branch_key, words,
                                               branch['n_words'], n_candidates)
                    continue
                self._cur_candidate = None      # coordinating, no candidate in flight
                self._heartbeat(branch_key, branch['n_words'], None, None,
                                None, None, force=True)
                self.queue.reclaim_stale_claims(HB_TIMEOUT_SECONDS)
                time.sleep(0.1)
                continue
            bundle_id, indices, forced = claim
            if self.evaluate_bundle(branch_key, words, branch['n_words'], bundle_id,
                                    indices, forced, budget=budget):
                if self.queue.branch_done_candidates(branch_key) >= n_candidates:
                    if self.maybe_finalize(branch_key, words, n_candidates):
                        break


def swarm_worker(worker_id, cache_path, queue_path, stop_event,  # pragma: no cover
                 n_workers=1, enable_adaptive_decomposition=True):
    """Process entry point for a swarm worker (target= for mp.Process)."""
    # Drop the supervisor's inherited handler during startup; a signal here would
    # otherwise run the parent's handler against the shared stop_event.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _setup_logging(worker_id)
    logger.info('worker-%d starting (pid=%d)', worker_id, os.getpid())
    w = _BranchWorker(worker_id, cache_path, queue_path, stop_event,
                      n_workers=n_workers,
                      enable_adaptive_decomposition=enable_adaptive_decomposition)

    # Now that the worker exists, handle termination cooperatively: request a stop
    # so run() returns and the finally below runs close(), which clears this
    # worker's heartbeat row.  request_stop sets a process-local flag only, so a
    # single recycled worker does not stop the rest of the pool.
    def _graceful_stop(signum, frame):
        w.request_stop()
    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT, _graceful_stop)

    try:
        w.run()
    finally:
        w.close()
        logger.info('worker-%d exiting (%d claims done)',
                    worker_id, w.claims_done)


def _setup_logging(worker_id):  # pragma: no cover
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    log_path = worker_log_path(worker_id)
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    # The supervisor sends SIGUSR1 when it trips the queue WAL hard ceiling
    # (erd_search._enforce_wal_hard_ceiling); faulthandler then writes this
    # worker's all-thread stacks into its log, so a post-mortem shows exactly
    # which code path it was in when the WAL ran away.  chain=False: SIGUSR1
    # has no other handler to fall through to.
    faulthandler.register(signal.SIGUSR1, file=h.stream,
                          all_threads=True, chain=False)
