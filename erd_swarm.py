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

import logging
import math
import os
import signal
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
    cache_all_scores,
    evaluate_candidate,
    estimate_candidate_work,
    load_word_list,
    _cache_reuse,
)
from erd_queue import (ERDQueue, decode_subset, encode_subset,
                       guess_depth_from_spine,
                       DEFAULT_SMALL_COUNT, DEFAULT_COUNT_CAP,
                       DEFAULT_REPUBLISH_LIMIT)
from wordle_ui import fmt_pattern

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'

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
# amortization/crash-reclaim-window reasoning behind 8 and 500).
#
# bundle_node_cap bounds how much genuinely-heavy search a small bundle's
# strong-splitter head can accumulate before the rest of the bundle is
# handed back for re-packing against a tighter B (§7a cross-candidate
# overrun) — large enough that ordinary multi-thousand-node candidates
# finish without a spurious republish, small enough that a real heavy
# candidate hands off its siblings within seconds rather than stalling a
# whole bundle.  A bulk bundle's members are each O(1) (§3 monotonicity), so
# this cap essentially never fires for one.
BUNDLE_NODE_CAP = int(os.environ.get('BUNDLE_NODE_CAP', '50000'))
# bundle_wall_cap_seconds is the only cap a bulk bundle can hit, since its
# members can't cost nodes: it exists purely to bound the crash-reclaim
# window (HB_TIMEOUT_SECONDS) a dead worker's held bundle would otherwise
# widen, not to catch misprediction — there is no prediction anywhere in
# this design (§3).
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
        (nodes_at_entry, predicted, entry_time, branch_words, budget) for any
        non-trivial frame (>= MIN_PUBLISH_BRANCH_WORDS answer words).  predicted
        may be None (cold model) — the node-proportionate check then can't arm,
        but the wall-clock backstop in check() still fires off entry_time, and
        record_inline() still warms the model on frame completion.
        """
        n = len(branch_words)
        if n < MIN_PUBLISH_BRANCH_WORDS:
            return None
        predicted = self._worker._typical(n, budget)
        return (self._worker._nodes, predicted, time.time(), branch_words, budget)

    def check(self, token, candidate_list, last_index,
              best_guess, best_erd, budget):
        """Called every loop iteration (before status-continue checks).

        candidate_list is the frame's full ordered candidate list and last_index
        is the index just evaluated, so remaining_count is derived cheaply and
        the evaluated prefix is sliced only on the rare iteration the overrun
        actually fires (avoiding an O(n²) per-iteration copy).

        Fires on either of two triggers:
        - node-proportionate: the frame has spent > OVERRUN_K * predicted nodes
          since enter() (warm model only — disabled when predicted is None);
        - wall-clock backstop: the frame has run longer than COLD_BACKSTOP_SECONDS
          since enter() (always armed, the only guard while the model is cold).

        When either fires and enough candidates remain to be worth handing off,
        emits the promotion sentinel, publishes the remainder as a cooperative
        branch, and returns the cooperative result so the engine can short-
        circuit.  Returns None to continue inline.
        """
        if token is None:
            return None
        nodes_at_entry, predicted, entry_time, branch_words, entry_budget = token
        delta = self._worker._nodes - nodes_at_entry
        node_overrun = predicted is not None and delta > OVERRUN_K * predicted
        elapsed = time.time() - entry_time
        time_overrun = elapsed > COLD_BACKSTOP_SECONDS
        if not (node_overrun or time_overrun):
            return None
        remaining_count = len(candidate_list) - (last_index + 1)
        if remaining_count < MIN_HANDOFF_CANDIDATES:
            return None

        n = len(branch_words)
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
        branch_key = encode_subset(branch_words)

        # Spine sentinel: mark this frame as handed off in the heartbeat display.
        self._worker._note_depth(entry_budget, -n, None, None)

        # Create the cooperative branch; idempotent if another worker raced us.
        # First writer for this path, so the composed spine must be supplied here
        # (cooperative_solve's later create_branch is a no-op once this row exists).
        self._worker.queue.create_branch(
            branch_key, n, self._worker.n_candidates,
            priority=PROMOTED_PRIORITY,
            source_word=self._worker._top_source_word,
            source_pattern=self._worker._top_source_pattern,
            budget=budget, spine=self._worker._promoted_spine(),
            root_budget=self._worker.root_budget)

        # Mark the already-evaluated candidates done by their all_words index so
        # cooperative workers claim only the unevaluated remainder.  The prefix
        # slice is built here — only when an overrun actually fires — not on
        # every loop iteration.
        word_idx = self._worker._word_idx
        done_indices = [word_idx[w] for w in candidate_list[:last_index + 1]
                        if w in word_idx]
        if done_indices:
            self._worker.queue.mark_claims_done(branch_key, done_indices)

        # Seed the cooperative branch's bound only when we have an achieved cost —
        # a None best_guess means no feasible candidate yet; seeding inf is a no-op.
        if best_guess is not None:
            self._worker.queue.update_branch_best(branch_key, best_guess, best_erd)

        return self._worker.cooperative_solve(branch_words, budget)

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
        nodes_at_entry, _predicted, _entry_time, branch_words, entry_budget = token
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
        self._last_ram_check = time.time()
        self._cand_max_depth = 0     # deepest guess_depth reached this candidate
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
        self.queue.clear_heartbeat(self.name)
        self.score_cache.checkpoint()
        self.score_cache.close()
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

    def _promoted_spine(self):
        """Absolute root -> promoted-branch spine: the claimed branch's base plus
        the live descent guesses (guess_depth-ordered "GUESS pattern" tokens).
        Returns None when the base is unknown, leaving the branch row to fall back
        to the source word.  Sentinel/size-only spine entries (no guess) are skipped.
        """
        base = getattr(self, '_claimed_branch_spine', None)
        if not base:
            return None
        # The live descent dict keeps a shallow entry until a shallower frame
        # overwrites it, so after the base is advanced it still holds the guess
        # that reached the base — whose edge then repeats the base's tail.  A
        # real spine never replays a guess (a replay gains no information and is
        # never selected), so an edge identical to the one before it is always
        # that seam artifact.  Dropping it keeps budget + guess_depth = GAME_GUESSES.
        edges = base.split()   # flat "GUESS pattern GUESS pattern ..." tokens
        for d in sorted(getattr(self, '_spine', {})):
            _size, guess, pattern = self._spine[d]
            if not (guess and guess != '•' and pattern):
                continue
            guess = guess.upper()
            if len(edges) >= 2 and edges[-2] == guess and edges[-1] == pattern:
                continue
            edges.extend((guess, pattern))
        return ' '.join(edges)

    # -- cost model ---------------------------------------------------------

    def _typical(self, n, budget=None):
        """Return the cost model's geometric-mean node count for sub-branches of
        size n at remaining-guess `budget`, or None when the model is cold.

        The model is keyed on (size, budget); a cold (size, budget) cell falls
        back to the budget-aggregate inside the queue.  Results are cached
        in-memory keyed by (n, budget) for the life of the worker; entries are
        invalidated on cooperative finalize so new samples take effect without
        re-querying on every enter() call.
        """
        key = (n, budget)
        if key in self._typical_cache:
            return self._typical_cache[key]
        result = self.queue.get_cost_typical(ERD_ALL, n, budget)
        self._typical_cache[key] = result
        return result

    def _update_cost_model(self, n_words, nodes, budget=None, wall_millis=None):
        """Update the cost model with a finalized cooperative branch's node count.

        budget keys the (size, budget) cell (and the aggregate); wall_millis is
        the branch's wall span, the only per-solve wall figure, recorded on the
        raw sample.
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
        now = time.time()
        if force or now - self._last_checkpoint > CHECKPOINT_SECONDS:
            self._flush_cost_model_buffer()
            self.score_cache.checkpoint()
            self._last_checkpoint = now

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
                   bound_erd=None, cur_candidate=None):
        # Count every invocation (one per node) BEFORE the throttle, so the
        # node counter is exact even though we only write every HB_SECONDS.
        self._nodes += 1
        now = time.time()
        if not force and now - self._last_hb < HB_SECONDS:
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
            cur_candidate=cur_candidate,
            cur_max_depth=self._cand_max_depth,
            cur_nodes=self._nodes, node_rate=node_rate,
            cur_path=self._hb_spine_str())
        self._hb_max_spine = {}
        if cur_candidate and now - self._last_progress_log >= PROGRESS_LOG_SECONDS:  # pragma: no cover
            self._last_progress_log = now
            be = f'{best_erd:.4f}' if best_erd is not None else '-'
            bg = (best_guess or '-').upper()
            logger.info('%s claim %d: %s in progress  '
                        'guess_depth=%d path=%s  %.1fM nodes %.0fk/s  best=%s %s',
                        self.name, claim_idx,
                        cur_candidate.upper(),
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
        — see ERDQueue.claim_next_bundle."""
        order, cost_lower_bound = self._packing_stats(branch_key, words)
        return self.queue.claim_next_bundle(
            branch_key, self.name, n_candidates, order, cost_lower_bound,
            small_count=self.small_count, count_cap=self.count_cap,
            republish_limit=self.republish_limit)

    # -- evaluate one candidate claim ---------------------------------------

    def evaluate_claim(self, branch_key, words, n_words, idx, budget=None):
        """Evaluate the single candidate self.all_words[idx] against branch_key.

        Folds the result into the branch's shared best and marks the claim
        done=1.  Returns True if the candidate was fully evaluated; False if
        cancelled mid-evaluation (claim left done=0 for reclaim/redo).

        budget is the branch's guess budget (depth-limited ERD): a candidate
        whose strategy can't win within budget is infeasible (and taints the
        branch — see ERDQueue.mark_branch_tainted).
        """
        candidate = self.all_words[idx]
        self._hb_max_spine = {}
        self._log_max_spine = {}
        local_candidate, local_best = self.queue.read_branch_best(branch_key)
        local_md = None
        shared_best = local_best
        last_refresh = time.time()
        claim_started = int(time.time())
        t0 = time.time()

        def _bound_provider():
            # Refreshes shared_best from the queue at most every
            # BEST_REFRESH_SECONDS, then returns the tightest known bound as a
            # float (inf when nothing is known yet).  Called by evaluate_candidate
            # at each sub-branch level throughout the full recursion.
            nonlocal shared_best, last_refresh
            now = time.time()
            if now - last_refresh > BEST_REFRESH_SECONDS:
                _, new = self.queue.read_branch_best(branch_key)
                if new is not None and (shared_best is None or new < shared_best):
                    shared_best = new
                last_refresh = now
            bests = [b for b in (local_best, shared_best) if b is not None]
            return min(bests) if bests else float('inf')

        def _eff_bound():
            v = _bound_provider()
            return None if v == float('inf') else v

        self._heartbeat(branch_key, n_words, idx, claim_started,
                        local_candidate, local_best, force=True,
                        bound_erd=_eff_bound(), cur_candidate=candidate)

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
                local_candidate, local_best, bound_erd=_eff_bound(),
                cur_candidate=candidate))
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
                n_words, int(full_coord_seconds * 1e3), nodes_delta, self.n_workers)
        self.claims_done += 1
        self._heartbeat(branch_key, n_words, idx, claim_started,
                        local_candidate, local_best, bound_erd=_eff_bound(),
                        force=True)
        return True

    # -- evaluate a packer-issued bundle of candidate claims -----------------

    def _evaluate_bundle_member(self, branch_key, words, n_words, idx, budget,
                                bundle_id, nodes_at_bundle_start, wall_t0):
        """evaluate_claim for one bundle member; on cancellation/abort,
        records the bundle as censored.  Returns True to keep going, False
        for the caller to abort evaluate_bundle immediately."""
        if self.cancel():
            self._finish_bundle(branch_key, bundle_id, nodes_at_bundle_start,
                                wall_t0, censored=True)
            return False
        if not self.evaluate_claim(branch_key, words, n_words, idx,
                                   budget=budget):
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
        claims.  A bulk bundle's members are each O(1) (§3 monotonicity), so
        in practice only the wall cap ever fires for one; the node cap is
        what protects a small bundle's strong-splitter head from stranding
        its siblings.

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
        for pos, idx in enumerate(indices):
            if not self._evaluate_bundle_member(
                    branch_key, words, n_words, idx, budget, bundle_id,
                    nodes_at_bundle_start, wall_t0):
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
                                bundle_id, nodes_at_bundle_start, wall_t0):
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

    def maybe_finalize(self, branch_key, words, n_candidates):
        """If every candidate is done, finalize the branch exactly once."""
        if self.queue.branch_done_candidates(branch_key) < n_candidates:
            return
        if not self.queue.try_finalize_branch(branch_key):  # pragma: no cover
            return  # another worker won the finalize
        meta = self.queue.read_branch_meta(branch_key)
        best_guess, best_erd, max_depth, tainted, budget = meta
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
        n_claims = self.queue.branch_done_candidates(branch_key)
        if best_guess is not None:
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
        else:  # pragma: no cover
            # No feasible guess within budget: this branch is a loss.  There is
            # no winning strategy to record in branch_best_by_policy, but the
            # loss itself is persisted so the branch is never re-swept at this
            # (or any smaller) budget.
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
        n_bundles, max_bundle_nodes, total_bundle_wall_millis, censored_units = (
            self.queue.finalize_bundle_stats(branch_key))
        self.queue.add_branch_finalize_log(
            branch_key, spine, len(words), budget, created_at, finalized_at,
            nodes_spent, n_claims, n_bundles=n_bundles,
            max_bundle_nodes=max_bundle_nodes,
            total_bundle_wall_millis=total_bundle_wall_millis,
            censored_units=censored_units)
        self.queue.mark_done(branch_key)        # pending_branches row -> done
        self.queue.delete_branch(branch_key)    # drop transient coordination
        self._packing_stats_cache.pop(branch_key, None)

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

    def _subbranch_solver(self, words, budget):
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
        Returns the engine's (status, cost, max_depth, floor) tuple, or None to
        inline.  Cooperative results are always exact (no cutoff).
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
            return self.cooperative_solve(words, budget) if n >= PROMOTE_MIN_SIZE \
                else None
        predicted = self._typical(n)
        if predicted is None:
            # Cold model: promote large branches by size, inline small ones; the
            # wall-clock backstop bounds a cold inline tarpit.
            return self.cooperative_solve(words, budget) if n >= PROMOTE_MIN_SIZE \
                else None
        if predicted < self._publish_threshold():
            return None
        return self.cooperative_solve(words, budget)

    def cooperative_solve(self, words, budget):
        """Solve sub-branch `words` at `budget` cooperatively, returning the
        engine's (cost, max_depth, floor, cutoff) tuple (cutoff always False —
        a cooperative solve runs to the exact optimum).

        Registers the sub-branch as a first-class swarm branch, then *helps*
        solve it — claiming and evaluating candidates alongside any other
        worker that needs it — until it is finalized, then reads the result
        from cache.  Never idle-blocks (the worker that needs it drives it),
        and deadlock-free (a waiting worker holds no claim).
        """
        # Absolute spine of the branch being promoted, composed from the descent
        # that reached it before this frame's own work overwrites self._spine.
        child_spine = self._promoted_spine()
        saved_spine = self._claimed_branch_spine
        try:
            branch_key = encode_subset(words)
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

            n_words = len(words)
            self.queue.create_branch(
                branch_key, n_words, self.n_candidates,
                priority=PROMOTED_PRIORITY, source_word=self._top_source_word,
                source_pattern=self._top_source_pattern, budget=budget,
                spine=child_spine, root_budget=self.root_budget)
            # Descents into this branch promote grandchildren relative to its spine.
            self._claimed_branch_spine = child_spine

            while not self.cancel():
                # Finished?  Check before claiming so we never touch a branch that
                # another worker just finalized and deleted.
                reuse = _cache_reuse(
                    self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
                if reuse is not None:
                    return (SOLVED, *reuse)
                if self.queue.get_branch(branch_key) is None:
                    break                       # finalized as a loss + deleted
                claim = self._claim_bundle(branch_key, self.n_candidates, words)
                if claim is not None:
                    bundle_id, indices, forced = claim
                    if self.evaluate_bundle(branch_key, words, n_words, bundle_id,
                                            indices, forced, budget=budget):
                        self.maybe_finalize(branch_key, words, self.n_candidates)
                    self._maybe_checkpoint()    # drain WAL during deep solving
                elif self.queue.branch_done_candidates(branch_key) >= self.n_candidates:
                    self.maybe_finalize(branch_key, words, self.n_candidates)
                else:  # pragma: no cover
                    # Every candidate is claimed but coverage isn't complete: some
                    # are held by other workers.  Heartbeat first (so THIS worker,
                    # which still holds its own parent claim up the stack, isn't
                    # itself presumed dead while it waits), then free any claim whose
                    # holder has died so we can re-claim it rather than wait forever
                    # — there may be no supervisor in the standalone solve path.
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
        # claim can be None only if another worker grabbed every candidate
        # between create and claim — rare; treat as "nothing for me right now".
        if claim is None:  # pragma: no cover
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

    # -- focused single-branch loop ------------------------------------------

    def solve_branch_focused(self, branch_key):
        """Help solve one already-registered branch to completion: claim and
        evaluate its candidates alongside any sibling workers, finalizing it
        once every candidate is done."""
        branch = self.queue.get_branch(branch_key)
        if branch is None or branch['status'] != 'open':
            return
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
                    self.maybe_finalize(branch_key, words, n_candidates)
                    break
                self._heartbeat(branch_key, branch['n_words'], None, None,
                                None, None, force=True)
                self.queue.reclaim_stale_claims(HB_TIMEOUT_SECONDS)
                time.sleep(0.1)
                continue
            bundle_id, indices, forced = claim
            if self.evaluate_bundle(branch_key, words, branch['n_words'], bundle_id,
                                    indices, forced, budget=budget):
                if self.queue.branch_done_candidates(branch_key) >= n_candidates:
                    self.maybe_finalize(branch_key, words, n_candidates)
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
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'erd_worker_{worker_id}.log')
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
