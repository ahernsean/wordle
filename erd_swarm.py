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

from cache_sqlite import ScoreCache, mem_cache_limit
from wordle_engine import (
    ERD_ALL,
    ResponseCache,
    CANCEL_RECVD,
    SOLVED,
    OVER_DEPTH_BUDGET,
    OVER_ERD_LIMIT,
    NO_INFORMATION,
    _ABORT_STATUSES,
    cache_all_scores,
    evaluate_candidate,
    load_word_list,
    _cache_reuse,
)
from erd_queue import ERDQueue, decode_subset, encode_subset
from wordle_ui import fmt_pattern

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'

BEST_REFRESH_SECONDS = 0.25   # how often a worker re-reads the shared bound
HB_SECONDS = 2.0              # liveness heartbeat cadence during a long chunk
# A worker that hasn't heartbeat within this many seconds is presumed dead, and
# only then are its in-flight chunk claims reclaimed.  Live workers heartbeat
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

GAME_GUESSES = 6              # a Wordle game allows 6 guesses total
# Queue branches are positions AFTER the opener (guess 1), so they are solved
# under the remaining budget.  Depth-limited ERD: a branch unsolvable within
# this many guesses is a loss, not a finite cost.
ROOT_BUDGET = GAME_GUESSES - 1

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

    def enter(self, branch_words, depth):
        """Called just before the candidate loop of each _solve_subset frame.

        Returns an opaque token
        (nodes_at_entry, predicted, entry_time, branch_words, depth) for any
        non-trivial frame (>= MIN_PUBLISH_BRANCH_WORDS answer words).  predicted
        may be None (cold model) — the node-proportionate check then can't arm,
        but the wall-clock backstop in check() still fires off entry_time, and
        record_inline() still warms the model on frame completion.
        """
        n = len(branch_words)
        if n < MIN_PUBLISH_BRANCH_WORDS:
            return None
        predicted = self._worker._typical(n)
        return (self._worker._nodes, predicted, time.time(), branch_words, depth)

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
        nodes_at_entry, predicted, entry_time, branch_words, depth = token
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
                n, depth, int(elapsed * 1000), delta, predicted, remaining_count)
        branch_key = encode_subset(branch_words)

        # Spine sentinel: mark this frame as handed off in the heartbeat display.
        self._worker._note_depth(depth, -n, None, None)

        # Create the cooperative branch; idempotent if another worker raced us.
        self._worker.queue.create_branch(
            branch_key, n, self._worker.n_candidates,
            priority=PROMOTED_PRIORITY,
            source_word=self._worker._top_source_word,
            source_pattern=self._worker._top_source_pattern,
            budget=budget, depth=self._worker._coop_depth + 1)

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
        nodes_at_entry, _predicted, _entry_time, branch_words, _depth = token
        n = len(branch_words)
        nodes = self._worker._nodes - nodes_at_entry
        if nodes <= 0:
            return
        log_n = math.log(nodes)
        buf = self._worker._cost_model_buffer
        if n in buf:
            s, sq, c = buf[n]
            buf[n] = (s + log_n, sq + log_n * log_n, c + 1)
        else:
            buf[n] = (log_n, log_n * log_n, 1)


class _BranchWorker:
    """One worker process's state and operations on branches/chunks."""

    def __init__(self, worker_id, cache_path, queue_path, stop_event,
                 budget=ROOT_BUDGET, n_workers=1,
                 enable_adaptive_decomposition=True):
        self.name = f'worker-{worker_id}'
        self.stop_event = stop_event
        self.budget = budget
        self.n_workers = n_workers
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
        self._cand_max_depth = 0
        # Live search probe (transparency): a monotonic node counter plus the
        # active recursion spine, so a long candidate evaluation never looks
        # frozen — node count climbs every heartbeat even mid-candidate.
        self._nodes = 0              # candidate evaluations this chunk
        self._nodes_at_last_hb = 0
        self._cur_depth = 0
        self._spine = {}             # depth -> subset size on the active descent
        self._hb_max_spine = {}      # deepest spine since last heartbeat (→ DB)
        self._log_max_spine = {}     # deepest spine since last progress log (→ log file)
        self._last_progress_log = 0.0
        # Attribution for promoted sub-branches: which top-level (opener,pattern)
        # tree the worker is currently descending.
        self._top_source_word = None
        self._top_source_pattern = None
        # Cooperative nesting depth: incremented on entry to cooperative_solve,
        # decremented on exit.  Passed to create_branch so the status display
        # can distinguish user-queued branches (depth 0) from sub-branches.
        self._coop_depth = 0
        # In-memory cache of cost-model predictions keyed by sub-branch size.
        # Cleared on any cost-model write so new samples take effect.
        self._typical_cache = {}
        # In-memory buffer for inline node-cost samples:
        # {n: (sum_log, sum_log_sq, count)}.  Flushed to the DB at each
        # checkpoint to avoid per-frame SQLite writes.
        self._cost_model_buffer: dict[int, tuple[float, float, int]] = {}
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

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        self.queue.clear_heartbeat(self.name)
        self.score_cache.checkpoint()
        self.score_cache.close()
        self.queue.close()

    def cancel(self):
        return self.stop_event is not None and self.stop_event.is_set()

    # -- depth instrumentation ----------------------------------------------

    def _note_depth(self, depth, n, guess=None, pattern=None):
        # Per-position observer: track max depth and the live recursion spine.
        # Each entry stores (size, guess, pattern): the sub-branch size and the
        # candidate+response that produced it.  n < 0 is the cooperative-
        # promotion sentinel: preserve the stored guess/pattern, replace size
        # with '•' to mark that the sub-branch was handed to the swarm.
        if n < 0:
            prev = self._spine.get(depth, (None, None, None))
            self._spine[depth] = ('•', prev[1], prev[2])
            deeper = [d for d in self._spine if d > depth]
            for d in deeper:
                del self._spine[d]
            if len(self._spine) >= len(self._hb_max_spine):
                self._hb_max_spine = dict(self._spine)
            if len(self._spine) >= len(self._log_max_spine):
                self._log_max_spine = dict(self._spine)
            return
        if depth > self._cand_max_depth:
            self._cand_max_depth = depth
        self._cur_depth = depth
        pattern_str = fmt_pattern(pattern) if isinstance(pattern, int) else pattern
        self._spine[depth] = (n, guess, pattern_str)
        deeper = [d for d in self._spine if d > depth]
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

    def _hb_spine_str(self):
        return '→'.join(
            self._fmt_spine_entry(self._hb_max_spine[d])
            for d in sorted(self._hb_max_spine))

    def _log_spine_str(self):
        return '→'.join(
            self._fmt_spine_entry(self._log_max_spine[d])
            for d in sorted(self._log_max_spine))

    # -- cost model ---------------------------------------------------------

    def _typical(self, n):
        """Return the cost model's geometric-mean node count for sub-branches of
        size n, or None when the model is cold for this size bucket.

        Results are cached in-memory for the life of the worker; the cache entry
        for a given size is invalidated on cooperative finalize so new samples
        take effect without re-querying on every enter() call.
        """
        if n in self._typical_cache:
            return self._typical_cache[n]
        result = self.queue.get_cost_typical(ERD_ALL, n)
        self._typical_cache[n] = result
        return result

    def _update_cost_model(self, n_words, nodes):
        """Update the cost model with a finalized cooperative branch's node count."""
        self.queue.update_cost_model(ERD_ALL, n_words, nodes)
        self.queue.add_cost_sample(ERD_ALL, n_words, nodes, 'finalize')
        self._typical_cache.clear()   # bucket changed: drop cached predictions

    def _flush_cost_model_buffer(self):
        """Flush the in-memory inline-sample buffer to the DB and clear it.

        The buffered (Σ ln, Σ ln², count) accumulators are folded straight into
        the cost model, so each sample's magnitude reaches the geometric mean and
        the second log-moment without an exp/int/log round-trip.
        """
        for n, (sum_log, sum_log_sq, count) in self._cost_model_buffer.items():
            self.queue.update_cost_model_logsums(
                ERD_ALL, n, sum_log, sum_log_sq, float(count))
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
                        'depth=%d path=%s  %.1fM nodes %.0fk/s  best=%s %s',
                        self.name, claim_idx,
                        cur_candidate.upper(),
                        self._cand_max_depth, self._log_spine_str(),
                        self._nodes / 1e6, node_rate / 1e3, bg, be)
            self._log_max_spine = {}
            self._cand_max_depth = 0

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
        status, cost, cand_md, budget_tainted = evaluate_candidate(
            words, candidate, self.rcache, self.score_cache,
            n=n_words, best_erd=float('inf'), guesses=self.all_words,
            policy=ERD_ALL, cancel_check=self.cancel,
            depth=0, note_depth=self._note_depth, budget=budget,
            subbranch_solver=self._subbranch_solver,
            bound_provider=_bound_provider,
            mid_loop_publisher=self._mid_loop_publisher,
            heartbeat=lambda: self._heartbeat(
                branch_key, n_words, idx, claim_started,
                local_candidate, local_best, bound_erd=_eff_bound(),
                cur_candidate=candidate))
        cand_elapsed = time.time() - cand_t0
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
            # Feed the in-memory estimators behind the adaptive publish threshold
            # in seconds (the ratio is unit-free).  These are the control-path
            # twins of the outbound claim_telemetry row, which the table never
            # feeds back; the table stores milliseconds for readable offline rows.
            self._coord_ema.add(coord_seconds)
            if nodes_delta > 0 and cand_elapsed > 0:
                self._node_time_ema.add(cand_elapsed / nodes_delta)
            self.queue.add_claim_telemetry(
                n_words, int(coord_seconds * 1e3), nodes_delta, self.n_workers)
        self.claims_done += 1
        self._heartbeat(branch_key, n_words, idx, claim_started,
                        local_candidate, local_best, bound_erd=_eff_bound(),
                        force=True)
        return True

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
                self._update_cost_model(len(words), nodes_spent)
            # NB: no per-finalize checkpoint — with recursive promotion a worker
            # finalizes thousands of sub-branches; checkpointing each one is
            # ruinous.  WAL is drained by the periodic _maybe_checkpoint instead.
            logger.info('%s finalized branch (%d words) -> %s erd=%.4f '
                        'max_depth=%s budget=%s%s', self.name, len(words),
                        best_guess, best_erd, max_depth, budget,
                        ' TAINTED' if tainted else '')
        else:  # pragma: no cover
            # No feasible guess within budget: this branch is a loss.  Don't
            # write an ERD entry (there is no winning strategy to cache).
            logger.warning('%s branch (%d words) UNSOLVABLE within budget %s '
                           '(loss) src=%s', self.name, len(words), budget,
                           branch_key[:25])
        self.queue.mark_done(branch_key)        # pending_branches row -> done
        self.queue.delete_branch(branch_key)    # drop transient coordination

    # -- recursive cooperative solving --------------------------------------

    def _help_other_branch(self, exclude_branch_key: bytes) -> bool:
        """Evaluate one candidate claim from any open branch other than exclude_branch_key.

        Called when the worker is waiting on a dependency branch whose remaining
        candidates are all held by other workers.  Instead of sleeping, the
        worker drains useful work from the queue.  Returns True if a candidate
        was evaluated, False if there was nothing to claim.
        """
        for branch in self.queue.branches_in_progress():
            other_key = bytes(branch['branch_key'])
            if other_key == bytes(exclude_branch_key):
                continue
            n_candidates = branch['n_candidates']
            idx = self.queue.claim_candidate(other_key, self.name, n_candidates)
            if idx is None:
                continue
            words = decode_subset(other_key)
            budget = branch['budget'] or self.budget
            if self.evaluate_claim(other_key, words, branch['n_words'], idx,
                                   budget=budget):
                self.maybe_finalize(other_key, words, n_candidates)
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
        solve it — claiming and evaluating its chunks alongside any other
        worker that needs it — until it is finalized, then reads the result
        from cache.  Never idle-blocks (the worker that needs it drives it),
        and deadlock-free (a waiting worker holds no chunk).
        """
        self._coop_depth += 1
        try:
            branch_key = encode_subset(words)
            # Already solved by someone? reuse without re-promoting.
            reuse = _cache_reuse(
                self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
            if reuse is not None:
                return (SOLVED, *reuse)

            n_words = len(words)
            self.queue.create_branch(
                branch_key, n_words, self.n_candidates,
                priority=PROMOTED_PRIORITY, source_word=self._top_source_word,
                source_pattern=self._top_source_pattern, budget=budget,
                depth=self._coop_depth)

            while not self.cancel():
                # Finished?  Check before claiming so we never touch a branch that
                # another worker just finalized and deleted.
                reuse = _cache_reuse(
                    self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
                if reuse is not None:
                    return (SOLVED, *reuse)
                if self.queue.get_branch(branch_key) is None:
                    break                       # finalized as a loss + deleted
                idx = self.queue.claim_candidate(branch_key, self.name,
                                                 self.n_candidates)
                if idx is not None:
                    if self.evaluate_claim(branch_key, words, n_words, idx,
                                           budget=budget):
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
            self._coop_depth -= 1

    # -- scheduling: claim one chunk of the best available branch -----------

    def claim_one(self):
        """Return (branch_row_dict, claim_idx) for the next candidate to work,
        or None if there is nothing to do right now.

        Prefers JOINING an in-progress branch (to finish branches already
        underway, concentrating workers) over PROMOTING a new one from the
        queue.  Promotion claims a pending branch and registers it so others
        can join.  A claimed branch already solved at this budget (e.g.
        synced in from elsewhere) is marked done without being promoted —
        no candidate work is needed.
        """
        for b in self.queue.branches_in_progress():
            idx = self.queue.claim_candidate(b['branch_key'], self.name,
                                             b['n_candidates'])
            if idx is not None:
                return dict(b), idx

        while True:
            claimed = self.queue.claim_next(self.name)
            if claimed is None:
                return None
            reuse = _cache_reuse(
                self.score_cache.read_with_depth(claimed['branch_key'], ERD_ALL),
                self.budget)
            if reuse is None:
                break
            self.queue.mark_done(claimed['branch_key'])

        n_words = claimed['n_words']
        self.queue.create_branch(
            claimed['branch_key'], n_words, self.n_candidates,
            priority=claimed['priority'], source_word=claimed['source_word'],
            source_pattern=claimed['source_pattern'], budget=self.budget)
        idx = self.queue.claim_candidate(claimed['branch_key'], self.name,
                                         self.n_candidates)
        branch = {
            'branch_key': claimed['branch_key'], 'n_words': n_words,
            'n_candidates': self.n_candidates,
            'source_word': claimed['source_word'],
            'source_pattern': claimed['source_pattern'],
            'budget': self.budget,
        }
        # idx can be None only if another worker grabbed every candidate between
        # create and claim — rare; treat as "nothing for me right now".
        return (branch, idx) if idx is not None else None

    # -- main loop ----------------------------------------------------------

    def run(self):  # pragma: no cover
        idle_since = None
        while not self.cancel():
            work = self.claim_one()
            if work is None:
                # Nothing claimable: queue empty or all chunks in flight.
                if idle_since is None:
                    idle_since = time.time()
                self._heartbeat(None, None, None, None,
                                None, None, force=True)
                time.sleep(0.5)
                continue
            idle_since = None
            branch, idx = work
            branch_key = branch['branch_key']
            # Attribute any sub-branches this worker promotes to the tree it is
            # descending (best-effort; for status display).
            self._top_source_word = branch.get('source_word')
            self._top_source_pattern = branch.get('source_pattern')
            words = decode_subset(branch_key)
            n_candidates = branch['n_candidates']
            if self.cancel():
                break
            completed = self.evaluate_claim(
                branch_key, words, branch['n_words'], idx,
                budget=branch.get('budget') or self.budget)
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
        budget = branch['budget'] or ROOT_BUDGET
        n_candidates = branch['n_candidates']
        while not self.cancel():
            # Stop the moment the branch is finalized (and its rows deleted) by
            # any worker: otherwise claim_candidate, seeing no claim rows for the
            # now-deleted branch, would re-create them and redo the whole branch
            # from scratch — doubling (or worse) the work for a large branch.
            if self.queue.get_branch(branch_key) is None:
                break
            idx = self.queue.claim_candidate(branch_key, self.name, n_candidates)
            if idx is None:
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
            if self.evaluate_claim(branch_key, words, branch['n_words'], idx,
                                   budget=budget):
                if self.queue.branch_done_candidates(branch_key) >= n_candidates:
                    self.maybe_finalize(branch_key, words, n_candidates)
                    break


def swarm_worker(worker_id, cache_path, queue_path, stop_event,  # pragma: no cover
                 n_workers=1, enable_adaptive_decomposition=True):
    """Process entry point for a swarm worker (target= for mp.Process)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _setup_logging(worker_id)
    logger.info('worker-%d starting (pid=%d)', worker_id, os.getpid())
    w = _BranchWorker(worker_id, cache_path, queue_path, stop_event,
                      n_workers=n_workers,
                      enable_adaptive_decomposition=enable_adaptive_decomposition)
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
