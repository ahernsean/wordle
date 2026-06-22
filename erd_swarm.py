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

PROMOTE_MIN_SIZE = 60   # a sub-branch with >= this many words is promoted to the
                        # queue and solved cooperatively; smaller ones are solved
                        # inline (cheap, and visible — the engine heartbeat is
                        # already threaded through the recursion).  Pure
                        # granularity knob; any reasonable value is correct.
PROMOTED_PRIORITY = 1_000_000  # promoted sub-branches outrank fresh top branches
                               # so freed workers prefer joining in-flight depth.

GAME_GUESSES = 6              # a Wordle game allows 6 guesses total
# Queue branches are positions AFTER the opener (guess 1), so they are solved
# under the remaining budget.  Depth-limited ERD: a branch unsolvable within
# this many guesses is a loss, not a finite cost.
ROOT_BUDGET = GAME_GUESSES - 1

logger = logging.getLogger('wordle')


class _BranchWorker:
    """One worker process's state and operations on branches/chunks."""

    def __init__(self, worker_id, cache_path, queue_path, stop_event,
                 budget=ROOT_BUDGET, n_workers=1):
        self.name = f'worker-{worker_id}'
        self.stop_event = stop_event
        self.budget = budget

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
                   cand_rate, best_guess, best_erd, force=False,
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
            claim_started_at=claim_started_at, cand_rate=cand_rate,
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

        self._heartbeat(branch_key, n_words, idx, claim_started, None,
                        local_candidate, local_best, force=True,
                        bound_erd=_eff_bound(), cur_candidate=candidate)

        if self.cancel():
            return False

        self._cand_max_depth = 0
        cand_t0 = time.time()
        status, cost, cand_md, budget_tainted = evaluate_candidate(
            words, candidate, self.rcache, self.score_cache,
            n=n_words, best_erd=float('inf'), guesses=self.all_words,
            policy=ERD_ALL, cancel_check=self.cancel,
            depth=0, note_depth=self._note_depth, budget=budget,
            subbranch_solver=self._subbranch_solver,
            bound_provider=_bound_provider,
            heartbeat=lambda: self._heartbeat(
                branch_key, n_words, idx, claim_started, None,
                local_candidate, local_best, bound_erd=_eff_bound(),
                cur_candidate=candidate))
        cand_elapsed = time.time() - cand_t0
        if cand_elapsed > 10:  # pragma: no cover
            logger.warning('%s slow candidate %s (idx=%d): %.1fs  '
                           'status=%s  max_depth=%d', self.name, candidate,
                           idx, cand_elapsed, status, self._cand_max_depth)

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
        self.claims_done += 1
        self._heartbeat(branch_key, n_words, idx, claim_started,
                        1.0 / max(1e-6, elapsed),
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
        if best_guess is not None:
            # Untainted => unconstrained optimum, reusable at any budget >=
            # max_depth (solve_budget NULL).  Tainted => valid only at this
            # budget (solve_budget = budget).  See cache_sqlite reuse rule.
            solve_budget = budget if tainted else None
            self.score_cache.write(branch_key, ERD_ALL, best_guess, best_erd,
                                   max_depth=max_depth, solve_budget=solve_budget)
            cache_all_scores(best_guess, words, self.score_cache, branch_key,
                             cache=self.rcache)
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

        Large sub-branches are promoted to the swarm and solved across workers;
        small ones return None so the engine solves them inline (cheap, and
        still visible — the heartbeat is threaded through that recursion).
        Returns the engine's (cost, max_depth, floor, cutoff) tuple, or None to
        inline.  Cooperative results are always exact, so cutoff is False.
        """
        if budget is None or len(words) < PROMOTE_MIN_SIZE:
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
                    self._heartbeat(branch_key, n_words, None, None, None,
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
                self._heartbeat(None, None, None, None, None, None, None,
                                force=True)
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
                self._heartbeat(branch_key, branch['n_words'], None, None, None,
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
                 n_workers=1):
    """Process entry point for a swarm worker (target= for mp.Process)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _setup_logging(worker_id)
    logger.info('worker-%d starting (pid=%d)', worker_id, os.getpid())
    w = _BranchWorker(worker_id, cache_path, queue_path, stop_event,
                      n_workers=n_workers)
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
