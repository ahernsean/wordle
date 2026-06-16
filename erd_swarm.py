"""erd_swarm.py — Branch-swarm worker for the ERD_ALL precache.

The precache is built by workers cooperating on one branch at a time.  A
*branch* is a position to solve: the answer words left after a guess+response
(e.g. SALET ----- = 315 words).  To solve a branch we evaluate ~12,972
*candidate* guesses against it and keep the lowest-ERD one.  That candidate
list is cut into *chunks* (contiguous slices); each worker claims one chunk at
a time, so several workers pour into the same branch at once while sharing a
single running-best ERD as a branch-and-bound bound.

There is no separate algorithm for "big" vs "small" branches: an easy branch
is a one-chunk branch a single worker disposes of; a hard branch is a
many-chunk branch a crowd swarms.  Chunk count scales with branch difficulty
(n_words) — granularity only, never a different code path.

Trust model (see erd_queue.py): a claimed chunk is advisory; only a done=1
chunk is authoritative.  A branch is finalized — its ERD written to the
persistent cache — only once every chunk is done, by whichever worker observes
full coverage.  So a crashed worker's chunk is redone, never skipped.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time

from cache_sqlite import ScoreCache
from wordle_engine import (
    ERD_ALL,
    ResponseCache,
    cache_all_scores,
    evaluate_guess,
    load_word_list,
    rank_guesses_by_group_then_entropy,
    _cache_reuse,
)
from erd_queue import ErdQueue, decode_subset, encode_subset

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'

BEST_REFRESH_SECONDS = 0.25   # how often a worker re-reads the shared bound
HB_SECONDS = 2.0              # liveness heartbeat cadence during a long chunk
# A worker that hasn't heartbeat within this many seconds is presumed dead, and
# only then are its in-flight chunks reclaimed.  Must be many multiples of
# HB_SECONDS so a live-but-busy worker is never mistaken for dead (which would
# let its slice be redone and finalized before it folds in a better candidate).
HB_TIMEOUT_SECONDS = 120
CHECKPOINT_SECONDS = 300      # WAL checkpoint interval (5 min)
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
                 divisor, max_chunks, budget=ROOT_BUDGET):
        self.name = f'worker-{worker_id}'
        self.stop_event = stop_event
        self.divisor = divisor
        self.max_chunks = max_chunks
        self.budget = budget

        self.all_answers = load_word_list(ANSWER_FILE)
        self.all_words = load_word_list(WORDS_FILE)
        self.n_candidates = len(self.all_words)
        self.score_cache = ScoreCache(cache_path, self.all_answers)
        self.rcache = ResponseCache(self.all_answers, self.score_cache)
        self.queue = ErdQueue(queue_path)

        self.started = int(time.time())
        self.chunks_done = 0
        self.n_ok = 0
        self.n_pruned = 0
        self.n_useless = 0
        self._ranked_key = None      # cache last branch's ranked candidate list
        self._ranked = None
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
        # Attribution for promoted sub-branches: which top-level (opener,pattern)
        # tree the worker is currently descending.
        self._top_source_word = None
        self._top_source_pattern = None

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        self.queue.clear_heartbeat(self.name)
        self.score_cache.checkpoint()
        self.score_cache.close()
        self.queue.close()

    def cancel(self):
        return self.stop_event is not None and self.stop_event.is_set()

    # -- candidate ranking (deterministic, so every worker agrees) ----------

    def _ranked_for(self, branch_key, words):
        if self._ranked_key == branch_key:
            return self._ranked
        ranked = rank_guesses_by_group_then_entropy(
            words, self.all_words, self.rcache, self.score_cache,
            cancel_check=self.cancel)
        self._ranked_key = branch_key
        self._ranked = ranked
        return ranked

    # -- depth instrumentation ----------------------------------------------

    def _note_depth(self, depth, n):
        # Per-position observer: track max depth and the live recursion spine
        # (subset size at each depth on the active descent).  Observation only.
        if depth > self._cand_max_depth:
            self._cand_max_depth = depth
        self._cur_depth = depth
        self._spine[depth] = n
        deeper = [d for d in self._spine if d > depth]
        for d in deeper:
            del self._spine[d]

    def _spine_str(self):
        return '>'.join(str(self._spine[d]) for d in sorted(self._spine))

    # -- RAM check and WAL checkpoint ---------------------------------------

    def _free_ram_mb(self):
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

    def _check_ram(self):
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

    def _heartbeat(self, branch_key, n_words, chunk_idx, chunk_started_at,
                   cand_rate, best_guess, best_erd, force=False,
                   cur_candidate=None, cand_n_seen=None, cand_chunk_size=None):
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
            self.chunks_done, chunk_idx=chunk_idx,
            chunk_started_at=chunk_started_at, cand_rate=cand_rate,
            cache_hits=self.score_cache.read_hits,
            cache_misses=self.score_cache.read_misses,
            n_pruned=self.n_pruned, n_ok=self.n_ok,
            best_guess=best_guess, best_erd=best_erd,
            cur_candidate=cur_candidate, cand_n_seen=cand_n_seen,
            cand_chunk_size=cand_chunk_size,
            cur_max_depth=self._cand_max_depth,
            cur_nodes=self._nodes, node_rate=node_rate,
            cur_path=self._spine_str())

    # -- evaluate one chunk -------------------------------------------------

    def evaluate_chunk(self, branch_key, words, n_words, ranked, idx,
                       chunk_size, budget=None):
        """Evaluate one chunk's candidate slice, folding results into the
        branch's shared best.  Returns True if the chunk completed, False if
        cancelled mid-way (the chunk is left done=0 for reclaim/redo).

        budget is the branch's guess budget (depth-limited ERD): a candidate
        whose strategy can't win within budget is infeasible (and taints the
        branch — see ErdQueue.mark_branch_tainted).
        """
        lo, hi = ErdQueue.chunk_range(idx, chunk_size, self.n_candidates)
        chunk_total = hi - lo
        local_word, local_best = self.queue.read_branch_best(branch_key)
        local_md = None
        shared_best = local_best
        last_refresh = time.time()
        last_log = 0.0
        chunk_started = int(time.time())
        t0 = time.time()
        self._heartbeat(branch_key, n_words, idx, chunk_started, None,
                        local_word, local_best, force=True,
                        cand_chunk_size=chunk_total)

        for n_seen, ci in enumerate(range(lo, hi), start=1):
            if self.cancel():
                return False
            now = time.time()
            if now - last_refresh > BEST_REFRESH_SECONDS:
                _, shared_best = self.queue.read_branch_best(branch_key)
                last_refresh = now
            if now - last_log > 30:
                logger.info('%s chunk %d: evaluating %s (%d/%d)  best=%s  '
                            'max_depth=%d', self.name, idx, ranked[ci], n_seen,
                            chunk_total, local_word or '-', self._cand_max_depth)
                last_log = now
            bound = float('inf')
            for b in (local_best, shared_best):
                if b is not None and b < bound:
                    bound = b

            self._cand_max_depth = 0
            cand_t0 = time.time()
            status, cost, cand_md, floor_hit = evaluate_guess(
                words, ranked[ci], self.rcache, self.score_cache,
                n=n_words, best_erd=bound, guesses=self.all_words,
                policy=ERD_ALL, cancel_check=self.cancel,
                depth=0, depth_observer=self._note_depth, budget=budget,
                subbranch_solver=self._subbranch_solver,
                heartbeat=lambda: self._heartbeat(
                    branch_key, n_words, idx, chunk_started, None,
                    local_word, local_best,
                    cur_candidate=ranked[ci], cand_n_seen=n_seen,
                    cand_chunk_size=chunk_total))
            cand_elapsed = time.time() - cand_t0
            if cand_elapsed > 10:
                logger.warning('%s slow candidate %s in chunk %d: %.1fs  '
                               'status=%s  max_depth=%d', self.name, ranked[ci],
                               idx, cand_elapsed, status, self._cand_max_depth)

            if status == 'abort':
                return False
            # A candidate excluded by the depth cap (anywhere in its subtree)
            # taints the branch: its ERD is only valid at this budget.  Marked
            # for any candidate, winner or not — see the taint rule.
            if floor_hit:
                self.queue.mark_branch_tainted(branch_key)
            if status == 'ok':
                self.n_ok += 1
                if local_best is None or cost < local_best:
                    local_best, local_word, local_md = cost, ranked[ci], cand_md
                    self.queue.update_branch_best(branch_key, local_word,
                                                  local_best, local_md)
                    shared_best = local_best
            elif status in ('pruned', 'cutoff'):
                self.n_pruned += 1
            else:
                self.n_useless += 1

            rate = n_seen / max(1e-6, now - t0)
            self._heartbeat(branch_key, n_words, idx, chunk_started, rate,
                            local_word, local_best,
                            cur_candidate=ranked[ci], cand_n_seen=n_seen,
                            cand_chunk_size=chunk_total)

        self.queue.complete_chunk(branch_key, idx)
        self.chunks_done += 1
        elapsed = time.time() - t0
        rate = chunk_total / max(1e-6, elapsed)
        logger.info('%s chunk %d done: %d cands in %.1fs (%.1f/s)  '
                    'ok=%d pruned=%d useless=%d  best=%s %.4f',
                    self.name, idx, chunk_total, elapsed, rate,
                    self.n_ok, self.n_pruned, self.n_useless,
                    local_word or '-', local_best if local_best else 0)
        self._heartbeat(branch_key, n_words, idx, chunk_started, rate,
                        local_word, local_best, force=True)
        return True

    # -- finalize -----------------------------------------------------------

    def maybe_finalize(self, branch_key, words, n_chunks):
        """If every chunk is done, finalize the branch exactly once."""
        if self.queue.branch_done_chunks(branch_key) < n_chunks:
            return
        if not self.queue.try_finalize_branch(branch_key):
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
        else:
            # No feasible guess within budget: this branch is a loss.  Don't
            # write an ERD entry (there is no winning strategy to cache).
            logger.warning('%s branch (%d words) UNSOLVABLE within budget %s '
                           '(loss) src=%s', self.name, len(words), budget,
                           branch_key[:25])
        self.queue.mark_done(branch_key)        # pending_subgroups row -> done
        self.queue.delete_branch(branch_key)    # drop transient coordination

    # -- recursive cooperative solving --------------------------------------

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
        branch_key = encode_subset(words)
        # Already solved by someone? reuse without re-promoting.
        reuse = _cache_reuse(
            self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
        if reuse is not None:
            return (*reuse, False)

        n_words = len(words)
        chunk_size = ErdQueue.chunk_size_for(
            n_words, self.n_candidates, self.divisor, self.max_chunks)
        self.queue.create_branch(
            branch_key, n_words, self.n_candidates, chunk_size,
            priority=PROMOTED_PRIORITY, source_word=self._top_source_word,
            source_pattern=self._top_source_pattern, budget=budget)
        n_chunks = ErdQueue.n_chunks_for(self.n_candidates, chunk_size)
        ranked = self._ranked_for(branch_key, words)

        while not self.cancel():
            # Finished?  Check before claiming so we never touch a branch that
            # another worker just finalized and deleted.
            reuse = _cache_reuse(
                self.score_cache.read_with_depth(branch_key, ERD_ALL), budget)
            if reuse is not None:
                return (*reuse, False)
            if self.queue.get_branch(branch_key) is None:
                break                       # finalized as a loss + deleted
            idx = self.queue.claim_chunk(branch_key, self.name, n_chunks)
            if idx is not None:
                if self.evaluate_chunk(branch_key, words, n_words, ranked, idx,
                                       chunk_size, budget=budget):
                    self.maybe_finalize(branch_key, words, n_chunks)
                self._maybe_checkpoint()    # drain WAL during deep solving
            elif self.queue.branch_done_chunks(branch_key) >= n_chunks:
                self.maybe_finalize(branch_key, words, n_chunks)
            else:
                # Every chunk is claimed but coverage isn't complete: some are
                # held by other workers.  Heartbeat first (so THIS worker, which
                # still holds its own parent chunk up the stack, isn't itself
                # presumed dead while it waits), then free any sub-chunk whose
                # holder has died so we can re-claim it rather than wait forever
                # — there may be no supervisor in the standalone solve path.
                self._heartbeat(branch_key, n_words, None, None, None,
                                None, None, force=True)
                self.queue.reclaim_stale_chunks(HB_TIMEOUT_SECONDS)
                time.sleep(0.05)            # chunks in flight elsewhere; let them land

        if self.cancel():
            return None
        # Finalized as a loss: proven unsolvable (not a cutoff).
        return (float('inf'), None, True, False)

    # -- scheduling: claim one chunk of the best available branch -----------

    def claim_one(self):
        """Return (branch_row_dict, chunk_idx) for the next chunk to work, or
        None if there is nothing to do right now.

        Prefers JOINING an in-progress branch (to finish branches already
        underway, concentrating workers) over PROMOTING a new one from the
        queue.  Promotion claims a pending branch and registers it so others
        can join.
        """
        for b in self.queue.branches_in_progress():
            n_chunks = ErdQueue.n_chunks_for(b['n_candidates'], b['chunk_size'])
            idx = self.queue.claim_chunk(b['branch_key'], self.name, n_chunks)
            if idx is not None:
                return dict(b), idx

        claimed = self.queue.claim_next(self.name)
        if claimed is None:
            return None
        n_words = claimed['n_words']
        chunk_size = ErdQueue.chunk_size_for(
            n_words, self.n_candidates, self.divisor, self.max_chunks)
        self.queue.create_branch(
            claimed['branch_key'], n_words, self.n_candidates, chunk_size,
            priority=claimed['priority'], source_word=claimed['source_word'],
            source_pattern=claimed['source_pattern'], budget=self.budget)
        n_chunks = ErdQueue.n_chunks_for(self.n_candidates, chunk_size)
        idx = self.queue.claim_chunk(claimed['branch_key'], self.name, n_chunks)
        branch = {
            'branch_key': claimed['branch_key'], 'n_words': n_words,
            'n_candidates': self.n_candidates, 'chunk_size': chunk_size,
            'source_word': claimed['source_word'],
            'source_pattern': claimed['source_pattern'],
            'budget': self.budget,
        }
        # idx can be None only if another worker grabbed every chunk between
        # create and claim — rare; treat as "nothing for me right now".
        return (branch, idx) if idx is not None else None

    # -- main loop ----------------------------------------------------------

    def run(self):
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
            n_chunks = ErdQueue.n_chunks_for(branch['n_candidates'],
                                             branch['chunk_size'])
            ranked = self._ranked_for(branch_key, words)
            if self.cancel():
                break
            completed = self.evaluate_chunk(
                branch_key, words, branch['n_words'], ranked, idx,
                branch['chunk_size'], budget=branch.get('budget') or self.budget)
            if completed:
                self.maybe_finalize(branch_key, words, n_chunks)
            self._maybe_checkpoint()
            self._check_ram()

    # -- focused single-branch loop (standalone solve-branch) ---------------

    def solve_branch_focused(self, branch_key):
        """Help solve one already-registered branch to completion: claim and
        evaluate its chunks alongside any sibling workers, finalizing it once
        every chunk is done.  The body of _focused_worker, factored out so it
        can be driven directly (signal setup stays in the process wrapper)."""
        branch = self.queue.get_branch(branch_key)
        if branch is None or branch['status'] != 'open':
            return
        words = decode_subset(branch_key)
        budget = branch['budget'] or ROOT_BUDGET
        n_chunks = ErdQueue.n_chunks_for(branch['n_candidates'],
                                         branch['chunk_size'])
        while not self.cancel():
            # Stop the moment the branch is finalized (and its rows deleted) by
            # any worker: otherwise claim_chunk, seeing no chunk rows for the
            # now-deleted branch, would re-create them and redo the whole branch
            # from scratch — doubling (or worse) the work for a large branch.
            if self.queue.get_branch(branch_key) is None:
                break
            idx = self.queue.claim_chunk(branch_key, self.name, n_chunks)
            if idx is None:
                # Every chunk is claimed.  If coverage is complete, finalize and
                # stop.  Otherwise some chunks are held by siblings — there is NO
                # supervisor in this path, so free any whose holder has died and
                # retry, rather than abandoning the branch one chunk short of
                # finalizing (which would strand it forever).
                if self.queue.branch_done_chunks(branch_key) >= n_chunks:
                    self.maybe_finalize(branch_key, words, n_chunks)
                    break
                self._heartbeat(branch_key, branch['n_words'], None, None, None,
                                None, None, force=True)
                self.queue.reclaim_stale_chunks(HB_TIMEOUT_SECONDS)
                time.sleep(0.1)
                continue
            if self.evaluate_chunk(branch_key, words, branch['n_words'],
                                   self._ranked_for(branch_key, words), idx,
                                   branch['chunk_size'], budget=budget):
                if self.queue.branch_done_chunks(branch_key) >= n_chunks:
                    self.maybe_finalize(branch_key, words, n_chunks)
                    break


def swarm_worker(worker_id, cache_path, queue_path, stop_event,
                 divisor=3, max_chunks=256):
    """Process entry point for a swarm worker (target= for mp.Process)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _setup_logging(worker_id)
    logger.info('worker-%d starting (pid=%d)', worker_id, os.getpid())
    w = _BranchWorker(worker_id, cache_path, queue_path, stop_event,
                      divisor, max_chunks)
    try:
        w.run()
    finally:
        w.close()
        logger.info('worker-%d exiting (%d chunks done)',
                    worker_id, w.chunks_done)


def _setup_logging(worker_id):
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'erd_worker_{worker_id}.log')
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Standalone single-branch solve (the `solve-branch` CLI), reusing the swarm
# machinery: register one branch directly, then point N focused workers at it.
# ---------------------------------------------------------------------------

def _focused_worker(branch_key, worker_id, cache_path, queue_path,
                    divisor, max_chunks):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    w = _BranchWorker(worker_id, cache_path, queue_path, None,
                      divisor, max_chunks)
    try:
        w.solve_branch_focused(branch_key)
    finally:
        w.close()


def run_branch_solve(branch_key, words, n_workers, cache_path, queue_path,
                     divisor=3, max_chunks=256, priority=1,
                     source_word=None, source_pattern=None, budget=ROOT_BUDGET,
                     timeout=None):
    """Solve one branch by swarming N workers across its candidates.

    n_workers is capped at the number of chunks so we never spawn a process
    that will immediately find no work and exit.

    Returns (best_guess, best_erd) or None.  If timeout is given (seconds),
    any worker still running after that long is killed and the result will be
    None (branch unfinished).
    """
    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    score_cache = ScoreCache(cache_path, all_answers)
    queue = ErdQueue(queue_path)

    existing = score_cache.read(branch_key, ERD_ALL)
    if existing is not None:
        queue.close()
        score_cache.close()
        return existing

    chunk_size = ErdQueue.chunk_size_for(
        len(words), len(all_words), divisor, max_chunks)
    n_chunks = ErdQueue.n_chunks_for(len(all_words), chunk_size)
    actual_workers = min(n_workers, n_chunks)
    queue.create_branch(branch_key, len(words), len(all_words), chunk_size,
                        priority=priority, source_word=source_word,
                        source_pattern=source_pattern, budget=budget)
    queue.close()
    score_cache.close()

    procs = [mp.Process(target=_focused_worker,
                        args=(branch_key, w, cache_path, queue_path,
                              divisor, max_chunks))
             for w in range(actual_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()

    score_cache = ScoreCache(cache_path, all_answers)
    try:
        return score_cache.read(branch_key, ERD_ALL)
    finally:
        score_cache.close()
