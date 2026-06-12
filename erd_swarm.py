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
)
from erd_queue import ErdQueue, decode_subset, encode_subset

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'

BEST_REFRESH_SECONDS = 0.25   # how often a worker re-reads the shared bound
HB_SECONDS = 2.0              # liveness heartbeat cadence during a long chunk

logger = logging.getLogger('wordle')


class _BranchWorker:
    """One worker process's state and operations on branches/chunks."""

    def __init__(self, worker_id, cache_path, queue_path, stop_event,
                 divisor, max_chunks):
        self.name = f'worker-{worker_id}'
        self.stop_event = stop_event
        self.divisor = divisor
        self.max_chunks = max_chunks

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

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        self.queue.clear_heartbeat(self.name)
        self.score_cache.checkpoint()
        self.score_cache.close()
        self.queue.close()

    def cancel(self):
        return self.stop_event is not None and self.stop_event.is_set()

    # -- candidate ranking (deterministic, so every worker agrees) ----------

    def _ranked_for(self, subset_key, words):
        if self._ranked_key == subset_key:
            return self._ranked
        ranked = rank_guesses_by_group_then_entropy(
            words, self.all_words, self.rcache, self.score_cache,
            cancel_check=self.cancel)
        self._ranked_key = subset_key
        self._ranked = ranked
        return ranked

    # -- heartbeat ----------------------------------------------------------

    def _heartbeat(self, subset_key, n_words, chunk_idx, chunk_started_at,
                   cand_rate, best_word, best_erd, force=False):
        now = time.time()
        if not force and now - self._last_hb < HB_SECONDS:
            return
        self._last_hb = now
        self.queue.heartbeat(
            self.name, os.getpid(), subset_key, n_words, self.started,
            self.chunks_done, chunk_idx=chunk_idx,
            chunk_started_at=chunk_started_at, cand_rate=cand_rate,
            cache_hits=self.score_cache.read_hits,
            cache_misses=self.score_cache.read_misses,
            n_pruned=self.n_pruned, n_ok=self.n_ok,
            best_word=best_word, best_erd=best_erd)

    # -- evaluate one chunk -------------------------------------------------

    def evaluate_chunk(self, subset_key, words, n_words, ranked, idx,
                       chunk_size):
        """Evaluate one chunk's candidate slice, folding results into the
        branch's shared best.  Returns True if the chunk completed, False if
        cancelled mid-way (the chunk is left done=0 for reclaim/redo).
        """
        lo, hi = ErdQueue.chunk_range(idx, chunk_size, self.n_candidates)
        local_word, local_best = self.queue.read_branch_best(subset_key)
        shared_best = local_best
        last_refresh = time.time()
        chunk_started = int(time.time())
        t0 = time.time()
        self._heartbeat(subset_key, n_words, idx, chunk_started, None,
                        local_word, local_best, force=True)

        for n_seen, ci in enumerate(range(lo, hi), start=1):
            if self.cancel():
                return False
            now = time.time()
            if now - last_refresh > BEST_REFRESH_SECONDS:
                _, shared_best = self.queue.read_branch_best(subset_key)
                last_refresh = now
            bound = float('inf')
            for b in (local_best, shared_best):
                if b is not None and b < bound:
                    bound = b

            status, cost = evaluate_guess(
                words, ranked[ci], self.rcache, self.score_cache,
                n=n_words, best_erd=bound, guesses=self.all_words,
                policy=ERD_ALL, cancel_check=self.cancel)

            if status == 'abort':
                return False
            if status == 'ok':
                self.n_ok += 1
                if local_best is None or cost < local_best:
                    local_best, local_word = cost, ranked[ci]
                    self.queue.update_branch_best(subset_key, local_word,
                                                  local_best)
                    shared_best = local_best
            elif status == 'pruned':
                self.n_pruned += 1
            else:
                self.n_useless += 1

            rate = n_seen / max(1e-6, now - t0)
            self._heartbeat(subset_key, n_words, idx, chunk_started, rate,
                            local_word, local_best)

        self.queue.complete_chunk(subset_key, idx)
        self.chunks_done += 1
        rate = (hi - lo) / max(1e-6, time.time() - t0)
        self._heartbeat(subset_key, n_words, idx, chunk_started, rate,
                        local_word, local_best, force=True)
        return True

    # -- finalize -----------------------------------------------------------

    def maybe_finalize(self, subset_key, words, n_chunks):
        """If every chunk is done, finalize the branch exactly once."""
        if self.queue.branch_done_chunks(subset_key) < n_chunks:
            return
        if not self.queue.try_finalize_branch(subset_key):
            return  # another worker won the finalize
        best_word, best_erd = self.queue.read_branch_best(subset_key)
        if best_word is not None:
            self.score_cache.write(subset_key, ERD_ALL, best_word, best_erd)
            cache_all_scores(best_word, words, self.score_cache, subset_key,
                             cache=self.rcache)
            self.score_cache.checkpoint()
        self.queue.mark_done(subset_key)        # pending_subgroups row -> done
        self.queue.delete_branch(subset_key)    # drop transient coordination
        logger.info('%s finalized branch (%d words) -> %s erd=%.4f',
                    self.name, len(words), best_word,
                    best_erd if best_erd is not None else -1)

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
            idx = self.queue.claim_chunk(b['subset_key'], self.name, n_chunks)
            if idx is not None:
                return dict(b), idx

        claimed = self.queue.claim_next(self.name)
        if claimed is None:
            return None
        n_words = claimed['n_words']
        chunk_size = ErdQueue.chunk_size_for(
            n_words, self.n_candidates, self.divisor, self.max_chunks)
        self.queue.create_branch(
            claimed['subset_key'], n_words, self.n_candidates, chunk_size,
            priority=claimed['priority'], source_word=claimed['source_word'],
            source_pattern=claimed['source_pattern'])
        n_chunks = ErdQueue.n_chunks_for(self.n_candidates, chunk_size)
        idx = self.queue.claim_chunk(claimed['subset_key'], self.name, n_chunks)
        branch = {
            'subset_key': claimed['subset_key'], 'n_words': n_words,
            'n_candidates': self.n_candidates, 'chunk_size': chunk_size,
            'source_word': claimed['source_word'],
            'source_pattern': claimed['source_pattern'],
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
            subset_key = branch['subset_key']
            words = decode_subset(subset_key)
            n_chunks = ErdQueue.n_chunks_for(branch['n_candidates'],
                                             branch['chunk_size'])
            ranked = self._ranked_for(subset_key, words)
            if self.cancel():
                break
            completed = self.evaluate_chunk(
                subset_key, words, branch['n_words'], ranked, idx,
                branch['chunk_size'])
            if completed:
                self.maybe_finalize(subset_key, words, n_chunks)


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

def _focused_worker(subset_key, worker_id, cache_path, queue_path,
                    divisor, max_chunks):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    w = _BranchWorker(worker_id, cache_path, queue_path, None,
                      divisor, max_chunks)
    try:
        branch = w.queue.get_branch(subset_key)
        if branch is None or branch['status'] != 'open':
            return
        words = decode_subset(subset_key)
        n_chunks = ErdQueue.n_chunks_for(branch['n_candidates'],
                                         branch['chunk_size'])
        while True:
            idx = w.queue.claim_chunk(subset_key, w.name, n_chunks)
            if idx is None:
                if w.queue.branch_done_chunks(subset_key) >= n_chunks:
                    w.maybe_finalize(subset_key, words, n_chunks)
                break
            if w.evaluate_chunk(subset_key, words, branch['n_words'],
                                w._ranked_for(subset_key, words), idx,
                                branch['chunk_size']):
                if w.queue.branch_done_chunks(subset_key) >= n_chunks:
                    w.maybe_finalize(subset_key, words, n_chunks)
                    break
    finally:
        w.close()


def run_branch_solve(subset_key, words, n_workers, cache_path, queue_path,
                     divisor=3, max_chunks=256, priority=1,
                     source_word=None, source_pattern=None):
    """Solve one branch by swarming N workers across its candidates.

    Returns (best_word, best_erd) or None.
    """
    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    score_cache = ScoreCache(cache_path, all_answers)
    queue = ErdQueue(queue_path)

    existing = score_cache.read(subset_key, ERD_ALL)
    if existing is not None:
        queue.close()
        score_cache.close()
        return existing

    chunk_size = ErdQueue.chunk_size_for(
        len(words), len(all_words), divisor, max_chunks)
    queue.create_branch(subset_key, len(words), len(all_words), chunk_size,
                        priority=priority, source_word=source_word,
                        source_pattern=source_pattern)
    queue.close()
    score_cache.close()

    procs = [mp.Process(target=_focused_worker,
                        args=(subset_key, w, cache_path, queue_path,
                              divisor, max_chunks))
             for w in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    score_cache = ScoreCache(cache_path, all_answers)
    try:
        return score_cache.read(subset_key, ERD_ALL)
    finally:
        score_cache.close()
