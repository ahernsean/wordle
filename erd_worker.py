"""erd_worker.py — One ERD_ALL precache worker process.

Entry point for multiprocessing.Process(target=erd_worker.main, ...).
Each worker opens its own sqlite3 connections (not fork/thread-shareable),
claims the highest-priority / largest pending subgroup from the queue,
skips it if already cached as a recursion side-effect of another worker's
larger subgroup, and calls min_expected_guesses on the rest.

The worker exits cleanly after --recycle-after completed subgroups so the
supervisor can respawn it with a fresh address space, controlling the
unbounded growth of ScoreCache._mem_cache over a long run.
"""

from __future__ import annotations

import logging
import os
import time

from cache_sqlite import ScoreCache
from wordle_engine import (
    ERD_ALL,
    ResponseCache,
    load_word_list,
    min_expected_guesses,
    rank_guesses_by_group_then_entropy,
)
from erd_queue import ErdQueue, decode_subset

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'

logger = logging.getLogger('wordle')


def main(worker_id: int, cache_path: str, queue_path: str,
         recycle_after: int, checkpoint_every: int, stop_event):
    """Worker process entry point.

    Runs until stop_event is set or recycle_after subgroups complete,
    then exits so the supervisor respawns it fresh.
    """
    _setup_logging(worker_id)
    logger.info('worker-%d starting (pid=%d)', worker_id, os.getpid())

    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    score_cache = ScoreCache(cache_path, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ErdQueue(queue_path)

    worker_name = f'worker-{worker_id}'
    started_at = int(time.time())
    done_count = 0
    last_hb_time = 0.0

    def cancel_check() -> bool:
        return stop_event.is_set()

    def _write_heartbeat(current_key=None, n=None):
        nonlocal last_hb_time
        now = time.time()
        if now - last_hb_time < 2.0:
            return
        last_hb_time = now
        queue.heartbeat(worker_name, os.getpid(), current_key, n,
                        started_at, done_count)

    try:
        while not stop_event.is_set() and done_count < recycle_after:
            claimed = queue.claim_next(worker_name)
            if claimed is None:
                # Queue momentarily empty — bootstrap may still be running,
                # or all remaining work is in_progress with other workers.
                time.sleep(2)
                continue

            subset_key, n_words = claimed
            _write_heartbeat(subset_key, n_words)

            # Fast path: already filled in as a recursion side-effect of
            # another worker completing a larger subgroup that contained this
            # one as a branch.
            if score_cache.read(subset_key, ERD_ALL) is not None:
                queue.mark_done(subset_key)
                done_count += 1
                continue

            words = decode_subset(subset_key)

            # Pre-rank candidates so the best guesses come first — identical
            # to BranchPrecacheSolver's approach.  Passing the ranked list to
            # min_expected_guesses lets cost_lb pruning eliminate bad candidates
            # much earlier, since best_erd is established quickly.
            ranked = rank_guesses_by_group_then_entropy(
                words, all_words, rcache, score_cache,
                cancel_check=cancel_check)
            if stop_event.is_set():
                # Leave row as in_progress; reset_stale_in_progress() reclaims
                # it on next run.
                break

            def heartbeat():
                _write_heartbeat(subset_key, n_words)

            result = min_expected_guesses(
                words, rcache, score_cache,
                guesses=ranked,
                policy=ERD_ALL,
                cancel_check=cancel_check,
                heartbeat=heartbeat,
            )

            if result is None:
                # Cancelled mid-computation — partial work is discarded
                # (min_expected_guesses writes nothing on abort), leave
                # in_progress for reset_stale_in_progress() on next run.
                break

            queue.mark_done(subset_key)
            done_count += 1
            logger.info('worker-%d subgroup done (%d words, erd=%.4f) '
                        '[%d/%d]',
                        worker_id, n_words, result, done_count, recycle_after)

            if done_count % checkpoint_every == 0:
                score_cache.checkpoint()

    finally:
        queue.clear_heartbeat(worker_name)
        score_cache.checkpoint()
        score_cache.close()
        queue.close()
        logger.info('worker-%d exiting, completed %d subgroups',
                    worker_id, done_count)


def _setup_logging(worker_id: int):
    """Configure the wordle logger for this worker process.

    Removes any handlers inherited from the parent process (fork artifact)
    and installs a worker-specific file handler so workers don't write to
    the supervisor's erd_search.log.
    """
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'erd_worker_{worker_id}.log')
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
