"""erd_split.py — Cooperative candidate-level solve of a single subgroup.

The normal precache parallelizes at the granularity of whole subgroups: one
worker owns a subgroup and grinds all ~12,972 candidate guesses serially.  For
a hard subgroup (a large, low-information branch such as an opener's all-gray
response) that single-owner grind never finishes in a useful time, which is
why such branches were capped out of the main queue entirely.

This module removes that seam.  A subgroup is solved by *many* workers, each
evaluating a disjoint slice (chunk) of the candidate guesses while sharing one
running best ERD through the split tables in erd_queue.sqlite3.  The candidate
order is ranked once (best-first) so a tight bound emerges immediately and the
long tail of weak candidates is pruned hard across all workers.

It is the *same* algorithm as the single-owner path — evaluate_guess against a
shared branch-and-bound bound — only with the candidate loop hoisted out so it
can be distributed.  A subgroup small enough to fit in one chunk is simply
solved by one worker, identically to before; chunk size is a granularity knob,
not an algorithm switch.

Trust model (see the long comments in erd_queue.py):
  - A claimed chunk is ADVISORY: a hint that someone is probably on that slice.
  - A done=1 chunk is AUTHORITATIVE: that slice is fully folded into best_erd.
  - Finalization (writing the subgroup's ERD to the persistent cache) waits on
    full done=1 coverage, never on claims, so a crash causes redo, never skip.
"""

from __future__ import annotations

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


def _finalize(queue, score_cache, rcache, subset_key, words):
    """Write the split's top-level ERD result to the persistent cache.

    Called by exactly one worker (the winner of try_finalize_split) once every
    chunk is done.  The recursion inside evaluate_guess already wrote every
    sub-subgroup as a side effect; this writes the subgroup's own entry, the
    one piece no single candidate evaluation produces.
    """
    best_word, best_erd = queue.read_split_best(subset_key)
    if best_word is None:
        return
    score_cache.write(subset_key, ERD_ALL, best_word, best_erd)
    cache_all_scores(best_word, words, score_cache, subset_key, cache=rcache)
    score_cache.checkpoint()
    queue.mark_done(subset_key)   # no-op if the subgroup has no pending row


def process_split(subset_key, worker_id, cache_path, queue_path,
                  stop_event=None):
    """Worker entry point: help solve one split until its chunks run out."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    score_cache = ScoreCache(cache_path, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ErdQueue(queue_path)
    worker_name = f'split-{worker_id}'

    def cancel():
        return stop_event is not None and stop_event.is_set()

    try:
        split = queue.get_split(subset_key)
        if split is None or split['status'] != 'open':
            return
        n_candidates = split['n_candidates']
        chunk = split['chunk']
        n_words = split['n_words']
        n_chunks = ErdQueue.n_chunks_for(n_candidates, chunk)
        ranked = decode_subset(bytes(split['ranked_blob']))
        words = decode_subset(subset_key)

        started = int(time.time())
        local_best = None
        local_word = None
        shared_best = None
        last_refresh = 0.0

        while not cancel():
            idx = queue.claim_split_chunk(subset_key, worker_name, n_chunks)
            if idx is None:
                # Every chunk has a row.  If all are done we may need to
                # finalize; otherwise other workers are still grinding and we
                # simply stop helping — never block.
                if queue.split_done_chunks(subset_key) >= n_chunks:
                    if queue.try_finalize_split(subset_key):
                        _finalize(queue, score_cache, rcache, subset_key, words)
                break

            lo, hi = ErdQueue.chunk_range(idx, chunk, n_candidates)
            for ci in range(lo, hi):
                if cancel():
                    return  # leave chunk done=0 — reclaim redoes the slice
                now = time.time()
                if now - last_refresh > BEST_REFRESH_SECONDS:
                    _, shared_best = queue.read_split_best(subset_key)
                    last_refresh = now
                bound = float('inf')
                for b in (local_best, shared_best):
                    if b is not None and b < bound:
                        bound = b
                status, cost = evaluate_guess(
                    words, ranked[ci], rcache, score_cache,
                    n=n_words, best_erd=bound, guesses=all_words,
                    policy=ERD_ALL, cancel_check=cancel)
                if status == 'abort':
                    return
                if status == 'ok' and (local_best is None or cost < local_best):
                    local_best, local_word = cost, ranked[ci]
                    queue.update_split_best(subset_key, local_word, local_best)
                    shared_best = local_best

            queue.complete_split_chunk(subset_key, idx)
            queue.heartbeat(worker_name, os.getpid(), subset_key, n_words,
                            started, idx + 1,
                            candidates_done=hi, candidates_total=n_candidates,
                            best_word=local_word, best_erd=local_best)

            if queue.split_done_chunks(subset_key) >= n_chunks:
                if queue.try_finalize_split(subset_key):
                    _finalize(queue, score_cache, rcache, subset_key, words)
                break
    finally:
        queue.clear_heartbeat(worker_name)
        score_cache.checkpoint()
        score_cache.close()
        queue.close()


def run_split_solve(subset_key, words, n_workers, chunk,
                    cache_path, queue_path,
                    priority=0, source_word=None, source_pattern=None):
    """Rank candidates once, register the split, fan out N workers, finalize.

    Returns (best_word, best_erd) or None if nothing was solvable.
    """
    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    score_cache = ScoreCache(cache_path, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ErdQueue(queue_path)

    existing = score_cache.read(subset_key, ERD_ALL)
    if existing is not None:
        queue.close()
        score_cache.close()
        return existing

    ranked = rank_guesses_by_group_then_entropy(
        words, all_words, rcache, score_cache)
    # NOT encode_subset: that sorts for a canonical key, which would destroy
    # the best-first ranking.  Order must be preserved so chunk index maps to
    # the same candidates for every worker.
    ranked_blob = "".join(ranked).encode("utf-8")
    n_candidates = len(ranked)
    n_chunks = ErdQueue.n_chunks_for(n_candidates, chunk)

    queue.create_split(subset_key, len(words), n_candidates, chunk,
                       ranked_blob, priority, source_word, source_pattern)
    score_cache.checkpoint()
    score_cache.close()
    queue.close()

    procs = []
    for w in range(n_workers):
        p = mp.Process(target=process_split,
                       args=(subset_key, w, cache_path, queue_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Backstop: if workers exited (e.g. all chunks claimed by the tail worker)
    # without anyone finalizing, do it here once coverage is complete.
    queue = ErdQueue(queue_path)
    score_cache = ScoreCache(cache_path, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    try:
        result = score_cache.read(subset_key, ERD_ALL)
        if result is None and queue.split_done_chunks(subset_key) >= n_chunks:
            if queue.try_finalize_split(subset_key):
                _finalize(queue, score_cache, rcache, subset_key, words)
            result = score_cache.read(subset_key, ERD_ALL)
        return result
    finally:
        score_cache.close()
        queue.close()
