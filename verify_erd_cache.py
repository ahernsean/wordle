#!/usr/bin/env python3.13
"""verify_erd_cache.py — Re-verify all ERD_ALL cache entries against the true optimum.

The reclaim-while-alive bug (fixed 2026-06-15, commit 774ac29) could produce
cached ERD values that are >= the true optimum.  Every entry in
branch_best_by_policy for policy erd_words_unfiltered is therefore suspect.

Each entry is re-verified by checking whether any candidate achieves a strictly
lower ERD than the stored best_score, reading sub-branch costs directly from the
cache rather than recursing through the solver.  This avoids budget-mismatch
re-evaluations that would make the pass prohibitively slow.

Entries are processed in ascending branch-size order (leaves before parents)
so that corrected sub-branch values are in place before any parent is
re-evaluated.  Within each wave (fixed branch size) workers run in parallel.

IMPORTANT: stop all swarm workers before running this script.  Active workers
modifying the cache while entries are being deleted and re-written can corrupt
results.

Usage:
    python3.13 verify_erd_cache.py [--workers N] [--cache PATH] [--log PATH]
    python3.13 verify_erd_cache.py --start-size 15   # resume from wave 15
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from collections import defaultdict

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, ResponseCache, load_word_list, _ALL_GREEN_PATTERN
from erd_queue import decode_subset
from erd_swarm import ROOT_BUDGET

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'
DEFAULT_CACHE = 'wordle_cache.sqlite3'
DEFAULT_LOG = 'verify_erd_cache.log'


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def _erd_from_cache(branch_words, candidate, rcache, sc, n, best_erd):
    """Compute the ERD for candidate on branch_words, reading sub-branch costs
    directly from cache.  Returns the cost, or None if any sub-branch is
    missing from the cache or if alpha-beta pruning shows this candidate
    can't beat best_erd.

    Unlike evaluate_candidate / _solve_subset, this never recurses: it reads
    sub-branch ERD values directly via sc.read(), bypassing budget-reuse
    rules.  This is safe for verification because sub-branches were already
    verified in earlier waves; budget-reuse rules guard correctness in the
    solver but are irrelevant when we're simply reading the stored optimum.
    """
    groups = rcache.group_words(candidate, branch_words)

    # Admissible lower bound: 3 - (# groups + has_self) / n
    has_self = _ALL_GREEN_PATTERN in groups
    cost_lb = 3.0 - (len(groups) + (1 if has_self else 0)) / n
    if cost_lb >= best_erd:
        return None  # provably can't beat bound

    # Sort largest sub-branches first so cost accumulates fast (early pruning).
    ordered = sorted(groups.values(), key=len, reverse=True)

    cost = 1.0
    for sub_branch in ordered:
        k = len(sub_branch)
        if k == 0:
            continue
        if k == 1 and sub_branch[0] == candidate:
            continue  # self: 0 extra cost
        if k >= n:
            return None  # useless candidate
        if k == 1:
            cost += 1.0 / n
        else:
            branch_key = ScoreCache.encode_subset(sub_branch)
            cached = sc.read(branch_key, ERD_ALL)
            if cached is None:
                # Sub-branch not yet in cache; skip this candidate.
                return None
            cost += (k / n) * cached[1]
        if cost >= best_erd - 1e-9:
            return None  # alpha-beta: partial cost already too high

    return cost


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _verify_chunk(args):
    """Re-verify one chunk of (branch_key, old_guess, old_score, old_md, old_sb).

    For each entry: delete it, evaluate every candidate with old_score as a
    strict ceiling (only accept a strictly lower ERD), then restore or correct.

    Returns (results, elapsed_seconds) where results is a list of
    (status, n_words, old_guess, old_score, new_guess, new_score) and
    elapsed_seconds is the wall time this chunk spent in the worker thread.
    """
    chunk_t0 = time.time()
    rows, cache_path, answer_file, words_file = args

    all_answers = load_word_list(answer_file)
    all_words = load_word_list(words_file)
    sc = ScoreCache(cache_path, all_answers, checkpoint_on_close=False)
    rcache = ResponseCache(all_answers, sc)

    results = []

    for branch_key, old_guess, old_score, old_md, old_sb in rows:
        n_words = len(branch_key) // 5

        sc.delete(branch_key, ERD_ALL)
        branch_words = decode_subset(branch_key)

        # Strict ceiling: only accept a candidate strictly better than old_score.
        # A correct entry has no such candidate; a wrong entry does.
        best_erd = old_score
        best_guess_new = None

        for candidate in all_words:
            cost = _erd_from_cache(branch_words, candidate, rcache, sc,
                                   n_words, best_erd)
            if cost is not None and cost < best_erd - 1e-9:
                best_erd = cost
                best_guess_new = candidate

        if best_guess_new is None:
            # No candidate beat old_score → confirmed; restore the old entry.
            sc.write(branch_key, ERD_ALL, old_guess, old_score, old_md, old_sb)
            results.append(('CONFIRMED', n_words,
                            old_guess, old_score, old_guess, old_score))
        else:
            # Found strictly lower ERD → the old value was wrong.
            # Re-derive max_depth and solve_budget by checking what the solver
            # would compute.  For most corrected entries, the new candidate's
            # sub-branches are already in cache with known max_depth, so we can
            # reconstruct max_depth from the sub-branch reads.
            new_md = _max_depth_from_cache(branch_words, best_guess_new,
                                           rcache, sc, n_words)
            # Taint: if the new strategy relies on a tainted sub-branch, set
            # solve_budget = ROOT_BUDGET so the entry is only reused at that
            # budget.  Otherwise, untainted.
            is_tainted = _is_tainted(branch_words, best_guess_new, rcache, sc,
                                     n_words)
            solve_budget = ROOT_BUDGET if is_tainted else None
            sc.write(branch_key, ERD_ALL, best_guess_new, best_erd,
                     max_depth=new_md, solve_budget=solve_budget)
            results.append(('SCORE_CORRECTED', n_words,
                            old_guess, old_score, best_guess_new, best_erd))

    sc.checkpoint()
    sc.close()
    return results, time.time() - chunk_t0


def _max_depth_from_cache(branch_words, candidate, rcache, sc, n):
    """Return 1 + max(sub_max_depth) for candidate on branch_words, reading
    sub-branch depths from cache.  Returns None if any sub-branch depth is
    unknown (legacy NULL or missing entry)."""
    groups = rcache.group_words(candidate, branch_words)
    max_sub_depth = 0
    for sub_branch in groups.values():
        k = len(sub_branch)
        if k <= 1:
            continue
        entry = sc.read_with_depth(ScoreCache.encode_subset(sub_branch), ERD_ALL)
        if entry is None or entry[2] is None:
            return None
        if entry[2] > max_sub_depth:
            max_sub_depth = entry[2]
    return 1 + max_sub_depth


def _is_tainted(branch_words, candidate, rcache, sc, n):
    """Return True if any sub-branch of candidate is tainted (solve_budget set)."""
    groups = rcache.group_words(candidate, branch_words)
    for sub_branch in groups.values():
        k = len(sub_branch)
        if k <= 1:
            continue
        entry = sc.read_with_depth(ScoreCache.encode_subset(sub_branch), ERD_ALL)
        if entry is not None and entry[3] is not None:  # solve_budget
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Target entries per chunk; controls how often progress lines appear.
# Small chunks matter on iOS where the GIL serialises CPU-bound threads —
# a thread pool gives no parallel speedup there, so the first progress line
# appears only after the first chunk completes.  1k entries keeps that delay
# to a few seconds on any device.  A 5-second throttle on the main-thread
# printer prevents spam on Linux where chunks complete in milliseconds.
_PROGRESS_CHUNK_SIZE = 1_000
_PROGRESS_MIN_INTERVAL = 5.0  # seconds between printed progress lines


def _fmt_eta(seconds: int) -> str:
    if seconds <= 0:
        return '0s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f'{h}h{m:02d}m'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH',
                        help=f'Cache DB (default: {DEFAULT_CACHE})')
    parser.add_argument('--log', default=DEFAULT_LOG, metavar='PATH',
                        help=f'Output log file (default: {DEFAULT_LOG})')
    parser.add_argument('--workers', type=int, default=6, metavar='N',
                        help='Parallel workers per wave (default: 6)')
    parser.add_argument('--start-size', type=int, default=2, metavar='N',
                        help='Skip branches with fewer than N words — for '
                             'resuming an interrupted run.  Assumes all '
                             'smaller waves are already verified; appends to '
                             'the existing log.  (default: 2 = full run)')
    args = parser.parse_args()

    print(f'Cache    : {os.path.abspath(args.cache)}')
    print(f'Log      : {os.path.abspath(args.log)}')
    print(f'Workers  : {args.workers}')
    if args.start_size > 2:
        print(f'Resuming from wave n={args.start_size}')
    print()

    # Run schema migrations single-threaded before spawning workers.
    all_answers = load_word_list(ANSWER_FILE)
    sc = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)
    answer_list_id = sc.answer_list_id

    total_in_scope = sc._conn.execute(
        "SELECT COUNT(*) FROM branch_best_by_policy "
        "WHERE policy=? AND answer_list_id=? AND length(branch_key)/5 >= ?",
        (ERD_ALL, answer_list_id, args.start_size)
    ).fetchone()[0]
    total_all = sc._conn.execute(
        "SELECT COUNT(*) FROM branch_best_by_policy "
        "WHERE policy=? AND answer_list_id=?",
        (ERD_ALL, answer_list_id)
    ).fetchone()[0]

    wave_sizes = [r[0] for r in sc._conn.execute(
        "SELECT DISTINCT length(branch_key)/5 FROM branch_best_by_policy "
        "WHERE policy=? AND answer_list_id=? AND length(branch_key)/5 >= ? "
        "ORDER BY length(branch_key)",
        (ERD_ALL, answer_list_id, args.start_size)
    ).fetchall()]
    sc.close()

    if not wave_sizes:
        print('No entries to verify.')
        return

    size_range = f'{min(wave_sizes)}..{max(wave_sizes)}'
    print(f'Entries in scope : {total_in_scope:,}  (of {total_all:,} total)')
    print(f'Branch sizes     : {size_range}  ({len(wave_sizes)} distinct sizes)')
    print()

    t0 = time.time()
    n_checked = 0
    n_score_corrected = 0

    log_mode = 'a' if args.start_size > 2 else 'w'
    with open(args.log, log_mode) as logf:
        if log_mode == 'w':
            logf.write('status\tn\told_guess\told_score\tnew_guess\tnew_score\n')

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for wave_size in wave_sizes:
                wave_t0 = time.time()

                sc2 = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)
                wave_rows_raw = sc2._conn.execute(
                    "SELECT branch_key, best_guess, best_score, "
                    "       max_depth, solve_budget "
                    "FROM branch_best_by_policy "
                    "WHERE policy=? AND answer_list_id=? "
                    "  AND length(branch_key)/5=?",
                    (ERD_ALL, sc2.answer_list_id, wave_size)
                ).fetchall()
                sc2.close()

                wave_data = [
                    (bytes(r['branch_key']), r['best_guess'], r['best_score'],
                     r['max_depth'], r['solve_budget'])
                    for r in wave_rows_raw
                ]
                n_wave = len(wave_data)

                # At least args.workers chunks (keep all threads busy); at most
                # one chunk per _PROGRESS_CHUNK_SIZE entries (progress granularity).
                n_chunks = max(args.workers,
                               (n_wave + _PROGRESS_CHUNK_SIZE - 1) // _PROGRESS_CHUNK_SIZE)
                chunk_size = max(1, (n_wave + n_chunks - 1) // n_chunks)
                chunks = [wave_data[i:i + chunk_size]
                          for i in range(0, n_wave, chunk_size)]
                n_chunks = len(chunks)  # actual after ceiling division

                overall_pct = 100.0 * n_checked / total_in_scope if total_in_scope else 0
                print(f'{time.strftime("%H:%M:%S")}  '
                      f'n={wave_size}: {n_wave:,} entries  '
                      f'({n_chunks} chunks, {chunk_size:,}/chunk)  '
                      f'overall {overall_pct:.1f}% done so far',
                      flush=True)

                futures = {
                    pool.submit(_verify_chunk,
                                (chunk, args.cache, ANSWER_FILE, WORDS_FILE)): idx
                    for idx, chunk in enumerate(chunks)
                }

                wave_done = 0
                wave_corrected = 0
                chunks_done = 0
                last_progress = t0
                total_chunk_cpu = 0.0  # sum of worker wall times across all chunks

                for future in as_completed(futures):
                    chunk_results, chunk_elapsed = future.result()
                    chunks_done += 1
                    n_chunk = len(chunk_results)
                    total_chunk_cpu += chunk_elapsed
                    chunk_rate = n_chunk / chunk_elapsed if chunk_elapsed > 0 else 0
                    for status, n, og, os_, ng, ns in chunk_results:
                        n_checked += 1
                        wave_done += 1
                        logf.write(
                            f'{status}\t{n}\t{og}\t{os_:.6f}\t{ng}\t{ns:.6f}\n')
                        if status == 'SCORE_CORRECTED':
                            n_score_corrected += 1
                            wave_corrected += 1

                    now = time.time()
                    is_last = chunks_done == n_chunks
                    if is_last or now - last_progress >= _PROGRESS_MIN_INTERVAL:
                        last_progress = now
                        elapsed_total = now - t0
                        rate = n_checked / elapsed_total if elapsed_total > 0 else 0
                        remaining = total_in_scope - n_checked
                        eta_s = int(remaining / rate) if rate > 0 else 0
                        wave_pct = 100.0 * wave_done / n_wave if n_wave else 100.0
                        overall_pct = 100.0 * n_checked / total_in_scope if total_in_scope else 100.0
                        corr_str = f'  {wave_corrected} corrected' if wave_corrected else ''
                        print(f'{time.strftime("%H:%M:%S")}  '
                              f'[{chunks_done:3d}/{n_chunks}] '
                              f'wave {wave_pct:3.0f}%  overall {overall_pct:.1f}%  '
                              f'chunk: {chunk_elapsed:.1f}s/{n_chunk}ent/{chunk_rate:,.0f}s⁻¹  '
                              f'overall: {rate:,.0f}/s  ETA {_fmt_eta(eta_s)}{corr_str}',
                              flush=True)

                logf.flush()
                wave_wall = time.time() - wave_t0
                parallelism = total_chunk_cpu / wave_wall if wave_wall > 0 else 1.0
                print(f'{time.strftime("%H:%M:%S")}  '
                      f'wave done: {n_wave:,} checked  {wave_corrected} corrected  '
                      f'wall {_fmt_eta(int(wave_wall))}  '
                      f'CPU {total_chunk_cpu:.1f}s  '
                      f'thread efficiency {parallelism:.2f}x '
                      f'(1.00=serial  {args.workers}.00=fully parallel)',
                      flush=True)
                print(flush=True)

    elapsed = int(time.time() - t0)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    n_confirmed = n_checked - n_score_corrected
    print(f'\nDone in {h}h{m:02d}m{s:02d}s')
    print(f'  Checked         : {n_checked:,}')
    print(f'  Confirmed       : {n_confirmed:,}')
    print(f'  Score corrected : {n_score_corrected:,}  (old ERD was too high)')
    print(f'  Log             : {os.path.abspath(args.log)}')


if __name__ == '__main__':
    main()
