#!/usr/bin/env python3.13
"""verify_erd_losses.py — Re-verify suspect proven-loss cache entries.

The mid-loop overrun publisher could hand off a frame still riding its entry
alpha-beta ceiling, marking ceiling-priced prefix candidates done in an
*exact* cooperative branch (fixed in PR #115).  A branch the swarm finalized
while the publisher was live may therefore have cached a **false loss**: the
remainder was all infeasible, but a discarded prefix candidate (priced out
>= ceiling, never proven infeasible) may have been feasible.  A false loss
poisons every budget below the stored one and can propagate upward — a
parent that trusted it may have discarded a feasible candidate.

Suspect set: loss entries whose branch key appears in the swarm's
branch_finalize_log (every swarm-finalized branch has a row there; losses
proven inline by the engine never went through finalize and are sound).

Each suspect is re-verified by deleting the loss row and re-running the
budget-capped solve with the fixed engine, to the exact optimum:

  REFUTED    a winning strategy exists within the stored budget.  The solve
             itself writes the exact best through the normal cache path; a
             mere feasibility witness is never what gets cached, because the
             solve runs to completion before any durable write.
  CONFIRMED  the exhaustive disproof succeeds again.  The engine re-writes
             the loss row; an unconditional write_loss here backstops the
             certain-loss short-circuits that return before touching the
             cache (write_loss keeps the widest budget, so it is idempotent).

Entries are processed leaves-first (ascending branch size) so a refuted
child loss is corrected before any parent that may have inherited its poison
is examined.  Within each wave (fixed branch size) workers run in parallel;
same-size branches never nest, so wave-internal order is free.

A killed run loses deleted-but-unverified rows in the wave that was running:
the poison is gone (nothing wrong remains cached) but the disproof work is
lost — the swarm re-proves such losses on demand.  Resume with --start-size
to skip completed waves.

IMPORTANT: stop all swarm workers before running this script.  Active
workers reading losses while rows are being deleted and re-proven can build
results on a verdict mid-verification.

Usage:
    python3.13 verify_erd_losses.py [--workers N] [--cache PATH]
                                    [--telemetry PATH] [--log PATH]
    python3.13 verify_erd_losses.py --start-size 15   # resume from wave 15
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import sqlite3
import time
from collections import defaultdict

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, ResponseCache, load_word_list, min_expected_guesses
from erd_queue import decode_subset
from pattern_matrix import PatternMatrix

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'
DEFAULT_CACHE = 'wordle_cache.sqlite3'
DEFAULT_TELEMETRY = 'erd_queue_telemetry.sqlite3'
DEFAULT_LOG = 'verify_erd_losses.log'


# ---------------------------------------------------------------------------
# Suspect-set derivation
# ---------------------------------------------------------------------------

def _suspect_losses(cache_path, telemetry_path, answer_list_id):
    """All (branch_key, loss_budget) loss rows whose key the swarm finalized.

    The finalize-log keys are loaded into a set and intersected in Python:
    the loss table is ~560k rows and the log ~33k, so a SQL anti-join without
    a covering index on the attached side is far slower than two scans.
    """
    tel = sqlite3.connect(telemetry_path)
    finalized_keys = {bytes(row[0]) for row in tel.execute(
        "SELECT DISTINCT branch_key FROM branch_finalize_log "
        "WHERE branch_key IS NOT NULL")}
    tel.close()

    conn = sqlite3.connect(cache_path)
    suspects = [
        (bytes(key), loss_budget)
        for key, loss_budget in conn.execute(
            "SELECT branch_key, loss_budget FROM branch_loss_by_policy "
            "WHERE policy = ? AND answer_list_id = ?",
            (ERD_ALL, answer_list_id))
        if bytes(key) in finalized_keys
    ]
    conn.close()
    return suspects


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _verify_chunk(args):
    """Re-verify one chunk of (branch_key, loss_budget) suspect loss rows.

    Returns (results, elapsed_seconds) where results is a list of
    (status, n_words, loss_budget, new_guess, new_score) and elapsed_seconds
    is the wall time this chunk spent in the worker.
    """
    chunk_t0 = time.time()
    rows, cache_path, answer_file, words_file = args

    all_answers = load_word_list(answer_file)
    all_words = load_word_list(words_file)
    sc = ScoreCache(cache_path, all_answers, checkpoint_on_close=False)
    rcache = ResponseCache(all_answers, sc)
    pattern_matrix = PatternMatrix.load_or_build(
        cache_path, all_words, all_answers, sc)

    results = []

    for branch_key, loss_budget in rows:
        branch_words = decode_subset(branch_key)
        n_words = len(branch_words)

        # Delete first: _solve_subset reuses cached losses, so the suspect
        # row would otherwise confirm itself without any re-proof.
        sc.delete_loss(branch_key, ERD_ALL)

        cost = min_expected_guesses(
            branch_words, rcache, sc,
            guesses=all_words, policy=ERD_ALL, budget=loss_budget,
            pattern_matrix=pattern_matrix)

        if cost is None:
            sc.write_loss(branch_key, ERD_ALL, loss_budget)
            results.append(
                ('CONFIRMED', n_words, loss_budget, '-', float('inf')))
        else:
            entry = sc.read(branch_key, ERD_ALL)
            new_guess = entry[0] if entry is not None else '?'
            results.append(
                ('REFUTED', n_words, loss_budget, new_guess, cost))

    sc.checkpoint()
    sc.close()
    return results, time.time() - chunk_t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Target entries per chunk.  Loss re-verification cost varies wildly per
# entry (a budget-2 disproof is milliseconds; a budget-3 exact solve on a
# large branch can take seconds), so chunks are small to keep the pool
# load-balanced and progress lines frequent.
_PROGRESS_CHUNK_SIZE = 50
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
    parser.add_argument('--telemetry', default=DEFAULT_TELEMETRY, metavar='PATH',
                        help='Telemetry DB holding branch_finalize_log '
                             f'(default: {DEFAULT_TELEMETRY})')
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

    print(f'Cache     : {os.path.abspath(args.cache)}')
    print(f'Telemetry : {os.path.abspath(args.telemetry)}')
    print(f'Log       : {os.path.abspath(args.log)}')
    print(f'Workers   : {args.workers}')
    if args.start_size > 2:
        print(f'Resuming from wave n={args.start_size}')
    print()

    # Run schema migrations single-threaded before spawning workers.
    all_answers = load_word_list(ANSWER_FILE)
    sc = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)
    answer_list_id = sc.answer_list_id
    sc.close()

    suspects = _suspect_losses(args.cache, args.telemetry, answer_list_id)
    by_size = defaultdict(list)
    for branch_key, loss_budget in suspects:
        by_size[len(branch_key) // 5].append((branch_key, loss_budget))

    wave_sizes = sorted(n for n in by_size if n >= args.start_size)
    total_in_scope = sum(len(by_size[n]) for n in wave_sizes)

    if not wave_sizes:
        print('No suspect losses to verify.')
        return

    size_range = f'{min(wave_sizes)}..{max(wave_sizes)}'
    print(f'Suspect losses   : {len(suspects):,}  ({total_in_scope:,} in scope)')
    print(f'Branch sizes     : {size_range}  ({len(wave_sizes)} distinct sizes)')
    print()

    t0 = time.time()
    n_checked = 0
    n_refuted = 0

    log_mode = 'a' if args.start_size > 2 else 'w'
    with open(args.log, log_mode) as logf:
        if log_mode == 'w':
            logf.write('status\tn\tloss_budget\tnew_guess\tnew_score\n')

        # Processes where fork exists (Linux) for true parallelism; threads on
        # iOS where fork is unavailable.
        Executor = ProcessPoolExecutor if hasattr(os, 'fork') else ThreadPoolExecutor
        print(f'Executor: {"processes" if Executor is ProcessPoolExecutor else "threads"}',
              flush=True)
        with Executor(max_workers=args.workers) as pool:
            for wave_size in wave_sizes:
                wave_t0 = time.time()
                wave_data = by_size[wave_size]
                n_wave = len(wave_data)

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
                wave_refuted = 0
                chunks_done = 0
                last_progress = t0
                total_chunk_cpu = 0.0  # sum of worker wall times across all chunks

                for future in as_completed(futures):
                    chunk_results, chunk_elapsed = future.result()
                    chunks_done += 1
                    n_chunk = len(chunk_results)
                    total_chunk_cpu += chunk_elapsed
                    chunk_rate = n_chunk / chunk_elapsed if chunk_elapsed > 0 else 0
                    for status, n, loss_budget, new_guess, new_score in chunk_results:
                        n_checked += 1
                        wave_done += 1
                        logf.write(f'{status}\t{n}\t{loss_budget}\t'
                                   f'{new_guess}\t{new_score:.6f}\n')
                        if status == 'REFUTED':
                            n_refuted += 1
                            wave_refuted += 1

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
                        ref_str = f'  {wave_refuted} refuted' if wave_refuted else ''
                        print(f'{time.strftime("%H:%M:%S")}  '
                              f'[{chunks_done:3d}/{n_chunks}] '
                              f'wave {wave_pct:3.0f}%  overall {overall_pct:.1f}%  '
                              f'chunk: {chunk_elapsed:.1f}s/{n_chunk}ent/{chunk_rate:,.0f}s⁻¹  '
                              f'overall: {rate:,.0f}/s  ETA {_fmt_eta(eta_s)}{ref_str}',
                              flush=True)

                logf.flush()
                wave_wall = time.time() - wave_t0
                parallelism = total_chunk_cpu / wave_wall if wave_wall > 0 else 1.0
                print(f'{time.strftime("%H:%M:%S")}  '
                      f'wave done: {n_wave:,} checked  {wave_refuted} refuted  '
                      f'wall {_fmt_eta(int(wave_wall))}  '
                      f'CPU {total_chunk_cpu:.1f}s  '
                      f'thread efficiency {parallelism:.2f}x '
                      f'(1.00=serial  {args.workers}.00=fully parallel)',
                      flush=True)
                print(flush=True)

    elapsed = int(time.time() - t0)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    n_confirmed = n_checked - n_refuted
    print(f'\nDone in {h}h{m:02d}m{s:02d}s')
    print(f'  Checked   : {n_checked:,}')
    print(f'  Confirmed : {n_confirmed:,}')
    print(f'  Refuted   : {n_refuted:,}  (false loss corrected to exact best)')
    print(f'  Log       : {os.path.abspath(args.log)}')


if __name__ == '__main__':
    main()
