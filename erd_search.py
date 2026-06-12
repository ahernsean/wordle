#!/usr/bin/env python3
"""erd_search.py — Parallel ERD_ALL precache CLI.

Subcommands
-----------
bootstrap   Walk root words, build response decompositions, and populate
            erd_queue.sqlite3 with every branch subgroup (size >= 2).
            Idempotent / resumable.

run         Start the supervisor: spawn N worker processes, monitor and
            respawn on recycle/crash, until the queue is empty or the
            process receives SIGTERM / Ctrl-C (graceful drain).

status      Read-only progress snapshot: queue counts, cache throughput,
            per-worker heartbeats.  --watch loops the display.

reset-stale Reset any 'in_progress' queue rows to 'pending'.  Done
            automatically by 'run' on startup; exposed here for manual
            recovery without restarting workers.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from datetime import datetime

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, ResponseCache, load_word_list
from erd_queue import ErdQueue, encode_subset
import erd_worker

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'
DEFAULT_CACHE = 'wordle_cache.sqlite3'
DEFAULT_QUEUE = 'erd_queue.sqlite3'

logger = logging.getLogger('wordle')


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def cmd_bootstrap(args):
    all_answers = load_word_list(ANSWER_FILE)
    root_words = load_word_list(args.root_words or WORDS_FILE)
    priority_words = {w.strip().lower() for w in (args.priority_words or [])}
    unknown = priority_words - set(root_words)
    if unknown:
        print(f'Warning: priority words not in root-word list: '
              f'{", ".join(sorted(unknown))}')

    score_cache = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ErdQueue(args.queue)

    # Reset any stale in_progress rows from a previous interrupted run.
    stale = queue.reset_stale_in_progress()
    if stale:
        print(f'Reset {stale} stale in_progress rows to pending.')

    # Print universe_id so the user can verify it matches their iPhone.
    print(f'Universe ID : {score_cache.universe_id[:16]}...'
          f'  ({len(all_answers)} answers)')
    print(f'Root words  : {len(root_words):,}  '
          f'({len(priority_words)} priority)')
    print(f'Cache       : {os.path.abspath(args.cache)}')
    print(f'Queue       : {os.path.abspath(args.queue)}')
    print()

    # Resumability: track which root words we've already enqueued.
    raw = queue.get_meta('bootstrap_done_roots') or ''
    done_roots: set[str] = set(raw.split(',')) if raw else set()
    remaining = [w for w in root_words if w not in done_roots]
    print(f'{len(done_roots):,} already done, '
          f'{len(remaining):,} to process.')

    try:
        for i, word in enumerate(remaining):
            groups = rcache.group_words(word, all_answers)
            priority = 1 if word in priority_words else 0
            rows = [
                (encode_subset(branch), len(branch), priority)
                for branch in groups.values() if len(branch) >= 2
            ]
            if rows:
                queue.add_pending_many(rows)
            done_roots.add(word)

            if (i + 1) % 100 == 0 or i == len(remaining) - 1:
                queue.set_meta('bootstrap_done_roots',
                               ','.join(sorted(done_roots)))
                score_cache.checkpoint()
                total = queue.total_subgroups()
                pct = (len(done_roots) / len(root_words)) * 100
                print(f'\r  [{len(done_roots):5d}/{len(root_words)}]'
                      f' {word.upper():<10s}'
                      f'  {total:,} subgroups queued'
                      f'  ({pct:.1f}%)',
                      end='', flush=True)

        print()
        queue.set_meta('bootstrap_status', 'done')
        queue.set_meta('bootstrap_completed_at', str(int(time.time())))
        total = queue.total_subgroups()
        counts = queue.counts_by_status()
        print(f'\nBootstrap complete.')
        print(f'  {total:,} distinct subgroups queued')
        print(f'  {counts.get("done", 0):,} already done '
              f'(cached from iPhone)')

    except KeyboardInterrupt:
        print('\n\nBootstrap interrupted — progress saved, resumable.')
    finally:
        score_cache.checkpoint()
        score_cache.close()
        queue.close()


# ---------------------------------------------------------------------------
# run (supervisor)
# ---------------------------------------------------------------------------

def cmd_run(args):
    queue = ErdQueue(args.queue)
    stale = queue.reset_stale_in_progress()
    if stale:
        print(f'Reset {stale} stale in_progress rows to pending.')

    bootstrap_status = queue.get_meta('bootstrap_status')
    if bootstrap_status != 'done' and not args.allow_partial_queue:
        print('Error: bootstrap not marked complete.  '
              'Run bootstrap first, or pass --allow-partial-queue to proceed anyway.',
              file=sys.stderr)
        queue.close()
        sys.exit(1)
    queue.close()

    _setup_supervisor_logging()
    logger.info('Supervisor starting: %d workers, recycle_after=%d, '
                'recycle_hours=%.1f', args.workers, args.recycle_after,
                args.recycle_hours)

    stop_event = multiprocessing.Event()

    def _sighandler(signum, frame):
        logger.info('Supervisor received signal %d — stopping...', signum)
        print(f'\nReceived signal {signum}, draining workers...')
        stop_event.set()

    signal.signal(signal.SIGTERM, _sighandler)
    signal.signal(signal.SIGINT, _sighandler)

    procs: dict[int, tuple] = {}
    for i in range(args.workers):
        procs[i] = _spawn_worker(i, args, stop_event)
    print(f'Started {args.workers} workers  (pid={os.getpid()}).')
    print(f'Monitor: python3.13 erd_search.py status --watch')
    print(f'Stop:    kill {os.getpid()}  or  Ctrl-C')

    while not stop_event.is_set():
        time.sleep(5)

        for wid, (p, started_at) in list(procs.items()):
            age = time.time() - started_at
            if not p.is_alive():
                logger.info('Worker %d exited (age=%.0fs), respawning', wid, age)
                procs[wid] = _spawn_worker(wid, args, stop_event)
            elif age > args.recycle_hours * 3600:
                logger.info('Worker %d recycle-hours hit (age=%.0fs), '
                            'terminating and respawning', wid, age)
                p.terminate()
                p.join(timeout=10)
                procs[wid] = _spawn_worker(wid, args, stop_event)

        # Check for overall completion.
        q = ErdQueue(args.queue)
        counts = q.counts_by_status()
        q.close()
        if (counts.get('pending', 0) == 0
                and counts.get('in_progress', 0) == 0
                and counts):
            logger.info('Queue drained — all subgroups done.')
            print('\nQueue empty — all subgroups done.')
            stop_event.set()

    logger.info('Supervisor stopping all workers...')
    for wid, (p, _) in procs.items():
        if p.is_alive():
            p.terminate()
    for wid, (p, _) in procs.items():
        p.join(timeout=30)
    logger.info('Supervisor exited.')
    print('All workers stopped.')


def _spawn_worker(worker_id: int, args, stop_event):
    p = multiprocessing.Process(
        target=erd_worker.main,
        args=(worker_id, args.cache, args.queue,
              args.recycle_after, args.checkpoint_every,
              stop_event),
        daemon=False,
        name=f'erd-worker-{worker_id}',
    )
    p.start()
    logger.info('Spawned worker-%d (pid=%d)', worker_id, p.pid)
    return p, time.time()


def _setup_supervisor_logging():
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'erd_search.log')
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    print(f'Logging to {log_path}')


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    if args.watch is not None:
        interval = args.watch if args.watch > 0 else 30
        try:
            while True:
                print('\033[2J\033[H', end='')  # clear screen
                _print_status(args)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    else:
        _print_status(args)


def _print_status(args):
    now_ts = int(time.time())

    # Queue stats
    try:
        queue = ErdQueue(args.queue)
        counts = queue.counts_by_status()
        words = queue.words_by_status()
        total = queue.total_subgroups()
        hbs = queue.heartbeats()
        bootstrap_status = queue.get_meta('bootstrap_status')
        queue.close()
        queue_ok = True
    except Exception as e:
        print(f'Queue unavailable: {e}')
        queue_ok = False
        counts = words = {}
        total = 0
        hbs = []
        bootstrap_status = None

    # Cache stats
    try:
        all_answers = load_word_list(ANSWER_FILE)
        sc = ScoreCache(args.cache, all_answers)
        total_erd = sc._conn.execute(
            "SELECT COUNT(*) FROM subgroup_best_by_policy "
            "WHERE policy=? AND universe_id=?",
            (ERD_ALL, sc.universe_id)).fetchone()[0]
        recent = sc._conn.execute(
            "SELECT COUNT(*) FROM subgroup_best_by_policy "
            "WHERE policy=? AND universe_id=? AND updated_at>?",
            (ERD_ALL, sc.universe_id, now_ts - 300)).fetchone()[0]
        uid_short = sc.universe_id[:16]
        sc.close()
        cache_ok = True
    except Exception as e:
        print(f'Cache unavailable: {e}')
        cache_ok = False
        total_erd = recent = 0
        uid_short = '?'

    print(f'ERD_ALL Precache — '
          f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if bootstrap_status != 'done':
        print(f'  !! bootstrap: {bootstrap_status or "not started"} !!')
    print()

    if queue_ok:
        print('Queue  (erd_queue.sqlite3):')
        for status in ('pending', 'in_progress', 'done'):
            n = counts.get(status, 0)
            w = words.get(status, 0)
            print(f'  {status:12s}  {n:8,} subgroups   {w:12,} words')
        print(f'  {"total":12s}  {total:8,} subgroups')
        print()

    if cache_ok:
        rate = recent / 5.0   # rows per minute
        print(f'Cache  (wordle_cache.sqlite3, universe {uid_short}...):')
        print(f'  {total_erd:,} ERD_ALL rows total')
        print(f'  +{recent:,} in last 5 min  (~{rate:.1f}/min)')
        if rate > 0 and queue_ok:
            remaining = (counts.get('pending', 0) +
                         counts.get('in_progress', 0))
            eta_min = remaining / rate
            print(f'  ETA (rough, ignores side-effect fill-in): '
                  f'~{_fmt_duration(int(eta_min * 60))}')
        print()

    print('Workers:')
    if not hbs:
        print('  (none active)')
    for hb in hbs:
        age_s = now_ts - hb['updated_at']
        uptime_s = now_ts - (hb['started_at'] or now_ts)
        subject = ''
        if hb['current_subset_key']:
            from erd_queue import decode_subset
            ws = decode_subset(bytes(hb['current_subset_key']))
            subject = (f' {ws[0].upper()}..{ws[-1].upper()}'
                       f' ({hb["n_words"]}w)')
        stale = '  !! STALE !!' if age_s > 120 else ''
        print(f'  {hb["worker_id"]:<12s}  pid={hb["pid"]:<7d}'
              f'{subject:<30s}'
              f'  done={hb["subgroups_done"]:<6d}'
              f'  up={_fmt_duration(uptime_s)}'
              f'  hb={age_s}s ago'
              f'{stale}')


def _fmt_duration(seconds: int) -> str:
    if seconds < 0:
        return '0s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f'{h}h{m:02d}m'


# ---------------------------------------------------------------------------
# reset-stale
# ---------------------------------------------------------------------------

def cmd_reset_stale(args):
    queue = ErdQueue(args.queue)
    n = queue.reset_stale_in_progress()
    queue.close()
    print(f'Reset {n} in_progress row(s) to pending.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # -- bootstrap --
    p_boot = sub.add_parser('bootstrap',
                             help='Populate the work queue from root words')
    p_boot.add_argument('--root-words', metavar='FILE',
                        help=f'Root-word list (default: {WORDS_FILE})')
    p_boot.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH',
                        help=f'Main cache DB (default: {DEFAULT_CACHE})')
    p_boot.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH',
                        help=f'Queue DB (default: {DEFAULT_QUEUE})')
    p_boot.add_argument('--priority-words', nargs='+', metavar='WORD',
                        help='Root words whose branches are worked first '
                             '(e.g. --priority-words salet crane)')

    # -- run --
    p_run = sub.add_parser('run', help='Start the parallel precache supervisor')
    p_run.add_argument('--workers', type=int, default=6, metavar='N',
                       help='Number of worker processes (default: 6)')
    p_run.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_run.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_run.add_argument('--recycle-after', type=int, default=1000,
                       metavar='N',
                       help='Respawn worker after N completed subgroups '
                            '(controls _mem_cache growth, default: 1000)')
    p_run.add_argument('--recycle-hours', type=float, default=3.0,
                       metavar='H',
                       help='Also respawn after H hours wall time '
                            '(default: 3)')
    p_run.add_argument('--checkpoint-every', type=int, default=100,
                       metavar='N',
                       help='WAL checkpoint every N completed subgroups '
                            'per worker (default: 100)')
    p_run.add_argument('--allow-partial-queue', action='store_true',
                       help='Start workers even if bootstrap is not complete')

    # -- status --
    p_stat = sub.add_parser('status', help='Show progress snapshot')
    p_stat.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_stat.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_stat.add_argument('--watch', nargs='?', const=30, type=int,
                        metavar='SECONDS',
                        help='Repeat every SECONDS (default 30)')

    # -- reset-stale --
    p_rst = sub.add_parser('reset-stale',
                            help='Reset in_progress rows to pending')
    p_rst.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    args = parser.parse_args()

    dispatch = {
        'bootstrap': cmd_bootstrap,
        'run': cmd_run,
        'status': cmd_status,
        'reset-stale': cmd_reset_stale,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
