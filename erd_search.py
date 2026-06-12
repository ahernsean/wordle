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

export      Create a trimmed snapshot of the cache containing only the
            tables needed on iPhone (universe, response_decomposition,
            subgroup_best_by_policy).  Safe to run while workers are
            active.  Re-running is incremental (INSERT OR IGNORE).
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
                (encode_subset(branch), len(branch), priority, word, code)
                for code, branch in groups.items()
                if 2 <= len(branch) <= args.max_size
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
        hbs = queue.heartbeats_with_source()
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
        print(f'  +{recent:,} in last 5 min  (~{rate:.1f}/min fill-in rate)')
        print()

    print('Workers:')
    if not hbs:
        print('  (none active)')
    from erd_queue import fmt_pattern, decode_subset
    for hb in hbs:
        age_s  = now_ts - hb['updated_at']
        up_s   = now_ts - (hb['started_at'] or now_ts)
        stale  = '  !!STALE!!' if age_s > 120 else ''

        pri    = hb['priority']
        tag    = '[P1]' if pri == 1 else '[P0]' if pri == 0 else '[??]'

        if hb['source_word'] and hb['source_pattern'] is not None:
            src = f'{hb["source_word"].upper()} {fmt_pattern(hb["source_pattern"])}'
        elif hb['current_subset_key']:
            ws  = decode_subset(bytes(hb['current_subset_key']))
            src = f'{ws[0].upper()}..{ws[-1].upper()}'
        else:
            src = '-----'

        n = hb['n_words'] or 0

        cd = hb['candidates_done']
        ct = hb['candidates_total']
        if cd is not None and ct:
            cands = f'{cd:5d}/{ct}'
        elif ct:
            cands = f'    0/{ct}'
        else:
            cands = '    -/-----'

        bw  = (hb['best_word'] or '-----').upper()
        be  = f'{hb["best_erd"]:.3f}' if hb['best_erd'] is not None else '  ---'

        print(f'  {hb["worker_id"]:<10s} {tag}  {src:<13s} {n:5d}w'
              f'  {cands}  {bw} {be:>5s}'
              f'  d={hb["subgroups_done"]:<4d} {_fmt_duration(up_s):>7s}  hb={age_s}s'
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
# export
# ---------------------------------------------------------------------------

EXPORT_TABLES = ['universe', 'response_decomposition', 'subgroup_best_by_policy']
DEFAULT_EXPORT = 'wordle_erd_export.sqlite3'


def cmd_export(args):
    """Create a trimmed export file with only the iPhone-useful tables.

    Safe to run while workers are active: WAL mode allows concurrent reads,
    so the export sees a consistent snapshot without stopping anything.
    Re-running is incremental: INSERT OR IGNORE skips rows already present,
    so you can refresh the export file at any time.
    """
    import sqlite3 as _sqlite3
    import re

    export_path = args.output or DEFAULT_EXPORT
    cache_path = os.path.abspath(args.cache)
    export_path = os.path.abspath(export_path)

    print(f'Source : {cache_path}')
    print(f'Export : {export_path}')
    print()

    conn = _sqlite3.connect(export_path, timeout=30.0, isolation_level=None)
    conn.row_factory = _sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute(f"ATTACH DATABASE '{cache_path}' AS src")

    try:
        conn.execute('BEGIN')

        total_new = 0
        for table in EXPORT_TABLES:
            # Copy CREATE TABLE statement from source, adding IF NOT EXISTS.
            schema_row = conn.execute(
                "SELECT sql FROM src.sqlite_master "
                "WHERE type='table' AND name=?", (table,)).fetchone()
            if schema_row is None:
                print(f'  {table}: not found in source, skipping')
                continue

            create_sql = re.sub(
                r'^(CREATE\s+TABLE\s+)',
                r'\1IF NOT EXISTS ',
                schema_row[0],
                count=1, flags=re.IGNORECASE)
            conn.execute(create_sql)

            # Copy indexes.
            for idx_row in conn.execute(
                    "SELECT sql FROM src.sqlite_master "
                    "WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                    (table,)):
                idx_sql = re.sub(
                    r'^(CREATE\s+(?:UNIQUE\s+)?INDEX\s+)',
                    r'\1IF NOT EXISTS ',
                    idx_row[0],
                    count=1, flags=re.IGNORECASE)
                try:
                    conn.execute(idx_sql)
                except _sqlite3.OperationalError:
                    pass  # already exists

            cols = [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]
            col_list = ', '.join(cols)
            conn.execute(f"""
                INSERT OR IGNORE INTO main.{table} ({col_list})
                SELECT {col_list} FROM src.{table}
            """)
            n = conn.execute('SELECT changes()').fetchone()[0]
            total = conn.execute(
                f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            print(f'  {table}: +{n:,} new rows  ({total:,} total)')
            total_new += n

        conn.execute('COMMIT')
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')

        size_mb = os.path.getsize(export_path) / 1e6
        print(f'\nDone.  {export_path}  ({size_mb:.0f} MB)')
        if total_new == 0:
            print('(Already up to date.)')

    except Exception:
        try:
            conn.execute('ROLLBACK')
        except Exception:
            pass
        raise
    finally:
        try:
            conn.execute('DETACH DATABASE src')
        except Exception:
            pass
        conn.close()


# ---------------------------------------------------------------------------
# solve-branch (cooperative candidate-level solve of one subgroup)
# ---------------------------------------------------------------------------

def cmd_solve_branch(args):
    """Solve one subgroup by swarming N workers across the candidate guesses.

    For a branch too large for a single worker to finish (e.g. an opener's
    all-gray response), this fans the ~12,972 candidate guesses out across
    workers sharing one running best ERD, instead of grinding them serially.
    Writes the result (and every sub-subgroup, as a recursion side effect) to
    the persistent cache.
    """
    import threading
    from erd_queue import parse_pattern, fmt_pattern
    from erd_split import run_split_solve

    all_answers = load_word_list(ANSWER_FILE)
    word = args.word.strip().lower()
    code = parse_pattern(args.pattern)

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    groups = rcache.group_words(word, all_answers)
    branch = groups.get(code, [])
    pat = fmt_pattern(code)

    if len(branch) < 2:
        print(f'{word.upper()} {pat}: branch has {len(branch)} word(s) — '
              f'nothing to solve (singletons resolve on the next guess).')
        sc.close()
        return

    subset_key = encode_subset(branch)
    existing = sc.read(subset_key, ERD_ALL)
    sc.close()

    print(f'{word.upper()} {pat}  —  {len(branch)} words')
    if existing is not None and not args.force:
        print(f'Already cached: {existing[0].upper()} ERD={existing[1]:.4f}  '
              f'(pass --force to recompute)')
        return

    n_chunks_box = {}
    stop = threading.Event()

    def monitor():
        """Poll the split row and print live progress until solving finishes."""
        started = time.time()
        while not stop.is_set():
            try:
                q = ErdQueue(args.queue)
                row = q.get_split(subset_key)
                if row is not None:
                    n_cand = row['n_candidates']
                    chunk = row['chunk']
                    n_chunks = ErdQueue.n_chunks_for(n_cand, chunk)
                    n_chunks_box['n'] = n_chunks
                    done = q.split_done_chunks(subset_key)
                    bw, be = q.read_split_best(subset_key)
                    el = int(time.time() - started)
                    pct = 100.0 * done / n_chunks if n_chunks else 0.0
                    best = (f'{bw.upper()} {be:.4f}'
                            if bw is not None else 'searching...')
                    eta = ''
                    if done > 0 and done < n_chunks:
                        rate = done / max(1, el)            # chunks/sec
                        rem = (n_chunks - done) / rate if rate else 0
                        eta = f'  ETA {_fmt_duration(int(rem))}'
                    print(f'\r  chunks {done:4d}/{n_chunks:<4d} ({pct:4.0f}%)  '
                          f'best={best:<18s}  {_fmt_duration(el)}{eta}   ',
                          end='', flush=True)
                q.close()
            except Exception:
                pass
            stop.wait(3.0)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    t0 = time.time()
    if args.force:
        # Recompute: drop any stale split + cached entry so we start clean.
        sc2 = ScoreCache(args.cache, all_answers)
        sc2.delete(subset_key, ERD_ALL)
        sc2.close()

    result = run_split_solve(
        subset_key, branch, n_workers=args.workers, chunk=args.chunk,
        cache_path=args.cache, queue_path=args.queue,
        priority=args.priority, source_word=word, source_pattern=code)

    stop.set()
    mon.join(timeout=4)
    print()
    elapsed = _fmt_duration(int(time.time() - t0))
    if result is not None:
        print(f'Done in {elapsed}:  {word.upper()} {pat}  ->  '
              f'{result[0].upper()}  ERD={result[1]:.4f}')
    else:
        print(f'No result after {elapsed} — check workers / rerun to resume.')


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
    p_boot.add_argument('--max-size', type=int, default=300, metavar='N',
                        help='Skip branches with more than N answer words '
                             '(default: 300; excludes computationally '
                             'infeasible large subgroups from poor root words)')

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

    # -- solve-branch --
    p_sb = sub.add_parser('solve-branch',
                          help='Swarm N workers to solve one large branch')
    p_sb.add_argument('--word', required=True, metavar='WORD',
                      help='Opening guess word (e.g. salet)')
    p_sb.add_argument('--pattern', required=True, metavar='PAT',
                      help="Response pattern, 5 chars: g=green y=yellow, gray "
                           "as . or - (use dots to avoid the shell/argparse "
                           "leading-dash trap, e.g. --pattern ..... for "
                           "all-gray, or --pattern=-y-g-)")
    p_sb.add_argument('--workers', type=int, default=6, metavar='N',
                      help='Worker processes to swarm the branch (default: 6)')
    p_sb.add_argument('--chunk', type=int, default=128, metavar='N',
                      help='Candidate guesses per claimable chunk '
                           '(default: 128; smaller = finer load balance, '
                           'more coordination writes)')
    p_sb.add_argument('--priority', type=int, default=1, metavar='P',
                      help='Priority recorded on the split (default: 1)')
    p_sb.add_argument('--force', action='store_true',
                      help='Recompute even if the branch is already cached')
    p_sb.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_sb.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- reset-stale --
    p_rst = sub.add_parser('reset-stale',
                            help='Reset in_progress rows to pending')
    p_rst.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- export --
    p_exp = sub.add_parser('export',
                            help='Create a trimmed iPhone-ready cache snapshot')
    p_exp.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH',
                       help=f'Source cache (default: {DEFAULT_CACHE})')
    p_exp.add_argument('--output', default=DEFAULT_EXPORT, metavar='PATH',
                       help=f'Output file (default: {DEFAULT_EXPORT})')

    args = parser.parse_args()

    dispatch = {
        'bootstrap': cmd_bootstrap,
        'run': cmd_run,
        'status': cmd_status,
        'reset-stale': cmd_reset_stale,
        'export': cmd_export,
        'solve-branch': cmd_solve_branch,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
