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
import erd_swarm

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

def _checkpoint_cache_on_start(cache_path):
    """Flush any leftover WAL into the main DB through SQLite's own recovery.

    A worker killed mid-write leaves a -wal holding committed transactions.
    NEVER delete that file: SQLite replays it on the next open, and removing
    it discards committed data and corrupts the main DB.  Instead open the
    DB single-threaded and TRUNCATE-checkpoint, which is the blessed way to
    drain the WAL cleanly before the worker swarm starts hammering it.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(cache_path, timeout=60)
        conn.execute('PRAGMA busy_timeout = 60000')
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.close()
        print('Startup: WAL checkpointed clean.')
    except sqlite3.Error as e:
        print(f'Startup: WAL checkpoint failed: {e}', file=sys.stderr)


def cmd_run(args):
    _checkpoint_cache_on_start(args.cache)
    queue = ErdQueue(args.queue)
    stale = queue.reset_stale_in_progress()
    nb, nc = queue.reset_active_branches()
    if stale or nb or nc:
        print(f'Recovery: {stale} pending rows reset, '
              f'{nb} in-progress branches / {nc} chunk claims cleared.')

    bootstrap_status = queue.get_meta('bootstrap_status')
    if bootstrap_status != 'done' and not args.allow_partial_queue:
        print('Error: bootstrap not marked complete.  '
              'Run bootstrap first, or pass --allow-partial-queue to proceed anyway.',
              file=sys.stderr)
        queue.close()
        sys.exit(1)
    queue.close()

    _setup_supervisor_logging()
    logger.info('Supervisor starting: %d workers, divisor=%d, max_chunks=%d, '
                'recycle_hours=%.1f', args.workers, args.divisor,
                args.max_chunks, args.recycle_hours)

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
    print(f'Started {args.workers} swarm workers  (pid={os.getpid()}).')
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

        # Free chunks claimed by a worker that died mid-chunk so they get
        # redone rather than stranding a branch one chunk short of finalizing.
        q = ErdQueue(args.queue)
        freed = q.reclaim_stale_chunks(args.stale_chunk_seconds)
        if freed:
            logger.info('Reclaimed %d stale chunk claim(s).', freed)
        counts = q.counts_by_status()
        in_flight = len(q.branches_in_progress())
        q.close()

        # Done when the queue holds no pending or in-progress branches and no
        # branch is still being swarmed.
        if (counts.get('pending', 0) == 0
                and counts.get('in_progress', 0) == 0
                and in_flight == 0
                and counts):
            logger.info('Queue drained — all branches done.')
            print('\nQueue empty — all branches done.')
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
        target=erd_swarm.swarm_worker,
        args=(worker_id, args.cache, args.queue, stop_event,
              args.divisor, args.max_chunks),
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
    from erd_queue import fmt_pattern

    # Queue + swarm state
    try:
        queue = ErdQueue(args.queue)
        counts = queue.counts_by_status()
        branches = queue.branches_in_progress()
        hbs = queue.heartbeats_with_branch()
        worker_counts = queue.worker_counts_by_branch()
        bootstrap_status = queue.get_meta('bootstrap_status')
        # Per-branch done-chunk counts for the in-progress branches.
        done_chunks = {bytes(b['subset_key']): queue.branch_done_chunks(b['subset_key'])
                       for b in branches}
        queue.close()
        queue_ok = True
    except Exception as e:
        print(f'Queue unavailable: {e}')
        queue_ok = False
        counts = {}
        branches = []
        hbs = []
        worker_counts = {}
        done_chunks = {}
        bootstrap_status = None

    # Cache throughput
    try:
        all_answers = load_word_list(ANSWER_FILE)
        sc = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)
        total_erd = sc._conn.execute(
            "SELECT COUNT(*) FROM subgroup_best_by_policy "
            "WHERE policy=? AND universe_id=?",
            (ERD_ALL, sc.universe_id)).fetchone()[0]
        recent = sc._conn.execute(
            "SELECT COUNT(*) FROM subgroup_best_by_policy "
            "WHERE policy=? AND universe_id=? AND updated_at>?",
            (ERD_ALL, sc.universe_id, now_ts - 300)).fetchone()[0]
        sc.close()
        cache_ok = True
    except Exception as e:
        print(f'Cache unavailable: {e}')
        cache_ok = False
        total_erd = recent = 0

    # Aggregate cache-hit and pruning effectiveness across live workers.
    live = [h for h in hbs if now_ts - h['updated_at'] <= 120]
    hits = sum(h['cache_hits'] or 0 for h in live)
    misses = sum(h['cache_misses'] or 0 for h in live)
    n_ok = sum(h['n_ok'] or 0 for h in live)
    n_pruned = sum(h['n_pruned'] or 0 for h in live)
    hit_pct = (100.0 * hits / (hits + misses)) if (hits + misses) else None
    prune_pct = (100.0 * n_pruned / (n_ok + n_pruned)) if (n_ok + n_pruned) else None

    print(f'ERD_ALL Precache — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if bootstrap_status != 'done':
        print(f'  !! bootstrap: {bootstrap_status or "not started"} !!')

    if queue_ok:
        print(f'Queue:  pending {counts.get("pending", 0):,}   '
              f'done {counts.get("done", 0):,}   '
              f'branches in progress {len(branches)}')
    if cache_ok:
        extra = ''
        if hit_pct is not None:
            extra += f'   hits {hit_pct:.0f}%'
        if prune_pct is not None:
            extra += f'   pruned {prune_pct:.0f}%'
        print(f'Cache:  {total_erd:,} rows   +{recent:,}/5m{extra}')
    print()

    # Branches in progress — the real progress unit.
    print('Branches in progress:')
    if not branches:
        print('  (none)')
    for b in branches:
        key = bytes(b['subset_key'])
        n_chunks = ErdQueue.n_chunks_for(b['n_candidates'], b['chunk_size'])
        done = done_chunks.get(key, 0)
        pct = 100.0 * done / n_chunks if n_chunks else 0.0
        src = (f'{b["source_word"].upper()} {fmt_pattern(b["source_pattern"])}'
               if b['source_word'] and b['source_pattern'] is not None
               else '-----')
        bw = (b['best_word'] or '-----').upper()
        be = f'{b["best_erd"]:.3f}' if b['best_erd'] is not None else '  ---'
        nw = b['n_words'] or 0
        wk = worker_counts.get(key, 0)
        created = b['created_at'] or now_ts
        el = now_ts - created
        eta = ''
        if 0 < done < n_chunks and el > 0:
            rem = (n_chunks - done) / (done / el)
            eta = f'  ~{_fmt_duration(int(rem))}'
        print(f'  {src:<13s} {nw:4d}w  chunks {done:3d}/{n_chunks:<3d} '
              f'({pct:3.0f}%)  {bw} {be:>5s}  {wk}w{eta}')
    print()

    # Workers — liveness only (alive and moving, or stuck/idle).
    print('Workers (health):')
    if not hbs:
        print('  (none active)')
    for h in sorted(hbs, key=lambda r: r['worker_id']):
        age = now_ts - h['updated_at']
        flag = '  !!STALE' if age > 120 else ''
        key = h['current_subset_key']
        if key is None:
            print(f'  {h["worker_id"]:<10s} idle{"":40s}hb={age}s{flag}')
            continue
        src = (f'{h["source_word"].upper()} {fmt_pattern(h["source_pattern"])}'
               if h['source_word'] and h['source_pattern'] is not None
               else '-----')
        chunk = h['chunk_idx'] if h['chunk_idx'] is not None else '-'
        held = now_ts - (h['chunk_started_at'] or now_ts)
        rate = h['cand_rate']
        rate_s = f'{rate/1000:.1f}k c/s' if rate else '  -  '
        done = h['chunks_done'] or 0
        cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
        n_seen = (h['cand_n_seen'] if 'cand_n_seen' in h.keys() else None) or 0
        c_total = (h['cand_chunk_size'] if 'cand_chunk_size' in h.keys() else None) or 0
        mdepth = (h['cur_max_depth'] if 'cur_max_depth' in h.keys() else None) or 0
        nodes = (h['cur_nodes'] if 'cur_nodes' in h.keys() else None) or 0
        nrate = (h['node_rate'] if 'node_rate' in h.keys() else None) or 0.0
        path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
        # Forward-progress flag: heartbeat fresh but no nodes moving == real hang.
        moving = '  !!HANG' if (age <= 10 and nrate == 0 and nodes) else ''
        cand_s = (f' [{cur} {n_seen}/{c_total} d{mdepth} '
                  f'{nodes/1e6:.1f}M nodes {nrate/1000:.0f}k/s sp:{path}]'
                  if cur else '')
        print(f'  {h["worker_id"]:<10s} {src:<13s} chunk {str(chunk):>3s} '
              f'held {_fmt_duration(held):>5s}  '
              f'done {done:<4d} hb={age}s{flag}{moving}{cand_s}')


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
# solve-branch (swarm N workers on one branch's candidate guesses)
# ---------------------------------------------------------------------------

def cmd_solve_branch(args):
    """Solve one branch by swarming N workers across its candidate guesses.

    For a branch too large for a single worker to finish (e.g. an opener's
    all-gray response), this fans the ~12,972 candidate guesses out across
    workers sharing one running best ERD, instead of grinding them serially.
    Writes the result (and every sub-branch, as a recursion side effect) to
    the persistent cache.
    """
    import threading
    from erd_queue import parse_pattern, fmt_pattern
    from erd_swarm import run_branch_solve

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

    stop = threading.Event()

    def monitor():
        """Poll the branch row and print live progress until it finishes."""
        started = time.time()
        while not stop.is_set():
            try:
                q = ErdQueue(args.queue)
                row = q.get_branch(subset_key)
                if row is not None:
                    n_chunks = ErdQueue.n_chunks_for(
                        row['n_candidates'], row['chunk_size'])
                    done = q.branch_done_chunks(subset_key)
                    bw, be = q.read_branch_best(subset_key)
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

    if args.force:
        sc2 = ScoreCache(args.cache, all_answers)
        sc2.delete(subset_key, ERD_ALL)
        sc2.close()

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    t0 = time.time()
    result = run_branch_solve(
        subset_key, branch, n_workers=args.workers,
        cache_path=args.cache, queue_path=args.queue,
        divisor=args.divisor, max_chunks=args.max_chunks,
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
                       help='Number of swarm worker processes (default: 6)')
    p_run.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_run.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_run.add_argument('--divisor', type=int, default=3, metavar='D',
                       help='Chunk-count divisor: a branch is cut into about '
                            'n_words/D chunks (default: 3). Lower = finer '
                            'chunks / more sharing on hard branches.')
    p_run.add_argument('--max-chunks', type=int, default=256, metavar='N',
                       help='Cap on chunks per branch (default: 256)')
    p_run.add_argument('--recycle-hours', type=float, default=3.0,
                       metavar='H',
                       help='Respawn each worker after H hours wall time, to '
                            'bound ScoreCache memory growth (default: 3)')
    p_run.add_argument('--stale-chunk-seconds', type=int, default=600,
                       metavar='S',
                       help='Reclaim a chunk claimed but not completed within '
                            'S seconds (crashed worker; default: 600)')
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
    p_sb.add_argument('--divisor', type=int, default=3, metavar='D',
                      help='Chunk-count divisor: ~n_words/D chunks (default: 3)')
    p_sb.add_argument('--max-chunks', type=int, default=256, metavar='N',
                      help='Cap on chunks for the branch (default: 256)')
    p_sb.add_argument('--priority', type=int, default=1, metavar='P',
                      help='Priority recorded on the branch (default: 1)')
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
