#!/usr/bin/env python3.13
"""erd_search.py — Parallel ERD_ALL precache CLI.

Subcommands
-----------
start           Start the supervisor via systemd (systemctl --user start).
stop            Stop the supervisor via systemd (systemctl --user stop).
restart         Restart the supervisor via systemd (systemctl --user restart):
                a stop followed by a start in one step.

view            Shared swarm reports in text, JSON, or watched JSON Lines.

run             Start the supervisor directly (without systemd), for
                development or one-shot use.  All output goes to erd_search.log.

queue           Queue mutation operations.
queue add       Add branches for a word or word list to the work queue.
                Idempotent: existing branches are never duplicated; priority
                is upgraded if the new request is higher.  With --word-list,
                --priority-words marks a subset of the list's words as
                higher priority.  --delete-erd-cache forces a recompute of
                branches that are already cached.

queue clear     Wipe all queue state (pending branches, active state, candidate
                claims, heartbeats).  Does not touch the ERD cache.

queue remove    Remove a pending branch from the queue.  Use --force to also
                cancel an in-progress branch (workers move on after their
                current candidate evaluation completes).

queue priority  Change the priority of a queued branch.  Higher numbers are
                worked sooner; 0 is the default.

For exporting a trimmed cache snapshot to sync to the iPhone, or importing
one from another machine, see export_cache.py and import_cache.py — the
cache is shared with interactive play (wordle.py), not swarm-specific.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sqlite3
import signal
import sys
import time
from datetime import datetime

from cache_sqlite import ScoreCache
from report_model import (
    ReportFilters,
    ReportRequest,
    WORKER_LIVENESS_SECONDS,
    parse_report_selector,
    parse_rich_spine as _parse_spine,
    validate_report_request,
)
from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    DEFAULT_QUEUE_PATH,
)
from wordle_engine import ERD_ALL, ResponseCache, load_word_list
from erd_queue import (
    DISK_STOP_FRACTION,
    ERDQueue,
    disk_stats,
    encode_subset,
)
import erd_swarm

ANSWER_FILE = DEFAULT_ANSWER_LIST_PATH
WORDS_FILE = DEFAULT_CANDIDATE_LIST_PATH
DEFAULT_CACHE = DEFAULT_CACHE_PATH
DEFAULT_QUEUE = DEFAULT_QUEUE_PATH

logger = logging.getLogger('wordle')


def _view_watch_interval(value):
    interval = float(value)
    if interval < 0.2:
        raise argparse.ArgumentTypeError("watch interval must be at least 0.2 seconds")
    return interval


def cmd_view(args):
    from report_terminal import run_view

    run_view(args)


# ---------------------------------------------------------------------------
# queue add
# ---------------------------------------------------------------------------

def cmd_queue_add(args):
    """Add branches for one word (or a word-list file) to the queue.

    With --word: adds all response branches for that word with at least 2
    answer words (and at most --max-branch-size, if given).  With --pattern
    as well: adds only that single branch.

    With --word-list: walks every word in the file, same as --word repeated.
    --priority-words marks a subset of those words as higher priority: they
    are queued at --priority while the rest are queued at 0.

    Already-queued branches are never duplicated; their priority is upgraded
    if the new request is higher.  --delete-erd-cache deletes each queued
    branch's existing ERD cache entry first, so it gets recomputed instead of
    being claimed and immediately marked done as already-cached.
    """
    from wordle_ui import parse_pattern, fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    priority_words = {w.strip().lower() for w in (args.priority_words or [])}
    if priority_words and not args.word_list:
        print('Warning: --priority-words only applies with --word-list; '
              'ignoring it.  Use --priority directly for a single --word.')
        priority_words = set()

    score_cache = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ERDQueue(args.queue)

    if args.word:
        words_to_process = [args.word.strip().lower()]
    else:
        words_to_process = load_word_list(args.word_list)

    unknown = priority_words - set(words_to_process)
    if unknown:
        print(f'Warning: priority words not in the word list: '
              f'{", ".join(sorted(unknown))}')

    n_added = 0
    try:
        for word in words_to_process:
            priority = (args.priority if (not priority_words
                                           or word in priority_words)
                        else 0)
            if args.pattern:
                code = parse_pattern(args.pattern)
                groups = rcache.group_words(word, all_answers)
                branch = groups.get(code, [])
                if len(branch) < 2:
                    print(f'{word.upper()} {fmt_pattern(code)}: '
                          f'{len(branch)} answer word(s) — nothing to queue.')
                    continue
                if (args.max_branch_size is not None
                        and len(branch) > args.max_branch_size):
                    print(f'{word.upper()} {fmt_pattern(code)}: '
                          f'{len(branch)} words exceeds --max-branch-size '
                          f'{args.max_branch_size}, skipping.')
                    continue
                rows = [(encode_subset(branch), len(branch), priority,
                         word, code)]
            else:
                groups = rcache.group_words(word, all_answers)
                rows = [
                    (encode_subset(branch), len(branch), priority, word, code)
                    for code, branch in groups.items()
                    if len(branch) >= 2
                    and (args.max_branch_size is None
                         or len(branch) <= args.max_branch_size)
                ]
            if rows:
                if args.delete_erd_cache:
                    for branch_key, *_rest in rows:
                        score_cache.delete(branch_key, ERD_ALL)
                queue.add_pending_many(rows)
                n_added += len(rows)

        total = queue.total_branches()
        print(f'Added {n_added:,} branch(es).  Queue total: {total:,}.')

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        score_cache.checkpoint()
        score_cache.close()
        queue.close()


# ---------------------------------------------------------------------------
# queue clear
# ---------------------------------------------------------------------------

def cmd_queue_clear_disk_stop(args):
    """Release the disk-stop latch so `run` will start again."""
    queue = ERDQueue(args.queue)
    try:
        latch = queue.disk_stop()
        if latch is None:
            print('No disk-stop latch is set.')
            return
        used_fraction = disk_stats(args.queue)['used_fraction']
        queue.clear_disk_stop()
        print(f'Disk-stop latch cleared (was: {latch["reason"]}).  '
              f'Disk is now {100 * used_fraction:.1f}% full.')
        if used_fraction >= DISK_STOP_FRACTION:
            print(f'Warning: still at or above the '
                  f'{100 * DISK_STOP_FRACTION:.0f}% stop threshold — '
                  f'run will refuse to start until space is freed.',
                  file=sys.stderr)
    finally:
        queue.close()


def cmd_queue_clear(args):
    """Wipe all queue state (pending branches, active branches, candidate claims,
    heartbeats, and run metadata).  The ERD cache is not touched.

    Requires confirmation unless --yes is passed.
    """
    queue = ERDQueue(args.queue)
    try:
        counts = queue.counts_by_status()
        pending = counts.get('pending', 0)
        done = counts.get('done', 0)
        in_prog = len(queue.branches_in_progress())

        print(f'Queue: {pending:,} pending   {done:,} done   {in_prog} in progress')
        if not args.yes:
            ans = input('Clear all queue state? [y/N] ').strip().lower()
            if ans != 'y':
                print('Aborted.')
                return

        queue.clear()
        print('Queue cleared.')
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# queue remove
# ---------------------------------------------------------------------------

def cmd_queue_remove(args):
    """Remove a branch from the pending queue.

    Only removes branches with status='pending'.  If the branch is currently
    in-progress (being worked by a worker), use --force to also cancel it by
    clearing its active_branches and candidate_claims rows so the worker's next heartbeat
    yields no further claims.  The worker will eventually notice the branch
    is gone and move on.
    """
    from wordle_ui import parse_pattern, fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    word = args.word.strip().lower()
    code = parse_pattern(args.pattern)
    pat = fmt_pattern(code)

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    groups = rcache.group_words(word, all_answers)
    branch = groups.get(code, [])
    sc.close()

    if not branch:
        print(f'{word.upper()} {pat}: no branch.')
        return

    branch_key = encode_subset(branch)
    queue = ERDQueue(args.queue)

    active = queue.get_active_branch(branch_key)
    if active and not args.force:
        print(f'{word.upper()} {pat}: branch is in-progress.  '
              f'Use --force to also cancel the active work.')
        queue.close()
        return

    if active and args.force:
        # Atomically clear candidate claims, the active_branches row, and the
        # pending_branches row.  All three DELETEs run in one transaction so a
        # crash partway through cannot leave orphaned rows.  (remove_pending()
        # alone would silently no-op here because the pending row still has
        # status='in_progress' after the active state is cleared.)
        queue.cancel_active_branch(branch_key, remove_from_queue=True)
        queue.close()
        print(f'Cancelled in-progress work and removed {word.upper()} {pat} from queue.')
        return

    removed = queue.remove_pending(branch_key)
    queue.close()

    if removed:
        print(f'Removed {word.upper()} {pat} from queue.')
    else:
        print(f'{word.upper()} {pat}: not found in pending queue '
              f'(may already be done or not queued).')


# ---------------------------------------------------------------------------
# queue priority
# ---------------------------------------------------------------------------

def cmd_queue_priority(args):
    """Set the priority of a queued branch.

    Priority is an integer; higher numbers are worked sooner.  The internal
    swarm uses priority 1,000,000 for cooperative sub-branches so they always
    drain before fresh top-level branches.  User-settable values in the range
    0–999 are reserved for normal use: 0 = default, higher = sooner.
    """
    from wordle_ui import parse_pattern, fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    word = args.word.strip().lower()
    code = parse_pattern(args.pattern)
    pat = fmt_pattern(code)

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    groups = rcache.group_words(word, all_answers)
    branch = groups.get(code, [])
    sc.close()

    if not branch:
        print(f'{word.upper()} {pat}: no branch.')
        return

    branch_key = encode_subset(branch)
    queue = ERDQueue(args.queue)
    updated = queue.set_priority(branch_key, args.priority)
    queue.close()

    if updated:
        print(f'{word.upper()} {pat}: priority set to {args.priority}.')
    else:
        print(f'{word.upper()} {pat}: not found in pending queue.')


# ---------------------------------------------------------------------------
# start / stop  (systemd delegation)
# ---------------------------------------------------------------------------

_SYSTEMD_SERVICE = 'wordle-erd'


def _run_systemctl(action: str, *extra: str) -> int:
    """Run `systemctl --user <action> <service> [extra...]` and return the
    exit code."""
    import subprocess
    result = subprocess.run(
        ['systemctl', '--user', action, _SYSTEMD_SERVICE, *extra],
        capture_output=False)
    return result.returncode


def cmd_start(_args):
    """Start the supervisor via systemd."""
    rc = _run_systemctl('start')
    if rc == 0:
        _run_systemctl('status', '--no-pager')
    else:
        print(f'systemctl start failed (exit {rc}).  '
              f'Is the service installed?  '
              f'Check: systemctl --user status {_SYSTEMD_SERVICE}',
              file=sys.stderr)
        sys.exit(rc)


def cmd_stop(_args):
    """Stop the supervisor via systemd."""
    rc = _run_systemctl('stop')
    if rc == 0:
        print(f'Supervisor stopped.')
    else:
        print(f'systemctl stop failed (exit {rc}).',
              file=sys.stderr)
        sys.exit(rc)


def cmd_restart(_args):
    """Restart the supervisor via systemd (stop + start in one step)."""
    rc = _run_systemctl('restart')
    if rc == 0:
        _run_systemctl('status', '--no-pager')
    else:
        print(f'systemctl restart failed (exit {rc}).  '
              f'Is the service installed?  '
              f'Check: systemctl --user status {_SYSTEMD_SERVICE}',
              file=sys.stderr)
        sys.exit(rc)


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


DISK_SAMPLE_SECONDS = 30
QUEUE_WAL_QUIESCE_BYTES = 2 * 1024 ** 3
TRUNCATE_RETRY_SECONDS = 15


def _disk_guard(queue, queue_path) -> bool:
    """Stop and latch the swarm when disk use reaches its reserved margin."""
    used_fraction = disk_stats(queue_path)['used_fraction']
    if used_fraction < DISK_STOP_FRACTION:
        return False
    logger.critical('Disk %.1f%% full (>= %.0f%% stop threshold) — stopping '
                    'swarm and latching down.  Clear with: '
                    'erd_search.py queue clear-disk-stop',
                    100 * used_fraction, 100 * DISK_STOP_FRACTION)
    try:
        queue.set_disk_stop(f'supervisor: disk {100 * used_fraction:.1f}% full')
    except sqlite3.OperationalError as exc:
        # The startup fullness check keeps the swarm down without a latch row.
        logger.critical('Could not write disk_stop latch: %s', exc)
    return True


def _supervisor_checkpoint(queue):
    """Backfill the queue WAL without taking the writer lock."""
    queue.checkpoint('PASSIVE')


def _maybe_quiesce_truncate(queue):
    """Quiesce queue readers and truncate an oversized WAL."""
    wal_bytes = queue.wal_size_bytes()
    if wal_bytes < QUEUE_WAL_QUIESCE_BYTES:
        return
    logger.info('Queue WAL at %.2f GB — quiescing workers for TRUNCATE.',
                wal_bytes / 1e9)
    queue.set_checkpoint_pause(True)
    try:
        deadline = time.time() + TRUNCATE_RETRY_SECONDS
        while True:
            result = queue.checkpoint('TRUNCATE')
            if result is not None and result[0] == 0:
                logger.info('Queue WAL truncated (%.2f GB reclaimed).',
                            wal_bytes / 1e9)
                return
            if time.time() >= deadline:
                logger.warning('Queue WAL TRUNCATE still busy after %ds '
                               '(wal=%.2f GB); will retry next pass.',
                               TRUNCATE_RETRY_SECONDS,
                               queue.wal_size_bytes() / 1e9)
                return
            time.sleep(0.5)
    finally:
        queue.set_checkpoint_pause(False)


def cmd_run(args):
    _checkpoint_cache_on_start(args.cache)
    # Apply any pending ScoreCache schema migrations single-threaded now, before
    # the worker processes open the cache concurrently — concurrent first-open
    # would race on ALTER TABLE ADD COLUMN ("duplicate column name").
    ScoreCache(args.cache, load_word_list(ANSWER_FILE),
               checkpoint_on_close=False).close()
    queue = ERDQueue(args.queue)
    latch = queue.disk_stop()
    if latch is not None:
        at = (
            datetime.fromtimestamp(latch['at']).isoformat(' ')
            if latch.get('at') else 'unknown time'
        )
        print(f'Refusing to start: disk-stop latch is set ({latch["reason"]}, '
              f'at {at}).\nFree disk space, then clear it with: '
              f'erd_search.py queue clear-disk-stop', file=sys.stderr)
        queue.close()
        return
    used_fraction = disk_stats(args.queue)['used_fraction']
    if used_fraction >= DISK_STOP_FRACTION:
        queue.set_disk_stop(f'startup: disk {100 * used_fraction:.1f}% full')
        print(f'Refusing to start: disk {100 * used_fraction:.1f}% full '
              f'(>= {100 * DISK_STOP_FRACTION:.0f}% stop threshold); '
              f'latching down.', file=sys.stderr)
        queue.close()
        return
    stale = queue.reset_stale_in_progress()
    nb, nc = queue.recover_active_branches()
    if stale or nb or nc:
        print(f'Recovery: {stale} pending rows reset, '
              f'{nb} active branches resumed, {nc} in-flight claims freed.')

    counts = queue.counts_by_status()
    if not counts.get('pending') and not counts.get('in_progress'):
        print('Warning: queue appears empty.  '
              'Run queue add to load branches before starting workers.',
              file=sys.stderr)
    # Telemetry epoch the workers will stamp their rows with.  Epoch 0 is the
    # single-candidate-atom baseline; a claiming-regime change calls set_epoch.
    print(f'Telemetry epoch: {queue.epoch}')
    queue.close()

    _setup_supervisor_logging()
    logger.info('Supervisor starting: %d workers, recycle_hours=%.1f',
                args.workers, args.recycle_hours)

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
    logger.info('Started %d workers (supervisor pid=%d).', args.workers, os.getpid())

    q = ERDQueue(args.queue)
    last_checkpoint = time.time()
    last_disk_sample = 0.0
    while not stop_event.is_set():
        time.sleep(5)
        if stop_event.is_set():
            break

        try:
            if _disk_guard(q, args.queue):
                stop_event.set()
                break

            now = time.time()
            if now - last_disk_sample > DISK_SAMPLE_SECONDS:
                q.record_disk_sample(disk_stats(args.queue)['avail_bytes'])
                last_disk_sample = now

            if now - last_checkpoint > erd_swarm.CHECKPOINT_SECONDS:
                _supervisor_checkpoint(q)
                last_checkpoint = time.time()
            _maybe_quiesce_truncate(q)

            for wid, (p, started_at) in list(procs.items()):
                age = time.time() - started_at
                if not p.is_alive():
                    logger.info('Worker %d exited (age=%.0fs), respawning',
                                wid, age)
                    _reap_worker(q, wid)
                    procs[wid] = _spawn_worker(wid, args, stop_event)
                elif age > args.recycle_hours * 3600:
                    logger.info('Worker %d recycle-hours hit (age=%.0fs), '
                                'terminating and respawning', wid, age)
                    p.terminate()
                    p.join(timeout=10)
                    if p.is_alive():
                        logger.warning('Worker %d did not exit on SIGTERM; '
                                       'killing', wid)
                        p.kill()
                        p.join(timeout=10)
                    _reap_worker(q, wid)
                    procs[wid] = _spawn_worker(wid, args, stop_event)

            # Liveness-gated reclaim never frees work held by a live worker.
            freed = q.reclaim_stale_claims(args.worker_timeout_seconds)
            if freed:
                logger.info('Reclaimed %d stale candidate claim(s).', freed)
            counts = q.counts_by_status()
            in_flight = len(q.branches_in_progress())

            if (counts.get('pending', 0) == 0
                    and counts.get('in_progress', 0) == 0
                    and in_flight == 0
                    and counts):
                logger.info('Queue drained — all branches done.')
                print('\nQueue empty — all branches done.')
                stop_event.set()
        except sqlite3.OperationalError as exc:
            logger.critical('Queue database error in supervisor loop: %s — '
                            'stopping swarm.', exc)
            stop_event.set()

    logger.info('Supervisor stopping all workers...')
    for wid, (p, _) in procs.items():
        if p.is_alive():
            p.terminate()
    for wid, (p, _) in procs.items():
        p.join(timeout=30)
    q.checkpoint()
    q.close()
    logger.info('Supervisor exited.')
    print('All workers stopped.')


def _reap_worker(queue, worker_id: int):
    """Free a dead/killed worker's in-flight candidate claims and clear its
    heartbeat, BEFORE a replacement of the same name starts heartbeating.

    Without this, a respawned worker-N would refresh the 'worker-N' heartbeat,
    making the previous instance's orphaned (done=0) claims look like they're
    held by a live worker — so the liveness-gated reclaim would never free them
    and the affected branches would never finalize.
    """
    name = f'worker-{worker_id}'
    freed = queue.reclaim_claims_of_worker(name)
    queue.clear_heartbeat(name)
    if freed:
        logger.info('Reaped worker %d: freed %d candidate claim(s).',
                    worker_id, freed)


def _spawn_worker(worker_id: int, args, stop_event):
    p = multiprocessing.Process(
        target=erd_swarm.swarm_worker,
        args=(worker_id, args.cache, args.queue, stop_event, args.workers),
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
# reset-stale
# ---------------------------------------------------------------------------

def cmd_reset_stale(args):
    queue = ERDQueue(args.queue)
    n = queue.reset_stale_in_progress()
    queue.close()
    print(f'Reset {n} in_progress row(s) to pending.')


def _normalize_queue_cli_args(args):
    """Apply the queue-level path to nested mutation commands."""
    if args.cmd != 'queue':
        return
    if not hasattr(args, 'queue'):
        args.queue = args.queue_path or DEFAULT_QUEUE


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # -- run --
    p_run = sub.add_parser('run', help='Start the parallel precache supervisor')
    p_run.add_argument('--workers', type=int, default=6, metavar='N',
                       help='Number of swarm worker processes (default: 6)')
    p_run.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_run.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_run.add_argument('--recycle-hours', type=float, default=3.0,
                       metavar='H',
                       help='Respawn each worker after H hours wall time '
                            '(default: 3).  Bounds per-worker ScoreCache '
                            'memory growth while in-progress work is preserved '
                            'in the queue and resumed by the fresh worker.')
    p_run.add_argument('--worker-timeout-seconds', type=int,
                       default=WORKER_LIVENESS_SECONDS,
                       metavar='S',
                       help='Declare a worker dead and reclaim its candidate '
                            'claims after S seconds of missed heartbeats '
                            '(default: 30).  Live workers heartbeat every '
                            '~2s regardless of how long a single candidate '
                            'takes, so only a crashed process triggers this.')

    # -- view --
    p_view = sub.add_parser('view', help='View shared swarm reports')
    p_view.add_argument('--watch', nargs='?', const=30.0,
                        type=_view_watch_interval, metavar='SECONDS')
    p_view.add_argument('--format', choices=('text', 'json', 'jsonl'),
                        default='text')
    p_view.add_argument('--no-color', action='store_true')
    p_view.add_argument('--queue-path', default=DEFAULT_QUEUE, metavar='PATH')
    p_view.add_argument('--cache-path', default=DEFAULT_CACHE, metavar='PATH')
    p_view.add_argument('--claims', action='store_true')
    p_view.add_argument('--answers', action='store_true')
    p_view.add_argument('--tree', action='store_true')
    view_kind = p_view.add_mutually_exclusive_group()
    view_kind.add_argument('--queue', dest='view_queue', action='store_true')
    view_kind.add_argument('--workers', action='store_true')
    view_kind.add_argument('--worker', metavar='N')
    view_kind.add_argument('--cache', dest='view_cache', action='store_true')
    view_kind.add_argument('--hotspots', action='store_true')
    lifecycle_filter = p_view.add_mutually_exclusive_group()
    lifecycle_filter.add_argument('--active-only', action='store_true')
    lifecycle_filter.add_argument(
        '--status', action='append', default=[],
        choices=('pending', 'active', 'finalizing', 'done', 'unqueued'))
    p_view.add_argument('--minimum-answer-count', type=int, metavar='N')
    p_view.add_argument('--maximum-answer-count', type=int, metavar='N')
    p_view.add_argument('--budget', type=int, metavar='N')
    p_view.add_argument('--priority', type=int, metavar='N')
    p_view.add_argument('--sort',
                        choices=('default', 'age', 'size', 'workers',
                                 'priority', 'nodes', 'slowest'))
    p_view.add_argument('--limit', type=int, metavar='N')
    p_view.add_argument(
        '--by', choices=(
            'nodes', 'age', 'size', 'workers', 'priority', 'slowest',
            'evaluated-candidates', 'bulk-completed-candidates',
            'cut-reuse', 'coordination'))
    p_view.add_argument('--epoch', type=int, metavar='N')
    p_view.add_argument('--since-seconds', type=int, metavar='N')
    p_view.add_argument('--sample-size', type=int, metavar='N')
    p_view.add_argument('spine', nargs='*', metavar='SPINE')

    # -- queue --
    p_queue = sub.add_parser('queue', help='Manage queue work')
    p_queue.add_argument('--queue', dest='queue_path', default=None, metavar='PATH')
    qsub = p_queue.add_subparsers(dest='queue_cmd', required=True)

    # -- queue add --
    p_qa = qsub.add_parser('add',
                           help='Add branches for a word (or word list) to the queue')
    qa_word = p_qa.add_mutually_exclusive_group(required=True)
    qa_word.add_argument('--word', metavar='WORD',
                         help='Single guess word (e.g. salet)')
    qa_word.add_argument('--word-list', metavar='FILE',
                         help=f'File of words to add (default list: {WORDS_FILE})')
    p_qa.add_argument('--pattern', metavar='PAT',
                      help='Only add this specific response pattern for --word '
                           '(5 chars: g=green y=yellow -=gray).  '
                           'Omit to add all patterns for the word.')
    p_qa.add_argument('--priority', type=int, default=0, metavar='N',
                      help='Priority for queued branches (default: 0).  '
                           'Higher numbers are worked sooner.')
    p_qa.add_argument('--priority-words', nargs='+', metavar='WORD',
                      help='With --word-list: only these words are queued at '
                           '--priority, the rest at 0 (e.g. '
                           '--priority-words salet crane)')
    p_qa.add_argument('--max-branch-size', type=int, default=None, metavar='N',
                      help='Skip branches with more than N answer words '
                           '(default: unlimited)')
    p_qa.add_argument('--delete-erd-cache', action='store_true',
                      help='Delete any existing ERD cache entry for each '
                           'queued branch first, so it is recomputed instead '
                           'of being skipped as already-cached')
    p_qa.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qa.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- queue clear --
    p_qc = qsub.add_parser('clear',
                            help='Wipe all queue state (does not touch the ERD cache)')
    p_qc.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qc.add_argument('--yes', action='store_true',
                      help='Skip confirmation prompt')

    # -- queue remove --
    p_qr = qsub.add_parser('remove',
                            help='Remove a pending branch from the queue')
    p_qr.add_argument('--word', required=True, metavar='WORD')
    p_qr.add_argument('--pattern', required=True, metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qr.add_argument('--force', action='store_true',
                      help='Also cancel an in-progress branch (clears active '
                           'state so the worker moves on after its current candidate)')
    p_qr.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qr.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- queue priority --
    p_qp = qsub.add_parser('priority',
                            help='Set the priority of a queued branch')
    p_qp.add_argument('--word', required=True, metavar='WORD')
    p_qp.add_argument('--pattern', required=True, metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qp.add_argument('--priority', required=True, type=int, metavar='N',
                      help='New priority (higher = worked sooner; '
                           'use values 0–999 for normal work)')
    p_qp.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qp.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- start --
    sub.add_parser('start',
                   help='Start the supervisor via systemd '
                        '(systemctl --user start wordle-erd)')

    # -- stop --
    sub.add_parser('stop',
                   help='Stop the supervisor via systemd '
                        '(systemctl --user stop wordle-erd)')

    # -- restart --
    sub.add_parser('restart',
                   help='Restart the supervisor via systemd '
                        '(systemctl --user restart wordle-erd)')

    # -- queue reset-stale --
    p_rst = qsub.add_parser('reset-stale',
                             help='Reset in_progress rows to pending')
    p_rst.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    p_cds = qsub.add_parser(
        'clear-disk-stop', help='Release the disk-stop latch so run can start'
    )
    p_cds.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    args = parser.parse_args()
    if args.cmd == 'view' and args.format == 'json' and args.watch is not None:
        parser.error('--format json cannot be used with --watch; use jsonl')
    if args.cmd == 'view':
        try:
            args.selector = parse_report_selector(args.spine)
        except ValueError as error:
            parser.error(str(error))
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be at least 1')
        if (args.minimum_answer_count is not None
                and args.maximum_answer_count is not None
                and args.minimum_answer_count > args.maximum_answer_count):
            parser.error('--minimum-answer-count cannot exceed --maximum-answer-count')
        args.report_kind = (
            'queue' if args.view_queue else
            'workers' if args.workers or args.worker is not None else
            'cache' if args.view_cache else
            'hotspots' if args.hotspots else 'auto'
        )
        if args.by is not None and not args.hotspots:
            parser.error('--by requires --hotspots')
        if not args.hotspots and any(
                value is not None
                for value in (args.epoch, args.since_seconds, args.sample_size)):
            parser.error('--epoch, --since-seconds, and --sample-size require --hotspots')
        if args.since_seconds is not None and args.since_seconds < 1:
            parser.error('--since-seconds must be at least 1')
        if args.sample_size is not None and args.sample_size < 1:
            parser.error('--sample-size must be at least 1')
        hotspot_field = args.by or 'nodes'
        if args.hotspots and args.limit is None:
            args.limit = 10
        args.filters = ReportFilters(
            active_only=args.active_only,
            statuses=tuple(args.status),
            minimum_answer_count=args.minimum_answer_count,
            maximum_answer_count=args.maximum_answer_count,
            budget=args.budget,
            priority=args.priority,
            sort=args.sort,
            limit=args.limit,
        )
        args.hotspot_field = hotspot_field if args.hotspots else None
        args.sample_size = min(args.sample_size or 50_000, 1_000_000)
        args.since_seconds = args.since_seconds or 3600
        try:
            validate_report_request(ReportRequest(
                report_kind=args.report_kind,
                selector=args.selector,
                include_claims=args.claims,
                include_answers=args.answers,
                tree=args.tree,
                filters=args.filters,
                worker_id=args.worker,
                hotspot_field=args.hotspot_field,
                epoch=args.epoch,
                since_seconds=args.since_seconds,
                sample_size=args.sample_size,
            ))
        except ValueError as error:
            parser.error(str(error))
    _normalize_queue_cli_args(args)

    if args.cmd == 'queue':
        qdispatch = {
            'add': cmd_queue_add,
            'clear': cmd_queue_clear,
            'remove': cmd_queue_remove,
            'priority': cmd_queue_priority,
            'reset-stale': cmd_reset_stale,
            'clear-disk-stop': cmd_queue_clear_disk_stop,
        }
        qdispatch[args.queue_cmd](args)
        return

    dispatch = {
        'start': cmd_start,
        'stop': cmd_stop,
        'restart': cmd_restart,
        'run': cmd_run,
        'view': cmd_view,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
