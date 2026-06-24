#!/usr/bin/env python3.13
"""erd_search.py — Parallel ERD_ALL precache CLI.

Subcommands
-----------
start           Start the supervisor via systemd (systemctl --user start).
stop            Stop the supervisor via systemd (systemctl --user stop).
restart         Restart the supervisor via systemd (systemctl --user restart):
                a stop followed by a start in one step.

status          Read-only progress snapshot: queue counts, cache throughput,
                per-worker heartbeats.  --watch loops the display.

run             Start the supervisor directly (without systemd), for
                development or one-shot use.  All output goes to erd_search.log.

queue-add       Add branches for a word or word list to the work queue.
                Idempotent: existing branches are never duplicated; priority
                is upgraded if the new request is higher.  With --word-list,
                --priority-words marks a subset of the list's words as
                higher priority.  --delete-erd-cache forces a recompute of
                branches that are already cached.

queue-clear     Wipe all queue state (pending branches, active state, chunk
                claims, heartbeats).  Does not touch the ERD cache.

queue-inspect   Show queue and worker detail for a specific branch.

queue-remove    Remove a pending branch from the queue.  Use --force to also
                cancel an in-progress branch (workers move on after their
                current chunk completes).

queue-priority  Change the priority of a queued branch.  Higher numbers are
                worked sooner; 0 is the default.

export          Create a trimmed snapshot of the cache for the iPhone
                (answer_list, response_decomposition, branch_best_by_policy).
                Safe while workers are active; re-running is incremental.

cache-status    Show ERD cache coverage for a given word: which response
                patterns are cached and which are missing.

queue-status    Show swarm queue coverage for a given word: which response
                patterns are pending, in progress, done, or not yet queued.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import multiprocessing
import os
import select
from collections import defaultdict
import signal
import sys
import termios
import time
from datetime import datetime

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, ResponseCache, load_word_list
from erd_queue import ERDQueue, encode_subset, guess_depth_from_spine
import erd_swarm

ANSWER_FILE = 'NYT_wordlist.txt'
WORDS_FILE = 'wordle.txt'
DEFAULT_CACHE = 'wordle_cache.sqlite3'
DEFAULT_QUEUE = 'erd_queue.sqlite3'

logger = logging.getLogger('wordle')


# ---------------------------------------------------------------------------
# cache-status
# ---------------------------------------------------------------------------

def cmd_cache_status(args):
    """Show ERD cache coverage for a given word.

    For each of the 242 possible response patterns for WORD, checks whether
    the branch has a cached ERD value.  Reports the cached best guess and
    score for hits, and flags misses so you can see what still needs work.
    """
    from wordle_ui import fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    word = args.word.strip().lower()

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    groups = rcache.group_words(word, all_answers)
    sc.close()

    # Re-open with full ScoreCache to read branch entries.
    sc = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)

    total = len(groups)
    cached_count = 0
    missing = []
    hits = []

    for code, branch in sorted(groups.items()):
        pat = fmt_pattern(code)
        n = len(branch)
        if n < 2:
            # Singleton or zero: no ERD needed (answer is identified immediately).
            continue
        branch_key = ScoreCache.encode_subset(branch)
        entry = sc.read_detail(branch_key, ERD_ALL)
        if entry is None:
            missing.append((pat, n))
        else:
            best_guess, best_score, updated_at = entry
            cached_count += 1
            hits.append((pat, n, best_guess, best_score, updated_at))

    sc.close()

    n_trivial = sum(1 for branch in groups.values() if len(branch) < 2)
    n_branches = total - n_trivial

    print(f'{word.upper()}:  {n_branches} branches with ≥2 answers  '
          f'({n_trivial} trivial patterns skipped)')
    print(f'  Cached : {cached_count:4d}')
    print(f'  Missing: {len(missing):4d}')
    print()

    if hits and not args.missing_only:
        print(f'{"Pattern":<8}  {"Ans":>4}  {"Best guess":<12}  {"ERD":>7}  Updated')
        for pat, n, best_guess, best_score, updated_at in hits:
            from datetime import datetime
            dt = datetime.fromtimestamp(updated_at).strftime('%Y-%m-%d %H:%M')
            print(f'  {pat:<8}  {n:4d}  {best_guess.upper():<12}  '
                  f'{best_score:7.4f}  {dt}')
        if missing:
            print()

    if missing:
        print(f'{"Pattern":<8}  {"Ans":>4}  (missing)')
        for pat, n in missing:
            print(f'  {pat:<8}  {n:4d}')


# ---------------------------------------------------------------------------
# queue-status
# ---------------------------------------------------------------------------

def cmd_queue_status(args):
    """Show swarm queue coverage for a given word.

    For each of the 242 possible response patterns for WORD, reports the
    branch's status in the queue (pending_branches): pending, in_progress,
    done, or not yet queued at all.
    """
    from wordle_ui import fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    word = args.word.strip().lower()

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    groups = rcache.group_words(word, all_answers)
    sc.close()

    branches = {code: branch for code, branch in groups.items() if len(branch) >= 2}
    branch_keys = {code: encode_subset(branch) for code, branch in branches.items()}

    queue = ERDQueue(args.queue)
    pending_rows = queue.status_by_branch_keys(list(branch_keys.values()))
    # Cooperative sub-branches exist only in active_branches (no pending_branches
    # row): check for any keys not found in pending_branches.
    missing = [bk for bk in branch_keys.values() if bk not in pending_rows]
    active_rows = queue.active_branches_by_keys(missing)
    queue.close()

    pending, in_progress, done, unqueued = [], [], [], []
    for code, branch in sorted(branches.items()):
        pat = fmt_pattern(code)
        n = len(branch)
        bk = branch_keys[code]
        row = pending_rows.get(bk)
        active = active_rows.get(bk)
        if row is None and active is None:
            unqueued.append((pat, n))
        elif row is None:
            in_progress.append((pat, n, active['priority'] or 0, None))
        elif row['status'] == 'pending':
            pending.append((pat, n, row['priority']))
        elif row['status'] == 'in_progress':
            in_progress.append((pat, n, row['priority'], row['claimed_by']))
        else:
            done.append((pat, n))

    n_trivial = len(groups) - len(branches)

    print(f'{word.upper()}:  {len(branches)} branches with ≥2 answers  '
          f'({n_trivial} trivial patterns skipped)')
    print(f'  Pending    : {len(pending):4d}')
    print(f'  In progress: {len(in_progress):4d}')
    print(f'  Done       : {len(done):4d}')
    print(f'  Not queued : {len(unqueued):4d}')
    print()

    if in_progress:
        _COOP_PRI = 1_000_000
        print(f'{"Pattern":<8}  {"Ans":>4}  {"Pri":>4}  Claimed by')
        for pat, n, priority, claimed_by in in_progress:
            pri_str = 'COOP' if priority >= _COOP_PRI else str(priority)
            print(f'  {pat:<8}  {n:4d}  {pri_str:>4}  {claimed_by or "---"}')
        print()

    if pending:
        print(f'{"Pattern":<8}  {"Ans":>4}  {"Pri":>4}')
        for pat, n, priority in pending:
            print(f'  {pat:<8}  {n:4d}  {priority:4d}')
        print()

    if unqueued and not args.queued_only:
        print(f'{"Pattern":<8}  {"Ans":>4}  (not queued)')
        for pat, n in unqueued:
            print(f'  {pat:<8}  {n:4d}')


# ---------------------------------------------------------------------------
# queue-add
# ---------------------------------------------------------------------------

def cmd_queue_add(args):
    """Add branches for one word (or a word-list file) to the queue.

    With --word: adds all response branches for that word whose answer-word
    count is between 2 and --max-branch-size.  With --pattern as well: adds
    only that single branch.

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
                if len(branch) > args.max_branch_size:
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
                    if 2 <= len(branch) <= args.max_branch_size
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
# queue-clear
# ---------------------------------------------------------------------------

def cmd_queue_clear(args):
    """Wipe all queue state (pending branches, active branches, chunk claims,
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
# queue-inspect
# ---------------------------------------------------------------------------

def cmd_queue_inspect(args):
    """Show the queue entry for a specific branch (word + pattern)."""
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
        print(f'{word.upper()} {pat}: no branch (that pattern yields 0 answers).')
        return

    branch_key = encode_subset(branch)
    queue = ERDQueue(args.queue)

    pb = queue.get_pending_branch(branch_key)
    ab = queue.get_active_branch(branch_key)
    chunks = queue.claims_for_branch(branch_key) if ab else []
    queue.close()

    print(f'{word.upper()} {pat}  ({len(branch)} answer words)')
    print(f'  branch_key: {branch_key[:20].hex()}...')

    if pb is None:
        print('  Not in queue.')
    else:
        print(f'  Status   : {pb["status"]}')
        print(f'  Priority : {pb["priority"]}  '
              f'(higher number = worked sooner)')

    if ab:
        n_candidates = ab['n_candidates']
        done_ct = sum(1 for c in chunks if c['done'])
        best_g = ab['best_guess'] or '---'
        best_e = f'{ab["best_erd"]:.4f}' if ab['best_erd'] is not None else '---'
        print(f'  In-progress: claims {done_ct}/{n_candidates}  '
              f'best {best_g.upper()} {best_e}')
        print(f'  Claim detail:')
        for c in chunks:
            holder = c['claimed_by'] or '(unclaimed)'
            status = 'done' if c['done'] else f'held by {holder}'
            print(f'    claim {c["idx"]:5d}: {status}')


# ---------------------------------------------------------------------------
# queue-remove
# ---------------------------------------------------------------------------

def cmd_queue_remove(args):
    """Remove a branch from the pending queue.

    Only removes branches with status='pending'.  If the branch is currently
    in-progress (being worked by a worker), use --force to also cancel it by
    clearing its active_branches and chunk rows so the worker's next heartbeat
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
        # Atomically clear chunk claims, the active_branches row, and the
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
# queue-priority
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


def cmd_run(args):
    _checkpoint_cache_on_start(args.cache)
    # Apply any pending ScoreCache schema migrations single-threaded now, before
    # the worker processes open the cache concurrently — concurrent first-open
    # would race on ALTER TABLE ADD COLUMN ("duplicate column name").
    ScoreCache(args.cache, load_word_list(ANSWER_FILE),
               checkpoint_on_close=False).close()
    queue = ERDQueue(args.queue)
    stale = queue.reset_stale_in_progress()
    nb, nc = queue.reset_active_branches()
    if stale or nb or nc:
        print(f'Recovery: {stale} pending rows reset, '
              f'{nb} in-progress branches / {nc} candidate claims cleared.')

    counts = queue.counts_by_status()
    if not counts.get('pending') and not counts.get('in_progress'):
        print('Warning: queue appears empty.  '
              'Run queue-add to load branches before starting workers.',
              file=sys.stderr)
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
    while not stop_event.is_set():
        time.sleep(5)
        if stop_event.is_set():
            break

        for wid, (p, started_at) in list(procs.items()):
            age = time.time() - started_at
            if not p.is_alive():
                logger.info('Worker %d exited (age=%.0fs), respawning', wid, age)
                _reap_worker(q, wid)
                procs[wid] = _spawn_worker(wid, args, stop_event)
            elif age > args.recycle_hours * 3600:
                logger.info('Worker %d recycle-hours hit (age=%.0fs), '
                            'terminating and respawning', wid, age)
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning('Worker %d did not exit on SIGTERM; killing',
                                   wid)
                    p.kill()
                    p.join(timeout=10)
                _reap_worker(q, wid)
                procs[wid] = _spawn_worker(wid, args, stop_event)

        # Backstop: free chunks held by any worker that died WITHOUT being
        # reaped above (e.g. it crashed and we haven't noticed yet).  Gated on
        # heartbeat liveness, so a slow-but-alive worker is never reclaimed.
        freed = q.reclaim_stale_claims(args.worker_timeout_seconds)
        if freed:
            logger.info('Reclaimed %d stale candidate claim(s).', freed)
        counts = q.counts_by_status()
        in_flight = len(q.branches_in_progress())

        # Done when the queue holds no pending or in-progress branches and no
        # branch is still being swarmed.
        if (counts.get('pending', 0) == 0
                and counts.get('in_progress', 0) == 0
                and in_flight == 0
                and counts):
            logger.info('Queue drained — all branches done.')
            print('\nQueue empty — all branches done.')
            stop_event.set()

    q.close()
    logger.info('Supervisor stopping all workers...')
    for wid, (p, _) in procs.items():
        if p.is_alive():
            p.terminate()
    for wid, (p, _) in procs.items():
        p.join(timeout=30)
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
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    if args.watch is not None:
        interval = args.watch if args.watch > 0 else 30
        if sys.stdin.isatty():
            _watch_with_keys(args, interval)
        else:
            try:
                sys.stdout.write('\033[?25l\033[2J\033[H')
                sys.stdout.flush()
                _redraw_status.prev_sections = []
                while True:
                    _redraw_status(args)
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass
            finally:
                sys.stdout.write('\033[?25h')
                sys.stdout.flush()
    else:
        _print_status(args, selected_worker=args.worker,
                      selected_branch=args.branch)


def _watch_with_keys(args, interval):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    # Disable canonical mode and echo without touching output processing (OPOST).
    # tty.setraw() also clears OPOST, which breaks \n → \r\n translation.
    new_settings[3] &= ~(termios.ICANON | termios.ECHO)
    selected_worker = None
    selected_branch = None
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        sys.stdout.write('\033[?25l\033[2J\033[H')
        sys.stdout.flush()
        _redraw_status.prev_sections = []
        while True:
            _redraw_status(args, selected_worker=selected_worker,
                           selected_branch=selected_branch)
            deadline = time.monotonic() + interval
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.2))
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ('q', 'Q', '\x04'):  # q, Ctrl-D
                        return
                    if ch == ' ':
                        break  # force refresh now
                    if ch.isdigit():
                        w = int(ch)
                        selected_worker = None if selected_worker == w else w
                        _redraw_status.prev_sections = []  # force full redraw
                        break
                    # A branch hotkey letter drills into that branch (by its
                    # stable id, so the selection survives position shifts).
                    hotkeys = getattr(_print_status, '_branch_hotkeys', {})
                    if ch in hotkeys:
                        bid = hotkeys[ch]
                        selected_branch = None if selected_branch == bid else bid
                        _redraw_status.prev_sections = []
                        break
    except KeyboardInterrupt:
        # ISIG stays enabled (only ICANON/ECHO are off), so Ctrl-C raises here
        # rather than arriving as a '\x03' byte from stdin.read().
        pass
    finally:
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _branch_id(branch_key):
    """Stable 4-hex-char label for a branch, derived from its key.

    Same branch_key always yields the same id, so a branch keeps its identity
    across refreshes even as its display position (and hotkey letter) shifts.
    Collisions among the few dozen simultaneously-active branches are vanishingly
    unlikely over a 16-bit space.
    """
    return hashlib.sha1(bytes(branch_key)).hexdigest()[:4]


# Hotkey letters offered for branch drill-down, in display order.  'q' is the
# quit key and is skipped; digits stay reserved for worker selection.
_BRANCH_HOTKEYS = 'abcdefghijklmnoprstuvwxyz'


def _fmt_spine_path(spine):
    """Render a stored branch spine ('GUESS pattern GUESS pattern ...') as a
    human path 'GUESS pattern ▸ GUESS pattern'.  Empty/None yields ''."""
    if not spine:
        return ''
    toks = spine.split()
    guesses = [' '.join(toks[i:i + 2]) for i in range(0, len(toks), 2)]
    return ' ▸ '.join(guesses)


def _compact_spine_path(spine, source_word, source_pattern, fmt_pattern,
                        guess_depth, width=40):
    """One-line spine for the overview row, tail-truncated to `width`.

    With a stored spine, shows the deepest `width` characters of the full path
    (the most recent guesses — where the branch actually sits), prefixed '…' when
    truncated.  Without one (a row predating spine capture), falls back to the
    source word plus a '▸ ?×N' marker for the N unrecorded guesses.
    """
    full = _fmt_spine_path(spine)
    if not full:
        source = (f'{source_word.upper()} {fmt_pattern(source_pattern)}'
                  if source_word and source_pattern is not None else '-----')
        return f'{source} ▸ ?×{guess_depth}' if guess_depth else source
    if len(full) <= width:
        return full
    return '…' + full[-(width - 1):]


def _spine_sizes(path):
    """Extract the branch-size portion from each level of a rich spine string.

    Rich format: 'GUESS:pattern/size→GUESS:pattern/size'.  Returns the size-
    only string ('size→size') used in the compact status row.
    """
    if not path:
        return ''
    parts = []
    for tok in path.split('→'):
        if '/' in tok:
            parts.append(tok.rsplit('/', 1)[1])
        else:
            parts.append(tok)
    return '→'.join(parts)


def _parse_spine(path):
    """Parse a rich spine string into a list of (guess, pattern, size) tuples.

    Each token in the '→'-separated path is either a bare size (root level,
    no guess/pattern yet) or 'GUESS:pattern/size'.  Returns one tuple per
    level in depth order.
    """
    if not path:
        return []
    result = []
    for tok in path.split('→'):
        if '/' in tok:
            gp, size_str = tok.rsplit('/', 1)
            if ':' in gp:
                guess, pattern = gp.split(':', 1)
            else:
                guess, pattern = None, gp
            result.append((guess, pattern, size_str))
        elif tok:
            result.append((None, None, tok))
    return result


# Sentinel prefix for section-boundary markers emitted by _print_status in
# interactive mode.  A marker line is '<prefix><section name>'.  The NUL byte
# guarantees the prefix never collides with real status output.
_SECTION_MARK = '\x00SECTION:'


def _section_break(name, interactive):
    """Emit a named section marker so _redraw_status can diff sections independently.

    Change detection runs per section, matched by name across refreshes, so a
    section that grows or shrinks (e.g. a branch added) repaints only its own
    rows instead of shifting later sections into a spurious all-changed diff.
    The marker is emitted only in interactive mode and is stripped before
    display.
    """
    if interactive:
        print(f'{_SECTION_MARK}{name}')


def _split_sections(lines):
    """Split captured status lines into (name, section_lines) pairs on markers.

    Lines preceding the first marker form a leading section keyed ''.  Marker
    lines are consumed as boundaries and do not appear in any section.
    """
    sections = []
    name = ''
    current = []
    for line in lines:
        if line.startswith(_SECTION_MARK):
            sections.append((name, current))
            name = line[len(_SECTION_MARK):]
            current = []
        else:
            current.append(line)
    sections.append((name, current))
    return sections


def _highlight_changes(new_line, old_line):
    """Return new_line with runs of changed characters highlighted in bold red."""
    if new_line == old_line:
        return new_line
    result = []
    in_change = False
    for i, ch in enumerate(new_line):
        changed = i >= len(old_line) or ch != old_line[i]
        if changed and not in_change:
            result.append('\033[1;31m')
            in_change = True
        elif not changed and in_change:
            result.append('\033[0m')
            in_change = False
        result.append(ch)
    if in_change:
        result.append('\033[0m')
    return ''.join(result)


def _redraw_status(args, selected_worker=None, selected_branch=None):
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        _print_status(args, selected_worker=selected_worker,
                      selected_branch=selected_branch, interactive=True)
    finally:
        sys.stdout = old_stdout
    new_sections = _split_sections(buf.getvalue().splitlines())
    prev_sections = dict(getattr(_redraw_status, 'prev_sections', []))
    _redraw_status.prev_sections = new_sections

    rendered = []
    for name, lines in new_sections:
        prev = prev_sections.get(name)
        for i, line in enumerate(lines):
            old_line = prev[i] if prev is not None and i < len(prev) else None
            if old_line is not None and line != old_line:
                rendered.append(_highlight_changes(line, old_line) + '\033[K')
            else:
                rendered.append(line + '\033[K')

    # \033[H  — cursor to top-left (no screen erase, so no flash)
    # \033[J  — erase from last line to end of screen (removes stale lines if output shrank)
    out = '\033[H' + '\n'.join(rendered) + '\033[J'
    sys.stdout.write(out)
    sys.stdout.flush()


def _print_status(args, selected_worker=None, selected_branch=None,
                  interactive=False):
    now_ts = int(time.time())
    from wordle_ui import fmt_pattern

    _section_break('header', interactive)

    # Queue + swarm state
    try:
        queue = ERDQueue(args.queue)
        counts = queue.counts_by_status()
        branches = queue.branches_in_progress()
        hbs = queue.heartbeats_with_branch()
        worker_counts = queue.worker_counts_by_branch()
        # Per-branch done-chunk counts for the in-progress branches.
        done_chunks = {bytes(b['branch_key']): queue.branch_done_candidates(b['branch_key'])
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

    # Cache throughput
    try:
        all_answers = load_word_list(ANSWER_FILE)
        sc = ScoreCache(args.cache, all_answers, checkpoint_on_close=False)
        total_erd = sc._conn.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy "
            "WHERE policy=? AND answer_list_id=?",
            (ERD_ALL, sc.answer_list_id)).fetchone()[0]
        recent = sc._conn.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy "
            "WHERE policy=? AND answer_list_id=? AND updated_at>?",
            (ERD_ALL, sc.answer_list_id, now_ts - 300)).fetchone()[0]
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
    n_cutoff = sum(h['n_cutoff'] or 0 for h in live)
    n_pruned = sum(h['n_pruned'] or 0 for h in live)
    total_evals = n_ok + n_cutoff + n_pruned
    hit_total = hits + misses
    hit_pct = (100.0 * hits / hit_total) if hit_total else None
    cutoff_pct = (100.0 * n_cutoff / total_evals) if total_evals else None
    pruned_pct = (100.0 * n_pruned / total_evals) if total_evals else None

    def _abbrev(n):
        if n >= 1_000_000:
            return f'{n/1_000_000:.1f}M'
        if n >= 1_000:
            return f'{n/1_000:.1f}k'
        return str(n)

    def _fmt_pct(pct):
        if pct >= 99.95:
            return f'{pct:.3f}%'
        if pct >= 99.5:
            return f'{pct:.2f}%'
        return f'{pct:.1f}%'

    print(f'ERD_ALL Precache — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cache_line = f'Cache: {total_erd:,}  +{recent:,}/5m' if cache_ok else None
    if cache_line and hit_pct is not None:
        cache_line += f'  hits {_fmt_pct(hit_pct)} ({_abbrev(hits)}/{_abbrev(hit_total)})'
    if cache_line:
        print(cache_line)
    print()

    _section_break('branches', interactive)

    if not hasattr(_print_status, '_answer_set'):
        _print_status._answer_set = set(load_word_list(ANSWER_FILE))
    answer_set = _print_status._answer_set

    # Branches in progress — the real progress unit.  Each branch carries a
    # stable short #id (durable across refreshes) and a positional hotkey letter
    # for drill-down; the workers cooperating on it are listed beneath it,
    # matched by current_branch_key, so cooperation is spatial rather than a
    # bare count.
    _COOP_PRIORITY = 1_000_000   # sentinel: cooperative sub-branch, not user-queued
    branch_hdr = 'Branches:'
    if queue_ok:
        n_in_prog = counts.get('in_progress', 0)
        n_pending = counts.get('pending', 0)
        n_done = counts.get('done', 0)
        # User-queued in-progress lives in pending_branches (counts['in_progress']);
        # cooperative sub-branches have no pending_branches row and live only in
        # active_branches.  They are independent quantities, so report them side by
        # side rather than nesting coop inside a user-queued count it isn't part of.
        n_coop = sum(1 for b in branches if (b['priority'] or 0) >= _COOP_PRIORITY)
        parts = [f'done {n_done:,}', f'user {n_in_prog:,}',
                 f'coop {n_coop:,}', f'pending {n_pending:,}']
        branch_hdr += ' ' + '  '.join(parts)
    print(branch_hdr)
    if live and total_evals:
        stat_parts = []
        if cutoff_pct is not None:
            stat_parts.append(
                f'ERD {_abbrev(n_cutoff)}/{_abbrev(total_evals)} ({_fmt_pct(cutoff_pct)})')
        if pruned_pct is not None:
            stat_parts.append(
                f'depth {_abbrev(n_pruned)}/{_abbrev(total_evals)} ({_fmt_pct(pruned_pct)})')
        if stat_parts:
            print('pruned: ' + '  '.join(stat_parts))
    print()

    # guess_depth = guesses already played to reach a branch = the number of
    # guesses on its absolute spine.  A seed with no full spine yet but a recorded
    # source word is one guess deep (the opener); the bare root (no guess) is 0.
    def _branch_guess_depth(b):
        spine = b['spine'] if 'spine' in b.keys() else None
        if spine:
            return guess_depth_from_spine(spine)
        return 1 if (b['source_word'] and b['source_pattern'] is not None) else 0

    branch_guess_depth = {bytes(b['branch_key']): _branch_guess_depth(b)
                          for b in branches}

    # Group live heartbeats under the branch each worker is contributing to.
    workers_by_branch = defaultdict(list)
    idle_workers = []
    for h in hbs:
        k = h['current_branch_key']
        if k is None:
            idle_workers.append(h)
        else:
            workers_by_branch[bytes(k)].append(h)

    def _worker_num(h):
        wid = h['worker_id']
        return wid.split('-')[-1] if '-' in wid else wid

    def _print_worker_row(h):
        # One nested row, kept within 59 cols: worker, sweep index, current
        # candidate, deepest guess_depth reached, node rate, then the descent
        # sizes truncated to whatever width remains.
        age = now_ts - h['updated_at']
        flag = ' !!' if age > 120 else ''
        chunk = str(h['claim_idx']) if h['claim_idx'] is not None else '-'
        cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
        mdepth = (h['cur_max_depth'] if 'cur_max_depth' in h.keys() else None) or 0
        nodes = (h['cur_nodes'] if 'cur_nodes' in h.keys() else None) or 0
        nrate = (h['node_rate'] if 'node_rate' in h.keys() else None) or 0.0
        path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
        if '>' in path:
            path = path.replace('>', '→')
        # Forward-progress flag: heartbeat fresh but evaluation rate is zero == hang.
        if age <= 10 and nrate == 0 and nodes:
            flag += ' ~?'
        cur_disp = (cur.upper() + ('*' if cur.lower() in answer_set else ' ')) if cur else '-----'
        krate = f'{int(nrate / 1000)}k/s' if nrate else ''
        head = (f' W{_worker_num(h):<2} {chunk:>5} {cur_disp:<6} '
                f'd{mdepth} {krate:>6}')
        tail = f' {age}s{flag}'
        sizes = _spine_sizes(path)
        room = 59 - len(head) - len(tail) - 1
        if sizes and room > 1:
            if len(sizes) > room:
                sizes = sizes[:room - 1] + '…'
            print(f'{head} {sizes}{tail}')
        else:
            print(f'{head}{tail}')

    # Show only branches a worker is currently on, so the screen stays within a
    # phone's height: the worker count (<= 6 here) caps the number of branch
    # blocks.  No-worker branches stay accounted for in the "Branches:" counts.
    active = [b for b in branches
              if workers_by_branch.get(bytes(b['branch_key']))]
    if not active:
        print('(no branches being worked)')
    # Stable per-branch hotkey letters: a branch keeps its letter across refreshes,
    # and a selected branch keeps it even after it finalizes, so its detail panel's
    # "press X to dismiss" stays valid and letters don't shuffle under the user.
    prev_letter_by_bid = getattr(_print_status, '_branch_letter_by_bid', {})
    keep_bids = [_branch_id(bytes(b['branch_key'])) for b in active]
    if selected_branch is not None and selected_branch not in keep_bids:
        keep_bids.append(selected_branch)
    branch_hotkeys = {}      # letter -> bid, consumed by the interactive input loop
    letter_by_bid = {}       # bid -> letter, persisted across refreshes for stability
    used = set()
    for bid in keep_bids:
        lt = prev_letter_by_bid.get(bid)
        if lt and lt not in used:
            letter_by_bid[bid] = lt
            branch_hotkeys[lt] = bid
            used.add(lt)
    free_letters = (c for c in _BRANCH_HOTKEYS if c not in used)
    for bid in keep_bids:
        if bid not in letter_by_bid:
            lt = next(free_letters, ' ')
            if lt != ' ':
                letter_by_bid[bid] = lt
                branch_hotkeys[lt] = bid
                used.add(lt)
    for b in active:
        key = bytes(b['branch_key'])
        bid = _branch_id(key)
        letter = letter_by_bid.get(bid, ' ')
        n_cands = b['n_candidates'] or 0
        done = done_chunks.get(key, 0)
        pct = int(100.0 * done / n_cands) if n_cands else 0
        bw = (b['best_guess'] or '-----').upper()
        bstar = '*' if (b['best_guess'] or '').lower() in answer_set else ' '
        be = f'{b["best_erd"]:.3f}' if b['best_erd'] is not None else '-----'
        nw = b['n_words'] or 0
        wk = worker_counts.get(key, 0)
        guess_depth = branch_guess_depth.get(key, 0)
        created = b['created_at'] or now_ts
        el = now_ts - created
        eta = '-'
        if wk > 0 and 0 < done < n_cands and el > 0:
            rem = (n_cands - done) / (done / el)
            eta = _fmt_duration(int(rem))
        # Line 1: branch stats.  Line 2: the spine (guesses played), truncated.
        print(f'{letter} #{bid} {nw:>3}w d{guess_depth} {pct:>3}% '
              f'{bw}{bstar} {be:>5} {wk}wk {eta:>6}')
        spine = b['spine'] if 'spine' in b.keys() else None
        spine_disp = _compact_spine_path(
            spine, b['source_word'], b['source_pattern'], fmt_pattern,
            guess_depth, width=57)
        print(spine_disp)
        for h in sorted(workers_by_branch.get(key, []),
                        key=lambda r: int(_worker_num(r)) if _worker_num(r).isdigit()
                        else 0):
            _print_worker_row(h)

    # Workers attached to a branch no longer in the in-progress list (just
    # finalized) or idle between claims.
    branch_keys = {bytes(b['branch_key']) for b in branches}
    detached = [h for k, hs in workers_by_branch.items() if k not in branch_keys
                for h in hs]
    if idle_workers or detached:
        print()
        for h in sorted(idle_workers, key=lambda r: _worker_num(r)):
            age = now_ts - h['updated_at']
            flag = ' !!' if age > 120 else ''
            print(f'W{_worker_num(h):<2} (idle)  {age}s{flag}')
        for h in sorted(detached, key=lambda r: _worker_num(r)):
            age = now_ts - h['updated_at']
            print(f'W{_worker_num(h):<2} (branch finalizing)  {age}s')

    _print_status._branch_hotkeys = branch_hotkeys
    _print_status._branch_letter_by_bid = letter_by_bid

    if selected_worker is not None:
        _section_break('detail', interactive)
        target = str(selected_worker)
        detail_hb = next(
            (h for h in hbs
             if (h['worker_id'].split('-')[-1]
                 if '-' in h['worker_id'] else h['worker_id']) == target),
            None)
        print()
        if detail_hb is None:
            print(f'Worker {selected_worker} not found')
        else:
            bkey = bytes(detail_hb['current_branch_key']) if detail_hb['current_branch_key'] else None
            base_k = branch_guess_depth.get(bkey, 1) if bkey else 1
            print(f'Worker {selected_worker}: branch '
                  f'#{_branch_id(bkey) if bkey else "?"}, depth {base_k}')
            # Path to the worker's branch: its absolute spine, one guess per line
            # (d1..dK).  The first guess is d1; the bare root is d0 with no guess.
            branch_info = next((b for b in branches
                                if bkey and bytes(b['branch_key']) == bkey), None)
            branch_spine = (branch_info['spine']
                            if branch_info and 'spine' in branch_info.keys() else None)
            full = _fmt_spine_path(branch_spine)
            if full:
                for di, guess_step in enumerate(full.split(' ▸ '), start=1):
                    parts = guess_step.split()
                    word = parts[0] if parts else ''
                    rest = ' '.join(parts[1:])
                    star = '*' if word.lower() in answer_set else ' '
                    print(f'{f"d{di}":<4} {word}{star} {rest}')
            elif detail_hb['source_word']:
                star = '*' if detail_hb['source_word'].lower() in answer_set else ' '
                pat = (fmt_pattern(detail_hb['source_pattern'])
                       if detail_hb['source_pattern'] is not None else '-----')
                print(f'{"d1":<4} {detail_hb["source_word"].upper()}{star} {pat}')
            rich_path = (detail_hb['cur_path'] if 'cur_path' in detail_hb.keys() else None) or ''
            cur_cand = (detail_hb['cur_candidate'] if 'cur_candidate' in detail_hb.keys() else None) or ''
            chunk_held = (detail_hb['claim_idx'] if 'claim_idx' in detail_hb.keys() else None) is not None
            # The worker's live descent below its branch (dK+1..).  The first
            # cur_path level is the claimed branch itself (size only) — skipped
            # by filtering to levels that carry a guess.
            descent = [(g, p, s) for (g, p, s) in _parse_spine(rich_path) if g and p]
            if descent:
                for di, (guess, pattern, size) in enumerate(descent, start=base_k + 1):
                    star = '*' if guess.lower() in answer_set else ' '
                    print(f'{f"d{di}":<4} {guess.upper()}{star} {pattern} {size:>4}w')
            elif cur_cand:
                # The candidate under evaluation is the first guess of the descent
                # (d{base_k+1}): render it as the spine line it is starting, not as
                # a separate "no spine yet" message.
                star = '*' if cur_cand.lower() in answer_set else ' '
                print(f'{f"d{base_k + 1}":<4} {cur_cand.upper()}{star} (evaluating)')
            elif not chunk_held:
                bkey = bytes(detail_hb['current_branch_key']) if detail_hb['current_branch_key'] else None
                w_guess_depth = branch_guess_depth.get(bkey, 0) if bkey else 0
                if w_guess_depth > 1 and bkey:
                    branch_info = next((b for b in branches
                                        if bytes(b['branch_key']) == bkey), None)
                    if branch_info:
                        n_cands = branch_info['n_candidates'] or 0
                        total_chunks = n_cands
                        done_ct = done_chunks.get(bkey, 0)
                        co_workers = [
                            h['worker_id'].split('-')[-1]
                            if '-' in h['worker_id'] else h['worker_id']
                            for h in hbs
                            if h['current_branch_key']
                            and bytes(h['current_branch_key']) == bkey
                            and h['worker_id'] != detail_hb['worker_id']
                        ]
                        co_str = (f', W{",".join(co_workers)} also active'
                                  if co_workers else '')
                        print(f'(cooperating — {done_ct}/{total_chunks} chunks done{co_str})')
                    else:
                        print('(cooperating — sub-branch finalizing)')
                else:
                    print('(between chunks)')
            else:
                print('(no spine data yet)')
            if interactive:
                print(f'[press {selected_worker} to dismiss]')

    if selected_branch is not None:
        _section_break('branchdetail', interactive)
        target = next((b for b in branches
                       if _branch_id(b['branch_key']) == selected_branch), None)
        print()
        if target is None:
            print(f'Branch #{selected_branch} not found')
            if interactive:
                # The branch's letter is held reserved while it stays selected, so
                # re-pressing it still toggles the (now finalized) panel closed.
                letter = next((lt for lt, bid
                               in getattr(_print_status, '_branch_hotkeys', {}).items()
                               if bid == selected_branch), None)
                if letter:
                    print(f'[press {letter} to dismiss]')
        else:
            key = bytes(target['branch_key'])
            n_cands = target['n_candidates'] or 0
            done = done_chunks.get(key, 0)
            guess_depth = branch_guess_depth.get(key, 0)
            print(f'Branch #{selected_branch}: {target["n_words"] or 0} words: '
                  f'depth {guess_depth}')
            # Full spine, one guess per line: the first guess is d1 (the root,
            # before any guess, is guess_depth 0 and has no guess to show).
            spine = target['spine'] if 'spine' in target.keys() else None
            full = _fmt_spine_path(spine)
            if full:
                for di, guess_step in enumerate(full.split(' ▸ '), start=1):
                    parts = guess_step.split()
                    word = parts[0] if parts else ''
                    rest = ' '.join(parts[1:])
                    star = '*' if word.lower() in answer_set else ' '
                    print(f'{f"d{di}":<4} {word}{star} {rest}')
            else:
                src = (f'{target["source_word"].upper()} '
                       f'{fmt_pattern(target["source_pattern"])}'
                       if target['source_word'] and target['source_pattern'] is not None
                       else '?????')
                print(f'{"d1":<4} {src}')
                if guess_depth > 1:
                    print(f'd2..d{guess_depth}  (intermediate guesses not recorded)')
            # Candidate sweep with each cooperating worker's position marked.
            best_disp = (f'{(target["best_guess"] or "-----").upper()} '
                         f'{target["best_erd"]:.3f}'
                         if target['best_erd'] is not None else 'none yet')
            # The sweep searches the next guess (one past the branch's spine), so
            # label it at that absolute depth to read as a continuation of d1..dK.
            print(f'{f"d{guess_depth + 1}":<4} sweep {done}/{n_cands}  '
                  f'best {best_disp}')
            workers_here = sorted(
                [h for h in hbs if h['current_branch_key']
                 and bytes(h['current_branch_key']) == key],
                key=lambda r: (r['claim_idx'] if r['claim_idx'] is not None else -1))
            if workers_here and n_cands:
                bar_w = 40
                marks = [' '] * bar_w
                for h in workers_here:
                    idx = h['claim_idx']
                    if idx is not None:
                        pos = min(bar_w - 1, int(bar_w * idx / n_cands))
                        # Workers on adjacent candidates can map to the same cell;
                        # nudge right to the next free one so both digits stay
                        # visible.  The bar is approximate, so showing every worker
                        # matters more than the exact cell.
                        while pos < bar_w and marks[pos] != ' ':
                            pos += 1
                        if pos >= bar_w:
                            pos = bar_w - 1
                        marks[pos] = _worker_num(h)[-1]
                print(f'[{"".join(marks)}] (worker by pos)')
            # Each cooperating worker's downward exploration spine, stacked.
            for h in workers_here:
                path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
                path = path.replace('>', '→')
                cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
                cur_disp = cur.upper() if cur else '-----'
                print(f'W{_worker_num(h):<2} idx {h["claim_idx"]}  {cur_disp:<6} '
                      f'{_spine_sizes(path)}')
            if interactive:
                letter = next((lt for lt, bid
                               in getattr(_print_status, '_branch_hotkeys', {}).items()
                               if bid == selected_branch), None)
                if letter:
                    print(f'[press {letter} to dismiss]')


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

EXPORT_TABLES = ['answer_list', 'response_decomposition', 'branch_best_by_policy']
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
# reset-stale
# ---------------------------------------------------------------------------

def cmd_reset_stale(args):
    queue = ERDQueue(args.queue)
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
    p_run.add_argument('--worker-timeout-seconds', type=int, default=30,
                       metavar='S',
                       help='Declare a worker dead and reclaim its candidate '
                            'claims after S seconds of missed heartbeats '
                            '(default: 30).  Live workers heartbeat every '
                            '~2s regardless of how long a single candidate '
                            'takes, so only a crashed process triggers this.')

    # -- status --
    p_stat = sub.add_parser('status', help='Show progress snapshot')
    p_stat.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_stat.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_stat.add_argument('--worker', type=int, default=None, metavar='N',
                        help='Print spine detail for worker N (one-shot, no --watch needed)')
    p_stat.add_argument('--branch', default=None, metavar='ID',
                        help='Print drill-down detail for branch #ID (the stable '
                             '4-hex id shown in the overview; one-shot)')
    p_stat.add_argument('--watch', nargs='?', const=30, type=int,
                        metavar='SECONDS',
                        help='Repeat every SECONDS (default 30)')

    # -- cache-status --
    p_cs = sub.add_parser('cache-status',
                           help='Show ERD cache coverage for a word')
    p_cs.add_argument('--word', required=True, metavar='WORD',
                      help='Guess word to inspect (e.g. salet)')
    p_cs.add_argument('--missing-only', action='store_true',
                      help='Only list patterns whose branches are not yet cached')
    p_cs.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')

    # -- queue-status --
    p_qs = sub.add_parser('queue-status',
                           help='Show swarm queue coverage for a word')
    p_qs.add_argument('--word', required=True, metavar='WORD',
                      help='Guess word to inspect (e.g. salet)')
    p_qs.add_argument('--queued-only', action='store_true',
                      help='Only list patterns that are pending or in progress')
    p_qs.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qs.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- queue-add --
    p_qa = sub.add_parser('queue-add',
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
    p_qa.add_argument('--max-branch-size', type=int, default=300, metavar='N',
                      help='Skip branches with more than N answer words '
                           '(default: 300)')
    p_qa.add_argument('--delete-erd-cache', action='store_true',
                      help='Delete any existing ERD cache entry for each '
                           'queued branch first, so it is recomputed instead '
                           'of being skipped as already-cached')
    p_qa.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qa.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- queue-clear --
    p_qc = sub.add_parser('queue-clear',
                           help='Wipe all queue state (does not touch the ERD cache)')
    p_qc.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_qc.add_argument('--yes', action='store_true',
                      help='Skip confirmation prompt')

    # -- queue-inspect --
    p_qi = sub.add_parser('queue-inspect',
                           help='Show queue detail for a specific branch')
    p_qi.add_argument('--word', required=True, metavar='WORD')
    p_qi.add_argument('--pattern', required=True, metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qi.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qi.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- queue-remove --
    p_qr = sub.add_parser('queue-remove',
                           help='Remove a pending branch from the queue')
    p_qr.add_argument('--word', required=True, metavar='WORD')
    p_qr.add_argument('--pattern', required=True, metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qr.add_argument('--force', action='store_true',
                      help='Also cancel an in-progress branch (clears active '
                           'state so the worker moves on after its current chunk)')
    p_qr.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qr.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

    # -- queue-priority --
    p_qp = sub.add_parser('queue-priority',
                           help='Set the priority of a queued branch')
    p_qp.add_argument('--word', required=True, metavar='WORD')
    p_qp.add_argument('--pattern', required=True, metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qp.add_argument('--priority', required=True, type=int, metavar='N',
                      help='New priority (higher = worked sooner; '
                           'use values 0–999 for normal work)')
    p_qp.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qp.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

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
        'cache-status': cmd_cache_status,
        'queue-status': cmd_queue_status,
        'queue-add': cmd_queue_add,
        'queue-clear': cmd_queue_clear,
        'queue-inspect': cmd_queue_inspect,
        'queue-remove': cmd_queue_remove,
        'queue-priority': cmd_queue_priority,
        'start': cmd_start,
        'stop': cmd_stop,
        'restart': cmd_restart,
        'run': cmd_run,
        'status': cmd_status,
        'reset-stale': cmd_reset_stale,
        'export': cmd_export,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
