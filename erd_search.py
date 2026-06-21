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
import io
import logging
import multiprocessing
import os
import select
import signal
import sys
import termios
import time
from datetime import datetime

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, ResponseCache, load_word_list
from erd_queue import ERDQueue, encode_subset
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
    chunks = queue.chunks_for_branch(branch_key) if ab else []
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
        n_chunks = ERDQueue.n_chunks_for(ab['n_candidates'], ab['chunk_size'])
        done_ct = sum(1 for c in chunks if c['done'])
        best_g = ab['best_guess'] or '---'
        best_e = f'{ab["best_erd"]:.4f}' if ab['best_erd'] is not None else '---'
        print(f'  In-progress: chunks {done_ct}/{n_chunks}  '
              f'best {best_g.upper()} {best_e}')
        print(f'  Chunk detail:')
        for c in chunks:
            holder = c['claimed_by'] or '(unclaimed)'
            status = 'done' if c['done'] else f'held by {holder}'
            print(f'    chunk {c["idx"]:3d}: {status}')


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
              f'{nb} in-progress branches / {nc} chunk claims cleared.')

    counts = queue.counts_by_status()
    if not counts.get('pending') and not counts.get('in_progress'):
        print('Warning: queue appears empty.  '
              'Run queue-add to load branches before starting workers.',
              file=sys.stderr)
    queue.close()

    _setup_supervisor_logging()
    logger.info('Supervisor starting: %d workers, min_words_per_chunk=%d, '
                'max_chunk_count=%d, recycle_hours=%.1f',
                args.workers, args.min_words_per_chunk,
                args.max_chunk_count, args.recycle_hours)

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
        freed = q.reclaim_stale_chunks(args.worker_timeout_seconds)
        if freed:
            logger.info('Reclaimed %d stale chunk claim(s).', freed)
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
    """Free a dead/killed worker's in-flight chunk claims and clear its
    heartbeat, BEFORE a replacement of the same name starts heartbeating.

    Without this, a respawned worker-N would refresh the 'worker-N' heartbeat,
    making the previous instance's orphaned (done=0) chunks look like they're
    held by a live worker — so the liveness-gated reclaim would never free them
    and the affected branches would never finalize.
    """
    name = f'worker-{worker_id}'
    freed = queue.reclaim_chunks_of_worker(name)
    queue.clear_heartbeat(name)
    if freed:
        logger.info('Reaped worker %d: freed %d chunk claim(s).',
                    worker_id, freed)


def _spawn_worker(worker_id: int, args, stop_event):
    p = multiprocessing.Process(
        target=erd_swarm.swarm_worker,
        args=(worker_id, args.cache, args.queue, stop_event,
              args.min_words_per_chunk, args.max_chunk_count, args.workers),
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
                _redraw_status.prev_lines = []
                while True:
                    _redraw_status(args)
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass
            finally:
                sys.stdout.write('\033[?25h')
                sys.stdout.flush()
    else:
        _print_status(args)


def _watch_with_keys(args, interval):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    # Disable canonical mode and echo without touching output processing (OPOST).
    # tty.setraw() also clears OPOST, which breaks \n → \r\n translation.
    new_settings[3] &= ~(termios.ICANON | termios.ECHO)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        sys.stdout.write('\033[?25l\033[2J\033[H')
        sys.stdout.flush()
        _redraw_status.prev_lines = []
        while True:
            _redraw_status(args)
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
    except KeyboardInterrupt:
        # ISIG stays enabled (only ICANON/ECHO are off), so Ctrl-C raises here
        # rather than arriving as a '\x03' byte from stdin.read().
        pass
    finally:
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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


def _redraw_status(args):
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        _print_status(args)
    finally:
        sys.stdout = old_stdout
    new_lines = buf.getvalue().splitlines()
    prev_lines = getattr(_redraw_status, 'prev_lines', [])
    _redraw_status.prev_lines = new_lines

    rendered = []
    for i, line in enumerate(new_lines):
        old_line = prev_lines[i] if i < len(prev_lines) else None
        if old_line is not None and line != old_line:
            rendered.append(_highlight_changes(line, old_line) + '\033[K')
        else:
            rendered.append(line + '\033[K')

    # \033[H  — cursor to top-left (no screen erase, so no flash)
    # \033[J  — erase from last line to end of screen (removes stale lines if output shrank)
    out = '\033[H' + '\n'.join(rendered) + '\033[J'
    sys.stdout.write(out)
    sys.stdout.flush()


def _print_status(args):
    now_ts = int(time.time())
    from wordle_ui import fmt_pattern

    # Queue + swarm state
    try:
        queue = ERDQueue(args.queue)
        counts = queue.counts_by_status()
        branches = queue.branches_in_progress()
        hbs = queue.heartbeats_with_branch()
        worker_counts = queue.worker_counts_by_branch()
        # Per-branch done-chunk counts for the in-progress branches.
        done_chunks = {bytes(b['branch_key']): queue.branch_done_chunks(b['branch_key'])
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
    cache_line = f'Cache:  {total_erd:,} ERD entries   +{recent:,} in last 5m' if cache_ok else None
    if cache_line and hit_pct is not None:
        cache_line += f'   hits {_abbrev(hits)}/{_abbrev(hit_total)} ({_fmt_pct(hit_pct)})'
    if cache_line:
        print(cache_line)
    print()

    # Branches in progress — the real progress unit.
    _COOP_PRIORITY = 1_000_000   # sentinel: cooperative sub-branch, not user-queued
    branch_hdr = 'Branches:'
    if queue_ok:
        n_in_prog = counts.get('in_progress', 0)
        n_pending = counts.get('pending', 0)
        n_done = counts.get('done', 0)
        n_coop = sum(1 for b in branches if (b['priority'] or 0) >= _COOP_PRIORITY)
        in_prog_str = f'in progress {n_in_prog}'
        if n_coop:
            in_prog_str += f' ({n_coop} cooperative)'
        parts = [f'done {n_done:,}', in_prog_str, f'pending {n_pending:,}']
        branch_hdr += ' ' + ', '.join(parts)
    print(branch_hdr)
    denom_w = 2
    if branches:
        max_nc = max(ERDQueue.n_chunks_for(b['n_candidates'] or 1, b['chunk_size'])
                     for b in branches)
        denom_w = len(str(max_nc))
    chunks_col_w = 2 * denom_w + 5   # num/denom + ' ' + 2-digit int pct + '%'
    if branches:
        print(f'  {"Root":<11s} {"D":>1s} {"Ans":>4s} '
              f'{"Chunks":<{chunks_col_w}s} '
              f'{"Best":<5s} {"ERD":>5s}  {"Pri":>3s} {"Wk":>2s} ETA')
    else:
        print('  (none)')
    for b in branches:
        key = bytes(b['branch_key'])
        n_cands = b['n_candidates'] or 0
        n_chunks = ERDQueue.n_chunks_for(n_cands, b['chunk_size'])
        done = done_chunks.get(key, 0)
        pct = 100.0 * done / n_chunks if n_chunks else 0.0
        src = (f'{b["source_word"].upper()} {fmt_pattern(b["source_pattern"])}'
               if b['source_word'] and b['source_pattern'] is not None
               else '-----')
        bw = (b['best_guess'] or '-----').upper()
        be = f'{b["best_erd"]:.3f}' if b['best_erd'] is not None else '---'
        nw = b['n_words'] or 0
        wk = worker_counts.get(key, 0)
        raw_pri = b['priority'] or 0
        pri_str = 'COOP' if raw_pri >= _COOP_PRIORITY else str(raw_pri)
        depth_val = b['depth'] if 'depth' in b.keys() else 0
        created = b['created_at'] or now_ts
        el = now_ts - created
        eta = ''
        if wk > 0 and 0 < done < n_chunks and el > 0:
            rem = (n_chunks - done) / (done / el)
            eta = _fmt_duration(int(rem))
        chunks_str = f'{done:>{denom_w}d}/{n_chunks:<{denom_w}d} {int(pct):2d}%'
        print(f'  {src:<11s} {depth_val:1d} {nw:4d} '
              f'{chunks_str} '
              f'{bw:<5s} {be:>5s} {pri_str:>4s} {wk:2d} {eta}')
    print()

    # Workers — liveness and forward progress.
    if not hasattr(_print_status, '_answer_set'):
        _print_status._answer_set = set(load_word_list(ANSWER_FILE))
    answer_set = _print_status._answer_set
    worker_hdr = 'Workers:'
    if live and total_evals:
        parts = []
        if cutoff_pct is not None:
            parts.append(f'ERD-pruned {_abbrev(n_cutoff)}/{_abbrev(total_evals)} ({_fmt_pct(cutoff_pct)})')
        if pruned_pct is not None:
            parts.append(f'depth-pruned {_abbrev(n_pruned)}/{_abbrev(total_evals)} ({_fmt_pct(pruned_pct)})')
        if parts:
            # Replace the worker header
            worker_hdr = f'Workers: {", ".join(parts)}:'
    print(worker_hdr)
    branch_depth_map = {bytes(b['branch_key']): (b['depth'] if 'depth' in b.keys() else 0)
                        for b in branches}
    # Fresh accumulator each cycle: path shown reflects this refresh only.
    max_paths = {}
    if not hbs:
        print('  (none active)')
    else:
        print(f'  {"W":>2}  {"Root":<11}  {"D":>1}  {"Chk":>3}  {"Held":>6}  '
              f'{"Best":<5}  {"ERD":>5}  {"Bound":>5}  HB')
    for h in sorted(hbs, key=lambda r: r['worker_id']):
        age = now_ts - h['updated_at']
        wnum = h['worker_id'].split('-')[-1] if '-' in h['worker_id'] else h['worker_id']
        flag = ' !!' if age > 120 else ''
        key = h['current_branch_key']
        if key is None:
            print(f'  {wnum:>2}  {"(idle)":<11}  {0:1d}  {"":>3}  {"":>6}  '
                  f'{"":5}  {"":>5}  {"":>5}  {age}s{flag}')
            continue
        src = (f'{h["source_word"].upper()} {fmt_pattern(h["source_pattern"])}'
               if h['source_word'] and h['source_pattern'] is not None
               else '-----')
        w_depth = branch_depth_map.get(bytes(key), 0)
        chunk = str(h['chunk_idx']) if h['chunk_idx'] is not None else '-'
        held = now_ts - (h['chunk_started_at'] or now_ts)
        cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
        n_seen = (h['cand_n_seen'] if 'cand_n_seen' in h.keys() else None) or 0
        c_total = (h['cand_chunk_size'] if 'cand_chunk_size' in h.keys() else None) or 0
        mdepth = (h['cur_max_depth'] if 'cur_max_depth' in h.keys() else None) or 0
        nodes = (h['cur_nodes'] if 'cur_nodes' in h.keys() else None) or 0
        nrate = (h['node_rate'] if 'node_rate' in h.keys() else None) or 0.0
        path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
        if '>' in path:
            path = path.replace('>', '→')
        # Show the deepest path seen across this watch cycle.  Each heartbeat
        # writes the max spine from its own 2-second window (_hb_max_spine in
        # erd_swarm.py, reset after each write); max_paths accumulates across
        # successive heartbeat reads so the display shows the maximum over the
        # full watch interval.  Tied depth overwrites so the displayed path is
        # the most recent at that depth.
        path_key = (h['worker_id'], bytes(key) if key else None)
        prev_max = max_paths.get(path_key, '')
        if path.count('→') >= prev_max.count('→'):
            max_paths[path_key] = path
        else:
            path = prev_max
        best_g = (h['best_guess'] if 'best_guess' in h.keys() else None) or ''
        best_e = (h['best_erd'] if 'best_erd' in h.keys() else None)
        bound_e = (h['bound_erd'] if 'bound_erd' in h.keys() else None)
        # Forward-progress flag: heartbeat fresh but evaluation rate is zero == hang.
        if age <= 10 and nrate == 0 and nodes:
            flag += ' ~?'
        best_g_disp = best_g.upper() if best_g else '-----'
        best_e_disp = f'{best_e:.3f}' if best_e is not None else '-----'
        bound_e_disp = f'{bound_e:.3f}' if bound_e is not None else '-----'
        print(f'  {wnum:>2}  {src:<11}  {w_depth:1d}  {chunk:>3}  {_fmt_duration(held):>6}  '
              f'{best_g_disp:<5}  {best_e_disp:>5}  {bound_e_disp:>5}  {age}s{flag}')
        if cur:
            cur_disp = cur.upper() + ('*' if cur.lower() in answer_set else ' ')
            krate = f'{int(nrate / 1000)}kN/s'
            print(f'       {cur_disp}  {n_seen:>4}/{c_total:<4}  MaxD:{mdepth}  {krate:>7}  {path}')
        else:
            print()


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
    p_run.add_argument('--min-words-per-chunk', type=int, default=3, metavar='N',
                       help='Minimum answer-word count per chunk of work: '
                            'a branch is split into ceil(n_words/N) chunks '
                            '(default: 3).  Lower = more chunks = more '
                            'worker sharing on hard branches.')
    p_run.add_argument('--max-chunk-count', type=int, default=256, metavar='N',
                       help='Cap on the number of chunks per branch (default: 256). '
                            'When --min-words-per-chunk would produce more chunks '
                            'than this cap, the cap wins and chunks become larger.')
    p_run.add_argument('--recycle-hours', type=float, default=3.0,
                       metavar='H',
                       help='Respawn each worker after H hours wall time '
                            '(default: 3).  Bounds per-worker ScoreCache '
                            'memory growth while in-progress work is preserved '
                            'in the queue and resumed by the fresh worker.')
    p_run.add_argument('--worker-timeout-seconds', type=int, default=30,
                       metavar='S',
                       help='Declare a worker dead and reclaim its chunk '
                            'claims after S seconds of missed heartbeats '
                            '(default: 30).  Live workers heartbeat every '
                            '~2s regardless of how long a single candidate '
                            'takes, so only a crashed process triggers this.')

    # -- status --
    p_stat = sub.add_parser('status', help='Show progress snapshot')
    p_stat.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')
    p_stat.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
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
