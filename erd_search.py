#!/usr/bin/env python3
"""erd_search.py — Parallel ERD_ALL precache CLI.

Subcommands
-----------
start           Start the supervisor via systemd (systemctl --user start).
stop            Stop the supervisor via systemd (systemctl --user stop).

status          Read-only progress snapshot: queue counts, cache throughput,
                per-worker heartbeats.  --watch loops the display.

run             Start the supervisor directly (without systemd), for
                development or one-shot use.  All output goes to erd_search.log.

queue-add       Add branches for a word or word list to the work queue.
                Replaces the old 'bootstrap' command.  Idempotent: existing
                branches are never duplicated; priority is upgraded if the new
                request is higher.

queue-clear     Wipe all queue state (pending branches, active state, chunk
                claims, heartbeats).  Does not touch the ERD cache.

queue-inspect   Show queue and worker detail for a specific branch.

queue-remove    Remove a pending branch from the queue.  Use --force to also
                cancel an in-progress branch (workers move on after their
                current chunk completes).

queue-priority  Change the priority of a queued branch.  Higher numbers are
                worked sooner; 0 is the default.

solve-branch    Swarm N workers onto one specific branch to completion.
                Useful for large branches that the regular queue would take
                very long to reach.

export          Create a trimmed snapshot of the cache for the iPhone
                (answer_list, response_decomposition, branch_best_by_policy).
                Safe while workers are active; re-running is incremental.

cache-status    Show ERD cache coverage for a given word: which response
                patterns are cached and which are missing.

bootstrap       (Legacy) Walk a word list and populate the queue.  Use
                queue-add instead.
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
from erd_queue import ERDQueue, encode_subset
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
    queue = ERDQueue(args.queue)

    # Reset any stale in_progress rows from a previous interrupted run.
    stale = queue.reset_stale_in_progress()
    if stale:
        print(f'Reset {stale} stale in_progress rows to pending.')

    # Print answer_list_id so the user can verify it matches their iPhone.
    print(f'Universe ID : {score_cache.answer_list_id[:16]}...'
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
                total = queue.total_branches()
                pct = (len(done_roots) / len(root_words)) * 100
                print(f'\r  [{len(done_roots):5d}/{len(root_words)}]'
                      f' {word.upper():<10s}'
                      f'  {total:,} branches queued'
                      f'  ({pct:.1f}%)',
                      end='', flush=True)

        print()
        queue.set_meta('bootstrap_status', 'done')
        queue.set_meta('bootstrap_completed_at', str(int(time.time())))
        total = queue.total_branches()
        counts = queue.counts_by_status()
        print(f'\nBootstrap complete.')
        print(f'  {total:,} distinct branches queued')
        print(f'  {counts.get("done", 0):,} already done '
              f'(cached from iPhone)')

    except KeyboardInterrupt:
        print('\n\nBootstrap interrupted — progress saved, resumable.')
    finally:
        score_cache.checkpoint()
        score_cache.close()
        queue.close()


# ---------------------------------------------------------------------------
# queue-add  (replaces bootstrap for targeted loading)
# ---------------------------------------------------------------------------

def cmd_queue_add(args):
    """Add branches for one word (or a word-list file) to the queue.

    With --word: adds all response branches for that word whose answer-word
    count is between 2 and --max-branch-size.  With --pattern as well: adds
    only that single branch.

    With --word-list: walks every word in the file, same as --word repeated.
    Equivalent to the old 'bootstrap' command.

    Already-queued branches are never duplicated; their priority is upgraded
    if the new request is higher.
    """
    from erd_queue import parse_pattern, fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    priority = args.priority

    score_cache = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ERDQueue(args.queue)

    if args.word:
        words_to_process = [args.word.strip().lower()]
    else:
        words_to_process = load_word_list(args.word_list)

    n_added = 0
    try:
        for word in words_to_process:
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
    counts = queue.counts_by_status()
    pending = counts.get('pending', 0)
    done = counts.get('done', 0)
    in_prog = len(queue.branches_in_progress())
    queue.close()

    print(f'Queue: {pending:,} pending   {done:,} done   {in_prog} in progress')
    if not args.yes:
        ans = input('Clear all queue state? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    queue = ERDQueue(args.queue)
    queue.clear()
    queue.close()
    print('Queue cleared.')


# ---------------------------------------------------------------------------
# queue-inspect
# ---------------------------------------------------------------------------

def cmd_queue_inspect(args):
    """Show the queue entry for a specific branch (word + pattern)."""
    from erd_queue import parse_pattern, fmt_pattern

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
    from erd_queue import parse_pattern, fmt_pattern

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
        # Clear chunk claims and the active_branches row so the worker's next
        # heartbeat can't claim more chunks for this branch.
        queue._conn.execute(
            "DELETE FROM branch_chunks WHERE branch_key = ?", (branch_key,))
        queue._conn.execute(
            "DELETE FROM active_branches WHERE branch_key = ?", (branch_key,))
        print(f'Cancelled in-progress work for {word.upper()} {pat}.')

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
    from erd_queue import parse_pattern, fmt_pattern

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


def _run_systemctl(action: str) -> int:
    """Run `systemctl --user <action> <service>` and return the exit code."""
    import subprocess
    result = subprocess.run(
        ['systemctl', '--user', action, _SYSTEMD_SERVICE],
        capture_output=False)
    return result.returncode


def cmd_start(_args):
    """Start the supervisor via systemd."""
    rc = _run_systemctl('start')
    if rc == 0:
        _run_systemctl('status')
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

    bootstrap_status = queue.get_meta('bootstrap_status')
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

    while not stop_event.is_set():
        time.sleep(5)

        q = ERDQueue(args.queue)
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
              args.min_words_per_chunk, args.max_chunk_count),
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
        queue = ERDQueue(args.queue)
        counts = queue.counts_by_status()
        branches = queue.branches_in_progress()
        hbs = queue.heartbeats_with_branch()
        worker_counts = queue.worker_counts_by_branch()
        bootstrap_status = queue.get_meta('bootstrap_status')
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
        bootstrap_status = None

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
        key = bytes(b['branch_key'])
        n_chunks = ERDQueue.n_chunks_for(b['n_candidates'], b['chunk_size'])
        done = done_chunks.get(key, 0)
        pct = 100.0 * done / n_chunks if n_chunks else 0.0
        src = (f'{b["source_word"].upper()} {fmt_pattern(b["source_pattern"])}'
               if b['source_word'] and b['source_pattern'] is not None
               else '-----')
        bw = (b['best_guess'] or '-----').upper()
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
        key = h['current_branch_key']
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

    branch_key = encode_subset(branch)
    existing = sc.read(branch_key, ERD_ALL)
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
                q = ERDQueue(args.queue)
                row = q.get_branch(branch_key)
                if row is not None:
                    n_chunks = ERDQueue.n_chunks_for(
                        row['n_candidates'], row['chunk_size'])
                    done = q.branch_done_chunks(branch_key)
                    bw, be = q.read_branch_best(branch_key)
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
        sc2.delete(branch_key, ERD_ALL)
        sc2.close()

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    t0 = time.time()
    result = run_branch_solve(
        branch_key, branch, n_workers=args.workers,
        cache_path=args.cache, queue_path=args.queue,
        min_words_per_chunk=args.min_words_per_chunk,
        max_chunk_count=args.max_chunk_count,
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
                             'infeasible large branches from poor root words)')

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
    p_sb.add_argument('--min-words-per-chunk', type=int, default=3, metavar='N',
                      help='Minimum answer-word count per chunk (default: 3); '
                           'see `run --min-words-per-chunk`')
    p_sb.add_argument('--max-chunk-count', type=int, default=256, metavar='N',
                      help='Cap on number of chunks for the branch (default: 256)')
    p_sb.add_argument('--priority', type=int, default=1, metavar='P',
                      help='Priority recorded on the branch (default: 1)')
    p_sb.add_argument('--force', action='store_true',
                      help='Recompute even if the branch is already cached')
    p_sb.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_sb.add_argument('--queue', default=DEFAULT_QUEUE, metavar='PATH')

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
    p_qa.add_argument('--max-branch-size', type=int, default=300, metavar='N',
                      help='Skip branches with more than N answer words '
                           '(default: 300)')
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
        'queue-add': cmd_queue_add,
        'queue-clear': cmd_queue_clear,
        'queue-inspect': cmd_queue_inspect,
        'queue-remove': cmd_queue_remove,
        'queue-priority': cmd_queue_priority,
        'start': cmd_start,
        'stop': cmd_stop,
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
