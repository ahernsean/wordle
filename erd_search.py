#!/usr/bin/env python3.13
"""erd_search.py — Parallel ERD_ALL precache CLI.

Subcommands
-----------
start           Start the supervisor and the report web server via systemd
                (systemctl --user start), unless each is already running.
                --swarm-only starts the supervisor alone; --web-only starts
                the report web server alone.
stop            Stop the supervisor and the report web server via systemd
                (systemctl --user stop).  --swarm-only stops the supervisor
                alone; --web-only stops the report web server alone.
restart         Restart the supervisor and the report web server via systemd
                (systemctl --user restart): a stop followed by a start in one
                step, per service.  --swarm-only restarts the supervisor
                alone; --web-only restarts the report web server alone.

view            Shared swarm reports in text, JSON, or watched JSON Lines.

run             Start the supervisor directly (without systemd), for
                development or one-shot use.  All output goes to
                runtime/erd_search.log.

queue           Queue mutation operations.
queue add       Add branches for one or more words to the work queue.
                Idempotent: existing branches are never duplicated; priority
                is upgraded if the new request is higher.  With --words-file,
                --priority-words marks a subset of the file's words as
                higher priority.  --delete-erd-cache forces a recompute of
                branches that are already cached.  Reports how many branches
                were newly queued, already queued, and already cached.

queue clear     Wipe all queue state (pending branches, active state, candidate
                claims, heartbeats).  Does not touch the ERD cache.

queue remove    Remove a pending branch from the queue.  Use --force to also
                cancel an in-progress branch (workers move on after their
                current candidate evaluation completes).

queue priority  Change the priority of a queued branch.  Higher numbers are
                worked sooner; 0 is the default.

queue source-priority
                Change the requested priority of a source-work request by
                word.  Takes effect immediately for both its pending roots
                and its active/promoted descendants.

epoch           Show or change the telemetry epoch used to compare swarm
                telemetry from one claiming regime.

queue set-disk-stop
                Keep the swarm down across reboots and systemd restarts.

queue reconcile-orphaned-ownership
                Demote open owned branches whose source-work membership was
                lost while they were still open (see check_source_work_
                invariants' "source-owned open branch_id ... has no live
                membership"), making them claimable again.  The run loop
                self-heals this on every membership resolution; this command
                is for branches stranded before that (e.g. accumulated while
                the swarm was down).

For exporting a trimmed cache snapshot to sync to the iPhone, or importing
one from another machine, see export_cache.py and import_cache.py — the
cache is shared with interactive play (wordle.py), not swarm-specific.
"""

from __future__ import annotations

import argparse
import json
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
    BRANCH_PHASES,
    BRANCH_STATUSES,
    ReportFilters,
    ReportRequest,
    WORKER_LIVENESS_SECONDS,
    parse_branch_filter,
    parse_report_branch_target,
    parse_rich_spine as _parse_spine,
    validate_report_request,
)
from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_SEARCH_LOG_PATH,
    ensure_runtime_dir,
)
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache, load_word_list
from erd_queue import (
    DISK_STOP_FRACTION,
    ERDQueue,
    QUEUE_WAL_HARD_CEILING_BYTES,
    check_source_priority_range,
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


def _comma_separated_filter(value, option_name, choices):
    try:
        return parse_branch_filter(value, option_name, choices)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _branch_status_filter(value):
    return _comma_separated_filter(value, "branch status", BRANCH_STATUSES)


def _branch_phase_filter(value):
    return _comma_separated_filter(value, "branch phase", BRANCH_PHASES)


def cmd_view(args):
    from report_terminal import run_view

    run_view(args)


# ---------------------------------------------------------------------------
# queue add
# ---------------------------------------------------------------------------

def cmd_queue_add(args):
    """Add branches for one or more words (or a words-file) to the queue.

    With --word: adds all response branches for each given word with at
    least 2 answer words (and at most --max-branch-size, if given).  With
    --pattern as well: adds only that single branch per word.

    With --words-file: walks every word in the file, same as --word with
    that file's contents.  --priority-words marks a subset of those words as
    higher priority: they are queued at --priority while the rest are queued
    at 0.

    Already-queued branches are never duplicated; their priority is upgraded
    if the new request is higher.  For each word, reports how many of its
    branches are newly queued, were already queued, and are already cached
    (so a worker will resolve them instantly without doing any search).
    --delete-erd-cache deletes each queued branch's existing ERD cache entry
    first, so it gets recomputed instead of being claimed and immediately
    marked done as already-cached.
    """
    from wordle_ui import parse_pattern, fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    if args.word:
        words_to_process = [word.strip().lower() for word in args.word]
    else:
        words_to_process = [word.strip().lower()
                            for word in load_word_list(args.words_file)]

    candidate_words = set(load_word_list(WORDS_FILE))
    invalid_words = [word for word in words_to_process
                     if len(word) != 5 or word not in candidate_words]
    if invalid_words:
        invalid_display = ', '.join(sorted(set(invalid_words)))
        raise ValueError(
            f'invalid candidate word(s): {invalid_display}; expected '
            f'five-letter words from {WORDS_FILE}')

    priority_words = {w.strip().lower() for w in (args.priority_words or [])}
    if priority_words and not args.words_file:
        print('Warning: --priority-words only applies with --words-file; '
              'ignoring it.  Use --priority directly with --word.')
        priority_words = set()

    score_cache = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, score_cache)
    queue = ERDQueue(args.queue)

    unknown = priority_words - set(words_to_process)
    if unknown:
        print(f'Warning: priority words not in the word list: '
              f'{", ".join(sorted(unknown))}')

    # A branch reached by --word has guess_depth 1 (one guess played), so it
    # is solved at ROOT_BUDGET - 1 == GAME_GUESSES - 1.
    branch_budget = GAME_GUESSES - 1

    n_new = 0
    n_already_queued = 0
    n_already_cached = 0
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
                branch_keys = [branch_key for branch_key, *_rest in rows]
                already_queued_keys = set(
                    queue.status_by_branch_keys(branch_keys))
                cache_states = score_cache.report_branch_states(
                    branch_keys, ERD_ALL, budget=branch_budget)
                already_cached_keys = {
                    key for key, state in cache_states.items()
                    if state['cache_state'] in ('exact', 'loss')}

                if args.delete_erd_cache:
                    for branch_key in branch_keys:
                        score_cache.delete(branch_key, ERD_ALL)
                queue.add_pending_many(rows)

                word_already_queued = len(already_queued_keys)
                word_already_cached = len(already_cached_keys)
                word_new = len(rows) - word_already_queued
                n_new += word_new
                n_already_queued += word_already_queued
                n_already_cached += word_already_cached
                print(f'{word.upper()}: {len(rows):,} branch(es) — '
                      f'{word_new:,} new, {word_already_queued:,} already '
                      f'queued, {word_already_cached:,} already cached '
                      f'(resolved instantly, no search needed).')

        total = queue.total_branches()
        n_added = n_new + n_already_queued
        print(f'\n{n_added:,} branch(es) processed across '
              f'{len(words_to_process):,} word(s): {n_new:,} new, '
              f'{n_already_queued:,} already queued, {n_already_cached:,} '
              f'already cached.  Queue total: {total:,}.')

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


def cmd_queue_set_disk_stop(args):
    """Latch the swarm down without replacing an existing latch reason."""
    queue = ERDQueue(args.queue)
    try:
        if queue.set_disk_stop_if_unset(args.reason):
            print(f'Disk-stop latch set: {args.reason}.')
            return
        latch = queue.disk_stop()
        print(f'Disk-stop latch is already set ({latch["reason"]}); '
              'it remains unchanged.')
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

    Priority is an integer; higher numbers are worked sooner.  User-settable
    values in the range 0–999 are reserved for normal use: 0 = default,
    higher = sooner.
    """
    if getattr(args, 'source_word', None):
        queue = ERDQueue(args.queue)
        try:
            updated = queue.set_ownerless_active_priority(
                args.source_word.strip().lower(), args.priority)
        except ValueError as error:
            print(error)
            return
        finally:
            queue.close()
        print(f'{updated:,} ownerless open branch(es) for '
              f'{args.source_word.strip().upper()}: priority set to '
              f'{args.priority}.')
        return

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
# queue source-priority
# ---------------------------------------------------------------------------

def cmd_queue_source_priority(args):
    """Set the requested priority of a source-work request, by word.

    Resolves the word to an open (non-complete) source_work_id via
    ERDQueue.source_work_candidates() and defers to
    ERDQueue.set_source_work_priority(), which applies the change to both the
    request's pending roots and its active/promoted descendants in one
    transaction.  A branch owned by more than one live request keeps the
    higher of their requested priorities (MAX(owner_priority) at the branch
    level), so lowering one request's priority does not necessarily lower a
    branch it shares with a higher-priority request.

    A word with more than one open request is ambiguous; --source-work-id
    picks one.  --source-work-id may also name a completed request directly,
    which is reported as such rather than as "not found".  A word whose
    requests are all complete is reported distinctly from a word with none.
    """
    word = args.word.strip().lower()

    try:
        check_source_priority_range(args.priority)
    except ValueError as error:
        print(error)
        return

    queue = ERDQueue(args.queue)
    try:
        all_rows = {row['source_work_id']: row
                    for row in queue.source_work_rows()
                    if row['source_word'] == word}

        if args.source_work_id is not None:
            if args.source_work_id not in all_rows:
                print(f'{word.upper()}: no source-work request with id '
                      f'{args.source_work_id}.')
                return
            source_work_id = args.source_work_id
        else:
            open_ids = [row['source_work_id']
                        for row in queue.source_work_candidates()
                        if row['source_word'] == word]
            if not open_ids:
                if all_rows:
                    print(f'{word.upper()}: all {len(all_rows)} '
                          f'source-work request(s) are complete.')
                else:
                    print(f'{word.upper()}: no source-work request found.')
                return
            if len(open_ids) > 1:
                print(f'{word.upper()}: ambiguous, {len(open_ids)} open '
                      f'source-work requests match.  '
                      f'Use --source-work-id to disambiguate.')
                for candidate_id in sorted(open_ids):
                    row = all_rows[candidate_id]
                    requested_at = datetime.fromtimestamp(
                        row['requested_at']).strftime('%Y-%m-%d %H:%M')
                    print(f'  id {candidate_id}  '
                          f'priority {row["requested_priority"]}  '
                          f'{row["state"]}  {row["root_count"]} root(s), '
                          f'{row["branch_count"]} branch(es)  '
                          f'requested {requested_at}')
                return
            source_work_id = open_ids[0]

        updated = queue.set_source_work_priority(source_work_id, args.priority)
    finally:
        queue.close()

    if updated:
        print(f'{word.upper()} (id {source_work_id}): '
              f'requested priority set to {args.priority}.')
    else:
        print(f'{word.upper()} (id {source_work_id}): '
              f'request is complete, cannot reprioritize.')


# ---------------------------------------------------------------------------
# start / stop  (systemd delegation)
# ---------------------------------------------------------------------------

_SYSTEMD_SERVICE = 'wordle-erd'
_REPORT_SERVER_SYSTEMD_SERVICE = 'wordle-report-server'


def _run_systemctl(service: str, action: str, *extra: str) -> int:
    """Run `systemctl --user <action> <service> [extra...]` and return the
    exit code."""
    import subprocess
    result = subprocess.run(
        ['systemctl', '--user', action, service, *extra],
        capture_output=False)
    return result.returncode


def _run_journalctl(service: str, since: float) -> int:
    """Print journal entries for ``service`` written since ``since``."""
    import subprocess
    result = subprocess.run(
        ['journalctl', '--user', '--unit', service, '--since', f'@{since}',
         '--no-pager', '--full'],
        capture_output=False)
    return result.returncode


def _service_scope_noun(args) -> str:
    """The subject describing which services a scoped command acted on."""
    if args.swarm_only:
        return 'Supervisor'
    if args.web_only:
        return 'Report server'
    return 'Supervisor and report server'


def _add_service_scope_flags(parser, verb: str) -> None:
    """Add the mutually exclusive --swarm-only / --web-only scope flags shared
    by start, stop, and restart.  Neither flag means act on both services."""
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        '--swarm-only', action='store_true',
        help=f'{verb} the supervisor only; leave the report web server alone')
    scope.add_argument(
        '--web-only', action='store_true',
        help=f'{verb} the report web server only; leave the supervisor alone')


def _start_or_restart_services(args, action: str) -> None:
    """Shared body for `start` and `restart`: run `systemctl <action>` on the
    supervisor (unless --web-only) and the report server (unless --swarm-only).

    A supervisor failure aborts before touching the report server: a broken
    supervisor is the primary problem, and the report server has nothing new
    to add while it's down.  Once the supervisor action succeeds, the report
    server is attempted independently and its failure is reported without
    undoing the supervisor action -- these are two separately-managed
    services, not a transaction."""
    diagnostics_since = time.time()
    if not args.web_only:
        rc = _run_systemctl(_SYSTEMD_SERVICE, action)
        if rc != 0:
            print(f'systemctl {action} failed (exit {rc}).  '
                  f'Is the service installed?  '
                  f'Check: systemctl --user status {_SYSTEMD_SERVICE}',
                  file=sys.stderr)
            sys.exit(rc)

    server_rc = 0
    if not args.swarm_only:
        server_rc = _run_systemctl(_REPORT_SERVER_SYSTEMD_SERVICE, action)
        if server_rc != 0:
            print(f'systemctl {action} failed (exit {server_rc}).  '
                  f'Is the service installed?  '
                  f'Check: systemctl --user status '
                  f'{_REPORT_SERVER_SYSTEMD_SERVICE}',
                  file=sys.stderr)

    if not args.web_only:
        _run_systemctl(_SYSTEMD_SERVICE, 'status', '--no-pager', '--lines=0')
        _run_journalctl(_SYSTEMD_SERVICE, diagnostics_since)
    if not args.swarm_only:
        _run_systemctl(
            _REPORT_SERVER_SYSTEMD_SERVICE, 'status', '--no-pager',
            '--lines=0')
        _run_journalctl(_REPORT_SERVER_SYSTEMD_SERVICE, diagnostics_since)
    if server_rc != 0:
        sys.exit(server_rc)


def cmd_start(args):
    """Start the supervisor and the report web server via systemd, scoped by
    --swarm-only / --web-only.  Starting a service that is already running is a
    no-op."""
    _start_or_restart_services(args, 'start')


def cmd_stop(args):
    """Stop the supervisor and the report web server via systemd, scoped by
    --swarm-only / --web-only.

    Both stops are attempted even if the first fails: stopping is best-effort
    cleanup, not a pipeline, so a failure on one service must never skip the
    other."""
    rc = 0
    if not args.web_only:
        rc = _run_systemctl(_SYSTEMD_SERVICE, 'stop')
        if rc != 0:
            print(f'systemctl stop failed (exit {rc}).', file=sys.stderr)

    server_rc = 0
    if not args.swarm_only:
        server_rc = _run_systemctl(_REPORT_SERVER_SYSTEMD_SERVICE, 'stop')
        if server_rc != 0:
            print(f'systemctl stop failed for report server (exit '
                  f'{server_rc}).', file=sys.stderr)

    if rc == 0 and server_rc == 0:
        print(f'{_service_scope_noun(args)} stopped.')
    else:
        sys.exit(rc or server_rc)


def cmd_restart(args):
    """Restart the supervisor and the report web server via systemd, scoped by
    --swarm-only / --web-only (stop + start in one step, per service)."""
    _start_or_restart_services(args, 'restart')


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
# Cadence for the supervisor's check_source_work_invariants() sweep.  The
# reconciliation in _resolve_branch_memberships already demotes routine
# pruning residue on every membership resolution, so a violation surviving
# to this sweep is a genuine anomaly worth logging rather than the expected
# output of alpha-beta pruning.
INVARIANT_CHECK_SECONDS = 300
QUEUE_WAL_QUIESCE_BYTES = 2 * 1024 ** 3
TRUNCATE_RETRY_SECONDS = 15
# Hard ceiling on the queue WAL (QUEUE_WAL_HARD_CEILING_BYTES, shared with
# the workers' backstop via erd_queue).  In healthy operation the
# quiesce/TRUNCATE protocol keeps the WAL near QUEUE_WAL_QUIESCE_BYTES; a WAL
# an order of magnitude larger means TRUNCATE has been losing for many cycles
# and the file is on course to fill the disk — the failure mode that corrupts
# the database.  At this size the supervisor stops trying to recover in place:
# it captures a diagnostic snapshot, signals every worker to dump its stacks,
# latches the swarm down (a manual `queue clear-disk-stop` is then required),
# and exits.
# Fill/drain rates below this magnitude read as "steady": smaller than this
# and the trend is within statvfs sampling noise (page cache, unrelated
# processes on the same filesystem).  Anything at or above it is shown via
# _fmt_size, whose adaptive K/M/G unit keeps a reportable rate from ever
# rounding to a bare "0".
DISK_RATE_FLOOR_BYTES = 10_000   # 10 kB/s


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


def _check_source_work_invariants(queue):
    """Log any check_source_work_invariants() violations found right now."""
    violations = queue.check_source_work_invariants()
    if violations:
        logger.warning('Source-work invariant check found %d violation(s):',
                       len(violations))
        for violation in violations:
            logger.warning('  %s', violation)


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


def _dump_worker_stacks(procs):
    """SIGUSR1 every live worker so it dumps all-thread stacks to its log.

    Workers register faulthandler on SIGUSR1 (erd_swarm._setup_logging), whose
    C-level handler dumps even from inside a native call (numpy, sqlite).
    Best-effort: a dead pid is skipped, and a worker deep in C returns its dump
    when the call unwinds."""
    for wid, (p, _started_at) in procs.items():
        if not p.is_alive():
            continue
        try:
            os.kill(p.pid, signal.SIGUSR1)
            logger.critical('Requested stack dump from worker-%d (pid=%d).',
                            wid, p.pid)
        except OSError as exc:
            logger.warning('Could not signal worker-%d (pid=%d): %s',
                           wid, p.pid, exc)


def _enforce_wal_hard_ceiling(queue, procs) -> bool:
    """Backstop for a quiesce/TRUNCATE that never wins: when the WAL breaches
    QUEUE_WAL_HARD_CEILING_BYTES, capture why and stop before it fills the disk
    and corrupts the database.  Returns True once it has latched the swarm
    down; the caller then stops."""
    wal_bytes = queue.wal_size_bytes()
    if wal_bytes < QUEUE_WAL_HARD_CEILING_BYTES:
        return False
    logger.critical(
        'Queue WAL %.2f GB breached hard ceiling %.2f GB — TRUNCATE never '
        'reclaimed it. Latching swarm down before the disk fills. Per-table '
        'WAL attribution and worker stacks are in the runtime/erd_worker_*.log files.',
        wal_bytes / 1e9, QUEUE_WAL_HARD_CEILING_BYTES / 1e9)
    now = time.time()
    for h in queue.heartbeats_with_branch():
        started = h['claim_started_at']
        updated = h['updated_at']
        logger.critical(
            '  worker-%s pid=%s cand=%s in_candidate=%ss since_heartbeat=%ss '
            'nodes=%s node_rate=%s/s',
            h['worker_id'], h['pid'], h['cur_candidate'],
            f'{now - started:.0f}' if started else '?',
            f'{now - updated:.0f}' if updated else '?',
            h['cur_nodes'], h['node_rate'])
    _dump_worker_stacks(procs)
    # Let workers flush their dumps to their logs before the supervisor tears
    # them down at the end of the run loop.
    time.sleep(2)
    try:
        queue.set_disk_stop(
            f'supervisor: queue WAL {wal_bytes / 1e9:.1f} GB breached hard '
            f'ceiling ({QUEUE_WAL_HARD_CEILING_BYTES / 1e9:.0f} GB)')
    except sqlite3.OperationalError as exc:
        logger.critical('Could not write disk_stop latch on WAL ceiling: %s',
                        exc)
    return True


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
    last_invariant_check = time.time()
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
            if now - last_invariant_check > INVARIANT_CHECK_SECONDS:
                # Purely diagnostic (six read queries, one log line): a
                # transient OperationalError here must not escalate to the
                # fail-stop handler below, which is for the load-bearing
                # writes elsewhere in this loop.
                try:
                    _check_source_work_invariants(q)
                except sqlite3.OperationalError as exc:
                    logger.warning('Source-work invariant check skipped: %s',
                                   exc)
                last_invariant_check = time.time()
            _maybe_quiesce_truncate(q)
            if _enforce_wal_hard_ceiling(q, procs):
                stop_event.set()
                break

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
        if p.is_alive():
            # A worker that ignores SIGTERM must not outlive the supervisor:
            # every WAL/disk safeguard below assumes no writers remain, and an
            # orphaned writer can fill the disk unsupervised.
            logger.warning('Worker %d did not exit on SIGTERM; killing.', wid)
            p.kill()
            p.join(timeout=10)
    # Workers are gone: the WAL truncates uncontested.
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
    log_path = DEFAULT_SEARCH_LOG_PATH
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


# ---------------------------------------------------------------------------
# queue reconcile-orphaned-ownership
# ---------------------------------------------------------------------------

def cmd_queue_reconcile_orphaned_ownership(args):
    queue = ERDQueue(args.queue)
    branch_ids = queue.reconcile_orphaned_branch_ownership()
    queue.close()
    if not branch_ids:
        print('No orphaned owned branches found.')
        return
    print(f'Demoted {len(branch_ids)} orphaned owned branch(es) to direct '
          f'(claimable without a live source-work membership): '
          f'{", ".join(str(b) for b in branch_ids)}')


def _normalize_queue_cli_args(args):
    """Apply the queue-level path to nested mutation commands."""
    if args.cmd != 'queue':
        return
    if not hasattr(args, 'queue'):
        args.queue = args.queue_path or DEFAULT_QUEUE


def _normalize_epoch_cli_args(args):
    """Apply the epoch-level path to nested epoch commands."""
    if args.cmd != 'epoch':
        return
    if not hasattr(args, 'queue'):
        args.queue = args.queue_path or DEFAULT_QUEUE


# ---------------------------------------------------------------------------
# telemetry epoch
# ---------------------------------------------------------------------------

def _current_git_sha():
    """Return the current checkout's abbreviated commit SHA, if available."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, check=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def cmd_epoch_show(args):
    queue = ERDQueue(args.queue)
    try:
        print(json.dumps(queue.epoch_metadata(), sort_keys=True))
    finally:
        queue.close()


def cmd_epoch_set(args):
    queue = ERDQueue(args.queue)
    try:
        now = int(time.time())
        live_workers = [
            row for row in queue.heartbeats_with_branch()
            if now - row['updated_at'] <= WORKER_LIVENESS_SECONDS
        ]
        if live_workers and not args.force:
            worker_ids = ', '.join(row['worker_id'] for row in live_workers)
            print(
                f'Refusing to change telemetry epoch while live workers are '
                f'heartbeating: {worker_ids}. Stop the swarm first, or use '
                f'--force to override.',
                file=sys.stderr,
            )
            return 1
        git_sha = args.git_sha if args.git_sha is not None else _current_git_sha()
        queue.set_epoch(
            args.epoch, label=args.label, git_sha=git_sha, notes=args.notes,
        )
        print(json.dumps(queue.epoch_metadata(), sort_keys=True))
        return 0
    finally:
        queue.close()


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
    p_view = sub.add_parser(
        'view', help='View shared swarm reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Branch status and phase
  Status describes the branch's relationship to current work:
    active    unfinished branch with at least one current worker
    pending   unfinished branch without a current worker
    done      result persisted; no work remains
    unqueued  semantic branch with no scheduled or persisted work

  Phase describes the durable search milestone normally reached in order:
    queued -> evaluating -> finalizing -> complete

  The axes are related; these are the normal combinations:
    unqueued / -           discovered, but not scheduled or cached
    pending / queued       scheduled and waiting for a worker
    active / evaluating    candidates are being evaluated now
    pending / evaluating   partial evaluation is waiting for a worker
    active / finalizing    a worker is persisting the completed evaluation
    pending / finalizing   an interrupted finalization is waiting for a worker
    done / complete        a reusable result has been persisted

  A branch normally moves through this lifecycle:

    unqueued / -
        -> pending / queued
        -> active / evaluating <-> pending / evaluating
        -> active / finalizing <-> pending / finalizing
        -> done / complete

  Worker arrival and departure change status without discarding phase progress.
  Cached or trivial branches can move directly to done / complete.  Recovery
  may briefly return an interrupted finalization to evaluating before retrying
  it.  Removing unfinished work returns the branch to unqueued / -.

  The overview defaults to --branch-status active.  Use comma-separated values
  such as --branch-status active,pending, or use all to disable that filter.
""")
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
    view_kind.add_argument('--leaderboard', action='store_true')
    view_kind.add_argument(
        '--sources', action='store_true',
        help='Show source-work requests and branch ownership/lineage')
    view_kind.add_argument(
        '--root-progress', dest='root_progress', action='store_true',
        help='Show per-response-group work totals and a completion estimate '
             'for one root word')
    p_view.add_argument(
        '--branch-status', type=_branch_status_filter, metavar='STATUSES',
        help='Comma-separated active,pending,done,unqueued, or all')
    p_view.add_argument(
        '--branch-phase', type=_branch_phase_filter, metavar='PHASES',
        help='Comma-separated queued,evaluating,finalizing,complete, or all')
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
            'evaluated-candidates', 'one-level-erd-prunes',
            'two-level-erd-prunes',
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
                           help='Add branches for one or more words to the queue')
    qa_word = p_qa.add_mutually_exclusive_group(required=True)
    qa_word.add_argument('--word', nargs='+', metavar='WORD',
                         help='One or more guess words, space-separated '
                              '(e.g. --word salet crane raise)')
    qa_word.add_argument('--words-file', metavar='FILE',
                         help=f'File of words to add, one per line '
                              f'(default list: {WORDS_FILE})')
    p_qa.add_argument('--pattern', metavar='PAT',
                      help='Only add this specific response pattern for --word '
                           '(5 chars: g=green y=yellow -=gray).  '
                           'Omit to add all patterns for the word(s).')
    p_qa.add_argument('--priority', type=int, default=0, metavar='N',
                      help='Priority for queued branches (default: 0).  '
                           'Higher numbers are worked sooner.')
    p_qa.add_argument('--priority-words', nargs='+', metavar='WORD',
                      help='With --words-file: only these words are queued at '
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
    qp_target = p_qp.add_mutually_exclusive_group(required=True)
    qp_target.add_argument('--word', metavar='WORD')
    qp_target.add_argument('--source-word', metavar='WORD',
                           help='Set every ownerless open branch attributed '
                                'to this word')
    p_qp.add_argument('--pattern', metavar='PAT',
                      help='Response pattern (5 chars: g=green y=yellow -=gray)')
    p_qp.add_argument('--priority', required=True, type=int, metavar='N',
                      help='New priority (higher = worked sooner; '
                           'use values 0–999 for normal work)')
    p_qp.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qp.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- queue source-priority --
    p_qsp = qsub.add_parser(
        'source-priority',
        help='Set the requested priority of a source-work request')
    p_qsp.add_argument('--word', required=True, metavar='WORD')
    p_qsp.add_argument('--priority', required=True, type=int, metavar='N',
                       help='New requested priority (higher = worked sooner; '
                            'use values 0–999)')
    p_qsp.add_argument('--source-work-id', type=int, default=None, metavar='N',
                       help='source_work_id to disambiguate, when --word '
                            'owns more than one open request')
    p_qsp.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- start --
    p_start = sub.add_parser(
        'start',
        help='Start the supervisor and the report web server via systemd '
             '(systemctl --user start wordle-erd wordle-report-server)')
    _add_service_scope_flags(p_start, 'Start')

    # -- stop --
    p_stop = sub.add_parser(
        'stop',
        help='Stop the supervisor and the report web server via systemd '
             '(systemctl --user stop wordle-erd wordle-report-server)')
    _add_service_scope_flags(p_stop, 'Stop')

    # -- restart --
    p_restart = sub.add_parser(
        'restart',
        help='Restart the supervisor and the report web server via systemd '
             '(systemctl --user restart wordle-erd wordle-report-server)')
    _add_service_scope_flags(p_restart, 'Restart')

    # -- queue reset-stale --
    p_rst = qsub.add_parser('reset-stale',
                             help='Reset in_progress rows to pending')
    p_rst.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- queue reconcile-orphaned-ownership --
    p_qro = qsub.add_parser(
        'reconcile-orphaned-ownership',
        help='Demote open owned branches whose source-work membership was '
             'lost while they were still open, making them claimable again')
    p_qro.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    p_cds = qsub.add_parser(
        'clear-disk-stop', help='Release the disk-stop latch so run can start'
    )
    p_cds.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    p_sds = qsub.add_parser(
        'set-disk-stop',
        help='Keep the swarm down across reboots and systemd restarts',
    )
    p_sds.add_argument('--reason', required=True, metavar='TEXT',
                       help='Reason shown when run refuses to start')
    p_sds.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    # -- epoch --
    p_epoch = sub.add_parser(
        'epoch', help='Show or change the telemetry epoch'
    )
    p_epoch.add_argument('--queue', dest='queue_path', default=None, metavar='PATH')
    esub = p_epoch.add_subparsers(dest='epoch_cmd', required=True)

    p_epoch_show = esub.add_parser(
        'show', help='Print the active telemetry epoch metadata'
    )
    p_epoch_show.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    p_epoch_set = esub.add_parser(
        'set', help='Change the active telemetry epoch while workers are stopped'
    )
    p_epoch_set.add_argument('epoch', type=int, metavar='N')
    p_epoch_set.add_argument('--label', metavar='TEXT')
    p_epoch_set.add_argument('--git-sha', metavar='SHA')
    p_epoch_set.add_argument('--notes', metavar='TEXT')
    p_epoch_set.add_argument(
        '--force', action='store_true',
        help='Allow the change despite live worker heartbeats',
    )
    p_epoch_set.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    args = parser.parse_args()
    ensure_runtime_dir()
    if args.cmd == 'view' and args.format == 'json' and args.watch is not None:
        parser.error('--format json cannot be used with --watch; use jsonl')
    if (args.cmd == 'queue' and args.queue_cmd == 'priority'
            and args.word is not None and args.pattern is None):
        parser.error('--pattern is required with --word')
    if (args.cmd == 'queue' and args.queue_cmd == 'priority'
            and args.source_word is not None and args.pattern is not None):
        parser.error('--pattern can only be used with --word')
    if args.cmd == 'view':
        try:
            args.branch_target = parse_report_branch_target(args.spine)
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
            'hotspots' if args.hotspots else
            'leaderboard' if args.leaderboard else
            'sources' if args.sources else
            'root_progress' if args.root_progress else 'auto'
        )
        if args.by is not None and not args.hotspots:
            parser.error('--by requires --hotspots')
        if not args.hotspots and any(
                value is not None
                for value in (args.since_seconds, args.sample_size)):
            parser.error('--since-seconds and --sample-size require --hotspots')
        if args.epoch is not None and not (args.hotspots or args.root_progress):
            parser.error('--epoch requires --hotspots or --root-progress')
        if args.since_seconds is not None and args.since_seconds < 1:
            parser.error('--since-seconds must be at least 1')
        if args.sample_size is not None and args.sample_size < 1:
            parser.error('--sample-size must be at least 1')
        hotspot_field = args.by or 'nodes'
        if args.hotspots and args.limit is None:
            args.limit = 10
        branch_statuses = args.branch_status
        if branch_statuses is None:
            branch_statuses = (
                ("active",)
                if (args.report_kind == "auto"
                    and args.branch_target.kind == "root" and not args.tree)
                else ()
            )
        args.filters = ReportFilters(
            branch_statuses=branch_statuses,
            branch_phases=args.branch_phase or (),
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
                branch_target=args.branch_target,
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
    _normalize_epoch_cli_args(args)

    if args.cmd == 'queue':
        qdispatch = {
            'add': cmd_queue_add,
            'clear': cmd_queue_clear,
            'remove': cmd_queue_remove,
            'priority': cmd_queue_priority,
            'source-priority': cmd_queue_source_priority,
            'reset-stale': cmd_reset_stale,
            'reconcile-orphaned-ownership': cmd_queue_reconcile_orphaned_ownership,
            'clear-disk-stop': cmd_queue_clear_disk_stop,
            'set-disk-stop': cmd_queue_set_disk_stop,
        }
        if args.queue_cmd == 'add':
            try:
                qdispatch[args.queue_cmd](args)
            except ValueError as error:
                parser.error(str(error))
        else:
            qdispatch[args.queue_cmd](args)
        return

    if args.cmd == 'epoch':
        epoch_dispatch = {
            'show': cmd_epoch_show,
            'set': cmd_epoch_set,
        }
        exit_code = epoch_dispatch[args.epoch_cmd](args)
        if isinstance(exit_code, int) and exit_code != 0:
            sys.exit(exit_code)
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
