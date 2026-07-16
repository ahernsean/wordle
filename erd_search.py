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

view            Shared report overview in text, JSON, or watched JSON Lines.

run             Start the supervisor directly (without systemd), for
                development or one-shot use.  All output goes to erd_search.log.

queue           Queue dashboard and nested queue operations.
queue add       Add branches for a word or word list to the work queue.
                Idempotent: existing branches are never duplicated; priority
                is upgraded if the new request is higher.  With --word-list,
                --priority-words marks a subset of the list's words as
                higher priority.  --delete-erd-cache forces a recompute of
                branches that are already cached.

queue clear     Wipe all queue state (pending branches, active state, candidate
                claims, heartbeats).  Does not touch the ERD cache.

queue show      Show queue and worker detail for a specific branch.

queue remove    Remove a pending branch from the queue.  Use --force to also
                cancel an in-progress branch (workers move on after their
                current candidate evaluation completes).

queue priority  Change the priority of a queued branch.  Higher numbers are
                worked sooner; 0 is the default.

cache-status    Show ERD cache coverage for a given word: which response
                patterns are cached and which are missing.

queue coverage  Show swarm queue coverage for a given word/path: which response
                patterns are pending, in progress, done, or not yet queued.

For exporting a trimmed cache snapshot to sync to the iPhone, or importing
one from another machine, see export_cache.py and import_cache.py — the
cache is shared with interactive play (wordle.py), not swarm-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import multiprocessing
import os
import select
import sqlite3
from collections import defaultdict
import signal
import sys
import termios
import time
from datetime import datetime

from cache_sqlite import ScoreCache
from report_model import (
    ReportFilters,
    WORKER_LIVENESS_SECONDS,
    parse_report_selector,
    parse_rich_spine as _parse_spine,
)
from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    DEFAULT_QUEUE_PATH,
)
from wordle_engine import ERD_ALL, ResponseCache, load_word_list
from erd_queue import (ERDQueue, encode_subset, guess_depth_from_spine,
                       disk_stats, DISK_WARN_FRACTION, DISK_STOP_FRACTION)
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
# queue visibility
# ---------------------------------------------------------------------------

class QueueRefError(ValueError):
    pass


def normalize_queue_pattern(token: str) -> str:
    """Normalize a response token to canonical g/y/- syntax."""
    from wordle_ui import fmt_pattern, parse_pattern
    try:
        return fmt_pattern(parse_pattern(token))
    except ValueError as e:
        raise QueueRefError(str(e)) from e


def parse_queue_branch_ref(text) -> list[tuple[str, str | None]]:
    """Parse WORD [PATTERN] ... refs used by queue visibility commands."""
    if isinstance(text, (list, tuple)):
        text = ' '.join(text)
    text = (text or '').strip()
    if not text:
        raise QueueRefError('empty branch reference')
    raw = text.split()
    tokens = []
    i = 0
    while i < len(raw):
        word = raw[i].strip().upper()
        if len(word) != 5 or not word.isalpha():
            raise QueueRefError(
                'malformed branch ref: expected alternating WORD [PATTERN] tokens')
        pat = None
        if i + 1 < len(raw):
            nxt = raw[i + 1]
            try:
                pat = normalize_queue_pattern(nxt)
                i += 1
            except QueueRefError:
                if len(nxt) != 5:
                    raise QueueRefError(
                        'malformed branch ref: expected alternating WORD [PATTERN] tokens')
                # A five-letter token that is not a valid response pattern is
                # interpreted as the next word, then validated by the final
                # alternating-token check below.
                pass
            if pat is None and len(nxt) != 5:
                raise QueueRefError(
                    'malformed branch ref: expected alternating WORD [PATTERN] tokens')
        tokens.append((word, pat))
        i += 1
    if any(pat is None for _word, pat in tokens[:-1]):
        raise QueueRefError(
            'malformed branch ref: expected alternating WORD [PATTERN] tokens')
    return tokens


def spine_prefix_sql(tokens) -> str:
    parts = []
    for word, pat in tokens:
        parts.append(word.upper())
        if pat is not None:
            parts.append(pat)
    return ' '.join(parts)


def _parse_optional_queue_ref(parts):
    if not parts:
        return None
    return parse_queue_branch_ref(parts)


def _json_safe(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _print_json(value):
    print(json.dumps(_json_safe(value), indent=2, sort_keys=True))


def _branch_ref_to_key(tokens, all_answers, cache_path):
    """Resolve a fully-patterned path to its answer subset branch key."""
    from wordle_ui import parse_pattern
    sc = ScoreCache(cache_path, all_answers)
    rcache = ResponseCache(all_answers, sc)
    branch = list(all_answers)
    try:
        for word, pat in tokens:
            if pat is None:
                break
            code = parse_pattern(pat)
            groups = rcache.group_words(word.lower(), branch)
            branch = groups.get(code, [])
        return encode_subset(branch) if len(branch) >= 2 else None, branch
    finally:
        sc.close()


def _row_path(row):
    if row.get('spine'):
        return _fmt_spine_path(row['spine'])
    if row.get('source_word') and row.get('source_pattern_text'):
        return f'{row["source_word"].upper()} {row["source_pattern_text"]}'
    if row.get('source_word'):
        return row['source_word'].upper()
    return ''


def _print_queue_table(rows, title=None):
    if title:
        print(title)
    if not rows:
        print('  (none)')
        return
    table_rows = []
    for row in rows:
        branch_id = _branch_id(row['branch_key'])
        done_candidates = (f'{row["done_candidates"]}/{row["n_candidates"]}'
                           if row.get('n_candidates') else '---')
        priority = ('COOP' if (row.get('priority') or 0) >= 1_000_000
                    else str(row.get('priority') or 0))
        table_rows.append((
            branch_id,
            row['kind'],
            row['status'],
            priority,
            str(row['n_words']),
            done_candidates,
            str(row.get('worker_count', 0)),
            str(row.get('nodes_spent') or 0),
            _row_path(row),
        ))

    headings = ('ID', 'Kind', 'Status', 'Pri', 'Words', 'Done', 'W', 'Nodes')
    column_widths = [
        max(len(heading), *(len(row[index]) for row in table_rows))
        for index, heading in enumerate(headings)
    ]

    def format_table_row(row_values):
        left = (
            f'{row_values[0]:<{column_widths[0]}} '
            f'{row_values[1]:<{column_widths[1]}} '
            f'{row_values[2]:<{column_widths[2]}}'
        )
        right = ' '.join(
            f'{row_values[index]:>{column_widths[index]}}'
            for index in range(3, 8)
        )
        return f'{left} {right}  {row_values[8]}'

    print(format_table_row((*headings, 'Spine')))
    for table_row in table_rows:
        print(format_table_row(table_row))


def cmd_queue_dashboard(args):
    queue = ERDQueue(args.queue)
    try:
        data = queue.queue_dashboard(args.limit)
    finally:
        queue.close()
    if args.json:
        _print_json(data)
        return
    s = data['summary']
    status = '  '.join(f'{k} {v:,}' for k, v in sorted(s['by_status'].items()))
    kind = '  '.join(f'{k} {v:,}' for k, v in sorted(s['by_kind'].items()))
    print(f'Queue: {s["total"]:,} branches')
    print(f'Status: {status or "none"}')
    print(f'Kind: {kind or "none"}')
    print()
    _print_queue_table(data['active'], 'Active')
    print()
    _print_queue_table(data['pending'], 'Top pending')


def cmd_queue_ls(args):
    filters = {
        'status': args.status,
        'min_words': args.min_words,
        'max_words': args.max_words,
        'budget': args.budget,
        'priority': args.priority,
        'source_word': args.source_word,
    }
    if args.prefix:
        filters['prefix'] = spine_prefix_sql(parse_queue_branch_ref(args.prefix))
    queue = ERDQueue(args.queue)
    try:
        rows = queue.list_queue_rows(filters, limit=args.limit)
    finally:
        queue.close()
    if args.json:
        _print_json(rows)
    else:
        _print_queue_table(rows)


def cmd_queue_tree(args):
    try:
        tokens = _parse_optional_queue_ref(args.prefix)
        prefix = spine_prefix_sql(tokens) if tokens else None
    except QueueRefError as e:
        print(f'Invalid branch ref: {e}')
        return
    queue = ERDQueue(args.queue)
    try:
        rows = queue.queue_tree_rows(
            prefix, active_only=args.active_only,
            max_depth=args.max_depth, limit=args.limit)
    finally:
        queue.close()
    if args.json:
        _print_json(rows)
        return
    if not rows:
        print('(none)')
        return
    for r in rows:
        depth = guess_depth_from_spine(r['spine'])
        if not r['spine'] and r.get('source_word'):
            depth = 1
        indent = '  ' * max(0, depth - 1)
        print(f'{indent}{_branch_id(r["branch_key"])} {r["status"]} '
              f'{r["n_words"]}w {_row_path(r)}')


def cmd_queue_summary(args):
    queue = ERDQueue(args.queue)
    try:
        summary = queue.queue_summary()
    finally:
        queue.close()
    if args.json:
        _print_json(summary)
        return
    print(f'Total: {summary["total"]:,}')
    for label, key in [('Status', 'by_status'), ('Kind', 'by_kind'),
                       ('Budget', 'by_budget'), ('Priority', 'by_priority'),
                       ('Size', 'by_size')]:
        parts = '  '.join(f'{k} {v:,}' for k, v in sorted(summary[key].items()))
        print(f'{label}: {parts or "none"}')
    for label in ('largest_pending', 'oldest_pending',
                  'largest_active', 'oldest_active'):
        row = summary[label]
        if row:
            print(f'{label.replace("_", " ").title()}: '
                  f'{_branch_id(row["branch_key"])} {row["n_words"]}w {_row_path(row)}')


def cmd_queue_top(args):
    try:
        tokens = _parse_optional_queue_ref(args.prefix)
        prefix = spine_prefix_sql(tokens) if tokens else None
    except QueueRefError as e:
        print(f'Invalid branch ref: {e}')
        return
    queue = ERDQueue(args.queue)
    try:
        rows = queue.queue_top_rows(args.by, args.limit, prefix)
    finally:
        queue.close()
    if args.json:
        _print_json(rows)
    else:
        _print_queue_table(rows)


def cmd_queue_show(args):
    ref_text = ' '.join(args.branch_ref)
    queue = ERDQueue(args.queue)
    try:
        try:
            tokens = parse_queue_branch_ref(ref_text)
            ref = spine_prefix_sql(tokens)
        except QueueRefError:
            ref = ref_text.strip()
            tokens = []
        matches = queue.resolve_branch_ref(ref)
        if not matches and tokens and tokens[-1][1] is not None:
            all_answers = load_word_list(ANSWER_FILE)
            key, _branch = _branch_ref_to_key(tokens, all_answers, args.cache)
            matches = queue.resolve_branch_ref(key.hex() if key else '')
        if len(matches) != 1:
            if args.json:
                _print_json(matches)
            elif not matches:
                print(f'{ref_text}: no matching branch.')
            else:
                print(f'{ref_text}: {len(matches)} matching branches. '
                      f'Narrow the spine/pattern or use ID.')
                _print_queue_table(matches)
            return
        detail = queue.branch_detail(matches[0]['branch_key'], args.claims)
    finally:
        queue.close()
    if args.json:
        _print_json(detail)
        return
    print(f'{_branch_id(detail["branch_key"])} {detail["branch_key_hex"][:20]}...')
    print(f'  Kind/status : {detail["kind"]} / {detail["status"]}')
    print(f'  Spine       : {_row_path(detail) or "---"}')
    print(f'  Words       : {detail["n_words"]}')
    print(f'  Priority    : {detail["priority"]}')
    print(f'  Budget      : {detail["budget"] if detail["budget"] is not None else "---"}')
    if detail.get('n_candidates'):
        print(f'  Candidates  : {detail["done_candidates"]}/{detail["n_candidates"]}')
    best = (f'{detail["best_guess"].upper()} {detail["best_erd"]:.4f}'
            if detail.get('best_guess') and detail.get('best_erd') is not None
            else '---')
    print(f'  Best        : {best}')
    print(f'  Workers     : {len(detail["workers"])}')
    print(f'  Nodes spent : {detail["nodes_spent"]}')
    print(f'  Tainted     : {"yes" if detail["tainted"] else "no"}')
    if detail['bundle_stats']:
        print(f'  Bundles     : {len(detail["bundle_stats"])}')
    if detail['republish']:
        print(f'  Republish   : {len(detail["republish"])} candidate(s)')
    if detail['finalize_log']:
        print(f'  Finalized   : {len(detail["finalize_log"])} logged run(s)')
    if args.claims and detail['claims']:
        print('  Claims:')
        for c in detail['claims']:
            status = 'done' if c['done'] else f'held by {c["claimed_by"] or "---"}'
            print(f'    {c["idx"]:5d} {status}')

def cmd_queue_status(args):
    """Show swarm queue coverage for a given word or partial path.

    For each of the 242 possible response patterns for WORD, reports the
    branch's status in the queue (pending_branches): pending, in_progress,
    done, or not yet queued at all.
    """
    from wordle_ui import fmt_pattern

    all_answers = load_word_list(ANSWER_FILE)
    try:
        tokens = parse_queue_branch_ref(getattr(args, 'branch_ref', None)
                                        or getattr(args, 'word', ''))
    except QueueRefError as e:
        print(f'Invalid branch ref: {e}')
        return
    if tokens[-1][1] is not None:
        # Fully-patterned refs are interpreted as "coverage at the last guess",
        # matching the old word/pattern mental model as closely as possible.
        path_tokens = tokens[:-1]
        word = tokens[-1][0].lower()
    else:
        path_tokens = tokens[:-1]
        word = tokens[-1][0].lower()

    sc = ScoreCache(args.cache, all_answers)
    rcache = ResponseCache(all_answers, sc)
    answer_branch = list(all_answers)
    for prev_word, prev_pat in path_tokens:
        if prev_pat is None:
            print('Invalid branch ref: a non-final word must include a pattern')
            sc.close()
            return
        from wordle_ui import parse_pattern
        answer_branch = rcache.group_words(
            prev_word.lower(), answer_branch).get(parse_pattern(prev_pat), [])
    groups = rcache.group_words(word, answer_branch)
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

    if getattr(args, 'json', False):
        _print_json({
            'word': word.upper(),
            'prefix': spine_prefix_sql(path_tokens) if path_tokens else '',
            'n_branches': len(branches),
            'n_trivial': n_trivial,
            'pending': pending,
            'in_progress': in_progress,
            'done': done,
            'unqueued': unqueued,
        })
        return

    print(f'{word.upper()}:  {len(branches)} branches with ≥2 answers  '
          f'({n_trivial} trivial patterns skipped)')
    print(f'  Pending    : {len(pending):4d}')
    print(f'  In progress: {len(in_progress):4d}')
    print(f'  Done       : {len(done):4d}')
    print(f'  Not queued : {len(unqueued):4d}')
    print()

    if in_progress and not getattr(args, 'missing_only', False):
        _COOP_PRI = 1_000_000
        print(f'{"Pattern":<8}  {"Ans":>4}  {"Pri":>4}  Claimed by')
        for pat, n, priority, claimed_by in in_progress:
            pri_str = 'COOP' if priority >= _COOP_PRI else str(priority)
            print(f'  {pat:<8}  {n:4d}  {pri_str:>4}  {claimed_by or "---"}')
        print()

    if pending and not getattr(args, 'missing_only', False):
        print(f'{"Pattern":<8}  {"Ans":>4}  {"Pri":>4}')
        for pat, n, priority in pending:
            print(f'  {pat:<8}  {n:4d}  {priority:4d}')
        print()

    if unqueued and not args.queued_only:
        print(f'{"Pattern":<8}  {"Ans":>4}  (not queued)')
        for pat, n in unqueued:
            print(f'  {pat:<8}  {n:4d}')


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
# legacy word/pattern queue inspection helper
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
    claims = queue.claims_for_branch(branch_key) if ab else []
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
        done_ct = sum(1 for c in claims if c['done'])
        best_g = ab['best_guess'] or '---'
        best_e = f'{ab["best_erd"]:.4f}' if ab['best_erd'] is not None else '---'
        print(f'  In-progress: claims {done_ct}/{n_candidates}  '
              f'best {best_g.upper()} {best_e}')
        print(f'  Claim detail:')
        for c in claims:
            holder = c['claimed_by'] or '(unclaimed)'
            status = 'done' if c['done'] else f'held by {holder}'
            print(f'    claim {c["idx"]:5d}: {status}')


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


# Supervisor disk-sample cadence for the status display's growth rate.
DISK_SAMPLE_SECONDS = 30
# Queue WAL size above which the supervisor quiesces workers to TRUNCATE.
# Below it, PASSIVE backfill each cycle is enough; the WAL file is reused once
# a TRUNCATE eventually wins, so a bounded WAL is not worth pausing for.
QUEUE_WAL_QUIESCE_BYTES = 2 * 1024 ** 3
# How long the supervisor retries a quiesced TRUNCATE before giving up until
# the next checkpoint cycle.
TRUNCATE_RETRY_SECONDS = 15
# Hard ceiling on the queue WAL.  In healthy operation the quiesce/TRUNCATE
# protocol keeps the WAL near QUEUE_WAL_QUIESCE_BYTES; a WAL an order of
# magnitude larger means TRUNCATE has been losing for many cycles and the file
# is on course to fill the disk — the failure mode that corrupts the database.
# At this size the supervisor stops trying to recover in place: it captures a
# diagnostic snapshot, signals every worker to dump its stacks, latches the
# swarm down (a manual `queue clear-disk-stop` is then required), and exits.
# Overridable via QUEUE_WAL_HARD_CEILING_GIB.
QUEUE_WAL_HARD_CEILING_BYTES = int(
    float(os.environ.get('QUEUE_WAL_HARD_CEILING_GIB', '32')) * 1024 ** 3)
# Fill/drain rates below this magnitude read as "steady": smaller than this
# and the trend is within statvfs sampling noise (page cache, unrelated
# processes on the same filesystem).  Anything at or above it is shown via
# _fmt_size, whose adaptive K/M/G unit keeps a reportable rate from ever
# rounding to a bare "0".
DISK_RATE_FLOOR_BYTES = 10_000   # 10 kB/s


def _disk_guard(queue, queue_path) -> bool:
    """True if the disk crossed DISK_STOP_FRACTION: the swarm must stop and
    latch down, reserving the remaining space for the rest of the OS."""
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
        # The startup live-fullness check still refuses to run, so the swarm
        # stays down even without the latch row.
        logger.critical('Could not write disk_stop latch: %s', exc)
    return True


def _supervisor_checkpoint(queue):
    """Periodic PASSIVE backfill: folds whatever it can into the main file
    without taking the writer lock, keeping the backfill debt small so a
    later quiesced TRUNCATE has little left to do."""
    queue.checkpoint('PASSIVE')


def _maybe_quiesce_truncate(queue):
    """Called on every supervisor pass: when the WAL has grown past
    QUEUE_WAL_QUIESCE_BYTES, quiesce workers and TRUNCATE.  The size test is
    a file stat, so checking each pass caps the sawtooth at roughly the
    threshold plus one pass of growth — not one full checkpoint cycle's.

    TRUNCATE needs a moment with no readers inside the WAL, which a busy swarm
    never yields naturally — so the checkpoint_pause flag asks workers to stay
    off the database (their compute continues) while TRUNCATE retries.  The
    flag self-expires; failure here is logged and retried next pass, never
    fatal."""
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
        'WAL attribution and worker stacks are in the erd_worker_*.log files.',
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
        at = datetime.fromtimestamp(latch['at']).isoformat(' ') \
            if latch.get('at') else 'unknown time'
        print(f'Refusing to start: disk-stop latch is set ({latch["reason"]}, '
              f'at {at}).\n'
              f'Free disk space, then clear it with: '
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

            # Backstop: free candidate claims held by any worker that died
            # WITHOUT being reaped above (e.g. it crashed and we haven't
            # noticed yet).  Gated on heartbeat liveness, so a slow-but-alive
            # worker is never reclaimed.
            freed = q.reclaim_stale_claims(args.worker_timeout_seconds)
            if freed:
                logger.info('Reclaimed %d stale candidate claim(s).', freed)
            counts = q.counts_by_status()
            in_flight = len(q.branches_in_progress())

            # Done when the queue holds no pending or in-progress branches and
            # no branch is still being swarmed.
            if (counts.get('pending', 0) == 0
                    and counts.get('in_progress', 0) == 0
                    and in_flight == 0
                    and counts):
                logger.info('Queue drained — all branches done.')
                print('\nQueue empty — all branches done.')
                stop_event.set()
        except sqlite3.OperationalError as exc:
            # Queue database writes failing (disk full, corruption) must stop
            # the swarm cleanly, not kill the supervisor with a traceback and
            # leave zombie workers behind.
            logger.critical('Queue database error in supervisor loop: %s — '
                            'stopping swarm.', exc)
            stop_event.set()

    logger.info('Supervisor stopping all workers...')
    for wid, (p, _) in procs.items():
        if p.is_alive():
            p.terminate()
    for wid, (p, _) in procs.items():
        p.join(timeout=30)
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
    across refreshes even as its other live display fields change.
    Collisions among the few dozen simultaneously-active branches are vanishingly
    unlikely over a 16-bit space.
    """
    return hashlib.sha1(bytes(branch_key)).hexdigest()[:4]


# Hotkey letters offered for branch drill-down, in display order.  'q' is the
# quit key and is skipped; digits stay reserved for worker selection.
_BRANCH_HOTKEYS = 'abcdefghijklmnoprstuvwxyz'


def _row_get(row, key, default=None):
    """Small adapter for sqlite3.Row and dict-like test rows."""
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return default
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _branch_display_sort_key(branch):
    """Stable topology key for branches that are new to the status display.

    This intentionally ignores volatile progress fields such as workers, percent
    done, ETA, best ERD, and updated timestamps.  Existing visible branches keep
    their sticky order; this key only decides the order among branches first seen
    on the same refresh.
    """
    spine = _row_get(branch, 'spine')
    source_word = (_row_get(branch, 'source_word') or '').upper()
    source_pattern = _row_get(branch, 'source_pattern')
    if spine:
        toks = spine.split()
        parent = ' '.join(toks[:-2])
        last_guess = toks[-2].upper() if len(toks) >= 2 else ''
        last_pattern = toks[-1] if len(toks) >= 2 else ''
        depth = len(toks) // 2
    else:
        parent = source_word
        last_guess = source_word
        last_pattern = '' if source_pattern is None else f'{source_pattern:03d}'
        depth = 1 if source_word and source_pattern is not None else 0
    return (parent, depth, last_pattern, last_guess,
            _branch_id(_row_get(branch, 'branch_key', b'')))


def _stable_branch_display_order(branches, previous_order):
    """Return branches in watch-stable display order plus the next order state."""
    by_bid = {_branch_id(_row_get(b, 'branch_key', b'')): b for b in branches}
    ordered_bids = []
    seen = set()

    for bid in previous_order or []:
        if bid in by_bid and bid not in seen:
            ordered_bids.append(bid)
            seen.add(bid)

    new_branches = [b for bid, b in by_bid.items() if bid not in seen]
    for b in sorted(new_branches, key=_branch_display_sort_key):
        bid = _branch_id(_row_get(b, 'branch_key', b''))
        ordered_bids.append(bid)
        seen.add(bid)

    return [by_bid[bid] for bid in ordered_bids], ordered_bids


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


def _worker_guess_depth_label(parent_guess_depth, current_max_guess_depth,
                              current_path, current_candidate):
    """Return the compact worker-row guess_depth label.

    Any rendered dN is an absolute guess_depth from the root.  Unknown live
    descent below a known branch is rendered as dN+?.
    """
    parsed_path = _parse_spine(current_path)
    explicit_guess_depths = [depth for depth, _guess, _pattern, _size in parsed_path
                             if depth is not None]
    if explicit_guess_depths:
        return f'd{max(explicit_guess_depths)}'

    if current_max_guess_depth and current_max_guess_depth > parent_guess_depth:
        return f'd{current_max_guess_depth}'

    if parsed_path:
        return f'd{parent_guess_depth + len(parsed_path)}'

    if current_candidate:
        return f'd{parent_guess_depth + 1}'

    return f'd{parent_guess_depth}+?'


def _sweep_progress_bar(n_candidates, done_indices, worker_positions,
                        width=40):
    """Render compressed candidate completion and worker positions.

    Each cell covers a range of candidate indices.  Unicode block heights show
    how much of the cell is done; worker labels overlay that conceptual
    completion map at their current claim positions.
    """
    if n_candidates <= 0 or width <= 0:
        return ''
    done_counts = [0] * width
    done_seen = set()
    for idx in done_indices or ():
        if idx is None or idx < 0:
            continue
        if idx in done_seen:
            continue
        done_seen.add(idx)
        pos = min(width - 1, int(width * idx / n_candidates))
        done_counts[pos] += 1
    ramp = ' ▁▂▃▄▅▆▇█'
    marks = [' '] * width
    for pos, done_count in enumerate(done_counts):
        if done_count == 0:
            continue
        start = (pos * n_candidates + width - 1) // width
        end = ((pos + 1) * n_candidates + width - 1) // width
        cell_size = end - start
        level = min(len(ramp) - 1,
                    (done_count * (len(ramp) - 1) + cell_size - 1) // cell_size)
        marks[pos] = ramp[level]
    for idx, label in worker_positions or ():
        if idx is None or idx < 0:
            continue
        pos = min(width - 1, int(width * idx / n_candidates))
        # Preserve multiple adjacent worker digits where possible, while still
        # allowing a worker to replace a progress marker in its approximate
        # bucket.
        while pos < width and marks[pos].isdigit():
            pos += 1
        if pos >= width:
            pos = width - 1
        marks[pos] = str(label)[-1]
    return ''.join(marks)


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


def _fmt_fill_eta(seconds):
    if seconds < 3600:
        return f'{seconds / 60:.0f} min'
    if seconds < 172800:
        return f'{seconds / 3600:.1f} h'
    return f'{seconds / 86400:.1f} d'


def _fmt_size(n_bytes):
    """Binary-unit size string matching df -h's numbers, with thousands
    separators so large byte counts stay readable.

    Also used for rates (bytes/s in, ".../s" appended by the caller) so the
    fill figure is in the same units as the WAL and fullness figures beside
    it, not decimal MB against their binary GiB/MiB.  The adaptive unit means
    a nonzero value never collapses to "0": it drops to the next smaller unit
    (a fixed-M formatter would round anything under ~0.5 MiB away — exactly
    the range slow fill rates live in).
    """
    gib = n_bytes / 2 ** 30
    if gib >= 100:
        return f'{gib:,.0f}G'
    if gib >= 1:
        return f'{gib:.1f}G'
    mib = n_bytes / 2 ** 20
    if mib >= 1:
        return f'{mib:,.0f}M'
    return f'{n_bytes / 2 ** 10:,.0f}K'


def _disk_fill_rate(samples, now):
    """Bytes/second the filesystem is filling (negative = freeing), by
    ordinary-least-squares slope of avail_bytes over fresh samples.

    Reducing the ring to an oldest-vs-newest secant, as this once did, meant
    the rate rode on whichever two samples happened to land at the ends of a
    ~10 min window, so one noisy statvfs reading (a concurrent checkpoint
    TRUNCATE, unrelated filesystem churn) swung the whole figure regardless
    of how many samples sat between.  Regressing over every fresh sample uses
    that history to average the noise out.  Returns None when there isn't
    enough fresh history to fit a slope; a stopped swarm has no fresh samples
    and so shows no rate rather than a stale one.
    """
    fresh = [s for s in samples if now - s[0] <= 900]
    if len(fresh) < 2:
        return None
    t0 = fresh[0][0]
    xs = [s[0] - t0 for s in fresh]
    ys = [s[1] for s in fresh]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x == 0:
        return None
    covariance_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance_xy / variance_x    # d(avail_bytes)/dt
    return -slope                         # positive = filling


def _disk_status_line(queue_path, samples):
    """One-line disk report: fullness (drawn red at DISK_WARN_FRACTION), the
    queue WAL size, and — when the supervisor's sample ring is fresh — the
    fill rate with the time remaining until the DISK_STOP_FRACTION stop
    threshold."""
    st = disk_stats(queue_path)
    capacity = st['used_bytes'] + st['avail_bytes']
    fullness = (f"{_fmt_size(st['used_bytes'])}/{_fmt_size(capacity)} "
                f"({100 * st['used_fraction']:.0f}%)")
    if st['used_fraction'] >= DISK_WARN_FRACTION:
        fullness = f'\033[31m{fullness}\033[0m'
    try:
        wal_bytes = os.path.getsize(f'{queue_path}-wal')
    except OSError:
        wal_bytes = 0
    line = f'Disk: {fullness}  queue WAL {_fmt_size(wal_bytes)}'

    rate = _disk_fill_rate(samples, time.time())
    if rate is not None:
        if rate > DISK_RATE_FLOOR_BYTES:
            avail_at_stop = (1 - DISK_STOP_FRACTION) * capacity
            eta = max(st['avail_bytes'] - avail_at_stop, 0) / rate
            line += (f'  filling {_fmt_size(rate)}/s: '
                     f'{100 * DISK_STOP_FRACTION:.0f}% in '
                     f'~{_fmt_fill_eta(eta)}')
        elif rate < -DISK_RATE_FLOOR_BYTES:
            line += f'  freeing {_fmt_size(-rate)}/s'
        else:
            line += '  steady'
    return line


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
        open_branch_keys = {bytes(b['branch_key']) for b in branches}
        hb_branch_keys = {
            bytes(h['current_branch_key'])
            for h in hbs
            if h['current_branch_key']
        }
        finalizing_branch_rows = queue.active_branches_by_keys(
            list(hb_branch_keys - open_branch_keys))
        detail_branches = branches + list(finalizing_branch_rows.values())
        # Per-branch done-candidate counts for branches the status display can
        # inspect, including finalized rows still referenced by worker heartbeats.
        done_candidates = {
            bytes(b['branch_key']): queue.branch_done_candidates(b['branch_key'])
            for b in detail_branches
        }
        selected_done_indices = {}
        if selected_branch is not None:
            for b in detail_branches:
                bkey = bytes(b['branch_key'])
                if _branch_id(bkey) == selected_branch:
                    selected_done_indices[bkey] = [
                        c['idx'] for c in queue.claims_for_branch(bkey)
                        if c['done']
                    ]
                    break
        disk_sample_ring = queue.disk_samples()
        queue.close()
        queue_ok = True
    except Exception as e:
        print(f'Queue unavailable: {e}')
        queue_ok = False
        disk_sample_ring = []
        counts = {}
        branches = []
        detail_branches = []
        finalizing_branch_rows = {}
        hbs = []
        worker_counts = {}
        done_candidates = {}
        selected_done_indices = {}

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
    live = [h for h in hbs if now_ts - h['updated_at'] <= WORKER_LIVENESS_SECONDS]
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
    print(_disk_status_line(args.queue, disk_sample_ring))
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
                          for b in detail_branches}

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

    def _print_worker_row(h, parent_guess_depth):
        # One nested row, kept within 59 cols: worker, sweep index, current
        # candidate, absolute guess_depth reached, node rate, then the descent
        # sizes truncated to whatever width remains.
        age = now_ts - h['updated_at']
        flag = ' !!' if age > WORKER_LIVENESS_SECONDS else ''
        claim_idx_str = str(h['claim_idx']) if h['claim_idx'] is not None else '-'
        cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
        current_max_guess_depth = (
            h['cur_max_depth'] if 'cur_max_depth' in h.keys() else None) or 0
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
        guess_depth_label = _worker_guess_depth_label(
            parent_guess_depth, current_max_guess_depth, path, cur)
        head = (f' W{_worker_num(h):<2} {claim_idx_str:>5} {cur_disp:<6} '
                f'{guess_depth_label} {krate:>6}')
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
    active, branch_display_order = _stable_branch_display_order(
        active, getattr(_print_status, '_branch_display_order', []))
    _print_status._branch_display_order = branch_display_order
    branch_keys = {bytes(b['branch_key']) for b in branches}
    detached = [h for k, hs in workers_by_branch.items() if k not in branch_keys
                for h in hs]
    if not active:
        print('(no branches being worked)')
    # Stable per-branch hotkey letters: a branch keeps its letter across refreshes,
    # and a selected branch keeps it even after it finalizes, so its detail panel's
    # "press X to dismiss" stays valid and letters don't shuffle under the user.
    prev_letter_by_bid = getattr(_print_status, '_branch_letter_by_bid', {})
    snapshots = getattr(_print_status, '_branch_detail_snapshots', {})
    keep_bids = [_branch_id(bytes(b['branch_key'])) for b in active]
    for h in detached:
        if h['current_branch_key']:
            bkey = bytes(h['current_branch_key'])
            bid = _branch_id(bkey)
            if (bkey in finalizing_branch_rows or bid in snapshots) and bid not in keep_bids:
                keep_bids.append(bid)
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
        done = done_candidates.get(key, 0)
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
            _print_worker_row(h, guess_depth)

    # Workers attached to a branch no longer in the in-progress list (just
    # finalized) or idle between claims.
    if idle_workers or detached:
        print()
        for h in sorted(idle_workers, key=lambda r: _worker_num(r)):
            age = now_ts - h['updated_at']
            flag = ' !!' if age > WORKER_LIVENESS_SECONDS else ''
            print(f'W{_worker_num(h):<2} (idle)  {age}s{flag}')
        for h in sorted(detached, key=lambda r: _worker_num(r)):
            age = now_ts - h['updated_at']
            bkey = bytes(h['current_branch_key']) if h['current_branch_key'] else None
            bid = _branch_id(bkey) if bkey else '????'
            letter = letter_by_bid.get(bid, ' ')
            prefix = f'[{letter}] ' if letter != ' ' else ''
            print(f'W{_worker_num(h):<2} {prefix}(branch #{bid} finalizing)  {age}s')

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
            bid = _branch_id(bkey) if bkey else None
            letter = letter_by_bid.get(bid) if bid else None
            branch_ref = f'[{letter}] #{bid}' if letter else f'#{bid or "?"}'
            print(f'Worker {selected_worker}: branch {branch_ref}, depth {base_k}')
            # Path to the worker's branch: its absolute spine, one guess per line
            # (d1..dK).  The first guess is d1; the bare root is d0 with no guess.
            branch_info = next((b for b in detail_branches
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
            # Live descent below the claimed branch (d{base_k+1}..).
            # New-format heartbeats carry an explicit guess_depth on each token;
            # filter out outer-frame entries (depth <= base_k) and use the stored
            # depth directly as the label.  Old-format heartbeats (depth=None) fall
            # back to the spine-edge heuristic: drop the leading entry if its
            # guess+pattern matches the last edge of the stored branch spine.
            descent_all = [(d, g, p, s) for (d, g, p, s) in _parse_spine(rich_path) if g and p]
            if descent_all and descent_all[0][0] is not None:
                descent = [(d, g, p, s) for (d, g, p, s) in descent_all if d > base_k]
            else:
                descent = descent_all
                if descent and branch_spine:
                    spine_tokens = branch_spine.split()
                    if len(spine_tokens) >= 2:
                        last_guess = spine_tokens[-2].upper()
                        last_pattern = spine_tokens[-1]
                        if descent[0][1].upper() == last_guess and descent[0][2] == last_pattern:
                            descent = descent[1:]
            if descent:
                for i, (depth, guess, pattern, size) in enumerate(descent, start=base_k + 1):
                    di = depth if depth is not None else i
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
                    branch_info = next((b for b in detail_branches
                                        if bytes(b['branch_key']) == bkey), None)
                    if branch_info:
                        n_cands = branch_info['n_candidates'] or 0
                        total_candidates = n_cands
                        done_ct = done_candidates.get(bkey, 0)
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
                        print(f'(cooperating — {done_ct}/{total_candidates} candidates done{co_str})')
                    else:
                        print('(cooperating — sub-branch finalizing)')
                else:
                    print('(between candidates)')
            else:
                print('(no spine data yet)')
            if interactive:
                print(f'[press {selected_worker} to dismiss]')

    if selected_branch is not None:
        _section_break('branchdetail', interactive)
        selected_branch_letter = next(
            (lt for lt, bid
             in getattr(_print_status, '_branch_hotkeys', {}).items()
             if bid == selected_branch), None)
        selected_branch_ref = (f'[{selected_branch_letter}] #{selected_branch}'
                               if selected_branch_letter
                               else f'#{selected_branch}')
        target = next((b for b in detail_branches
                       if _branch_id(b['branch_key']) == selected_branch), None)
        print()
        if target is None:
            snap = snapshots.get(selected_branch)
            if snap is None:
                print(f'Branch {selected_branch_ref} not found')
            else:
                key = snap['branch_key']
                n_cands = snap['n_candidates']
                done = max(snap['done'], done_candidates.get(key, 0))
                guess_depth = snap['guess_depth']
                workers_here = sorted(
                    [h for h in hbs if h['current_branch_key']
                     and bytes(h['current_branch_key']) == key],
                    key=lambda r: (r['claim_idx'] if r['claim_idx'] is not None else -1))
                state = 'finalizing' if workers_here else 'finished'
                print(f'Branch {selected_branch_ref}: {snap["n_words"]} words: '
                      f'depth {guess_depth}  {state}')
                full = _fmt_spine_path(snap['spine'])
                if full:
                    for di, guess_step in enumerate(full.split(' ▸ '), start=1):
                        parts = guess_step.split()
                        word = parts[0] if parts else ''
                        rest = ' '.join(parts[1:])
                        star = '*' if word.lower() in answer_set else ' '
                        print(f'{f"d{di}":<4} {word}{star} {rest}')
                else:
                    src = (f'{snap["source_word"].upper()} '
                           f'{fmt_pattern(snap["source_pattern"])}'
                           if snap['source_word'] and snap['source_pattern'] is not None
                           else '?????')
                    print(f'{"d1":<4} {src}')
                    if guess_depth > 1:
                        print(f'd2..d{guess_depth}  (intermediate guesses not recorded)')
                best_disp = (f'{(snap["best_guess"] or "-----").upper()} '
                             f'{snap["best_erd"]:.3f}'
                             if snap['best_erd'] is not None else 'none yet')
                print(f'{f"d{guess_depth + 1}":<4} sweep {done}/{n_cands}  '
                      f'best {best_disp}')
                for h in workers_here:
                    path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
                    path = path.replace('>', '→')
                    cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
                    cur_disp = cur.upper() if cur else '-----'
                    print(f'W{_worker_num(h):<2} idx {h["claim_idx"]}  {cur_disp:<6} '
                          f'{_spine_sizes(path)}')
            if interactive:
                # The branch's letter is held reserved while it stays selected, so
                # re-pressing it still toggles the (now finalized) panel closed.
                if selected_branch_letter:
                    print(f'[press {selected_branch_letter} to dismiss]')
        else:
            key = bytes(target['branch_key'])
            n_cands = target['n_candidates'] or 0
            done = done_candidates.get(key, 0)
            guess_depth = branch_guess_depth.get(key, 0)
            snapshots[selected_branch] = {
                'branch_key': key,
                'n_words': target['n_words'] or 0,
                'n_candidates': n_cands,
                'done': done,
                'guess_depth': guess_depth,
                'spine': target['spine'] if 'spine' in target.keys() else None,
                'source_word': target['source_word'],
                'source_pattern': target['source_pattern'],
                'best_guess': target['best_guess'],
                'best_erd': target['best_erd'],
            }
            _print_status._branch_detail_snapshots = snapshots
            print(f'Branch {selected_branch_ref}: {target["n_words"] or 0} words: '
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
            if n_cands and (workers_here or selected_done_indices.get(key)):
                worker_positions = [
                    (h['claim_idx'], _worker_num(h)) for h in workers_here
                ]
                bar = _sweep_progress_bar(
                    n_cands, selected_done_indices.get(key, ()),
                    worker_positions)
                print(f'[{bar}]')
            # Each cooperating worker's downward exploration spine, stacked.
            for h in workers_here:
                path = (h['cur_path'] if 'cur_path' in h.keys() else None) or ''
                path = path.replace('>', '→')
                cur = (h['cur_candidate'] if 'cur_candidate' in h.keys() else None) or ''
                cur_disp = cur.upper() if cur else '-----'
                print(f'W{_worker_num(h):<2} idx {h["claim_idx"]}  {cur_disp:<6} '
                      f'{_spine_sizes(path)}')
            if interactive:
                if selected_branch_letter:
                    print(f'[press {selected_branch_letter} to dismiss]')


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
    queue = ERDQueue(args.queue)
    n = queue.reset_stale_in_progress()
    queue.close()
    print(f'Reset {n} in_progress row(s) to pending.')


def _normalize_queue_cli_args(args):
    """Apply queue-level options to nested queue commands.

    Nested parsers use SUPPRESS defaults for duplicated options so they do not
    overwrite queue-level values parsed before the subcommand.
    """
    if args.cmd != 'queue':
        return
    if not hasattr(args, 'queue'):
        args.queue = args.queue_path or DEFAULT_QUEUE
    if not hasattr(args, 'json'):
        args.json = bool(args.queue_json)
    if not hasattr(args, 'limit'):
        if args.queue_limit is not None:
            args.limit = args.queue_limit
        elif args.queue_cmd is None:
            args.limit = 8
        elif args.queue_cmd == 'top':
            args.limit = 10
        else:
            args.limit = None


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
    p_view.add_argument('spine', nargs='*', metavar='SPINE')

    # -- cache-status --
    p_cs = sub.add_parser('cache-status',
                           help='Show ERD cache coverage for a word')
    p_cs.add_argument('--word', required=True, metavar='WORD',
                      help='Guess word to inspect (e.g. salet)')
    p_cs.add_argument('--missing-only', action='store_true',
                      help='Only list patterns whose branches are not yet cached')
    p_cs.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')

    # -- queue --
    p_queue = sub.add_parser('queue', help='Inspect and manage queue work')
    p_queue.add_argument('--queue', dest='queue_path', default=None, metavar='PATH')
    p_queue.add_argument('--limit', dest='queue_limit', type=int, default=None,
                         metavar='N')
    p_queue.add_argument('--json', dest='queue_json', action='store_true')
    qsub = p_queue.add_subparsers(dest='queue_cmd')

    p_qls = qsub.add_parser('ls', help='List queue rows')
    p_qls.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qls.add_argument('--status', choices=['pending', 'in_progress', 'done', 'open'])
    p_qls.add_argument('--min-words', type=int, metavar='N')
    p_qls.add_argument('--max-words', type=int, metavar='N')
    p_qls.add_argument('--budget', type=int, metavar='N')
    p_qls.add_argument('--priority', type=int, metavar='N')
    p_qls.add_argument('--source-word', metavar='WORD')
    p_qls.add_argument('--prefix', metavar='SPINE')
    p_qls.add_argument('--limit', type=int, default=argparse.SUPPRESS, metavar='N')
    p_qls.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

    p_qtree = qsub.add_parser('tree', help='Show queue rows grouped by spine')
    p_qtree.add_argument('prefix', nargs='*', metavar='SPINE')
    p_qtree.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qtree.add_argument('--active-only', action='store_true')
    p_qtree.add_argument('--max-depth', type=int, metavar='N')
    p_qtree.add_argument('--limit', type=int, default=argparse.SUPPRESS, metavar='N')
    p_qtree.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

    p_qshow = qsub.add_parser('show', help='Show detail for a branch')
    p_qshow.add_argument('branch_ref', nargs='+', metavar='BRANCH_REF')
    p_qshow.add_argument('--claims', action='store_true')
    p_qshow.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qshow.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qshow.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

    p_qsum = qsub.add_parser('summary', help='Show aggregate queue counts')
    p_qsum.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qsum.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

    p_qtop = qsub.add_parser('top', help='Show queue hotspots')
    p_qtop.add_argument('prefix', nargs='*', metavar='SPINE')
    p_qtop.add_argument('--by', choices=['nodes', 'age', 'size', 'workers',
                                         'priority', 'slowest'],
                        default='nodes')
    p_qtop.add_argument('--limit', type=int, default=argparse.SUPPRESS, metavar='N')
    p_qtop.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qtop.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

    p_qcov = qsub.add_parser('coverage',
                             help='Show swarm queue coverage for a spine prefix')
    p_qcov.add_argument('branch_ref', nargs='+', metavar='BRANCH_REF')
    p_qcov.add_argument('--queued-only', action='store_true',
                        help='Only list patterns that are pending or in progress')
    p_qcov.add_argument('--missing-only', action='store_true',
                        help='Only list patterns not yet queued')
    p_qcov.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH')
    p_qcov.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')
    p_qcov.add_argument('--json', action='store_true', default=argparse.SUPPRESS)

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

    p_cds = qsub.add_parser('clear-disk-stop',
                            help='Release the disk-stop latch so run can start')
    p_cds.add_argument('--queue', default=argparse.SUPPRESS, metavar='PATH')

    args = parser.parse_args()
    if args.cmd == 'view' and args.format == 'json' and args.watch is not None:
        parser.error('--format json cannot be used with --watch; use jsonl')
    if args.cmd == 'view':
        try:
            args.selector = parse_report_selector(args.spine)
        except ValueError as error:
            parser.error(str(error))
        if args.tree and args.view_cache:
            parser.error('--tree cannot be used with --cache')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be at least 1')
        if (args.minimum_answer_count is not None
                and args.maximum_answer_count is not None
                and args.minimum_answer_count > args.maximum_answer_count):
            parser.error('--minimum-answer-count cannot exceed --maximum-answer-count')
        args.report_kind = (
            'queue' if args.view_queue else
            'workers' if args.workers or args.worker is not None else
            'cache' if args.view_cache else 'auto'
        )
        if args.claims and (
                args.tree or args.report_kind != 'auto'
                or args.selector.kind not in ('branch', 'branch_reference')):
            parser.error('--claims requires a singular branch selector')
        if args.answers and (
                args.tree or args.report_kind in ('queue', 'workers')
                or (args.report_kind == 'auto' and args.selector.kind == 'root')):
            parser.error('--answers requires a word or branch report without --tree')
        if (args.report_kind == 'auto' and args.selector.kind == 'word'
                and args.sort is not None
                and args.sort not in ('default', 'size', 'workers', 'priority')):
            parser.error(
                '--sort for word reports must be default, size, workers, or priority'
            )
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
    _normalize_queue_cli_args(args)

    if args.cmd == 'queue':
        qdispatch = {
            None: cmd_queue_dashboard,
            'ls': cmd_queue_ls,
            'tree': cmd_queue_tree,
            'show': cmd_queue_show,
            'summary': cmd_queue_summary,
            'top': cmd_queue_top,
            'coverage': cmd_queue_status,
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
        'cache-status': cmd_cache_status,
        'start': cmd_start,
        'stop': cmd_stop,
        'restart': cmd_restart,
        'run': cmd_run,
        'status': cmd_status,
        'view': cmd_view,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
