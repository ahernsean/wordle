#!/usr/bin/env python3.13
"""merge_cache.py — Merge a source wordle_cache.sqlite3 into the local one.

Usage
-----
  python3.13 merge_cache.py <source_db> [--target PATH] [--dry-run]

Adds rows from <source_db> not already present in --target across all four
cache tables.  Three of them (answer_list, response_decomposition, candidate_scores)
are deterministic given the same answer-word universe — matching keys imply
identical values — so INSERT OR IGNORE is exact.

branch_best_by_policy is the exception: its primary key
(branch_key, policy, answer_list_id) does NOT include solve_budget, so two caches
can legitimately hold DIFFERENT entries for the same key — e.g. one solved
unconstrained (solve_budget NULL, reusable at any budget) and one solved under
a depth cap (solve_budget set, reusable only at that budget).  The unconstrained
entry strictly dominates (see wordle_engine._cache_reuse), so for that table we
let an incoming untainted row replace an existing tainted one, rather than
silently keeping whichever the target happened to have first.

Run with --dry-run first to see how many rows would be added/upgraded.  Prefer
to run while workers are stopped (or at least briefly paused) to avoid competing
for the SQLite write lock, though the 30s timeout makes concurrent use safe.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

BATCH = 20000   # rows per progress step / commit

TABLES = [
    'answer_list',
    'response_decomposition',
    'branch_best_by_policy',
    'candidate_scores',
]

DEFAULT_TARGET = 'wordle_cache.sqlite3'

# Columns the untainted-preference UPSERT for branch_best_by_policy needs.
_ERD_UPSERT_COLS = {'branch_key', 'policy', 'answer_list_id', 'solve_budget',
                    'best_guess', 'best_score', 'updated_at', 'max_depth'}


def _insert_sql(table: str, cols: list[str]) -> str:
    """INSERT statement for a merge batch.

    For branch_best_by_policy, a key collision can pit a tainted entry against
    an untainted one (the PK omits solve_budget); the untainted entry is
    strictly more reusable, so an incoming untainted row replaces an existing
    tainted one.  Every other table — and any conflict that isn't
    tainted->untainted — is a plain INSERT OR IGNORE (keep what's there).
    """
    col_list = ', '.join(cols)
    placeholders = ', '.join('?' * len(cols))
    if table == 'branch_best_by_policy' and _ERD_UPSERT_COLS <= set(cols):
        return f"""
            INSERT INTO main.{table} ({col_list}) VALUES ({placeholders})
            ON CONFLICT(branch_key, policy, answer_list_id) DO UPDATE SET
                best_guess   = excluded.best_guess,
                best_score   = excluded.best_score,
                updated_at   = excluded.updated_at,
                max_depth    = excluded.max_depth,
                solve_budget = excluded.solve_budget
            WHERE main.{table}.solve_budget IS NOT NULL
              AND excluded.solve_budget IS NULL
        """
    return (f'INSERT OR IGNORE INTO main.{table} ({col_list}) '
            f'VALUES ({placeholders})')


def _pk_cols(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')
            if r[5] > 0]


def _all_cols(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'


def _copy_table_with_progress(conn, table) -> int:
    """Copy src.table -> main.table in rowid-paged batches with live progress.

    INSERT OR IGNORE makes each batch idempotent, and each batch commits on
    its own, so an interrupted merge can simply be re-run to resume.  Paging
    by rowid (rather than one opaque INSERT...SELECT of millions of rows) is
    what lets us show a percentage and ETA instead of an apparent hang.
    """
    cols = _all_cols(conn, table)
    col_list = ', '.join(cols)
    insert_sql = _insert_sql(table, cols)
    total = conn.execute(f'SELECT COUNT(*) FROM src.{table}').fetchone()[0]
    if total == 0:
        print(f'  {table}: source empty, skipping')
        return 0

    inserted = scanned = 0
    last_rowid = 0
    start = time.time()
    last_print = 0.0
    while True:
        rows = conn.execute(
            f'SELECT rowid, {col_list} FROM src.{table} '
            f'WHERE rowid > ? ORDER BY rowid LIMIT ?',
            (last_rowid, BATCH)).fetchall()
        if not rows:
            break
        last_rowid = rows[-1][0]
        before = conn.total_changes
        conn.execute('BEGIN')
        conn.executemany(insert_sql, [tuple(r[1:]) for r in rows])
        conn.execute('COMMIT')
        inserted += conn.total_changes - before
        scanned += len(rows)

        now = time.time()
        if now - last_print > 0.5 or scanned >= total:
            el = now - start
            rate = scanned / el if el > 0 else 0
            eta = (total - scanned) / rate if rate > 0 else 0
            print(f'\r  {table}: {scanned:,}/{total:,} '
                  f'({100 * scanned / total:3.0f}%) scanned, '
                  f'{inserted:,} new, {rate:,.0f}/s, ETA {_fmt_eta(eta)}    ',
                  end='', flush=True)
            last_print = now
    print()
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='?', default='wordle_erd_export.sqlite3',
                        help='Source cache file (default: wordle_erd_export.sqlite3)')
    parser.add_argument('--target', default=DEFAULT_TARGET, metavar='PATH',
                        help=f'Target cache file (default: {DEFAULT_TARGET})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report counts without writing anything')
    parser.add_argument('--delete-source', action='store_true',
                        help='Delete source file after successful merge')
    args = parser.parse_args()

    conn = sqlite3.connect(args.target, timeout=30.0, isolation_level=None)
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute(f"ATTACH DATABASE '{args.source}' AS src")

        src_tables = {r[0] for r in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'")}

        total_new = 0
        for table in TABLES:
            if table not in src_tables:
                print(f'  {table}: not in source, skipping')
                continue

            if args.dry_run:
                pks = _pk_cols(conn, table)
                if pks:
                    join = ' AND '.join(
                        f'main.{table}.{c} = src.{table}.{c}' for c in pks)
                    n = conn.execute(f"""
                        SELECT COUNT(*) FROM src.{table}
                        WHERE NOT EXISTS (
                            SELECT 1 FROM main.{table} WHERE {join}
                        )
                    """).fetchone()[0]
                else:
                    n = conn.execute(
                        f'SELECT COUNT(*) FROM src.{table}').fetchone()[0]
                msg = f'  {table}: {n:,} rows would be inserted'
                # Account for the tainted->untainted upgrades INSERT OR IGNORE
                # would silently miss (and the real merge now performs).
                if (table == 'branch_best_by_policy'
                        and 'solve_budget' in _all_cols(conn, table)):
                    up = conn.execute(f"""
                        SELECT COUNT(*) FROM src.{table} s
                        JOIN main.{table} m
                          ON m.branch_key     = s.branch_key
                         AND m.policy         = s.policy
                         AND m.answer_list_id = s.answer_list_id
                        WHERE m.solve_budget IS NOT NULL
                          AND s.solve_budget IS NULL
                    """).fetchone()[0]
                    if up:
                        msg += f' (+{up:,} tainted->untainted upgrades)'
                print(msg)
            else:
                n = _copy_table_with_progress(conn, table)

            total_new += n

        if args.dry_run:
            print(f'\nTotal: {total_new:,} rows would be inserted '
                  f'(dry run — no changes made)')
        else:
            print('  checkpointing...', flush=True)
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            print(f'Total: {total_new:,} rows inserted')
            if args.delete_source:
                import os
                for suffix in ('', '-shm', '-wal'):
                    p = args.source + suffix
                    if os.path.exists(p):
                        os.remove(p)
                print(f'Deleted {args.source}')

    except Exception as e:
        try:
            conn.execute('ROLLBACK')
        except Exception:
            pass
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            conn.execute('DETACH DATABASE src')
        except Exception:
            pass
        conn.close()


if __name__ == '__main__':
    main()
