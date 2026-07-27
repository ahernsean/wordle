#!/usr/bin/env python3.13
"""export_cache.py — Create a trimmed iPhone-ready snapshot of the cache.

Usage
-----
  python3.13 export_cache.py [--cache PATH] [--output PATH]

Creates a trimmed export file with only the iPhone-useful tables:
answer_list, response_decomposition, branch_best_by_policy, and
candidate_scores. A phone without a cached ERD result for its current
position still needs candidate_scores' entropy/max-group-size numbers to
rank candidates, and that need isn't limited to the opening guess, so the
whole table is carried, not just one position's rows.

Safe to run while workers are active: WAL mode allows concurrent reads, so
the export sees a consistent snapshot without stopping anything. Re-running
is incremental: INSERT OR IGNORE skips rows already present, so you can
refresh the export file at any time.

--since UNIXTIME restricts every table with an updated_at column to rows
newer than that watermark, for a delta export. answer_list has no
updated_at (it's one tiny row per answer list, the namespace key), so it is
always copied in full regardless of --since.

Import the result into another cache with import_cache.py.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3

from runtime_paths import DEFAULT_CACHE_PATH

DEFAULT_CACHE = DEFAULT_CACHE_PATH
DEFAULT_EXPORT = 'wordle_erd_export.sqlite3'

EXPORT_TABLES = ['answer_list', 'response_decomposition',
                  'branch_best_by_policy', 'candidate_scores']


def cmd_export(args):
    export_path = args.output or DEFAULT_EXPORT
    cache_path = os.path.abspath(args.cache)
    export_path = os.path.abspath(export_path)

    print(f'Source : {cache_path}')
    print(f'Export : {export_path}')
    if args.since is not None:
        print(f'Since  : {args.since}')
    print()

    conn = sqlite3.connect(export_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
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
                except sqlite3.OperationalError:  # pragma: no cover
                    # IF NOT EXISTS already suppresses a same-name collision;
                    # this only guards a source schema too stale for the
                    # table just created above (a state that can't arise
                    # from a real cache, which is always current).
                    pass

            cols = [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]
            col_list = ', '.join(cols)
            if args.since is not None and 'updated_at' in cols:
                conn.execute(f"""
                    INSERT OR IGNORE INTO main.{table} ({col_list})
                    SELECT {col_list} FROM src.{table}
                    WHERE updated_at > ?
                """, (args.since,))
            else:
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
        except Exception:  # pragma: no cover — only if conn is already dead
            pass
        raise
    finally:
        try:
            conn.execute('DETACH DATABASE src')
        except Exception:  # pragma: no cover — only if conn is already dead
            pass
        conn.close()


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', default=DEFAULT_CACHE, metavar='PATH',
                        help=f'Source cache (default: {DEFAULT_CACHE})')
    parser.add_argument('--output', default=DEFAULT_EXPORT, metavar='PATH',
                        help=f'Output file (default: {DEFAULT_EXPORT})')
    parser.add_argument('--since', type=int, default=None, metavar='UNIXTIME',
                        help='Only copy rows updated after UNIXTIME '
                             '(default: full export)')
    args = parser.parse_args()
    cmd_export(args)


if __name__ == '__main__':
    main()
