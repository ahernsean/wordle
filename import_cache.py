#!/usr/bin/env python3.13
"""import_cache.py — Import a source wordle_cache.sqlite3 into runtime/wordle_cache.sqlite3.

Usage
-----
  python3.13 import_cache.py <source_db> [--target PATH] [--dry-run]
                             [--keep-source]

Creates the target cache if it doesn't exist yet, or merges into it if it
already does — adding rows from <source_db> not already present in
--target across all five cache tables. After a successful merge the source
file (and its -wal/-shm siblings) is deleted; pass --keep-source to retain
it. The source is a transitory export snapshot, so an import that leaves
nothing behind is the normal end of the export/import cycle. Four of them
(answer_list, response_decomposition, candidate_scores,
candidate_erd_by_policy) are deterministic given the same answer-word
universe — matching keys imply identical values — so INSERT OR IGNORE is
exact.

A missing target restores a working cache rather than erroring: before any
row is merged, the target is opened once through ScoreCache itself (the
same construction path every normal cache open uses), so a target built
here is schema-identical to one that has always been managed by
ScoreCache — a table under a pre-rename legacy name gets migrated in
place, schema_migrations is populated, and all of this happens before the
target holds any of the merged rows, not after. Merging into an
already-current target is a fast no-op bootstrap.

A branch's two kinds of exact result are keyed apart — the unrestricted
optimum in branch_best_by_policy, one per budget in
branch_best_by_policy_and_budget — so a collision in either table is two caches
holding the same fact, and INSERT OR IGNORE is exact there too.

A source written before that split carries its budget-specific results in
branch_best_by_policy, marked by a non-NULL solve_budget.  Merging those into
the target's canonical table would put two different facts back on one key, so
they are routed into the budget table under their own budget and skipped by the
canonical copy.  An older reader handed a newer export sees a table it does not
know and ignores it, keeping the unrestricted rows it does understand.

A key collision in either exact-result table is normally one fact reached
twice, but that is checked rather than assumed: before any row moves, colliding
rows that are not the same certificate are reported and the merge refuses.
That preflight describes the target as it was when the import started, so each
batch re-checks inside its own write transaction — a live worker can store a
disagreeing result after a clean scan, and INSERT OR IGNORE would drop the
source's row without a word.  Two exact
results for one scope cannot both be right, and keeping either displaces
whichever the other cache's ancestors folded.  --dry-run reports them and
continues.

Run with --dry-run first to see how many rows would be added/upgraded.  Prefer
to run while workers are stopped (or at least briefly paused) to avoid competing
for the SQLite write lock, though the 30s timeout makes concurrent use safe.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

from cache_sqlite import (EXACT_SCORE_TOLERANCE, ScoreCache,
                          branch_reference, exact_results_agree)
from wordle_engine import load_word_list

from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CACHE_PATH, ensure_runtime_dir

ANSWER_FILE = DEFAULT_ANSWER_LIST_PATH
BATCH = 20000   # rows per progress step / commit

TABLES = [
    'answer_list',
    'response_decomposition',
    'branch_best_by_policy',
    'branch_best_by_policy_and_budget',
    'candidate_scores',
    'candidate_erd_by_policy',
]

# A branch's budget-specific results live in their own table.  A source written
# before that split carries them in branch_best_by_policy instead, marked by a
# non-NULL solve_budget; they are routed to the budget table and never merged
# into the canonical one, whose rows are unrestricted optima.
CANONICAL_TABLE = 'branch_best_by_policy'
BUDGET_TABLE = 'branch_best_by_policy_and_budget'

DEFAULT_TARGET = DEFAULT_CACHE_PATH

# Columns the legacy budget-specific routing needs.
_ERD_ROUTING_COLS = {'branch_key', 'policy', 'answer_list_id', 'solve_budget',
                     'best_guess', 'best_score', 'updated_at', 'max_depth'}


def _insert_sql(table: str, cols: list[str]) -> str:
    """INSERT statement for a merge batch: keep whatever the target holds.

    Every table is a plain INSERT OR IGNORE.  Both branch tables now key a row
    by the scope it belongs to, so a collision is two caches holding the same
    fact rather than two different facts competing, and there is nothing to
    prefer between them.
    """
    col_list = ', '.join(cols)
    placeholders = ', '.join('?' * len(cols))
    return (f'INSERT OR IGNORE INTO main.{table} ({col_list}) '
            f'VALUES ({placeholders})')


def _target_has_table(conn, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchone() is not None


def _droppable_indexes(conn, table: str):
    """[(name, create_sql), ...] for main.{table}'s explicit, non-PK indexes.

    An index backing a PRIMARY KEY constraint has no `sql` (SQLite tracks it
    as an automatic index), so filtering on `sql IS NOT NULL` naturally
    excludes it — dropping it would break the ON CONFLICT/INSERT OR IGNORE
    the merge relies on for correctness.
    """
    return conn.execute(
        "SELECT name, sql FROM main.sqlite_master "
        "WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
        (table,)).fetchall()


def _bootstrap_target_schema(target_path: str) -> None:
    """Ensure target_path has a fully current, migrated schema.

    Opens it once through ScoreCache itself -- the same construction path
    every normal cache open uses -- before any row is merged into it. A
    table hiding under a pre-rename legacy name (word_scores, universe,
    subgroup_best_by_policy) is migrated to its current name here; without
    this, a later ScoreCache open would find that legacy table still
    present, "helpfully" migrate over it, and drop everything a merge had
    just written into the current-named table. A no-op on an
    already-current target.
    """
    ScoreCache(target_path, load_word_list(ANSWER_FILE)).close()


def _merge_table(conn, table: str) -> int:
    """Merge src.table into main.table, returning rows inserted/upgraded.

    Defers a from-scratch table's secondary indexes until after the bulk
    copy (dropped before, recreated after) so a multi-million-row disaster
    recovery doesn't pay incremental B-tree maintenance on every insert;
    an incremental merge into an already-populated table is untouched.
    """
    row_count = conn.execute(f'SELECT COUNT(*) FROM main.{table}').fetchone()[0]
    deferred_indexes = []
    if row_count == 0:
        deferred_indexes = _droppable_indexes(conn, table)
        for name, _ in deferred_indexes:
            conn.execute(f'DROP INDEX IF EXISTS {name}')
    n = _copy_table_with_progress(conn, table)
    for _, create_sql in deferred_indexes:
        conn.execute(create_sql)
    return n


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


class ImportConflict(Exception):
    """A colliding exact result was found while the merge was under way.

    The preflight catches what is already there when the import starts; this
    is the one a live writer introduced after that scan.  Batches that already
    committed stand — INSERT OR IGNORE makes them idempotent, so re-running
    after the disagreement is resolved resumes rather than duplicates.
    """


def _legacy_budget_rows_predicate(conn) -> str:
    """SQL restricting src.branch_best_by_policy to its budget-specific rows.

    A source written after the split has none — its canonical table holds only
    unrestricted optima — and one written before it marks them with a non-NULL
    solve_budget.  A source old enough to lack the column has none either.
    """
    if 'solve_budget' not in _all_cols_src(conn, CANONICAL_TABLE):
        return '0'
    return 's.solve_budget IS NOT NULL'


def _all_cols_src(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA src.table_info({table})')]


def _route_legacy_budget_rows(conn) -> int:
    """Move a legacy source's budget-specific rows into the budget table.

    They arrive in the canonical table because they predate the split.  Merging
    them there would put two different facts back on one key, which is the
    thing the split exists to stop, so they are routed by their own budget
    instead.  Rows the target already holds are left alone.
    """
    predicate = _legacy_budget_rows_predicate(conn)
    if predicate == '0':
        return 0
    # The predicate above already established the source has solve_budget.
    available = set(_all_cols_src(conn, CANONICAL_TABLE))
    cols = [column for column in
            ('branch_key', 'branch_reference', 'policy', 'answer_list_id',
             'solve_budget', 'best_guess', 'best_score', 'updated_at',
             'max_depth')
            if column in available]
    col_list = ', '.join(cols)
    before = conn.total_changes
    conn.execute('BEGIN IMMEDIATE')
    try:
        conn.execute(f"""
            INSERT OR IGNORE INTO main.{BUDGET_TABLE} ({col_list})
            SELECT {col_list} FROM src.{CANONICAL_TABLE} s WHERE {predicate}
        """)
        # Same reason the batch copy checks inside its transaction: a live
        # worker can store a disagreeing budget-specific result after the
        # preflight came back clean, and INSERT OR IGNORE would drop the
        # source's row silently.
        conflict = conn.execute(f"""
            SELECT s.branch_key, s.policy, s.solve_budget,
                   m.best_guess, m.best_score, m.max_depth,
                   s.best_guess, s.best_score, s.max_depth
            FROM src.{CANONICAL_TABLE} s
            JOIN main.{BUDGET_TABLE} m
              ON m.branch_key     = s.branch_key
             AND m.policy         = s.policy
             AND m.answer_list_id = s.answer_list_id
             AND m.solve_budget   = s.solve_budget
            WHERE {predicate}
              AND (abs(m.best_score - s.best_score) > ?
                   OR m.max_depth IS NOT s.max_depth)
            LIMIT 1
        """, (EXACT_SCORE_TOLERANCE,)).fetchone()
        if conflict is not None:
            conn.execute('ROLLBACK')
            found = ('budget-specific (routed)', branch_reference(conflict[0]),
                     conflict[1], conflict[2],
                     (conflict[3], conflict[4], conflict[5]),
                     (conflict[6], conflict[7], conflict[8]))
            print()
            _report_conflicts([found])
            raise ImportConflict(found[1])
    except ImportConflict:
        raise
    except Exception:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')
    return conn.total_changes - before


def _conflicting_exact_results(conn, src_tables, limit=20):
    """Colliding exact results whose costs disagree, across both scopes.

    A key collision between two caches is normally one fact reached twice, and
    INSERT OR IGNORE is exact for that.  It is not safe to *assume*: two caches
    can hold different exact results for one branch at one scope, and whichever
    the merge keeps displaces a result the other cache's ancestors folded.

    Sameness is cache_sqlite.exact_results_agree — equal cost AND equal
    max_depth, expressed here in SQL so whole tables compare at once.  Equal
    cost alone is not enough: max_depth is ancestor-visible, so keeping the
    target's child while admitting source-only parents folded from the source's
    deeper worst case makes those imported parents inconsistent on arrival.
    Within one file ScoreCache.write reconciles the same situation by handing
    the incumbent back for the caller to adopt; a merge has no such option,
    because the incoming ancestors are already computed and stored.

    Returns up to `limit` (scope, reference, policy, budget, target, incoming)
    rows.  A legacy source carries its budget-specific results in the canonical
    table, so those are compared against the target's budget table, where the
    routing would put them.
    """
    src_has_budget_column = 'solve_budget' in _all_cols_src(conn, CANONICAL_TABLE)
    canonical_src_filter = ('s.solve_budget IS NULL' if src_has_budget_column
                            else '1')
    comparisons = [
        ('unrestricted', CANONICAL_TABLE, CANONICAL_TABLE,
         canonical_src_filter, ''),
    ]
    if src_has_budget_column:
        # A pre-split source's budget rows are routed to the budget table, so
        # that is where they can collide.
        comparisons.append(
            ('budget-specific (routed)', BUDGET_TABLE, CANONICAL_TABLE,
             _legacy_budget_rows_predicate(conn),
             ' AND m.solve_budget = s.solve_budget'))
    if BUDGET_TABLE in src_tables and _target_has_table(conn, BUDGET_TABLE):
        comparisons.append(
            ('budget-specific', BUDGET_TABLE, BUDGET_TABLE, '1',
             ' AND m.solve_budget = s.solve_budget'))

    conflicts = []
    for scope, target_table, source_table, source_filter, budget_join in comparisons:
        if not _target_has_table(conn, target_table):
            continue
        rows = conn.execute(f"""
            SELECT s.branch_key, s.policy, s.solve_budget,
                   m.best_guess, m.best_score, m.max_depth,
                   s.best_guess, s.best_score, s.max_depth
            FROM src.{source_table} s
            JOIN main.{target_table} m
              ON m.branch_key     = s.branch_key
             AND m.policy         = s.policy
             AND m.answer_list_id = s.answer_list_id{budget_join}
            WHERE ({source_filter})
              AND (abs(m.best_score - s.best_score) > ?
                   OR m.max_depth IS NOT s.max_depth)
            LIMIT ?
        """, (EXACT_SCORE_TOLERANCE, limit - len(conflicts))).fetchall()
        for row in rows:
            conflicts.append((scope, branch_reference(row[0]), row[1], row[2],
                              (row[3], row[4], row[5]),
                              (row[6], row[7], row[8])))
        if len(conflicts) >= limit:
            break
    return conflicts


def _report_conflicts(conflicts) -> None:
    print(f'\n{len(conflicts):,} colliding exact result(s) are not the same '
          f'certificate:', file=sys.stderr)
    for scope, reference, policy, budget, target, incoming in conflicts:
        print(f'  {reference} {scope} policy={policy} budget={budget}: '
              f'target {target[0]}/{target[1]!r}/depth {target[2]}, '
              f'source {incoming[0]}/{incoming[1]!r}/depth {incoming[2]}',
              file=sys.stderr)
    print("Two exact results for one scope cannot both be right, and equal "
          "cost with a different worst case is still two certificates: "
          "whichever side the merge keeps, the other cache's ancestors folded "
          "the one it dropped.  Resolve the disagreement before importing.",
          file=sys.stderr)


def _batch_conflict(conn, table, row_filter, low_rowid, high_rowid):
    """A non-equivalent collision among src rows in (low_rowid, high_rowid].

    Called inside the batch's own write transaction, so no other connection
    can commit between this check and the insert it validates.  Returns one
    conflict row, or None.
    """
    if table not in (CANONICAL_TABLE, BUDGET_TABLE):
        return None
    budget_join = ('' if table == CANONICAL_TABLE
                   else ' AND m.solve_budget = s.solve_budget')
    rows = conn.execute(f"""
        SELECT s.branch_key, s.policy, s.solve_budget,
               m.best_guess, m.best_score, m.max_depth,
               s.best_guess, s.best_score, s.max_depth
        FROM src.{table} s
        JOIN main.{table} m
          ON m.branch_key     = s.branch_key
         AND m.policy         = s.policy
         AND m.answer_list_id = s.answer_list_id{budget_join}
        WHERE s.rowid > ? AND s.rowid <= ?{row_filter}
          AND (abs(m.best_score - s.best_score) > ?
               OR m.max_depth IS NOT s.max_depth)
        LIMIT 1
    """, (low_rowid, high_rowid, EXACT_SCORE_TOLERANCE)).fetchall()
    if not rows:
        return None
    row = rows[0]
    scope = 'unrestricted' if table == CANONICAL_TABLE else 'budget-specific'
    return (scope, branch_reference(row[0]), row[1], row[2],
            (row[3], row[4], row[5]), (row[6], row[7], row[8]))


def _copy_table_with_progress(conn, table) -> int:
    """Copy src.table -> main.table in rowid-paged batches with live progress.

    INSERT OR IGNORE makes each batch idempotent, and each batch commits on
    its own, so an interrupted merge can simply be re-run to resume.  Paging
    by rowid (rather than one opaque INSERT...SELECT of millions of rows) is
    what lets us show a percentage and ETA instead of an apparent hang.

    For the two exact-result tables each batch validates inside its own write
    transaction: the preflight only describes the target as it was when the
    import started, and a live worker can store a disagreeing result after
    that scan came back clean — INSERT OR IGNORE would then drop the source's
    row without a word.  Holding the write lock across the insert and the
    check is what makes the two one step.  A conflict rolls the batch back and
    raises ImportConflict.

    BEGIN IMMEDIATE takes that lock up front rather than at the first write,
    so the guarantee does not rest on the insert happening to be the first
    statement in the block.  With the statements in their current order a
    deferred BEGIN would hold too, which is why no test separates them.
    """
    cols = _all_cols(conn, table)
    col_list = ', '.join(cols)
    insert_sql = _insert_sql(table, cols)
    # A legacy source's budget-specific rows are routed to the budget table by
    # _route_legacy_budget_rows; they must not also land in the canonical one.
    row_filter = ''
    if table == CANONICAL_TABLE:
        predicate = _legacy_budget_rows_predicate(conn)
        if predicate != '0':
            row_filter = f' AND NOT ({predicate})'
    total = conn.execute(
        f'SELECT COUNT(*) FROM src.{table} s WHERE 1{row_filter}').fetchone()[0]
    if total == 0:
        print(f'  {table}: source empty, skipping')
        return 0

    inserted = scanned = 0
    last_rowid = 0
    start = time.time()
    last_print = 0.0
    while True:
        rows = conn.execute(
            f'SELECT s.rowid, {col_list} FROM src.{table} s '
            f'WHERE s.rowid > ?{row_filter} ORDER BY s.rowid LIMIT ?',
            (last_rowid, BATCH)).fetchall()
        if not rows:
            break
        batch_low, last_rowid = last_rowid, rows[-1][0]
        before = conn.total_changes
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.executemany(insert_sql, [tuple(r[1:]) for r in rows])
            conflict = _batch_conflict(conn, table, row_filter,
                                       batch_low, last_rowid)
            if conflict is not None:
                conn.execute('ROLLBACK')
                print()
                _report_conflicts([conflict])
                raise ImportConflict(conflict[1])
            conn.execute('COMMIT')
        except ImportConflict:
            raise
        except Exception:
            conn.execute('ROLLBACK')
            raise
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


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='?', default='wordle_erd_export.sqlite3',
                        help='Source cache file (default: wordle_erd_export.sqlite3)')
    parser.add_argument('--target', default=DEFAULT_TARGET, metavar='PATH',
                        help=f'Target cache file (default: {DEFAULT_TARGET})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report counts without writing anything')
    parser.add_argument('--keep-source', action='store_true',
                        help='Keep the source file after a successful merge '
                             '(the default deletes it)')
    args = parser.parse_args()
    ensure_runtime_dir()

    if not os.path.exists(args.source):
        # ATTACH would silently create an empty database for a missing path,
        # turning a typo into a "0 rows inserted" non-merge.
        print(f'Error: source file {args.source} does not exist',
              file=sys.stderr)
        sys.exit(1)

    if (not args.dry_run and not args.keep_source
            and os.path.exists(args.target)
            and os.path.samefile(args.source, args.target)):
        print(f'Error: source {args.source!r} and target {args.target!r} '
              'refer to the same file; refusing to delete the target',
              file=sys.stderr)
        sys.exit(1)

    target_existed = os.path.exists(args.target)
    if not target_existed:
        action = ('a new cache would be created' if args.dry_run
                  else 'creating a new cache')
        print(f'Target {args.target} does not exist — {action} '
              f'with a fully migrated schema.')

    if not args.dry_run:
        _bootstrap_target_schema(args.target)

    conn = sqlite3.connect(args.target, timeout=30.0, isolation_level=None)
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute(f"ATTACH DATABASE '{args.source}' AS src")

        src_tables = {r[0] for r in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'")}

        # Before a single row moves: a disagreement means one of the two
        # caches is wrong, and no merge direction is safe until that is
        # settled.  A dry run reports and continues; a real merge refuses,
        # leaving the target as it was rather than half-merged.
        if CANONICAL_TABLE in src_tables:
            conflicts = _conflicting_exact_results(conn, src_tables)
            if conflicts:
                _report_conflicts(conflicts)
                if not args.dry_run:
                    sys.exit(1)

        total_new = 0
        for table in TABLES:
            if table not in src_tables:
                print(f'  {table}: not in source, skipping')
                continue

            if args.dry_run:
                if not _target_has_table(conn, table):
                    n = conn.execute(
                        f'SELECT COUNT(*) FROM src.{table}').fetchone()[0]
                    print(f'  {table}: {n:,} rows would be inserted '
                          f'(table would be created in target)')
                    total_new += n
                    continue
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
                # A legacy source's budget-specific rows are counted where they
                # will land, not where they currently sit.
                if table == CANONICAL_TABLE:
                    predicate = _legacy_budget_rows_predicate(conn)
                    if predicate != '0':
                        routed = conn.execute(
                            f'SELECT COUNT(*) FROM src.{table} s '
                            f'WHERE {predicate}').fetchone()[0]
                        if routed:
                            n -= routed
                            msg = (f'  {table}: {n:,} rows would be inserted '
                                   f'({routed:,} budget-specific rows routed '
                                   f'to {BUDGET_TABLE})')
                print(msg)
            else:
                try:
                    n = _merge_table(conn, table)
                    if table == CANONICAL_TABLE:
                        n += _route_legacy_budget_rows(conn)
                except ImportConflict:
                    print('\nMerge stopped. Batches already committed stand '
                          'and are idempotent: resolve the disagreement and '
                          're-run to resume.', file=sys.stderr)
                    sys.exit(1)

            total_new += n

        if args.dry_run:
            print(f'\nTotal: {total_new:,} rows would be inserted '
                  f'(dry run — no changes made)')
        else:
            print('  checkpointing...', flush=True)
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            print(f'Total: {total_new:,} rows inserted')
            if not args.keep_source:
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
        if args.dry_run and not target_existed:
            # Connecting created an empty shell where no target existed;
            # a dry run must leave no trace.
            for suffix in ('', '-shm', '-wal'):
                p = args.target + suffix
                if os.path.exists(p):
                    os.remove(p)


if __name__ == '__main__':
    main()
