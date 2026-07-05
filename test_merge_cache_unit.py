"""Unit tests for merge_cache.py utility functions.

The merge_cache module is a CLI tool (main() is # pragma: no cover), but its
utility functions — _fmt_eta, _all_cols, _pk_cols, _insert_sql, and
_copy_table_with_progress — contain real logic worth pinning.

test_erd_fixes.TestMergeUntaintedWins already covers the tainted-vs-untainted
UPSERT semantics by calling _all_cols and _insert_sql directly; those tests
continue to provide that coverage.  The tests here focus on the helper
functions and the end-to-end copy path.
"""
import os
import sqlite3
import tempfile
import unittest

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL
import merge_cache

WORDS = ["crane", "slate", "trace", "stale", "tales"]


class _TmpDB(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def path(self, name):
        return os.path.join(self._tmp.name, name)

    def _cache(self, name):
        return ScoreCache(self.path(name), WORDS)


class TestFmtEta(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(merge_cache._fmt_eta(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(merge_cache._fmt_eta(75), "1m15s")

    def test_hours_and_minutes(self):
        self.assertEqual(merge_cache._fmt_eta(3661), "1h01m")


class TestAllCols(_TmpDB):
    def test_returns_column_names_for_branch_table(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            cols = merge_cache._all_cols(conn, "branch_best_by_policy")
        finally:
            conn.close()
        self.assertIn("branch_key", cols)
        self.assertIn("best_guess", cols)
        self.assertIn("solve_budget", cols)
        self.assertIn("answer_list_id", cols)


class TestPkCols(_TmpDB):
    def test_returns_primary_key_columns(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            pks = merge_cache._pk_cols(conn, "branch_best_by_policy")
        finally:
            conn.close()
        self.assertIn("branch_key", pks)
        self.assertIn("policy", pks)
        self.assertIn("answer_list_id", pks)


class TestInsertSql(unittest.TestCase):
    def _all_erd_cols(self):
        return list(merge_cache._ERD_UPSERT_COLS) + ["extra_col"]

    def test_branch_best_with_all_required_cols_returns_upsert(self):
        sql = merge_cache._insert_sql("branch_best_by_policy", self._all_erd_cols())
        self.assertIn("DO UPDATE", sql)
        self.assertIn("solve_budget IS NOT NULL", sql)

    def test_other_table_returns_insert_or_ignore(self):
        sql = merge_cache._insert_sql("response_decomposition", ["guess", "patterns"])
        self.assertIn("INSERT OR IGNORE", sql)
        self.assertNotIn("DO UPDATE", sql)

    def test_branch_best_missing_required_col_falls_back_to_insert_or_ignore(self):
        # Without solve_budget the tainted-vs-untainted rule cannot be applied.
        cols = [c for c in merge_cache._ERD_UPSERT_COLS if c != "solve_budget"]
        sql = merge_cache._insert_sql("branch_best_by_policy", cols)
        self.assertIn("INSERT OR IGNORE", sql)
        self.assertNotIn("DO UPDATE", sql)


class TestCopyTableWithProgress(_TmpDB):
    def _make_src_with_entry(self):
        sc = self._cache("source.sqlite3")
        key = ScoreCache.encode_subset(WORDS)
        sc.write(key, ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        sc.close()

    def _copy(self, table):
        conn = sqlite3.connect(self.path("target.sqlite3"), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{self.path('source.sqlite3')}' AS src")
        try:
            return merge_cache._copy_table_with_progress(conn, table)
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

    def test_copy_inserts_rows_into_empty_target(self):
        # Initialize target schema via ScoreCache, then copy from source.
        target_sc = self._cache("target.sqlite3")
        target_sc.close()
        self._make_src_with_entry()

        n = self._copy("branch_best_by_policy")
        self.assertGreater(n, 0)

        sc = ScoreCache(self.path("target.sqlite3"), WORDS, checkpoint_on_close=False)
        result = sc.read(ScoreCache.encode_subset(WORDS), ERD_ALL)
        sc.close()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "crane")

    def test_copy_is_idempotent(self):
        target_sc = self._cache("target.sqlite3")
        target_sc.close()
        self._make_src_with_entry()

        n1 = self._copy("branch_best_by_policy")
        n2 = self._copy("branch_best_by_policy")
        self.assertGreater(n1, 0)
        self.assertEqual(n2, 0)   # INSERT OR IGNORE skips already-present rows

    def test_copy_empty_source_returns_zero_and_skips(self):
        target_sc = self._cache("target.sqlite3")
        target_sc.close()
        source_sc = self._cache("source.sqlite3")
        source_sc.close()
        # No entries in source: _copy_table_with_progress should print a skip
        # message and return 0.
        n = self._copy("branch_best_by_policy")
        self.assertEqual(n, 0)


class TestCreateTargetTableFromSource(_TmpDB):
    """Merging into a target that has no tables (a fresh device recovering a
    lost cache) must create each table from the source's schema."""

    def _make_src_with_entry(self):
        sc = self._cache("source.sqlite3")
        key = ScoreCache.encode_subset(WORDS)
        sc.write(key, ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        sc.close()

    def _connect_empty_target(self):
        conn = sqlite3.connect(self.path("target.sqlite3"),
                               isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{self.path('source.sqlite3')}' AS src")
        return conn

    def test_target_has_table_reflects_main_schema(self):
        self._make_src_with_entry()
        conn = self._connect_empty_target()
        try:
            self.assertFalse(
                merge_cache._target_has_table(conn, "branch_best_by_policy"))
            merge_cache._create_target_table_from_source(
                conn, "branch_best_by_policy")
            self.assertTrue(
                merge_cache._target_has_table(conn, "branch_best_by_policy"))
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

    def test_created_table_matches_source_columns_and_indexes(self):
        self._make_src_with_entry()
        conn = self._connect_empty_target()
        try:
            merge_cache._create_target_table_from_source(
                conn, "branch_best_by_policy")
            self.assertEqual(
                merge_cache._all_cols(conn, "branch_best_by_policy"),
                [r[1] for r in conn.execute(
                    "PRAGMA src.table_info(branch_best_by_policy)")])
            src_indexes = {r[0] for r in conn.execute(
                "SELECT name FROM src.sqlite_master WHERE type='index' "
                "AND tbl_name='branch_best_by_policy' AND sql IS NOT NULL")}
            main_indexes = {r[0] for r in conn.execute(
                "SELECT name FROM main.sqlite_master WHERE type='index' "
                "AND tbl_name='branch_best_by_policy' AND sql IS NOT NULL")}
            self.assertEqual(src_indexes, main_indexes)
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

    def test_copy_into_created_table_yields_readable_cache(self):
        # End-to-end: schema from source, rows copied, and the result opens
        # cleanly through ScoreCache (its migrations must all be no-ops).
        self._make_src_with_entry()
        conn = self._connect_empty_target()
        try:
            merge_cache._create_target_table_from_source(
                conn, "branch_best_by_policy")
            n = merge_cache._copy_table_with_progress(
                conn, "branch_best_by_policy")
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()
        self.assertGreater(n, 0)

        sc = ScoreCache(self.path("target.sqlite3"), WORDS,
                        checkpoint_on_close=False)
        result = sc.read(ScoreCache.encode_subset(WORDS), ERD_ALL)
        sc.close()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "crane")


if __name__ == "__main__":
    unittest.main()
