"""Unit tests for import_cache.py utility functions.

The import_cache module is a CLI tool (main() is # pragma: no cover), but its
utility functions — _fmt_eta, _all_cols, _pk_cols, _insert_sql, and
_copy_table_with_progress — contain real logic worth pinning.

test_erd_fixes.TestMergeUntaintedWins already covers the tainted-vs-untainted
UPSERT semantics by calling _all_cols and _insert_sql directly; those tests
continue to provide that coverage.  The tests here focus on the helper
functions and the end-to-end copy path.
"""
import hashlib
import io
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL
import import_cache

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
        self.assertEqual(import_cache._fmt_eta(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(import_cache._fmt_eta(75), "1m15s")

    def test_hours_and_minutes(self):
        self.assertEqual(import_cache._fmt_eta(3661), "1h01m")


class TestAllCols(_TmpDB):
    def test_returns_column_names_for_branch_table(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            cols = import_cache._all_cols(conn, "branch_best_by_policy")
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
            pks = import_cache._pk_cols(conn, "branch_best_by_policy")
        finally:
            conn.close()
        self.assertIn("branch_key", pks)
        self.assertIn("policy", pks)
        self.assertIn("answer_list_id", pks)


class TestTargetHasTable(_TmpDB):
    def test_true_for_a_table_that_exists(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            self.assertTrue(
                import_cache._target_has_table(conn, "branch_best_by_policy"))
        finally:
            conn.close()

    def test_false_for_a_table_that_does_not_exist(self):
        conn = sqlite3.connect(self.path("empty.sqlite3"))
        try:
            self.assertFalse(
                import_cache._target_has_table(conn, "branch_best_by_policy"))
        finally:
            conn.close()


class TestInsertSql(unittest.TestCase):
    def _all_erd_cols(self):
        return list(import_cache._ERD_UPSERT_COLS) + ["extra_col"]

    def test_branch_best_with_all_required_cols_returns_upsert(self):
        sql = import_cache._insert_sql("branch_best_by_policy", self._all_erd_cols())
        self.assertIn("DO UPDATE", sql)
        self.assertIn("solve_budget IS NOT NULL", sql)

    def test_other_table_returns_insert_or_ignore(self):
        sql = import_cache._insert_sql("response_decomposition", ["guess", "patterns"])
        self.assertIn("INSERT OR IGNORE", sql)
        self.assertNotIn("DO UPDATE", sql)

    def test_branch_best_missing_required_col_falls_back_to_insert_or_ignore(self):
        # Without solve_budget the tainted-vs-untainted rule cannot be applied.
        cols = [c for c in import_cache._ERD_UPSERT_COLS if c != "solve_budget"]
        sql = import_cache._insert_sql("branch_best_by_policy", cols)
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
            return import_cache._copy_table_with_progress(conn, table)
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


class TestDroppableIndexes(_TmpDB):
    def test_returns_secondary_indexes_excluding_primary_key(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            indexes = import_cache._droppable_indexes(
                conn, "branch_best_by_policy")
        finally:
            conn.close()
        names = {name for name, _ in indexes}
        self.assertEqual(
            names, {"idx_branch_best_by_policy", "idx_branch_updated"})

    def test_returns_empty_for_table_with_only_a_primary_key(self):
        sc = self._cache("c.sqlite3")
        sc.close()
        conn = sqlite3.connect(self.path("c.sqlite3"))
        try:
            indexes = import_cache._droppable_indexes(conn, "answer_list")
        finally:
            conn.close()
        self.assertEqual(indexes, [])


class TestBootstrapTargetSchema(_TmpDB):
    """Merging into a target with no tables at all (a fresh device
    recovering a lost cache) must produce a fully migrated schema, and a
    target whose candidate_scores table is still under its pre-rename
    legacy name (word_scores) must be migrated in place, not left for a
    later ScoreCache open to clobber."""

    def test_creates_fully_migrated_schema_for_a_brand_new_target(self):
        target_path = self.path("target.sqlite3")
        import_cache._bootstrap_target_schema(target_path)

        conn = sqlite3.connect(target_path)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue(
                {"schema_migrations", "answer_list", "response_decomposition",
                 "branch_best_by_policy", "candidate_scores"} <= tables)
            migrations_done = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
            self.assertGreater(migrations_done, 0)
            answer_list_rows = conn.execute(
                "SELECT COUNT(*) FROM answer_list").fetchone()[0]
            self.assertEqual(answer_list_rows, 1)
        finally:
            conn.close()

    def test_migrates_legacy_table_name_in_place_without_losing_its_row(self):
        target_path = self.path("target.sqlite3")
        # Hand-build a target whose candidate_scores table is still under
        # its pre-rename name (word_scores) with the current, subset_hash
        # schema -- the exact legacy state that used to be silently wiped
        # if a merge wrote new rows into a candidate_scores table that
        # ScoreCache's rename_word_scores migration hadn't run over yet.
        conn = sqlite3.connect(target_path)
        conn.execute("""
            CREATE TABLE word_scores (
                subset_hash    TEXT    NOT NULL,
                word           TEXT    NOT NULL,
                method         TEXT    NOT NULL,
                score          REAL    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, answer_list_id, word)
            )
        """)
        conn.execute(
            "INSERT INTO word_scores VALUES "
            "('deadbeef', 'crane', 'entropy_gain', 42.0, 'legacy_universe', 0)")
        conn.commit()
        conn.close()

        import_cache._bootstrap_target_schema(target_path)

        conn = sqlite3.connect(target_path)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("word_scores", tables)
            self.assertIn("candidate_scores", tables)
            rows = conn.execute(
                "SELECT word, score FROM candidate_scores").fetchall()
            self.assertEqual(rows, [("crane", 42.0)])
        finally:
            conn.close()


class TestMainSourceDeletion(_TmpDB):
    def test_refuses_to_delete_source_that_is_also_target(self):
        cache_path = self.path("cache.sqlite3")
        cache = ScoreCache(cache_path, WORDS)
        cache.close()

        stderr = io.StringIO()
        with mock.patch("sys.argv", [
                "import_cache.py", cache_path, "--target", cache_path]), \
                mock.patch("sys.stderr", stderr), \
                self.assertRaises(SystemExit) as raised:
            import_cache.main()

        self.assertEqual(raised.exception.code, 1)
        error = stderr.getvalue()
        self.assertIn(f"source {cache_path!r}", error)
        self.assertIn(f"target {cache_path!r}", error)
        self.assertIn("refer to the same file", error)
        self.assertTrue(os.path.exists(cache_path))


class TestMergeTable(_TmpDB):
    def _make_src_with_entry(self):
        sc = self._cache("source.sqlite3")
        key = ScoreCache.encode_subset(WORDS)
        sc.write(key, ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        sc.close()

    def test_defers_indexes_for_an_empty_target_and_recreates_them_after(self):
        self._make_src_with_entry()
        target_sc = self._cache("target.sqlite3")
        target_sc.close()

        conn = sqlite3.connect(self.path("target.sqlite3"),
                               isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{self.path('source.sqlite3')}' AS src")

        seen_indexes_during_copy = {}
        real_copy = import_cache._copy_table_with_progress

        def spy_copy(conn_, table):
            seen_indexes_during_copy[table] = {
                name for name, _ in
                import_cache._droppable_indexes(conn_, table)}
            return real_copy(conn_, table)

        try:
            with mock.patch.object(import_cache, "_copy_table_with_progress",
                                   spy_copy):
                n = import_cache._merge_table(conn, "branch_best_by_policy")
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

        self.assertGreater(n, 0)
        # Indexes were gone while the bulk copy ran ...
        self.assertEqual(
            seen_indexes_during_copy["branch_best_by_policy"], set())

        # ... and are back afterward.
        conn = sqlite3.connect(self.path("target.sqlite3"))
        try:
            final_indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='branch_best_by_policy' AND sql IS NOT NULL")}
        finally:
            conn.close()
        self.assertEqual(
            final_indexes, {"idx_branch_best_by_policy", "idx_branch_updated"})

    def test_does_not_defer_indexes_for_an_already_populated_target(self):
        self._make_src_with_entry()
        target_sc = self._cache("target.sqlite3")
        key = ScoreCache.encode_subset(WORDS)
        target_sc.write(key, ERD_ALL, "slate", 2.0, max_depth=3,
                        solve_budget=None)
        target_sc.close()

        conn = sqlite3.connect(self.path("target.sqlite3"),
                               isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{self.path('source.sqlite3')}' AS src")

        seen_indexes_during_copy = {}
        real_copy = import_cache._copy_table_with_progress

        def spy_copy(conn_, table):
            seen_indexes_during_copy[table] = {
                name for name, _ in
                import_cache._droppable_indexes(conn_, table)}
            return real_copy(conn_, table)

        try:
            with mock.patch.object(import_cache, "_copy_table_with_progress",
                                   spy_copy):
                import_cache._merge_table(conn, "branch_best_by_policy")
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

        # Target already had rows: indexes must stay live throughout.
        self.assertEqual(
            seen_indexes_during_copy["branch_best_by_policy"],
            {"idx_branch_best_by_policy", "idx_branch_updated"})


class TestLegacyTableNameRegression(_TmpDB):
    """End-to-end regression test for the data-loss bug: merging into a
    target whose candidate_scores table was still named word_scores used
    to have the merged rows silently destroyed the next time the cache was
    opened normally, because the rename migration would drop whatever
    candidate_scores table the merge had created and put the untouched
    legacy word_scores rows in its place."""

    def test_merge_into_legacy_named_target_preserves_both_old_and_new_rows(self):
        source_path = self.path("source.sqlite3")
        source_sc = ScoreCache(source_path, WORDS)
        source_sc.write_scores(
            ScoreCache.encode_subset(WORDS), [("slate", 1.1)], "entropy_gain")
        source_sc.close()

        target_path = self.path("target.sqlite3")
        conn = sqlite3.connect(target_path)
        conn.execute("""
            CREATE TABLE word_scores (
                subset_hash    TEXT    NOT NULL,
                word           TEXT    NOT NULL,
                method         TEXT    NOT NULL,
                score          REAL    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, answer_list_id, word)
            )
        """)
        root_hash = ScoreCache._subset_hash(ScoreCache.encode_subset(WORDS))
        answer_list_id = hashlib.sha256(
            "\n".join(WORDS).encode()).hexdigest()
        conn.execute(
            "INSERT INTO word_scores VALUES (?, 'crane', 'entropy_gain', "
            "42.0, ?, 0)", (root_hash, answer_list_id))
        conn.commit()
        conn.close()

        import_cache._bootstrap_target_schema(target_path)
        conn = sqlite3.connect(target_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{source_path}' AS src")
        try:
            n = import_cache._merge_table(conn, "candidate_scores")
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()
        self.assertEqual(n, 1)

        # Opening the target again, as the app normally would, must not
        # lose either the pre-existing legacy row or the merged one.
        sc = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        scores = sc.read_scores(
            ScoreCache.encode_subset(WORDS), "entropy_gain")
        sc.close()
        self.assertCountEqual(scores, [("crane", 42.0), ("slate", 1.1)])


if __name__ == "__main__":
    unittest.main()
