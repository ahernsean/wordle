"""Tests for the split between a branch's unrestricted and budget-specific results.

A branch has two kinds of exact result and they are different facts: the
optimum over all strategies, and the optimum among those feasible at one
remaining-depth budget.  They live in separate tables so neither write can
destroy the other, because an ancestor may already have folded the one being
replaced and nothing records which it folded (issue #302).
"""
import os
import shutil
import sqlite3
import tempfile
import unittest

from cache_sqlite import CacheWriteConflict, ScoreCache
from wordle_engine import ERD_ALL, _cache_reuse

WORDS = ["crane", "slate", "trace", "stale", "tales"]
CANONICAL = "branch_best_by_policy"
BUDGETED = "branch_best_by_policy_and_budget"


class _CacheTest(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        self.path = os.path.join(self._dir, "cache.sqlite3")
        self.key = ScoreCache.encode_subset(WORDS)

    def cache(self, name="cache.sqlite3"):
        score_cache = ScoreCache(os.path.join(self._dir, name), WORDS,
                                 checkpoint_on_close=False)
        self.addCleanup(score_cache.close)
        return score_cache

    def count(self, score_cache, table):
        return score_cache._conn.execute(
            f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class WriteRoutingTest(_CacheTest):
    """Which table each kind of result lands in, and in what order."""

    def test_the_two_scopes_land_in_their_own_tables(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0,
                          max_depth=4, solve_budget=None)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5,
                          max_depth=3, solve_budget=3)
        self.assertEqual(self.count(score_cache, CANONICAL), 1)
        self.assertEqual(self.count(score_cache, BUDGETED), 1)
        self.assertEqual(score_cache._conn.execute(
            f"SELECT COUNT(*) FROM {CANONICAL} "
            "WHERE solve_budget IS NOT NULL").fetchone()[0], 0)

    def test_both_write_orders_leave_the_same_two_results(self):
        first = self.cache("first.sqlite3")
        first.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        first.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3, solve_budget=3)

        second = self.cache("second.sqlite3")
        second.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3, solve_budget=3)
        second.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)

        for score_cache in (first, second):
            self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL),
                             ("crane", 2.0, 4, None))
            self.assertEqual(score_cache.read_for_budget(self.key, ERD_ALL, 3),
                             ("slate", 2.5, 3, 3))

    def test_two_budgets_coexist_in_either_order(self):
        first = self.cache("first.sqlite3")
        first.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3, solve_budget=3)
        first.write(self.key, ERD_ALL, "trace", 2.2, max_depth=4, solve_budget=4)

        second = self.cache("second.sqlite3")
        second.write(self.key, ERD_ALL, "trace", 2.2, max_depth=4, solve_budget=4)
        second.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3, solve_budget=3)

        for score_cache in (first, second):
            self.assertEqual(self.count(score_cache, BUDGETED), 2)
            self.assertEqual(
                score_cache.read_for_budget(self.key, ERD_ALL, 3)[0], "slate")
            self.assertEqual(
                score_cache.read_for_budget(self.key, ERD_ALL, 4)[0], "trace")

    def test_a_repeated_result_is_idempotent(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        self.assertEqual(self.count(score_cache, CANONICAL), 1)
        self.assertEqual(score_cache.redundant_write_count, 1)

    def test_an_equal_cost_rewrite_keeps_the_depth_ancestors_folded(self):
        # Two strategies can tie on ERD and differ in worst case.  Replacing
        # the stored one would leave every ancestor that folded its depth
        # describing a subtree the cache no longer holds.
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3)
        score_cache.write(self.key, ERD_ALL, "slate", 2.0, max_depth=4)
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 3, None))

    def test_a_disagreeing_result_is_refused_loudly(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3)
        with self.assertRaises(CacheWriteConflict) as raised:
            score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3)
        self.assertIn("crane" if "crane" in str(raised.exception) else "2.0",
                      str(raised.exception))
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 3, None))

    def test_a_disagreement_at_one_budget_does_not_implicate_another(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3,
                          solve_budget=3)
        score_cache.write(self.key, ERD_ALL, "trace", 2.2, max_depth=4,
                          solve_budget=4)
        with self.assertRaises(CacheWriteConflict):
            score_cache.write(self.key, ERD_ALL, "slate", 2.9, max_depth=3,
                              solve_budget=3)
        self.assertEqual(score_cache.read_for_budget(self.key, ERD_ALL, 4)[0],
                         "trace")


class ReadSelectionTest(_CacheTest):
    """Which of a branch's results a search at a given budget may reuse."""

    def seed(self, score_cache, *, canonical=None, budgets=()):
        if canonical is not None:
            score_cache.write(self.key, ERD_ALL, "crane", 2.0,
                              max_depth=canonical)
        for budget, depth in budgets:
            score_cache.write(self.key, ERD_ALL, f"g{budget:04d}"[:5], 2.5,
                              max_depth=depth, solve_budget=budget)

    def test_a_fitting_unrestricted_result_wins_over_a_budget_row(self):
        score_cache = self.cache()
        self.seed(score_cache, canonical=3, budgets=[(3, 3)])
        self.assertEqual(score_cache.read_for_budget(self.key, ERD_ALL, 3)[0],
                         "crane")

    def test_a_budget_row_answers_when_the_unrestricted_one_does_not_fit(self):
        score_cache = self.cache()
        self.seed(score_cache, canonical=5, budgets=[(3, 3)])
        entry = score_cache.read_for_budget(self.key, ERD_ALL, 3)
        self.assertEqual(entry[3], 3)

    def test_a_row_from_another_budget_is_a_miss(self):
        score_cache = self.cache()
        self.seed(score_cache, canonical=5, budgets=[(4, 4)])
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, 3))

    def test_an_unlimited_read_never_returns_a_budget_row(self):
        score_cache = self.cache()
        self.seed(score_cache, budgets=[(3, 3), (4, 4)])
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, None))

    def test_only_the_requested_budget_is_eligible(self):
        score_cache = self.cache()
        self.seed(score_cache, budgets=[(2, 2), (3, 3), (4, 4)])
        for budget in (2, 3, 4):
            self.assertEqual(
                score_cache.read_for_budget(self.key, ERD_ALL, budget)[3], budget)
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, 5))

    def test_every_selected_entry_passes_the_engine_s_own_reuse_rule(self):
        # read_for_budget selects; _cache_reuse remains the one place the rule
        # is stated, and must agree with every selection.
        score_cache = self.cache()
        self.seed(score_cache, canonical=4, budgets=[(2, 2), (3, 3)])
        for budget in (None, 2, 3, 4, 5):
            entry = score_cache.read_for_budget(self.key, ERD_ALL, budget)
            if entry is not None:
                self.assertIsNotNone(_cache_reuse(entry, budget),
                                     f"selected an entry _cache_reuse rejects "
                                     f"at budget {budget}")

    def test_the_memory_mirror_keeps_the_scopes_apart(self):
        score_cache = self.cache()
        self.seed(score_cache, canonical=5, budgets=[(3, 3)])
        # Prime both scopes, then confirm neither shadows the other.
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL)[0], "crane")
        self.assertEqual(score_cache.read_for_budget(self.key, ERD_ALL, 3)[3], 3)
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL)[0], "crane")
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, 2))


class AncestorSurvivesBothOrdersTest(_CacheTest):
    """A parent's fold stays true whichever of a child's results arrives later."""

    def child_key(self):
        return ScoreCache.encode_subset(WORDS[:2])

    def test_a_budget_specific_child_survives_its_unrestricted_sibling(self):
        score_cache = self.cache()
        child = self.child_key()
        score_cache.write(child, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        # A parent at budget 4 folds that child and records its own result.
        score_cache.write(self.key, ERD_ALL, "crane", 2.8, max_depth=4,
                          solve_budget=4)
        # The child is later solved unrestricted, at a different cost.
        score_cache.write(child, ERD_ALL, "trace", 2.1, max_depth=5)

        self.assertEqual(score_cache.read_for_budget(child, ERD_ALL, 3),
                         ("slate", 2.5, 3, 3))
        self.assertEqual(score_cache.read_for_budget(self.key, ERD_ALL, 4),
                         ("crane", 2.8, 4, 4))

    def test_an_unrestricted_child_survives_a_later_budget_specific_one(self):
        score_cache = self.cache()
        child = self.child_key()
        score_cache.write(child, ERD_ALL, "trace", 2.1, max_depth=3)
        score_cache.write(self.key, ERD_ALL, "crane", 2.6, max_depth=4)
        score_cache.write(child, ERD_ALL, "slate", 2.5, max_depth=2,
                          solve_budget=2)

        self.assertEqual(score_cache.read_with_depth(child, ERD_ALL),
                         ("trace", 2.1, 3, None))
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.6, 4, None))


class MigrationTest(_CacheTest):
    """Opening a pre-split cache moves its budget-specific rows, once."""

    def legacy_cache(self, name="legacy.sqlite3"):
        """A cache in the shape the split migration finds: budget-specific
        results sharing the canonical table's key."""
        path = os.path.join(self._dir, name)
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        answer_list_id = score_cache.answer_list_id
        score_cache._conn.execute(
            "DELETE FROM schema_migrations "
            "WHERE name = 'split_budget_specific_branch_results'")
        rows = [
            (ScoreCache.encode_subset(WORDS), "crane", 2.0, 4, None),
            (ScoreCache.encode_subset(WORDS[:2]), "slate", 2.5, 3, 3),
            (ScoreCache.encode_subset(WORDS[:3]), "trace", 2.2, 2, 4),
        ]
        for branch_key, guess, score, depth, budget in rows:
            score_cache._conn.execute(
                f"INSERT OR REPLACE INTO {CANONICAL} "
                "(branch_key, policy, answer_list_id, best_guess, best_score, "
                " updated_at, max_depth, solve_budget) "
                "VALUES (?, ?, ?, ?, ?, 100, ?, ?)",
                (branch_key, ERD_ALL, answer_list_id, guess, score, depth, budget))
        score_cache._conn.execute(
            "INSERT OR REPLACE INTO candidate_erd_by_policy "
            "(subset_hash, candidate_word, policy, answer_list_id, erd, "
            " max_remaining_depth, response_group_count, updated_at) "
            "VALUES ('h', 'crane', ?, ?, 2.0, 3, 2, 100)",
            (ERD_ALL, answer_list_id))
        score_cache.close()
        return path

    def test_budget_specific_rows_move_and_unrestricted_ones_stay_put(self):
        path = self.legacy_cache()
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(score_cache.close)

        self.assertEqual(self.count(score_cache, CANONICAL), 1)
        self.assertEqual(self.count(score_cache, BUDGETED), 2)
        self.assertEqual(score_cache.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 4, None))
        self.assertEqual(
            score_cache.read_for_budget(
                ScoreCache.encode_subset(WORDS[:2]), ERD_ALL, 3),
            ("slate", 2.5, 3, 3))
        self.assertEqual(
            score_cache.read_for_budget(
                ScoreCache.encode_subset(WORDS[:3]), ERD_ALL, 4),
            ("trace", 2.2, 2, 4))

    def test_the_unrestricted_row_keeps_its_own_timestamp(self):
        path = self.legacy_cache()
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(score_cache.close)
        self.assertEqual(score_cache._conn.execute(
            f"SELECT updated_at FROM {CANONICAL}").fetchone()[0], 100)
        self.assertEqual(sorted(row[0] for row in score_cache._conn.execute(
            f"SELECT updated_at FROM {BUDGETED}")), [100, 100])

    def test_the_derived_candidate_erd_folds_are_dropped(self):
        # Every one memoised a fold under the old branch-row identity.
        path = self.legacy_cache()
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(score_cache.close)
        self.assertEqual(self.count(score_cache, "candidate_erd_by_policy"), 0)

    def test_reopening_is_idempotent(self):
        path = self.legacy_cache()
        ScoreCache(path, WORDS, checkpoint_on_close=False).close()
        ScoreCache(path, WORDS, checkpoint_on_close=False).close()
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(score_cache.close)
        self.assertEqual(self.count(score_cache, CANONICAL), 1)
        self.assertEqual(self.count(score_cache, BUDGETED), 2)

    def test_the_canonical_table_is_not_rebuilt(self):
        # Moving the minority of rows is the point; rebuilding a multi-GB
        # table to alter a primary key is what this design avoids.
        path = self.legacy_cache()
        before = ScoreCache(path, WORDS, checkpoint_on_close=False)
        original_sql = before._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?",
            (CANONICAL,)).fetchone()[0]
        before.close()
        after = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(after.close)
        self.assertEqual(after._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?",
            (CANONICAL,)).fetchone()[0], original_sql)


class ReportingTest(_CacheTest):
    """Counts and states say which kind of fact they mean."""

    def test_a_branch_with_several_budgets_counts_once_as_a_branch(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        score_cache.write(self.key, ERD_ALL, "trace", 2.2, max_depth=2,
                          solve_budget=2)
        summary = score_cache.erd_report_summary(ERD_ALL, 0)
        self.assertEqual(summary["exact_branch_count"], 1)
        self.assertEqual(summary["budgeted_branch_count"], 1)
        self.assertEqual(summary["budgeted_result_count"], 2)

    def test_a_report_reads_the_result_its_budget_would_reuse(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=5)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        at_three = score_cache.report_branch_state(self.key, ERD_ALL, budget=3)
        self.assertEqual(at_three["cache_state"], "exact")
        self.assertEqual(at_three["best_guess"], "slate")
        self.assertTrue(at_three["tainted"])

        at_five = score_cache.report_branch_state(self.key, ERD_ALL, budget=5)
        self.assertEqual(at_five["best_guess"], "crane")
        self.assertFalse(at_five["tainted"])

        self.assertEqual(
            score_cache.report_branch_state(self.key, ERD_ALL, budget=2)
            ["cache_state"], "missing")

    def test_the_bulk_maps_select_the_same_result_as_a_single_lookup(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=5)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        exact_by_key, loss_by_key = score_cache.report_branch_row_maps(ERD_ALL)
        for budget in (None, 2, 3, 5):
            self.assertEqual(
                score_cache.report_branch_states_from_maps(
                    [self.key], exact_by_key, loss_by_key, budget)[self.key],
                score_cache.report_branch_states([self.key], ERD_ALL, budget)[self.key],
                f"maps and query disagree at budget {budget}")

    def test_delete_clears_every_scope(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        score_cache.delete(self.key, ERD_ALL)
        self.assertIsNone(score_cache.read_with_depth(self.key, ERD_ALL))
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, 3))
        self.assertEqual(self.count(score_cache, BUDGETED), 0)

    def test_last_write_ts_sees_a_budget_specific_write(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        self.assertIsNotNone(score_cache.last_write_ts())

    def test_a_budget_specific_branch_resolves_from_its_reference(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        from cache_sqlite import branch_reference
        self.assertIn(self.key, score_cache.branch_keys_for_reference_prefix(
            branch_reference(self.key)[:6]))


if __name__ == "__main__":
    unittest.main()


class TransferTest(_CacheTest):
    """What crosses between caches, including from a writer that predates the split."""

    def _import(self, target_path, source_path):
        import import_cache
        conn = sqlite3.connect(target_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"ATTACH DATABASE '{source_path}' AS src")
        try:
            inserted = import_cache._copy_table_with_progress(conn, CANONICAL)
            inserted += import_cache._route_legacy_budget_rows(conn)
            if conn.execute(
                    "SELECT COUNT(*) FROM src.sqlite_master WHERE name = ?",
                    (BUDGETED,)).fetchone()[0]:
                inserted += import_cache._copy_table_with_progress(conn, BUDGETED)
            return inserted
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()

    def _seed(self, name):
        score_cache = ScoreCache(os.path.join(self._dir, name), WORDS,
                                 checkpoint_on_close=False)
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        score_cache.close()
        return os.path.join(self._dir, name)

    def test_a_round_trip_between_current_caches_preserves_both_tables(self):
        source = self._seed("source.sqlite3")
        target_path = os.path.join(self._dir, "target.sqlite3")
        ScoreCache(target_path, WORDS, checkpoint_on_close=False).close()
        self._import(target_path, source)

        target = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        self.addCleanup(target.close)
        self.assertEqual(target.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 4, None))
        self.assertEqual(target.read_for_budget(self.key, ERD_ALL, 3),
                         ("slate", 2.5, 3, 3))

    def legacy_source(self, name="legacy.sqlite3"):
        """A source written before the split: its budget-specific result sits
        in the canonical table, where merging it would recreate the aliasing."""
        path = os.path.join(self._dir, name)
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        answer_list_id = score_cache.answer_list_id
        score_cache._conn.execute(
            f"INSERT OR REPLACE INTO {CANONICAL} "
            "(branch_key, policy, answer_list_id, best_guess, best_score, "
            " updated_at, max_depth, solve_budget) VALUES (?, ?, ?, ?, ?, 5, ?, ?)",
            (self.key, ERD_ALL, answer_list_id, "slate", 2.5, 3, 3))
        score_cache._conn.execute(f"DROP TABLE {BUDGETED}")
        score_cache.close()
        return path

    def test_a_legacy_source_s_budget_row_is_routed_not_aliased(self):
        source = self.legacy_source()
        target_path = os.path.join(self._dir, "target.sqlite3")
        target = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        target.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        target.close()

        self._import(target_path, source)

        merged = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        self.addCleanup(merged.close)
        # The target's unrestricted result is untouched, and the incoming
        # budget-specific one landed where budget-specific results live.
        self.assertEqual(merged.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 4, None))
        self.assertEqual(merged.read_for_budget(self.key, ERD_ALL, 3),
                         ("slate", 2.5, 3, 3))
        self.assertEqual(merged._conn.execute(
            f"SELECT COUNT(*) FROM {CANONICAL} "
            "WHERE solve_budget IS NOT NULL").fetchone()[0], 0)

    def test_a_legacy_budget_row_does_not_also_reach_the_canonical_table(self):
        source = self.legacy_source()
        target_path = os.path.join(self._dir, "target.sqlite3")
        ScoreCache(target_path, WORDS, checkpoint_on_close=False).close()
        self._import(target_path, source)

        merged = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        self.addCleanup(merged.close)
        self.assertEqual(self.count(merged, CANONICAL), 0)
        self.assertEqual(self.count(merged, BUDGETED), 1)

    def test_both_tables_are_carried_by_export_and_import(self):
        import export_cache
        import import_cache
        self.assertIn(BUDGETED, export_cache.EXPORT_TABLES)
        self.assertIn(BUDGETED, import_cache.TABLES)
        self.assertIn(CANONICAL, export_cache.EXPORT_TABLES)

    def test_a_source_predating_solve_budget_routes_nothing(self):
        # A cache old enough to lack the column has no budget-specific results
        # to route, and its rows merge as ordinary unrestricted ones.
        import import_cache
        path = os.path.join(self._dir, "ancient.sqlite3")
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        answer_list_id = score_cache.answer_list_id
        score_cache._conn.execute(f"DROP TABLE {BUDGETED}")
        score_cache._conn.execute(f"ALTER TABLE {CANONICAL} DROP COLUMN solve_budget")
        score_cache._conn.execute(
            f"INSERT OR REPLACE INTO {CANONICAL} "
            "(branch_key, policy, answer_list_id, best_guess, best_score, "
            " updated_at, max_depth) VALUES (?, ?, ?, 'crane', 2.0, 5, 4)",
            (self.key, ERD_ALL, answer_list_id))
        score_cache.close()

        target_path = os.path.join(self._dir, "target.sqlite3")
        target = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        target.close()
        conn = sqlite3.connect(target_path, isolation_level=None)
        conn.execute(f"ATTACH DATABASE '{path}' AS src")
        try:
            self.assertEqual(
                import_cache._legacy_budget_rows_predicate(conn), '0')
            self.assertEqual(import_cache._route_legacy_budget_rows(conn), 0)
            self.assertEqual(
                conn.execute(
                    f"SELECT COUNT(*) FROM main.{BUDGETED}").fetchone()[0], 0)
        finally:
            conn.execute("DETACH DATABASE src")
            conn.close()
