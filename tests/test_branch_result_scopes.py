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
import threading
import unittest
from unittest import mock

from cache_sqlite import (CacheWriteConflict, ScoreCache,
                          branch_reference, exact_results_agree)
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


class ConcurrentWriteTest(_CacheTest):
    """Two writers reaching one scope at once.

    Creating the row is the check: a read followed by an insert leaves a
    window both writers pass through, and the loser's insert would then
    displace a result an ancestor may already have folded, with neither
    noticing.  These drive that window directly.
    """

    SCOPES = [('unrestricted', None), ('budget-specific', 3)]

    def test_a_stale_miss_cannot_displace_a_durable_result(self):
        for label, scope in self.SCOPES:
            with self.subTest(scope=label):
                observer = self.cache(f'{label}-a.sqlite3')
                writer = ScoreCache(observer.db_path, WORDS,
                                    checkpoint_on_close=False)
                self.addCleanup(writer.close)
                # The observer looks first and sees nothing...
                self.assertIsNone(
                    observer.read_for_budget(self.key, ERD_ALL, scope))
                # ...the writer stores a result...
                writer.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3,
                             solve_budget=scope)
                # ...and the observer writes on the strength of its stale miss.
                with self.assertLogs('wordle', level='ERROR') as logged:
                    with self.assertRaises(CacheWriteConflict):
                        observer.write(self.key, ERD_ALL, "slate", 2.5,
                                       max_depth=3, solve_budget=scope)
                self.assertIn('conflicting exact results',
                              '\n'.join(logged.output))
                self.assertEqual(
                    writer.read_for_budget(self.key, ERD_ALL, scope),
                    ("crane", 2.0, 3, scope))

    def test_a_stale_miss_with_an_agreeing_value_keeps_the_incumbent(self):
        # Equal cost, different worst case: the stored depth is the one an
        # ancestor folded, so the loser must not replace it.
        for label, scope in self.SCOPES:
            with self.subTest(scope=label):
                observer = self.cache(f'{label}-b.sqlite3')
                writer = ScoreCache(observer.db_path, WORDS,
                                    checkpoint_on_close=False)
                self.addCleanup(writer.close)
                self.assertIsNone(
                    observer.read_for_budget(self.key, ERD_ALL, scope))
                writer.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3,
                             solve_budget=scope)

                observer.write(self.key, ERD_ALL, "slate", 2.0, max_depth=4,
                               solve_budget=scope)
                self.assertEqual(observer.redundant_write_count, 1)
                self.assertEqual(
                    observer.read_for_budget(self.key, ERD_ALL, scope),
                    ("crane", 2.0, 3, scope))

    def _race(self, scope, scores):
        """Two connections writing one scope, released together."""
        path = os.path.join(self._dir, f'race-{scope}-{scores[1]}.sqlite3')
        ScoreCache(path, WORDS, checkpoint_on_close=False).close()
        barrier = threading.Barrier(len(scores))
        outcomes = {}

        def run(name, guess, score):
            score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
            try:
                score_cache.read_for_budget(self.key, ERD_ALL, scope)
                barrier.wait()
                score_cache.write(self.key, ERD_ALL, guess, score,
                                  max_depth=3, solve_budget=scope)
                outcomes[name] = None
            except CacheWriteConflict as conflict:
                outcomes[name] = conflict
            finally:
                score_cache.close()

        threads = [threading.Thread(target=run, args=(name, guess, score))
                   for name, guess, score in scores]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        after = ScoreCache(path, WORDS, checkpoint_on_close=False)
        self.addCleanup(after.close)
        table = CANONICAL if scope is None else BUDGETED
        rows = after._conn.execute(
            f"SELECT best_guess, best_score FROM {table}").fetchall()
        return outcomes, [tuple(row) for row in rows]

    def test_racing_writers_that_disagree_leave_one_result_and_one_conflict(self):
        for label, scope in self.SCOPES:
            with self.subTest(scope=label):
                with self.assertLogs('wordle', level='ERROR'):
                    outcomes, rows = self._race(
                        scope, [("a", "crane", 2.0), ("b", "slate", 2.5)])
                self.assertEqual(len(rows), 1)
                raised = [name for name, outcome in outcomes.items()
                          if outcome is not None]
                self.assertEqual(len(raised), 1, outcomes)
                # The survivor is whoever created the row, and the loser is
                # the one that was told about it.
                self.assertIn(rows[0], [("crane", 2.0), ("slate", 2.5)])

    def test_racing_writers_that_agree_leave_one_result_and_no_conflict(self):
        for label, scope in self.SCOPES:
            with self.subTest(scope=label):
                outcomes, rows = self._race(
                    scope, [("a", "crane", 2.0), ("b", "crane", 2.0)])
                self.assertEqual(rows, [("crane", 2.0)])
                self.assertEqual([o for o in outcomes.values() if o], [])

    def test_an_equal_cost_result_is_adopted_rather_than_kept_locally(self):
        # The incumbent stands, and write hands it back: max_depth is
        # ancestor-visible, so a solver that kept its own worst case would
        # fold a parent the stored child does not support -- the same
        # inconsistent ancestry, reached without any overwrite.
        for label, scope in self.SCOPES:
            with self.subTest(scope=label):
                score_cache = self.cache(f'{label}-adopt.sqlite3')
                score_cache.write(self.key, ERD_ALL, "crane", 2.0,
                                  max_depth=3, solve_budget=scope)
                durable = score_cache.write(self.key, ERD_ALL, "slate", 2.0,
                                            max_depth=4, solve_budget=scope)
                self.assertEqual(durable, ("crane", 2.0, 3, scope))
                self.assertEqual(score_cache.adopted_depth_count, 1)
                self.assertEqual(
                    score_cache.read_for_budget(self.key, ERD_ALL, scope),
                    ("crane", 2.0, 3, scope))

    def test_creating_the_row_returns_what_was_stored(self):
        score_cache = self.cache()
        self.assertEqual(
            score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3),
            ("crane", 2.0, 3, None))
        self.assertEqual(score_cache.adopted_depth_count, 0)

    def test_the_memory_cache_holds_the_same_invariant(self):
        from cache_sqlite import MemoryScoreCache
        memory = MemoryScoreCache()
        memory.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3)
        self.assertEqual(
            memory.write(self.key, ERD_ALL, "slate", 2.0, max_depth=4),
            ("crane", 2.0, 3, None))
        self.assertEqual(memory.read_with_depth(self.key, ERD_ALL),
                         ("crane", 2.0, 3, None))
        with self.assertRaises(CacheWriteConflict):
            memory.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3)

    def _solve(self, answers, score_cache, budget=4):
        from wordle_engine import ResponseCache, _solve_subset
        return _solve_subset(answers, ResponseCache(answers), score_cache,
                             budget, None, answers, ERD_ALL,
                             None, None, None, None)

    def test_a_solve_returns_the_depth_the_cache_holds(self):
        # The engine must adopt before folding, so what _solve_subset hands
        # its caller is what a parent reads back.  Driving that needs a
        # competing incumbent the solve does not see until it writes -- the
        # real interleaving, where another worker stored first.
        from wordle_engine import SOLVED
        answers = ["crane", "slate", "trace", "stale"]
        branch_key = ScoreCache.encode_subset(answers)

        reference = self.cache('adopt-ref.sqlite3')
        status, cost, depth, _floor = self._solve(answers, reference)
        self.assertEqual(status, SOLVED)

        # Another worker got there first with an equal-cost strategy whose
        # worst case is one guess deeper.
        score_cache = self.cache('adopt-engine.sqlite3')
        score_cache.write(branch_key, ERD_ALL, "trace", cost,
                          max_depth=depth + 1, solve_budget=None)

        real_read = ScoreCache.read_for_budget

        def blind_to_this_branch(self, key, policy, budget):
            if key == branch_key:
                return None          # the stale miss this solve started from
            return real_read(self, key, policy, budget)

        with mock.patch.object(ScoreCache, 'read_for_budget',
                               blind_to_this_branch):
            _status, adopted_cost, adopted_depth, _f = self._solve(
                answers, score_cache)

        self.assertEqual(adopted_depth, depth + 1,
                         "solve returned its own worst case, not the stored one")
        self.assertEqual(adopted_cost, cost)
        stored = score_cache._read_stored_row(branch_key, ERD_ALL, None)
        self.assertEqual((adopted_cost, adopted_depth), (stored[1], stored[2]))

    def test_a_result_deleted_before_the_reconcile_is_not_resurrected(self):
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3)
        with mock.patch.object(ScoreCache, '_read_stored_row',
                               return_value=None):
            score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=4)
        # Nothing was written and nothing is mirrored, so the next read goes
        # back to the database rather than serving a value it never stored.
        self.assertIsNone(
            score_cache._mem_cache.get((self.key, ERD_ALL, None)))
        self.assertEqual(self.count(score_cache, CANONICAL), 1)

    def test_delete_clears_every_mirrored_scope_for_the_branch(self):
        # The mirror can hold a scope no query of the delete's own would list
        # -- one written after such a query and removed by the delete's WHERE
        # clause -- and serving that from memory outlives the row itself.
        score_cache = self.cache()
        score_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=4)
        score_cache.write(self.key, ERD_ALL, "slate", 2.5, max_depth=3,
                          solve_budget=3)
        self.assertIsNotNone(score_cache.read_with_depth(self.key, ERD_ALL))
        self.assertIsNotNone(score_cache.read_for_budget(self.key, ERD_ALL, 3))
        score_cache._mem_cache[(self.key, ERD_ALL, 4)] = ("trace", 2.2, 2, 4)

        score_cache.delete(self.key, ERD_ALL)
        for scope in (None, 3, 4):
            self.assertIsNone(
                score_cache._mem_cache.get((self.key, ERD_ALL, scope)),
                f"scope {scope} still mirrored after delete")
        self.assertIsNone(score_cache.read_for_budget(self.key, ERD_ALL, 4))


class ImportConflictTest(_CacheTest):
    """A key collision between two caches is checked, not assumed.

    Normally it is one fact reached twice.  When the costs disagree, one of
    the caches is wrong and whichever the merge keeps displaces a result the
    other's ancestors folded — the failure CacheWriteConflict refuses within a
    file, applied across files.
    """

    def _attached(self, target_path, source_path):
        conn = sqlite3.connect(target_path, isolation_level=None)
        conn.execute(f"ATTACH DATABASE '{source_path}' AS src")
        self.addCleanup(conn.close)
        return conn

    def _cache_with(self, name, guess, score, *, solve_budget=None,
                    max_depth=3, legacy_budget_in_canonical=False):
        import_cache = __import__('import_cache')
        path = os.path.join(self._dir, name)
        score_cache = ScoreCache(path, WORDS, checkpoint_on_close=False)
        if legacy_budget_in_canonical:
            score_cache._conn.execute(
                f"INSERT OR REPLACE INTO {CANONICAL} "
                "(branch_key, policy, answer_list_id, best_guess, best_score, "
                " updated_at, max_depth, solve_budget) VALUES (?,?,?,?,?,7,?,?)",
                (self.key, ERD_ALL, score_cache.answer_list_id, guess, score,
                 max_depth, solve_budget))
        else:
            score_cache.write(self.key, ERD_ALL, guess, score,
                              max_depth=max_depth, solve_budget=solve_budget)
        score_cache.close()
        del import_cache
        return path

    def _conflicts(self, target_path, source_path):
        import import_cache
        conn = self._attached(target_path, source_path)
        src_tables = {row[0] for row in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'")}
        return import_cache._conflicting_exact_results(conn, src_tables)

    def test_disagreeing_unrestricted_results_are_reported_both_directions(self):
        a = self._cache_with("a.sqlite3", "crane", 2.0)
        b = self._cache_with("b.sqlite3", "slate", 2.5)
        for target, source in ((a, b), (b, a)):
            with self.subTest(direction=os.path.basename(target)):
                conflicts = self._conflicts(target, source)
                self.assertEqual(len(conflicts), 1, conflicts)
                self.assertEqual(conflicts[0][0], 'unrestricted')

    def test_disagreeing_budget_results_are_reported_both_directions(self):
        a = self._cache_with("ba.sqlite3", "crane", 2.0, solve_budget=3)
        b = self._cache_with("bb.sqlite3", "slate", 2.5, solve_budget=3)
        for target, source in ((a, b), (b, a)):
            with self.subTest(direction=os.path.basename(target)):
                conflicts = self._conflicts(target, source)
                self.assertEqual(len(conflicts), 1, conflicts)
                self.assertEqual(conflicts[0][0], 'budget-specific')

    def test_a_legacy_budget_row_is_compared_where_it_would_be_routed(self):
        # It arrives in the source's canonical table but lands in the target's
        # budget table, so that is where a disagreement has to be caught.
        target = self._cache_with("lt.sqlite3", "crane", 2.0, solve_budget=3)
        source = self._cache_with("ls.sqlite3", "slate", 2.5, solve_budget=3,
                                  legacy_budget_in_canonical=True)
        conflicts = self._conflicts(target, source)
        self.assertEqual([c[0] for c in conflicts], ['budget-specific (routed)'])

    def test_a_different_strategy_at_the_same_cost_and_depth_is_one_fact(self):
        # Different guesses are harmless when both folded outputs agree: the
        # cost and the worst case are what an ancestor folded.
        a = self._cache_with("sa.sqlite3", "crane", 2.0)
        b = self._cache_with("sb.sqlite3", "slate", 2.0)
        self.assertEqual(self._conflicts(a, b), [])
        self.assertEqual(self._conflicts(b, a), [])

    def test_equal_cost_with_a_different_worst_case_is_a_conflict(self):
        # max_depth is ancestor-visible, so equal cost does not make two
        # certificates interchangeable: keeping the target's child while
        # admitting source parents folded from the source's depth makes those
        # parents inconsistent on arrival.
        a = self._cache_with("da.sqlite3", "crane", 2.0, max_depth=3)
        b = self._cache_with("db.sqlite3", "slate", 2.0, max_depth=4)
        for target, source in ((a, b), (b, a)):
            with self.subTest(direction=os.path.basename(target)):
                conflicts = self._conflicts(target, source)
                self.assertEqual(len(conflicts), 1, conflicts)
                self.assertNotEqual(conflicts[0][4][2], conflicts[0][5][2])

    def test_a_source_parent_is_not_admitted_above_a_retained_child(self):
        # End to end: the source holds a child at depth 4 and a parent folded
        # from it; the target holds the same child at depth 3.  INSERT OR
        # IGNORE would keep the target's child and import the source's parent,
        # leaving that parent describing a subtree the retained child does not
        # support.  The merge has to refuse before any row moves.
        child = ScoreCache.encode_subset(WORDS[:2])
        parent = ScoreCache.encode_subset(WORDS[:3])

        target_path = os.path.join(self._dir, "pt.sqlite3")
        target = ScoreCache(target_path, WORDS, checkpoint_on_close=False)
        target.write(child, ERD_ALL, "crane", 2.0, max_depth=3)
        target.close()

        source_path = os.path.join(self._dir, "ps.sqlite3")
        source = ScoreCache(source_path, WORDS, checkpoint_on_close=False)
        source.write(child, ERD_ALL, "slate", 2.0, max_depth=4)
        source.write(parent, ERD_ALL, "trace", 2.5, max_depth=5)
        source.close()

        conflicts = self._conflicts(target_path, source_path)
        self.assertEqual(len(conflicts), 1, conflicts)
        self.assertEqual(conflicts[0][1], branch_reference(child))

    def test_a_scope_only_one_cache_holds_is_not_a_conflict(self):
        target = self._cache_with("oa.sqlite3", "crane", 2.0)
        source = self._cache_with("ob.sqlite3", "slate", 2.5, solve_budget=3)
        self.assertEqual(self._conflicts(target, source), [])

    def test_the_report_names_both_sides(self):
        import io
        import import_cache
        from contextlib import redirect_stderr
        conflicts = [('unrestricted', 'abc123def456', ERD_ALL, None,
                      ('crane', 2.0, 3), ('slate', 2.0, 4))]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            import_cache._report_conflicts(conflicts)
        report = stderr.getvalue()
        self.assertIn('abc123def456', report)
        self.assertIn('crane', report)
        self.assertIn('slate', report)
        # The worst cases are what differ here, so the report has to show them.
        self.assertIn('depth 3', report)
        self.assertIn('depth 4', report)

    def test_a_target_without_the_budget_table_is_skipped_not_an_error(self):
        # An older target has no budget table to collide with; the comparison
        # that would reach it is passed over rather than failing the import.
        import import_cache
        source = self._cache_with("nt_src.sqlite3", "slate", 2.5, solve_budget=3)
        target = os.path.join(self._dir, "nt_target.sqlite3")
        target_cache = ScoreCache(target, WORDS, checkpoint_on_close=False)
        target_cache.write(self.key, ERD_ALL, "crane", 2.0, max_depth=3)
        target_cache._conn.execute(f"DROP TABLE {BUDGETED}")
        target_cache.close()

        conn = self._attached(target, source)
        src_tables = {row[0] for row in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'")}
        self.assertEqual(
            import_cache._conflicting_exact_results(conn, src_tables), [])

    def test_the_conflict_report_is_bounded(self):
        import import_cache
        target_cache = self.cache("many_t.sqlite3")
        source_cache = self.cache("many_s.sqlite3")
        for index in range(4):
            branch = ScoreCache.encode_subset(WORDS[:2] + [f"aaa{index}a"])
            target_cache.write(branch, ERD_ALL, "crane", 2.0, max_depth=3)
            source_cache.write(branch, ERD_ALL, "slate", 2.5, max_depth=3)
        target_cache.close()
        source_cache.close()

        conn = self._attached(os.path.join(self._dir, "many_t.sqlite3"),
                              os.path.join(self._dir, "many_s.sqlite3"))
        src_tables = {row[0] for row in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table'")}
        conflicts = import_cache._conflicting_exact_results(
            conn, src_tables, limit=2)
        self.assertEqual(len(conflicts), 2)


class EquivalenceRuleTest(unittest.TestCase):
    """The rule that decides sameness, in its two expressions."""

    CASES = [
        ((2.0, 3), (2.0, 3), True),
        ((2.0, 3), (2.0, 4), False),
        ((2.0, 3), (2.5, 3), False),
        ((2.0, 3), (2.5, 4), False),
        ((2.0, None), (2.0, None), True),
        ((2.0, None), (2.0, 3), False),
        ((2.0, 3), (2.0 + 1e-12, 3), True),
    ]

    def test_the_python_rule_is_cost_and_worst_case(self):
        for (stored, incoming, agree) in self.CASES:
            with self.subTest(stored=stored, incoming=incoming):
                self.assertEqual(
                    exact_results_agree(stored[0], stored[1],
                                        incoming[0], incoming[1]),
                    agree)

    def test_the_sql_equivalence_rule_matches_the_python_one(self):
        # import_cache compares whole tables in SQL; the two expressions of
        # one rule have to answer alike or a merge and a write disagree.
        import import_cache
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        for (stored, incoming, agree) in self.CASES:
            with self.subTest(stored=stored, incoming=incoming):
                flagged = conn.execute(
                    "SELECT (abs(?1 - ?3) > ?5 OR ?2 IS NOT ?4)",
                    (stored[0], stored[1], incoming[0], incoming[1],
                     import_cache.EXACT_SCORE_TOLERANCE)).fetchone()[0]
                self.assertEqual(not flagged, agree)
