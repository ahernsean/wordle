"""The fold that turns an opener's response groups into its own ERD."""

import os
import sqlite3
import tempfile
import unittest

from cache_sqlite import ScoreCache, answer_list_id
from opener_erd import (
    ALL_GREEN_PATTERN_TEXT,
    fold_opener,
    opener_response_groups,
    response_groups_from_patterns,
    store_opener_verdict,
)
from pattern_matrix import PatternMatrix
from wordle_engine import ERD_ALL, GAME_GUESSES

ANSWERS = ["crane", "slate", "shale", "stale", "brine", "swine"]
GROUP_BUDGET = GAME_GUESSES - 1


class OpenerResponseGroupTest(unittest.TestCase):
    def test_a_pattern_no_answer_produces_is_not_a_group(self):
        groups = response_groups_from_patterns({0: [], 1: ["crane"], 2: []})
        self.assertEqual([(pattern, count) for pattern, count, _key in groups],
                         [("----y", 1)])

    def test_groups_are_ordered_by_pattern_whatever_partitioned_them(self):
        # A pattern matrix and a response cache partition the same opener by
        # different machinery, and the fold is handed the result of either.  An
        # order that depended on which one ran would make two callers' stored
        # verdicts differ in nothing but their group order.
        unordered = {7: ["brine"], 2: ["crane"], 5: ["slate", "stale"]}
        groups = response_groups_from_patterns(unordered)
        self.assertEqual([count for _pattern, count, _key in groups],
                         [1, 2, 1])

    def test_a_group_carries_the_branch_key_of_its_answers(self):
        groups = response_groups_from_patterns({1: ["slate", "crane"]})
        self.assertEqual(groups[0][2],
                         ScoreCache.encode_subset(["slate", "crane"]))


class FoldOpenerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = os.path.join(self.directory.name, "cache.sqlite3")
        self.cache = ScoreCache(self.cache_path, ANSWERS,
                                checkpoint_on_close=False)
        self.addCleanup(self.cache.close)

    def _groups(self, opener, candidates=("gippy", "raise")):
        matrix = PatternMatrix.load_or_build(
            self.cache_path, list(candidates), ANSWERS, self.cache)
        return opener_response_groups(matrix, opener, ANSWERS)

    def test_an_opener_partitions_the_whole_answer_list(self):
        groups = self._groups("raise")
        self.assertEqual(sum(count for _p, count, _k in groups), len(ANSWERS))

    def test_an_opener_that_is_an_answer_holds_an_all_green_group(self):
        groups = self._groups("crane", candidates=("crane", "raise"))
        self.assertIn(ALL_GREEN_PATTERN_TEXT,
                      [pattern for pattern, _count, _key in groups])

    def test_a_group_with_no_stored_result_leaves_the_opener_pending(self):
        summary = fold_opener(
            self.cache, self._groups("gippy"), ERD_ALL, GROUP_BUDGET)
        self.assertEqual(summary["state"], "pending")
        self.assertIsNone(summary["erd"])

    def test_a_proven_loss_in_a_group_makes_the_opener_infeasible(self):
        groups = self._groups("gippy")
        self.cache.write_loss(groups[0][2], ERD_ALL, GROUP_BUDGET)
        summary = fold_opener(
            self.cache, groups, ERD_ALL, GROUP_BUDGET)
        self.assertEqual(summary["state"], "infeasible")

    def test_a_result_solved_at_another_budget_is_not_folded_in(self):
        # The fold reads each group at the budget the opener would play it at,
        # so a result that only exists at some other budget is missing rather
        # than close enough -- the same scope rule the solver reuses a child
        # under.
        groups = self._groups("gippy")
        for _pattern, _count, key in groups:
            self.cache.write(key, ERD_ALL, "crane", 2.0, max_depth=2,
                             solve_budget=GROUP_BUDGET - 1)
        self.assertEqual(
            fold_opener(self.cache, groups, ERD_ALL, GROUP_BUDGET)["state"],
            "pending")
        self.assertEqual(
            fold_opener(self.cache, groups, ERD_ALL, GROUP_BUDGET - 1)["state"],
            "complete")


class StoreOpenerVerdictTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = os.path.join(self.directory.name, "cache.sqlite3")
        self.cache = ScoreCache(self.cache_path, ANSWERS,
                                checkpoint_on_close=False)
        self.addCleanup(self.cache.close)
        matrix = PatternMatrix.load_or_build(
            self.cache_path, ["gippy", "raise"], ANSWERS, self.cache)
        self.groups = opener_response_groups(matrix, "gippy", ANSWERS)

    def _store(self):
        return store_opener_verdict(self.cache, "gippy", self.groups,
                                    ERD_ALL, GROUP_BUDGET)

    def test_a_complete_fold_is_stored_with_its_erd(self):
        for _pattern, _count, key in self.groups:
            self.cache.write(key, ERD_ALL, "crane", 2.0, max_depth=2)
        summary = self._store()
        self.assertEqual(summary["state"], "complete")
        verdict = self.cache.opener_erd_verdict(ERD_ALL, "gippy")
        self.assertEqual(verdict["state"], "complete")
        self.assertEqual(verdict["erd"], summary["erd"])
        self.assertEqual(verdict["max_remaining_depth"],
                         summary["max_remaining_depth"])
        self.assertEqual(verdict["response_group_count"], len(self.groups))

    def test_an_infeasible_fold_is_stored_with_no_erd(self):
        # The opener finished and cannot be solved within budget, so it has no
        # finite ERD to rank and the row says so rather than carrying a number
        # that would sort somewhere.
        self.cache.write_loss(self.groups[0][2], ERD_ALL, GROUP_BUDGET)
        for _pattern, _count, key in self.groups[1:]:
            self.cache.write(key, ERD_ALL, "crane", 2.0, max_depth=2)
        self.assertEqual(self._store()["state"], "infeasible")
        verdict = self.cache.opener_erd_verdict(ERD_ALL, "gippy")
        self.assertEqual(verdict["state"], "infeasible")
        self.assertIsNone(verdict["erd"])
        self.assertIsNone(verdict["max_remaining_depth"])
        self.assertEqual(
            self.cache.opener_erd_ranking(ERD_ALL), [],
            "an opener with no finite ERD was ranked")

    def test_a_pending_fold_stores_nothing(self):
        # Reaching here says the tree is finished, so a group with no result
        # means one is missing from this cache.  A row would assert a tree that
        # is not there; no row reports the opener as unfinished, which is what
        # it is.
        self.assertEqual(self._store()["state"], "pending")
        self.assertIsNone(self.cache.opener_erd_verdict(ERD_ALL, "gippy"))
        self.assertEqual(self.cache.opener_erd_verdict_counts(ERD_ALL),
                         {"complete": 0, "infeasible": 0,
                          "maximum_response_group_count": 0})


# The pre-verdict DDL, verbatim, so the test builds exactly the schema older
# code left behind: a row meant "complete" by existing at all.
_LEGACY_OPENER_ERD_SCHEMA = """
CREATE TABLE opener_erd_by_policy (
    opener TEXT NOT NULL, policy TEXT NOT NULL,
    answer_list_id TEXT NOT NULL, erd REAL NOT NULL,
    max_remaining_depth INTEGER NOT NULL,
    response_group_count INTEGER NOT NULL,
    folded_at INTEGER NOT NULL,
    PRIMARY KEY (opener, policy, answer_list_id)
);
CREATE INDEX idx_opener_erd_rank ON opener_erd_by_policy
    (policy, answer_list_id, erd, max_remaining_depth, opener);
"""


class OpenerVerdictMigrationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "legacy.sqlite3")
        legacy = sqlite3.connect(self.path)
        legacy.executescript(_LEGACY_OPENER_ERD_SCHEMA)
        legacy.execute(
            "INSERT INTO opener_erd_by_policy VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("tarse", ERD_ALL, answer_list_id(ANSWERS), 3.5562, 5, 94, 1000))
        legacy.commit()
        legacy.close()

    def _open(self):
        cache = ScoreCache(self.path, ANSWERS, checkpoint_on_close=False)
        self.addCleanup(cache.close)
        return cache

    def test_a_legacy_row_keeps_its_erd_and_becomes_complete(self):
        # Nothing but a complete fold was ever stored, so every legacy row is
        # one.
        cache = self._open()
        row = cache._conn.execute(
            "SELECT opener, state, erd, max_remaining_depth,"
            " response_group_count, folded_at FROM opener_erd_by_policy"
        ).fetchone()
        self.assertEqual(tuple(row), ("tarse", "complete", 3.5562, 5, 94, 1000))

    def test_the_rebuilt_table_can_hold_a_verdict_with_no_erd(self):
        # The point of the rebuild: `erd NOT NULL` left an opener the swarm
        # finished and found unsolvable with nowhere to go.
        cache = self._open()
        cache.write_opener_erd("gippy", ERD_ALL, "infeasible", None, None, 7)
        self.assertEqual(
            cache.opener_erd_verdict(ERD_ALL, "gippy")["state"], "infeasible")

    def test_the_ranking_index_is_rebuilt_on_the_verdict_first_key(self):
        # `CREATE INDEX IF NOT EXISTS` is a no-op against an index that already
        # exists under the old column list, so an index left in place would
        # keep the legacy definition forever and the ranking's own range scan
        # would be a filter over every row.
        cache = self._open()
        definition = cache._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_opener_erd_rank'"
        ).fetchone()[0]
        self.assertIn("state", definition)
        plan = cache._conn.execute(
            "EXPLAIN QUERY PLAN SELECT erd, max_remaining_depth, opener"
            " FROM opener_erd_by_policy WHERE policy = ?"
            " AND answer_list_id = ? AND state = 'complete'"
            " ORDER BY erd, max_remaining_depth, opener",
            (ERD_ALL, cache.answer_list_id)).fetchall()
        detail = " ".join(str(step[-1]) for step in plan)
        self.assertIn("idx_opener_erd_rank", detail)
        self.assertNotIn("TEMP B-TREE", detail,
                         "the ranking had to sort rather than walk the index")

    def test_the_migration_leaves_the_ranking_index_on_its_own(self):
        """The migration must not depend on being followed by schema setup.

        An index follows its table through a RENAME and keeps its name, so
        `CREATE INDEX IF NOT EXISTS` no-ops against the legacy index and
        dropping the legacy table then takes the only index with it -- leaving
        the rebuilt table unindexed.  `_ensure_schema` calls the schema setup
        again after the migration and would rebuild it, which is precisely why
        this asserts what the migration itself leaves behind.
        """
        cache = self._open()
        cache._conn.execute("DELETE FROM schema_migrations WHERE name = ?",
                            ("opener_erd_verdict_state",))
        cache._conn.executescript(
            "DROP TABLE opener_erd_by_policy;" + _LEGACY_OPENER_ERD_SCHEMA)

        cache._migrate_opener_erds_to_state()

        definition = cache._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_opener_erd_rank'"
        ).fetchone()
        self.assertIsNotNone(definition, "the migration left no ranking index")
        self.assertIn("state", definition[0])

    def test_the_rebuild_does_not_run_twice(self):
        cache = self._open()
        cache.close()
        again = ScoreCache(self.path, ANSWERS, checkpoint_on_close=False)
        self.addCleanup(again.close)
        self.assertEqual(
            again._conn.execute(
                "SELECT COUNT(*) FROM opener_erd_by_policy").fetchone()[0], 1)
        self.assertFalse(again._table_exists("opener_erd_legacy"))
