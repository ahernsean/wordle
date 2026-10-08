"""An opener's cost is read from its own branches' finalize rows, through
the spine index rather than a scan of the whole log."""

import os
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue


class _Log(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "queue.sqlite3"))
        self.addCleanup(self.q.close)
        self.now = int(time.time())

    def _finalized(self, spine, nodes, evaluation_time_millis=100,
                   coordination_time_millis=10, wall_time_millis=150,
                   created_ago=60):
        self.q.add_branch_finalize_log(
            ScoreCache.encode_subset([spine[:5].lower()]), spine, 5, 4,
            self.now - created_ago, self.now, nodes, 2, n_bundles=1,
            total_bundle_wall_time_millis=wall_time_millis,
            evaluation_time_millis=evaluation_time_millis,
            coordination_time_millis=coordination_time_millis)

    def _plans_of(self, call):
        """The query plan of every statement `call` runs on the finalize log,
        captured as SQLite expands it, so the plan is of the method's own
        query and not of a copy of it."""
        statements = []
        self.q._conn.set_trace_callback(statements.append)
        try:
            call()
        finally:
            self.q._conn.set_trace_callback(None)
        return [" ".join(row["detail"] for row in self.q._conn.execute(
                    "EXPLAIN QUERY PLAN " + statement))
                for statement in statements
                if "branch_finalize_log" in statement]

    def assertEveryLookupUsesTheSpineIndex(self, call):
        plans = self._plans_of(call)
        self.assertTrue(plans)
        for plan in plans:
            self.assertIn("idx_branch_finalize_log_spine", plan)


class TestCompletedOpenerTiming(_Log):

    def test_an_opener_totals_only_its_own_branches(self):
        self._finalized("CRANE -----", 100, created_ago=600)
        self._finalized("CRANE ----- LUBES -y---", 40)
        self._finalized("CRANK -----", 9_999)
        self._finalized("CRAN", 9_999)

        timing = self.q.completed_opener_timing("crane")

        self.assertEqual(timing["search_node_count"], 140)
        self.assertEqual(timing["evaluation_time_millis"], 200)
        self.assertEqual(timing["coordination_time_millis"], 20)
        self.assertEqual(timing["worker_time_millis"], 300)
        self.assertEqual(timing["first_created_at"], self.now - 600)
        self.assertEqual(timing["completed_at"], self.now)

    def test_a_branch_missing_a_figure_withholds_that_total(self):
        self._finalized("CRANE -----", 100)
        self._finalized("CRANE ----y", 40, evaluation_time_millis=None)

        timing = self.q.completed_opener_timing("CRANE")

        self.assertIsNone(timing["evaluation_time_millis"])
        self.assertEqual(timing["coordination_time_millis"], 20)
        self.assertEqual(timing["search_node_count"], 140)

    def test_the_lookup_is_an_index_range(self):
        self.assertEveryLookupUsesTheSpineIndex(
            lambda: self.q.completed_opener_timing("crane"))


class TestSpineSubtreeRollup(_Log):

    def test_a_subtree_is_its_root_and_every_descendant(self):
        self._finalized("CRANE -----", 100)
        self._finalized("CRANE ----- LUBES -y---", 40)
        self._finalized("CRANE ----- LUBES -y--- PILCH ----g", 7)
        self._finalized("CRANE ----y", 9_000)
        self._finalized("CRANE -----X", 9_000)

        rollup = self.q.roll_up_spine_subtrees(["CRANE -----"])["CRANE -----"]

        self.assertEqual(rollup["branch_count"], 3)
        self.assertEqual(rollup["search_node_count"], 147)
        self.assertEqual(rollup["wall_time_millis"], 450)

    def test_an_empty_subtree_reports_nothing_spent(self):
        rollup = self.q.roll_up_spine_subtrees(["SALET -----"])["SALET -----"]
        self.assertEqual(rollup["branch_count"], 0)
        self.assertIsNone(rollup["first_created_at"])

    def test_a_rollup_is_an_index_range(self):
        self.assertEveryLookupUsesTheSpineIndex(
            lambda: self.q.roll_up_spine_subtrees(["CRANE -----"]))

    def test_root_progress_groups_through_the_index(self):
        self._finalized("CRANE -----", 100)
        self.assertEveryLookupUsesTheSpineIndex(
            lambda: self.q.report_root_progress("CRANE", None, 600))


class TestCompletedOpenerSummaryColumns(unittest.TestCase):

    def test_an_opener_summary_keeps_its_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ScoreCache(os.path.join(directory, "cache.sqlite3"),
                               ["crane", "slate"], checkpoint_on_close=False)
            try:
                cache.write_completed_opener_summary(
                    "crane", "erd_all", 100, 60_000, 4_000, (23,),
                    search_node_count=900, evaluation_time_millis=3_000,
                    coordination_time_millis=120)
                summary = cache.completed_opener_summary_map(
                    "erd_all")["crane"]
            finally:
                cache.close()
        self.assertEqual(
            (summary["search_node_count"], summary["evaluation_time_millis"],
             summary["coordination_time_millis"]), (900, 3_000, 120))


if __name__ == "__main__":
    unittest.main()
