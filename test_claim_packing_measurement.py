"""Unit tests for the adaptive-claim-packing measurement layer (issue #67, §9).

Covers the telemetry/epoch instrumentation and the budget-keyed cost model that
land before the packer:
  - cost_model re-key migration: an old (policy, size_bucket) DB carries its
    accumulators forward as the budget-aggregate (-1) cell;
  - budget-keyed reads with the specific-cell -> aggregate fallback;
  - the new telemetry inserts (branch_finalize_log, candidate_accuracy) and the
    extended cost_samples / claim_telemetry columns;
  - epoch tagging (baseline epoch 0, set_epoch);
  - estimate_candidate_work (the metric the §10 go/no-go gate validates).
"""
import math
import os
import sqlite3
import tempfile
import unittest

from erd_queue import ERDQueue, cost_size_bucket, _AGGREGATE_BUDGET
from wordle_engine import (estimate_candidate_work, estimate_candidate_work_cutoff,
                           evaluate_candidate, ERD_ANSWERS, ERD_ALL)


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.path)
        self.addCleanup(self.q.close)


class TestEpochBaseline(_TmpQueue):
    def test_fresh_db_has_baseline_epoch_zero(self):
        self.assertEqual(self.q.epoch, 0)
        self.assertEqual(self.q.get_meta("epoch"), "0")
        row = self.q._conn.execute(
            "SELECT label FROM telemetry_epoch WHERE epoch = 0").fetchone()
        self.assertIsNotNone(row)

    def test_set_epoch_activates_and_stamps(self):
        self.q.set_epoch(1, label="pack+overrun", git_sha="abc123")
        self.assertEqual(self.q.epoch, 1)
        self.assertEqual(self.q.get_meta("epoch"), "1")
        # New telemetry now carries epoch 1.
        self.q.add_claim_telemetry(50, 12, 3, 6)
        ep = self.q._conn.execute(
            "SELECT epoch FROM claim_telemetry").fetchone()["epoch"]
        self.assertEqual(ep, 1)


class TestCostModelBudgetKey(_TmpQueue):
    def test_specific_budget_cell_isolated_from_other_budget(self):
        self.q.update_cost_model(ERD_ALL, 100, 1000, budget=3)
        self.q.update_cost_model(ERD_ALL, 100, 1_000_000, budget=5)
        self.assertAlmostEqual(
            self.q.get_cost_typical(ERD_ALL, 100, budget=3), 1000, delta=1)
        self.assertAlmostEqual(
            self.q.get_cost_typical(ERD_ALL, 100, budget=5), 1_000_000, delta=1)

    def test_cold_specific_cell_falls_back_to_aggregate(self):
        # Warm budget=3 only; reading an unseen budget=4 falls back to the
        # budget-aggregate, which every sample also feeds.
        self.q.update_cost_model(ERD_ALL, 100, 1000, budget=3)
        self.assertIsNotNone(self.q.get_cost_typical(ERD_ALL, 100, budget=4))
        self.assertAlmostEqual(
            self.q.get_cost_typical(ERD_ALL, 100, budget=4), 1000, delta=1)

    def test_budget_none_reads_aggregate(self):
        self.q.update_cost_model(ERD_ALL, 100, 2000, budget=2)
        # No budget arg == aggregate, warm because every sample feeds -1.
        self.assertAlmostEqual(
            self.q.get_cost_typical(ERD_ALL, 100), 2000, delta=1)

    def test_aggregate_row_present_under_minus_one(self):
        self.q.update_cost_model(ERD_ALL, 100, 1234, budget=2)
        row = self.q._cost_bucket_row(ERD_ALL, 100, _AGGREGATE_BUDGET)
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["weight_sum"], 1.0)

    def test_spread_falls_back_to_aggregate(self):
        for x in (100, 1000, 10000):
            self.q.update_cost_model(ERD_ALL, 100, x, budget=2)
        # Unseen budget reads the aggregate spread rather than None.
        self.assertIsNotNone(self.q.get_cost_spread(ERD_ALL, 100, budget=9))


class TestCostModelMigration(unittest.TestCase):
    """An old (policy, size_bucket)-keyed cost_model carries forward to -1."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "old.sqlite3")
        bucket = cost_size_bucket(100)
        log_n = math.log(1000)
        conn = sqlite3.connect(self.path)
        conn.executescript("""
            CREATE TABLE cost_model (
                policy           TEXT    NOT NULL,
                size_bucket      INTEGER NOT NULL,
                weighted_log_sum REAL    NOT NULL DEFAULT 0,
                weight_sum       REAL    NOT NULL DEFAULT 0,
                weighted_log_sq  REAL    NOT NULL DEFAULT 0,
                last_updated     INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (policy, size_bucket)
            );
        """)
        conn.execute(
            "INSERT INTO cost_model VALUES (?, ?, ?, ?, ?, 0)",
            (ERD_ALL, bucket, log_n * 5, 5.0, log_n * log_n * 5))
        conn.commit()
        conn.close()

    def test_old_rows_become_budget_aggregate(self):
        q = ERDQueue(self.path)
        self.addCleanup(q.close)
        cols = {r["name"]
                for r in q._conn.execute("PRAGMA table_info(cost_model)")}
        self.assertIn("budget", cols)
        # The migrated accumulator is readable as the aggregate (and via any
        # budget through the fallback), preserving the typical value (~1000).
        self.assertAlmostEqual(q.get_cost_typical(ERD_ALL, 100), 1000, delta=1)
        self.assertAlmostEqual(
            q.get_cost_typical(ERD_ALL, 100, budget=4), 1000, delta=1)

    def test_migration_is_idempotent(self):
        ERDQueue(self.path).close()
        q = ERDQueue(self.path)   # second open must not fail or double-rebuild
        self.addCleanup(q.close)
        self.assertAlmostEqual(q.get_cost_typical(ERD_ALL, 100), 1000, delta=1)


class TestTelemetryInserts(_TmpQueue):
    def test_cost_sample_records_budget_wall_censored_epoch(self):
        self.q.add_cost_sample(ERD_ALL, 30, 500, "finalize",
                               budget=4, wall_millis=1234)
        self.q.add_cost_sample(ERD_ALL, 30, 999, "censored",
                               budget=4, censored=1)
        rows = self.q._conn.execute(
            "SELECT nodes, wall_millis, budget, censored, source, epoch "
            "FROM cost_samples ORDER BY id").fetchall()
        self.assertEqual(rows[0]["wall_millis"], 1234)
        self.assertEqual(rows[0]["budget"], 4)
        self.assertEqual(rows[0]["censored"], 0)
        self.assertEqual(rows[1]["censored"], 1)
        self.assertEqual(rows[1]["source"], "censored")
        self.assertEqual(rows[0]["epoch"], 0)

    def test_claim_telemetry_carries_contention_and_epoch(self):
        self.q._last_claim_busy_millis = 7
        self.q._last_claim_retries = 0
        self.q.add_claim_telemetry(40, 21, 1500, 6)
        row = self.q._conn.execute(
            "SELECT busy_wait_millis, claim_retries, epoch "
            "FROM claim_telemetry").fetchone()
        self.assertEqual(row["busy_wait_millis"], 7)
        self.assertEqual(row["claim_retries"], 0)
        self.assertEqual(row["epoch"], 0)

    def test_branch_finalize_log_persists_branch_timing(self):
        self.q.add_branch_finalize_log(
            b"key", "SALET --g-- ", 30, 4, created_at=100,
            finalized_at=160, nodes_spent=12345, n_claims=2315)
        row = self.q._conn.execute(
            "SELECT * FROM branch_finalize_log").fetchone()
        self.assertEqual(row["n_words"], 30)
        self.assertEqual(row["budget"], 4)
        self.assertEqual(row["nodes_spent"], 12345)
        self.assertEqual(row["n_claims"], 2315)
        self.assertEqual(row["epoch"], 0)
        # Packer-era columns stay NULL under single-candidate claiming.
        self.assertIsNone(row["max_bundle_nodes"])

    def test_candidate_accuracy_records_prediction_point(self):
        self.q.add_candidate_accuracy(
            b"key", 30, 4, predicted_work=820.0, bound_erd=3.41,
            cost_lb=2.9, gated=False, actual_nodes=771)
        self.q.add_candidate_accuracy(
            b"key", 30, 4, predicted_work=0.0, bound_erd=3.41,
            cost_lb=3.5, gated=True, actual_nodes=4)
        rows = self.q._conn.execute(
            "SELECT predicted_work, gated, actual_nodes, epoch "
            "FROM candidate_accuracy ORDER BY id").fetchall()
        self.assertEqual(rows[0]["gated"], 0)
        self.assertAlmostEqual(rows[0]["predicted_work"], 820.0)
        self.assertEqual(rows[1]["gated"], 1)
        self.assertEqual(rows[1]["actual_nodes"], 4)
        self.assertEqual(rows[0]["epoch"], 0)


class TestEstimateCandidateWork(unittest.TestCase):
    """estimate_candidate_work: gated -> 0, else sum typical over recursed groups."""

    @staticmethod
    def _warm(value):
        return lambda k, budget: value

    def test_gated_candidate_predicts_zero(self):
        # cost_lb = 3 - (2+0)/7 ~= 2.714; best_erd below it -> provably gated.
        work = estimate_candidate_work(
            [4, 3], has_self=False, n=7, best_erd=2.0, budget=4,
            typical=self._warm(10.0))
        self.assertEqual(work, 0.0)

    def test_non_gated_sums_typical_over_recursed_groups(self):
        # cost_lb = 3 - (4+1)/7 ~= 2.286 < best_erd 3.0 -> not gated.
        # Groups [2,3,1,1]: singletons skipped, two recursed -> 10 + 10.
        work = estimate_candidate_work(
            [2, 3, 1, 1], has_self=True, n=7, best_erd=3.0, budget=4,
            typical=self._warm(10.0))
        self.assertAlmostEqual(work, 20.0)

    def test_cold_model_falls_back_to_group_size(self):
        work = estimate_candidate_work(
            [2, 3, 1], has_self=False, n=7, best_erd=3.0, budget=4,
            typical=lambda k, b: None)
        self.assertAlmostEqual(work, 5.0)   # 2 + 3, singleton skipped

    def test_full_branch_group_excluded(self):
        # A single group of size n gains no information -> contributes nothing.
        work = estimate_candidate_work(
            [7], has_self=False, n=7, best_erd=3.0, budget=4,
            typical=self._warm(99.0))
        self.assertEqual(work, 0.0)

    def test_sub_budget_threaded_to_typical(self):
        seen = {}

        def typical(k, budget):
            seen["budget"] = budget
            return 1.0
        estimate_candidate_work(
            [2], has_self=False, n=7, best_erd=3.0, budget=4, typical=typical)
        self.assertEqual(seen["budget"], 3)   # budget - 1


class TestCutoffMetric(unittest.TestCase):
    """estimate_candidate_work_cutoff stops at the engine's accumulated-cost cutoff,
    so a weak splitter near the bound predicts ~0 where the uncut sum predicts large."""

    @staticmethod
    def _warm(value):
        return lambda k, budget: value

    def test_gated_predicts_zero(self):
        # cost_lb = 3 - (2)/7 ~= 2.71; bound below it -> gated.
        self.assertEqual(
            estimate_candidate_work_cutoff([4, 3], False, 7, 2.0, 4, self._warm(99.0)),
            0.0)

    def test_inf_bound_falls_back_to_uncut(self):
        sizes, n = [3, 2, 2], 7
        self.assertEqual(
            estimate_candidate_work_cutoff(sizes, False, n, float("inf"), 4, self._warm(10.0)),
            estimate_candidate_work(sizes, False, n, float("inf"), 4, self._warm(10.0)))

    def test_weak_splitter_cut_early_predicts_far_less_than_uncut(self):
        # Several large groups, cost_lb just under the bound: the estimated ERD
        # accumulation busts the bound after the first couple of groups, so later
        # groups are never reached and not counted — far below the uncut sum.
        sizes, n, bound = [40, 30, 20, 10], 80, 2.96
        cut = estimate_candidate_work_cutoff(sizes, False, n, bound, 4, self._warm(1000.0))
        uncut = estimate_candidate_work(sizes, False, n, bound, 4, self._warm(1000.0))
        self.assertLess(cut, uncut)

    def test_strong_splitter_reaches_many_groups(self):
        # Many small groups, cost_lb far below the bound: the cutoff is not reached
        # early, so most groups contribute -> a large estimate.
        sizes = [2] * 20
        n = 41
        work = estimate_candidate_work_cutoff(sizes, False, n, 3.0, 4, self._warm(5.0))
        self.assertGreater(work, 5.0 * 5)   # many groups counted, not just one

    def test_cutoff_never_underestimates_reached_groups(self):
        # Lower-bound accumulation => cutoff fires no earlier than the engine's,
        # so cutoff work is an UPPER bound on the truly-reached groups: <= uncut.
        sizes, n = [10, 8, 6, 4, 2], 41
        cut = estimate_candidate_work_cutoff(sizes, False, n, 2.9, 4, self._warm(7.0))
        uncut = estimate_candidate_work(sizes, False, n, 2.9, 4, self._warm(7.0))
        self.assertLessEqual(cut, uncut)


class TestCandidateAccuracyGroupSizes(_TmpQueue):
    def test_group_sizes_persisted(self):
        self.q.add_candidate_accuracy(
            b"k", 30, 4, 820.0, 3.41, 2.9, gated=False, actual_nodes=771,
            group_sizes="12-8-5-3-1")
        row = self.q._conn.execute(
            "SELECT group_sizes FROM candidate_accuracy").fetchone()
        self.assertEqual(row["group_sizes"], "12-8-5-3-1")

    def test_group_sizes_optional_defaults_null(self):
        self.q.add_candidate_accuracy(
            b"k", 30, 4, 0.0, 3.41, 3.5, gated=True, actual_nodes=4)
        row = self.q._conn.execute(
            "SELECT group_sizes FROM candidate_accuracy").fetchone()
        self.assertIsNone(row["group_sizes"])

    def test_source_word_persisted_for_per_opener_segmentation(self):
        self.q.add_candidate_accuracy(
            b"k", 30, 4, 820.0, 3.41, 2.9, gated=False, actual_nodes=771,
            group_sizes="12-8-5", source_word="salet")
        row = self.q._conn.execute(
            "SELECT source_word FROM candidate_accuracy").fetchone()
        self.assertEqual(row["source_word"], "salet")


class TestMetricObserverHook(unittest.TestCase):
    """evaluate_candidate reports the work-metric inputs exactly once per call."""

    def test_observer_called_with_costlb_groups_bound(self):
        remaining = ["crane", "slate", "trace", "stale", "tales"]
        seen = []

        def observer(group_sizes, has_self, cost_lb, bound, gated):
            seen.append((group_sizes, has_self, cost_lb, bound, gated))

        evaluate_candidate(remaining, "crane", None, None,
                           guesses=remaining, policy=ERD_ANSWERS,
                           best_erd=float('inf'), metric_observer=observer)
        self.assertEqual(len(seen), 1)
        group_sizes, has_self, cost_lb, bound, gated = seen[0]
        self.assertEqual(sum(group_sizes), len(remaining))
        # Loose (inf) bound never gates: cost_lb >= inf is impossible.
        self.assertFalse(gated)
        self.assertEqual(bound, float('inf'))

    def test_observer_reports_gated_under_tight_bound(self):
        remaining = ["crane", "slate", "trace", "stale", "tales"]
        seen = []

        def observer(group_sizes, has_self, cost_lb, bound, gated):
            seen.append(gated)

        # A bound at the cost-floor (1.0) gates every candidate immediately.
        evaluate_candidate(remaining, "crane", None, None,
                           guesses=remaining, policy=ERD_ANSWERS,
                           best_erd=1.0, metric_observer=observer)
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
