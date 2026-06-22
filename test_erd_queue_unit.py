"""Unit tests for ERDQueue: covers methods not exercised by the integration tests.

The integration tests (test_erd_parallel, test_erd_scaling, test_erd_fixes)
exercise ERDQueue through _BranchWorker, which reaches claim_next,
claim_candidate, complete_candidate, update_branch_best, read_branch_best,
mark_branch_tainted, read_branch_meta, branch_done_candidates,
try_finalize_branch, delete_branch, and branches_in_progress.  What they do
NOT reach are the operator/supervisor methods only called from erd_search.py:
reset_active_branches, reset_stale_in_progress, cancel_active_branch,
status_by_branch_keys, active_branches_by_keys, get_active_branch,
claims_for_branch, heartbeats_with_branch, worker_counts_by_branch,
set_meta/get_meta, and total_branches.  claim_next/mark_done are also tested
here for completeness.
"""
import os
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue

WORDS = ["crane", "slate", "trace", "stale", "tales"]
N_CANDIDATES = 20


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(self.q.close)
        self.key = ScoreCache.encode_subset(WORDS)


class TestBranchLifecycle(_TmpQueue):
    def test_create_branch_returns_true_first_call(self):
        self.assertTrue(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES))

    def test_create_branch_returns_false_on_duplicate(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertFalse(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES))

    def test_get_branch_returns_row(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        row = self.q.get_branch(self.key)
        self.assertIsNotNone(row)
        self.assertEqual(row["n_words"], len(WORDS))
        self.assertEqual(row["budget"], 5)

    def test_get_branch_returns_none_after_delete(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.delete_branch(self.key)
        self.assertIsNone(self.q.get_branch(self.key))

    def test_get_active_branch_agrees_with_get_branch(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertEqual(
            self.q.get_active_branch(self.key)["n_words"],
            self.q.get_branch(self.key)["n_words"])

    def test_get_active_branch_returns_none_when_absent(self):
        self.assertIsNone(self.q.get_active_branch(self.key))

    def test_update_branch_best_sets_value(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.update_branch_best(self.key, "crane", 2.0, max_depth=3)
        guess, erd = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_update_branch_best_is_monotone(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.update_branch_best(self.key, "crane", 2.0)
        self.q.update_branch_best(self.key, "slate", 3.0)  # worse — must be rejected
        guess, erd = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_read_branch_best_returns_none_none_for_missing_key(self):
        self.assertEqual(self.q.read_branch_best(b"notakey"), (None, None))

    def test_mark_branch_tainted_sets_flag(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=3)
        self.q.mark_branch_tainted(self.key)
        meta = self.q.read_branch_meta(self.key)
        self.assertTrue(meta[3])  # tainted

    def test_mark_branch_tainted_is_monotone(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=3)
        self.q.mark_branch_tainted(self.key)
        self.q.mark_branch_tainted(self.key)  # no-op — can't un-taint
        self.assertTrue(self.q.read_branch_meta(self.key)[3])

    def test_read_branch_meta_full_tuple(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4)
        self.q.update_branch_best(self.key, "crane", 1.5, max_depth=3)
        meta = self.q.read_branch_meta(self.key)
        self.assertEqual(meta[0], "crane")
        self.assertAlmostEqual(meta[1], 1.5)
        self.assertEqual(meta[2], 3)    # best_max_depth
        self.assertFalse(meta[3])       # not tainted
        self.assertEqual(meta[4], 4)    # budget

    def test_try_finalize_branch_exactly_one_winner(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertTrue(self.q.try_finalize_branch(self.key))
        self.assertFalse(self.q.try_finalize_branch(self.key))

    def test_read_branch_meta_returns_none_for_missing_key(self):
        self.assertIsNone(self.q.read_branch_meta(b"notakey"))


class TestCandidateClaiming(_TmpQueue):
    def setUp(self):
        super().setUp()
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)

    def test_claim_candidate_returns_none_for_finalized_branch(self):
        self.q.try_finalize_branch(self.key)  # transitions status to 'finalized'
        idx = self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        self.assertIsNone(idx)

    def test_claim_candidate_returns_none_for_missing_branch(self):
        # Branch doesn't exist at all (never created).
        key2 = ScoreCache.encode_subset(WORDS[:2])
        idx = self.q.claim_candidate(key2, "worker-0", N_CANDIDATES)
        self.assertIsNone(idx)

    def test_claim_candidate_returns_each_index_exactly_once(self):
        claimed = set()
        for _ in range(N_CANDIDATES):
            idx = self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
            self.assertIsNotNone(idx)
            claimed.add(idx)
        self.assertEqual(len(claimed), N_CANDIDATES)
        # All candidates claimed — next call returns None.
        self.assertIsNone(self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES))

    def test_complete_candidate_marks_done(self):
        idx = self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        self.q.complete_candidate(self.key, idx)
        self.assertEqual(self.q.branch_done_candidates(self.key), 1)

    def test_claims_for_branch_returns_rows(self):
        self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        rows = self.q.claims_for_branch(self.key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["idx"], 0)

    def test_claims_for_branch_empty_after_delete(self):
        self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        self.q.delete_branch(self.key)
        self.assertEqual(self.q.claims_for_branch(self.key), [])


class TestStartupRecovery(_TmpQueue):
    def test_reset_stale_in_progress_restores_pending(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "salet", 0)])
        self.q.claim_next("worker-0")   # transitions row to in_progress
        n = self.q.reset_stale_in_progress()
        self.assertEqual(n, 1)
        row = self.q.get_pending_branch(self.key)
        self.assertEqual(row["status"], "pending")

    def test_reset_stale_in_progress_returns_zero_when_nothing_stale(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "salet", 0)])
        self.assertEqual(self.q.reset_stale_in_progress(), 0)

    def test_reset_active_branches_clears_d0_branches_and_claims(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        n_b, n_c = self.q.reset_active_branches()
        self.assertEqual(n_b, 1)
        self.assertGreaterEqual(n_c, 1)
        self.assertIsNone(self.q.get_branch(self.key))
        self.assertEqual(self.q.claims_for_branch(self.key), [])

    def test_reset_active_branches_preserves_cooperative_branch_progress(self):
        # D=0 branch (has pending_branches backup — wiped on reset).
        d0_key = self.key
        self.q.create_branch(d0_key, len(WORDS), N_CANDIDATES, depth=0)

        # D=1 cooperative branch (no pending_branches row — must survive reset).
        coop_words = WORDS[:3]
        coop_key = ScoreCache.encode_subset(coop_words)
        self.q.create_branch(coop_key, len(coop_words), N_CANDIDATES, depth=1)
        # Simulate two claims: idx 0 completed, idx 1 stale in-flight.
        idx0 = self.q.claim_candidate(coop_key, "worker-0", N_CANDIDATES)
        self.q.complete_candidate(coop_key, idx0)
        idx1 = self.q.claim_candidate(coop_key, "worker-0", N_CANDIDATES)
        self.assertIsNotNone(idx1)

        n_b, n_c = self.q.reset_active_branches()

        # D=0 branch is gone.
        self.assertEqual(n_b, 1)
        self.assertIsNone(self.q.get_branch(d0_key))

        # Cooperative branch row survives.
        self.assertIsNotNone(self.q.get_branch(coop_key))

        claims = self.q.claims_for_branch(coop_key)
        # Completed claim (done=1) is preserved so its progress isn't lost.
        done_claims = [c for c in claims if c["done"] == 1]
        self.assertEqual(len(done_claims), 1)
        self.assertEqual(done_claims[0]["idx"], idx0)

        # Stale in-flight claim (done=0) is freed so the candidate becomes a
        # re-claimable gap for the next worker.
        inflight_claims = [c for c in claims if c["done"] == 0]
        self.assertEqual(inflight_claims, [])

        # The freed gap is claimable again.
        reclaimed = self.q.claim_candidate(coop_key, "worker-1", N_CANDIDATES)
        self.assertEqual(reclaimed, idx1)


class TestCancelAndInspection(_TmpQueue):
    def test_cancel_active_branch_removes_branch_and_claims(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.claim_candidate(self.key, "worker-0", N_CANDIDATES)
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.cancel_active_branch(self.key, remove_from_queue=False)
        self.assertIsNone(self.q.get_branch(self.key))
        self.assertEqual(self.q.claims_for_branch(self.key), [])
        self.assertIsNotNone(self.q.get_pending_branch(self.key))  # pending survives

    def test_cancel_active_branch_with_remove_from_queue(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.cancel_active_branch(self.key, remove_from_queue=True)
        self.assertIsNone(self.q.get_pending_branch(self.key))

    def test_status_by_branch_keys_returns_only_requested_keys(self):
        k2 = ScoreCache.encode_subset(WORDS[:3])
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.add_pending_many([(k2, 3, 0, "crane", 0)])
        result = self.q.status_by_branch_keys([self.key])
        self.assertIn(self.key, result)
        self.assertNotIn(k2, result)

    def test_status_by_branch_keys_empty_input_returns_empty(self):
        self.assertEqual(self.q.status_by_branch_keys([]), {})

    def test_active_branches_by_keys_returns_only_requested_keys(self):
        k2 = ScoreCache.encode_subset(WORDS[:3])
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.create_branch(k2, 3, N_CANDIDATES)
        result = self.q.active_branches_by_keys([self.key])
        self.assertIn(self.key, result)
        self.assertNotIn(k2, result)

    def test_active_branches_by_keys_empty_input_returns_empty(self):
        self.assertEqual(self.q.active_branches_by_keys([]), {})

    def test_heartbeats_with_branch_joins_worker_and_branch(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, claims_done=0)
        rows = self.q.heartbeats_with_branch()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["worker_id"], "worker-0")

    def test_worker_counts_by_branch_counts_live_workers(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, claims_done=0)
        counts = self.q.worker_counts_by_branch(timeout_seconds=60)
        self.assertEqual(counts.get(self.key), 1)

    def test_worker_counts_by_branch_excludes_stale_heartbeats(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, claims_done=0)
        self.q._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = 0 WHERE worker_id = 'worker-0'")
        counts = self.q.worker_counts_by_branch(timeout_seconds=30)
        self.assertEqual(counts.get(self.key, 0), 0)

    def test_branches_in_progress_returns_open_branches(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertEqual(len(self.q.branches_in_progress()), 1)

    def test_branches_in_progress_excludes_finalized_branches(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.try_finalize_branch(self.key)
        self.assertEqual(len(self.q.branches_in_progress()), 0)

    def test_total_branches(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.assertEqual(self.q.total_branches(), 1)


class TestMetaKV(_TmpQueue):
    def test_set_and_get_meta_round_trip(self):
        self.q.set_meta("run_id", "abc123")
        self.assertEqual(self.q.get_meta("run_id"), "abc123")

    def test_get_meta_returns_none_for_missing_key(self):
        self.assertIsNone(self.q.get_meta("nonexistent"))

    def test_set_meta_overwrites_existing_value(self):
        self.q.set_meta("k", "v1")
        self.q.set_meta("k", "v2")
        self.assertEqual(self.q.get_meta("k"), "v2")


class TestClaimNext(_TmpQueue):
    def test_claim_next_returns_none_when_queue_empty(self):
        self.assertIsNone(self.q.claim_next("worker-0"))

    def test_claim_next_returns_branch_info(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        claimed = self.q.claim_next("worker-0")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["branch_key"], self.key)
        self.assertEqual(claimed["n_words"], len(WORDS))

    def test_claim_next_then_mark_done(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.claim_next("worker-0")
        self.q.mark_done(self.key)
        row = self.q.get_pending_branch(self.key)
        self.assertEqual(row["status"], "done")

    def test_claim_next_second_call_returns_none_after_only_one_pending(self):
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.claim_next("worker-0")       # claims the only pending branch
        self.assertIsNone(self.q.claim_next("worker-1"))  # nothing left


class TestCostModel(_TmpQueue):
    """Cost model: cold read, warm read, geometric mean, policy isolation,
    mark_claims_done, add_nodes_spent, add_claim_telemetry."""

    def test_cold_read_returns_none(self):
        self.assertIsNone(self.q.get_cost_typical("erd_all", 10))

    def test_single_sample_round_trips(self):
        import math
        self.q.update_cost_model("erd_all", 10, 1000)
        result = self.q.get_cost_typical("erd_all", 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1000.0, delta=1.0)

    def test_geometric_mean_not_arithmetic(self):
        import math
        # Two samples: 100 and 10000.  Geometric mean = 1000; arithmetic = 5050.
        self.q.update_cost_model("erd_all", 5, 100)
        self.q.update_cost_model("erd_all", 5, 10000)
        result = self.q.get_cost_typical("erd_all", 5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1000.0, delta=50.0)

    def test_weighted_batch_update(self):
        # weight=3 is equivalent to adding the sample 3 times.
        import math
        self.q.update_cost_model("erd_all", 8, 500, weight=3.0)
        result = self.q.get_cost_typical("erd_all", 8)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 500.0, delta=5.0)

    def test_policy_isolation(self):
        self.q.update_cost_model("erd_all", 12, 200)
        self.q.update_cost_model("max_group_size", 12, 9999)
        erd = self.q.get_cost_typical("erd_all", 12)
        mgs = self.q.get_cost_typical("max_group_size", 12)
        self.assertAlmostEqual(erd, 200.0, delta=5.0)
        self.assertAlmostEqual(mgs, 9999.0, delta=5.0)

    def test_size_bucket_isolation(self):
        self.q.update_cost_model("erd_all", 10, 100)
        self.q.update_cost_model("erd_all", 20, 999)
        r10 = self.q.get_cost_typical("erd_all", 10)
        r20 = self.q.get_cost_typical("erd_all", 20)
        self.assertAlmostEqual(r10, 100.0, delta=5.0)
        self.assertAlmostEqual(r20, 999.0, delta=5.0)

    def test_mark_claims_done_inserts_done_rows(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.q.mark_claims_done(self.key, [0, 2, 4])
        rows = self.q.claims_for_branch(self.key)
        done = {r['idx'] for r in rows if r['done'] == 1}
        self.assertEqual(done, {0, 2, 4})

    def test_add_nodes_spent_accumulates(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.q.add_nodes_spent(self.key, 100)
        self.q.add_nodes_spent(self.key, 50)
        row = self.q.get_branch(self.key)
        self.assertEqual(row['nodes_spent'], 150)

    def test_add_claim_telemetry_inserts_row(self):
        self.q.add_claim_telemetry(10, 5000, 300, 4)
        row = self.q._conn.execute(
            "SELECT * FROM claim_telemetry ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['n_words'], 10)
        self.assertEqual(row['coordination_millis'], 5000)
        self.assertEqual(row['work_nodes'], 300)
        self.assertEqual(row['worker_count'], 4)


if __name__ == "__main__":
    unittest.main()
