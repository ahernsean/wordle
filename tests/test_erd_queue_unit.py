"""Unit tests for ERDQueue: covers methods not exercised by the integration tests.

The integration tests (test_erd_parallel, test_erd_scaling, test_erd_fixes)
exercise ERDQueue through _BranchWorker, which reaches claim_next,
claim_next_bundle, complete_candidate, update_branch_best, read_branch_best,
mark_branch_tainted, read_branch_meta, branch_done_candidates,
try_finalize_branch, delete_branch, and branches_in_progress.  What they do
NOT reach are the operator/supervisor methods only called from erd_search.py:
recover_active_branches, reset_stale_in_progress, cancel_active_branch,
status_by_branch_keys, active_branches_by_keys, get_active_branch,
claims_for_branch, heartbeats_with_branch, worker_counts_by_branch,
set_meta/get_meta, and total_branches.  claim_next/mark_done are also tested
here for completeness.
"""
import os
import sqlite3
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue, cost_size_bucket

WORDS = ["crane", "slate", "trace", "stale", "tales"]
N_CANDIDATES = 20
# Fixed remaining-guess budget for TestCostModel: those tests exercise
# policy/size-bucket mechanics, which are independent of the budget value.
COST_MODEL_BUDGET = 5


# Identity best-first order with a uniform cost_lower_bound: nothing is ever
# ERD-pruned (cost_lower_bound 0.0 never meets any real bound), so
# claim_next_bundle(small_count=1, count_cap=1) hands out exactly one idx per
# call, in ascending order — the single-candidate-claim contract these tests
# were written against.
_IDENTITY_ORDER = list(range(N_CANDIDATES))
_ZERO_LOWER_BOUND = [0.0] * N_CANDIDATES


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.queue_path)
        self.addCleanup(self.q.close)
        self.key = ScoreCache.encode_subset(WORDS)

    def _claim_one_idx(self, branch_key, worker_id="worker-0",
                       expected_source_work_id=None,
                       expected_source_priority=None):
        """claim_next_bundle(small_count=1, count_cap=1) unwrapped to a bare
        idx (or None) — the old claim_candidate single-index contract, for
        tests exercising something other than the packer itself."""
        claim = self.q.claim_next_bundle(
            branch_key, worker_id, N_CANDIDATES, _IDENTITY_ORDER,
            _ZERO_LOWER_BOUND, small_count=1, count_cap=1,
            expected_source_work_id=expected_source_work_id,
            expected_source_priority=expected_source_priority)
        return claim[1][0] if claim is not None else None


class TestBranchLifecycle(_TmpQueue):
    def test_create_branch_returns_true_first_call(self):
        self.assertTrue(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES))

    def test_create_branch_returns_false_on_duplicate(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertFalse(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES))

    def test_create_branch_commits_active_row_and_ownership_together(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 42)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        statements = []
        self.q._conn.set_trace_callback(statements.append)
        try:
            self.assertTrue(self.q.create_branch(
                self.key, len(WORDS), N_CANDIDATES, budget=5,
                source_word="crane", source_pattern=42,
                source_work_id=source_work_id))
        finally:
            self.q._conn.set_trace_callback(None)
        transaction = "\n".join(statements)
        self.assertLess(transaction.index("BEGIN IMMEDIATE"),
                        transaction.index("INSERT OR IGNORE INTO active_branches"))
        self.assertLess(transaction.index("INSERT OR IGNORE INTO active_branches"),
                        transaction.index("INSERT INTO branch_source_work"))
        self.assertLess(transaction.index("INSERT INTO branch_source_work"),
                        transaction.index("COMMIT"))

    def test_losing_create_does_not_attach_incompatible_source_work(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 0)])
        source_a = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                             budget=4, source_work_id=source_a)
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "slate", 42)])
        source_b = max(row["source_work_id"] for row in self.q.source_work_rows())

        self.assertFalse(self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES,
            budget=5, source_work_id=source_b))
        self.assertEqual(self.q.branches_in_progress(source_b), [])

    def test_attach_source_work_rejects_replaced_incompatible_branch(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 0)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)

        self.assertFalse(self.q.attach_branch_source_work(
            self.key, source_work_id, budget=4, ceiling=None,
            n_words=len(WORDS), source_pattern=0))
        self.assertEqual(self.q.branches_in_progress(source_work_id), [])

    def test_attach_source_work_accepts_compatible_branch(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 0)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)

        self.assertTrue(self.q.attach_branch_source_work(
            self.key, source_work_id, budget=5, ceiling=None,
            n_words=len(WORDS), source_pattern=0))
        self.assertEqual(len(self.q.branches_in_progress(source_work_id)), 1)

    def test_attach_source_work_compares_compatible_lattice_ceilings(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 0)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                             budget=5, ceiling=2.0)

        self.assertTrue(self.q.attach_branch_source_work(
            self.key, source_work_id, budget=5, ceiling=1.8,
            n_words=len(WORDS), source_pattern=0))

    def test_attach_source_work_compares_off_lattice_ceilings(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 0)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                             budget=5, ceiling=2.01)

        self.assertTrue(self.q.attach_branch_source_work(
            self.key, source_work_id, budget=5, ceiling=1.99,
            n_words=len(WORDS), source_pattern=0))

    def test_direct_claim_rejects_newly_owned_branch(self):
        other_key = ScoreCache.encode_subset(WORDS[:4])
        self.q.add_pending_many([(other_key, 4, 1, "crane", 0)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.assertTrue(self.q.attach_branch_source_work(
            self.key, source_work_id, budget=5, ceiling=None,
            n_words=len(WORDS), source_pattern=0))

        self.assertIsNone(self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _IDENTITY_ORDER,
            _ZERO_LOWER_BOUND, small_count=1, count_cap=1,
            expected_source_work_id=None))

    def test_claim_rejects_resolved_expected_owner(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 7)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, budget=5,
            source_work_id=source_work_id, source_pattern=7)
        self.q.mark_done(self.key)

        self.assertIsNone(self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _IDENTITY_ORDER,
            _ZERO_LOWER_BOUND, small_count=1, count_cap=1,
            expected_source_work_id=source_work_id,
            expected_source_priority=1))

    def test_claim_rejects_stale_expected_source_priority(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 7)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, budget=5,
            source_work_id=source_work_id, source_pattern=7)
        self.q.set_source_work_priority(source_work_id, 9)

        self.assertIsNone(self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _IDENTITY_ORDER,
            _ZERO_LOWER_BOUND, small_count=1, count_cap=1,
            expected_source_work_id=source_work_id,
            expected_source_priority=1))
        self.assertEqual(self.q.claims_for_branch(self.key), [])

    def test_malformed_pending_schema_reports_schema_drift(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 0)])
        queue_path = self.queue_path
        self.q.close()
        conn = sqlite3.connect(queue_path)
        try:
            conn.executescript("""
                CREATE TABLE pending_branches_without_pattern (
                    branch_id INTEGER PRIMARY KEY,
                    n_words INTEGER NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    source_word TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    claimed_by TEXT,
                    claimed_at INTEGER,
                    completed_at INTEGER
                );
                INSERT INTO pending_branches_without_pattern
                    (branch_id, n_words, priority, source_word, status,
                     claimed_by, claimed_at, completed_at)
                    SELECT branch_id, n_words, priority, source_word, status,
                           claimed_by, claimed_at, completed_at
                    FROM pending_branches;
                DROP TABLE pending_branches;
                ALTER TABLE pending_branches_without_pattern
                    RENAME TO pending_branches;
            """)
            conn.commit()
        finally:
            conn.close()

        with self.assertRaisesRegex(RuntimeError,
                                    "pending_branches is missing column"):
            ERDQueue(queue_path)

    def test_source_work_migration_preserves_root_pattern_idempotently(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 42)])
        self.q._conn.execute("DELETE FROM branch_source_work")
        self.q._conn.execute("DELETE FROM source_work")
        self.q.close()

        migrated = ERDQueue(self.queue_path)
        try:
            claimed = migrated.claim_next("worker-0")
            self.assertEqual(claimed["source_pattern"], 42)
            self.assertEqual(claimed["source_word"], "crane")
        finally:
            migrated.close()

        reopened = ERDQueue(self.queue_path)
        try:
            self.assertEqual(len(reopened.source_work_rows()), 1)
            self.assertEqual(reopened.source_work_rows()[0]["root_count"], 1)
        finally:
            reopened.close()

    def test_source_work_migration_marks_terminal_roots_complete(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 42)])
        self.q.mark_done(self.key)
        self.q._conn.execute("DELETE FROM branch_source_work")
        self.q._conn.execute("DELETE FROM source_work")
        self.q.close()

        migrated = ERDQueue(self.queue_path)
        try:
            rows = migrated.source_work_rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["state"], "complete")
        finally:
            migrated.close()

        reopened = ERDQueue(self.queue_path)
        try:
            self.assertEqual(reopened.source_work_rows()[0]["state"], "complete")
        finally:
            reopened.close()

    def test_get_branch_returns_row(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        row = self.q.get_branch(self.key)
        self.assertIsNotNone(row)
        self.assertEqual(row["n_words"], len(WORDS))
        self.assertEqual(row["budget"], 5)

    def test_create_branch_accepts_consistent_spine_budget(self):
        # SALET ABORT DOGMA = guess_depth 3; budget 3; 3 + 3 == root_budget 6.
        spine = 'SALET -g-g- ABORT y---- DOGMA y---y'
        self.assertTrue(self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES,
            budget=3, spine=spine, root_budget=6))

    def test_create_branch_rejects_spine_budget_mismatch(self):
        # The duplicated-guess spine: guess_depth 4 with budget 3 violates
        # budget + guess_depth == root_budget.  Must raise, not persist.
        bad = 'SALET -g-g- ABORT y---- ABORT y---- DOGMA y---y'
        with self.assertRaises(ValueError):
            self.q.create_branch(
                self.key, len(WORDS), N_CANDIDATES,
                budget=3, spine=bad, root_budget=6)
        self.assertIsNone(self.q.get_branch(self.key))

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
        guess, erd, _ceiling = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_update_branch_best_is_monotone(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.q.update_branch_best(self.key, "crane", 2.0)
        self.q.update_branch_best(self.key, "slate", 3.0)  # worse — must be rejected
        guess, erd, _ceiling = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_read_branch_best_returns_none_none_for_missing_key(self):
        self.assertEqual(self.q.read_branch_best(b"notakey"), (None, None, None))

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

    def test_claim_returns_none_for_finalized_branch(self):
        self.q.try_finalize_branch(self.key)  # transitions status to 'finalized'
        idx = self._claim_one_idx(self.key, "worker-0")
        self.assertIsNone(idx)

    def test_claim_returns_none_for_missing_branch(self):
        # Branch doesn't exist at all (never created).
        key2 = ScoreCache.encode_subset(WORDS[:2])
        idx = self._claim_one_idx(key2, "worker-0")
        self.assertIsNone(idx)

    def test_claim_returns_each_index_exactly_once(self):
        claimed = set()
        for _ in range(N_CANDIDATES):
            idx = self._claim_one_idx(self.key, "worker-0")
            self.assertIsNotNone(idx)
            claimed.add(idx)
        self.assertEqual(len(claimed), N_CANDIDATES)
        # All candidates claimed — next call returns None.
        self.assertIsNone(self._claim_one_idx(self.key, "worker-0"))

    def test_forward_path_does_not_rewind_for_holes_before_exhaustion(self):
        # Claim the first three slots, then free the middle one the way
        # reclaim_stale_claims does: the row is deleted, reopening a gap.
        for _ in range(3):
            self._claim_one_idx(self.key, "worker-0")
        self.q._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_id = ? AND idx = 1",
            (self.q._intern_branch(self.key),))
        # pack_cursor is at 3 (< N_CANDIDATES): the forward path claims the
        # next fresh slot (3), not the hole at idx 1 — holes are picked up
        # only once the cursor exhausts the full best-first order
        # (adaptive_claim_packing.md §5), the O(1)-amortized common path.
        idx = self._claim_one_idx(self.key, "worker-1")
        self.assertEqual(idx, 3)

    def test_holes_pass_refills_reclaimed_hole_after_cursor_exhausted(self):
        claimed = [self._claim_one_idx(self.key, "worker-0")
                  for _ in range(N_CANDIDATES)]
        hole = claimed[5]
        self.q._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_id = ? AND idx = ?",
            (self.q._intern_branch(self.key), hole))
        # pack_cursor == N_CANDIDATES now, so this call falls back to
        # holes_pass and picks the freed slot back up.
        idx = self._claim_one_idx(self.key, "worker-1")
        self.assertEqual(idx, hole)

    def test_complete_candidate_marks_done(self):
        idx = self._claim_one_idx(self.key, "worker-0")
        self.q.complete_candidate(self.key, idx)
        self.assertEqual(self.q.branch_done_candidates(self.key), 1)

    def test_claims_for_branch_returns_rows(self):
        self._claim_one_idx(self.key, "worker-0")
        rows = self.q.claims_for_branch(self.key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["idx"], 0)

    def test_claims_for_branch_empty_after_delete(self):
        self._claim_one_idx(self.key, "worker-0")
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

    def test_recover_active_branches_frees_in_flight_claims_only(self):
        # A user-queued branch is identified by its pending_branches row.
        self.q.add_pending_many([(self.key, len(WORDS), 0, None, None)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self._claim_one_idx(
            self.key, "worker-0", source_work_id, 0)
        n_b, n_c = self.q.recover_active_branches()
        self.assertEqual(n_b, 1)
        self.assertEqual(n_c, 1)
        # The branch row survives so the re-promoting worker adopts it.
        self.assertIsNotNone(self.q.get_branch(self.key))
        # The in-flight (done=0) claim is freed.
        self.assertEqual(self.q.claims_for_branch(self.key), [])

    def test_recover_active_branches_reopens_dead_finalize(self):
        # A finalizer killed between try_finalize_branch and delete_branch
        # leaves the row 'finalized' with no live owner; recovery reopens it.
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertTrue(self.q.try_finalize_branch(self.key))
        self.q.recover_active_branches()
        row = self.q.get_branch(self.key)
        self.assertEqual(row["status"], "open")
        self.assertIsNone(row["finalized_at"])
        self.assertTrue(self.q.try_finalize_branch(self.key))

    def test_reclaim_stale_finalize(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self.assertTrue(self.q.try_finalize_branch(self.key))
        # A fresh finalize belongs to a live rival: not reclaimable.
        self.assertFalse(self.q.reclaim_stale_finalize(self.key, 60))
        # Past the takeover window it is reclaimable, exactly once.
        self.q._conn.execute(
            "UPDATE active_branches SET finalized_at = finalized_at - 120 "
            "WHERE branch_id = ?", (self.q._intern_branch(self.key),))
        self.assertTrue(self.q.reclaim_stale_finalize(self.key, 60))
        self.assertEqual(self.q.get_branch(self.key)["status"], "open")
        self.assertFalse(self.q.reclaim_stale_finalize(self.key, 60))

    def test_recover_active_branches_preserves_completed_claims(self):
        # User-queued branch: has a pending_branches row.
        d0_key = self.key
        self.q.add_pending_many([(d0_key, len(WORDS), 0, None, None)])
        source_work_id = self.q.source_work_rows()[0]["source_work_id"]
        self.q.create_branch(d0_key, len(WORDS), N_CANDIDATES)
        idx_done = self._claim_one_idx(
            d0_key, "worker-0", source_work_id, 0)
        self.q.complete_candidate(d0_key, idx_done)

        # Cooperative branch: no pending_branches row.
        coop_words = WORDS[:3]
        coop_key = ScoreCache.encode_subset(coop_words)
        self.q.create_branch(coop_key, len(coop_words), N_CANDIDATES)
        # Simulate two claims: idx 0 completed, idx 1 stale in-flight.
        idx0 = self._claim_one_idx(coop_key, "worker-0")
        self.q.complete_candidate(coop_key, idx0)
        idx1 = self._claim_one_idx(coop_key, "worker-0")
        self.assertIsNotNone(idx1)

        n_b, n_c = self.q.recover_active_branches()
        self.assertEqual(n_b, 2)
        self.assertEqual(n_c, 1)

        # Both branch rows survive, and each keeps its done=1 claims.
        for key, done_idx in ((d0_key, idx_done), (coop_key, idx0)):
            self.assertIsNotNone(self.q.get_branch(key))
            claims = self.q.claims_for_branch(key)
            done_claims = [c for c in claims if c["done"] == 1]
            self.assertEqual([c["idx"] for c in done_claims], [done_idx])
            # Stale in-flight claims (done=0) are freed so those candidates
            # become re-claimable gaps for the next worker.
            self.assertEqual([c for c in claims if c["done"] == 0], [])

        # The freed gap is not lost, but it isn't immediately reclaimable
        # either: pack_cursor is past it (2), and holes below the cursor are
        # only picked up by holes_pass once the cursor exhausts the full
        # best-first order (adaptive_claim_packing.md §5) — so the next claim
        # takes the next fresh slot instead.
        next_idx = self._claim_one_idx(coop_key, "worker-1")
        self.assertNotEqual(next_idx, idx1)
        self.assertEqual(next_idx, 2)


class TestCancelAndInspection(_TmpQueue):
    def test_cancel_active_branch_removes_branch_and_claims(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES)
        self._claim_one_idx(self.key, "worker-0")
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


class TestCheckpoint(_TmpQueue):
    def _wal_size(self):
        wal_path = os.path.join(self._tmp.name, "q.sqlite3-wal")
        return os.path.getsize(wal_path) if os.path.exists(wal_path) else 0

    def test_checkpoint_truncates_wal_after_writes(self):
        for i in range(50):
            self.q.add_pending_many(
                [(f"{i:05d}".encode(), 1, 0, None, 0)])
        self.assertGreater(self._wal_size(), 0)
        self.q.checkpoint()
        self.assertEqual(self._wal_size(), 0)

    def test_checkpoint_does_not_raise_on_empty_queue(self):
        self.q.checkpoint()  # no rows written yet — must be a no-op, not an error


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

    def test_equal_priority_source_work_keeps_started_source_admitted(self):
        later_key = ScoreCache.encode_subset(WORDS[:3])
        earlier_second_key = ScoreCache.encode_subset(WORDS[1:])
        self.q.add_pending_many([
            (self.key, len(WORDS), 5, "crane", 0),
            (earlier_second_key, 4, 5, "crane", 1),
        ])
        self.q.add_pending_many([(later_key, 3, 5, "slate", 0)])

        first = self.q.claim_next("worker-0")
        self.assertEqual(first["source_word"], "crane")
        second = self.q.claim_next("worker-1")
        self.assertEqual(second["source_word"], "crane")

    def test_source_priority_updates_active_and_pending_membership(self):
        self.assertFalse(self.q.has_source_work())
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 0)])
        self.assertTrue(self.q.has_source_work())
        claimed = self.q.claim_next("worker-0")
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                             priority=claimed["priority"],
                             source_work_id=claimed["source_work_id"])
        self.assertTrue(self.q.set_source_work_priority(
            claimed["source_work_id"], 9))
        self.assertEqual(self.q.get_branch(self.key)["priority"], 9)
        self.assertEqual(self.q.source_work_candidates()[0]["requested_priority"], 9)
        rows = self.q.source_work_rows()
        self.assertEqual(rows[0]["root_count"], 1)
        self.assertEqual(rows[0]["branch_count"], 1)

    def test_shared_root_claim_uses_selected_owner_pattern(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 7)])
        self.q.add_pending_many([(self.key, len(WORDS), 9, "slate", 42)])
        slate = next(row for row in self.q.source_work_rows()
                     if row["source_word"] == "slate")
        claimed = self.q.claim_next("worker-0", slate["source_work_id"])
        self.assertEqual(claimed["source_word"], "slate")
        self.assertEqual(claimed["source_pattern"], 42)

    def test_active_shared_root_uses_selected_owner_metadata(self):
        self.q.add_pending_many([(self.key, len(WORDS), 1, "crane", 7)])
        self.q.add_pending_many([(self.key, len(WORDS), 9, "slate", 42)])
        rows = {row["source_word"]: row for row in self.q.source_work_rows()}
        crane = self.q.claim_next("worker-0", rows["crane"]["source_work_id"])
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, priority=crane["priority"],
            source_word=crane["source_word"],
            source_pattern=crane["source_pattern"],
            source_work_id=crane["source_work_id"])

        active = self.q.branches_in_progress(rows["slate"]["source_work_id"])[0]
        self.assertEqual(active["owner_source_word"], "slate")
        self.assertEqual(active["owner_root_pattern"], 42)
        self.assertEqual(active["owner_priority"], 9)

    def test_completed_owner_stays_resolved_when_branch_is_reactivated(self):
        self.q.add_pending_many([(self.key, len(WORDS), 9, "crane", 7)])
        crane = self.q.claim_next("worker-0")
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, budget=5,
            priority=crane["priority"], source_word=crane["source_word"],
            source_pattern=crane["source_pattern"],
            source_work_id=crane["source_work_id"])
        self.q.mark_done(self.key)
        self.q.delete_branch(self.key)

        self.q.add_pending_many([(self.key, len(WORDS), 1, "slate", 42)])
        slate = next(row for row in self.q.source_work_rows()
                     if row["source_word"] == "slate")
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, budget=4,
            priority=1, source_word="slate", source_pattern=42,
            source_work_id=slate["source_work_id"])

        candidates = self.q.source_work_candidates()
        self.assertEqual([row["source_word"] for row in candidates], ["slate"])
        provenance = self.q._conn.execute("""
            SELECT source.source_word, membership.resolved_at
            FROM branch_source_work AS membership
            JOIN source_work AS source USING (source_work_id)
            WHERE membership.branch_id = ?
            ORDER BY source.source_work_id
        """, (self.q._intern_branch(self.key),)).fetchall()
        self.assertEqual(provenance[0]["source_word"], "crane")
        self.assertIsNotNone(provenance[0]["resolved_at"])
        self.assertEqual(provenance[1]["source_word"], "slate")
        self.assertIsNone(provenance[1]["resolved_at"])

    def test_cut_branch_keeps_live_membership_for_pending_exact_work(self):
        self.q.add_pending_many([(self.key, len(WORDS), 9, "crane", 7)])
        crane = self.q.claim_next("worker-0")
        self.q.create_branch(
            self.key, len(WORDS), N_CANDIDATES, budget=5,
            priority=crane["priority"], source_word=crane["source_word"],
            source_pattern=crane["source_pattern"],
            source_work_id=crane["source_work_id"])
        self.q.requeue_pending(self.key)
        self.q.delete_branch(self.key)

        candidates = self.q.source_work_candidates()
        self.assertEqual([row["source_word"] for row in candidates], ["crane"])
        membership = self.q._conn.execute("""
            SELECT resolved_at FROM branch_source_work
            WHERE branch_id = ? AND source_work_id = ?
        """, (self.q._intern_branch(self.key),
              crane["source_work_id"])).fetchone()
        self.assertIsNone(membership["resolved_at"])

    def test_clear_removes_source_work_before_readding_shared_root(self):
        self.q.add_pending_many([(self.key, len(WORDS), 999, "audio", 0)])
        self.q.clear()
        self.q.add_pending_many([(self.key, len(WORDS), 1, "penis", 0)])
        claimed = self.q.claim_next("worker-0")
        self.assertEqual(claimed["source_word"], "penis")
        self.assertEqual(claimed["priority"], 1)

    def test_remove_pending_removes_source_work_membership(self):
        self.q.add_pending_many([(self.key, len(WORDS), 9, "crane", 0)])
        branch_id = self.q._intern_branch(self.key)

        self.assertTrue(self.q.remove_pending(self.key))
        self.assertEqual(self.q._conn.execute(
            "SELECT COUNT(*) FROM branch_source_work WHERE branch_id = ?",
            (branch_id,)).fetchone()[0], 0)

        self.q.add_pending_many([(self.key, len(WORDS), 1, "slate", 42)])
        claimed = self.q.claim_next("worker-0")
        self.assertEqual(claimed["source_word"], "slate")
        self.assertEqual(claimed["priority"], 1)

    def test_forced_active_removal_removes_source_work_membership(self):
        self.q.add_pending_many([(self.key, len(WORDS), 9, "crane", 0)])
        claimed = self.q.claim_next("worker-0")
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                             priority=claimed["priority"],
                             source_work_id=claimed["source_work_id"])
        branch_id = self.q._intern_branch(self.key)

        self.q.cancel_active_branch(self.key, remove_from_queue=True)
        self.assertEqual(self.q._conn.execute(
            "SELECT COUNT(*) FROM branch_source_work WHERE branch_id = ?",
            (branch_id,)).fetchone()[0], 0)

        self.q.add_pending_many([(self.key, len(WORDS), 1, "slate", 42)])
        claimed = self.q.claim_next("worker-1")
        self.assertEqual(claimed["source_word"], "slate")
        self.assertEqual(claimed["priority"], 1)


class TestCostModel(_TmpQueue):
    """Cost model: cold read, warm read, geometric mean, policy isolation,
    mark_claims_done, add_nodes_spent, add_claim_telemetry."""

    def test_cold_read_returns_none(self):
        self.assertIsNone(self.q.get_cost_typical("erd_all", 10, budget=COST_MODEL_BUDGET))

    def test_single_sample_round_trips(self):
        import math
        self.q.update_cost_model("erd_all", 10, 1000, budget=COST_MODEL_BUDGET)
        result = self.q.get_cost_typical("erd_all", 10, budget=COST_MODEL_BUDGET)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1000.0, delta=1.0)

    def test_geometric_mean_not_arithmetic(self):
        import math
        # Two samples: 100 and 10000.  Geometric mean = 1000; arithmetic = 5050.
        self.q.update_cost_model("erd_all", 5, 100, budget=COST_MODEL_BUDGET)
        self.q.update_cost_model("erd_all", 5, 10000, budget=COST_MODEL_BUDGET)
        result = self.q.get_cost_typical("erd_all", 5, budget=COST_MODEL_BUDGET)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1000.0, delta=50.0)

    def test_weighted_batch_update(self):
        # weight=3 is equivalent to adding the sample 3 times.
        import math
        self.q.update_cost_model("erd_all", 8, 500, weight=3.0, budget=COST_MODEL_BUDGET)
        result = self.q.get_cost_typical("erd_all", 8, budget=COST_MODEL_BUDGET)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 500.0, delta=5.0)

    def test_policy_isolation(self):
        self.q.update_cost_model("erd_all", 12, 200, budget=COST_MODEL_BUDGET)
        self.q.update_cost_model("max_group_size", 12, 9999, budget=COST_MODEL_BUDGET)
        erd = self.q.get_cost_typical("erd_all", 12, budget=COST_MODEL_BUDGET)
        mgs = self.q.get_cost_typical("max_group_size", 12, budget=COST_MODEL_BUDGET)
        self.assertAlmostEqual(erd, 200.0, delta=5.0)
        self.assertAlmostEqual(mgs, 9999.0, delta=5.0)

    def test_size_bucket_isolation(self):
        self.q.update_cost_model("erd_all", 10, 100, budget=COST_MODEL_BUDGET)
        self.q.update_cost_model("erd_all", 20, 999, budget=COST_MODEL_BUDGET)
        r10 = self.q.get_cost_typical("erd_all", 10, budget=COST_MODEL_BUDGET)
        r20 = self.q.get_cost_typical("erd_all", 20, budget=COST_MODEL_BUDGET)
        self.assertAlmostEqual(r10, 100.0, delta=5.0)
        self.assertAlmostEqual(r20, 999.0, delta=5.0)

    def test_mark_claims_done_inserts_done_rows(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.q.mark_claims_done(self.key, [0, 2, 4])
        rows = self.q.claims_for_branch(self.key)
        done = {r['idx'] for r in rows if r['done'] == 1}
        self.assertEqual(done, {0, 2, 4})

    def test_mark_claims_done_skips_finalized_branch(self):
        # A publisher call that races finalization must not re-create claim
        # rows: delete_branch has removed the active row, so a fresh insert
        # would orphan the rows past the branch they belong to.
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.q.delete_branch(self.key)
        self.q.mark_claims_done(self.key, [0, 2, 4])
        self.assertEqual(self.q.claims_for_branch(self.key), [])

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

    def test_add_backstop_telemetry_inserts_row(self):
        self.q.add_backstop_telemetry(8, 2, 65000, 500, None, 6)
        row = self.q._conn.execute(
            "SELECT * FROM backstop_telemetry ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['n_words'], 8)
        self.assertEqual(row['budget'], 2)
        self.assertEqual(row['elapsed_millis'], 65000)
        self.assertEqual(row['nodes'], 500)
        self.assertIsNone(row['predicted_nodes'])
        self.assertEqual(row['remaining_candidates'], 6)

    def test_add_backstop_telemetry_with_warm_prediction(self):
        self.q.add_backstop_telemetry(12, 1, 120000, 1000, 800.5, 4)
        row = self.q._conn.execute(
            "SELECT * FROM backstop_telemetry ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertAlmostEqual(row['predicted_nodes'], 800.5, places=1)

    def test_add_nodes_spent_noop_for_nonpositive_delta(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=5)
        self.q.add_nodes_spent(self.key, 0)
        self.q.add_nodes_spent(self.key, -10)
        row = self.q.get_branch(self.key)
        self.assertEqual(row['nodes_spent'], 0)

    def test_update_cost_model_noop_for_nonpositive_nodes(self):
        from wordle_engine import ERD_ALL
        self.q.update_cost_model(ERD_ALL, 10, 0, budget=COST_MODEL_BUDGET)
        self.q.update_cost_model(ERD_ALL, 10, -5, budget=COST_MODEL_BUDGET)
        self.assertIsNone(self.q.get_cost_typical(ERD_ALL, 10, budget=COST_MODEL_BUDGET))

    def test_update_cost_model_logsums_noop_for_nonpositive_weight(self):
        from wordle_engine import ERD_ALL
        self.q.update_cost_model_logsums(ERD_ALL, 10, 3.0, 9.0, 0.0, budget=COST_MODEL_BUDGET)
        self.assertIsNone(self.q.get_cost_typical(ERD_ALL, 10, budget=COST_MODEL_BUDGET))

    def test_get_cost_spread_returns_none_for_cold_bucket(self):
        from wordle_engine import ERD_ALL
        result = self.q.get_cost_spread(ERD_ALL, 10, budget=COST_MODEL_BUDGET)
        self.assertIsNone(result)

    def test_cost_size_bucket_returns_zero_for_n_below_one(self):
        # n_words < 1 hits the early-return guard (line 33); log(0) is undefined
        # so this guard is the only valid path for zero or negative inputs.
        self.assertEqual(cost_size_bucket(0), 0)
        self.assertEqual(cost_size_bucket(-5), 0)


class TestBranchCeiling(_TmpQueue):
    """The active_branches.ceiling column: stored at create, immutable, read
    back by read_branch_best / read_branch_meta, and flagged by
    mark_branch_cut."""

    def test_create_branch_stores_ceiling(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4,
                             ceiling=2.5)
        _, _, ceiling = self.q.read_branch_best(self.key)
        self.assertAlmostEqual(ceiling, 2.5)

    def test_create_branch_default_ceiling_is_null(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4)
        _, _, ceiling = self.q.read_branch_best(self.key)
        self.assertIsNone(ceiling)

    def test_racing_create_does_not_overwrite_ceiling(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4,
                             ceiling=2.5)
        created = self.q.create_branch(self.key, len(WORDS), N_CANDIDATES,
                                       budget=4, ceiling=9.0)
        self.assertFalse(created)
        _, _, ceiling = self.q.read_branch_best(self.key)
        self.assertAlmostEqual(ceiling, 2.5)

    def test_mark_branch_cut_sets_flag_in_meta(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4,
                             ceiling=2.5)
        meta = self.q.read_branch_meta(self.key)
        self.assertAlmostEqual(meta[5], 2.5)   # ceiling
        self.assertFalse(meta[6])              # cut_occurred
        self.q.mark_branch_cut(self.key)
        self.q.mark_branch_cut(self.key)       # monotone — still set
        self.assertTrue(self.q.read_branch_meta(self.key)[6])


class TestCutResults(_TmpQueue):
    """cut_results: the durable-proof channel for ceilinged solves that ended
    as a lower bound instead of an exact optimum.  Keyed by (branch_id,
    budget, tainted); each class keeps its own maximum bound (issue #185)."""

    def test_round_trip(self):
        self.q.add_cut_result(self.key, budget=4, bound=2.5)
        self.assertEqual(self.q.read_cut_result(self.key), [(2.5, 4, False)])

    def test_missing_key_reads_empty(self):
        self.assertEqual(self.q.read_cut_result(b"notakey"), [])

    def test_higher_bound_at_same_budget_and_taint_wins(self):
        self.q.add_cut_result(self.key, budget=4, bound=2.5)
        self.q.add_cut_result(self.key, budget=4, bound=3.0)
        self.assertEqual(self.q.read_cut_result(self.key), [(3.0, 4, False)])

    def test_lower_bound_at_same_budget_and_taint_does_not_regress(self):
        self.q.add_cut_result(self.key, budget=4, bound=3.0)
        self.q.add_cut_result(self.key, budget=4, bound=2.5)
        self.assertEqual(self.q.read_cut_result(self.key), [(3.0, 4, False)])

    def test_different_budgets_kept_as_separate_rows(self):
        # A higher bound proven at a lower budget must not evict a lower
        # bound proven at a higher budget: only the budget-5 row can serve a
        # budget-5 consumer.
        self.q.add_cut_result(self.key, budget=2, bound=5.0)
        self.q.add_cut_result(self.key, budget=5, bound=1.0)
        self.assertEqual(
            sorted(self.q.read_cut_result(self.key)),
            sorted([(5.0, 2, False), (1.0, 5, False)]))

    def test_different_taint_kept_as_separate_rows(self):
        # A higher tainted bound must not evict a lower untainted one at the
        # same budget: the untainted row serves a consumer for free, while
        # the tainted row forces the consumer to inherit the taint.
        self.q.add_cut_result(self.key, budget=4, bound=2.0, tainted=False)
        self.q.add_cut_result(self.key, budget=4, bound=5.0, tainted=True)
        self.assertEqual(
            sorted(self.q.read_cut_result(self.key)),
            sorted([(2.0, 4, False), (5.0, 4, True)]))

    def test_tainted_round_trip(self):
        self.q.add_cut_result(self.key, budget=4, bound=2.5, tainted=True)
        self.assertEqual(self.q.read_cut_result(self.key), [(2.5, 4, True)])

    def test_recover_active_branches_preserves_cut_results(self):
        # A cut bound is a proof, not in-flight state: a restart kills
        # waiters, but the proof stays true.
        self.q.add_cut_result(self.key, budget=4, bound=2.5)
        self.q.recover_active_branches()
        self.assertEqual(self.q.read_cut_result(self.key), [(2.5, 4, False)])


class TestPendingRowHelpers(_TmpQueue):
    """has_pending_row / requeue_pending: the user-queued (exact-consumer)
    guards around ceilinged promotion."""

    def _queue_pending(self):
        self.q.add_pending_many(
            [(self.key, len(WORDS), 1, "crane", 0)])

    def test_has_pending_row(self):
        self.assertFalse(self.q.has_pending_row(self.key))
        self._queue_pending()
        self.assertTrue(self.q.has_pending_row(self.key))

    def test_requeue_pending_resets_in_progress(self):
        self._queue_pending()
        claimed = self.q.claim_next("worker-0")
        self.assertEqual(bytes(claimed["branch_key"]), self.key)
        self.assertIsNone(self.q.claim_next("worker-1"))  # in_progress: unclaimable
        self.assertTrue(self.q.requeue_pending(self.key))
        reclaimed = self.q.claim_next("worker-1")
        self.assertIsNotNone(reclaimed)
        self.assertEqual(bytes(reclaimed["branch_key"]), self.key)

    def test_requeue_pending_leaves_done_rows(self):
        self._queue_pending()
        self.q.claim_next("worker-0")
        self.q.mark_done(self.key)
        self.assertFalse(self.q.requeue_pending(self.key))
        self.assertIsNone(self.q.claim_next("worker-1"))

    def test_requeue_pending_noop_without_row(self):
        self.assertFalse(self.q.requeue_pending(self.key))

    def test_complete_pending_for_loss_preserves_an_uncovered_request(self):
        self._queue_pending()
        self.q.claim_next("worker-0")
        self.assertFalse(self.q.complete_pending_for_loss(
            self.key, loss_budget=3, root_budget=6))
        self.assertEqual(self.q.get_pending_branch(self.key)["status"], "pending")

    def test_complete_pending_for_loss_retires_a_covered_request(self):
        self._queue_pending()
        self.q.claim_next("worker-0")
        self.assertTrue(self.q.complete_pending_for_loss(
            self.key, loss_budget=5, root_budget=6))
        self.assertEqual(self.q.get_pending_branch(self.key)["status"], "done")


class TestCeilingTelemetry(_TmpQueue):
    """The ceiling/outcome columns on branch_finalize_log, the
    cut_reuse_misses table, and the backstop epoch stamp."""

    def test_finalize_log_records_ceiling_and_outcome(self):
        self.q.add_branch_finalize_log(
            self.key, "SALET -g-g-", len(WORDS), 4, 100, 200, 5000, 3,
            ceiling=2.5, outcome="cut", best_guess="crane", best_erd=1.75)
        row = self.q._conn.execute(
            "SELECT ceiling, outcome, best_guess, best_erd "
            "FROM telemetry.branch_finalize_log"
        ).fetchone()
        self.assertAlmostEqual(row["ceiling"], 2.5)
        self.assertEqual(row["outcome"], "cut")
        self.assertEqual(row["best_guess"], "crane")
        self.assertAlmostEqual(row["best_erd"], 1.75)

    def test_cut_reuse_miss_row(self):
        self.q.set_epoch(7)
        self.q.add_cut_reuse_miss(self.key, len(WORDS), 4,
                                  wanted_ceiling=None,
                                  available_bound=2.5, available_budget=5)
        row = self.q._conn.execute(
            "SELECT n_words, budget, wanted_ceiling, available_bound, "
            "available_budget, epoch FROM telemetry.cut_reuse_misses").fetchone()
        self.assertEqual(row["n_words"], len(WORDS))
        self.assertEqual(row["budget"], 4)
        self.assertIsNone(row["wanted_ceiling"])
        self.assertAlmostEqual(row["available_bound"], 2.5)
        self.assertEqual(row["available_budget"], 5)
        self.assertEqual(row["epoch"], 7)

    def test_backstop_telemetry_stamps_epoch(self):
        self.q.set_epoch(7)
        self.q.add_backstop_telemetry(30, 4, 61000, 12345, None, 9)
        row = self.q._conn.execute(
            "SELECT epoch FROM telemetry.backstop_telemetry").fetchone()
        self.assertEqual(row["epoch"], 7)


class TestPackerUsesCeilingBound(_TmpQueue):
    """claim_next_bundle classifies candidates against min(best_erd, ceiling):
    on a fresh ceilinged branch with no achieved best, candidates whose lower
    bound reaches the ceiling are provably-pruned freebies, so the first
    bundle is a bulk sweep instead of a small_count-limited survivor bundle."""

    def test_ceiling_bulk_completes_provably_pruned_candidates(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4,
                             ceiling=1.5)
        lower_bounds = [2.0] * N_CANDIDATES
        claim = self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _IDENTITY_ORDER, lower_bounds,
            small_count=2, count_cap=N_CANDIDATES)
        self.assertIsNone(claim)
        self.assertEqual(self.q.branch_done_candidates(self.key), N_CANDIDATES)
        branch = self.q.get_branch(self.key)
        self.assertEqual(branch["cut_occurred"], 1)
        self.assertEqual(branch["pack_cursor"], N_CANDIDATES)

    def test_no_ceiling_same_bounds_pack_small_bundle(self):
        # Contrast case: identical lower bounds but no ceiling — nothing is
        # provably pruned against an infinite bound, so the packer stops at
        # small_count survivors.
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, budget=4)
        lower_bounds = [2.0] * N_CANDIDATES
        claim = self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _IDENTITY_ORDER, lower_bounds,
            small_count=2, count_cap=N_CANDIDATES)
        self.assertIsNotNone(claim)
        _, indices, _ = claim
        self.assertEqual(len(indices), 2)


if __name__ == "__main__":
    unittest.main()
