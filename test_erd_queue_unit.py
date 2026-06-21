"""Unit tests for ERDQueue: covers methods not exercised by the integration tests.

The integration tests (test_erd_parallel, test_erd_scaling, test_erd_fixes)
exercise ERDQueue through _BranchWorker, which reaches claim_next, claim_chunk,
complete_chunk, update_branch_best, read_branch_best, mark_branch_tainted,
read_branch_meta, branch_done_chunks, try_finalize_branch, delete_branch, and
branches_in_progress.  What they do NOT reach are the operator/supervisor
methods only called from erd_search.py: reset_active_branches,
reset_stale_in_progress, cancel_active_branch, status_by_branch_keys,
active_branches_by_keys, get_active_branch, chunks_for_branch,
heartbeats_with_branch, worker_counts_by_branch, set_meta/get_meta, and
total_branches.  The pure-math static methods (chunk_range, n_chunks_for,
chunk_size_for) and claim_next/mark_done are also tested here for completeness.
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


class TestChunkMath(unittest.TestCase):
    def test_chunk_range_first_chunk(self):
        lo, hi = ERDQueue.chunk_range(0, 5, 20)
        self.assertEqual((lo, hi), (0, 5))

    def test_chunk_range_last_chunk_clips_at_n_candidates(self):
        lo, hi = ERDQueue.chunk_range(3, 5, 18)
        self.assertEqual((lo, hi), (15, 18))

    def test_chunk_range_single_chunk(self):
        lo, hi = ERDQueue.chunk_range(0, 100, 15)
        self.assertEqual((lo, hi), (0, 15))

    def test_n_chunks_for_exact_division(self):
        self.assertEqual(ERDQueue.n_chunks_for(20, 5), 4)

    def test_n_chunks_for_with_remainder(self):
        self.assertEqual(ERDQueue.n_chunks_for(21, 5), 5)

    def test_chunk_size_for_one_word_branch(self):
        # 1 word → 1 chunk → chunk_size = n_candidates
        self.assertEqual(ERDQueue.chunk_size_for(1, 100, min_words_per_chunk=3), 100)

    def test_chunk_size_for_max_chunk_count_cap(self):
        # 1000 words / 3 min → 334 chunks, but cap=10 → chunk_size = ceil(100/10) = 10
        self.assertEqual(
            ERDQueue.chunk_size_for(1000, 100, min_words_per_chunk=3, max_chunk_count=10),
            10)

    def test_chunk_size_for_min_words_larger_than_n_words(self):
        # 5 words, min_words=100 → ceil(5/100)=1 chunk → chunk_size = n_candidates
        self.assertEqual(ERDQueue.chunk_size_for(5, 50, min_words_per_chunk=100), 50)


class TestBranchLifecycle(_TmpQueue):
    def test_create_branch_returns_true_first_call(self):
        self.assertTrue(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5))

    def test_create_branch_returns_false_on_duplicate(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.assertFalse(self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5))

    def test_get_branch_returns_row(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5, budget=5)
        row = self.q.get_branch(self.key)
        self.assertIsNotNone(row)
        self.assertEqual(row["n_words"], len(WORDS))
        self.assertEqual(row["budget"], 5)

    def test_get_branch_returns_none_after_delete(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.delete_branch(self.key)
        self.assertIsNone(self.q.get_branch(self.key))

    def test_get_active_branch_agrees_with_get_branch(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.assertEqual(
            self.q.get_active_branch(self.key)["n_words"],
            self.q.get_branch(self.key)["n_words"])

    def test_get_active_branch_returns_none_when_absent(self):
        self.assertIsNone(self.q.get_active_branch(self.key))

    def test_update_branch_best_sets_value(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.update_branch_best(self.key, "crane", 2.0, max_depth=3)
        guess, erd = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_update_branch_best_is_monotone(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.update_branch_best(self.key, "crane", 2.0)
        self.q.update_branch_best(self.key, "slate", 3.0)  # worse — must be rejected
        guess, erd = self.q.read_branch_best(self.key)
        self.assertEqual(guess, "crane")
        self.assertAlmostEqual(erd, 2.0)

    def test_read_branch_best_returns_none_none_for_missing_key(self):
        self.assertEqual(self.q.read_branch_best(b"notakey"), (None, None))

    def test_mark_branch_tainted_sets_flag(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5, budget=3)
        self.q.mark_branch_tainted(self.key)
        meta = self.q.read_branch_meta(self.key)
        self.assertTrue(meta[3])  # tainted

    def test_mark_branch_tainted_is_monotone(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5, budget=3)
        self.q.mark_branch_tainted(self.key)
        self.q.mark_branch_tainted(self.key)  # no-op — can't un-taint
        self.assertTrue(self.q.read_branch_meta(self.key)[3])

    def test_read_branch_meta_full_tuple(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5, budget=4)
        self.q.update_branch_best(self.key, "crane", 1.5, max_depth=3)
        meta = self.q.read_branch_meta(self.key)
        self.assertEqual(meta[0], "crane")
        self.assertAlmostEqual(meta[1], 1.5)
        self.assertEqual(meta[2], 3)    # best_max_depth
        self.assertFalse(meta[3])       # not tainted
        self.assertEqual(meta[4], 4)    # budget

    def test_try_finalize_branch_exactly_one_winner(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.assertTrue(self.q.try_finalize_branch(self.key))
        self.assertFalse(self.q.try_finalize_branch(self.key))

    def test_read_branch_meta_returns_none_for_missing_key(self):
        self.assertIsNone(self.q.read_branch_meta(b"notakey"))


class TestChunkCoordination(_TmpQueue):
    def setUp(self):
        super().setUp()
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 7)
        self.n_chunks = ERDQueue.n_chunks_for(N_CANDIDATES, 7)

    def test_claim_chunk_returns_none_for_finalized_branch(self):
        self.q.try_finalize_branch(self.key)  # transitions status to 'finalized'
        idx = self.q.claim_chunk(self.key, "worker-0", self.n_chunks)
        self.assertIsNone(idx)

    def test_claim_chunk_returns_none_for_missing_branch(self):
        # Branch doesn't exist at all (never created).
        key2 = ScoreCache.encode_subset(WORDS[:2])
        idx = self.q.claim_chunk(key2, "worker-0", 5)
        self.assertIsNone(idx)

    def test_claim_chunk_returns_each_index_exactly_once(self):
        claimed = set()
        for _ in range(self.n_chunks):
            idx = self.q.claim_chunk(self.key, "worker-0", self.n_chunks)
            self.assertIsNotNone(idx)
            claimed.add(idx)
        self.assertEqual(len(claimed), self.n_chunks)
        # All chunks claimed — next call returns None.
        self.assertIsNone(self.q.claim_chunk(self.key, "worker-0", self.n_chunks))

    def test_complete_chunk_marks_done(self):
        idx = self.q.claim_chunk(self.key, "worker-0", self.n_chunks)
        self.q.complete_chunk(self.key, idx)
        self.assertEqual(self.q.branch_done_chunks(self.key), 1)

    def test_chunks_for_branch_returns_rows(self):
        self.q.claim_chunk(self.key, "worker-0", self.n_chunks)
        rows = self.q.chunks_for_branch(self.key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["idx"], 0)

    def test_chunks_for_branch_empty_after_delete(self):
        self.q.claim_chunk(self.key, "worker-0", self.n_chunks)
        self.q.delete_branch(self.key)
        self.assertEqual(self.q.chunks_for_branch(self.key), [])


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

    def test_reset_active_branches_clears_d0_branches_and_chunks(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.claim_chunk(self.key, "worker-0",
                           ERDQueue.n_chunks_for(N_CANDIDATES, 5))
        n_b, n_c = self.q.reset_active_branches()
        self.assertEqual(n_b, 1)
        self.assertGreaterEqual(n_c, 1)
        self.assertIsNone(self.q.get_branch(self.key))
        self.assertEqual(self.q.chunks_for_branch(self.key), [])

    def test_reset_active_branches_preserves_cooperative_branch_progress(self):
        # D=0 branch (has pending_branches backup — wiped on reset).
        d0_key = self.key
        self.q.create_branch(d0_key, len(WORDS), N_CANDIDATES, 5, depth=0)

        # D=1 cooperative branch (no pending_branches row — must survive reset).
        coop_words = WORDS[:3]
        coop_key = ScoreCache.encode_subset(coop_words)
        n_chunks = ERDQueue.n_chunks_for(N_CANDIDATES, 5)
        self.q.create_branch(coop_key, len(coop_words), N_CANDIDATES, 5, depth=1)
        # Simulate two chunks: idx 0 completed, idx 1 stale in-flight.
        idx0 = self.q.claim_chunk(coop_key, "worker-0", n_chunks)
        self.q.complete_chunk(coop_key, idx0)
        idx1 = self.q.claim_chunk(coop_key, "worker-0", n_chunks)
        self.assertIsNotNone(idx1)

        n_b, n_c = self.q.reset_active_branches()

        # D=0 branch is gone.
        self.assertEqual(n_b, 1)
        self.assertIsNone(self.q.get_branch(d0_key))

        # Cooperative branch row survives.
        self.assertIsNotNone(self.q.get_branch(coop_key))

        chunks = self.q.chunks_for_branch(coop_key)
        # Completed chunk (done=1) is preserved so its progress isn't lost.
        done_chunks = [c for c in chunks if c["done"] == 1]
        self.assertEqual(len(done_chunks), 1)
        self.assertEqual(done_chunks[0]["idx"], idx0)

        # Stale in-flight claim (done=0) is freed so the chunk becomes a
        # re-claimable gap for the next worker.
        inflight_chunks = [c for c in chunks if c["done"] == 0]
        self.assertEqual(inflight_chunks, [])

        # The freed gap is claimable again.
        reclaimed = self.q.claim_chunk(coop_key, "worker-1", n_chunks)
        self.assertEqual(reclaimed, idx1)


class TestCancelAndInspection(_TmpQueue):
    def test_cancel_active_branch_removes_branch_and_chunks(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.claim_chunk(self.key, "worker-0",
                           ERDQueue.n_chunks_for(N_CANDIDATES, 5))
        self.q.add_pending_many([(self.key, len(WORDS), 0, "crane", 0)])
        self.q.cancel_active_branch(self.key, remove_from_queue=False)
        self.assertIsNone(self.q.get_branch(self.key))
        self.assertEqual(self.q.chunks_for_branch(self.key), [])
        self.assertIsNotNone(self.q.get_pending_branch(self.key))  # pending survives

    def test_cancel_active_branch_with_remove_from_queue(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
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
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.create_branch(k2, 3, N_CANDIDATES, 5)
        result = self.q.active_branches_by_keys([self.key])
        self.assertIn(self.key, result)
        self.assertNotIn(k2, result)

    def test_active_branches_by_keys_empty_input_returns_empty(self):
        self.assertEqual(self.q.active_branches_by_keys([]), {})

    def test_heartbeats_with_branch_joins_worker_and_branch(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, chunks_done=0)
        rows = self.q.heartbeats_with_branch()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["worker_id"], "worker-0")

    def test_worker_counts_by_branch_counts_live_workers(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, chunks_done=0)
        counts = self.q.worker_counts_by_branch(timeout_seconds=60)
        self.assertEqual(counts.get(self.key), 1)

    def test_worker_counts_by_branch_excludes_stale_heartbeats(self):
        now = int(time.time())
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.q.heartbeat("worker-0", pid=1, current_branch_key=self.key,
                         n_words=len(WORDS), started_at=now, chunks_done=0)
        self.q._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = 0 WHERE worker_id = 'worker-0'")
        counts = self.q.worker_counts_by_branch(timeout_seconds=30)
        self.assertEqual(counts.get(self.key, 0), 0)

    def test_branches_in_progress_returns_open_branches(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
        self.assertEqual(len(self.q.branches_in_progress()), 1)

    def test_branches_in_progress_excludes_finalized_branches(self):
        self.q.create_branch(self.key, len(WORDS), N_CANDIDATES, 5)
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


if __name__ == "__main__":
    unittest.main()
