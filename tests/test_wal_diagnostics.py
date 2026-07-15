"""Tests for the queue WAL diagnostics and hard-ceiling backstop:
ERDQueue's per-table WAL traffic attribution, and
erd_search._enforce_wal_hard_ceiling, which latches the swarm down (rather than
letting the WAL fill the disk and corrupt the database) when a quiesce/TRUNCATE
never wins.
"""
import os
import tempfile
import unittest
from unittest import mock

import erd_search
from erd_queue import ERDQueue, encode_subset

N_CANDIDATES = 40
_ORDER = list(range(N_CANDIDATES))


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.path)
        self.addCleanup(self.q.close)
        self.key = encode_subset(["crane", "slate", "trace", "stale", "tales"])
        self.q.create_branch(self.key, 5, N_CANDIDATES)


class TestWALTrafficAttribution(_TmpQueue):
    def test_bulk_elimination_is_attributed_to_candidate_claims(self):
        # A tight bound eliminates every candidate in one bulk INSERT OR REPLACE.
        self.q.update_branch_best(self.key, "salet", 1.0)
        self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES,
            small_count=5, count_cap=500)
        rows, byts = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/bulk-eliminate"], N_CANDIDATES)
        # The fat branch_key blob dominates the byte estimate, so it is well
        # above a rows-only count.
        self.assertGreater(byts["candidate_claims/bulk-eliminate"],
                           N_CANDIDATES * len(self.key))

    def test_survivor_claim_is_attributed(self):
        # A loose bound eliminates nothing and hands out a survivor bundle.
        _, indices, _ = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.9] * N_CANDIDATES,
            small_count=5, count_cap=500)
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/claim"], len(indices))

    def test_mark_claims_done_is_attributed(self):
        self.q.mark_claims_done(self.key, [0, 1, 2, 3])
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/publisher-mark-done"], 4)

    def test_add_pending_many_is_attributed(self):
        other = encode_subset(["abbey", "abide", "abode"])
        self.q.add_pending_many([(other, 3, 0, "crane", "ggggg")])
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["pending_branches/add"], 1)

    # -- the update/delete write paths, so the report is not insert-only -------

    def _claim_a_bundle(self):
        return self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.9] * N_CANDIDATES,
            small_count=5, count_cap=500)

    def test_complete_candidate_update_is_attributed(self):
        _, indices, _ = self._claim_a_bundle()
        self.q.complete_candidate(self.key, indices[0])
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/complete"], 1)

    def test_delete_branch_deletes_are_attributed(self):
        self._claim_a_bundle()
        self.q.delete_branch(self.key)
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertGreater(rows["candidate_claims/delete-branch"], 0)

    def test_republish_delete_and_upsert_are_attributed(self):
        bundle_id, indices, _ = self._claim_a_bundle()
        self.q.republish_remainder(self.key, bundle_id, indices)
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/republish-delete"], len(indices))
        self.assertEqual(
            rows["candidate_republish/republish-upsert"], len(indices))

    def test_reclaim_paths_are_attributed(self):
        self.q._conn.execute(
            "INSERT INTO candidate_claims (branch_key, idx, claimed_by, "
            "claimed_at, done) VALUES (?, 0, 'dead', 0, 0)", (self.key,))
        self.assertEqual(self.q.reclaim_stale_claims(120), 1)
        self.q._conn.execute(
            "INSERT INTO candidate_claims (branch_key, idx, claimed_by, "
            "claimed_at, done) VALUES (?, 1, 'w9', 0, 0)", (self.key,))
        self.assertEqual(self.q.reclaim_claims_of_worker("w9"), 1)
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["candidate_claims/reclaim-stale"], 1)
        self.assertEqual(rows["candidate_claims/reclaim-worker"], 1)

    def test_add_nodes_spent_update_is_attributed(self):
        self.q.add_nodes_spent(self.key, 500)
        rows, _ = self.q.wal_traffic_snapshot()
        self.assertEqual(rows["active_branches/nodes-spent"], 1)

    def test_snapshot_is_a_copy_so_deltas_are_stable(self):
        before, _ = self.q.wal_traffic_snapshot()
        self.q.mark_claims_done(self.key, [0, 1, 2])
        after, _ = self.q.wal_traffic_snapshot()
        # The earlier snapshot must not have moved when new traffic landed.
        self.assertEqual(before["candidate_claims/publisher-mark-done"], 0)
        self.assertEqual(after["candidate_claims/publisher-mark-done"], 3)

    def test_report_lists_categories_largest_first(self):
        self.assertEqual(self.q.wal_traffic_report(), "")
        self.q.update_branch_best(self.key, "salet", 1.0)
        self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES,
            small_count=5, count_cap=500)
        report = self.q.wal_traffic_report()
        self.assertIn("candidate_claims/bulk-eliminate", report)
        self.assertIn("rows", report)


class TestEnforceWALHardCeiling(_TmpQueue):
    def test_below_ceiling_is_a_noop(self):
        with mock.patch.object(self.q, "wal_size_bytes", return_value=1024):
            with mock.patch.object(self.q, "set_disk_stop") as stop:
                tripped = erd_search._enforce_wal_hard_ceiling(self.q, {})
        self.assertFalse(tripped)
        stop.assert_not_called()
        self.assertIsNone(self.q.disk_stop())

    def test_breach_latches_disk_stop_and_reports(self):
        over = erd_search.QUEUE_WAL_HARD_CEILING_BYTES + 1
        with mock.patch.object(self.q, "wal_size_bytes", return_value=over):
            with mock.patch("time.sleep"):  # skip the flush pause
                tripped = erd_search._enforce_wal_hard_ceiling(self.q, {})
        self.assertTrue(tripped)
        latch = self.q.disk_stop()
        self.assertIsNotNone(latch)
        self.assertIn("WAL", latch["reason"])

    def test_breach_signals_live_workers_for_stack_dump(self):
        over = erd_search.QUEUE_WAL_HARD_CEILING_BYTES + 1
        alive = mock.Mock()
        alive.is_alive.return_value = True
        alive.pid = 4242
        dead = mock.Mock()
        dead.is_alive.return_value = False
        dead.pid = 4243
        procs = {0: (alive, 0.0), 1: (dead, 0.0)}
        with mock.patch.object(self.q, "wal_size_bytes", return_value=over):
            with mock.patch("time.sleep"):
                with mock.patch("os.kill") as kill:
                    erd_search._enforce_wal_hard_ceiling(self.q, procs)
        # Only the live worker is signalled, and with SIGUSR1.
        kill.assert_called_once_with(4242, erd_search.signal.SIGUSR1)


if __name__ == "__main__":
    unittest.main()
