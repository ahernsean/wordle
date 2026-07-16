"""Every ERDQueue read/mutation that takes a branch_key must handle a key that
was never registered in the branches table (no claim, active, or pending row
was ever created for it) by returning the empty/false result via its
branch_id-is-None guard, never by raising.
"""
import os
import tempfile
import unittest

from erd_queue import ERDQueue, encode_subset


class TestUnregisteredBranchKey(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(self.q.close)
        # A key that has never been interned: no branches row exists for it.
        self.key = encode_subset(["crane", "slate", "trace"])

    def test_reads_return_empty_without_registering(self):
        self.assertIsNone(self.q._intern_branch(self.key))
        self.assertIsNone(self.q.get_branch(self.key))
        self.assertIsNone(self.q.get_active_branch(self.key))
        self.assertIsNone(self.q.get_pending_branch(self.key))
        self.assertEqual(self.q.read_branch_best(self.key), (None, None, None))
        self.assertIsNone(self.q.read_branch_meta(self.key))
        self.assertIsNone(self.q.read_cut_result(self.key))
        self.assertEqual(self.q.branch_done_candidates(self.key), 0)
        self.assertEqual(self.q.branch_bulk_done_candidates(self.key), 0)
        self.assertFalse(self.q.has_pending_row(self.key))
        self.assertEqual(self.q.claims_for_branch(self.key), [])
        self.assertEqual(self.q.status_by_branch_keys([self.key]), {})
        self.assertEqual(self.q.active_branches_by_keys([self.key]), {})
        # A pure read must not have created a registry row as a side effect.
        self.assertEqual(
            self.q._conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0],
            0)

    def test_mutations_are_noops_returning_false(self):
        self.assertFalse(self.q.requeue_pending(self.key))
        self.assertFalse(self.q.try_finalize_branch(self.key))
        self.assertFalse(self.q.reclaim_stale_finalize(self.key, 60))
        self.assertFalse(self.q.set_priority(self.key, 5))
        self.assertFalse(self.q.remove_pending(self.key))
        # These return None and must simply not raise.
        self.q.mark_done(self.key)
        self.q.delete_branch(self.key)
        self.q.cancel_active_branch(self.key, remove_from_queue=True)


if __name__ == "__main__":
    unittest.main()
