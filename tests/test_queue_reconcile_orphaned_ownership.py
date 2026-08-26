"""Unit tests for erd_search.cmd_queue_reconcile_orphaned_ownership and the
supervisor's periodic check_source_work_invariants() sweep (issue #215).

Exercises the CLI surface for ERDQueue.reconcile_orphaned_branch_ownership():
demoting a stranded open source-owned branch and verifying the demoted branch
is genuinely claimable afterward, not just visible. Also covers rejection of
an attempted source-owned branch creation after the response group completed,
and erd_search._check_source_work_invariants, the helper the supervisor loop's
periodic sweep calls.
"""
import contextlib
import io
import logging
import os
import tempfile
import types
import unittest

import erd_search
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue

WORDS = ["crane", "slate", "trace", "stale", "tales"]
N_CANDIDATES = 20
_ZERO_LOWER_BOUND = [0.0] * N_CANDIDATES


def _make_args(queue_path):
    return types.SimpleNamespace(queue=queue_path)


def _capture_warnings(callback):
    """Run callback() with a handler collecting erd_search's log records."""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    erd_search.logger.addHandler(handler)
    try:
        callback()
    finally:
        erd_search.logger.removeHandler(handler)
    return records


class TestQueueReconcileOrphanedOwnership(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, 'queue.sqlite3')

    def _run(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            erd_search.cmd_queue_reconcile_orphaned_ownership(args)
        return buf.getvalue()

    def test_reports_no_orphans_on_a_healthy_queue(self):
        queue = ERDQueue(self.queue_path)
        queue.close()

        output = self._run(_make_args(self.queue_path))
        self.assertIn('No orphaned owned branches found.', output)

    def test_demotes_reports_and_restores_claimability(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS)
        queue.add_pending_many([(key, len(WORDS), 9, 'crane', 7)])
        crane = queue.claim_next('worker-0')
        queue.create_branch(
            key, len(WORDS), N_CANDIDATES, budget=5,
            priority=crane['priority'], source_word=crane['source_word'],
            source_pattern=crane['source_pattern'],
            source_work_id=crane['source_work_id'])
        branch_id = queue._intern_branch(key)
        # This is the branch's only membership, so retracting it must also
        # retire the request itself to keep the rest of the database
        # internally consistent while isolating the one violation under
        # test.
        queue._conn.execute(
            'DELETE FROM branch_source_work WHERE branch_id = ?',
            (branch_id,))
        queue._conn.execute(
            "UPDATE source_work SET state = 'complete' "
            'WHERE source_work_id = ?', (crane['source_work_id'],))
        queue.close()

        output = self._run(_make_args(self.queue_path))
        self.assertIn('Demoted 1 orphaned owned branch(es)', output)
        self.assertIn('claimable without a live opener-work membership',
                       output)
        self.assertIn(str(branch_id), output)

        queue = ERDQueue(self.queue_path)
        self.assertEqual(
            queue.get_active_branch(key)['requires_source_membership'], 0)
        self.assertEqual(queue.check_source_work_invariants(), [])
        # Visible is not the same as claimable: claim_next_bundle's
        # direct-claim guard also requires no unresolved membership row, so
        # demotion must restore both, not just flip the flag.
        self.assertIsNotNone(queue.claim_next_bundle(
            key, 'worker-1', N_CANDIDATES, list(range(N_CANDIDATES)),
            _ZERO_LOWER_BOUND))
        queue.close()

    def test_rejects_branch_created_after_source_response_group_completed(self):
        queue = ERDQueue(self.queue_path)
        root_key = ScoreCache.encode_subset(WORDS)
        queue.add_pending_many([(root_key, len(WORDS), 9, 'crane', 7)])
        crane = queue.claim_next('worker-0')
        queue.create_branch(
            root_key, len(WORDS), N_CANDIDATES, budget=5,
            priority=crane['priority'], source_word=crane['source_word'],
            source_pattern=crane['source_pattern'],
            source_work_id=crane['source_work_id'])
        queue.mark_done(root_key)
        queue.delete_branch(root_key)  # the request completes here
        self.assertEqual(queue._conn.execute(
            "SELECT state FROM source_work WHERE source_work_id = ?",
            (crane['source_work_id'],)).fetchone()[0], 'complete')

        late_key = ScoreCache.encode_subset(WORDS[:4])
        self.assertFalse(queue.create_branch(
            late_key, 4, N_CANDIDATES, budget=4,
            priority=crane['priority'], source_word=crane['source_word'],
            source_pattern=crane['source_pattern'],
            source_work_id=crane['source_work_id']))
        self.assertEqual(queue.direct_branches_in_progress(), [])
        self.assertIsNone(queue.get_active_branch(late_key))
        self.assertEqual(queue.check_source_work_invariants(), [])
        queue.close()


class TestCheckSourceWorkInvariants(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, 'q.sqlite3')
        self.q = ERDQueue(self.queue_path)
        self.addCleanup(self.q.close)

    def test_clean_queue_logs_nothing(self):
        records = _capture_warnings(
            lambda: erd_search._check_source_work_invariants(self.q))
        self.assertEqual(records, [])

    def test_violation_is_logged(self):
        key = ScoreCache.encode_subset(WORDS)
        self.q.add_pending_many([(key, len(WORDS), 9, "crane", 7)])
        crane = self.q.claim_next("worker-0")
        self.q.create_branch(
            key, len(WORDS), N_CANDIDATES, budget=5,
            priority=crane["priority"], source_word=crane["source_word"],
            source_pattern=crane["source_pattern"],
            source_work_id=crane["source_work_id"])
        branch_id = self.q._intern_branch(key)
        self.q._conn.execute(
            "DELETE FROM branch_source_work WHERE branch_id = ?",
            (branch_id,))
        self.q._conn.execute(
            "UPDATE source_work SET state = 'complete' "
            "WHERE source_work_id = ?", (crane["source_work_id"],))

        records = _capture_warnings(
            lambda: erd_search._check_source_work_invariants(self.q))
        joined = "\n".join(record.getMessage() for record in records)
        self.assertIn("source-owned open branch_id", joined)


if __name__ == '__main__':
    unittest.main()
