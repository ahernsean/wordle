"""Unit tests for erd_search.cmd_queue_reconcile_stale_ownership (issue #215).

Exercises the CLI surface for ERDQueue.reconcile_orphaned_branch_ownership():
demoting a stranded open source-owned branch and reporting when there is
nothing to reconcile.
"""
import contextlib
import io
import os
import tempfile
import types
import unittest

import erd_search
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue

WORDS = ["crane", "slate", "trace", "stale", "tales"]
N_CANDIDATES = 20


def _make_args(queue_path):
    return types.SimpleNamespace(queue=queue_path)


class TestQueueReconcileStaleOwnership(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, 'queue.sqlite3')

    def _run(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            erd_search.cmd_queue_reconcile_stale_ownership(args)
        return buf.getvalue()

    def test_reports_no_orphans_on_a_healthy_queue(self):
        queue = ERDQueue(self.queue_path)
        queue.close()

        output = self._run(_make_args(self.queue_path))
        self.assertIn('No orphaned owned branches found.', output)

    def test_demotes_and_reports_a_stranded_branch(self):
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
        queue._conn.execute(
            'DELETE FROM branch_source_work WHERE branch_id = ?',
            (branch_id,))
        queue._conn.execute(
            "UPDATE source_work SET state = 'complete' "
            'WHERE source_work_id = ?', (crane['source_work_id'],))
        queue.close()

        output = self._run(_make_args(self.queue_path))
        self.assertIn('Demoted 1 orphaned owned branch(es)', output)
        self.assertIn(str(branch_id), output)

        queue = ERDQueue(self.queue_path)
        self.assertEqual(
            queue.get_active_branch(key)['requires_source_membership'], 0)
        self.assertEqual(queue.check_source_work_invariants(), [])
        queue.close()


if __name__ == '__main__':
    unittest.main()
