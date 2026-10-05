"""erd_search reconcile-opener-erds: store the ERD of openers that lack one."""

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import erd_search
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
from wordle_engine import ERD_ALL

ANSWERS = ["crane", "slate"]


class ReconcileOpenerERDsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.queue_path = os.path.join(tmp.name, "queue.sqlite3")
        self.cache_path = os.path.join(tmp.name, "cache.sqlite3")
        answer_path = os.path.join(tmp.name, "answers.txt")
        with open(answer_path, "w") as answer_file:
            answer_file.write("\n".join(ANSWERS) + "\n")
        patcher = mock.patch.object(erd_search, "ANSWER_FILE", answer_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.args = argparse.Namespace(queue=self.queue_path,
                                       cache=self.cache_path)

    def _finish_branches_of(self, opener, flip):
        """Resolve an opener's only branch; optionally flip it done."""
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(ANSWERS)
        queue.add_pending_many([(key, len(ANSWERS), 0, opener, 0)])
        queue.claim_next("worker-0")
        queue.mark_done(key)
        if flip:
            queue.mark_openers_complete(queue.openers_ready_to_complete())
        queue.close()

    def _run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            erd_search.cmd_reconcile_opener_erds(self.args)
        return out.getvalue()

    def _stored(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        try:
            return cache.opener_names_with_erd(ERD_ALL)
        finally:
            cache.close()

    def _state(self, opener):
        queue = ERDQueue(self.queue_path)
        try:
            return queue._conn.execute(
                "SELECT state FROM opener_work WHERE opener = ?",
                (opener,)).fetchone()[0]
        finally:
            queue.close()

    def test_a_done_opener_with_no_erd_gets_one(self):
        self._finish_branches_of("crane", flip=True)

        output = self._run()

        self.assertEqual(self._stored(), {"crane"})
        self.assertIn("stored 1", output)

    def test_an_opener_that_resolved_but_was_never_finished_is_stored_and_finished(self):
        self._finish_branches_of("crane", flip=False)
        self.assertNotEqual(self._state("crane"), "complete")

        self._run()

        self.assertEqual(self._stored(), {"crane"})
        self.assertEqual(self._state("crane"), "complete")

    def test_an_opener_that_already_has_its_erd_is_left_alone(self):
        self._finish_branches_of("crane", flip=True)
        self._run()

        output = self._run()

        self.assertIn("0 opener(s) owed", output)

    def test_an_opener_whose_groups_are_unsettled_is_reported_and_not_stored(self):
        # HOWDY puts both answers in one group, which no result settles.
        self._finish_branches_of("howdy", flip=True)

        with self.assertRaises(SystemExit) as exit_info:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                erd_search.cmd_reconcile_opener_erds(self.args)

        self.assertEqual(exit_info.exception.code, 1)
        self.assertIn("howdy", output.getvalue())
        self.assertEqual(self._stored(), set())

    def test_an_infeasible_opener_is_reported_and_finished_with_no_erd(self):
        self._finish_branches_of("howdy", flip=False)
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write_loss(ScoreCache.encode_subset(ANSWERS), ERD_ALL, 5)
        cache.close()

        output = self._run()

        self.assertIn("Infeasible", output)
        self.assertEqual(self._stored(), set())
        self.assertEqual(self._state("howdy"), "complete")

    def test_a_failed_store_is_reported_and_exits_nonzero(self):
        self._finish_branches_of("crane", flip=True)

        with mock.patch.object(
                ScoreCache, "write_opener_erd",
                side_effect=__import__("sqlite3").OperationalError("disk I/O error")):
            output = io.StringIO()
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(output):
                    erd_search.cmd_reconcile_opener_erds(self.args)

        self.assertIn("Could not store", output.getvalue())
        self.assertEqual(self._stored(), set())


if __name__ == "__main__":
    unittest.main()
