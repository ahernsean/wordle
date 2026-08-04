"""Unit tests for erd_search.cmd_queue_source_priority (issue #206).

Exercises the CLI surface for ERDQueue.set_source_work_priority(): word ->
source_work_id resolution, ambiguous-word disambiguation via --id, and the
distinct out-of-range / unknown-word / ambiguous / complete-request messages.
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

WORDS_A = ["crane", "slate", "trace", "stale", "tales"]
WORDS_B = ["crane", "slate", "trace", "stale"]
WORDS_C = ["crane", "slate", "trace"]


def _make_args(queue_path, **overrides):
    args = types.SimpleNamespace(
        word="salet",
        priority=1,
        id=None,
        queue=queue_path,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestQueueSourcePriority(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, 'queue.sqlite3')

    def _run(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            erd_search.cmd_queue_source_priority(args)
        return buf.getvalue()

    def test_sets_priority_for_pending_and_active_descendant(self):
        queue = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(WORDS_A)
        key_b = ScoreCache.encode_subset(WORDS_B)
        queue.add_pending_many([
            (key_a, len(WORDS_A), 0, 'salet', 0),
            (key_b, len(WORDS_B), 0, 'salet', 1),
        ])
        claimed = queue.claim_next('worker-0')
        queue.create_branch(key_a, len(WORDS_A), 20,
                             priority=claimed['priority'],
                             source_work_id=claimed['source_work_id'])
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=9))
        self.assertIn('priority set to 9', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_branch(key_a)['priority'], 9)
        self.assertEqual(
            queue.get_pending_branch(key_b)['priority'], 9)

    def test_out_of_range_priority_rejected_without_traceback(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key, len(WORDS_A), 0, 'salet', 0)])
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=1000))
        self.assertIn('between 0 and 999', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_pending_branch(key)['priority'], 0)

    def test_unknown_word_reported_distinctly(self):
        queue = ERDQueue(self.queue_path)
        queue.close()

        output = self._run(_make_args(self.queue_path, word='zzzzz'))
        self.assertIn('no source-work request found', output)

    def test_ambiguous_word_lists_candidate_ids_and_requires_disambiguation(self):
        queue = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(WORDS_A)
        key_b = ScoreCache.encode_subset(WORDS_B)
        # Distinct priorities -> distinct source_work rows for the same word.
        queue.add_pending_many([(key_a, len(WORDS_A), 0, 'salet', 0)])
        queue.add_pending_many([(key_b, len(WORDS_B), 1, 'salet', 0)])
        rows = queue.source_work_rows()
        self.assertEqual(len(rows), 2)
        ids = sorted(row['source_work_id'] for row in rows)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=5))
        self.assertIn('ambiguous', output)
        for source_work_id in ids:
            self.assertIn(str(source_work_id), output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        priorities = sorted(row['requested_priority']
                             for row in queue.source_work_rows())
        self.assertEqual(priorities, [0, 1])

        output = self._run(_make_args(self.queue_path, priority=5,
                                       id=ids[0]))
        self.assertIn('priority set to 5', output)

    def test_id_disambiguates(self):
        queue = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(WORDS_A)
        key_b = ScoreCache.encode_subset(WORDS_B)
        queue.add_pending_many([(key_a, len(WORDS_A), 0, 'salet', 0)])
        queue.add_pending_many([(key_b, len(WORDS_B), 1, 'salet', 0)])
        rows = {row['requested_priority']: row['source_work_id']
                for row in queue.source_work_rows()}
        target_id = rows[0]
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=7,
                                       id=target_id))
        self.assertIn('priority set to 7', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        by_id = {row['source_work_id']: row['requested_priority']
                  for row in queue.source_work_rows()}
        self.assertEqual(by_id[target_id], 7)
        self.assertEqual(by_id[rows[1]], 1)

    def test_unknown_id_for_word_reported_distinctly(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key, len(WORDS_A), 0, 'salet', 0)])
        rows = queue.source_work_rows()
        bogus_id = max(row['source_work_id'] for row in rows) + 1000
        queue.close()

        output = self._run(_make_args(self.queue_path, id=bogus_id))
        self.assertIn(f'no source-work request with id {bogus_id}', output)

    def test_completed_request_reported_distinctly(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_C)
        queue.add_pending_many([(key, len(WORDS_C), 0, 'salet', 0)])
        source_work_id = queue.source_work_rows()[0]['source_work_id']
        queue.claim_next('worker-0', source_work_id)
        queue.mark_done(key)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=2))
        self.assertIn('request is complete, cannot reprioritize', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(
            queue.source_work_rows()[0]['requested_priority'], 0)


if __name__ == '__main__':
    unittest.main()
