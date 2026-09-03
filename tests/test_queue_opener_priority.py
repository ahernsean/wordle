"""Unit tests for erd_search.cmd_queue_opener_priority (issue #206).

Exercises the CLI surface for ERDQueue.set_opener_work_priority(): word ->
opener_work_id resolution restricted to open (non-complete) requests, the
--opener-work-id disambiguation and direct-completed-request targeting, the
MAX(owner_priority) branch-sharing behaviour, and the distinct out-of-range /
unknown-word / all-complete / ambiguous / complete-request messages.
"""
import contextlib
import io
import os
import tempfile
import types
import unittest

import erd_search
from cache_sqlite import ScoreCache
from erd_queue import (
    OPENER_PRIORITY_MAX,
    OPENER_PRIORITY_MIN,
    ERDQueue,
)

WORDS_A = ["crane", "slate", "trace", "stale", "tales"]
WORDS_B = ["crane", "slate", "trace", "stale"]
WORDS_C = ["crane", "slate", "trace"]


def _make_args(queue_path, **overrides):
    args = types.SimpleNamespace(
        word="salet",
        opener_word=None,
        priority=1,
        opener_work_id=None,
        queue=queue_path,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestQueueOpenerPriority(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, 'queue.sqlite3')

    def _run(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            erd_search.cmd_queue_opener_priority(args)
        return buf.getvalue()

    def _run_branch_priority(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            erd_search.cmd_queue_priority(args)
        return buf.getvalue()

    def test_sets_priority_for_pending_root_active_root_and_descendant(self):
        queue = ERDQueue(self.queue_path)
        key_root = ScoreCache.encode_subset(WORDS_A)
        key_child = ScoreCache.encode_subset(WORDS_B)
        key_other_root = ScoreCache.encode_subset(WORDS_C)
        queue.add_pending_many([
            (key_root, len(WORDS_A), 0, 'salet', 0),
            (key_other_root, len(WORDS_C), 0, 'salet', 1),
        ])
        claimed = queue.claim_next('worker-0')
        queue.create_branch(key_root, len(WORDS_A), 20,
                             priority=claimed['priority'],
                             opener_work_id=claimed['opener_work_id'])
        # A promoted descendant discovered while solving key_root: created
        # directly (never queued), owned by the same opener-work request.
        queue.create_branch(key_child, len(WORDS_B), 5,
                             priority=claimed['priority'],
                             opener_work_id=claimed['opener_work_id'],
                             parent_branch_key=key_root)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=9))
        self.assertIn('requested priority set to 9', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_branch(key_root)['priority'], 9)
        self.assertEqual(queue.get_branch(key_child)['priority'], 9)
        self.assertEqual(
            queue.get_pending_branch(key_other_root)['priority'], 9)

    def test_branch_shared_by_two_live_requests_keeps_max_priority(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key, len(WORDS_A), 1, 'salet', 0)])
        queue.add_pending_many([(key, len(WORDS_A), 50, 'crane', 0)])
        self.assertEqual(queue.get_pending_branch(key)['priority'], 50)
        queue.close()

        # Lowering CRANE's priority must not drag the branch below what
        # SALET (still at 1) requires: MAX(owner_priority) wins.
        output = self._run(_make_args(self.queue_path, word='crane',
                                       priority=0))
        self.assertIn('requested priority set to 0', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_pending_branch(key)['priority'], 1)

    def test_out_of_range_priority_rejected_without_traceback(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key, len(WORDS_A), 0, 'salet', 0)])
        queue.close()

        output = self._run(_make_args(
            self.queue_path, priority=OPENER_PRIORITY_MAX + 1))
        self.assertIn(
            f'between {OPENER_PRIORITY_MIN} and {OPENER_PRIORITY_MAX}', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_pending_branch(key)['priority'], 0)

    def test_ownerless_opener_updates_only_open_ownerless_branches(self):
        queue = ERDQueue(self.queue_path)
        ownerless_key = ScoreCache.encode_subset(WORDS_A)
        owned_key = ScoreCache.encode_subset(WORDS_B)
        done_key = ScoreCache.encode_subset(WORDS_C)
        queue.create_branch(ownerless_key, len(WORDS_A), 20,
                            priority=0, opener='salet')
        queue.add_pending_many([(owned_key, len(WORDS_B), 1, 'salet', 0)])
        claimed = queue.claim_next('worker-0')
        queue.create_branch(owned_key, len(WORDS_B), 20,
                            priority=claimed['priority'], opener='salet',
                            opener_work_id=claimed['opener_work_id'])
        queue.create_branch(done_key, len(WORDS_C), 20,
                            priority=0, opener='salet')
        queue._conn.execute(
            "UPDATE active_branches SET status = 'finalized' WHERE branch_id = ?",
            (queue._intern_branch(done_key),))
        queue.close()

        output = self._run_branch_priority(_make_args(
            self.queue_path, opener_word='salet', priority=9))

        self.assertIn('1 ownerless open branch(es)', output)
        self.assertIn('ownerless', output)
        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(queue.get_active_branch(ownerless_key)['priority'], 9)
        self.assertEqual(queue.get_active_branch(owned_key)['priority'], 1)
        self.assertEqual(queue.get_active_branch(done_key)['priority'], 0)

    def test_unknown_word_reported_distinctly(self):
        queue = ERDQueue(self.queue_path)
        queue.close()

        output = self._run(_make_args(self.queue_path, word='zzzzz'))
        self.assertIn('no opener-work request found', output)

    def test_all_requests_complete_reported_distinctly_from_unknown_word(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_C)
        queue.add_pending_many([(key, len(WORDS_C), 0, 'salet', 0)])
        opener_work_id = queue.opener_work_rows()[0]['opener_work_id']
        queue.claim_next('worker-0', opener_work_id)
        queue.mark_done(key)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=2))
        self.assertIn('all 1 opener-work request(s) are complete', output)
        self.assertNotIn('no opener-work request found', output)

    def test_completed_request_does_not_make_later_request_ambiguous(self):
        # Regression for issue #206 review finding 1: a word's first request
        # completing must not leave the next queue add for that word stuck
        # behind a permanent "ambiguous" result.
        queue = ERDQueue(self.queue_path)
        key_done = ScoreCache.encode_subset(WORDS_C)
        queue.add_pending_many([(key_done, len(WORDS_C), 0, 'salet', 0)])
        done_id = queue.opener_work_rows()[0]['opener_work_id']
        queue.claim_next('worker-0', done_id)
        queue.mark_done(key_done)

        key_open = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key_open, len(WORDS_A), 0, 'salet', 0)])
        rows = queue.opener_work_rows()
        self.assertEqual(len(rows), 2)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=5))
        self.assertNotIn('ambiguous', output)
        self.assertIn('requested priority set to 5', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        by_id = {row['opener_work_id']: row['requested_priority']
                  for row in queue.opener_work_rows()}
        self.assertEqual(by_id[done_id], 0)

    def test_ambiguous_word_lists_candidate_details_and_requires_disambiguation(self):
        queue = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(WORDS_A)
        key_b = ScoreCache.encode_subset(WORDS_B)
        # Distinct priorities -> distinct opener_work rows for the same word.
        queue.add_pending_many([(key_a, len(WORDS_A), 0, 'salet', 0)])
        queue.add_pending_many([(key_b, len(WORDS_B), 1, 'salet', 0)])
        rows = queue.opener_work_rows()
        self.assertEqual(len(rows), 2)
        ids = sorted(row['opener_work_id'] for row in rows)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=5))
        self.assertIn('ambiguous', output)
        self.assertIn('--opener-work-id', output)
        for opener_work_id in ids:
            self.assertIn(f'id {opener_work_id}', output)
        self.assertIn('direct', output)
        self.assertIn('branch(es)', output)
        self.assertIn('queued', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        priorities = sorted(row['requested_priority']
                             for row in queue.opener_work_rows())
        self.assertEqual(priorities, [0, 1])

        output = self._run(_make_args(self.queue_path, priority=5,
                                       opener_work_id=ids[0]))
        self.assertIn('requested priority set to 5', output)

    def test_opener_work_id_disambiguates(self):
        queue = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(WORDS_A)
        key_b = ScoreCache.encode_subset(WORDS_B)
        queue.add_pending_many([(key_a, len(WORDS_A), 0, 'salet', 0)])
        queue.add_pending_many([(key_b, len(WORDS_B), 1, 'salet', 0)])
        rows = {row['requested_priority']: row['opener_work_id']
                for row in queue.opener_work_rows()}
        target_id = rows[0]
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=7,
                                       opener_work_id=target_id))
        self.assertIn('requested priority set to 7', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        by_id = {row['opener_work_id']: row['requested_priority']
                  for row in queue.opener_work_rows()}
        self.assertEqual(by_id[target_id], 7)
        self.assertEqual(by_id[rows[1]], 1)

    def test_unknown_id_for_word_reported_distinctly(self):
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_A)
        queue.add_pending_many([(key, len(WORDS_A), 0, 'salet', 0)])
        rows = queue.opener_work_rows()
        bogus_id = max(row['opener_work_id'] for row in rows) + 1000
        queue.close()

        output = self._run(_make_args(self.queue_path,
                                       opener_work_id=bogus_id))
        self.assertIn(f'no opener-work request with id {bogus_id}', output)

    def test_opener_work_id_can_target_completed_request_directly(self):
        # A completed id named explicitly is resolved (it belongs to the
        # word) and only then reported as complete -- not "not found".
        queue = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(WORDS_C)
        queue.add_pending_many([(key, len(WORDS_C), 0, 'salet', 0)])
        opener_work_id = queue.opener_work_rows()[0]['opener_work_id']
        queue.claim_next('worker-0', opener_work_id)
        queue.mark_done(key)
        queue.close()

        output = self._run(_make_args(self.queue_path, priority=2,
                                       opener_work_id=opener_work_id))
        self.assertIn('request is complete, cannot reprioritize', output)
        self.assertNotIn('no opener-work request with id', output)

        queue = ERDQueue(self.queue_path)
        self.addCleanup(queue.close)
        self.assertEqual(
            queue.opener_work_rows()[0]['requested_priority'], 0)


if __name__ == '__main__':
    unittest.main()
