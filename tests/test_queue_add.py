"""Unit tests for erd_search.cmd_queue_add's --max-branch-size handling.

Covers issue #77: the default must queue every branch with >= 2 answer
words, including branches too large for the old 300-word default cap.
"""
import os
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import erd_search
from erd_queue import ERDQueue, encode_subset
from wordle_engine import ResponseCache, load_word_list
from cache_sqlite import ScoreCache

# "fuzzy"'s all-gray branch (code 0) has ~1,868 answer words in
# all_answers.txt -- well over the old 300-word default cap.
LARGE_BRANCH_WORD = 'fuzzy'


def _make_args(tmp_dir, **overrides):
    args = types.SimpleNamespace(
        word=LARGE_BRANCH_WORD,
        word_list=None,
        pattern=None,
        priority=0,
        priority_words=None,
        max_branch_size=None,
        delete_erd_cache=False,
        cache=os.path.join(tmp_dir, 'cache.sqlite3'),
        queue=os.path.join(tmp_dir, 'queue.sqlite3'),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestQueueAddMaxBranchSize(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _all_gray_branch_key(self):
        all_answers = load_word_list(erd_search.ANSWER_FILE)
        score_cache = ScoreCache(
            os.path.join(self._tmp.name, 'probe.sqlite3'), all_answers)
        self.addCleanup(score_cache.close)
        rcache = ResponseCache(all_answers, score_cache)
        groups = rcache.group_words(LARGE_BRANCH_WORD, all_answers)
        branch = groups[0]
        self.assertGreater(len(branch), 300)
        return encode_subset(branch)

    def test_default_queues_branch_larger_than_old_cap(self):
        branch_key = self._all_gray_branch_key()
        args = _make_args(self._tmp.name)

        erd_search.cmd_queue_add(args)

        queue = ERDQueue(args.queue)
        self.addCleanup(queue.close)
        self.assertIsNotNone(queue.get_pending_branch(branch_key))

    def test_explicit_max_branch_size_reproduces_old_behaviour(self):
        branch_key = self._all_gray_branch_key()
        args = _make_args(self._tmp.name, max_branch_size=300)

        erd_search.cmd_queue_add(args)

        queue = ERDQueue(args.queue)
        self.addCleanup(queue.close)
        self.assertIsNone(queue.get_pending_branch(branch_key))

    def test_invalid_word_is_rejected_before_queue_creation(self):
        args = _make_args(self._tmp.name, word='blah')

        with self.assertRaisesRegex(ValueError, 'invalid candidate word'):
            erd_search.cmd_queue_add(args)

        self.assertFalse(os.path.exists(args.queue))

    def test_word_list_with_invalid_word_is_rejected_atomically(self):
        word_list_path = os.path.join(self._tmp.name, 'words.txt')
        with open(word_list_path, 'w') as word_file:
            word_file.write('fuzzy\nblah\n')
        args = _make_args(self._tmp.name, word=None, word_list=word_list_path)

        with self.assertRaisesRegex(ValueError, 'blah'):
            erd_search.cmd_queue_add(args)

        self.assertFalse(os.path.exists(args.queue))

    def test_cli_reports_invalid_word_as_an_error(self):
        args = _make_args(self._tmp.name, word='blah')
        error_output = StringIO()

        with patch.object(erd_search.sys, 'argv', [
                'erd_search.py', 'queue', 'add', '--word', 'blah',
                '--cache', args.cache, '--queue', args.queue]), \
                redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                erd_search.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn('invalid candidate word(s): blah', error_output.getvalue())
        self.assertFalse(os.path.exists(args.queue))


if __name__ == '__main__':
    unittest.main()
