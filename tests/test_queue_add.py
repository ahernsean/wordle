"""Unit tests for erd_search.cmd_queue_add's --max-branch-size handling.

Covers issue #77: the default must queue every branch with >= 2 answer
words, including branches too large for the old 300-word default cap.
"""
import os
import re
import tempfile
import types
import unittest
from collections import namedtuple
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import erd_search
from erd_queue import ERDQueue, encode_subset
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache, load_word_list
from cache_sqlite import ScoreCache

# "fuzzy"'s all-gray branch (code 0) has ~1,868 answer words in
# all_answers.txt -- well over the old 300-word default cap.
LARGE_BRANCH_WORD = 'fuzzy'
SECOND_WORD = 'salet'
NON_CANDIDATE_ENGLISH_WORD = 'bogon'
NON_CANDIDATE_SURNAME = 'ahern'


def _make_args(tmp_dir, **overrides):
    args = types.SimpleNamespace(
        word=[LARGE_BRANCH_WORD],
        words_file=None,
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


SummaryCounts = namedtuple(
    'SummaryCounts', ['new', 'already_queued', 'already_solved', 'total'])

_SUMMARY_LINE_RE = re.compile(
    r'branch\(es\) queued across [\d,]+ word\(s\): '
    r'([\d,]+) new, ([\d,]+) already queued, '
    r'([\d,]+) already solved\.\s+'
    r'Queue total: ([\d,]+)')


def _parse_summary_counts(output):
    """Extract counts from cmd_queue_add's closing summary line specifically
    (not any per-word line, which shares the same "N new (...), M already
    queued (...)" shape but reports one word's counts instead of the run's
    totals) -- anchored on the "processed across ... Queue total:" text that
    only the closing line contains."""
    match = _SUMMARY_LINE_RE.search(output)
    return SummaryCounts(*(int(group.replace(',', ''))
                           for group in match.groups()))


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

    def test_non_candidate_words_are_rejected_before_queue_creation(self):
        candidate_words = load_word_list(erd_search.WORDS_FILE)
        for word in (NON_CANDIDATE_ENGLISH_WORD, NON_CANDIDATE_SURNAME):
            with self.subTest(word=word):
                self.assertEqual(len(word), 5)
                self.assertNotIn(word, candidate_words)
                args = _make_args(self._tmp.name, word=[word])

                with self.assertRaisesRegex(ValueError, 'invalid candidate word'):
                    erd_search.cmd_queue_add(args)

                self.assertFalse(os.path.exists(args.queue))

    def test_words_file_with_invalid_word_is_rejected_atomically(self):
        words_file_path = os.path.join(self._tmp.name, 'words.txt')
        with open(words_file_path, 'w') as word_file:
            word_file.write(f'{LARGE_BRANCH_WORD}\n{NON_CANDIDATE_ENGLISH_WORD}\n')
        args = _make_args(self._tmp.name, word=None, words_file=words_file_path)

        with self.assertRaisesRegex(ValueError, NON_CANDIDATE_ENGLISH_WORD):
            erd_search.cmd_queue_add(args)

        self.assertFalse(os.path.exists(args.queue))

    def test_cli_reports_invalid_word_as_an_error(self):
        args = _make_args(self._tmp.name, word=[NON_CANDIDATE_ENGLISH_WORD])
        error_output = StringIO()

        with patch.object(erd_search.sys, 'argv', [
                'erd_search.py', 'queue', 'add', '--word',
                NON_CANDIDATE_ENGLISH_WORD,
                '--cache', args.cache, '--queue', args.queue]), \
                redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                erd_search.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            f'invalid candidate word(s): {NON_CANDIDATE_ENGLISH_WORD}',
            error_output.getvalue())
        self.assertFalse(os.path.exists(args.queue))

    def test_cli_word_flag_takes_multiple_space_separated_words(self):
        args = _make_args(self._tmp.name)

        with patch.object(erd_search.sys, 'argv', [
                'erd_search.py', 'queue', 'add', '--word',
                LARGE_BRANCH_WORD, SECOND_WORD,
                '--cache', args.cache, '--queue', args.queue]):
            erd_search.main()

        queue = ERDQueue(args.queue)
        self.addCleanup(queue.close)
        branch_key = self._all_gray_branch_key()
        self.assertIsNotNone(queue.get_pending_branch(branch_key))
        self.assertGreater(queue.total_branches(), 0)

        all_answers = load_word_list(erd_search.ANSWER_FILE)
        score_cache = ScoreCache(
            os.path.join(self._tmp.name, 'probe2.sqlite3'), all_answers)
        self.addCleanup(score_cache.close)
        rcache = ResponseCache(all_answers, score_cache)
        second_groups = rcache.group_words(SECOND_WORD, all_answers)
        second_branch_key = encode_subset(next(
            branch for branch in second_groups.values() if len(branch) >= 2))
        self.assertIsNotNone(queue.get_pending_branch(second_branch_key))

    def test_rerunning_the_same_word_reports_already_queued_not_new(self):
        args = _make_args(self._tmp.name)
        first_run_output = StringIO()
        with redirect_stdout(first_run_output):
            erd_search.cmd_queue_add(args)
        first_summary = _parse_summary_counts(first_run_output.getvalue())
        self.assertGreater(first_summary.new, 0)

        second_run_output = StringIO()
        with redirect_stdout(second_run_output):
            erd_search.cmd_queue_add(args)

        second_summary = _parse_summary_counts(second_run_output.getvalue())
        self.assertEqual(second_summary.new, 0)
        self.assertEqual(second_summary.already_queued, first_summary.new)

    def test_already_cached_branch_is_not_queued(self):
        # A reusable result is terminal work, not a new queue request.
        branch_key = self._all_gray_branch_key()
        args = _make_args(self._tmp.name)

        all_answers = load_word_list(erd_search.ANSWER_FILE)
        score_cache = ScoreCache(args.cache, all_answers)
        score_cache.write(branch_key, ERD_ALL, 'salet', 3.5,
                          max_depth=GAME_GUESSES - 2, solve_budget=None)
        score_cache.checkpoint()
        score_cache.close()

        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(args)

        summary = _parse_summary_counts(output.getvalue())
        self.assertEqual(summary.already_solved, 1)
        self.assertEqual(summary.new + summary.already_queued, summary.total)
        queue = ERDQueue(args.queue)
        self.addCleanup(queue.close)
        self.assertIsNone(queue.get_pending_branch(branch_key))

    def test_fully_cached_word_reports_already_solved_without_source_work(self):
        branch_key = self._all_gray_branch_key()
        args = _make_args(self._tmp.name, pattern='-----')

        all_answers = load_word_list(erd_search.ANSWER_FILE)
        score_cache = ScoreCache(args.cache, all_answers)
        score_cache.write(branch_key, ERD_ALL, 'salet', 3.5,
                          max_depth=GAME_GUESSES - 2, solve_budget=None)
        score_cache.checkpoint()
        score_cache.close()

        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(args)

        self.assertIn(
            f'{LARGE_BRANCH_WORD.upper()}: already solved — 1 response '
            'group already cached; nothing queued.', output.getvalue())
        summary = _parse_summary_counts(output.getvalue())
        self.assertEqual(summary, SummaryCounts(0, 0, 1, 0))
        queue = ERDQueue(args.queue)
        self.addCleanup(queue.close)
        self.assertEqual(queue.total_branches(), 0)
        self.assertEqual(queue.source_work_rows(), [])


if __name__ == '__main__':
    unittest.main()
