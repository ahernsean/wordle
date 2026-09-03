"""Unit tests for erd_search.cmd_queue_add.

Covers --max-branch-size handling (issue #77: the default must queue every
branch with >= 2 answer words, including branches too large for the old
300-word default cap) and the descending priority ladder that keeps a batch
of words from all starting at once.
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
from erd_queue import (
    OPENER_PRIORITY_MAX,
    OPENER_PRIORITY_MIN,
    ERDQueue,
    encode_subset,
)
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
        priority=None,
        priority_step=erd_search.DEFAULT_PRIORITY_STEP,
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

    def test_fully_cached_word_reports_already_solved_without_opener_work(self):
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
        self.assertEqual(queue.opener_work_rows(), [])


class TestQueueAddDeleteErdCache(unittest.TestCase):
    """--delete-erd-cache must leave a completed branch genuinely recomputable.

    Deleting the cache entry alone strands it: the pending row stays `done`,
    a surviving active_branches row defeats create_branch's INSERT OR IGNORE,
    and claims already marked done finalize it again without work.  The
    branch ends up neither cached nor scheduled.
    """

    N_CANDIDATES = 12

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.args = _make_args(self._tmp.name, pattern='g----')

    def _open_queue(self):
        queue = ERDQueue(self.args.queue)
        self.addCleanup(queue.close)
        return queue

    def _add(self, delete_erd_cache=False):
        self.args.delete_erd_cache = delete_erd_cache
        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(self.args)
        return output.getvalue()

    def _promote(self, queue, worker='worker-0'):
        """Claim the queued branch and register it as an open active branch."""
        branch_key = bytes(queue._conn.execute(
            'SELECT branch_key FROM branches LIMIT 1').fetchone()[0])
        n_words = queue._conn.execute(
            'SELECT n_words FROM pending_branches').fetchone()[0]
        claimed = queue.claim_next(worker)
        queue.create_branch(branch_key, n_words, self.N_CANDIDATES,
                            priority=claimed['priority'],
                            opener_work_id=claimed['opener_work_id'])
        return branch_key, claimed

    def _claim_a_bundle(self, queue, branch_key, worker='worker-0',
                        claimed=None):
        """Claim real candidate slots, so claim counts have something to prove.

        An opener-owned branch refuses a bundle whose expected_opener_work_id
        does not match its owner, so the claim row from _promote has to be
        carried through; a branch promoted without opener ownership passes
        None for both.
        """
        return queue.claim_next_bundle(
            branch_key, worker, self.N_CANDIDATES,
            list(range(self.N_CANDIDATES)), [0.0] * self.N_CANDIDATES,
            expected_opener_work_id=(
                claimed['opener_work_id'] if claimed else None),
            expected_opener_priority=(
                claimed['priority'] if claimed else None))

    def _status(self, queue):
        return queue._conn.execute(
            'SELECT status FROM pending_branches').fetchone()[0]

    def _counts(self, queue):
        one = lambda sql: queue._conn.execute(sql).fetchone()[0]
        return {
            'open_active': one("SELECT COUNT(*) FROM active_branches "
                               "WHERE status = 'open'"),
            'active': one('SELECT COUNT(*) FROM active_branches'),
            'claims': one('SELECT COUNT(*) FROM candidate_claims'),
        }

    def _finalize_as_the_swarm_does(self):
        """Drive a branch to the shape a real finalize leaves behind.

        BranchWorker calls mark_done and then delete_branch, so a completed
        branch carries no active row and no claims -- the state #277 is
        actually about.
        """
        self._add()
        queue = self._open_queue()
        branch_key, claimed = self._promote(queue)
        assert self._claim_a_bundle(queue, branch_key, claimed=claimed)
        queue.mark_done(branch_key)
        queue.delete_branch(branch_key)
        return queue, branch_key

    # -- the case #277 is about ------------------------------------------

    def test_branch_finalized_the_way_the_swarm_does_is_claimable_again(self):
        queue, _ = self._finalize_as_the_swarm_does()
        self.assertEqual(self._status(queue), 'done')
        self.assertEqual(self._counts(queue),
                         {'open_active': 0, 'active': 0, 'claims': 0})
        self.assertIsNone(queue.claim_next('worker-1'))

        self._add(delete_erd_cache=True)

        self.assertEqual(self._status(queue), 'pending')
        self.assertIsNotNone(queue.claim_next('worker-1'))

    def test_pending_row_is_reset_in_place_never_removed(self):
        queue, branch_key = self._finalize_as_the_swarm_does()

        self._add(delete_erd_cache=True)

        # has_pending_row is what suppresses the alpha-beta ceiling for a
        # user-queued branch.  A row that blinked out could be re-created
        # under an immutable ceiling and finalize as an uncacheable cut.
        self.assertTrue(queue.has_pending_row(branch_key))
        self.assertEqual(queue._conn.execute(
            'SELECT COUNT(*) FROM pending_branches').fetchone()[0], 1)

    # -- residue left by a crash between mark_done and delete_branch -------

    def test_branch_still_open_after_mark_done_is_left_entirely_alone(self):
        """mark_done without delete_branch leaves the branch open.

        Whether that is a crashed finalize or a live descendant solve cannot
        be told apart from these tables, so the branch is skipped -- and its
        cache entry is kept, because deleting the result while refusing to
        requeue would strand it exactly the way #277 describes.
        """
        self._add()
        queue = self._open_queue()
        branch_key, claimed = self._promote(queue)
        self.assertIsNotNone(
            self._claim_a_bundle(queue, branch_key, claimed=claimed))
        queue.mark_done(branch_key)          # no delete_branch
        before = self._counts(queue)
        self.assertEqual(self._status(queue), 'done')
        self.assertGreater(before['claims'], 0)
        self.assertEqual(before['open_active'], 1)

        output = self._add(delete_erd_cache=True)

        self.assertEqual(self._counts(queue), before)
        self.assertEqual(self._status(queue), 'done')
        self.assertNotIn('cleared for recompute', output)
        self.assertIn('being solved right now', output)

    def test_a_busy_branch_keeps_its_cache_entry(self):
        """The cache delete and the requeue are one decision, not two."""
        self._add()
        queue = self._open_queue()
        branch_key, claimed = self._promote(queue)
        self._claim_a_bundle(queue, branch_key, claimed=claimed)
        queue.mark_done(branch_key)

        all_answers = load_word_list(erd_search.ANSWER_FILE)
        score_cache = ScoreCache(self.args.cache, all_answers)
        score_cache.write(branch_key, ERD_ALL, 'salet', 3.5,
                          max_depth=GAME_GUESSES - 2, solve_budget=None)
        score_cache.checkpoint()
        score_cache.close()

        self._add(delete_erd_cache=True)

        score_cache = ScoreCache(self.args.cache, all_answers)
        self.addCleanup(score_cache.close)
        # Skipping the requeue but dropping the result would leave the branch
        # with neither -- the state the whole change exists to remove.
        self.assertIsNotNone(score_cache.read(branch_key, ERD_ALL))

    # -- branches a worker is still solving --------------------------------

    def test_in_progress_branch_is_not_reset_underneath_its_worker(self):
        self._add()
        queue = self._open_queue()
        self.assertIsNotNone(queue.claim_next('worker-0'))
        self.assertEqual(self._status(queue), 'in_progress')

        output = self._add(delete_erd_cache=True)

        self.assertEqual(self._status(queue), 'in_progress')
        self.assertIsNone(queue.claim_next('worker-1'))
        self.assertNotIn('cleared for recompute', output)

    def test_branch_re_promoted_as_a_descendant_is_not_reset(self):
        """A `done` pending row does not mean the branch is idle.

        create_branch takes any branch key regardless of pending status, so a
        branch finished for one request can be re-promoted as another
        request's descendant and hold live claims while its pending row still
        reads `done`.  Reading only pending_branches would delete an open
        active row and live claims out from under the worker holding them.
        """
        queue, branch_key = self._finalize_as_the_swarm_does()
        queue.create_branch(
            branch_key,
            queue._conn.execute(
                'SELECT n_words FROM pending_branches').fetchone()[0],
            self.N_CANDIDATES, priority=0)
        self.assertIsNotNone(self._claim_a_bundle(queue, branch_key, 'worker-9'))
        before = self._counts(queue)
        self.assertEqual(before['open_active'], 1)
        self.assertGreater(before['claims'], 0)
        self.assertEqual(self._status(queue), 'done')

        output = self._add(delete_erd_cache=True)

        self.assertEqual(self._counts(queue), before)
        self.assertEqual(self._status(queue), 'done')
        self.assertNotIn('cleared for recompute', output)

    # -- accounting and non-targets ----------------------------------------

    def test_reset_is_reported(self):
        self._finalize_as_the_swarm_does()

        output = self._add(delete_erd_cache=True)

        self.assertIn('1 completed branch(es) cleared for recompute', output)

    def test_without_the_flag_a_completed_branch_stays_done(self):
        queue, _ = self._finalize_as_the_swarm_does()

        output = self._add()

        self.assertEqual(self._status(queue), 'done')
        self.assertNotIn('cleared for recompute', output)

    def test_reset_ignores_branches_that_were_never_queued(self):
        output = self._add(delete_erd_cache=True)

        queue = self._open_queue()
        self.assertEqual(self._status(queue), 'pending')
        # A fresh branch reaches 'pending' whether it was ignored or reset and
        # re-added; only the count distinguishes them.
        self.assertNotIn('cleared for recompute', output)


class TestRequeueCompletedBranch(unittest.TestCase):
    """ERDQueue.requeue_completed_branch's refusal contract."""

    N_CANDIDATES = 8

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue = ERDQueue(os.path.join(self._tmp.name, 'q.sqlite3'))
        self.addCleanup(self.queue.close)
        self.branch_key = encode_subset(
            load_word_list(erd_search.ANSWER_FILE)[:6])
        self.queue.add_pending_many(
            [(self.branch_key, 6, 0, 'salet', 0)])

    def test_unknown_branch_is_refused(self):
        self.assertFalse(self.queue.requeue_completed_branch(b'nosuchbranch'))

    def test_pending_branch_is_refused(self):
        self.assertFalse(
            self.queue.requeue_completed_branch(self.branch_key))

    def test_in_progress_branch_is_refused(self):
        self.queue.claim_next('worker-0')
        self.assertFalse(
            self.queue.requeue_completed_branch(self.branch_key))

    def test_done_branch_with_an_open_active_row_is_refused(self):
        claimed = self.queue.claim_next('worker-0')
        self.queue.create_branch(self.branch_key, 6, self.N_CANDIDATES,
                                 priority=claimed['priority'],
                                 opener_work_id=claimed['opener_work_id'])
        self.queue.mark_done(self.branch_key)
        self.assertEqual(self.queue._conn.execute(
            "SELECT COUNT(*) FROM active_branches WHERE status = 'open'"
        ).fetchone()[0], 1)

        self.assertFalse(
            self.queue.requeue_completed_branch(self.branch_key))
        self.assertEqual(self.queue._conn.execute(
            'SELECT status FROM pending_branches').fetchone()[0], 'done')

    def test_done_branch_with_no_open_row_is_reset(self):
        claimed = self.queue.claim_next('worker-0')
        self.queue.create_branch(self.branch_key, 6, self.N_CANDIDATES,
                                 priority=claimed['priority'],
                                 opener_work_id=claimed['opener_work_id'])
        self.queue.mark_done(self.branch_key)
        self.queue.delete_branch(self.branch_key)

        self.assertTrue(
            self.queue.requeue_completed_branch(self.branch_key))
        self.assertEqual(self.queue._conn.execute(
            'SELECT status, claimed_by, claimed_at, completed_at '
            'FROM pending_branches').fetchone()[:],
            ('pending', None, None, None))


class TestPriorityLadder(unittest.TestCase):
    """priority_ladder's rung assignment, independent of the queue."""

    def test_first_word_takes_the_top_and_the_rest_descend(self):
        ladder = erd_search.priority_ladder(['alpha', 'bravo', 'delta'], 10, 5)

        self.assertEqual(ladder, {'alpha': 10, 'bravo': 5, 'delta': 0})

    def test_top_priority_lifts_the_whole_ladder(self):
        ladder = erd_search.priority_ladder(['alpha', 'bravo'], 105, 5)

        self.assertEqual(ladder, {'alpha': 105, 'bravo': 100})

    def test_zero_step_ties_every_word_at_the_top(self):
        ladder = erd_search.priority_ladder(['alpha', 'bravo', 'delta'], 7, 0)

        self.assertEqual(ladder, {'alpha': 7, 'bravo': 7, 'delta': 7})

    def test_single_word_sits_on_the_top(self):
        self.assertEqual(erd_search.priority_ladder(['alpha'], 4, 5),
                         {'alpha': 4})

    def test_no_rung_falls_below_the_opener_priority_minimum(self):
        words = [f'w{index:05d}' for index in range(500)]

        ladder = erd_search.priority_ladder(words, OPENER_PRIORITY_MAX, 5)

        self.assertLessEqual(max(ladder.values()), OPENER_PRIORITY_MAX)
        self.assertGreaterEqual(min(ladder.values()), 0)

    def test_overflowing_list_seats_leading_words_and_floors_the_tail(self):
        # 5 rungs fit at or above 0 below a top of 20: 20, 15, 10, 5, 0.
        words = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

        ladder = erd_search.priority_ladder(words, 20, 5)

        self.assertEqual(ladder, {'a': 20, 'b': 15, 'c': 10, 'd': 5,
                                  'e': 0, 'f': 0, 'g': 0})

    def test_range_seats_every_candidate_word_on_its_own_rung(self):
        # The sweep's end state: every candidate laddered at the default step,
        # each on a distinct rung, with room left below to append more.
        candidate_words = load_word_list(erd_search.WORDS_FILE)
        self.assertGreater(len(candidate_words), 14_000)

        ladder = erd_search.priority_ladder(
            candidate_words, OPENER_PRIORITY_MAX,
            erd_search.DEFAULT_PRIORITY_STEP)

        self.assertEqual(len(set(ladder.values())), len(candidate_words))
        self.assertGreater(min(ladder.values()), 0)

    def test_top_below_one_full_step_ties_the_tail_on_the_minimum(self):
        ladder = erd_search.priority_ladder(['a', 'b', 'c'], 2, 5)

        self.assertEqual(ladder, {'a': 2, 'b': 0, 'c': 0})


class TestQueueAddPriorityLadder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _requested_priority_by_word(self, queue_path):
        queue = ERDQueue(queue_path)
        self.addCleanup(queue.close)
        return {row['opener']: row['requested_priority']
                for row in queue.opener_work_rows()}

    def _write_words_file(self, words):
        path = os.path.join(self._tmp.name, 'words.txt')
        with open(path, 'w') as words_file:
            words_file.write(''.join(f'{word}\n' for word in words))
        return path

    def test_words_are_queued_on_a_descending_ladder_in_the_given_order(self):
        args = _make_args(self._tmp.name,
                          word=[LARGE_BRANCH_WORD, SECOND_WORD], pattern='-----')

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 5})

    def test_priority_step_sets_the_gap_between_rungs(self):
        args = _make_args(self._tmp.name, priority_step=50,
                          word=[LARGE_BRANCH_WORD, SECOND_WORD], pattern='-----')

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 50})

    def test_zero_step_restores_the_flat_single_priority_batch(self):
        args = _make_args(self._tmp.name, priority_step=0, priority=3,
                          word=[LARGE_BRANCH_WORD, SECOND_WORD], pattern='-----')

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(self._requested_priority_by_word(args.queue),
                         {LARGE_BRANCH_WORD: 3, SECOND_WORD: 3})

    def test_priority_words_are_laddered_and_the_rest_stay_flat_at_zero(self):
        third_word = 'crane'
        words_file = self._write_words_file(
            [LARGE_BRANCH_WORD, SECOND_WORD, third_word])
        args = _make_args(
            self._tmp.name, word=None, words_file=words_file, pattern='-----',
            priority=100, priority_words=[LARGE_BRANCH_WORD, third_word])

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: 105, third_word: 100, SECOND_WORD: 0})

    def test_repeated_word_is_queued_once_at_its_first_position(self):
        args = _make_args(
            self._tmp.name, pattern='-----',
            word=[LARGE_BRANCH_WORD, SECOND_WORD, LARGE_BRANCH_WORD])

        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 5})
        self.assertIn('across 2 word(s)', output.getvalue())

    def test_overflowing_ladder_warns_that_the_tail_starts_together(self):
        # An empty queue tops the append out at 5, which seats only one word
        # above the minimum and clamps the other two onto it.
        args = _make_args(self._tmp.name, pattern='-----', priority_step=5,
                          word=[LARGE_BRANCH_WORD, SECOND_WORD, 'crane'])

        output = StringIO()
        with patch.object(erd_search, 'OPENER_PRIORITY_MAX', 5), \
                redirect_stdout(output):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: 5, SECOND_WORD: 0, 'crane': 0})
        self.assertIn('do not fit on a ladder', output.getvalue())
        self.assertIn('the last 2 share priority 0', output.getvalue())
        self.assertIn('Raise the queued work with queue opener-priority',
                      output.getvalue())

    def test_fitting_ladder_does_not_warn(self):
        args = _make_args(self._tmp.name, pattern='-----',
                          word=[LARGE_BRANCH_WORD, SECOND_WORD])

        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(args)

        self.assertNotIn('do not fit on a ladder', output.getvalue())

    def test_negative_step_is_rejected_before_the_queue_is_created(self):
        args = _make_args(self._tmp.name, priority_step=-1)

        with self.assertRaisesRegex(ValueError, 'must not be negative'):
            erd_search.cmd_queue_add(args)

    def test_out_of_range_priority_is_rejected_before_any_branch_is_queued(self):
        args = _make_args(self._tmp.name, priority=OPENER_PRIORITY_MAX + 1)

        with self.assertRaisesRegex(ValueError, 'opener-work priority'):
            erd_search.cmd_queue_add(args)

        self.assertEqual(ERDQueue(args.queue).total_branches(), 0)

    def test_second_batch_is_appended_below_the_first(self):
        first = _make_args(self._tmp.name, pattern='-----',
                           word=[LARGE_BRANCH_WORD, SECOND_WORD])
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(first)

        second = _make_args(self._tmp.name, pattern='-----',
                            word=['crane', 'irate'],
                            cache=first.cache, queue=first.queue)
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(second)

        priorities = self._requested_priority_by_word(first.queue)
        self.assertEqual(priorities[LARGE_BRANCH_WORD], OPENER_PRIORITY_MAX)
        self.assertEqual(priorities[SECOND_WORD], OPENER_PRIORITY_MAX - 5)
        # The second batch descends from just below the first, so every one
        # of its words ranks under everything already queued.
        self.assertEqual(priorities['crane'], OPENER_PRIORITY_MAX - 6)
        self.assertEqual(priorities['irate'], OPENER_PRIORITY_MAX - 11)
        self.assertLess(max(priorities['crane'], priorities['irate']),
                        min(priorities[LARGE_BRANCH_WORD],
                            priorities[SECOND_WORD]))

    def test_appended_batch_ranks_below_high_priority_queued_work(self):
        first = _make_args(self._tmp.name, pattern='-----', priority=900,
                           word=[LARGE_BRANCH_WORD, SECOND_WORD])
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(first)

        second = _make_args(self._tmp.name, pattern='-----',
                            word=['crane', 'irate'],
                            cache=first.cache, queue=first.queue)
        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(second)

        priorities = self._requested_priority_by_word(first.queue)
        self.assertEqual(priorities[LARGE_BRANCH_WORD], 905)
        self.assertEqual(priorities[SECOND_WORD], 900)
        self.assertEqual(priorities['crane'], 899)
        self.assertEqual(priorities['irate'], 894)
        self.assertIn('behind queued work down to priority 900',
                      output.getvalue())

    def test_explicit_priority_preempts_queued_work_deliberately(self):
        first = _make_args(self._tmp.name, pattern='-----', priority=100,
                           word=[LARGE_BRANCH_WORD])
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(first)

        second = _make_args(self._tmp.name, pattern='-----', priority=500,
                            word=['crane'], cache=first.cache,
                            queue=first.queue)
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(second)

        priorities = self._requested_priority_by_word(first.queue)
        self.assertEqual(priorities[LARGE_BRANCH_WORD], 100)
        self.assertEqual(priorities['crane'], 500)

    def test_completed_work_does_not_hold_the_ceiling_down(self):
        first = _make_args(self._tmp.name, pattern='-----', priority=0,
                           word=[LARGE_BRANCH_WORD])
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(first)

        queue = ERDQueue(first.queue)
        queue._conn.execute("UPDATE opener_work SET state = 'complete'")
        queue._conn.commit()
        queue.close()

        second = _make_args(self._tmp.name, pattern='-----',
                            word=['crane', 'irate'], cache=first.cache,
                            queue=first.queue)
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(second)

        priorities = self._requested_priority_by_word(first.queue)
        self.assertEqual(priorities['crane'], OPENER_PRIORITY_MAX)
        self.assertEqual(priorities['irate'], OPENER_PRIORITY_MAX - 5)

    def test_first_batch_into_an_empty_queue_starts_at_the_ceiling(self):
        args = _make_args(self._tmp.name, pattern='-----',
                          word=[LARGE_BRANCH_WORD, SECOND_WORD])

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        # Nothing is queued yet, so the batch takes the top of the range and
        # leaves the whole space below it for later batches to append into.
        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 5})

    def test_explicit_priority_past_the_range_is_refused_not_clamped(self):
        # Seating the batch lower would hand back a last rung below the one
        # --priority names, and tie it with whatever already sits up there.
        args = _make_args(
            self._tmp.name, pattern='-----', priority=OPENER_PRIORITY_MAX - 5,
            word=[LARGE_BRANCH_WORD, SECOND_WORD, 'crane'])

        with self.assertRaisesRegex(ValueError, 'above the maximum'):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue), {})

    def test_explicit_priority_that_exactly_fits_is_accepted(self):
        args = _make_args(
            self._tmp.name, pattern='-----',
            priority=OPENER_PRIORITY_MAX - 10,
            word=[LARGE_BRANCH_WORD, SECOND_WORD, 'crane'])

        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 5,
             'crane': OPENER_PRIORITY_MAX - 10})

    def test_appending_onto_floor_priority_reports_a_tie_not_an_order(self):
        first = _make_args(self._tmp.name, pattern='-----',
                           priority=OPENER_PRIORITY_MIN,
                           word=[LARGE_BRANCH_WORD])
        with redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(first)

        second = _make_args(self._tmp.name, pattern='-----',
                            word=['crane', 'irate'],
                            cache=first.cache, queue=first.queue)
        output = StringIO()
        with redirect_stdout(output):
            erd_search.cmd_queue_add(second)

        text = output.getvalue()
        self.assertIn('TIED WITH queued work at priority 0', text)
        self.assertNotIn('behind queued work', text)
        # And the advice names a remedy that can actually work at the floor.
        self.assertIn('Raise the queued work with queue opener-priority', text)
        self.assertNotIn('smaller --priority-step', text)

    def test_ladder_top_priority_is_a_pure_function_of_its_arguments(self):
        # Taking lowest_queued as a value rather than re-querying keeps the
        # figure reported to the user and the one the ladder uses identical.
        self.assertEqual(
            erd_search.ladder_top_priority(None, None, 5, 3),
            OPENER_PRIORITY_MAX)
        self.assertEqual(erd_search.ladder_top_priority(900, None, 5, 3), 899)
        self.assertEqual(erd_search.ladder_top_priority(0, None, 5, 3),
                         OPENER_PRIORITY_MIN)
        self.assertEqual(erd_search.ladder_top_priority(900, 100, 5, 3), 110)

    def test_queue_add_queries_the_lowest_priority_once(self):
        args = _make_args(self._tmp.name, pattern='-----',
                          word=[LARGE_BRANCH_WORD, SECOND_WORD])
        calls = []
        original = ERDQueue.lowest_unfinished_opener_priority

        def counting(self):
            calls.append(1)
            return original(self)

        with patch.object(ERDQueue, 'lowest_unfinished_opener_priority',
                          counting), redirect_stdout(StringIO()):
            erd_search.cmd_queue_add(args)

        self.assertEqual(len(calls), 1)

    def test_cli_default_step_ladders_words_given_to_the_word_flag(self):
        args = _make_args(self._tmp.name)

        with patch.object(erd_search.sys, 'argv', [
                'erd_search.py', 'queue', 'add', '--word',
                LARGE_BRANCH_WORD, SECOND_WORD, '--pattern=-----',
                '--cache', args.cache, '--queue', args.queue]):
            erd_search.main()

        self.assertEqual(
            self._requested_priority_by_word(args.queue),
            {LARGE_BRANCH_WORD: OPENER_PRIORITY_MAX,
             SECOND_WORD: OPENER_PRIORITY_MAX - 5})


if __name__ == '__main__':
    unittest.main()
