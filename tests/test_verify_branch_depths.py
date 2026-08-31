"""Tests for verify_branch_depths.py — the max_depth fold audit.

The fixture is a real depth-limited solve over a sampled vocabulary, so the
rows carry the max_depth and solve_budget values the solver itself writes.
Two properties anchor everything else: a cache the solver just built agrees
with the fold on every row, and a cache where one branch's value has been
replaced — the tainted/untainted overwrite of issue #302 — disagrees at every
ancestor that folded the value it replaced.
"""
import io
import json
import os
import random
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import verify_branch_depths
from cache_sqlite import ScoreCache, branch_reference
from erd_queue import decode_subset
from runtime_paths import DEFAULT_ANSWER_LIST_PATH
from verify_branch_depths import DepthAudit, fold_branch, iter_rows
from wordle_engine import (
    ERD_ALL,
    ResponseCache,
    load_word_list,
    min_expected_guesses,
)

# 200 answers guessed from the answer list alone, solved under a budget of 4:
# small enough to rebuild in a fraction of a second, deep enough that the
# deepest branch sits three stored levels below the root, and tight enough
# that the budget floor fires on some branches (tainted rows).
FIXTURE_ANSWER_COUNT = 200
FIXTURE_BUDGET = 4
FIXTURE_SEED = 11


class _StubResponses:
    """Groups a guess's response partition straight from a supplied mapping."""

    def __init__(self, groups):
        self._groups = groups

    def group_words(self, guess, branch_words):
        return self._groups


class FoldBranchTest(unittest.TestCase):
    """fold_branch reproduces evaluate_candidate's max_depth recurrence."""

    def test_self_group_is_free_and_a_singleton_costs_one_more_guess(self):
        # CRANE is the guess and one of the branch words: its own group is
        # finished by playing it, while the other single word needs one more.
        responses = _StubResponses({0: ['crane'], 1: ['sound']})
        fold = fold_branch(['crane', 'sound'], 'crane', responses,
                           lambda key: self.fail('no group needs a lookup'))
        self.assertEqual(fold.depth, 2)
        self.assertEqual(fold.erd, 1.5)

    def test_a_stored_group_costs_one_more_than_its_own_worst_case(self):
        group = ['sound', 'spend', 'stand']
        responses = _StubResponses({0: ['crane'], 1: group})
        fold = fold_branch(['crane'] + group, 'crane', responses,
                           lambda key: (3, 2.0))
        self.assertEqual(fold.depth, 4)
        self.assertAlmostEqual(fold.erd, 1.0 + 0.75 * 2.0)

    def test_a_group_with_no_stored_row_leaves_the_fold_incomplete(self):
        group = ['sound', 'spend']
        responses = _StubResponses({0: ['crane'], 1: group})
        fold = fold_branch(['crane'] + group, 'crane', responses, lambda key: None)
        self.assertFalse(fold.complete)
        self.assertEqual(fold.missing, (ScoreCache.encode_subset(group),))

    def test_a_row_whose_max_depth_is_missing_reads_as_incomplete(self):
        group = ['sound', 'spend']
        responses = _StubResponses({0: ['crane'], 1: group})
        fold = fold_branch(['crane'] + group, 'crane', responses,
                           lambda key: (None, 2.0))
        self.assertFalse(fold.complete)

    def test_a_guess_that_separates_nothing_is_degenerate(self):
        branch = ['sound', 'spend']
        responses = _StubResponses({0: branch})
        fold = fold_branch(branch, 'crane', responses,
                           lambda key: self.fail('degenerate rows are not folded'))
        self.assertTrue(fold.degenerate)
        self.assertFalse(fold.complete)


class _CacheFixture(unittest.TestCase):
    """A cache the solver built, restored fresh for each test."""

    @classmethod
    def setUpClass(cls):
        random_generator = random.Random(FIXTURE_SEED)
        cls.answer_words = sorted(random_generator.sample(
            load_word_list(DEFAULT_ANSWER_LIST_PATH), FIXTURE_ANSWER_COUNT))
        cls._master_dir = tempfile.mkdtemp()
        cls.answers_path = os.path.join(cls._master_dir, 'answers.txt')
        with open(cls.answers_path, 'w') as handle:
            handle.write('\n'.join(cls.answer_words) + '\n')
        cls._master_cache = os.path.join(cls._master_dir, 'master.sqlite3')
        score_cache = ScoreCache(cls._master_cache, cls.answer_words,
                                 checkpoint_on_close=False)
        responses = ResponseCache(cls.answer_words, score_cache=score_cache)
        min_expected_guesses(cls.answer_words, responses, score_cache,
                             guesses=cls.answer_words, policy=ERD_ALL,
                             budget=FIXTURE_BUDGET)
        score_cache.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._master_dir, ignore_errors=True)

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        self.cache_path = os.path.join(self._dir, 'cache.sqlite3')
        shutil.copy(self._master_cache, self.cache_path)

    def open_cache(self):
        score_cache = ScoreCache(self.cache_path, self.answer_words,
                                 checkpoint_on_close=False)
        self.addCleanup(score_cache.close)
        return score_cache

    def audit(self, repair=False, list_limit=0):
        score_cache = ScoreCache(self.cache_path, self.answer_words,
                                 checkpoint_on_close=False)
        try:
            responses = ResponseCache(self.answer_words, score_cache=score_cache)
            audit = DepthAudit(score_cache, ERD_ALL, responses, repair=repair)
            audit.run(iter_rows(score_cache, ERD_ALL), list_limit=list_limit)
        finally:
            score_cache.close()
        return audit

    def row_for(self, branch_key):
        score_cache = self.open_cache()
        return score_cache._conn.execute(
            "SELECT best_guess, best_score, max_depth, solve_budget "
            "FROM branch_best_by_policy "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
            (branch_key, ERD_ALL, score_cache.answer_list_id)).fetchone()

    def deepest_chain(self):
        """Branch keys from the root down the groups that set its max_depth.

        Following the group that achieves `1 + child_depth == parent_depth` is
        what makes a change to the last link propagate all the way back up.
        """
        score_cache = self.open_cache()
        responses = ResponseCache(self.answer_words, score_cache=score_cache)
        depths = {bytes(row['branch_key']): row['max_depth']
                  for row in iter_rows(score_cache, ERD_ALL)}
        chain = [ScoreCache.encode_subset(self.answer_words)]
        while True:
            branch_key = chain[-1]
            row = self.row_for(branch_key)
            branch_words = decode_subset(branch_key)
            groups = responses.group_words(row['best_guess'], branch_words)
            for group in groups.values():
                if len(group) < 2:
                    continue
                group_key = ScoreCache.encode_subset(group)
                if depths.get(group_key) == row['max_depth'] - 1:
                    chain.append(group_key)
                    break
            else:
                return chain


class CleanRebuildTest(_CacheFixture):

    def test_the_solver_s_own_rows_agree_with_the_fold_on_every_one(self):
        audit = self.audit()
        self.assertGreater(audit.checked, 20)
        self.assertEqual(audit.incomplete, 0)
        self.assertEqual(audit.degenerate, 0)
        self.assertEqual(audit.legacy, 0)
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_too_high, 0)
        self.assertEqual(audit.score_stale, 0)

    def test_the_fixture_reaches_the_states_the_audit_is_about(self):
        # A fold over rows that were all shallow, or none of them tainted,
        # would assert nothing about either.
        score_cache = self.open_cache()
        rows = score_cache._conn.execute(
            "SELECT MAX(max_depth), SUM(solve_budget IS NOT NULL) "
            "FROM branch_best_by_policy WHERE policy = ? AND answer_list_id = ?",
            (ERD_ALL, score_cache.answer_list_id)).fetchone()
        self.assertGreaterEqual(rows[0], 4)
        self.assertGreater(rows[1], 0)
        self.assertGreaterEqual(len(self.deepest_chain()), 3)

    def test_a_legacy_row_is_counted_rather_than_contradicted(self):
        chain = self.deepest_chain()
        score_cache = self.open_cache()
        score_cache._conn.execute(
            "UPDATE branch_best_by_policy SET max_depth = NULL "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
            (chain[-1], ERD_ALL, score_cache.answer_list_id))
        audit = self.audit()
        self.assertEqual(audit.legacy, 1)
        # Its parent folds it as an unresolved group rather than as depth 0.
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_too_high, 0)
        self.assertGreaterEqual(audit.incomplete, 1)
        self.assertGreaterEqual(audit.unresolved_groups, audit.incomplete)

    def test_the_erd_fold_reproduces_each_stored_score_bit_for_bit(self):
        # Summing the response groups in evaluate_candidate's own order is
        # what makes this exact rather than merely close; a tolerance would
        # hide a genuinely stale score that happens to be near enough.
        score_cache = self.open_cache()
        responses = ResponseCache(self.answer_words, score_cache=score_cache)
        stored = {}
        for row in iter_rows(score_cache, ERD_ALL):
            branch_key = bytes(row['branch_key'])
            fold = fold_branch(decode_subset(branch_key), row['best_guess'],
                               responses, stored.get)
            self.assertTrue(fold.complete)
            self.assertEqual(fold.erd, row['best_score'])
            stored[branch_key] = (row['max_depth'], row['best_score'])
        self.assertGreater(len(stored), 20)


class AliasedOverwriteTest(_CacheFixture):
    """The overwrite of issue #302, and what the audit makes of it."""

    def restate(self, branch_key, max_depth):
        """Rewrite one branch's recorded depth, keeping its strategy.

        A branch whose row was replaced by the other of its two competing
        values leaves exactly this behind at every ancestor that folded the
        one it replaced: the same best_guess and best_score it always had,
        against a depth that no longer describes the subtree beneath it.
        """
        row = self.row_for(branch_key)
        score_cache = self.open_cache()
        score_cache.write(branch_key, ERD_ALL, row['best_guess'], row['best_score'],
                          max_depth=max_depth, solve_budget=FIXTURE_BUDGET)
        return row['max_depth']

    def understate_chain(self):
        """Understate every branch on the deepest chain by one guess.

        The deepest branch's own strategy still needs the depth it always
        needed, so correcting it re-opens the gap at its parent, and so on to
        the root — the ancestry the issue's global fold cannot count because
        it reads back the same understated children.
        """
        chain = self.deepest_chain()
        true_depths = [self.restate(key, self.row_for(key)['max_depth'] - 1)
                       for key in chain]
        return chain, true_depths

    def naive_flagged(self):
        """What a fold that reads each child's stored depth would flag."""
        score_cache = self.open_cache()
        responses = ResponseCache(self.answer_words, score_cache=score_cache)
        rows = [(bytes(row['branch_key']), row['best_guess'], row['max_depth'])
                for row in iter_rows(score_cache, ERD_ALL)]
        stored = {key: (max_depth, 0.0) for key, _guess, max_depth in rows}
        flagged = set()
        for branch_key, best_guess, max_depth in rows:
            fold = fold_branch(decode_subset(branch_key), best_guess, responses,
                               stored.get)
            if fold.complete and fold.depth != max_depth:
                flagged.add(branch_reference(branch_key))
        return flagged

    def test_an_understated_branch_is_reported_against_its_own_subtree(self):
        chain = self.deepest_chain()
        parent = chain[-2]
        true_depth = self.restate(parent, self.row_for(parent)['max_depth'] - 1)

        audit = self.audit(list_limit=len(chain))
        flagged = {finding['branch_reference']: finding for finding in audit.findings}
        self.assertIn(branch_reference(parent), flagged)
        self.assertEqual(flagged[branch_reference(parent)]['stored_max_depth'],
                         true_depth - 1)
        self.assertEqual(flagged[branch_reference(parent)]['folded_max_depth'],
                         true_depth)
        self.assertEqual(audit.depth_deltas[(true_depth - 1, true_depth)], 1)
        self.assertEqual(audit.tainted_split['tainted'], 1)
        self.assertEqual(audit.mismatch_sizes[len(decode_subset(parent))], 1)

    def test_every_ancestor_of_an_understated_branch_is_flagged(self):
        chain, true_depths = self.understate_chain()
        self.assertGreaterEqual(len(chain), 3)

        audit = self.audit(list_limit=len(chain))
        flagged = {finding['branch_reference'] for finding in audit.findings}
        for branch_key in chain:
            self.assertIn(branch_reference(branch_key), flagged,
                          f'{branch_reference(branch_key)} not flagged')
        self.assertEqual(audit.depth_too_low, len(chain))
        self.assertEqual(audit.depth_too_high, 0)

    def test_reading_stored_children_misses_part_of_the_ancestry(self):
        # The bottom-up pass is what finds the whole ancestry.  A fold that
        # re-reads each child's stored depth agrees with a parent that folded
        # the same understated value and moves on, so its count is a floor on
        # the real one rather than a measurement of it.
        chain, _true_depths = self.understate_chain()
        audit = self.audit(list_limit=len(chain))
        flagged = {finding['branch_reference'] for finding in audit.findings}
        naive = self.naive_flagged()
        self.assertIn(branch_reference(chain[-2]), flagged)
        self.assertNotIn(branch_reference(chain[-2]), naive)
        self.assertLess(naive, flagged)

    def test_an_overstated_branch_reads_as_conservative(self):
        chain = self.deepest_chain()
        parent = chain[-2]
        true_depth = self.restate(parent, self.row_for(parent)['max_depth'] + 1)
        audit = self.audit()
        self.assertEqual(audit.depth_too_high, 1)
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_deltas[(true_depth + 1, true_depth)], 1)

    def test_a_replaced_score_is_reported_but_never_rewritten(self):
        chain = self.deepest_chain()
        child = chain[-1]
        row = self.row_for(child)
        score_cache = self.open_cache()
        score_cache.write(child, ERD_ALL, row['best_guess'], row['best_score'] + 0.5,
                          max_depth=row['max_depth'], solve_budget=row['solve_budget'])
        audit = self.audit(repair=True)
        self.assertGreaterEqual(audit.score_stale, 1)
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_too_high, 0)
        self.assertAlmostEqual(self.row_for(child)['best_score'],
                               row['best_score'] + 0.5)


class RepairTest(_CacheFixture):

    def understate(self, branch_key):
        """Record one branch a guess shallower than its own strategy needs."""
        row = self.row_for(branch_key)
        score_cache = self.open_cache()
        score_cache.write(branch_key, ERD_ALL, row['best_guess'], row['best_score'],
                          max_depth=row['max_depth'] - 1, solve_budget=FIXTURE_BUDGET)
        return row

    def test_repair_rewrites_the_depth_and_leaves_the_strategy_alone(self):
        parent = self.deepest_chain()[-2]
        before = self.understate(parent)

        audit = self.audit(repair=True)
        self.assertEqual(audit.repaired, audit.depth_too_low + audit.depth_too_high)
        after = self.row_for(parent)
        self.assertEqual(after['max_depth'], before['max_depth'])
        self.assertEqual(after['best_guess'], before['best_guess'])
        self.assertEqual(after['best_score'], before['best_score'])
        # The overwrite's own taint marker stands: repair touches only depth.
        self.assertEqual(after['solve_budget'], FIXTURE_BUDGET)

    def test_a_repaired_cache_audits_clean_on_the_next_run(self):
        chain = self.deepest_chain()
        for branch_key in chain:
            self.understate(branch_key)
        self.assertEqual(self.audit(repair=True).repaired, len(chain))

        second = self.audit()
        self.assertEqual(second.depth_too_low, 0)
        self.assertEqual(second.depth_too_high, 0)
        self.assertEqual(second.repaired, 0)

    def test_an_audit_only_run_rewrites_nothing(self):
        parent = self.deepest_chain()[-2]
        before = self.understate(parent)
        audit = self.audit()
        self.assertGreaterEqual(audit.depth_too_low, 1)
        self.assertEqual(audit.repaired, 0)
        self.assertEqual(self.row_for(parent)['max_depth'], before['max_depth'] - 1)

    def test_repair_max_depth_reports_a_row_it_did_not_find(self):
        score_cache = self.open_cache()
        self.assertFalse(score_cache.repair_max_depth(
            ScoreCache.encode_subset(['aaaaa', 'bbbbb']), ERD_ALL, 3))

    def test_repair_max_depth_drops_the_row_from_the_session_mirror(self):
        branch_key = self.deepest_chain()[-1]
        score_cache = self.open_cache()
        original = score_cache.read_with_depth(branch_key, ERD_ALL)
        self.assertTrue(score_cache.repair_max_depth(
            branch_key, ERD_ALL, original[2] + 1))
        self.assertEqual(score_cache.read_with_depth(branch_key, ERD_ALL)[2],
                         original[2] + 1)


class CommandLineTest(_CacheFixture):

    def run_main(self, *args):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = verify_branch_depths.main(
                ['--cache', self.cache_path, '--answers', self.answers_path, *args])
        return status, stdout.getvalue()

    def test_a_clean_cache_reports_zero_and_exits_zero(self):
        status, output = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn('stored max_depth TOO LOW  (unsound reuse): 0', output)

    def understate(self, branch_key):
        row = self.row_for(branch_key)
        score_cache = self.open_cache()
        score_cache.write(branch_key, ERD_ALL, row['best_guess'], row['best_score'],
                          max_depth=row['max_depth'] - 1, solve_budget=FIXTURE_BUDGET)

    def test_an_understated_row_exits_nonzero_and_is_listed(self):
        parent = self.deepest_chain()[-2]
        self.understate(parent)
        status, output = self.run_main('--list', '10')
        self.assertEqual(status, 1)
        self.assertIn(branch_reference(parent), output)
        self.assertIn('deltas:', output)

    def test_repair_exits_zero_and_reports_what_it_rewrote(self):
        self.understate(self.deepest_chain()[-2])
        status, output = self.run_main('--repair')
        self.assertEqual(status, 0)
        self.assertIn('max_depth rows repaired:', output)
        self.assertEqual(self.run_main()[0], 0)

    def test_json_output_carries_the_whole_summary(self):
        status, output = self.run_main('--json')
        summary = json.loads(output)
        self.assertEqual(status, 0)
        self.assertEqual(summary['depth_too_low'], 0)
        self.assertGreater(summary['checked'], 20)
        self.assertIn('elapsed_seconds', summary)

    def test_thousands_separators_reach_the_report(self):
        report = verify_branch_depths.render_report(
            {'checked': 739662, 'legacy': 0, 'incomplete': 0, 'degenerate': 4,
             'unresolved_groups': 0,
             'depth_too_low': 1070, 'depth_too_high': 36, 'score_stale': 2500,
             'repaired': 1106,
             'depth_deltas': {'3 -> 4': 1065, '4 -> 5': 5},
             'tainted_split': {'tainted': 1070},
             'mismatch_sizes': {16: 500, 25: 570},
             'findings': []},
            30.0, repair=True)
        self.assertIn('checked 739,662', report)
        self.assertIn('3 -> 4: 1,065', report)
        self.assertIn('unsound reuse): 1,070', report)
        self.assertIn('n=16-25', report)
        self.assertIn('separates nothing:   4', report)


if __name__ == '__main__':
    unittest.main()
