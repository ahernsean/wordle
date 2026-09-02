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
import sqlite3
import os
import random
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

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
                                 checkpoint_on_close=False, read_only=not repair)
        try:
            responses = ResponseCache(
                self.answer_words,
                score_cache=verify_branch_depths._ReadOnlyDecompositions(score_cache))
            audit = DepthAudit(score_cache, ERD_ALL, responses, repair=repair)
            audit.run(iter_rows(score_cache, ERD_ALL), list_limit=list_limit)
        finally:
            score_cache.close()
        return audit

    def row_for(self, branch_key, solve_budget=None):
        """The stored result for one branch at one scope.

        None is the unrestricted result; an int is the one solved under that
        cap.  They live in different tables and are different facts.
        """
        score_cache = self.open_cache()
        if solve_budget is None:
            return score_cache._conn.execute(
                "SELECT best_guess, best_score, max_depth, solve_budget, updated_at "
                "FROM branch_best_by_policy "
                "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
                (branch_key, ERD_ALL, score_cache.answer_list_id)).fetchone()
        return score_cache._conn.execute(
            "SELECT best_guess, best_score, max_depth, solve_budget, updated_at "
            "FROM branch_best_by_policy_and_budget "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ? "
            "  AND solve_budget = ?",
            (branch_key, ERD_ALL, score_cache.answer_list_id,
             solve_budget)).fetchone()

    def deepest_chain(self):
        """Facts from the root down the groups that set its max_depth.

        Each element is (branch_key, solve_budget): a branch can hold an
        unrestricted result and one per budget, and only the fact a parent
        actually folded is on its chain.  Descending picks the group achieving
        `1 + child_depth == parent_depth` in the scope that parent read — one
        budget down, unrestricted first — so a change to the last link
        propagates all the way back up.
        """
        score_cache = self.open_cache()
        responses = ResponseCache(self.answer_words, score_cache=score_cache)
        depths = {(bytes(row['branch_key']), row['solve_budget']): row['max_depth']
                  for row in iter_rows(score_cache, ERD_ALL)}
        root = ScoreCache.encode_subset(self.answer_words)
        root_scope = None if (root, None) in depths else FIXTURE_BUDGET
        chain = [(root, root_scope)]
        while True:
            branch_key, scope = chain[-1]
            row = self.row_for(branch_key, scope)
            child_budget = None if scope is None else scope - 1
            groups = responses.group_words(row['best_guess'],
                                           decode_subset(branch_key))
            for group in groups.values():
                if len(group) < 2:
                    continue
                group_key = ScoreCache.encode_subset(group)
                fact = self._child_fact(depths, group_key, child_budget)
                if fact is not None and depths[fact] == row['max_depth'] - 1:
                    chain.append(fact)
                    break
            else:
                return chain

    @staticmethod
    def _child_fact(depths, group_key, child_budget):
        """Which of a group's facts a parent at `child_budget` would read."""
        canonical_depth = depths.get((group_key, None))
        if child_budget is None:
            return (group_key, None) if canonical_depth is not None else None
        if canonical_depth is not None and canonical_depth <= child_budget:
            return (group_key, None)
        if (group_key, child_budget) in depths:
            return (group_key, child_budget)
        return None

    def restate(self, fact, max_depth=None, best_score=None):
        """Replace one stored fact's recorded values, keeping its strategy.

        Written with SQL rather than ScoreCache.write, which now refuses a
        same-scope rewrite: this is reproducing the state a cache was left in
        *before* that refusal existed, which is exactly what the audit is for.
        """
        branch_key, scope = fact
        row = self.row_for(branch_key, scope)
        table = ('branch_best_by_policy' if scope is None
                 else 'branch_best_by_policy_and_budget')
        score_cache = self.open_cache()
        score_cache._conn.execute(
            f"UPDATE {table} SET max_depth = ?, best_score = ? "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
            (row['max_depth'] if max_depth is None else max_depth,
             row['best_score'] if best_score is None else best_score,
             branch_key, ERD_ALL, score_cache.answer_list_id))
        return row

    def understate(self, fact):
        """Record one fact a guess shallower than its own strategy needs."""
        return self.restate(fact, max_depth=self.row_for(*fact)['max_depth'] - 1)

    def overstate(self, fact):
        return self.restate(fact, max_depth=self.row_for(*fact)['max_depth'] + 1)


def _flagged(audit):
    """The (branch_reference, solve_budget) facts an audit reported."""
    return {(finding['branch_reference'], finding['solve_budget'])
            for finding in audit.findings}


def _fact_reference(fact):
    return (branch_reference(fact[0]), fact[1])


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
        # A fold over rows that were all shallow, or all of one scope, would
        # assert nothing about either.
        score_cache = self.open_cache()
        canonical = score_cache._conn.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy "
            "WHERE policy = ? AND answer_list_id = ?",
            (ERD_ALL, score_cache.answer_list_id)).fetchone()[0]
        budgeted = score_cache._conn.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy_and_budget "
            "WHERE policy = ? AND answer_list_id = ?",
            (ERD_ALL, score_cache.answer_list_id)).fetchone()[0]
        self.assertGreater(canonical, 0)
        self.assertGreater(budgeted, 0)
        chain = self.deepest_chain()
        self.assertGreaterEqual(len(chain), 3)
        self.assertGreaterEqual(self.row_for(*chain[0])['max_depth'], 4)

    def test_no_budget_specific_result_reaches_the_canonical_table(self):
        score_cache = self.open_cache()
        self.assertEqual(score_cache._conn.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy "
            "WHERE solve_budget IS NOT NULL").fetchone()[0], 0)

    def test_a_legacy_row_is_counted_rather_than_contradicted(self):
        branch_key, scope = self.deepest_chain()[-1]
        table = ('branch_best_by_policy' if scope is None
                 else 'branch_best_by_policy_and_budget')
        score_cache = self.open_cache()
        score_cache._conn.execute(
            f"UPDATE {table} SET max_depth = NULL "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
            (branch_key, ERD_ALL, score_cache.answer_list_id))
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
        known = {}
        checked = 0
        for row in iter_rows(score_cache, ERD_ALL):
            branch_key = bytes(row['branch_key'])
            scope = row['solve_budget']
            fold = fold_branch(
                decode_subset(branch_key), row['best_guess'], responses,
                _known_lookup(known, scope))
            self.assertTrue(fold.complete)
            self.assertEqual(fold.erd, row['best_score'])
            known[(branch_key, scope)] = (row['max_depth'], row['best_score'])
            checked += 1
        self.assertGreater(checked, 20)


def _known_lookup(known, parent_scope):
    """A fold lookup over a plain dict, scoped as DepthAudit scopes its own."""
    if parent_scope is None:
        return lambda key: known.get((key, None))
    child_budget = parent_scope - 1

    def lookup(key):
        canonical = known.get((key, None))
        if (canonical is not None and canonical[0] is not None
                and canonical[0] <= child_budget):
            return canonical
        return known.get((key, child_budget))

    return lookup


class AliasedOverwriteTest(_CacheFixture):
    """The overwrite of issue #302, and what the audit makes of it."""

    def understate_chain(self):
        """Understate every fact on the deepest chain by one guess.

        The deepest one's own strategy still needs the depth it always needed,
        so correcting it re-opens the gap at its parent, and so on to the root
        — the ancestry the issue's global fold cannot count because it reads
        back the same understated children.
        """
        chain = self.deepest_chain()
        for fact in chain:
            self.understate(fact)
        return chain

    def naive_flagged(self):
        """What a fold that reads each child's stored depth would flag."""
        score_cache = self.open_cache()
        responses = ResponseCache(self.answer_words, score_cache=score_cache)
        rows = [(bytes(row['branch_key']), row['solve_budget'],
                 row['best_guess'], row['max_depth'])
                for row in iter_rows(score_cache, ERD_ALL)]
        stored = {(key, scope): (max_depth, 0.0)
                  for key, scope, _guess, max_depth in rows}
        flagged = set()
        for branch_key, scope, best_guess, max_depth in rows:
            fold = fold_branch(decode_subset(branch_key), best_guess, responses,
                               _known_lookup(stored, scope))
            if fold.complete and fold.depth != max_depth:
                flagged.add((branch_reference(branch_key), scope))
        return flagged

    def test_an_understated_fact_is_reported_against_its_own_subtree(self):
        parent = self.deepest_chain()[-2]
        true_depth = self.row_for(*parent)['max_depth']
        self.understate(parent)

        audit = self.audit(list_limit=10)
        flagged = {(f['branch_reference'], f['solve_budget']): f
                   for f in audit.findings}
        self.assertIn(_fact_reference(parent), flagged)
        finding = flagged[_fact_reference(parent)]
        self.assertEqual(finding['stored_max_depth'], true_depth - 1)
        self.assertEqual(finding['folded_max_depth'], true_depth)
        self.assertEqual(audit.depth_deltas[(true_depth - 1, true_depth)], 1)
        self.assertEqual(audit.mismatch_sizes[len(decode_subset(parent[0]))], 1)

    def test_every_ancestor_of_an_understated_fact_is_flagged(self):
        chain = self.understate_chain()
        self.assertGreaterEqual(len(chain), 3)

        audit = self.audit(list_limit=len(chain) + 5)
        flagged = _flagged(audit)
        for fact in chain:
            self.assertIn(_fact_reference(fact), flagged,
                          f'{_fact_reference(fact)} not flagged')
        self.assertEqual(audit.depth_too_low, len(chain))
        self.assertEqual(audit.depth_too_high, 0)

    def test_reading_stored_children_misses_part_of_the_ancestry(self):
        # The bottom-up pass is what finds the whole ancestry.  A fold that
        # re-reads each child's stored depth agrees with a parent that folded
        # the same understated value and moves on, so its count is a floor on
        # the real one rather than a measurement of it.
        chain = self.understate_chain()
        audit = self.audit(list_limit=len(chain) + 5)
        flagged = _flagged(audit)
        naive = self.naive_flagged()
        self.assertIn(_fact_reference(chain[-2]), flagged)
        self.assertNotIn(_fact_reference(chain[-2]), naive)
        self.assertLess(naive, flagged)

    def test_an_overstated_fact_reads_as_conservative(self):
        parent = self.deepest_chain()[-2]
        true_depth = self.row_for(*parent)['max_depth']
        self.overstate(parent)
        audit = self.audit()
        self.assertEqual(audit.depth_too_high, 1)
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_deltas[(true_depth + 1, true_depth)], 1)

    def test_a_replaced_score_is_reported_but_never_rewritten(self):
        child = self.deepest_chain()[-1]
        row = self.restate(child, best_score=self.row_for(*child)['best_score'] + 0.5)
        audit = self.audit(repair=True)
        self.assertGreaterEqual(audit.score_stale, 1)
        self.assertEqual(audit.depth_too_low, 0)
        self.assertEqual(audit.depth_too_high, 0)
        self.assertAlmostEqual(self.row_for(*child)['best_score'],
                               row['best_score'] + 0.5)


class RepairTest(_CacheFixture):

    def test_repair_rewrites_the_depth_and_leaves_the_strategy_alone(self):
        parent = self.deepest_chain()[-2]
        before = self.understate(parent)

        audit = self.audit(repair=True)
        self.assertEqual(audit.repaired, audit.depth_too_low + audit.depth_too_high)
        after = self.row_for(*parent)
        self.assertEqual(after['max_depth'], before['max_depth'])
        self.assertEqual(after['best_guess'], before['best_guess'])
        self.assertEqual(after['best_score'], before['best_score'])
        self.assertEqual(after['solve_budget'], before['solve_budget'])

    def test_a_repair_reaches_a_budget_specific_result(self):
        # The two scopes live in different tables, so a repair that only knew
        # the canonical one would silently no-op on half the cache.
        budgeted = [fact for fact in self.deepest_chain() if fact[1] is not None]
        self.assertTrue(budgeted, 'fixture has no budget-specific facts')
        fact = budgeted[-1]
        before = self.understate(fact)
        self.assertGreater(self.audit(repair=True).repaired, 0)
        self.assertEqual(self.row_for(*fact)['max_depth'], before['max_depth'])

    def test_a_repaired_cache_audits_clean_on_the_next_run(self):
        chain = self.deepest_chain()
        for fact in chain:
            self.understate(fact)
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
        self.assertEqual(self.row_for(*parent)['max_depth'],
                         before['max_depth'] - 1)

    def test_repair_max_depth_reports_a_row_it_did_not_find(self):
        score_cache = self.open_cache()
        self.assertFalse(score_cache.repair_max_depth(
            ScoreCache.encode_subset(['aaaaa', 'bbbbb']), ERD_ALL, 3))

    def test_repair_max_depth_drops_the_row_from_the_session_mirror(self):
        branch_key, scope = self.deepest_chain()[-1]
        score_cache = self.open_cache()
        original = score_cache.read_for_budget(branch_key, ERD_ALL, scope)
        self.assertTrue(score_cache.repair_max_depth(
            branch_key, ERD_ALL, original[2] + 1, solve_budget=scope))
        self.assertEqual(
            score_cache.read_for_budget(branch_key, ERD_ALL, scope)[2],
            original[2] + 1)


class RepairSafetyTest(_CacheFixture):
    """Which direction of disagreement --repair is willing to act on."""

    def test_an_overstated_depth_is_lowered_when_the_score_holds_up(self):
        parent = self.deepest_chain()[-2]
        before = self.overstate(parent)
        audit = self.audit(repair=True)
        self.assertEqual(audit.repair_withheld, 0)
        self.assertEqual(audit.repaired, 1)
        self.assertEqual(self.row_for(*parent)['max_depth'], before['max_depth'])

    def test_an_overstated_depth_is_left_alone_when_the_score_is_stale(self):
        # Lowering it would offer the row at budgets that previously rejected
        # it, on the strength of an ERD this same pass just contradicted.
        parent = self.deepest_chain()[-2]
        before = self.row_for(*parent)
        self.restate(parent, max_depth=before['max_depth'] + 1,
                     best_score=before['best_score'] + 0.5)
        audit = self.audit(repair=True)
        # The replaced score also disagrees at whoever folded it, so the stale
        # count is the parent plus its own ancestry.
        self.assertGreaterEqual(audit.score_stale, 1)
        self.assertEqual(audit.depth_too_high, 1)
        self.assertEqual(audit.repaired, 0)
        self.assertEqual(audit.repair_withheld, 1)
        self.assertEqual(self.row_for(*parent)['max_depth'],
                         before['max_depth'] + 1)

    def test_an_understated_depth_is_raised_even_when_the_score_is_stale(self):
        # Raising only withdraws reuse, so a stale score is no reason to wait.
        parent = self.deepest_chain()[-2]
        before = self.row_for(*parent)
        self.restate(parent, max_depth=before['max_depth'] - 1,
                     best_score=before['best_score'] + 0.5)
        audit = self.audit(repair=True)
        self.assertGreaterEqual(audit.score_stale, 1)
        self.assertEqual(audit.depth_too_low, 1)
        self.assertEqual(audit.repaired, 1)
        self.assertEqual(audit.repair_withheld, 0)
        self.assertEqual(self.row_for(*parent)['max_depth'], before['max_depth'])


class RepairSideEffectsTest(_CacheFixture):
    """What a repair has to do beyond the column it corrects."""

    def updated_at_of(self, fact):
        return self.row_for(*fact)['updated_at']

    def test_a_repair_moves_updated_at_so_an_incremental_export_carries_it(self):
        # export_cache --since selects updated_at > ?, so a repair that left
        # the timestamp alone would be dropped from every incremental export.
        parent = self.deepest_chain()[-2]
        self.understate(parent)
        table = ('branch_best_by_policy' if parent[1] is None
                 else 'branch_best_by_policy_and_budget')
        score_cache = self.open_cache()
        score_cache._conn.execute(
            f"UPDATE {table} SET updated_at = 0 "
            "WHERE branch_key = ? AND policy = ? AND answer_list_id = ?",
            (parent[0], ERD_ALL, score_cache.answer_list_id))
        self.assertEqual(self.updated_at_of(parent), 0)

        self.assertEqual(self.audit(repair=True).repaired, 1)
        self.assertGreater(self.updated_at_of(parent), 0)

    def test_a_repair_touches_only_the_row_it_repairs(self):
        """A repair has nothing to invalidate above it.

        A candidate's own ERD is folded from its response groups' rows on
        every read, so the repaired depth reaches the next report without an
        invalidation step.  What the pass must not do is write anything beyond
        the branch results it is auditing.
        """
        parent = self.deepest_chain()[-2]
        self.understate(parent)
        before = self.table_names()

        audit = self.audit(repair=True)

        self.assertEqual(audit.repaired, 1)
        self.assertEqual(self.table_names(), before)
        self.assertNotIn('candidate_erd_by_policy', before)

    def table_names(self):
        return {row[0] for row in self.open_cache()._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}


class ReadOnlyAuditTest(_CacheFixture):
    """An audit-only run must leave the cache it inspects untouched."""

    def cache_bytes(self):
        with open(self.cache_path, 'rb') as handle:
            return handle.read()

    def decomposition_count(self):
        score_cache = self.open_cache()
        return score_cache._conn.execute(
            "SELECT COUNT(*) FROM response_decomposition").fetchone()[0]

    def test_a_missing_cache_is_an_error_not_a_clean_audit(self):
        # A mistyped path used to be created empty and certified as clean.
        missing = os.path.join(self._dir, 'nope.sqlite3')
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                verify_branch_depths.main(
                    ['--cache', missing, '--answers', self.answers_path])
        self.assertNotEqual(raised.exception.code, 0)
        self.assertFalse(os.path.exists(missing))

    def test_an_audit_only_run_adds_no_response_decompositions(self):
        score_cache = self.open_cache()
        score_cache._conn.execute("DELETE FROM response_decomposition")
        self.assertEqual(self.decomposition_count(), 0)

        with redirect_stdout(io.StringIO()):
            verify_branch_depths.main(
                ['--cache', self.cache_path, '--answers', self.answers_path])
        self.assertEqual(self.decomposition_count(), 0)

    def test_an_audit_only_run_leaves_the_file_byte_identical(self):
        before = self.cache_bytes()
        with redirect_stdout(io.StringIO()):
            verify_branch_depths.main(
                ['--cache', self.cache_path, '--answers', self.answers_path])
        self.assertEqual(self.cache_bytes(), before)

    def test_a_read_only_cache_refuses_a_write(self):
        score_cache = ScoreCache(self.cache_path, self.answer_words,
                                 checkpoint_on_close=False, read_only=True)
        self.addCleanup(score_cache.close)
        self.assertFalse(score_cache.checkpoint_on_close)
        with self.assertRaises(sqlite3.OperationalError):
            score_cache.repair_max_depth(self.deepest_chain()[-1][0], ERD_ALL, 9)

    def test_a_read_only_cache_reports_the_same_answer_list_id(self):
        writable = self.open_cache()
        read_only = ScoreCache(self.cache_path, self.answer_words,
                               checkpoint_on_close=False, read_only=True)
        self.addCleanup(read_only.close)
        self.assertEqual(read_only.answer_list_id, writable.answer_list_id)


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

    def test_an_understated_row_exits_nonzero_and_is_listed(self):
        parent = self.deepest_chain()[-2]
        self.understate(parent)
        status, output = self.run_main('--list', '10')
        self.assertEqual(status, 1)
        self.assertIn(branch_reference(parent[0]), output)
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
             'repaired': 1106, 'repair_withheld': 30,
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
