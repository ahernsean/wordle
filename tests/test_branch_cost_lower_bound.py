"""Tests for the branch ERD floor and the two-level candidate bound it feeds.

The floor answers "what is the least this word set can possibly cost", priced
by what the guess pool can actually do to those words rather than by assuming a
perfect all-singletons split. It must never exceed the set's true ERD — an
inadmissible floor would prune the optimum and silently return a wrong answer —
and the pattern-matrix and pure-Python implementations must agree, since the
engine chooses between them by which kernel a caller passes.
"""
import os
import random
import tempfile
import unittest

from cache_sqlite import ScoreCache
from pattern_matrix import PatternMatrix
from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH
from wordle_engine import (
    ERD_ALL, BranchFloorTable, ResponseCache, all_singletons_floor,
    evaluate_candidate,
    _candidate_cost_lower_bound, min_expected_guesses,
    sub_branch_cost_lower_bound,
)


def _load_words(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALL_GUESS_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_CANDIDATE_LIST_PATH))
_ALL_ANSWER_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_ANSWER_LIST_PATH))


class _VocabularyMixin:
    """A guess pool that is exactly the pattern matrix's rows, as in the game:
    the guess list is a superset of the answer list, so every branch word is
    also a candidate."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(11)
        cls.answer_words = sorted(rng.sample(_ALL_ANSWER_WORDS, 120))
        extra = rng.sample(
            [w for w in _ALL_GUESS_WORDS if w not in set(cls.answer_words)], 120)
        cls.guess_words = sorted(set(cls.answer_words) | set(extra))
        cls.pattern_matrix = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def _response_cache(self):
        return ResponseCache(self.answer_words)

    def _table(self, pool=None, with_matrix=True):
        return BranchFloorTable(
            self.guess_words if pool is None else pool,
            cache=self._response_cache(),
            pattern_matrix=self.pattern_matrix if with_matrix else None)

    def _branch(self, size, seed):
        return sorted(random.Random(seed).sample(self.answer_words, size))


class TestImplementationsAgree(_VocabularyMixin, unittest.TestCase):

    def test_matrix_and_python_floors_are_identical(self):
        vectorized = self._table(with_matrix=True)
        reference = self._table(with_matrix=False)
        for size in (2, 3, 5, 9, 17, 40, 75, 120):
            for seed in (size, size + 1000):
                words = self._branch(size, seed)
                with self.subTest(size=size, seed=seed):
                    self.assertEqual(
                        vectorized.branch_cost_lower_bound(words),
                        reference.branch_cost_lower_bound(words))

    def test_memo_hit_returns_the_computed_value(self):
        table = self._table()
        words = self._branch(30, seed=7)
        first = table.branch_cost_lower_bound(words)
        misses = table.misses
        second = table.branch_cost_lower_bound(words)
        self.assertEqual(first, second)
        self.assertEqual(table.misses, misses)
        self.assertGreaterEqual(table.hits, 1)

    def test_memo_is_keyed_by_the_set_not_the_ordering(self):
        table = self._table()
        words = self._branch(24, seed=3)
        shuffled = list(words)
        random.Random(99).shuffle(shuffled)
        table.branch_cost_lower_bound(words)
        misses = table.misses
        table.branch_cost_lower_bound(shuffled)
        self.assertEqual(table.misses, misses)

    def test_memo_evicts_oldest_past_capacity(self):
        table = BranchFloorTable(self.guess_words, cache=self._response_cache(),
                                 pattern_matrix=self.pattern_matrix, capacity=2)
        for seed in range(5):
            table.branch_cost_lower_bound(self._branch(6, seed=seed))
        self.assertLessEqual(len(table._floors), 2)


class TestAdmissibility(_VocabularyMixin, unittest.TestCase):
    """The floor must never exceed the exact ERD it bounds."""

    def _exact_erd(self, words):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as tmp:
            pass
        self.addCleanup(os.unlink, tmp.name)
        score_cache = ScoreCache(tmp.name, self.answer_words)
        self.addCleanup(score_cache.close)
        cache = ResponseCache(self.answer_words, score_cache)
        return min_expected_guesses(
            words, cache, score_cache, guesses=self.guess_words,
            policy=ERD_ALL, pattern_matrix=self.pattern_matrix)

    def test_floor_never_exceeds_exact_erd(self):
        for size in (2, 3, 4, 6, 9, 14, 22):
            words = self._branch(size, seed=size * 7)
            with self.subTest(size=size):
                floor = self._table().branch_cost_lower_bound(words)
                self.assertLessEqual(floor, self._exact_erd(words) + 1e-9)

    def test_floor_is_never_below_the_all_singletons_floor(self):
        for size in (2, 5, 13, 40, 120):
            words = self._branch(size, seed=size)
            with self.subTest(size=size):
                floor = self._table().branch_cost_lower_bound(words)
                self.assertGreaterEqual(floor, all_singletons_floor(size) - 1e-12)

    def test_two_level_bound_dominates_the_group_count_bound(self):
        """Summing per-sub-branch floors is never looser than 3 - (G + self)/n,
        which is that same sum with every floor at all_singletons_floor."""
        cache = self._response_cache()
        for size in (20, 60, 120):
            words = self._branch(size, seed=size + 5)
            for candidate in (self.guess_words[0], words[0], self.guess_words[-1]):
                with self.subTest(size=size, candidate=candidate):
                    groups = cache.group_words(candidate, words)
                    has_self = any(len(g) == 1 and g[0] == candidate
                                   for g in groups.values())
                    closed_form = _candidate_cost_lower_bound(
                        groups.values(), has_self, size)
                    table = self._table()
                    two_level = 1.0 + sum(
                        (len(g) / size) * sub_branch_cost_lower_bound(
                            g, candidate, table)
                        for g in groups.values())
                    self.assertGreaterEqual(two_level, closed_form - 1e-9)


class TestSubBranchFloorSemantics(_VocabularyMixin, unittest.TestCase):

    def test_singleton_of_the_candidate_costs_nothing_further(self):
        candidate = self.answer_words[0]
        self.assertEqual(
            sub_branch_cost_lower_bound([candidate], candidate, self._table()),
            0.0)

    def test_singleton_of_another_word_costs_one_guess(self):
        candidate = self.answer_words[0]
        self.assertEqual(
            sub_branch_cost_lower_bound([self.answer_words[1]], candidate,
                                        self._table()),
            1.0)

    def test_without_a_table_it_degrades_to_all_singletons(self):
        words = self._branch(30, seed=2)
        self.assertEqual(
            sub_branch_cost_lower_bound(words, self.guess_words[0]),
            all_singletons_floor(len(words)))

    def test_narrow_pool_table_is_priced_by_its_own_pool(self):
        """A table built on a pool that is not the matrix's rows must not be
        priced by the matrix, whose rows would credit splits that pool cannot
        play."""
        words = self._branch(20, seed=4)
        narrow_pool = self.guess_words[:40]
        table = self._table(pool=narrow_pool)
        self.assertEqual(
            table.branch_cost_lower_bound(words),
            ResponseCache(self.answer_words).branch_cost_lower_bound(
                words, narrow_pool))


class TestPoolIdentity(unittest.TestCase):
    """A floor is only a floor over the words that may actually be played.

    aided/bided/sided/tided is the smallest case that separates the two pools:
    no word among the four splits them fully, so their own floor is 2.25, while
    abacs distinguishes all four and brings the achievable ERD down to 2.0.
    Serving 2.25 to a search allowed to play abacs prunes the optimum.
    """

    ANSWERS = ['aided', 'bided', 'sided', 'tided']
    NARROW_POOL = ['aided', 'bided', 'sided', 'tided']
    WIDE_POOL = ['aided', 'bided', 'sided', 'tided', 'abacs']

    def test_memo_does_not_serve_a_narrower_pools_floor(self):
        cache = ResponseCache(self.ANSWERS)
        narrow = cache.branch_cost_lower_bound(self.ANSWERS, self.NARROW_POOL)
        widened = cache.branch_cost_lower_bound(self.ANSWERS, self.WIDE_POOL)
        self.assertEqual(narrow, 2.25)
        self.assertEqual(widened, 2.0)
        self.assertEqual(widened,
                         ResponseCache(self.ANSWERS).branch_cost_lower_bound(
                             self.ANSWERS, self.WIDE_POOL))

    def test_rebinding_back_to_the_narrow_pool_still_answers_for_it(self):
        cache = ResponseCache(self.ANSWERS)
        cache.branch_cost_lower_bound(self.ANSWERS, self.WIDE_POOL)
        self.assertEqual(
            cache.branch_cost_lower_bound(self.ANSWERS, self.NARROW_POOL), 2.25)

    def test_same_length_pool_is_not_the_matrix_vocabulary(self):
        matrix_words = ['aided', 'bided', 'sided', 'tided', 'immix']
        same_length_pool = ['aided', 'bided', 'sided', 'tided', 'abacs']
        matrix = PatternMatrix.build(matrix_words, self.ANSWERS)
        self.assertTrue(matrix.is_guess_pool(matrix_words))
        self.assertTrue(matrix.is_guess_pool(list(matrix_words)))
        self.assertFalse(matrix.is_guess_pool(same_length_pool))
        self.assertFalse(matrix.is_guess_pool(matrix_words[:4]))
        self.assertFalse(matrix.is_guess_pool(['aided'] * 5))

    def test_the_matrix_would_have_answered_for_the_wrong_pool(self):
        """The hazard itself, at the level the two kernels disagree: asked
        about the same four words, the matrix answers 2.25 for its own
        vocabulary while the pool that may play abacs can reach 2.0."""
        matrix_words = ['aided', 'bided', 'sided', 'tided', 'immix']
        same_length_pool = ['aided', 'bided', 'sided', 'tided', 'abacs']
        matrix = PatternMatrix.build(matrix_words, self.ANSWERS)
        self.assertEqual(
            matrix.branch_cost_lower_bound(matrix.answer_indices(self.ANSWERS)),
            2.25)
        self.assertEqual(
            ResponseCache(self.ANSWERS).branch_cost_lower_bound(
                self.ANSWERS, same_length_pool),
            2.0)
        self.assertFalse(matrix.is_guess_pool(same_length_pool))


class TestForeignPoolRoutesToTheReferencePath(_VocabularyMixin, unittest.TestCase):
    """Above the size threshold, a same-length pool that is not the matrix's
    vocabulary must be priced by the reference path, not the matrix."""

    def test_same_length_foreign_pool_uses_the_pool_it_was_given(self):
        foreign_pool = list(self.guess_words)
        foreign_pool[0] = next(w for w in _ALL_GUESS_WORDS
                               if w not in set(self.guess_words))
        self.assertEqual(len(foreign_pool), self.pattern_matrix.n_guesses)
        self.assertFalse(self.pattern_matrix.is_guess_pool(foreign_pool))
        words = self._branch(20, seed=41)
        self.assertEqual(
            self._table(pool=foreign_pool).branch_cost_lower_bound(words),
            ResponseCache(self.answer_words).branch_cost_lower_bound(
                words, foreign_pool))


class TestThePoolCannotDrift(_VocabularyMixin, unittest.TestCase):
    """The structural guarantee: a table's pool is fixed at construction, so
    there is no per-call pool that can disagree with the memo.

    The two admissibility defects this replaces were both a floor computed
    under one pool being served to a search allowed to play another. Checks
    catch the cases someone anticipated; having no pool argument at all is
    what makes the mistake unrepresentable.
    """

    def test_no_method_accepts_a_candidate_pool(self):
        import inspect
        for name in ('branch_cost_lower_bound', 'sub_branch_cost_lower_bound'):
            parameters = inspect.signature(
                getattr(BranchFloorTable, name)).parameters
            with self.subTest(method=name):
                self.assertNotIn('candidate_pool', parameters)
                self.assertNotIn('pool', parameters)

    def test_the_pool_is_snapshotted_against_later_mutation(self):
        pool = list(self.guess_words)
        table = BranchFloorTable(pool, cache=self._response_cache(),
                                 pattern_matrix=self.pattern_matrix)
        words = self._branch(20, seed=51)
        before = table.branch_cost_lower_bound(words)
        pool.append('zzzzz')
        pool[0] = 'aaaaa'
        after = BranchFloorTable(
            table.candidate_pool, cache=self._response_cache(),
            pattern_matrix=self.pattern_matrix).branch_cost_lower_bound(words)
        self.assertEqual(before, after)

    def test_two_pools_need_two_tables_and_disagree(self):
        narrow = BranchFloorTable(self.ANSWERS, cache=ResponseCache(self.ANSWERS))
        wide = BranchFloorTable(self.WIDE, cache=ResponseCache(self.ANSWERS))
        self.assertEqual(narrow.branch_cost_lower_bound(self.ANSWERS), 2.25)
        self.assertEqual(wide.branch_cost_lower_bound(self.ANSWERS), 2.0)

    ANSWERS = ['aided', 'bided', 'sided', 'tided']
    WIDE = ['aided', 'bided', 'sided', 'tided', 'abacs']


class TestASuppliedTableMustBeTheSearchesPool(_VocabularyMixin, unittest.TestCase):
    """Owning its memo stops a table serving a foreign pool internally; it does
    not stop a caller handing in a table built for different words. A table
    bound to a sub-branch alone reports a floor that ignores every other word
    the search may play, which prunes reachable strategies."""

    def _reject(self, table, guesses):
        words = self._branch(20, seed=61)
        with self.assertRaises(ValueError):
            evaluate_candidate(words, self.guess_words[0],
                               self._response_cache(), None, guesses=guesses,
                               policy=ERD_ALL, budget=5,
                               pattern_matrix=self.pattern_matrix,
                               branch_floor_table=table)

    def test_table_bound_to_a_narrower_pool_is_rejected(self):
        narrow = BranchFloorTable(self.guess_words[:40],
                                  cache=self._response_cache())
        self._reject(narrow, self.guess_words)

    def test_table_is_rejected_when_the_search_has_no_pool(self):
        self._reject(self._table(), None)

    def test_a_reused_pool_object_short_circuits_the_set_rebuild(self):
        table = self._table()
        equal_pool = list(self.guess_words)
        self.assertTrue(table.matches_pool(equal_pool))
        self.assertTrue(table.matches_pool(equal_pool))

    def test_reference_floor_of_a_trivial_branch_is_its_size(self):
        cache = self._response_cache()
        self.assertEqual(cache.branch_cost_lower_bound([], self.guess_words), 0.0)
        self.assertEqual(
            cache.branch_cost_lower_bound([self.answer_words[0]],
                                          self.guess_words), 1.0)

    def test_min_expected_guesses_rejects_a_foreign_table(self):
        narrow = BranchFloorTable(self.guess_words[:40],
                                  cache=self._response_cache())
        with self.assertRaises(ValueError):
            min_expected_guesses(
                self._branch(10, seed=71), self._response_cache(), None,
                guesses=self.guess_words, policy=ERD_ALL,
                pattern_matrix=self.pattern_matrix, branch_floor_table=narrow)

    def test_table_without_a_kernel_falls_back_to_all_singletons(self):
        """No matrix and no response cache leaves nothing to compute a floor
        with, so the all-singletons floor stands."""
        table = BranchFloorTable(self.guess_words)
        words = self._branch(20, seed=72)
        self.assertEqual(table.branch_cost_lower_bound(words),
                         all_singletons_floor(len(words)))

    def test_words_outside_the_matrix_use_the_reference_path(self):
        table = self._table()
        outside = [w for w in _ALL_ANSWER_WORDS
                   if w not in set(self.answer_words)][:20]
        self.assertEqual(
            table.branch_cost_lower_bound(outside),
            ResponseCache(self.answer_words).branch_cost_lower_bound(
                outside, self.guess_words))

    def test_matching_pool_is_accepted_by_word_set_not_by_object(self):
        table = self._table()
        self.assertTrue(table.matches_pool(list(self.guess_words)))
        self.assertFalse(table.matches_pool(self.guess_words[:-1]))
        self.assertFalse(table.matches_pool(None))


class TestEverySubBranchIsPriced(_VocabularyMixin, unittest.TestCase):
    """No size is too small for the computed floor.

    A size cutoff is the obvious economy — a floor costs one pass over the
    pool however few words it prices.  It is also a false one: a sub-branch of
    k words moves its parent's bound with weight k/n, so the cutoff that keeps
    the tightening at one parent size discards it at another, and the small
    sub-branches are the numerous ones.  These pin that no cutoff is present.
    """

    def test_the_smallest_multi_word_sub_branch_gets_a_computed_floor(self):
        words = self._branch(2, seed=21)
        table = self._table()
        self.assertEqual(
            sub_branch_cost_lower_bound(words, self.guess_words[0], table),
            max(all_singletons_floor(len(words)),
                table.branch_cost_lower_bound(words)))
        self.assertGreater(table.misses, 0, "no floor was built")

    def test_every_size_up_to_a_dozen_words_gets_a_computed_floor(self):
        for size in range(2, 13):
            table = self._table()
            words = self._branch(size, seed=30 + size)
            with self.subTest(size=size):
                self.assertEqual(
                    sub_branch_cost_lower_bound(
                        words, self.guess_words[0], table),
                    max(all_singletons_floor(size),
                        table.branch_cost_lower_bound(words)))

    def test_a_singleton_costs_one_guess_or_none(self):
        table = self._table()
        word = self.answer_words[0]
        self.assertEqual(sub_branch_cost_lower_bound([word], word, table), 0.0)
        self.assertEqual(
            sub_branch_cost_lower_bound([word], self.guess_words[-1], table),
            1.0)
        self.assertEqual(table.misses, 0,
                         "a singleton needs no pass over the pool")


class TestGatesAreObservedOnce(_VocabularyMixin, unittest.TestCase):
    """The measurement path must see the bound that decided the candidate."""

    def _observe(self, branch_words, candidate, best_erd):
        seen = []
        cache = ResponseCache(self.answer_words)
        evaluate_candidate(
            branch_words, candidate, cache, None, best_erd=best_erd,
            guesses=self.guess_words, policy=ERD_ALL, budget=5,
            pattern_matrix=self.pattern_matrix,
            branch_floor_table=self._table(),
            metric_observer=lambda sizes, has_self, bound, in_force, pruned:
                seen.append((bound, pruned)))
        return seen

    def test_observer_fires_exactly_once_per_candidate(self):
        words = self._branch(40, seed=31)
        for best_erd in (float('inf'), 3.0, 2.0):
            with self.subTest(best_erd=best_erd):
                self.assertEqual(
                    len(self._observe(words, self.guess_words[0], best_erd)), 1)

    def test_candidate_cut_only_by_the_two_level_gate_reports_pruned(self):
        """A bound between the closed form and the two-level bound is cut by
        the second gate alone, and must be reported as a prune with the bound
        that did it — not as unpruned work the closed form could not refute."""
        words = self._branch(40, seed=32)
        candidate = self.guess_words[0]
        cache = ResponseCache(self.answer_words)
        groups = cache.group_words(candidate, words)
        has_self = any(len(g) == 1 and g[0] == candidate for g in groups.values())
        closed_form = _candidate_cost_lower_bound(groups.values(), has_self, len(words))
        table = self._table()
        two_level = 1.0 + sum(
            (len(g) / len(words)) * sub_branch_cost_lower_bound(g, candidate, table)
            for g in groups.values())
        self.assertGreater(two_level, closed_form,
                           "fixture precondition: the two-level bound must be "
                           "strictly tighter for this candidate")
        between = (closed_form + two_level) / 2
        seen = self._observe(words, candidate, between)
        self.assertEqual(len(seen), 1)
        bound, pruned = seen[0]
        self.assertTrue(pruned, "the second gate cut it, so it must read as pruned")
        # The reported bound stays the closed form: analyze_swarm_telemetry
        # inverts 3 - (G + has_self)/n from this field to recover has_self, so
        # storing the tighter effective bound here would corrupt that.
        self.assertEqual(bound, closed_form)


if __name__ == "__main__":
    unittest.main()
