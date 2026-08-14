"""An exhaustive ERD oracle, and the engine checked against it.

The engine reaches its answer by pruning: a candidate whose cost cannot beat
the incumbent is abandoned, and the branch floor is what proves it cannot.
Pruning is only sound if that floor never exceeds what the branch can truly
cost, and the failure it causes is silent — a pruned optimum leaves a *higher*
ERD behind, with nothing to mark that a better line was discarded.  Recovering
a previously known value does not settle it either: the same inadmissible floor
can remove the competitors that would have beaten it.

So the check here does not use the engine's own machinery to grade the engine.
The oracle below is a plain minimizing recurrence with no floor, no incumbent,
no bound of any kind, and its own response function written from Wordle's
rules.  It enumerates every strategy the pool allows and reports the exact
optimum as an integer total, so a comparison never rests on float equality.
Anything the engine returns that the oracle does not is a defect in the engine
— whether it is too low (a wrong answer) or too high (a pruned optimum).

The layers, in the order they narrow the claim:

* `TestTheOracleIsSound` — the oracle's own response function and recurrence
  agree with hand-computable cases, so a later disagreement indicts the engine.
* `TestTheEngineMatchesTheOracle` — status, ERD, chosen guess, and worst-case
  line length over a seeded sweep of branches, pools, budgets, and kernels.
* `TestTheBoundIsAdmissible` — the floor itself, in exact rational arithmetic,
  never exceeds the exact optimum.
* `TestPruningBoundaries` — incumbents one ulp below, exactly at, and one ulp
  above the optimum, which is where a strict comparison is decided.
* `TestMetamorphicProperties` — relations that must hold between solves
  (widening a pool, reordering it, permuting the branch, reusing a table)
  whatever the individual values turn out to be.
"""
import math
import os
import random
import tempfile
import unittest
from fractions import Fraction

from cache_sqlite import ScoreCache
from erd_queue import encode_subset
from pattern_matrix import PatternMatrix
from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH
from wordle_engine import (
    ERD_ALL, OVER_ERD_LIMIT, SOLVED, BranchFloorTable, ResponseCache,
    _solve_subset, calculate_response, min_expected_guesses,
)


def _load_words(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALL_GUESS_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_CANDIDATE_LIST_PATH))
_ALL_ANSWER_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_ANSWER_LIST_PATH))


def oracle_response(guess, answer):
    """Wordle's response to `guess` given `answer`, as a five-character string.

    Written from the rules rather than taken from the engine, so that a
    disagreement between the two is visible instead of shared.  Greens are
    claimed first; each yellow then consumes one of the answer's remaining
    copies of that letter, which is what makes a repeated letter in the guess
    come back gray once the answer's supply runs out.
    """
    remaining = {}
    for letter in answer:
        remaining[letter] = remaining.get(letter, 0) + 1
    marks = [None] * len(guess)
    for i, letter in enumerate(guess):
        if answer[i] == letter:
            marks[i] = 'G'
            remaining[letter] -= 1
    for i, letter in enumerate(guess):
        if marks[i] is None:
            if remaining.get(letter, 0) > 0:
                marks[i] = 'Y'
                remaining[letter] -= 1
            else:
                marks[i] = '.'
    return ''.join(marks)


class ExactERD:
    """Exhaustive minimum expected remaining depth over a fixed candidate pool.

    `solve` returns `(total, depths, best_guesses)`, or None when the words
    cannot be solved within the budget.  `total` is the sum over answers of the
    guesses each one takes, an integer: ERD is `total / n`, and keeping the
    numerator lets every comparison be exact.  `best_guesses` is every
    candidate attaining that minimum, because which one the engine records
    depends on the order it happens to try them in, and no order is more
    correct than another among equals.

    `depths` is the *set* of worst-case line lengths reachable while still
    attaining that minimum, not the shortest of them.  The engine chooses on
    ERD alone and reports whatever depth its choice implies, so more than one
    depth can be correct for the same optimal ERD — but only the ones in this
    set are, and a stored depth outside it is wrong even when it falls between
    the shortest and the budget.  `guess_depths` gives the same set per optimal
    guess, which is what lets the recorded guess be checked against its own
    depths rather than against the best any guess could manage.

    A set is needed rather than a single value because the total is a sum over
    response groups while the depth is a max: minimizing the sum is separable
    group by group, but every combination of ERD-optimal sub-strategies then
    yields some max, and the engine may land on any of them.  A max of `v` is
    reachable exactly when some group can supply `v` and no group is forced
    above it.

    There is no pruning here of any kind.  The single candidate exclusion is a
    guess that leaves every word in one group and is not itself one of them:
    that reaches the identical branch one guess later, so it can never beat
    playing something that splits.  Keeping it would only add non-terminating
    work below the budget cap.
    """

    def __init__(self, pool):
        self.pool = tuple(pool)
        self._pool_set = frozenset(pool)
        self._memo = {}
        self._responses = {}

    def _groups(self, candidate, words):
        key = (candidate, words)
        cached = self._responses.get(key)
        if cached is None:
            grouped = {}
            for word in words:
                grouped.setdefault(oracle_response(candidate, word), []).append(word)
            cached = tuple(tuple(group) for group in grouped.values())
            self._responses[key] = cached
        return cached

    def solve(self, words, budget):
        words = tuple(sorted(words))
        key = (words, budget)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = result = self._solve(words, budget)
        return result

    def _outcome(self, candidate, words, sub_budget):
        """`(total, depths)` for playing `candidate` here, or None if infeasible.

        The depth set is every worst-case line length reachable while each
        response group is solved ERD-optimally.  A group can be held to any
        depth in its own set, so a max of `v` is reachable exactly when some
        group offers `v` and every group can stay at or below it — which is
        what the `floor` below expresses.
        """
        total = len(words)
        group_depths = []
        for group in self._groups(candidate, words):
            if len(group) == 1 and group[0] == candidate:
                continue
            sub = self.solve(group, sub_budget)
            if sub is None:
                return None
            total += sub[0]
            group_depths.append(sub[1])
        if not group_depths:
            return (total, frozenset({1}))
        floor = max(min(depths) for depths in group_depths)
        return (total, frozenset(
            1 + depth
            for depths in group_depths for depth in depths if depth >= floor))

    def _solve(self, words, budget):
        n = len(words)
        if n == 0:
            return (0, frozenset({0}), ())
        if budget is not None and budget < 1:
            return None
        if n == 1:
            # A lone survivor is named directly, whether or not the pool
            # happens to list it: every answer is a legal guess in Wordle, so
            # the engine resolves a singleton without consulting the pool and
            # the oracle must model the same game.
            return (1, frozenset({1}), (words[0],))
        sub_budget = None if budget is None else budget - 1
        best_total = None
        best_depths = frozenset()
        best_guesses = []
        for candidate in self.pool:
            groups = self._groups(candidate, words)
            if len(groups) == 1 and candidate not in words:
                continue
            outcome = self._outcome(candidate, words, sub_budget)
            if outcome is None:
                continue
            total, depths = outcome
            if best_total is None or total < best_total:
                best_total, best_depths, best_guesses = total, depths, [candidate]
            elif total == best_total:
                best_guesses.append(candidate)
                best_depths |= depths
        if best_total is None:
            return None
        return (best_total, best_depths, tuple(best_guesses))

    def guess_depths(self, words, budget):
        """`{guess: depths}` for every ERD-optimal guess, or {} if unsolvable.

        The engine records one guess and the depth its own descent produced.
        Checking that depth against this guess's own set is exact; checking it
        against the shortest depth any optimal guess could reach would accept a
        stored depth no strategy actually attains, and `max_remaining_depth` is
        the feasibility gate and the cache-reuse key.
        """
        words = tuple(sorted(words))
        result = self.solve(words, budget)
        if result is None:
            return {}
        sub_budget = None if budget is None else budget - 1
        return {candidate: self._outcome(candidate, words, sub_budget)[1]
                for candidate in result[2]}

    def erd(self, words, budget):
        """The exact optimum as a Fraction, or None when unsolvable."""
        result = self.solve(words, budget)
        return None if result is None else Fraction(result[0], len(words))


class _CaseMixin:
    """A small vocabulary whose branches are exhaustible by the oracle.

    The words are drawn from the real lists so that responses have the shapes
    the game produces — repeated letters, near-misses, whole groups that one
    guess cannot separate — rather than the tidy splits a synthetic alphabet
    gives.
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(2029)
        cls.answers = sorted(rng.sample(_ALL_ANSWER_WORDS, 60))
        extra = sorted(rng.sample(
            [w for w in _ALL_GUESS_WORDS if w not in set(cls.answers)], 40))
        cls.wide_pool = tuple(sorted(set(cls.answers) | set(extra)))
        cls.narrow_pool = tuple(cls.answers)
        # A pool of the right size drawn from words the branches never contain,
        # so no candidate is ever its own answer.
        cls.foreign_pool = tuple(sorted(rng.sample(
            [w for w in _ALL_GUESS_WORDS if w not in cls.wide_pool], 60)))
        cls.matrix = PatternMatrix.build(list(cls.wide_pool), cls.answers)

    def branch(self, size, seed):
        return sorted(random.Random(seed).sample(self.answers, size))

    def score_cache(self, words):
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return ScoreCache(handle.name, words)

    def engine_solve(self, branch, pool, budget, matrix=None, table=None):
        """Run the engine and return (erd, best_guess, max_remaining_depth)."""
        cache = ResponseCache(list(self.answers))
        score_cache = self.score_cache(list(self.answers))
        try:
            cost = min_expected_guesses(
                branch, cache, score_cache, guesses=pool, budget=budget,
                policy=ERD_ALL, pattern_matrix=matrix, branch_floor_table=table)
            if cost is None:
                return (None, None, None)
            recorded = score_cache.read_with_depth(encode_subset(branch), ERD_ALL)
            guess = recorded[0] if recorded else None
            depth = recorded[2] if recorded else None
            return (cost, guess, depth)
        finally:
            score_cache.close()


class TestTheOracleIsSound(_CaseMixin, unittest.TestCase):
    """Before the oracle can indict the engine it has to be right itself."""

    def test_the_response_function_matches_the_engines(self):
        rng = random.Random(7)
        colours = {'green': 'G', 'yellow': 'Y', 'gray': '.'}
        for _ in range(4000):
            guess = rng.choice(_ALL_GUESS_WORDS)
            answer = rng.choice(_ALL_ANSWER_WORDS)
            engine = ''.join(colours[c] for c in calculate_response(guess, answer))
            self.assertEqual(oracle_response(guess, answer), engine,
                             "%s vs %s" % (guess, answer))

    def test_repeated_letters_follow_the_answers_supply(self):
        # One 'l' in the answer, two in the guess: the aligned one is green and
        # the second has nothing left to claim.
        self.assertEqual(oracle_response('lolly', 'lucid'), 'G....')
        self.assertEqual(oracle_response('allay', 'lucid'), '.Y...')
        self.assertEqual(oracle_response('crane', 'crane'), 'GGGGG')

    def test_a_shattering_guess_costs_one_plus_one_each(self):
        """A pool word that separates every answer and is one of them.

        `chino` gives each of these four a different response, so playing it
        finds itself in one guess and pins each of the others for the second:
        the total is 1 + 2 + 2 + 2 = 7.
        """
        words = ['chino', 'fakir', 'regal', 'share']
        oracle = ExactERD(words)
        total, depths, guesses = oracle.solve(words, 5)
        self.assertEqual(len({oracle_response('chino', w) for w in words}), 4)
        self.assertEqual(total, 7)
        self.assertEqual(depths, frozenset({2}))
        self.assertEqual(oracle.erd(words, 5), Fraction(7, 4))
        self.assertIn('chino', guesses)

    def test_words_one_letter_apart_cost_a_guess_each(self):
        """The opposite shape, where nothing in the pool separates anything.

        These four differ only in their first letter, and no word here has
        anything to say about that letter unless it *is* the answer.  So the
        strategy is to name them one at a time: 1 + 2 + 3 + 4 = 10.
        """
        words = ['aided', 'bided', 'sided', 'tided']
        oracle = ExactERD(words)
        self.assertEqual(oracle.solve(words, 5)[:2], (10, frozenset({4})))
        self.assertIsNotNone(oracle.solve(words, 4))
        self.assertEqual(oracle.erd(words, 5), Fraction(10, 4))

    def test_a_lone_survivor_is_named_without_asking_the_pool(self):
        """The pool constrains what narrows a branch, not what ends one.

        Every answer is a legal guess, so a branch down to one word costs one
        guess even from a pool that does not list it — which is what lets a
        pool of foreign words solve anything at all.
        """
        self.assertEqual(ExactERD(['crane']).solve(['crane'], 1),
                         (1, frozenset({1}), ('crane',)))
        self.assertEqual(ExactERD(['slate']).solve(['crane'], 5),
                         (1, frozenset({1}), ('crane',)))
        self.assertIsNone(ExactERD(['slate']).solve(['crane'], 0))

    def test_a_budget_below_the_needed_depth_is_unsolvable(self):
        words = ['chino', 'fakir', 'regal', 'share']
        oracle = ExactERD(words)
        self.assertIsNone(oracle.solve(words, 1))
        self.assertIsNotNone(oracle.solve(words, 2))
        # The unseparable four need all four guesses, so three is not enough.
        one_apart = ExactERD(['aided', 'bided', 'sided', 'tided'])
        self.assertIsNone(one_apart.solve(['aided', 'bided', 'sided', 'tided'], 3))

    def test_an_in_range_depth_is_not_automatically_an_attainable_one(self):
        """Why the depth check reads a set and not an interval.

        These four are told apart by one guess, so every ERD-optimal strategy
        finishes in two and no reachable line is longer — even though the
        budget allows five.  A depth of 3 sits between the shortest optimum and
        the budget while corresponding to no strategy at all, so a check that
        only bounded the recorded depth from both sides would wave it through.
        """
        words = ['chino', 'fakir', 'regal', 'share']
        oracle = ExactERD(words)
        for guess, depths in oracle.guess_depths(words, 5).items():
            with self.subTest(guess=guess):
                self.assertEqual(depths, frozenset({2}))
                self.assertNotIn(3, depths)
                self.assertNotIn(5, depths)

    def test_the_depth_set_is_usually_narrower_than_the_budget_allows(self):
        """The tightening is not a technicality — it is most of the interval.

        Across the sweep the attainable depths for an optimal guess almost
        always leave some longer line unreachable, so accepting anything up to
        the budget would accept a stored depth no strategy produces.
        """
        rng = random.Random(31337)
        narrower = 0
        cases = 0
        for _ in range(60):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 7)))
            pool = rng.choice([self.narrow_pool, self.wide_pool])
            budget = rng.choice([4, 5])
            if ExactERD(pool).solve(branch, budget) is None:
                continue
            for depths in ExactERD(pool).guess_depths(branch, budget).values():
                cases += 1
                if set(range(min(depths), budget + 1)) - set(depths):
                    narrower += 1
        self.assertGreater(cases, 100)
        self.assertGreater(narrower, cases // 2,
                           "the depth set is no tighter than a range check")

    def test_a_non_splitting_candidate_is_never_the_optimum(self):
        """The one candidate the oracle skips could not have won anyway.

        `aided` and `bided` differ only where `stout` has nothing to say, so
        `stout` returns both words in one group and leaves the branch exactly
        as it found it.  Any strategy opening with it is one guess longer than
        the same strategy without it.
        """
        words = ['aided', 'bided']
        oracle = ExactERD(['stout', 'aided', 'bided'])
        self.assertEqual(oracle._groups('stout', tuple(words)), (tuple(words),))
        self.assertEqual(oracle.solve(words, 5)[0], 3)  # 1 + 2, not 2 + 3


class TestTheEngineMatchesTheOracle(_CaseMixin, unittest.TestCase):
    """The engine's answer, against an answer computed without any pruning."""

    def _compare(self, branch, pool, budget, matrix=None, table=None):
        oracle = ExactERD(pool)
        expected = oracle.solve(branch, budget)
        erd, guess, depth = self.engine_solve(branch, pool, budget,
                                              matrix=matrix, table=table)
        if expected is None:
            self.assertIsNone(erd, "engine solved what no strategy can")
            return
        total, _depths, best_guesses = expected
        self.assertIsNotNone(erd, "engine gave up on a solvable branch")
        self.assertEqual(round(erd * len(branch)), total,
                         "ERD %r is not %d/%d" % (erd, total, len(branch)))
        self.assertAlmostEqual(erd, total / len(branch), places=9)
        self.assertIn(guess, best_guesses,
                      "recorded %r, which is not ERD-optimal" % (guess,))
        if budget is not None:
            self.assertLessEqual(depth, budget)
            reachable = oracle.guess_depths(branch, budget)[guess]
            self.assertIn(depth, sorted(reachable),
                          "recorded depth %r for %r, which no ERD-optimal "
                          "strategy on that guess attains" % (depth, guess))

    def test_branch_sizes_against_the_wide_pool(self):
        for size in range(2, 9):
            with self.subTest(size=size):
                self._compare(self.branch(size, seed=100 + size),
                              self.wide_pool, 5, matrix=self.matrix)

    def test_narrow_and_wide_pools_on_the_same_branches(self):
        for size in (3, 5, 7):
            branch = self.branch(size, seed=200 + size)
            for name, pool in (('narrow', self.narrow_pool),
                               ('wide', self.wide_pool)):
                with self.subTest(size=size, pool=name):
                    self._compare(branch, pool, 5)

    def test_a_foreign_pool_that_never_contains_the_answer(self):
        """Every line needs one more guess than a self-containing pool's does."""
        for size in (2, 4, 6):
            with self.subTest(size=size):
                self._compare(self.branch(size, seed=300 + size),
                              self.foreign_pool, 5)

    def test_budgets_around_the_feasibility_boundary(self):
        for size in (4, 6, 8):
            branch = self.branch(size, seed=400 + size)
            for budget in (1, 2, 3, 4, None):
                with self.subTest(size=size, budget=budget):
                    self._compare(branch, self.wide_pool, budget)

    def test_a_reordered_pool_reaches_the_same_optimum(self):
        branch = self.branch(6, seed=501)
        shuffled = list(self.wide_pool)
        random.Random(9).shuffle(shuffled)
        self._compare(branch, tuple(shuffled), 5)

    def test_every_kernel_and_table_combination(self):
        branch = self.branch(6, seed=600)
        for matrix_name, matrix in (('none', None), ('exact', self.matrix)):
            for supplied in (False, True):
                table = (BranchFloorTable(
                    self.wide_pool, cache=ResponseCache(list(self.answers)),
                    pattern_matrix=matrix) if supplied else None)
                with self.subTest(matrix=matrix_name, table=supplied):
                    self._compare(branch, self.wide_pool, 5,
                                  matrix=matrix, table=table)

    def test_a_seeded_sweep_of_branches_and_pools(self):
        rng = random.Random(31337)
        for case in range(120):
            size = rng.randint(2, 7)
            branch = sorted(rng.sample(self.answers, size))
            pool = rng.choice([self.narrow_pool, self.wide_pool,
                               self.foreign_pool])
            budget = rng.choice([2, 3, 4, 5, None])
            matrix = self.matrix if pool is self.wide_pool and rng.random() < 0.5 else None
            with self.subTest(case=case, size=size, budget=budget,
                              pool=len(pool), matrix=matrix is not None):
                self._compare(branch, pool, budget, matrix=matrix)


class TestTheBoundIsAdmissible(_CaseMixin, unittest.TestCase):
    """The floor, checked against the exact optimum in rational arithmetic.

    The claim being tested is the one the whole PR rests on: for a branch of
    `k` words, playing candidate `c` leaves at most one word solved by this
    guess and at best one word per response group solved by the next, so the
    guesses summed over all answers is at least `3k - (G_c + has_self_c)`.
    Minimizing over a fixed pool gives `3 - max_c(G_c + has_self_c) / k`, and
    the maximum of that with the all-singletons floor `2 - 1/k` is admissible
    because both are.
    """

    def exact_bound(self, branch, pool):
        """The floor recomputed as a Fraction, from integer counts only."""
        k = len(branch)
        widest = 0
        for candidate in pool:
            groups = set()
            for word in branch:
                groups.add(oracle_response(candidate, word))
            has_self = 1 if candidate in branch else 0
            widest = max(widest, len(groups) + has_self)
        return max(Fraction(2) - Fraction(1, k), Fraction(3) - Fraction(widest, k))

    def test_the_bound_never_exceeds_the_exact_optimum(self):
        rng = random.Random(4242)
        checked = 0
        for _ in range(150):
            size = rng.randint(2, 7)
            branch = sorted(rng.sample(self.answers, size))
            pool = rng.choice([self.narrow_pool, self.wide_pool, self.foreign_pool])
            optimum = ExactERD(pool).erd(branch, None)
            if optimum is None:
                continue
            checked += 1
            bound = self.exact_bound(branch, pool)
            self.assertLessEqual(
                bound, optimum,
                "floor %s exceeds the optimum %s on %r" % (bound, optimum, branch))
        self.assertGreater(checked, 100, "the sweep solved almost nothing")

    def test_the_implementation_agrees_with_the_exact_bound(self):
        """What the table computes is that Fraction, up to float rounding.

        A float cannot hold 1/3, so the computed floor may sit a rounding
        above the rational one.  That is the same rounding the engine's own
        ERD carries, and the comparison between them is made in floats, so it
        cannot separate them — but it must stay a rounding and never a
        difference in the arithmetic.
        """
        rng = random.Random(99)
        for _ in range(40):
            size = rng.randint(2, 8)
            branch = sorted(rng.sample(self.answers, size))
            for pool in (self.narrow_pool, self.wide_pool, self.foreign_pool):
                exact = self.exact_bound(branch, pool)
                table = BranchFloorTable(pool, cache=ResponseCache(list(self.answers)))
                computed = table.branch_cost_lower_bound(branch)
                with self.subTest(branch=branch, pool=len(pool)):
                    self.assertLessEqual(abs(computed - float(exact)),
                                         4 * math.ulp(float(exact)))

    def test_the_matrix_kernel_computes_the_same_exact_bound(self):
        rng = random.Random(1234)
        reference = BranchFloorTable(self.wide_pool,
                                     cache=ResponseCache(list(self.answers)))
        vectorized = BranchFloorTable(self.wide_pool,
                                      cache=ResponseCache(list(self.answers)),
                                      pattern_matrix=self.matrix)
        for _ in range(40):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 9)))
            with self.subTest(branch=branch):
                self.assertEqual(reference.branch_cost_lower_bound(branch),
                                 vectorized.branch_cost_lower_bound(branch))


class TestPruningBoundaries(_CaseMixin, unittest.TestCase):
    """Incumbents placed exactly where the strict comparison is decided.

    The engine keeps a candidate only while `cost < best_erd`, so an incumbent
    equal to the optimum must reject the branch and one a single ulp above it
    must accept.  Anything looser than an ulp leaves the comparison itself
    untested — a `<=` written where `<` belongs passes every ordinary case.

    The incumbents are placed around the cost the engine's own arithmetic
    produces rather than around the exact rational, because that is the
    quantity the comparison actually reads.  How far the two can drift apart
    is a separate question, and `test_the_computed_cost_tracks_the_exact_optimum`
    is where it is bounded.
    """

    def _solve(self, branch, pool, budget, ceiling):
        cache = ResponseCache(list(self.answers))
        score_cache = self.score_cache(list(self.answers))
        try:
            return _solve_subset(
                branch, cache, score_cache, budget, None, tuple(pool), ERD_ALL,
                None, None, None, None,
                branch_floor_table=BranchFloorTable(tuple(pool), cache=cache),
                ceiling=ceiling)
        finally:
            score_cache.close()

    def test_an_incumbent_at_the_optimum_prunes_and_one_ulp_above_does_not(self):
        for size in (3, 5, 6):
            branch = self.branch(size, seed=700 + size)
            unbounded = self._solve(branch, self.wide_pool, 5, float('inf'))
            self.assertEqual(unbounded[0], SOLVED)
            optimum = unbounded[1]
            with self.subTest(size=size):
                below = self._solve(branch, self.wide_pool, 5,
                                    math.nextafter(optimum, 0.0))
                self.assertEqual(below[0], OVER_ERD_LIMIT)
                at = self._solve(branch, self.wide_pool, 5, optimum)
                self.assertEqual(at[0], OVER_ERD_LIMIT,
                                 "an incumbent equal to the optimum was beaten")
                # A few ulps of headroom, not one.  The bound handed to a
                # sub-branch is derived by subtracting the cost accumulated so
                # far, and that subtraction spends the margin: an optimum that
                # beats the incumbent by a single ulp cannot be proved to.
                # `test_distinct_erds_are_further_apart_than_the_rounding`
                # is why no reachable answer is lost in that gap.
                headroom = optimum
                for _ in range(4):
                    headroom = math.nextafter(headroom, math.inf)
                above = self._solve(branch, self.wide_pool, 5, headroom)
                self.assertEqual(above[0], SOLVED)
                self.assertEqual(above[1], optimum)

    def test_an_incumbent_far_below_the_optimum_prunes_the_whole_branch(self):
        branch = self.branch(6, seed=808)
        optimum = float(ExactERD(self.wide_pool).erd(branch, 5))
        status, _cost, _depth, _floor = self._solve(
            branch, self.wide_pool, 5, optimum - 0.5)
        self.assertEqual(status, OVER_ERD_LIMIT)

    def test_the_computed_cost_tracks_the_exact_optimum(self):
        """How far float accumulation can move the answer, stated as a number.

        The optimum is a rational — a sum of `(k/n) * sub_cost` terms — and the
        engine accumulates it in floats, so the value it reports sits a few
        ulps either side of the exact one.  That drift is arithmetic, not
        pruning: it is bounded here so that a real change in what the search
        finds cannot hide inside a tolerance nobody measured.
        """
        rng = random.Random(2468)
        worst = 0
        for _ in range(30):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 7)))
            exact = float(ExactERD(self.wide_pool).erd(branch, 5))
            computed = self._solve(branch, self.wide_pool, 5, float('inf'))[1]
            gap = 0
            low, high = min(exact, computed), max(exact, computed)
            while low < high:
                low = math.nextafter(low, math.inf)
                gap += 1
            worst = max(worst, gap)
            with self.subTest(branch=branch):
                self.assertLessEqual(gap, 8, "%r vs exact %r" % (computed, exact))
        self.assertGreater(worst, 0, "no case exercised the rounding at all")

    def test_distinct_erds_are_further_apart_than_the_rounding(self):
        """Why float accumulation cannot cost the search the right answer.

        Every answer in a branch is found in a whole number of guesses, so the
        expected remaining depth of any strategy over `n` words is an integer
        over `n`.  Two strategies that differ at all therefore differ by at
        least `1/n`.  The accumulated rounding measured above is a handful of
        ulps — around `1e-16` at these magnitudes, and around `1e-13` even on
        a branch of hundreds — so it is many orders of magnitude too small to
        move a value across the gap between two achievable ERDs.  Rounding can
        make the reported number's last digits wrong; it cannot make the search
        prefer a different strategy.
        """
        rng = random.Random(13579)
        for _ in range(20):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 7)))
            n = len(branch)
            total, _depths, _guesses = ExactERD(self.wide_pool).solve(branch, 5)
            computed = self._solve(branch, self.wide_pool, 5, float('inf'))[1]
            with self.subTest(branch=branch):
                # On the 1/n grid, and nowhere near the next point on it.
                self.assertEqual(round(computed * n), total)
                self.assertLess(abs(computed - total / n), (1.0 / n) / 1e6)


class TestTheOptimumSurvivesATightIncumbent(_CaseMixin, unittest.TestCase):
    """Where the floor is load-bearing: a bound the search has to argue against.

    With no incumbent the first candidate evaluated is costed in full and the
    floor never has to prove anything, so an inadmissible floor can sit in the
    code without changing a single answer.  The swarm never runs that way — it
    starts from an incumbent published by another worker, and every candidate
    after the first must be shown unable to beat it.  That is when a floor one
    part in a billion too high starts discarding the optimum.

    So each case here hands the search an incumbent a few ulps above the exact
    optimum.  The optimum still beats it, and the search must still find it; a
    floor that overstates any sub-branch by any margin prunes the only strategy
    that could, and the branch comes back OVER_ERD_LIMIT instead.
    """

    def _tight(self, branch, pool, budget, matrix=None):
        oracle = ExactERD(pool)
        expected = oracle.solve(branch, budget)
        if expected is None:
            return
        total = expected[0]
        optimum = total / len(branch)
        incumbent = optimum
        for _ in range(8):
            incumbent = math.nextafter(incumbent, math.inf)
        cache = ResponseCache(list(self.answers))
        score_cache = self.score_cache(list(self.answers))
        try:
            status, cost, _depth, _floor = _solve_subset(
                branch, cache, score_cache, budget, None, tuple(pool), ERD_ALL,
                None, None, None, None,
                branch_floor_table=BranchFloorTable(
                    tuple(pool), cache=cache, pattern_matrix=matrix),
                ceiling=incumbent, pattern_matrix=matrix)
        finally:
            score_cache.close()
        self.assertEqual(
            status, SOLVED,
            "the optimum %d/%d was pruned under an incumbent above it"
            % (total, len(branch)))
        self.assertEqual(round(cost * len(branch)), total)

    def test_branch_sizes_under_a_tight_incumbent(self):
        for size in range(2, 9):
            with self.subTest(size=size):
                self._tight(self.branch(size, seed=100 + size),
                            self.wide_pool, 5, matrix=self.matrix)

    def test_every_pool_under_a_tight_incumbent(self):
        for size in (3, 5, 7):
            branch = self.branch(size, seed=200 + size)
            for name, pool in (('narrow', self.narrow_pool),
                               ('wide', self.wide_pool),
                               ('foreign', self.foreign_pool)):
                with self.subTest(size=size, pool=name):
                    self._tight(branch, pool, 5)

    def test_a_seeded_sweep_under_a_tight_incumbent(self):
        rng = random.Random(80808)
        for case in range(80):
            size = rng.randint(2, 7)
            branch = sorted(rng.sample(self.answers, size))
            pool = rng.choice([self.narrow_pool, self.wide_pool,
                               self.foreign_pool])
            budget = rng.choice([3, 4, 5, None])
            matrix = self.matrix if pool is self.wide_pool and rng.random() < 0.5 else None
            with self.subTest(case=case, size=size, budget=budget,
                              pool=len(pool), matrix=matrix is not None):
                self._tight(branch, pool, budget, matrix=matrix)


class TestTheFloorIsLoadBearing(unittest.TestCase):
    """Branches where the floor is the thing being tested, not merely present.

    A branch a pool shatters in one guess never asks the floor anything: every
    response group is a singleton, priced at one guess without consulting it.
    Every case above is of that shape, so none of them would notice a floor
    that was wrong — which is the trap this class exists to avoid.

    The branches here are word families differing in a single letter, against
    pools that separate them poorly.  Response groups stay multi-word several
    levels down, so the floor is asked about real sub-branches, and one case
    sits above ERD 3.0 — the range where the group-count bound cannot reach and
    the branch floor is the only thing pruning at all.

    Each case asserts the floor was consulted.  Without that a later change
    could stop reaching it and leave these tests passing on nothing.
    """

    ATER = ['after', 'alter', 'aster', 'cater', 'dater', 'eater', 'hater',
            'later', 'rater', 'tater', 'water']
    IGHT = ['eight', 'fight', 'light', 'might', 'night', 'right', 'sight',
            'tight', 'wight']
    OUND = ['bound', 'found', 'hound', 'mound', 'pound', 'round', 'sound',
            'wound']

    @classmethod
    def setUpClass(cls):
        cls.vocabulary = sorted(set(_ALL_ANSWER_WORDS))

    def pool_for(self, family, extra, seed):
        """The family itself plus a seeded handful of unrelated guess words."""
        rng = random.Random(seed)
        outside = [w for w in _ALL_GUESS_WORDS
                   if w not in set(self.ATER) | set(self.IGHT) | set(self.OUND)]
        return tuple(sorted(set(family) | set(rng.sample(outside, extra))))

    # Each case is (name, branch, extra pool words, budget).  The families are
    # taken in list order so the branches are stable, and the budgets are ones
    # the branch can actually meet — an unsolvable case would exercise the
    # feasibility gate instead of the floor.
    CASES = (('ater-6', ATER[:6], 4, 5),
             ('ater-8', ATER[:8], 8, 5),
             ('ater-10', ATER[:10], 12, 5),
             ('ater-10-unlimited', ATER[:10], 4, None),
             ('ater-10-unlimited-wider', ATER[:10], 8, None),
             ('ight-6', IGHT[:6], 12, 5))

    def cases(self):
        for name, family, extra, budget in self.CASES:
            yield name, sorted(family), self.pool_for(family, extra, 5), budget

    def _solve(self, branch, pool, budget, ceiling, matrix=None):
        cache = ResponseCache(self.vocabulary)
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        score_cache = ScoreCache(handle.name, self.vocabulary)
        table = BranchFloorTable(tuple(pool), cache=cache, pattern_matrix=matrix)
        try:
            result = _solve_subset(
                branch, cache, score_cache, budget, None, tuple(pool), ERD_ALL,
                None, None, None, None, branch_floor_table=table,
                ceiling=ceiling, pattern_matrix=matrix)
        finally:
            score_cache.close()
        return result, table.hits + table.misses

    def _matrix_for(self, pool, branch):
        """A matrix whose rows are exactly the pool, so the table uses it.

        The vectorized kernel is what the swarm runs; the reference kernel is
        the one everything else here exercises.  A floor is only as admissible
        as the implementation that actually computed it.
        """
        answers = sorted(set(branch) | (set(pool) & set(self.vocabulary)))
        return PatternMatrix.build(list(pool), answers)

    def test_the_floor_is_consulted_on_every_case(self):
        for name, branch, pool, budget in self.cases():
            expected = ExactERD(pool).solve(branch, budget)
            self.assertIsNotNone(expected, "%s is unsolvable" % name)
            incumbent = expected[0] / len(branch)
            for _ in range(8):
                incumbent = math.nextafter(incumbent, math.inf)
            _result, consultations = self._solve(branch, pool, budget, incumbent)
            with self.subTest(case=name):
                self.assertGreater(consultations, 0,
                                   "the floor was never asked anything")

    def test_the_engine_matches_the_oracle_on_hard_branches(self):
        for name, branch, pool, budget in self.cases():
            oracle = ExactERD(pool)
            total, _depths, _best = oracle.solve(branch, budget)
            (status, cost, depth, _floor), _n = self._solve(
                branch, pool, budget, float('inf'))
            with self.subTest(case=name):
                self.assertEqual(status, SOLVED)
                self.assertEqual(round(cost * len(branch)), total,
                                 "%s: %r is not %d/%d"
                                 % (name, cost, total, len(branch)))
                if budget is not None:
                    self.assertLessEqual(depth, budget)
                    reachable = set()
                    for depths in oracle.guess_depths(branch, budget).values():
                        reachable |= depths
                    self.assertIn(depth, sorted(reachable),
                                  "%s: depth %r is unattainable" % (name, depth))

    def test_the_optimum_survives_a_tight_incumbent_on_hard_branches(self):
        """The regime the swarm runs in, on branches that need the floor.

        Every candidate but the first has to be shown unable to beat a bound
        the optimum only just clears.  A floor that overstates any sub-branch
        discards the strategy that clears it, and the branch comes back
        OVER_ERD_LIMIT with nothing to say a better line existed.
        """
        for name, branch, pool, budget in self.cases():
            total = ExactERD(pool).solve(branch, budget)[0]
            incumbent = total / len(branch)
            for _ in range(8):
                incumbent = math.nextafter(incumbent, math.inf)
            (status, cost, _depth, _floor), consultations = self._solve(
                branch, pool, budget, incumbent)
            with self.subTest(case=name):
                self.assertGreater(consultations, 0)
                self.assertEqual(
                    status, SOLVED,
                    "%s: the optimum %d/%d was pruned under an incumbent "
                    "above it" % (name, total, len(branch)))
                self.assertEqual(round(cost * len(branch)), total)

    def test_the_vectorized_kernel_survives_a_tight_incumbent(self):
        for name, branch, pool, budget in self.cases():
            matrix = self._matrix_for(pool, branch)
            total = ExactERD(pool).solve(branch, budget)[0]
            incumbent = total / len(branch)
            for _ in range(8):
                incumbent = math.nextafter(incumbent, math.inf)
            (status, cost, _depth, _floor), consultations = self._solve(
                branch, pool, budget, incumbent, matrix=matrix)
            with self.subTest(case=name):
                self.assertGreater(consultations, 0)
                self.assertEqual(status, SOLVED,
                                 "%s: the optimum was pruned" % name)
                self.assertEqual(round(cost * len(branch)), total)

    def test_both_kernels_reach_the_same_answer_on_hard_branches(self):
        for name, branch, pool, budget in self.cases():
            reference, _n = self._solve(branch, pool, budget, float('inf'))
            vectorized, _m = self._solve(branch, pool, budget, float('inf'),
                                         matrix=self._matrix_for(pool, branch))
            with self.subTest(case=name):
                self.assertEqual(reference[0], vectorized[0])
                self.assertEqual(reference[1], vectorized[1])

    def test_a_branch_above_erd_three_is_covered(self):
        """The range the old bound cannot reach, so the floor is all there is.

        `3 - (group_count + has_self) / n` is below 3.0 for every candidate by
        construction, so on a branch whose true ERD is 3.0 or more it prunes
        nothing whatsoever.  Ten near-identical words cost 3.2 each on average,
        which puts this case in that range.
        """
        branch = sorted(self.ATER[:10])
        pool = self.pool_for(self.ATER[:10], 4, 5)
        total = ExactERD(pool).solve(branch, None)[0]
        self.assertGreater(total / len(branch), 3.0)
        (status, cost, _depth, _floor), consultations = self._solve(
            branch, pool, None, float('inf'))
        self.assertEqual(status, SOLVED)
        self.assertEqual(round(cost * len(branch)), total)
        self.assertGreater(consultations, 0)


class TestMetamorphicProperties(_CaseMixin, unittest.TestCase):
    """Relations between solves that hold whatever the individual values are."""

    def test_widening_the_pool_cannot_raise_the_optimum(self):
        for size in (3, 5, 7):
            branch = self.branch(size, seed=900 + size)
            narrow = ExactERD(self.narrow_pool).erd(branch, 5)
            wide = ExactERD(self.wide_pool).erd(branch, 5)
            with self.subTest(size=size):
                self.assertLessEqual(wide, narrow)

    def test_widening_the_pool_cannot_raise_the_floor(self):
        """More words to play means finer splits are available, never fewer."""
        narrow = BranchFloorTable(self.narrow_pool,
                                  cache=ResponseCache(list(self.answers)))
        wide = BranchFloorTable(self.wide_pool,
                                cache=ResponseCache(list(self.answers)))
        rng = random.Random(55)
        for _ in range(60):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 9)))
            with self.subTest(branch=branch):
                self.assertLessEqual(wide.branch_cost_lower_bound(branch),
                                     narrow.branch_cost_lower_bound(branch))

    def test_pool_order_cannot_change_the_erd(self):
        branch = self.branch(6, seed=1001)
        shuffled = list(self.wide_pool)
        random.Random(17).shuffle(shuffled)
        first = self.engine_solve(branch, self.wide_pool, 5)[0]
        second = self.engine_solve(branch, tuple(shuffled), 5)[0]
        self.assertAlmostEqual(first, second, places=12)

    def test_pool_order_cannot_change_the_floor(self):
        shuffled = list(self.wide_pool)
        random.Random(23).shuffle(shuffled)
        ordered = BranchFloorTable(self.wide_pool,
                                   cache=ResponseCache(list(self.answers)))
        reordered = BranchFloorTable(tuple(shuffled),
                                     cache=ResponseCache(list(self.answers)))
        rng = random.Random(77)
        for _ in range(40):
            branch = sorted(rng.sample(self.answers, rng.randint(2, 9)))
            with self.subTest(branch=branch):
                self.assertEqual(ordered.branch_cost_lower_bound(branch),
                                 reordered.branch_cost_lower_bound(branch))

    def test_permuting_the_branch_cannot_change_the_erd(self):
        branch = self.branch(6, seed=1002)
        permuted = list(branch)
        random.Random(41).shuffle(permuted)
        first = self.engine_solve(branch, self.wide_pool, 5)[0]
        second = self.engine_solve(permuted, self.wide_pool, 5)[0]
        self.assertAlmostEqual(first, second, places=12)

    def test_a_reused_table_answers_as_a_fresh_one_does(self):
        """A memo shared across a solve must not accumulate an opinion.

        Reuse is the whole reason the floors are cheaper than the search they
        replace, so a table warmed by earlier branches has to give the same
        answers a table built for this branch alone would.
        """
        branches = [self.branch(size, seed=1100 + size) for size in (4, 5, 6)]
        warmed = BranchFloorTable(self.wide_pool,
                                  cache=ResponseCache(list(self.answers)))
        for branch in branches:
            self.engine_solve(branch, self.wide_pool, 5, table=warmed)
        for branch in branches:
            fresh = self.engine_solve(branch, self.wide_pool, 5)
            reused = self.engine_solve(branch, self.wide_pool, 5, table=warmed)
            with self.subTest(branch=branch):
                self.assertEqual(fresh, reused)

    def test_a_smaller_budget_cannot_lower_the_erd(self):
        """A depth cap only removes strategies; it never invents a cheaper one."""
        for size in (4, 6):
            branch = self.branch(size, seed=1200 + size)
            oracle = ExactERD(self.wide_pool)
            unlimited = oracle.erd(branch, None)
            for budget in (2, 3, 4, 5):
                capped = oracle.erd(branch, budget)
                with self.subTest(size=size, budget=budget):
                    if capped is not None:
                        self.assertGreaterEqual(capped, unlimited)


if __name__ == '__main__':
    unittest.main()
