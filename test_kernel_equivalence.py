"""§4 acceptance: the vectorized kernel (best-first sort + ERD-lower-bound
pruning, §4a/§4b) must be result-identical to the pure-Python path it replaces.

Covers full_tree_plan.md §4's acceptance list:
  - equivalence across branch sizes {8, 30, 81, 146} and budgets {None, 5, 4}:
    identical best_guess/ERD/max_remaining_depth/taint and identical rows
    written to a fresh ScoreCache.
  - invariant 2: a node where every candidate is ERD-pruned still returns a
    cutoff and writes no loss row (never a false "proven unsolvable").
  - the §4a index-alignment hazard: ERD-pruning a candidate against a
    misaligned (unpermuted) bound array changes the recorded winner/ERD.
"""
import os
import random
import tempfile
import unittest

import pattern_matrix as pattern_matrix_module
from cache_sqlite import ScoreCache
from pattern_matrix import PatternMatrix
from wordle_engine import (
    ResponseCache,
    evaluate_candidate,
    _solve_subset,
    ERD_ALL,
    SOLVED,
    OVER_ERD_LIMIT,
)

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_words(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


_ALL_GUESS_WORDS = _load_words(os.path.join(_REPO_DIR, "wordle.txt"))
_ALL_ANSWER_WORDS = _load_words(os.path.join(_REPO_DIR, "NYT_wordlist.txt"))


def _dump_cache_tables(score_cache):
    """(best_rows, loss_rows), sorted and excluding updated_at (a timestamp,
    not part of the search result), for equality comparison across runs."""
    best = score_cache._conn.execute(
        "SELECT branch_key, policy, best_guess, best_score, max_depth, solve_budget "
        "FROM branch_best_by_policy ORDER BY branch_key, policy"
    ).fetchall()
    loss = score_cache._conn.execute(
        "SELECT branch_key, policy, loss_budget "
        "FROM branch_loss_by_policy ORDER BY branch_key, policy"
    ).fetchall()
    return ([tuple(row) for row in best], [tuple(row) for row in loss])


class _TmpCacheMixin:
    def _fresh_score_cache(self, answer_words):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        sc = ScoreCache(tmp.name, answer_words)
        self.addCleanup(sc.close)
        return sc

    def _solve(self, branch_words, answer_words, guesses, budget,
               pattern_matrix, ceiling=float("inf"), heartbeat=None):
        score_cache = self._fresh_score_cache(answer_words)
        cache = ResponseCache(answer_words, score_cache)
        result = _solve_subset(
            branch_words, cache, score_cache, budget,
            deadline=None, guesses=guesses, policy=ERD_ALL,
            cancel_check=None, heartbeat=heartbeat, note_depth=None,
            progress_callback=None, subbranch_solver=None,
            ceiling=ceiling, pattern_matrix=pattern_matrix,
        )
        return result, _dump_cache_tables(score_cache)


@unittest.skipUnless(pattern_matrix_module.available(), "NumPy not available")
class TestKernelEquivalence(_TmpCacheMixin, unittest.TestCase):
    """The core §4 deliverable: matrix-on and matrix-off solves agree exactly."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(0)
        # A guess vocabulary of real 5-letter words, large enough to reach
        # ORDER_MIN_N and trigger vectorized ordering; the answer vocabulary
        # is a subset of it so every branch word is always a valid candidate
        # too (as in the real game, where the guess list is a superset of
        # the answer list).
        cls.answer_words = sorted(rng.sample(_ALL_ANSWER_WORDS, 150))
        extra_guesses = rng.sample(
            [w for w in _ALL_GUESS_WORDS if w not in set(cls.answer_words)], 150)
        cls.guess_words = sorted(set(cls.answer_words) | set(extra_guesses))
        cls.pattern_matrix = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def _branch(self, size, seed):
        rng = random.Random(seed)
        return sorted(rng.sample(self.answer_words, size))

    def test_equivalence_across_sizes_and_budgets(self):
        for size in (8, 30, 81, 146):
            branch_words = self._branch(size, seed=size)
            for budget in (None, 5, 4):
                with self.subTest(size=size, budget=budget):
                    result_on, dump_on = self._solve(
                        branch_words, self.answer_words, self.guess_words,
                        budget, self.pattern_matrix)
                    result_off, dump_off = self._solve(
                        branch_words, self.answer_words, self.guess_words,
                        budget, None)
                    self.assertEqual(result_on, result_off)
                    self.assertEqual(dump_on, dump_off)


@unittest.skipUnless(pattern_matrix_module.available(), "NumPy not available")
class TestAllCandidatesERDPrunedInvariant(_TmpCacheMixin, unittest.TestCase):
    """§4b invariant 2: a node where every candidate is ERD-pruned returns a
    cutoff, not a loss.

    3 - (n+1)/n is the lowest cost_lb any candidate can possibly achieve
    (C2.1: G <= n non-self-inclusive groups, plus has_self) — so seeding the
    ceiling below that floor (here 1.5, valid for every n >= 2) guarantees
    every candidate is ERD-pruned, regardless of branch content.
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        cls.answer_words = sorted(rng.sample(_ALL_ANSWER_WORDS, 150))
        cls.guess_words = cls.answer_words
        cls.pattern_matrix = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def test_all_candidates_erd_pruned_returns_cutoff_not_loss(self):
        branch_words = sorted(random.Random(2).sample(self.answer_words, 20))
        result, (best_rows, loss_rows) = self._solve(
            branch_words, self.answer_words, self.guess_words,
            budget=None, pattern_matrix=self.pattern_matrix, ceiling=1.5)
        status, cost, max_remaining_depth, tainted = result
        self.assertEqual(status, OVER_ERD_LIMIT)
        self.assertEqual(cost, 1.5)
        # The single worst bug §4b could introduce: a node where every
        # candidate is ERD-pruned falling through to the "proven unsolvable"
        # branch and writing a false loss row instead of reporting a cutoff.
        self.assertEqual(loss_rows, [])
        self.assertEqual(best_rows, [])

    def test_heartbeat_still_ticks_once_per_erd_pruned_candidate(self):
        # evaluate_candidate's own heartbeat call is the swarm's node counter
        # (erd_swarm.py's _BranchWorker._nodes), documented to count every
        # candidate considered, ERD-pruned or not. ERD-pruning must not
        # silently skip it, or nodes_spent telemetry undercounts whenever a
        # PatternMatrix is active. Total ticks = one from _solve_subset's own
        # node-entry heartbeat, plus one per candidate in the loop (all of
        # them ERD-pruned here, per this class's docstring).
        branch_words = sorted(random.Random(2).sample(self.answer_words, 20))
        ticks = [0]
        self._solve(
            branch_words, self.answer_words, self.guess_words,
            budget=None, pattern_matrix=self.pattern_matrix, ceiling=1.5,
            heartbeat=lambda: ticks.__setitem__(0, ticks[0] + 1))
        self.assertEqual(ticks[0], 1 + len(self.guess_words))


@unittest.skipUnless(pattern_matrix_module.available(), "NumPy not available")
class TestIndexAlignmentHazard(_TmpCacheMixin, unittest.TestCase):
    """§4a: the sort-key and cost-lower-bound arrays must be permuted by BOTH
    candidate_rows and the sort order. This fixture makes a candidate's raw
    vocabulary position (weak first, strong second) diverge from its
    best-first position (strong first, weak second) enough that ERD-pruning
    either candidate against the OTHER's bound flips the outcome from a
    concrete SOLVED winner to a cutoff — an unpermuted or half-permuted
    bound array cannot pass this test by accident.
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(3)
        cls.answer_words = sorted(rng.sample(_ALL_ANSWER_WORDS, 150))
        cls.branch_words = sorted(rng.sample(cls.answer_words, 8))
        pool = rng.sample(_ALL_GUESS_WORDS, 80)
        cls.guess_words = sorted(set(cls.answer_words) | set(pool) | set(cls.branch_words))
        cls.pattern_matrix = PatternMatrix.build(cls.guess_words, cls.answer_words)

        probe_cache = ResponseCache(cls.answer_words)
        scored = []
        for word in pool:
            counts = probe_cache.group_counts(word, cls.branch_words)
            has_self = word in cls.branch_words
            cost_lb = 3.0 - (len(counts) + (1 if has_self else 0)) / len(cls.branch_words)
            scored.append((cost_lb, word))
        scored.sort(key=lambda item: item[0])
        cls.strong = scored[0][1]
        cls.weak = scored[-1][1]
        weak_cost_lb = scored[-1][0]

        status, strong_cost, _, _ = evaluate_candidate(
            cls.branch_words, cls.strong, probe_cache, None,
            best_erd=float("inf"))
        if status != SOLVED:
            raise RuntimeError("fixture precondition: strong must be solvable")
        cls.strong_cost = strong_cost
        cls.ceiling = strong_cost + 1e-6
        if weak_cost_lb < cls.ceiling:
            raise RuntimeError(
                "fixture precondition failed: weak's cost_lb must dominate "
                "the ceiling seeded from strong's true cost — widen the "
                "candidate pool if this ever breaks")
        # candidate_list in raw (pre-sort) order: weak first, strong last —
        # the opposite of best-first order, which is exactly what an
        # unpermuted or half-permuted bound array would misalign.
        cls.candidate_list = [cls.weak, cls.strong]

    def test_misaligned_bound_would_change_the_winner(self):
        result, _ = self._solve(
            self.branch_words, self.answer_words, self.candidate_list,
            budget=None, pattern_matrix=self.pattern_matrix,
            ceiling=self.ceiling)
        status, cost, _max_remaining_depth, _tainted = result
        # Correct alignment: strong (sorted first) evaluates under its own
        # low bound, solves, and wins; weak (sorted second) is ERD-pruned by
        # the tightened best_erd. A misaligned array would instead swap which
        # candidate is ERD-pruned at which position, wrongly dropping strong
        # before it ever evaluates and reporting a cutoff instead of a
        # concrete winner (see the module docstring's full trace).
        self.assertEqual(status, SOLVED)
        self.assertEqual(cost, self.strong_cost)


if __name__ == "__main__":
    unittest.main()
