"""
Tests for Wordle engine correctness and cache behavior.
Run with:  python -m unittest discover -s tests -t . -p 'test_*.py'
"""
import io
import itertools
import math
import os
import platform
import re
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordle_engine import (
    Solution, ScoringMethod, ResponseCache, GuessUniverse, ComplianceFilter,
    calculate_response, score_groups, score_groups_multi, calculate_group_counts,
    score_word, score_word_multi, load_word_list, max_entropy,
    Restriction, answer_to_restriction, apply_guess, _encode_response,
    decode_response, _ALL_GREEN_PATTERN,
    min_expected_guesses, ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED,
    ERD_ANSWERS_UNFILTERED, cache_all_scores, verify_erd_cache,
    enumerate_branches, rank_candidates_by_max_group_size_then_entropy_gain, _cache_reuse,
    _solve_subset, max_solvable_within,
)
from cache_sqlite import ScoreCache, MemoryScoreCache
from wordle import (
    _multistep_stats, _erd_solve_scores, ERDSolver,
    _compare_words, set_display_context,
    BranchPrecacheSolver, format_response, print_status,
    _format_cache_timestamp, _current_candidate_tag,
    _format_scan_progress, _format_branch_header, _platform_label,
    print_line_with_pattern,
    colored_text, reset_color, print_error, print_success,
    print_colored_pattern, print_colored_word, ANSI_COLORS, ANSI_RESET,
    mark, render_markup, MARK_RESET, MARK_RED, MARK_GREEN, MARK_YELLOW,
    MARK_GRAY,
)


# Small deterministic word sets used across all tests.
ANSWERS = ["crane", "slate", "trace", "stale", "tales",
           "least", "heart", "earth", "share", "rates"]
GUESSES = ANSWERS + ["brain", "stove", "cloud", "piano", "train"]


def make_solution(db_path=None):
    sc = ScoreCache(db_path, ANSWERS) if db_path else None
    cache = ResponseCache(ANSWERS, score_cache=sc)
    return Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)


# ---------------------------------------------------------------------------
# calculate_response
# ---------------------------------------------------------------------------

class TestCalculateResponse(unittest.TestCase):

    def test_all_green(self):
        self.assertEqual(calculate_response("crane", "crane"),
                         ["green", "green", "green", "green", "green"])

    def test_all_gray(self):
        # logic / bumpy share no letters
        self.assertEqual(calculate_response("logic", "bumpy"),
                         ["gray", "gray", "gray", "gray", "gray"])

    def test_all_yellow(self):
        # RATES vs STARE: every letter present, none in correct position
        self.assertEqual(calculate_response("rates", "stare"),
                         ["yellow", "yellow", "yellow", "yellow", "yellow"])

    def test_mixed(self):
        # CRANE vs SLANT: A(2) and N(3) green, C R E gray
        self.assertEqual(calculate_response("crane", "slant"),
                         ["gray", "gray", "green", "green", "gray"])

    def test_yellow_and_green(self):
        # CRANE vs TRACE: C yellow, R green, A green, N gray, E green
        self.assertEqual(calculate_response("crane", "trace"),
                         ["yellow", "green", "green", "gray", "green"])

    def test_duplicate_letter_in_guess(self):
        # STEEL vs SLATE: only one E in SLATE (pos 4)
        # S(0)green, T(1)yellow, E(2)yellow, E(3)gray(E consumed), L(4)yellow
        self.assertEqual(calculate_response("steel", "slate"),
                         ["green", "yellow", "yellow", "gray", "yellow"])


# ---------------------------------------------------------------------------
# score_groups
# ---------------------------------------------------------------------------

class TestScoreGroups(unittest.TestCase):

    def test_entropy_uniform_4_groups(self):
        # 4 equal groups of 1: H = log2(4) = 2.0
        groups = {i: 1 for i in range(4)}
        self.assertAlmostEqual(
            score_groups(groups, ScoringMethod.ENTROPY_GAIN), 2.0)

    def test_entropy_one_group_is_zero(self):
        # Everything in one group → no information gained
        self.assertAlmostEqual(
            score_groups({0: 5}, ScoringMethod.ENTROPY_GAIN), 0.0)

    def test_entropy_two_equal_groups(self):
        # 2 groups of 2: H = 1.0
        self.assertAlmostEqual(
            score_groups({0: 2, 1: 2}, ScoringMethod.ENTROPY_GAIN), 1.0)

    def test_entropy_known_unequal(self):
        # groups of sizes 4, 2, 2: n=8
        groups = {0: 4, 1: 2, 2: 2}
        expected = -(4/8)*math.log2(4/8) - 2*(2/8)*math.log2(2/8)
        self.assertAlmostEqual(
            score_groups(groups, ScoringMethod.ENTROPY_GAIN), expected)

    def test_max_group_size(self):
        groups = {0: 5, 1: 3, 2: 7, 3: 2}
        self.assertEqual(score_groups(groups, ScoringMethod.MAX_GROUP_SIZE), 7)

    def test_weighted_avg(self):
        # sum(k^2) / N where N = sum(k)
        groups = {0: 3, 1: 2}   # N=5
        self.assertAlmostEqual(
            score_groups(groups, ScoringMethod.WEIGHTED_AVG), (9 + 4) / 5)

    def test_prob_finish(self):
        # 2 singletons out of 5 words total
        groups = {0: 1, 1: 1, 2: 3}
        self.assertAlmostEqual(
            score_groups(groups, ScoringMethod.PROB_FINISH), 2 / 5)

    def test_empty_groups_higher_is_better_returns_zero(self):
        self.assertEqual(score_groups({}, ScoringMethod.ENTROPY_GAIN), 0.0)
        self.assertEqual(score_groups({}, ScoringMethod.PROB_FINISH), 0.0)

    def test_empty_groups_lower_is_better_returns_infinity(self):
        self.assertEqual(score_groups({}, ScoringMethod.WEIGHTED_AVG), float('inf'))
        self.assertEqual(score_groups({}, ScoringMethod.MAX_GROUP_SIZE), float('inf'))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            score_groups({0: 1}, "not-a-scoring-method")


# ---------------------------------------------------------------------------
# score_word / score_word_multi: cache vs. no-cache agreement, callbacks
# ---------------------------------------------------------------------------

class TestScoreWordCacheAgreement(unittest.TestCase):
    """score_word/score_word_multi take an optional ResponseCache: with one,
    they read pre-built pattern mappings; without one, they fall back to
    calculate_group_counts. Both paths must produce identical scores."""

    def test_score_word_with_and_without_cache_agree(self):
        cache = ResponseCache(ANSWERS)
        with_cache = score_word("crane", ANSWERS, ScoringMethod.ENTROPY_GAIN, cache=cache)
        without_cache = score_word("crane", ANSWERS, ScoringMethod.ENTROPY_GAIN)
        self.assertAlmostEqual(with_cache, without_cache)

    def test_score_word_multi_with_and_without_cache_agree(self):
        cache = ResponseCache(ANSWERS)
        methods = list(ScoringMethod)
        with_cache = score_word_multi("crane", ANSWERS, methods, cache=cache)
        without_cache = score_word_multi("crane", ANSWERS, methods)
        self.assertEqual(with_cache, without_cache)

    def test_score_word_invokes_progress_callback(self):
        calls = []
        score_word("crane", ANSWERS, ScoringMethod.ENTROPY_GAIN,
                   progress_callback=lambda: calls.append(1))
        self.assertEqual(calls, [1])

    def test_score_word_multi_invokes_progress_callback(self):
        calls = []
        score_word_multi("crane", ANSWERS, [ScoringMethod.ENTROPY_GAIN],
                         progress_callback=lambda: calls.append(1))
        self.assertEqual(calls, [1])


# ---------------------------------------------------------------------------
# ResponseCache: is_cached and the not-yet-mapped fallback branch
# ---------------------------------------------------------------------------

class TestResponseCacheMembership(unittest.TestCase):

    def test_is_cached_reflects_first_use(self):
        cache = ResponseCache(ANSWERS)
        self.assertFalse(cache.is_cached("crane"))
        cache.group_counts("crane", ANSWERS[:3])
        self.assertTrue(cache.is_cached("crane"))

    def test_group_counts_handles_word_outside_answer_universe(self):
        """A subset word that wasn't in answer_words at cache-build time
        (e.g. a guess-only word reachable via fallback mode) isn't in the
        pre-built mapping — group_counts/group_words must still score it
        correctly via the inline calculate_response fallback."""
        small_universe = ANSWERS[:3]
        cache = ResponseCache(small_universe)
        outsider = "piano"
        subset = small_universe + [outsider]

        counts = cache.group_counts("crane", subset)
        groups = cache.group_words("crane", subset)

        expected_counts = calculate_group_counts("crane", subset)
        self.assertEqual(dict(counts), dict(expected_counts))
        self.assertEqual(sum(len(g) for g in groups.values()), len(subset))
        self.assertTrue(any(outsider in g for g in groups.values()))


# ---------------------------------------------------------------------------
# Engine utility functions: load_word_list, calculate_group_counts,
# max_entropy, Restriction, ScoringMethod display helpers
# ---------------------------------------------------------------------------

class TestLoadWordList(unittest.TestCase):

    def test_strips_blank_lines_and_whitespace(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write("crane\n  slate  \n\ntrace\n")
            path = f.name
        try:
            self.assertEqual(load_word_list(path), ["crane", "slate", "trace"])
        finally:
            os.unlink(path)


class TestCalculateGroupCounts(unittest.TestCase):

    def test_matches_per_word_calculate_response(self):
        words = ANSWERS[:6]
        counts = calculate_group_counts("crane", words)
        expected = defaultdict(int)
        for w in words:
            expected[_encode_response(calculate_response("crane", w))] += 1
        self.assertEqual(dict(counts), dict(expected))


class TestMaxEntropy(unittest.TestCase):

    def test_zero_or_one_remaining_is_zero(self):
        self.assertEqual(max_entropy(0), 0.0)
        self.assertEqual(max_entropy(1), 0.0)

    def test_matches_log2(self):
        self.assertAlmostEqual(max_entropy(8), 3.0)
        self.assertAlmostEqual(max_entropy(16), 4.0)


class TestRestriction(unittest.TestCase):

    def test_setitem_getitem_round_trip(self):
        r = Restriction()
        r[2] = ('a', 1, 'yellow')
        self.assertEqual(r[2], ['a', 1, 'yellow'])
        # Untouched slots remain at their defaults
        self.assertEqual(r[0], ['', 0, None])

    def test_apply_matches_apply_guess(self):
        guess, answer = "crane", "trace"
        response = calculate_response(guess, answer)
        restriction = answer_to_restriction(guess, response)
        via_restriction = restriction.apply(ANSWERS)
        via_apply_guess = apply_guess(ANSWERS, guess, response)
        self.assertEqual(via_restriction, via_apply_guess)

    def test_apply_handles_repeated_letters_in_guess(self):
        """A guess with a repeated letter ('sassy' has three S's) forces
        _ignore_word's duplicate-accounting to actually mask out earlier
        same-letter occurrences. Restriction-based filtering must agree
        with an independent per-word calculate_response comparison —
        two algorithms that should always produce the same subset."""
        guess = "sassy"
        candidates = ANSWERS + ["essay", "spans", "lasso"]
        for answer in candidates:
            response = calculate_response(guess, answer)
            filtered = apply_guess(candidates, guess, response)
            expected = [w for w in candidates
                        if calculate_response(guess, w) == response]
            self.assertEqual(filtered, expected, msg=f"answer={answer}")


class TestEnumerateBranches(unittest.TestCase):

    def test_partition_covers_all_words(self):
        words = ANSWERS
        guess_word = "heart"
        branches = enumerate_branches(words, guess_word)

        codes = [code for code, _ in branches]
        self.assertNotIn(_ALL_GREEN_PATTERN, codes)
        self.assertEqual(len(codes), len(set(codes)))

        for code, branch_words in branches:
            self.assertGreaterEqual(len(branch_words), 2)
            self.assertEqual(
                branch_words,
                apply_guess(words, guess_word, decode_response(code)))

        # Branches plus the excluded (<2-word, including the win) groups
        # exactly partition the original word list.
        branch_total = sum(len(bw) for _, bw in branches)
        excluded_total = sum(
            len(apply_guess(words, guess_word, decode_response(code)))
            for code in range(243) if code not in codes)
        self.assertEqual(branch_total + excluded_total, len(words))


class TestScoringMethodDisplay(unittest.TestCase):

    def test_label_is_distinct_per_method(self):
        labels = {m.label for m in ScoringMethod}
        self.assertEqual(len(labels), len(list(ScoringMethod)))

    def test_format_score_max_group_size_is_integer(self):
        self.assertEqual(ScoringMethod.MAX_GROUP_SIZE.format_score(7.0), "7")

    def test_format_score_prob_finish_is_percentage(self):
        self.assertEqual(ScoringMethod.PROB_FINISH.format_score(0.25), "25.00%")

    def test_format_score_default_is_four_decimals(self):
        self.assertEqual(ScoringMethod.ENTROPY_GAIN.format_score(1.5), "1.5000")


# ---------------------------------------------------------------------------
# Solution in-memory score cache lifecycle
# ---------------------------------------------------------------------------

class TestSolutionScoreCache(unittest.TestCase):

    def setUp(self):
        self.soln = make_solution()

    def test_word_scores_empty_before_compute(self):
        self.assertEqual(self.soln.word_scores, {})

    def test_word_scores_populated_after_compute(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        for w in GUESSES:
            self.assertIn(w, self.soln.word_scores)
            self.assertIn(ScoringMethod.ENTROPY_GAIN, self.soln.word_scores[w])

    def test_compute_scores_multi_populates_both_methods(self):
        methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MAX_GROUP_SIZE]
        self.soln.compute_scores_multi(GUESSES, methods)
        for w in GUESSES:
            for m in methods:
                self.assertIn(m, self.soln.word_scores.get(w, {}))

    def test_second_compute_uses_cache(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        calls = [0]
        def count(x=None): calls[0] += 1
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN,
                                 progress_callback=count)
        # All calls should be cache hits — same count as number of words
        self.assertEqual(calls[0], len(GUESSES))

    def test_second_method_extends_existing_word_entry(self):
        """Scoring under a second method must add to a word's existing
        word_scores entry rather than replace it — both methods' results
        stay available from the same cache."""
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        self.soln.compute_scores(GUESSES, ScoringMethod.MAX_GROUP_SIZE)
        for w in GUESSES:
            entry = self.soln.word_scores[w]
            self.assertIn(ScoringMethod.ENTROPY_GAIN, entry)
            self.assertIn(ScoringMethod.MAX_GROUP_SIZE, entry)

    def test_compute_scores_multi_second_call_is_pure_cache_hits(self):
        methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MAX_GROUP_SIZE]
        self.soln.compute_scores_multi(GUESSES, methods)
        calls = [0]
        def count(x=None): calls[0] += 1
        self.soln.compute_scores_multi(GUESSES, methods, progress_callback=count)
        self.assertEqual(calls[0], len(GUESSES))

    def test_invalidate_clears_word_scores(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        self.soln._invalidate_scores()
        self.assertEqual(self.soln.word_scores, {})
        self.assertEqual(self.soln._db_loaded_methods, set())

    def test_is_full_game_initial(self):
        self.assertTrue(self.soln._is_full_game())

    def test_is_full_game_after_guess(self):
        pattern = calculate_response("crane", "slate")
        self.soln.apply_guess("crane", pattern)
        self.assertFalse(self.soln._is_full_game())

    def test_is_full_game_after_reset(self):
        pattern = calculate_response("crane", "slate")
        self.soln.apply_guess("crane", pattern)
        self.soln.reset()
        self.assertTrue(self.soln._is_full_game())

    def test_scores_updated_flag(self):
        self.assertFalse(self.soln.scores_updated)
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        self.assertTrue(self.soln.scores_updated)
        self.soln._invalidate_scores()
        self.assertFalse(self.soln.scores_updated)

    def test_scores_sorted_entropy_descending(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        scores = [s for _, s in self.soln.scores]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_sorted_max_group_size_ascending(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.MAX_GROUP_SIZE)
        scores = [s for _, s in self.soln.scores]
        self.assertEqual(scores, sorted(scores))


# ---------------------------------------------------------------------------
# Solution: undo_guess, include_letters/exclude_letters, join
# ---------------------------------------------------------------------------

class TestUndoGuess(unittest.TestCase):

    def setUp(self):
        self.soln = make_solution()

    def test_no_op_on_empty_history(self):
        self.assertFalse(self.soln.undo_guess())
        self.assertEqual(self.soln.current_words, ANSWERS)

    def test_returns_true_and_restores_prior_word_count(self):
        before = self.soln.current_words[:]
        pattern = calculate_response("crane", "slate")
        self.soln.apply_guess("crane", pattern)
        self.assertNotEqual(self.soln.current_words, before)

        self.assertTrue(self.soln.undo_guess())
        self.assertEqual(self.soln.current_words, before)
        self.assertEqual(self.soln.guesses, [])

    def test_undo_replays_remaining_history(self):
        """Undoing the most recent of several guesses must leave current_words
        exactly as if only the earlier guesses had ever been applied."""
        p1 = calculate_response("crane", "heart")
        p2 = calculate_response("slate", "heart")
        self.soln.apply_guess("crane", p1)
        after_first = self.soln.current_words[:]
        self.soln.apply_guess("slate", p2)

        self.assertTrue(self.soln.undo_guess())
        self.assertEqual(self.soln.current_words, after_first)
        self.assertEqual(self.soln.guesses, [["crane", list(p1)]])

    def test_undo_clears_stale_score_cache(self):
        pattern = calculate_response("crane", "slate")
        self.soln.apply_guess("crane", pattern)
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        self.assertTrue(self.soln.word_scores)

        self.soln.undo_guess()
        self.assertEqual(self.soln.word_scores, {},
                         "stale per-word scores from the undone position must "
                         "not survive — they were computed against a different "
                         "remaining-word set")

    def test_undo_resets_fallback_flag(self):
        # Drive current_words to empty so the all_words fallback engages.
        soln = Solution(["crane"], GUESSES, cache=ResponseCache(["crane"]))
        pattern = calculate_response("slate", "heart")  # crane can't match this
        soln.apply_guess("slate", pattern)
        self.assertTrue(soln.fallback_active)

        soln.undo_guess()
        self.assertFalse(soln.fallback_active)


class TestIncludeExcludeLetters(unittest.TestCase):

    def setUp(self):
        self.soln = make_solution()

    def test_include_keeps_only_words_with_all_letters(self):
        self.soln.include_letters("ea")
        for w in self.soln.current_words:
            self.assertIn("e", w)
            self.assertIn("a", w)
        self.assertTrue(self.soln.current_words)

    def test_exclude_removes_words_with_any_letter(self):
        self.soln.exclude_letters("cs")
        for w in self.soln.current_words:
            self.assertNotIn("c", w)
            self.assertNotIn("s", w)
        self.assertTrue(self.soln.current_words)

    def test_include_then_exclude_compose(self):
        self.soln.include_letters("a")
        self.soln.exclude_letters("e")
        for w in self.soln.current_words:
            self.assertIn("a", w)
            self.assertNotIn("e", w)

    def test_filters_invalidate_score_cache(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        self.assertTrue(self.soln.word_scores)
        self.soln.include_letters("e")
        self.assertEqual(self.soln.word_scores, {})


class TestSolutionJoin(unittest.TestCase):

    def _soln(self, words):
        return Solution(ANSWERS, GUESSES, cache=ResponseCache(ANSWERS))

    def test_returns_none_for_empty_list(self):
        self.assertIsNone(Solution.join([]))

    def test_combines_unsolved_boards_only(self):
        s1 = self._soln(ANSWERS)
        s2 = self._soln(ANSWERS)
        s3 = self._soln(ANSWERS)

        s1.current_words = ["crane", "slate", "trace"]
        s2.current_words = ["slate", "stale", "tales"]
        s3.current_words = ["heart"]  # solved — single candidate, excluded

        merged = Solution.join([s1, s2, s3])
        self.assertEqual(merged.current_words,
                         sorted({"crane", "slate", "trace", "stale", "tales"}))
        self.assertNotIn("heart", merged.current_words)

    def test_join_carries_forward_first_solutions_caches(self):
        s1 = self._soln(ANSWERS)
        s2 = self._soln(ANSWERS)
        s1.current_words = ["crane", "slate"]
        s2.current_words = ["trace", "stale"]

        merged = Solution.join([s1, s2])
        self.assertIs(merged.all_answers, s1.all_answers)
        self.assertIs(merged.all_words, s1.all_words)
        self.assertIs(merged.cache, s1.cache)
        self.assertIs(merged.score_cache, s1.score_cache)


# ---------------------------------------------------------------------------
# ScoreCache SQLite round-trips
# ---------------------------------------------------------------------------

class TestScoreCacheSQLite(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_word_scores_round_trip(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write_scores(branch_key, [("crane", 3.14159), ("slate", 2.71828)],
                        "entropy_gain")
        result = dict(sc.read_scores(branch_key, "entropy_gain"))
        self.assertAlmostEqual(result["crane"], 3.14159)
        self.assertAlmostEqual(result["slate"], 2.71828)

    def test_word_scores_are_subset_scoped(self):
        sc = ScoreCache(self.db, ANSWERS)
        key1 = ScoreCache.encode_subset(["crane", "slate"])
        key2 = ScoreCache.encode_subset(["heart", "earth"])
        sc.write_scores(key1, [("brain", 1.0)], "entropy_gain")
        self.assertIsNone(sc.read_scores(key2, "entropy_gain"),
                          "scores cached for one remaining-word subset must "
                          "not leak into a lookup for a different subset")

    def test_subgroup_round_trip(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write(branch_key, "full", "heart", 2.5)
        word, ent = sc.read(branch_key, "full")
        self.assertEqual(word, "heart")
        self.assertAlmostEqual(ent, 2.5)

    def test_write_populates_mem_cache(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write(branch_key, "full", "heart", 2.5)
        self.assertEqual(sc._mem_cache[(branch_key, "full")],
                         ("heart", 2.5, None, None))

        # Closing the connection proves this read is served from memory,
        # not a fresh SQLite round trip.
        sc._conn.close()
        self.assertEqual(sc.read(branch_key, "full"), ("heart", 2.5))

    def test_read_hit_populates_mem_cache(self):
        sc1 = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        sc1.write(branch_key, "hard", "earth", 1.8)
        sc1.close()

        sc2 = ScoreCache(self.db, ANSWERS)
        first = sc2.read(branch_key, "hard")
        self.assertEqual(first, ("earth", 1.8))
        self.assertEqual(sc2._mem_cache[(branch_key, "hard")],
                         ("earth", 1.8, None, None))

        # A second read must not touch SQLite at all.
        sc2._conn.close()
        self.assertEqual(sc2.read(branch_key, "hard"), ("earth", 1.8))

    def test_read_miss_returns_none(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        self.assertIsNone(sc.read(branch_key, "full"))
        self.assertIsNone(sc.read_scores(branch_key, "entropy_gain"))

    def test_policy_separation(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write(branch_key, "full", "heart", 2.5)
        sc.write(branch_key, "hard", "earth", 1.8)
        self.assertEqual(sc.read(branch_key, "full")[0], "heart")
        self.assertEqual(sc.read(branch_key, "hard")[0], "earth")

    def test_different_universe_no_cross_contamination(self):
        alt_answers = ["brain", "stove", "cloud"]
        sc1 = ScoreCache(self.db, ANSWERS)
        sc2 = ScoreCache(self.db, alt_answers)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        sc1.write_scores(branch_key, [("crane", 3.14)], "entropy_gain")
        self.assertIsNone(sc2.read_scores(branch_key, "entropy_gain"))

    def test_overwrite_replaces_value(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write_scores(branch_key, [("crane", 1.0)], "entropy_gain")
        sc.write_scores(branch_key, [("crane", 9.9)], "entropy_gain")
        result = dict(sc.read_scores(branch_key, "entropy_gain"))
        self.assertAlmostEqual(result["crane"], 9.9)

    def test_encode_subset_is_compact(self):
        # Key length = 5 * number of words, no separators
        words = ["crane", "slate", "trace"]
        key = ScoreCache.encode_subset(words)
        self.assertEqual(len(key), 15)
        self.assertNotIn(b"\x00", key)

    def test_checkpoint_truncates_wal_and_persists_writes(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write(branch_key, "full", "heart", 2.5)
        sc.checkpoint()

        wal_path = self.db + "-wal"
        self.assertTrue(
            not os.path.exists(wal_path) or os.path.getsize(wal_path) == 0,
            "checkpoint should fold the WAL into the main db file")

        # A fresh connection (simulating a copy of just the .sqlite3 file)
        # must see the checkpointed write.
        sc2 = ScoreCache(self.db, ANSWERS)
        try:
            self.assertEqual(sc2.read(branch_key, "full"), ("heart", 2.5))
        finally:
            sc2.close()
        sc.close()

    def test_close_checkpoints(self):
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write(branch_key, "full", "heart", 2.5)
        sc.close()

        wal_path = self.db + "-wal"
        self.assertTrue(
            not os.path.exists(wal_path) or os.path.getsize(wal_path) == 0,
            "close() should leave a self-contained .sqlite3 file")

    def test_checkpoint_swallows_disk_io_error(self):
        """A transient OperationalError (e.g. iCloud File Provider Storage
        holding the lock TRUNCATE needs) must not propagate — the WAL still
        has every committed write, so a failed checkpoint loses nothing."""
        sc = ScoreCache(self.db, ANSWERS)

        class FailingCheckpoint:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if sql.startswith("PRAGMA wal_checkpoint"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn = sc._conn
        sc._conn = FailingCheckpoint(real_conn)
        try:
            sc.checkpoint()  # must not raise
        finally:
            sc._conn = real_conn
        sc.close()

    def test_write_swallows_disk_io_error(self):
        """A transient 'disk I/O error' from the INSERT must not propagate —
        it would unwind every enclosing min_expected_guesses recursion and
        abort the calling background-solver thread (see ScoreCache.write)."""
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate", "trace"])

        class FailingWrite:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if sql.lstrip().startswith("INSERT OR REPLACE INTO branch_best_by_policy"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn = sc._conn
        sc._conn = FailingWrite(real_conn)
        try:
            sc.write(branch_key, "full", "heart", 2.5)  # must not raise
        finally:
            sc._conn = real_conn

        self.assertIsNone(sc.read_detail(branch_key, "full"),
                          "failed write must not be persisted")
        self.assertEqual(sc._mem_cache[(branch_key, "full")],
                          ("heart", 2.5, None, None),
                          "result must still be memoized for this run")
        sc.close()

    def test_write_decomposition_swallows_disk_io_error(self):
        """Same as test_write_swallows_disk_io_error, for the
        response_decomposition cache populated via ResponseCache._ensure
        inside min_expected_guesses recursion."""
        sc = ScoreCache(self.db, ANSWERS)

        class FailingWrite:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if sql.lstrip().startswith("INSERT OR REPLACE INTO response_decomposition"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn = sc._conn
        sc._conn = FailingWrite(real_conn)
        try:
            sc.write_decomposition("crane", bytes([1] * len(ANSWERS)))  # must not raise
        finally:
            sc._conn = real_conn

        self.assertIsNone(sc.read_decomposition("crane"))
        sc.close()

    def test_write_scores_swallows_disk_io_error(self):
        """Same as test_write_swallows_disk_io_error, for the word_scores
        cache populated by cache_all_scores inside min_expected_guesses
        recursion."""
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])

        class FailingExecuteMany:
            def __init__(self, real):
                self._real = real

            def executemany(self, *a, **k):
                raise sqlite3.OperationalError("disk I/O error")

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn = sc._conn
        sc._conn = FailingExecuteMany(real_conn)
        try:
            sc.write_scores(branch_key, [("crane", 1.0)], "entropy_gain")  # must not raise
        finally:
            sc._conn = real_conn

        self.assertIsNone(sc.read_scores(branch_key, "entropy_gain"))
        sc.close()

    def test_close_releases_connection_when_checkpoint_fails(self):
        """close() must still release the connection even when its
        checkpoint() call hits a disk I/O error."""
        sc = ScoreCache(self.db, ANSWERS)

        class FailingCheckpoint:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if sql.startswith("PRAGMA wal_checkpoint"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._real.execute(sql, *a, **k)

            def close(self):
                self._real.close()

            def __getattr__(self, name):
                return getattr(self._real, name)

        sc._conn = FailingCheckpoint(sc._conn)
        sc.close()  # must not raise

        with self.assertRaises(sqlite3.ProgrammingError):
            sc.read(ScoreCache.encode_subset(["crane"]), "full")

    def test_encode_subset_is_order_independent(self):
        self.assertEqual(
            ScoreCache.encode_subset(["slate", "crane"]),
            ScoreCache.encode_subset(["crane", "slate"]),
        )

    def test_old_null_separated_entries_are_dropped(self):
        # Simulate a row written under BOTH legacy forms at once: the old
        # null-separated subset-key encoding, and the pre-rename
        # lookahead_result/best_word/best_entropy table+column names —
        # genuinely ancient data would carry both, since the encoding
        # predates the table/column rename.
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lookahead_result (
                subset_key BLOB NOT NULL, policy TEXT NOT NULL,
                universe_id TEXT NOT NULL, best_word TEXT NOT NULL,
                best_entropy REAL NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_key, policy, universe_id)
            )
        """)
        old_key = b"crane\x00slate"
        conn.execute(
            "INSERT OR REPLACE INTO lookahead_result VALUES (?,?,?,?,?,?)",
            (old_key, "full", "test_universe", "heart", 2.5, 0),
        )
        conn.commit()
        conn.close()

        # Opening ScoreCache should migrate through the full rename chain
        # (lookahead_result -> subgroup_best_by_policy -> branch_best_by_policy,
        # branch_key -> branch_key, best_word -> best_guess) AND delete the
        # old null-separated-key row.
        ScoreCache(self.db, ANSWERS)
        conn2 = _sqlite3.connect(self.db)
        rows = conn2.execute(
            "SELECT COUNT(*) FROM branch_best_by_policy "
            "WHERE instr(branch_key, char(0)) > 0"
        ).fetchone()[0]
        conn2.close()
        self.assertEqual(rows, 0)

    def test_old_lookahead_result_table_is_renamed(self):
        # Simulate a row persisted under the original pre-rename table/column
        # names, WITHOUT the legacy key encoding — i.e. data written between
        # the subset-key-encoding migration and the table/column rename.
        # Built entirely without a ScoreCache so neither subgroup_pick nor
        # branch_best_by_policy exists yet — otherwise the rename-in-place
        # path wouldn't trigger.
        import hashlib as _hashlib
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lookahead_result (
                subset_key BLOB NOT NULL, policy TEXT NOT NULL,
                universe_id TEXT NOT NULL, best_word TEXT NOT NULL,
                best_entropy REAL NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_key, policy, universe_id)
            )
        """)
        universe_id = _hashlib.sha256("\n".join(ANSWERS).encode()).hexdigest()
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        conn.execute(
            "INSERT OR REPLACE INTO lookahead_result VALUES (?,?,?,?,?,?)",
            (branch_key, "hard", universe_id, "heart", 2.5, 0),
        )
        conn.commit()
        conn.close()

        sc = ScoreCache(self.db, ANSWERS)
        hit = sc.read(branch_key, "hard")
        self.assertIsNotNone(hit)
        self.assertEqual(hit, ("heart", 2.5))

    def test_old_subgroup_pick_table_is_renamed(self):
        # Simulate a row persisted under the short-lived intermediate name
        # (subgroup_pick/picked_word/picked_score) — a real possible state
        # for any DB created against the commit that introduced it before
        # this rename landed. Built without a ScoreCache so
        # subgroup_best_by_policy doesn't exist yet.
        import hashlib as _hashlib
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subgroup_pick (
                subset_key BLOB NOT NULL, policy TEXT NOT NULL,
                universe_id TEXT NOT NULL, picked_word TEXT NOT NULL,
                picked_score REAL NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_key, policy, universe_id)
            )
        """)
        universe_id = _hashlib.sha256("\n".join(ANSWERS).encode()).hexdigest()
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        conn.execute(
            "INSERT OR REPLACE INTO subgroup_pick VALUES (?,?,?,?,?,?)",
            (branch_key, "hard", universe_id, "heart", 2.5, 0),
        )
        conn.commit()
        conn.close()

        sc = ScoreCache(self.db, ANSWERS)
        hit = sc.read(branch_key, "hard")
        self.assertIsNotNone(hit)
        self.assertEqual(hit, ("heart", 2.5))

    def test_old_minimax_method_key_is_migrated(self):
        # Simulate rows persisted under the pre-rename method key.
        # Compute answer_list_id directly (same formula as ScoreCache._ensure_answer_list)
        # so no ScoreCache open marks the migration done before we insert legacy data.
        import sqlite3 as _sqlite3
        import hashlib as _hashlib
        answer_list_id = _hashlib.sha256(
            "\n".join(ANSWERS).encode()
        ).hexdigest()
        conn = _sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS word_scores (
                subset_hash TEXT NOT NULL, word TEXT NOT NULL,
                method TEXT NOT NULL, score REAL NOT NULL,
                answer_list_id TEXT NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, answer_list_id, word)
            )
        """)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        subset_hash = ScoreCache._subset_hash(branch_key)
        conn.execute(
            "INSERT OR REPLACE INTO word_scores VALUES (?,?,?,?,?,?)",
            (subset_hash, "crane", "minimax", 4.0, answer_list_id, 0),
        )
        conn.commit()
        conn.close()

        # Re-opening ScoreCache should migrate the old method key forward —
        # the data must remain reachable under its new name.
        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read_scores(branch_key, "minimax"))
        migrated = sc.read_scores(branch_key, "max_group_size")
        self.assertIsNotNone(migrated)
        self.assertEqual(dict(migrated)["crane"], 4.0)

    def test_decomposition_round_trip(self):
        sc = ScoreCache(self.db, ANSWERS)
        patterns = bytes(range(len(ANSWERS)))
        sc.write_decomposition("crane", patterns)
        self.assertEqual(sc.read_decomposition("crane"), patterns)

    def test_decomposition_read_miss_returns_none(self):
        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read_decomposition("crane"))

    def test_decomposition_different_universe_no_cross_contamination(self):
        alt_answers = ["brain", "stove", "cloud"]
        sc1 = ScoreCache(self.db, ANSWERS)
        sc2 = ScoreCache(self.db, alt_answers)
        sc1.write_decomposition("crane", bytes(range(len(ANSWERS))))
        self.assertIsNone(sc2.read_decomposition("crane"))

    def test_decomposition_overwrite_replaces_value(self):
        sc = ScoreCache(self.db, ANSWERS)
        sc.write_decomposition("crane", bytes([1] * len(ANSWERS)))
        sc.write_decomposition("crane", bytes([2] * len(ANSWERS)))
        self.assertEqual(sc.read_decomposition("crane"), bytes([2] * len(ANSWERS)))

    def test_stats_reflects_row_counts_and_timestamp(self):
        """stats() backs the cache-status display ('d' command) — verify it
        reports the right counts per table, scoped to this universe, and a
        timestamp only once a subgroup result has been written."""
        sc = ScoreCache(self.db, ANSWERS)
        empty_sp, empty_ws, empty_rd, empty_mtime = sc.stats()
        self.assertEqual((empty_sp, empty_ws, empty_rd), (0, 0, 0))
        self.assertIsNone(empty_mtime)

        key1 = ScoreCache.encode_subset(["crane", "slate"])
        key2 = ScoreCache.encode_subset(["heart", "earth"])
        sc.write(key1, "full", "trace", 2.0)
        sc.write(key2, "hard", "stale", 1.5)
        sc.write_scores(key1, [("crane", 1.0), ("slate", 0.9)], "entropy_gain")
        sc.write_decomposition("crane", bytes([0] * len(ANSWERS)))

        sp_rows, ws_rows, rd_rows, mtime = sc.stats()
        self.assertEqual(sp_rows, 2)
        self.assertEqual(ws_rows, 2)
        self.assertEqual(rd_rows, 1)
        self.assertIsNotNone(mtime)

    def test_stats_scoped_to_universe(self):
        alt_answers = ["brain", "stove", "cloud"]
        sc1 = ScoreCache(self.db, ANSWERS)
        sc2 = ScoreCache(self.db, alt_answers)
        key = ScoreCache.encode_subset(["crane", "slate"])
        sc1.write(key, "full", "trace", 2.0)

        sp1, _, _, _ = sc1.stats()
        sp2, _, _, _ = sc2.stats()
        self.assertEqual(sp1, 1)
        self.assertEqual(sp2, 0,
                         "stats() must not count another universe's rows")

    def test_close_releases_connection(self):
        sc = ScoreCache(self.db, ANSWERS)
        sc.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            sc.read(ScoreCache.encode_subset(["crane"]), "full")

    def test_write_scores_rolls_back_on_failure(self):
        """A failure mid-write must not leave a partial commit — the
        transaction is rolled back and the exception re-raised."""
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write_scores(branch_key, [("crane", 1.0)], "entropy_gain")

        class FailingExecuteMany:
            """sqlite3.Connection.executemany is a read-only slot — wrap the
            real connection to inject a failure into one call only."""
            def __init__(self, real):
                self._real = real

            def executemany(self, *a, **k):
                raise sqlite3.OperationalError("boom")

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_conn = sc._conn
        sc._conn = FailingExecuteMany(real_conn)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                sc.write_scores(branch_key, [("slate", 2.0)], "entropy_gain")
        finally:
            sc._conn = real_conn

        # The pre-existing row must survive untouched, and no partial
        # row from the failed write should have leaked in.
        result = dict(sc.read_scores(branch_key, "entropy_gain"))
        self.assertEqual(result, {"crane": 1.0})

    def test_read_detail_returns_word_score_and_timestamp(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])
        before = int(time.time())
        sc.write(key, "erd_words_unfiltered", "crane", 1.5)
        after = int(time.time())

        detail = sc.read_detail(key, "erd_words_unfiltered")
        self.assertEqual(detail[0], "crane")
        self.assertAlmostEqual(detail[1], 1.5)
        self.assertGreaterEqual(detail[2], before)
        self.assertLessEqual(detail[2], after)

    def test_read_detail_miss_returns_none(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])
        self.assertIsNone(sc.read_detail(key, "erd_words_unfiltered"))

    def test_delete_removes_entry(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write(key, "erd_words_unfiltered", "crane", 1.5)

        sc.delete(key, "erd_words_unfiltered")

        self.assertIsNone(sc.read(key, "erd_words_unfiltered"))
        self.assertIsNone(sc.read_detail(key, "erd_words_unfiltered"))
        self.assertNotIn((key, "erd_words_unfiltered"), sc._mem_cache)

    def test_delete_persists_across_connections(self):
        key = ScoreCache.encode_subset(["crane", "slate"])
        sc1 = ScoreCache(self.db, ANSWERS)
        sc1.write(key, "erd_words_unfiltered", "crane", 1.5)
        sc1.delete(key, "erd_words_unfiltered")
        sc1.close()

        sc2 = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc2.read(key, "erd_words_unfiltered"))

    def test_delete_of_missing_entry_is_a_no_op(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])
        sc.delete(key, "erd_words_unfiltered")  # must not raise
        self.assertIsNone(sc.read(key, "erd_words_unfiltered"))


# ---------------------------------------------------------------------------
# Transparent SQLite persistence across sessions
# ---------------------------------------------------------------------------

class TestTransparentPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_scores_persist_across_sessions(self):
        # Session 1: compute and auto-persist
        s1 = make_solution(db_path=self.db)
        s1.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        scores1 = dict(s1.scores)

        # Session 2: new Solution, same DB — should auto-load
        s2 = make_solution(db_path=self.db)
        s2.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        scores2 = dict(s2.scores)

        for w in GUESSES:
            self.assertAlmostEqual(scores1[w], scores2[w], places=10,
                                   msg=f"Score mismatch for {w}")

    def test_mid_game_scores_persisted_subset_scoped(self):
        s = make_solution(db_path=self.db)
        pattern = calculate_response("crane", "slate")
        s.apply_guess("crane", pattern)
        self.assertFalse(s._is_full_game())

        s.compute_scores(s.current_words, ScoringMethod.ENTROPY_GAIN)

        # Scores for this position are written, keyed by its remaining-word subset.
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(s.current_words)
        cached = sc.read_scores(branch_key, "entropy_gain")
        self.assertIsNotNone(cached)
        self.assertIn(s.current_words[0], dict(cached))

    def test_mid_game_scores_reloaded_for_same_position(self):
        pattern = calculate_response("crane", "slate")

        s1 = make_solution(db_path=self.db)
        s1.apply_guess("crane", pattern)
        s1.compute_scores(s1.current_words, ScoringMethod.ENTROPY_GAIN)
        scores1 = dict(s1.scores)

        s2 = make_solution(db_path=self.db)
        s2.apply_guess("crane", pattern)
        s2.compute_scores(s2.current_words, ScoringMethod.ENTROPY_GAIN)
        scores2 = dict(s2.scores)

        for w in s1.current_words:
            self.assertAlmostEqual(scores1[w], scores2[w], places=10)
        self.assertIn(ScoringMethod.ENTROPY_GAIN, s2._db_loaded_methods)

    def test_different_positions_use_different_cache_entries(self):
        s1 = make_solution(db_path=self.db)
        pattern1 = calculate_response("crane", "slate")
        s1.apply_guess("crane", pattern1)
        s1.compute_scores(s1.current_words, ScoringMethod.ENTROPY_GAIN)

        s2 = make_solution(db_path=self.db)
        pattern2 = calculate_response("heart", "earth")
        s2.apply_guess("heart", pattern2)
        self.assertNotEqual(sorted(s1.current_words), sorted(s2.current_words))

        sc = ScoreCache(self.db, ANSWERS)
        s2_key = ScoreCache.encode_subset(s2.current_words)
        self.assertIsNone(
            sc.read_scores(s2_key, "entropy_gain"),
            "a different remaining-word set must not see another position's "
            "cached scores")

    def test_multi_method_persist_and_reload(self):
        methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MAX_GROUP_SIZE]
        s1 = make_solution(db_path=self.db)
        s1.compute_scores_multi(GUESSES, methods)

        s2 = make_solution(db_path=self.db)
        s2.compute_scores_multi(GUESSES, methods)

        for w in GUESSES:
            for m in methods:
                v1 = s1.word_scores[w][m]
                v2 = s2.word_scores[w][m]
                self.assertAlmostEqual(v1, v2, places=10,
                                       msg=f"{m} score mismatch for {w}")


# ---------------------------------------------------------------------------
# ResponseCache decomposition persistence
# ---------------------------------------------------------------------------

class TestResponseCachePersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_decomposition_persisted_on_first_build(self):
        sc = ScoreCache(self.db, ANSWERS)
        cache = ResponseCache(ANSWERS, score_cache=sc)
        cache._ensure("crane")
        self.assertIsNotNone(sc.read_decomposition("crane"))

    def test_decomposition_reused_without_recomputing(self):
        sc1 = ScoreCache(self.db, ANSWERS)
        cache1 = ResponseCache(ANSWERS, score_cache=sc1)
        cache1._ensure("crane")
        expected = bytes(cache1._cache["crane"])

        sc2 = ScoreCache(self.db, ANSWERS)
        cache2 = ResponseCache(ANSWERS, score_cache=sc2)
        with mock.patch('wordle_engine.calculate_response') as fake_calc:
            cache2._ensure("crane")
            fake_calc.assert_not_called()
        self.assertEqual(bytes(cache2._cache["crane"]), expected)

    def test_decomposition_reload_matches_freshly_computed(self):
        sc1 = ScoreCache(self.db, ANSWERS)
        cache1 = ResponseCache(ANSWERS, score_cache=sc1)
        cache1._ensure("crane")
        expected = bytes(cache1._cache["crane"])

        sc2 = ScoreCache(self.db, ANSWERS)
        cache2 = ResponseCache(ANSWERS, score_cache=sc2)
        cache2._ensure("crane")
        self.assertEqual(bytes(cache2._cache["crane"]), expected)


# ---------------------------------------------------------------------------
# cache_all_scores — comprehensive per-word persistence from one partition
# ---------------------------------------------------------------------------

class TestCacheAllScores(unittest.TestCase):
    """cache_all_scores is the single place that knows the full ScoringMethod
    roster, so algorithms (compute_lookahead, min_expected_guesses, ...) that
    merely want to remember a word's standing for a subgroup don't have to —
    they delegate "comprehensively" to this helper instead of enumerating
    ScoringMethod themselves. Adding a new ScoringMethod only changes this
    one function; no algorithm using it needs to change.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_every_scoring_method_persisted_from_single_partition(self):
        sc = ScoreCache(self.db, ANSWERS)
        cache = ResponseCache(ANSWERS, score_cache=sc)
        subgroup = ANSWERS[:6]
        branch_key = ScoreCache.encode_subset(subgroup)

        cache_all_scores("heart", subgroup, sc, branch_key, cache=cache)

        for method in ScoringMethod:
            cached = sc.read_scores(branch_key, method.name.lower())
            self.assertIsNotNone(
                cached, f"{method.name} should be persisted by cache_all_scores")
            expected = score_word("heart", subgroup, method, cache=cache)
            self.assertEqual(dict(cached)["heart"], expected)

    def test_no_op_without_score_cache(self):
        cache = ResponseCache(ANSWERS)
        # Must not raise when there's nothing to persist to.
        cache_all_scores("heart", ANSWERS[:6], None,
                         ScoreCache.encode_subset(ANSWERS[:6]), cache=cache)

    def test_no_op_with_memory_score_cache(self):
        """Hard-mode searches pass a MemoryScoreCache, whose minimal
        read/write interface deliberately has no write_scores — those ERD
        values are path-dependent and must never reach the persisted
        cross-game cache. cache_all_scores must skip silently rather than
        raise AttributeError (regression for the b67 hard-mode crash)."""
        cache = ResponseCache(ANSWERS)
        mc = MemoryScoreCache()
        mc.set_scope('test-scope')
        subgroup = ANSWERS[:6]
        cache_all_scores("heart", subgroup, mc,
                         ScoreCache.encode_subset(subgroup), cache=cache)


# ---------------------------------------------------------------------------
# rank_candidates_by_max_group_size_then_entropy_gain — shared word_scores cache with cmd_solve
# ---------------------------------------------------------------------------

class TestRankGuessesByGroupThenEntropy(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def _expected_order(self, words, candidates, cache):
        scored = []
        for word in candidates:
            groups = cache.group_counts(word, words)
            scores = score_groups_multi(
                groups, (ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN))
            scored.append((scores[ScoringMethod.MAX_GROUP_SIZE],
                            -scores[ScoringMethod.ENTROPY_GAIN], word))
        scored.sort()
        return [w for _, _, w in scored]

    def test_uses_word_scores_populated_by_compute_scores_multi(self):
        """If 's' (compute_scores_multi) already populated word_scores for
        this position, ranking must use those cached scores rather than
        recomputing via rcache.group_counts."""
        words = ANSWERS[:6]
        candidates = GUESSES

        sc = ScoreCache(self.db, ANSWERS)
        try:
            cache = ResponseCache(ANSWERS, score_cache=sc)
            soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
            soln.current_words = words
            soln.compute_scores_multi(
                candidates, [ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN])

            expected_order = self._expected_order(words, candidates, cache)

            class ExplodingResponseCache:
                @staticmethod
                def group_counts(word, subset):
                    raise AssertionError(
                        "ranking should use cached word_scores, not recompute")

            ranked = rank_candidates_by_max_group_size_then_entropy_gain(
                words, candidates, ExplodingResponseCache(), sc)
            self.assertEqual(ranked, expected_order)
        finally:
            sc.close()

    def test_populates_word_scores_for_later_compute_scores_multi(self):
        """If ERD ranking runs first at this position, it must persist
        MAX_GROUP_SIZE/ENTROPY_GAIN into word_scores so a later 's'
        (compute_scores_multi) at the same position hits the cache instead
        of recomputing via score_word_multi."""
        words = ANSWERS[:6]
        candidates = GUESSES

        sc = ScoreCache(self.db, ANSWERS)
        try:
            cache = ResponseCache(ANSWERS, score_cache=sc)

            rank_candidates_by_max_group_size_then_entropy_gain(words, candidates, cache, sc)

            soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
            soln.current_words = words
            with mock.patch('wordle_engine.score_word_multi') as spy:
                soln.compute_scores_multi(
                    candidates,
                    [ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN])
            spy.assert_not_called()
        finally:
            sc.close()

    def test_cancel_check_immediate_returns_input_unchanged(self):
        words = ANSWERS[:6]
        candidates = list(GUESSES)

        sc = ScoreCache(self.db, ANSWERS)
        try:
            cache = ResponseCache(ANSWERS, score_cache=sc)
            ranked = rank_candidates_by_max_group_size_then_entropy_gain(
                words, candidates, cache, sc, cancel_check=lambda: True)
            self.assertEqual(ranked, candidates)

            branch_key = ScoreCache.encode_subset(words)
            self.assertIsNone(sc.read_scores(branch_key, 'max_group_size'))
            self.assertIsNone(sc.read_scores(branch_key, 'entropy_gain'))
        finally:
            sc.close()

    def test_memory_score_cache_no_attribute_error(self):
        """MemoryScoreCache (hard mode) lacks read_scores/write_scores —
        ranking must degrade to 'always compute', not raise AttributeError."""
        words = ANSWERS[:6]
        candidates = list(GUESSES)
        cache = ResponseCache(ANSWERS)
        mc = MemoryScoreCache()
        mc.set_scope('test-scope')

        ranked = rank_candidates_by_max_group_size_then_entropy_gain(words, candidates, cache, mc)
        self.assertEqual(ranked, self._expected_order(words, candidates, cache))


# ---------------------------------------------------------------------------
# compute_lookahead — winner's max-group-size persisted alongside entropy
# ---------------------------------------------------------------------------

class TestComputeLookaheadCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_winners_other_scores_persisted_alongside_ranking_score(self):
        s = make_solution(db_path=self.db)
        first_ent = score_word("piano", s.current_words, ScoringMethod.ENTROPY_GAIN,
                               cache=s.cache)
        s.compute_lookahead([("piano", first_ent)])

        sc = ScoreCache(self.db, ANSWERS)
        grouped = s.cache.group_words("piano", s.current_words)
        scanned = [sg for sg in grouped.values() if len(sg) > 2]
        self.assertTrue(scanned, "expected at least one subgroup big enough to scan")

        for subgroup in scanned:
            branch_key = ScoreCache.encode_subset(subgroup)
            hit = sc.read(branch_key, "hard")
            self.assertIsNotNone(hit)
            best_word, _best_score = hit

            # Every method besides the ranking criterion (entropy, already
            # captured in branch_best_by_policy) should be persisted too — they
            # all come from the same group-count partition, so there's no
            # principled reason to single any of them out.
            for method in ScoringMethod:
                if method == ScoringMethod.ENTROPY_GAIN:
                    continue
                cached = sc.read_scores(branch_key, method.name.lower())
                self.assertIsNotNone(
                    cached,
                    f"{method.name}'s score for the cached winner should be "
                    f"persisted alongside its entropy, at near-zero extra cost")
                cached_value = dict(cached)[best_word]
                expected_value = score_word(best_word, subgroup, method, cache=s.cache)
                self.assertEqual(cached_value, expected_value)

    def test_size_two_subgroups_use_shortcut_without_scanning(self):
        """A response group of exactly 2 words always costs 1 extra guess
        (whichever the candidate scan would pick, the loser is then a
        singleton) — compute_lookahead special-cases it to skip the scan
        entirely rather than search a 2-candidate space for an answer that's
        always 1.0."""
        s = make_solution()
        first_ent = score_word("crane", s.current_words, ScoringMethod.ENTROPY_GAIN,
                               cache=s.cache)
        grouped = s.cache.group_words("crane", s.current_words)
        pair_groups = [sg for sg in grouped.values() if len(sg) == 2]
        self.assertTrue(pair_groups, "expected at least one size-2 group for this setup")

        with mock.patch('wordle_engine.score_word') as mocked:
            results = s.compute_lookahead([("crane", first_ent)])
        # None of the size-2 groups should have triggered a candidate scan —
        # only score_word calls from outside compute_lookahead (there are
        # none here) would show up.
        mocked.assert_not_called()
        self.assertEqual(len(results), 1)

    def test_full_mode_searches_provided_candidates(self):
        """With second_step_words supplied, the best second guess is chosen
        from that list — not restricted to the subgroup itself (hard mode) —
        and persisted under the 'full' policy namespace, distinct from
        hard mode's 'hard' so the two never collide in the cache."""
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            s = make_solution(db_path=tmp.name)
            first_ent = score_word("piano", s.current_words, ScoringMethod.ENTROPY_GAIN,
                                   cache=s.cache)
            second_step_words = ["brain", "stove", "cloud", "crane", "train"]

            results = s.compute_lookahead([("piano", first_ent)],
                                           second_step_words=second_step_words)
            self.assertEqual(len(results), 1)
            word, step1, step2, combined = results[0]
            self.assertEqual(word, "piano")
            self.assertAlmostEqual(combined, step1 + step2)

            grouped = s.cache.group_words("piano", s.current_words)
            scanned = [sg for sg in grouped.values() if len(sg) > 2]
            self.assertTrue(scanned, "expected at least one scannable subgroup")
            for subgroup in scanned:
                branch_key = ScoreCache.encode_subset(subgroup)
                hit = s.score_cache.read(branch_key, "full")
                self.assertIsNotNone(hit)
                self.assertIn(hit[0], second_step_words,
                              "full-mode winner must come from second_step_words")
                self.assertIsNone(
                    s.score_cache.read(branch_key, "hard"),
                    "full-mode results must not collide with hard mode's namespace")
        finally:
            os.unlink(tmp.name)

    def test_repeat_run_reuses_cached_subgroup_results(self):
        """A subgroup whose result is already in the SQLite lookahead cache
        must be reused verbatim on a later run — no rescanning. Uses
        "piano" specifically because it produces subgroups bigger than 2
        (unlike "crane" here, whose subgroups are all <= 2 and so take the
        cnt==2 shortcut, never touching the cache-hit-reuse code path)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            s = make_solution(db_path=tmp.name)
            first_ent = score_word("piano", s.current_words, ScoringMethod.ENTROPY_GAIN,
                                   cache=s.cache)
            grouped = s.cache.group_words("piano", s.current_words)
            self.assertTrue(any(len(sg) > 2 for sg in grouped.values()),
                            "test setup needs a subgroup bigger than 2 to "
                            "exercise the cache-hit-reuse path, not the shortcut")

            first_results = s.compute_lookahead([("piano", first_ent)])

            with mock.patch('wordle_engine.score_word',
                            wraps=score_word) as wrapped:
                second_results = s.compute_lookahead([("piano", first_ent)])
            self.assertFalse(
                wrapped.called,
                "every scannable subgroup was already cached by the first "
                "run — a second run must be pure cache reads")
            self.assertEqual(first_results, second_results)
        finally:
            os.unlink(tmp.name)

    def test_progress_callback_invoked_during_candidate_scan(self):
        """progress_callback (wired to the CLI's ProgressTracker) must fire
        once per candidate scanned in phase 2 — that's what drives the
        on-screen progress bar during a lookahead run."""
        s = make_solution()  # no SQLite cache — every scannable subgroup is scanned fresh
        first_ent = score_word("piano", s.current_words, ScoringMethod.ENTROPY_GAIN,
                               cache=s.cache)
        grouped = s.cache.group_words("piano", s.current_words)
        scannable = [sg for sg in grouped.values() if len(sg) > 2]
        self.assertTrue(scannable)
        expected_calls = sum(len(sg) for sg in scannable)

        calls = []
        s.compute_lookahead([("piano", first_ent)],
                            progress_callback=lambda: calls.append(1))
        self.assertEqual(len(calls), expected_calls)


# ---------------------------------------------------------------------------
# min_expected_guesses
# ---------------------------------------------------------------------------

class TestMinExpectedGuesses(unittest.TestCase):

    def setUp(self):
        self.cache = ResponseCache(ANSWERS)

    def test_singleton_returns_one(self):
        for w in ANSWERS:
            self.assertEqual(min_expected_guesses([w], self.cache, None), 1.0,
                             msg=f"singleton {w}")

    def test_any_pair_returns_one_point_five(self):
        for i in range(len(ANSWERS)):
            for j in range(i + 1, len(ANSWERS)):
                pair = [ANSWERS[i], ANSWERS[j]]
                self.assertAlmostEqual(
                    min_expected_guesses(pair, self.cache, None), 1.5, places=10,
                    msg=f"pair {pair}")

    def test_result_written_to_sqlite(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            subset = ANSWERS[:4]
            result = min_expected_guesses(subset, self.cache, sc)
            self.assertIsNotNone(result)
            hit = sc.read(ScoreCache.encode_subset(subset), ERD_ANSWERS)
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit[1], result, places=10)
        finally:
            os.unlink(tmp.name)

    def test_winner_scores_cached_for_every_method(self):
        """The ERD winner's standing under every ScoringMethod should be
        persisted too — same comprehensive treatment as a lookahead winner,
        via the same cache_all_scores helper, at near-zero extra cost since
        the partition is already in hand for the winning guess.
        """
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            subset = ANSWERS[:4]
            result = min_expected_guesses(subset, self.cache, sc)
            self.assertIsNotNone(result)
            best_word, _best_score = sc.read(ScoreCache.encode_subset(subset), ERD_ANSWERS)

            branch_key = ScoreCache.encode_subset(subset)
            for method in ScoringMethod:
                cached = sc.read_scores(branch_key, method.name.lower())
                self.assertIsNotNone(
                    cached, f"{method.name} should be persisted for the ERD winner")
                expected = score_word(best_word, subset, method, cache=self.cache)
                self.assertEqual(dict(cached)[best_word], expected)
        finally:
            os.unlink(tmp.name)

    def test_disk_io_error_during_write_does_not_abort_recursion(self):
        """A transient 'disk I/O error' from score_cache.write() must not
        propagate out of min_expected_guesses — it would unwind every
        enclosing recursive call and abort the calling background-solver
        thread, discarding the whole computation (see
        cache_sqlite.ScoreCache.write)."""
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            real_conn = sc._conn

            class FailingWrite:
                def __init__(self, real):
                    self._real = real

                def execute(self, sql, *a, **k):
                    if sql.lstrip().startswith("INSERT OR REPLACE INTO branch_best_by_policy"):
                        raise sqlite3.OperationalError("disk I/O error")
                    return self._real.execute(sql, *a, **k)

                def __getattr__(self, name):
                    return getattr(self._real, name)

            sc._conn = FailingWrite(real_conn)
            try:
                subset = ANSWERS[:4]
                result = min_expected_guesses(subset, self.cache, sc)
            finally:
                sc._conn = real_conn

            self.assertIsNotNone(result)
            # Still memoized in-process even though persistence failed.
            hit = sc.read(ScoreCache.encode_subset(subset), ERD_ANSWERS)
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit[1], result, places=10)
            sc.close()
        finally:
            os.unlink(tmp.name)

    def test_expired_deadline_returns_none(self):
        already_past = time.time() - 1
        result = min_expected_guesses(ANSWERS, self.cache, None,
                                      deadline=already_past)
        self.assertIsNone(result)

    def test_timeout_propagates_through_recursion(self):
        """If the deadline expires partway through a deeper recursive call,
        that call's `None` ("too big — move on") must propagate all the way
        back to the root rather than be silently swallowed as if the guess
        being explored were simply a bad one."""
        words = ANSWERS[:4]
        deadline = time.time() + 1000  # comfortably in the future for the root check

        real_time = time.time
        calls = [0]
        def fake_time():
            calls[0] += 1
            # Root's own deadline check passes; every check from then on
            # (i.e. every recursive call) finds the deadline already blown.
            return real_time() if calls[0] <= 1 else deadline + 1

        # Pin best-first ordering OFF for this test.  Timeout propagation is
        # independent of candidate ordering, but ordering can solve a tiny set
        # in one ply (all-singleton split, then ERD-lower-bound pruning
        # removes the rest),
        # leaving no deeper recursive call for the deadline to interrupt.  With
        # ordering off the set deterministically recurses, exercising the path
        # under test regardless of how ORDER_MIN_N is later tuned.
        with mock.patch('wordle_engine.ORDER_MIN_N', 10 ** 9), \
             mock.patch('wordle_engine.time.time', side_effect=fake_time):
            result = min_expected_guesses(words, self.cache, None,
                                           deadline=deadline, guesses=words)
        self.assertIsNone(result)

    def test_larger_set_costs_more_than_smaller(self):
        cost3 = min_expected_guesses(ANSWERS[:3], self.cache, None)
        cost5 = min_expected_guesses(ANSWERS[:5], self.cache, None)
        self.assertLess(cost3, cost5)

    def test_works_with_memory_score_cache(self):
        """Hard mode runs min_expected_guesses against a MemoryScoreCache
        (gs.constrained_erd_cache), not a SQLite-backed ScoreCache. Its
        write(...) call at the end must succeed, and the cache_all_scores(...)
        call right after it must not blow up on MemoryScoreCache's minimal
        interface — regression for the b67 hard-mode crash:
        AttributeError: 'MemoryScoreCache' object has no attribute 'write_scores'.
        """
        mc = MemoryScoreCache()
        mc.set_scope(MemoryScoreCache.fingerprint_vocabulary(ANSWERS))
        subset = ANSWERS[:4]
        result = min_expected_guesses(subset, self.cache, mc, guesses=subset)
        self.assertIsNotNone(result)
        hit = mc.read(ScoreCache.encode_subset(subset), ERD_ALL)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[1], result, places=10)

    def test_unrecognized_policy_raises(self):
        """An unrecognized policy must raise rather than silently write
        results into the wrong cache namespace — that would corrupt that
        mode's cache for every future game (see docstring)."""
        with self.assertRaises(ValueError):
            min_expected_guesses(ANSWERS[:3], self.cache, None,
                                 policy="not_a_real_policy")

    def test_heartbeat_fires_for_root_and_recursive_calls(self):
        """heartbeat fires once per min_expected_guesses invocation at
        every recursion depth — far more often than progress_callback,
        which only fires once per fully-evaluated top-level candidate and
        can go silent for a long time during a single candidate's deep
        recursive evaluation (best_erd still unbounded, nothing prunable
        yet)."""
        subset = ANSWERS[:4]
        heartbeats = []
        progress_calls = []
        min_expected_guesses(
            subset, self.cache, None, guesses=GUESSES,
            heartbeat=lambda: heartbeats.append(1),
            progress_callback=lambda *a: progress_calls.append(a))
        self.assertGreaterEqual(len(heartbeats), 1)
        self.assertGreater(len(heartbeats), len(progress_calls))

    def test_heartbeat_fires_once_on_cache_hit_root(self):
        """A cached root returns immediately with no recursion at all —
        but heartbeat must still fire for that single invocation, so a
        caller relying on it for liveness never sees total silence."""
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            subset = ANSWERS[:4]
            min_expected_guesses(subset, self.cache, sc)  # populate cache

            heartbeats = []
            result = min_expected_guesses(
                subset, self.cache, sc,
                heartbeat=lambda: heartbeats.append(1))
            self.assertIsNotNone(result)
            self.assertEqual(heartbeats, [1])
        finally:
            os.unlink(tmp.name)


class TestDepthLimitedERD(unittest.TestCase):
    """Budget-limited ERD: the minimum expected guesses among strategies that
    are guaranteed to win within `budget`.  A generous budget reproduces the
    unlimited optimum; a budget below a position's worst-case line makes it
    infeasible (None) — a position you would lose in a real game."""

    # 8 words sharing the suffix "ound"; each distinguishing first letter
    # (b/f/h/m/p/r/s/w) appears nowhere in "ound", so answers-only guessing
    # isolates exactly one word at a time — a clean linear probe with a
    # worst-case line of 8 and ERD = (1+..+8)/8 = 4.5.
    LINEAR = ["bound", "found", "hound", "mound",
              "pound", "round", "sound", "wound"]

    def setUp(self):
        self.cache = ResponseCache(ANSWERS)
        self.lcache = ResponseCache(self.LINEAR)

    def test_generous_budget_matches_unlimited(self):
        subset = ANSWERS[:6]
        unlimited = min_expected_guesses(subset, self.cache, None, guesses=subset)
        budgeted = min_expected_guesses(subset, self.cache, None,
                                        guesses=subset, budget=6)
        self.assertAlmostEqual(unlimited, budgeted, places=10)

    def test_pair_needs_budget_two(self):
        pair = [ANSWERS[0], ANSWERS[1]]
        self.assertAlmostEqual(
            min_expected_guesses(pair, self.cache, None, guesses=pair, budget=2),
            1.5, places=10)
        self.assertIsNone(
            min_expected_guesses(pair, self.cache, None, guesses=pair, budget=1))

    def test_infeasible_below_worst_case_depth(self):
        # Exactly fits at budget 8 (worst-case line length), infeasible at 7.
        fit = min_expected_guesses(self.LINEAR, self.lcache, None,
                                   guesses=self.LINEAR, budget=8)
        self.assertAlmostEqual(fit, 4.5, places=10)
        self.assertIsNone(
            min_expected_guesses(self.LINEAR, self.lcache, None,
                                 guesses=self.LINEAR, budget=7))

    def test_max_solvable_within_values(self):
        # M(b) = 1 + 242*M(b-1), M(0) = 0.
        self.assertEqual(max_solvable_within(0), 0)
        self.assertEqual(max_solvable_within(1), 1)
        self.assertEqual(max_solvable_within(2), 243)
        self.assertEqual(max_solvable_within(3), 58807)

    def test_oversized_branch_is_certain_loss(self):
        # A single guess yields at most 243 patterns, so >243 distinct words
        # cannot all be resolved in 2 guesses regardless of structure.  The size
        # bound certifies the loss without scanning any candidate.
        import itertools
        words = [''.join(p) for p in itertools.islice(
            itertools.product('abcdefg', repeat=5), max_solvable_within(2) + 1)]
        self.assertEqual(len(words), 244)
        cache = ResponseCache(words)
        self.assertIsNone(
            min_expected_guesses(words, cache, None, guesses=words, budget=2))

    def test_max_depth_persisted(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, self.LINEAR)
            lcache = ResponseCache(self.LINEAR, score_cache=sc)
            min_expected_guesses(self.LINEAR, lcache, sc,
                                 guesses=self.LINEAR, budget=8)
            entry = sc.read_with_depth(
                ScoreCache.encode_subset(self.LINEAR), ERD_ALL)
            self.assertIsNotNone(entry)
            _bw, _score, max_depth, solve_budget = entry
            self.assertEqual(max_depth, 8)         # worst-case line length
            self.assertIsNone(solve_budget)        # untainted: budget 8 fit exactly
        finally:
            os.unlink(tmp.name)

    def test_write_loss_keeps_widest_budget(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, self.LINEAR)
            key = b'aaaaabbbbb'
            sc.write_loss(key, ERD_ALL, 3)
            self.assertEqual(sc.read_loss(key, ERD_ALL), 3)
            sc.write_loss(key, ERD_ALL, 2)   # narrower — ignored
            self.assertEqual(sc.read_loss(key, ERD_ALL), 3)
            sc.write_loss(key, ERD_ALL, 5)   # wider — widens reuse range
            self.assertEqual(sc.read_loss(key, ERD_ALL), 5)
            sc.close()
            sc2 = ScoreCache(tmp.name, self.LINEAR)   # fresh: no session mirror
            self.assertEqual(sc2.read_loss(key, ERD_ALL), 5)
            self.assertIsNone(sc2.read_loss(b'never', ERD_ALL))
        finally:
            os.unlink(tmp.name)

    def test_proven_loss_is_persisted_and_reused(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        tmp.close()
        try:
            key = ScoreCache.encode_subset(self.LINEAR)
            sc = ScoreCache(tmp.name, self.LINEAR)
            lcache = ResponseCache(self.LINEAR, score_cache=sc)
            # LINEAR needs budget 8; budget 7 is a proven loss, now persisted.
            self.assertIsNone(min_expected_guesses(
                self.LINEAR, lcache, sc, guesses=self.LINEAR, budget=7))
            self.assertEqual(sc.read_loss(key, ERD_ALL), 7)
            sc.close()
            # A fresh process (no session mirror) reuses the persisted loss, and
            # a loss within 7 is also a loss within 6 — both return immediately.
            sc2 = ScoreCache(tmp.name, self.LINEAR)
            self.assertEqual(sc2.read_loss(key, ERD_ALL), 7)
            lcache2 = ResponseCache(self.LINEAR, score_cache=sc2)
            self.assertIsNone(min_expected_guesses(
                self.LINEAR, lcache2, sc2, guesses=self.LINEAR, budget=6))
        finally:
            os.unlink(tmp.name)


class TestSubbranchSolverHook(unittest.TestCase):
    """The subbranch_solver hook lets the swarm divert a sub-branch to a
    cooperative parallel solve.  It must be correctness-neutral: a hook that
    simply solves inline yields the identical ERD as no hook at all."""

    def setUp(self):
        self.cache = ResponseCache(ANSWERS)

    def test_inline_hook_matches_no_hook(self):
        subset = ANSWERS[:8]
        without = min_expected_guesses(subset, self.cache, None,
                                       guesses=GUESSES, budget=6)

        calls = []
        def inline_solver(words, budget):
            calls.append(len(words))
            # Solve inline (no further delegation) -> identical to recursion.
            return _solve_subset(words, self.cache, None, budget, None, GUESSES,
                                 ERD_ALL, None, None, None, None, None)

        with_hook = min_expected_guesses(subset, self.cache, None,
                                         guesses=GUESSES, budget=6,
                                         subbranch_solver=inline_solver)
        self.assertEqual(without, with_hook)
        self.assertTrue(calls)  # the hook actually fired on sub-branches

    def test_declining_hook_falls_through(self):
        subset = ANSWERS[:8]
        without = min_expected_guesses(subset, self.cache, None,
                                       guesses=GUESSES, budget=6)
        # A hook that always declines (returns None) must change nothing.
        with_decline = min_expected_guesses(
            subset, self.cache, None, guesses=GUESSES, budget=6,
            subbranch_solver=lambda w, b: None)
        self.assertEqual(without, with_decline)


class TestCacheReuseRule(unittest.TestCase):
    """_cache_reuse decides whether a cached (best_word, score, max_depth,
    solve_budget) entry is valid at a given remaining budget."""

    def test_unlimited_reuses_legacy_and_untainted_not_tainted(self):
        # legacy (max_depth None) and untainted (solve_budget None) are
        # unconstrained optima -> reusable unlimited; tainted is budget-specific.
        self.assertEqual(_cache_reuse(("w", 3.0, None, None), None), (3.0, None, False))
        self.assertEqual(_cache_reuse(("w", 3.0, 4, None), None), (3.0, 4, False))
        self.assertIsNone(_cache_reuse(("w", 3.0, 4, 5), None))
        self.assertIsNone(_cache_reuse(None, None))

    def test_budgeted_rejects_legacy(self):
        self.assertIsNone(_cache_reuse(("w", 3.0, None, None), 5))

    def test_budgeted_untainted_valid_when_fits(self):
        self.assertEqual(_cache_reuse(("w", 3.0, 4, None), 5), (3.0, 4, False))
        self.assertEqual(_cache_reuse(("w", 3.0, 5, None), 5), (3.0, 5, False))
        self.assertIsNone(_cache_reuse(("w", 3.0, 6, None), 5))   # too deep

    def test_budgeted_tainted_valid_only_at_exact_budget(self):
        self.assertEqual(_cache_reuse(("w", 3.0, 4, 5), 5), (3.0, 4, True))
        self.assertIsNone(_cache_reuse(("w", 3.0, 4, 5), 6))      # revive siblings
        self.assertIsNone(_cache_reuse(("w", 3.0, 4, 5), 4))


# ---------------------------------------------------------------------------
# min_expected_guesses: candidate_cost_lower_bound admissible-bound pruning
# ---------------------------------------------------------------------------

def _brute_force_erd(branch_words, cache, candidate_list):
    """Reference ERD computation with NO pruning of any kind: every candidate
    and every sub-branch is evaluated, with no candidate_cost_lower_bound
    check and no cost >= best_erd branch-and-bound break. Mirrors the base-case and
    skip-candidate handling of min_expected_guesses exactly so the two can be
    compared directly."""
    n = len(branch_words)
    if n == 1:
        return 1.0
    best = float('inf')
    for candidate in candidate_list:
        groups = cache.group_words(candidate, branch_words)
        cost = 1.0
        skip = False
        for sub_branch in groups.values():
            k = len(sub_branch)
            if k == 1 and sub_branch[0] == candidate:
                continue
            if k >= n:
                skip = True
                break
            cost += (k / n) * _brute_force_erd(sub_branch, cache, candidate_list)
        if skip:
            continue
        best = min(best, cost)
    return best


class TestMinExpectedGuessesLowerBoundPruning(unittest.TestCase):
    """The candidate_cost_lower_bound admissible lower bound
    (candidate_cost_lower_bound = 3 - (G + has_self) / n,
    derived from sub_erd(k) >= 2 - 1/k for any subgroup of size k) lets
    min_expected_guesses skip a candidate guess without recursing into any
    of its subgroups. Since the bound never overestimates the guess's true
    cost, the chosen ERD must be identical to an unpruned search."""

    def setUp(self):
        self.cache = ResponseCache(ANSWERS)

    def test_pruning_matches_brute_force(self):
        for size in (2, 3, 4):
            for subset in itertools.combinations(ANSWERS, size):
                subset = list(subset)
                pruned = min_expected_guesses(subset, self.cache, None,
                                               guesses=GUESSES)
                expected = _brute_force_erd(subset, self.cache, GUESSES)
                self.assertAlmostEqual(
                    pruned, expected, places=10,
                    msg=f"mismatch for {subset}")


# ---------------------------------------------------------------------------
# verify_erd_cache
# ---------------------------------------------------------------------------

class _MultiPartitionCache:
    """Stub for soln.cache: group_words(word, subset) returns a fixed
    partition keyed only by `word`, ignoring `subset`. Each test below
    only ever looks up the (word, subset) pairs it set up, so a
    word-keyed table is enough to drive verify_erd_cache deterministically
    without a real ResponseCache."""

    def __init__(self, partitions):
        self._partitions = partitions

    def group_words(self, word, subset):
        return self._partitions[word]


class TestVerifyERDCache(unittest.TestCase):
    """verify_erd_cache reconstructs 1 + sum (k_i/n)*sub_score from a cached
    entry's own subtree and compares it to the entry's stored best_score —
    without recomputing anything via min_expected_guesses."""

    WORDS = ["alpha", "beta", "c1", "c2", "c3", "c4"]

    # 'alpha' splits the 6 words into: itself (self, contributes 0),
    # 'beta' (non-self singleton, contributes 1/6), and two cached size-2
    # groups each worth 1.5 (contributing (2/6)*1.5 = 0.5 each).
    ROOT_PARTITION = {
        'self': ['alpha'], 'other': ['beta'],
        'g1': ['c1', 'c2'], 'g2': ['c3', 'c4'],
    }
    # Each size-2 group splits into a self singleton and one other
    # singleton: 1 + (1/2)*1 = 1.5.
    PAIR1_PARTITION = {'self': ['c1'], 'other': ['c2']}
    PAIR2_PARTITION = {'self': ['c3'], 'other': ['c4']}

    ROOT_SCORE = 13 / 6  # 1 + 1/6 + 0.5 + 0.5

    def _full_cache(self):
        return _MultiPartitionCache({
            'alpha': self.ROOT_PARTITION,
            'c1': self.PAIR1_PARTITION,
            'c3': self.PAIR2_PARTITION,
        })

    def test_uncached_root_reports_uncached(self):
        sc = MemoryScoreCache()
        sc.set_scope('test')
        cache = _MultiPartitionCache({})

        report = verify_erd_cache(self.WORDS, cache, sc, ERD_ALL)

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['status'], 'uncached')

    def test_fully_consistent_tree_reports_match(self):
        sc = MemoryScoreCache()
        sc.set_scope('test')
        sc.write(ScoreCache.encode_subset(self.WORDS), ERD_ALL,
                 'alpha', self.ROOT_SCORE)
        sc.write(ScoreCache.encode_subset(['c1', 'c2']), ERD_ALL, 'c1', 1.5)
        sc.write(ScoreCache.encode_subset(['c3', 'c4']), ERD_ALL, 'c3', 1.5)

        report = verify_erd_cache(self.WORDS, self._full_cache(), sc, ERD_ALL)

        self.assertEqual(len(report), 3)
        for r in report:
            self.assertEqual(r['status'], 'match', r)
            self.assertTrue(r['complete'], r)
            self.assertAlmostEqual(r['reconstructed'], r['best_score'])

    def test_uncached_subgroups_report_incomplete(self):
        # Root entry whose g1/g2 subgroups were never cached — e.g. ERD-pruned
        # away entirely by candidate_cost_lower_bound (b95) without recursing
        # into them.
        sc = MemoryScoreCache()
        sc.set_scope('test')
        sc.write(ScoreCache.encode_subset(self.WORDS), ERD_ALL, 'alpha', 10.0)
        cache = _MultiPartitionCache({'alpha': self.ROOT_PARTITION})

        report = verify_erd_cache(self.WORDS, cache, sc, ERD_ALL)

        self.assertEqual(len(report), 1)
        root = report[0]
        self.assertEqual(root['status'], 'incomplete')
        self.assertFalse(root['complete'])
        self.assertLess(root['reconstructed'], root['best_score'])

    def test_inconsistent_root_reports_mismatch(self):
        # Subtree (g1, g2) is internally consistent at 1.5 each, but the
        # root's stored score (2.0) doesn't match the reconstruction from
        # that subtree (13/6 ~= 2.1667) — a provable contradiction.
        sc = MemoryScoreCache()
        sc.set_scope('test')
        sc.write(ScoreCache.encode_subset(self.WORDS), ERD_ALL, 'alpha', 2.0)
        sc.write(ScoreCache.encode_subset(['c1', 'c2']), ERD_ALL, 'c1', 1.5)
        sc.write(ScoreCache.encode_subset(['c3', 'c4']), ERD_ALL, 'c3', 1.5)

        report = verify_erd_cache(self.WORDS, self._full_cache(), sc, ERD_ALL)

        root = report[0]
        self.assertEqual(root['status'], 'mismatch')
        self.assertAlmostEqual(root['reconstructed'], self.ROOT_SCORE)
        self.assertAlmostEqual(root['best_score'], 2.0)

    def test_partial_sum_exceeding_best_score_is_mismatch(self):
        # Even with g2 uncached, g1 alone (0.5) plus the 'beta' singleton
        # (1/6) plus the base 1.0 already exceeds a too-low claimed score.
        sc = MemoryScoreCache()
        sc.set_scope('test')
        sc.write(ScoreCache.encode_subset(self.WORDS), ERD_ALL, 'alpha', 1.5)
        sc.write(ScoreCache.encode_subset(['c1', 'c2']), ERD_ALL, 'c1', 1.5)
        cache = _MultiPartitionCache({
            'alpha': self.ROOT_PARTITION, 'c1': self.PAIR1_PARTITION,
        })

        report = verify_erd_cache(self.WORDS, cache, sc, ERD_ALL)

        root = report[0]
        self.assertEqual(root['status'], 'mismatch')
        self.assertFalse(root['complete'])
        self.assertGreater(root['reconstructed'], root['best_score'])

    def test_max_nodes_caps_report_length(self):
        sc = MemoryScoreCache()
        sc.set_scope('test')
        sc.write(ScoreCache.encode_subset(self.WORDS), ERD_ALL,
                 'alpha', self.ROOT_SCORE)
        sc.write(ScoreCache.encode_subset(['c1', 'c2']), ERD_ALL, 'c1', 1.5)
        sc.write(ScoreCache.encode_subset(['c3', 'c4']), ERD_ALL, 'c3', 1.5)

        report = verify_erd_cache(self.WORDS, self._full_cache(), sc, ERD_ALL,
                                   max_nodes=1)

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['status'], 'match')


# ---------------------------------------------------------------------------
# _erd_solve_scores
# ---------------------------------------------------------------------------

class TestERDSolveScores(unittest.TestCase):
    """
    Tests for _erd_solve_scores, with focus on the singleton subgroup
    edge case: min_expected_guesses never writes n==1 results to cache
    (it returns 1.0 as a base case), so _erd_solve_scores must handle
    them inline rather than via a cache read.
    """

    def _soln_with_words(self, words, sc):
        import types
        cache = ResponseCache(words)
        return types.SimpleNamespace(
            current_words=list(words),
            score_cache=sc,
            cache=cache,
        ), cache

    def test_singleton_subgroups_do_not_cause_none(self):
        """
        Every pair produces non-all-green singletons (k=1, sg[0]!=word).
        After min_expected_guesses populates the root, _erd_solve_scores
        must succeed — not return None — despite singletons being absent
        from the cache.
        """
        words = ANSWERS[:2]
        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), words)
            soln, cache = self._soln_with_words(words, sc)
            erd = min_expected_guesses(words, cache, sc, guesses=words)
            self.assertAlmostEqual(erd, 1.5)
            # Singletons are NOT in cache — confirm that
            for w in words:
                hit = sc.read(ScoreCache.encode_subset([w]), ERD_ALL)
                self.assertIsNone(hit, f"singleton {w} should not be cached")
            # _erd_solve_scores must still work
            scores = _erd_solve_scores(soln)
            self.assertIsNotNone(scores, "must not fail on singleton subgroups")
            self.assertEqual(len(scores), len(words))

    def test_all_candidates_return_correct_cost_for_pair(self):
        """For any 2-word set both candidates cost exactly 1.5."""
        words = ANSWERS[:2]
        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), words)
            soln, cache = self._soln_with_words(words, sc)
            min_expected_guesses(words, cache, sc, guesses=words)
            scores = _erd_solve_scores(soln)
            self.assertIsNotNone(scores)
            for word, cost in scores:
                self.assertAlmostEqual(cost, 1.5, places=10,
                                       msg=f"cost for {word}")

    def test_results_sorted_ascending(self):
        """Scores are returned lowest-first (best play first)."""
        words = ANSWERS[:5]
        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), words)
            soln, cache = self._soln_with_words(words, sc)
            min_expected_guesses(words, cache, sc, guesses=words)
            scores = _erd_solve_scores(soln)
            self.assertIsNotNone(scores)
            costs = [c for _, c in scores]
            self.assertEqual(costs, sorted(costs))

    def test_best_word_matches_root_cache(self):
        """The top-ranked word must match what min_expected_guesses chose."""
        words = ANSWERS[:5]
        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), words)
            soln, cache = self._soln_with_words(words, sc)
            min_expected_guesses(words, cache, sc, guesses=words)
            root_hit = sc.read(ScoreCache.encode_subset(words), ERD_ALL)
            self.assertIsNotNone(root_hit)
            scores = _erd_solve_scores(soln)
            self.assertIsNotNone(scores)
            best_word, best_cost = scores[0]
            self.assertEqual(best_word, root_hit[0])
            self.assertAlmostEqual(best_cost, root_hit[1], places=10)


# ---------------------------------------------------------------------------
# _multistep_stats participates in the transparent cache
# ---------------------------------------------------------------------------

class TestMultistepStatsCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    def test_step1_scores_persisted_to_sqlite(self):
        soln = make_solution(db_path=self.db)
        _multistep_stats("crane", soln)
        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(soln.current_words)
        rows = sc.read_scores(branch_key, "entropy_gain")
        self.assertIsNotNone(rows)
        self.assertIn("crane", dict(rows))

    def test_step1_scores_loaded_from_sqlite_on_second_session(self):
        soln1 = make_solution(db_path=self.db)
        stats1 = _multistep_stats("crane", soln1)

        soln2 = make_solution(db_path=self.db)
        stats2 = _multistep_stats("crane", soln2)
        self.assertAlmostEqual(stats1['step1'], stats2['step1'], places=10)

    def test_step1_scores_reused_from_word_scores(self):
        # Pre-populate word_scores via compute_scores; _multistep_stats
        # should not overwrite with a different value.
        soln = make_solution(db_path=self.db)
        soln.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        cached_entropy = soln.word_scores["crane"][ScoringMethod.ENTROPY_GAIN]

        stats = _multistep_stats("crane", soln)
        self.assertAlmostEqual(stats['step1'], cached_entropy, places=10)

    def test_mid_game_scores_persisted_subset_scoped(self):
        soln = make_solution(db_path=self.db)
        pattern = calculate_response("crane", "slate")
        soln.apply_guess("crane", pattern)
        self.assertFalse(soln._is_full_game())

        _multistep_stats("slate", soln)

        sc = ScoreCache(self.db, ANSWERS)
        branch_key = ScoreCache.encode_subset(soln.current_words)
        cached = sc.read_scores(branch_key, "entropy_gain")
        self.assertIsNotNone(cached)
        self.assertIn("slate", dict(cached))


# ---------------------------------------------------------------------------
# Bug: min_expected_guesses derives policy from guesses-is-not-None, ignoring
# the caller's intended policy.  The fix adds an explicit policy= parameter.
# ---------------------------------------------------------------------------

class TestERDPolicyParameter(unittest.TestCase):
    """
    min_expected_guesses must accept and honour an explicit policy= argument
    so callers can separate ERD_ANSWERS, ERD_ALL, ERD_CONSTRAINED in
    the cache independently of whether a guesses list is supplied.

    All tests in this class currently fail because no policy= parameter exists.
    """

    def setUp(self):
        self.cache = ResponseCache(ANSWERS)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmpdir.name, 'test.sqlite3')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _sc(self):
        return ScoreCache(self.db, ANSWERS)

    def test_explicit_policy_stored_not_derived(self):
        """Passing policy=ERD_ANSWERS with guesses= stores under ERD_ANSWERS, not ERD_ALL."""
        sc = self._sc()
        subset = ANSWERS[:4]
        key = ScoreCache.encode_subset(subset)

        min_expected_guesses(subset, self.cache, sc,
                             guesses=subset, policy=ERD_ANSWERS)

        self.assertIsNotNone(sc.read(key, ERD_ANSWERS),
                             "result must be stored under the explicit policy")
        self.assertIsNone(sc.read(key, ERD_ALL),
                          "result must NOT be stored under the derived policy")

    def test_possible_answers_results_readable_by_solve(self):
        """
        Results written with policy=ERD_ANSWERS (POSSIBLE_ANSWERS solver) are
        readable by _erd_solve_scores under policy=ERD_ANSWERS.

        With the bug, the solver calls min_expected_guesses(guesses=words) which
        stores under ERD_ALL; _erd_solve_scores looks under ERD_ANSWERS → miss.
        """
        words = ANSWERS[:5]
        sc = self._sc()
        import types
        soln = types.SimpleNamespace(
            current_words=list(words),
            score_cache=sc,
            cache=ResponseCache(words),
        )

        # Simulate POSSIBLE_ANSWERS solver: guesses=current_words, policy=ERD_ANSWERS
        min_expected_guesses(words, soln.cache, sc,
                             guesses=words, policy=ERD_ANSWERS)

        scores = _erd_solve_scores(soln, score_cache=sc, policy=ERD_ANSWERS)
        self.assertIsNotNone(scores,
                             "POSSIBLE_ANSWERS ERD must be readable after solver completes")

    def test_erd_all_and_erd_answers_do_not_share_cache_slots(self):
        """
        erd_answers and erd_all are stored under separate cache slots.
        Writing one does not contaminate the other.
        """
        subset = ANSWERS[:5]
        sc = self._sc()
        key = ScoreCache.encode_subset(subset)

        # Write only erd_answers (no guesses=)
        min_expected_guesses(subset, self.cache, sc, policy=ERD_ANSWERS)
        self.assertIsNotNone(sc.read(key, ERD_ANSWERS),
                             "erd_answers must be stored after first call")
        self.assertIsNone(sc.read(key, ERD_ALL),
                          "erd_all must be untouched after erd_answers-only call")

        # Now write erd_all (guesses=GUESSES, different vocabulary)
        min_expected_guesses(subset, self.cache, sc,
                             guesses=GUESSES, policy=ERD_ALL)
        self.assertIsNotNone(sc.read(key, ERD_ALL),
                             "erd_all must be stored after second call")
        self.assertIsNotNone(sc.read(key, ERD_ANSWERS),
                             "erd_answers must still be present — independent slot")


# ---------------------------------------------------------------------------
# Bug: _multistep_stats calls min_expected_guesses without guesses=, so it
# always uses policy=ERD_ANSWERS regardless of all_words mode.
# After the fix it passes guesses=all_words and policy=ERD_ALL.
# ---------------------------------------------------------------------------

class TestMultistepStatsERDPolicy(unittest.TestCase):
    """
    _multistep_stats surfaces ERD purely by reading the cache namespace that
    matches the current mode — ERD_ALL for any-word, ERD_ANSWERS for the
    default (possible-answers) mode, ERD_CONSTRAINED (a transient/supplied
    MemoryScoreCache, never SQLite) for hard mode. It must read from the
    *correct* namespace for the mode in play, ignoring values cached under
    any other policy — a cross-policy hit would surface a number computed
    against the wrong guess vocabulary.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmpdir.name, 'test.sqlite3')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mid_game_soln(self):
        """Apply 'piano' → leaves 6 words including subgroup-producing 'heart'."""
        soln = make_solution(db_path=self.db)
        # Guessing 'piano' against answer 'slate' leaves six words: slate,
        # trace, stale, least, heart, share.  'heart' then splits them into
        # a 2-word subgroup (slate, stale) plus singletons — exactly the
        # non-trivial shape these tests need.
        pattern = calculate_response("piano", "slate")
        soln.apply_guess("piano", pattern)
        self.assertFalse(soln._is_full_game())
        self.assertGreaterEqual(len(soln.current_words), 3,
                                "setup must leave 3+ words for non-trivial subgroups")
        return soln

    def _find_subgroup_key(self, word, remaining):
        """Return ScoreCache key for the first subgroup of size >= 2."""
        groups = ResponseCache(ANSWERS).group_words(word, remaining)
        for sg in groups.values():
            if len(sg) >= 2:
                return ScoreCache.encode_subset(sg)
        return None

    def test_all_words_mode_surfaces_from_erd_all_not_erd_answers(self):
        """
        In any-word mode (all_words supplied), _multistep_stats must read the
        subgroup ERD from ERD_ALL — and ignore a value parked under
        ERD_ANSWERS for the same subgroup, which was computed against a
        different (answers-only) guess vocabulary and would be the wrong number.
        """
        soln = self._mid_game_soln()
        key = self._find_subgroup_key("heart", soln.current_words)
        self.assertIsNotNone(key, "setup must produce a subgroup with k>=2")

        sc = ScoreCache(self.db, ANSWERS)
        sc.write(key, ERD_ANSWERS, 'wrong-vocab-value', 9.0)
        sc.write(key, ERD_ALL, 'crane', 1.5)

        st = _multistep_stats("heart", soln, all_words=GUESSES)
        self.assertIsNotNone(st['erd'],
                             "a cached ERD_ALL value must be surfaced")
        self.assertNotAlmostEqual(st['erd'], 1.0 + 9.0 * (2 / len(soln.current_words)),
                                  msg="must not surface the ERD_ANSWERS value")

    def test_default_mode_surfaces_from_erd_answers_not_erd_all(self):
        """
        Without all_words, _multistep_stats must read ERD_ANSWERS — and
        ignore a value parked under ERD_ALL for the same subgroup, which
        was computed against the full guess vocabulary and would be wrong here.
        """
        soln = self._mid_game_soln()
        key = self._find_subgroup_key("heart", soln.current_words)
        self.assertIsNotNone(key, "setup must produce a subgroup with k>=2")

        sc = ScoreCache(self.db, ANSWERS)
        sc.write(key, ERD_ALL, 'wrong-vocab-value', 9.0)
        sc.write(key, ERD_ANSWERS, 'crane', 1.5)

        st = _multistep_stats("heart", soln)  # all_words=None
        self.assertIsNotNone(st['erd'],
                             "a cached ERD_ANSWERS value must be surfaced")
        self.assertNotAlmostEqual(st['erd'], 1.0 + 9.0 * (2 / len(soln.current_words)),
                                  msg="must not surface the ERD_ALL value")

    def test_constraint_compliant_mode_does_not_fall_through_to_erd_all(self):
        """
        constraint_compliant=True must steer ERD to the hard-mode vocabulary
        and policy — not silently fall through to ERD_ALL just because
        all_words happens to be non-empty.  And because hard-mode ERDs are
        path-dependent (the eligible guess set depends on the exact
        constraints accumulated so far), they must never be written into
        the persisted, cross-game SQLite cache under any policy name.
        """
        soln = self._mid_game_soln()
        key = self._find_subgroup_key("heart", soln.current_words)
        self.assertIsNotNone(key, "setup must produce a subgroup with k>=2")

        _multistep_stats("heart", soln, constraint_compliant=True, all_words=GUESSES)

        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read(key, ERD_ALL),
                          "hard-mode ERD must not be computed/cached as ERD_ALL "
                          "merely because all_words was supplied")
        self.assertIsNone(sc.read(key, ERD_CONSTRAINED),
                          "hard-mode ERD values are path-dependent and must never "
                          "be persisted to the cross-game SQLite cache")


# ---------------------------------------------------------------------------
# _multistep_stats ERD must never block the interactive (main) thread.
#
# Exact ERD computation (min_expected_guesses) over a large guess vocabulary
# is combinatorially expensive — a single subgroup can take tens of seconds.
# That cost belongs solely to the background ERDSolver, which uses its own
# short deadlines to detect when it has bitten off more than it can chew.
# The foreground must instead simply *surface* whatever the solver has
# already cached: an instant, recursion-free cache read per subgroup. If a
# subgroup isn't cached yet, ERD is reported as unavailable (None) rather
# than computed live — exactly mirroring how print_status's ERD tag already
# works. As the solver (running continuously in the background) populates
# more of the cache over time, more subgroups become surfaceable for free.
# ---------------------------------------------------------------------------

class TestMultistepStatsERDNonBlocking(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmpdir.name, 'test.sqlite3')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mid_game_soln(self):
        soln = make_solution(db_path=self.db)
        pattern = calculate_response("piano", "slate")
        soln.apply_guess("piano", pattern)
        self.assertFalse(soln._is_full_game())
        return soln

    def _subgroup(self, word, remaining):
        """Return the first subgroup of size >= 2 that 'word' produces."""
        groups = ResponseCache(ANSWERS).group_words(word, remaining)
        for sg in groups.values():
            if len(sg) >= 2:
                return list(sg)
        return None

    def _refuse_to_compute(self):
        """Patch min_expected_guesses to fail the test if _multistep_stats
        ever calls it — the foreground must be a pure cache surface."""
        return mock.patch(
            'wordle.min_expected_guesses',
            side_effect=AssertionError(
                "_multistep_stats must never invoke min_expected_guesses — "
                "ERD computation belongs to the background solver only"))

    def test_uncached_subgroup_yields_none_without_computing(self):
        """A cache miss must surface as 'not yet available' (None), not
        trigger a live computation that could block for tens of seconds."""
        soln = self._mid_game_soln()
        sg = self._subgroup("heart", soln.current_words)
        self.assertIsNotNone(sg, "setup must produce a subgroup with k>=2")

        with self._refuse_to_compute():
            st = _multistep_stats("heart", soln, all_words=GUESSES)

        self.assertIsNone(st['erd'],
                          "ERD must be None when a subgroup isn't cached yet")

    def test_cached_subgroup_is_surfaced_instantly(self):
        """A pre-cached subgroup ERD must be read straight from the cache —
        contributing to the displayed ERD without any fresh computation."""
        soln = self._mid_game_soln()
        sg = self._subgroup("heart", soln.current_words)
        self.assertIsNotNone(sg)
        n = len(soln.current_words)
        key = ScoreCache.encode_subset(sg)

        sc = ScoreCache(self.db, ANSWERS)
        sc.write(key, ERD_ALL, 'crane', 1.5)

        with self._refuse_to_compute():
            st = _multistep_stats("heart", soln, all_words=GUESSES)

        self.assertIsNotNone(st['erd'],
                             "a fully-cached subgroup tree must surface a value")
        # erd = 1.0 + sum over non-trivial subgroups of (k/n) * cached_value;
        # the cached subgroup contributes exactly (len(sg)/n) * 1.5.
        self.assertGreaterEqual(st['erd'], 1.0 + (len(sg) / n) * 1.5 - 1e-9)

    def test_constraint_compliant_uses_supplied_cache_only(self):
        """Hard mode must surface from the caller-supplied erd_cache (the
        long-lived MemoryScoreCache shared with the solver) without ever
        computing — same non-blocking contract as any-word mode."""
        soln = self._mid_game_soln()
        sg = self._subgroup("heart", soln.current_words)
        self.assertIsNotNone(sg)
        n = len(soln.current_words)
        key = ScoreCache.encode_subset(sg)

        mc = MemoryScoreCache()
        eligible = soln.constraint_compliant_words(GUESSES)
        mc.set_scope(MemoryScoreCache.fingerprint_vocabulary(eligible))
        mc.write(key, ERD_CONSTRAINED, 'crane', 1.5)

        with self._refuse_to_compute():
            st = _multistep_stats("heart", soln, constraint_compliant=True,
                                  all_words=GUESSES, erd_cache=mc)

        self.assertIsNotNone(st['erd'],
                             "a fully-cached subgroup tree must surface a value")
        self.assertGreaterEqual(st['erd'], 1.0 + (len(sg) / n) * 1.5 - 1e-9)


# ---------------------------------------------------------------------------
# MemoryScoreCache scoping by eligible-vocabulary fingerprint
#
# Hard-mode ERD results are valid only for the exact eligible-guess
# vocabulary (the word set surviving every accumulated Restriction) that
# produced them — NOT merely for a particular `current_words` snapshot.
# Two different guess histories can coincidentally produce the same
# `current_words` while differing in eligible vocabulary (and vice versa
# after undo/replay), so the cache must be keyed on a fingerprint of the
# vocabulary itself.  Entries written under one scope must be invisible
# under another, and reachable again "for free" once that scope recurs —
# without any explicit eviction.
# ---------------------------------------------------------------------------

class TestMemoryScoreCacheScoping(unittest.TestCase):

    def test_close_is_a_harmless_no_op(self):
        """Unlike ScoreCache.close(), MemoryScoreCache holds no SQLite
        connection — close() must be safe to call (ERDSolver's run() calls
        it unconditionally) and leave the cache usable afterwards."""
        mc = MemoryScoreCache()
        mc.set_scope('test-scope')
        mc.write(b'key', 'full', 'crane', 1.5)
        mc.close()
        self.assertEqual(mc.read(b'key', 'full'), ('crane', 1.5))

    def test_fingerprint_is_order_independent(self):
        """Fingerprint depends only on the set of words, not their order."""
        a = MemoryScoreCache.fingerprint_vocabulary(["crane", "slate", "trace"])
        b = MemoryScoreCache.fingerprint_vocabulary(["trace", "crane", "slate"])
        self.assertEqual(a, b)

    def test_fingerprint_distinguishes_different_vocabularies(self):
        a = MemoryScoreCache.fingerprint_vocabulary(["crane", "slate", "trace"])
        b = MemoryScoreCache.fingerprint_vocabulary(["crane", "slate", "stale"])
        self.assertNotEqual(a, b)

    def test_write_invisible_under_different_scope(self):
        """An entry written under one vocabulary scope is a miss under another,
        even for the identical (branch_key, policy) — preventing false hits
        when the eligible-guess vocabulary changes but current_words coincides."""
        mc = MemoryScoreCache()
        key = ScoreCache.encode_subset(["crane", "slate"])

        mc.set_scope(MemoryScoreCache.fingerprint_vocabulary(GUESSES))
        mc.write(key, ERD_CONSTRAINED, 'heart', 1.5)
        self.assertEqual(mc.read(key, ERD_CONSTRAINED), ('heart', 1.5))

        mc.set_scope(MemoryScoreCache.fingerprint_vocabulary(ANSWERS))
        self.assertIsNone(mc.read(key, ERD_CONSTRAINED),
                          "entry from a different vocabulary scope must not be "
                          "visible — it was computed against a different "
                          "eligible-guess set and would be a false hit")

    def test_entries_reachable_again_when_scope_recurs(self):
        """Switching back to a previously-seen vocabulary makes its entries
        reachable again at no cost — exactly the reuse undo/replay wants."""
        mc = MemoryScoreCache()
        key = ScoreCache.encode_subset(["crane", "slate"])
        fp_guesses = MemoryScoreCache.fingerprint_vocabulary(GUESSES)
        fp_answers = MemoryScoreCache.fingerprint_vocabulary(ANSWERS)

        mc.set_scope(fp_guesses)
        mc.write(key, ERD_CONSTRAINED, 'heart', 1.5)

        mc.set_scope(fp_answers)
        mc.write(key, ERD_CONSTRAINED, 'stale', 2.0)

        mc.set_scope(fp_guesses)
        self.assertEqual(mc.read(key, ERD_CONSTRAINED), ('heart', 1.5),
                         "revisiting a scope must surface its own entries again")

    def test_unscoped_cache_keys_by_none(self):
        """Before set_scope is ever called, reads/writes use a stable (None)
        scope rather than raising — a freshly constructed cache is usable."""
        mc = MemoryScoreCache()
        key = ScoreCache.encode_subset(["crane", "slate"])
        mc.write(key, ERD_CONSTRAINED, 'heart', 1.5)
        self.assertEqual(mc.read(key, ERD_CONSTRAINED), ('heart', 1.5))


# ---------------------------------------------------------------------------
# Bug: _erd_solve_scores only iterates soln.current_words — in any-word mode
# non-answer candidates (e.g. 'brain', 'train') are never evaluated.
# After the fix, passing guesses= includes them.
# ---------------------------------------------------------------------------

class TestERDSolveScoresNonAnswerCandidates(unittest.TestCase):
    """
    _erd_solve_scores must accept an optional guesses= parameter.  When supplied,
    words outside current_words (non-answers) that have pre-solved subgroup ERDs
    must appear in the returned ranking.
    """

    def _soln(self, words, sc):
        import types
        return types.SimpleNamespace(
            current_words=list(words),
            score_cache=sc,
            cache=ResponseCache(words),
        )

    def test_non_answer_word_appears_when_guesses_supplied(self):
        """
        Non-answer words are eligible for the ranking when guesses= is supplied
        (the bug was that _erd_solve_scores only ever scored answer words).

        Assert that *some* non-answer appears — not a specific one.  Candidate
        ordering / ERD-lower-bound pruning may legitimately drop any individual
        weak guess from the ranking (an ERD-pruned candidate's subgroups are
        never cached, so it is skipped), and that is orthogonal to the
        invariant under test.
        """
        answers = ANSWERS[:5]

        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), answers)
            soln = self._soln(answers, sc)

            # Pre-solve all subgroup ERDs using the full GUESSES vocabulary
            min_expected_guesses(answers, soln.cache, sc,
                                 guesses=GUESSES, policy=ERD_ALL)

            scores = _erd_solve_scores(soln, score_cache=sc,
                                       policy=ERD_ALL, guesses=GUESSES)
            self.assertIsNotNone(scores)
            result_words = [w for w, _ in scores]
            non_answers = [w for w in result_words if w not in ANSWERS]
            self.assertTrue(
                non_answers,
                "at least one non-answer must appear in the ERD ranking when "
                f"guesses= is supplied; got only answers: {result_words}")

    def test_without_guesses_only_answer_words_returned(self):
        """
        Default behaviour (no guesses=) still limits candidates to current_words.
        This test must continue to pass after the fix.
        """
        answers = ANSWERS[:5]
        non_answer = next(w for w in GUESSES if w not in ANSWERS)

        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), answers)
            soln = self._soln(answers, sc)
            min_expected_guesses(answers, soln.cache, sc, guesses=answers)
            scores = _erd_solve_scores(soln, score_cache=sc)
            self.assertIsNotNone(scores)
            result_words = [w for w, _ in scores]
            for w in result_words:
                self.assertIn(w, answers,
                              f"without guesses=, only answer words should appear")


# ---------------------------------------------------------------------------
# Regression: ERD-lower-bound pruning (b95) means min_expected_guesses no
# longer recurses into every candidate's subgroups, so the cache can be
# missing subgroup entries for answer-word candidates that were pruned outright.
# _erd_solve_scores must skip those candidates rather than aborting the
# whole ranking.
# ---------------------------------------------------------------------------

class _FixedPartitionCache:
    """Stub for soln.cache: every word partitions current_words the same
    way (`full_groups`), except `split_word`, which produces a single
    oversized group (`split_groups`) whose ERD was never cached — as if
    ERD-lower-bound pruning skipped recursing into it entirely."""

    def __init__(self, answer_words, full_groups, split_word, split_groups):
        self.answer_words = answer_words
        self._full_groups = full_groups
        self._split_word = split_word
        self._split_groups = split_groups

    def group_words(self, word, subset):
        if word == self._split_word:
            return self._split_groups
        return self._full_groups


class TestERDSolveScoresUncachedSubgroupSkipped(unittest.TestCase):

    def test_candidate_with_uncached_subgroup_is_skipped_not_aborted(self):
        words = ["alpha", "beta", "c1", "c2", "c3", "c4"]
        # Every word except "beta" splits current_words into two cached
        # size-2 groups plus two self/other singletons (sums to 6).
        full_groups = {
            'g1': ["c1", "c2"],
            'g2': ["c3", "c4"],
            'self': ["alpha"],
            'other': ["beta"],
        }
        # "beta" instead produces one big size-4 group whose ERD is not
        # in the cache — the pruned candidate.
        split_groups = {
            'big': ["c1", "c2", "c3", "c4"],
            'self': ["beta"],
            'other': ["alpha"],
        }
        cache = _FixedPartitionCache(words, full_groups, "beta", split_groups)

        mc = MemoryScoreCache()
        mc.set_scope('test-scope')
        mc.write(ScoreCache.encode_subset(["c1", "c2"]), ERD_ALL, "x", 1.5)
        mc.write(ScoreCache.encode_subset(["c3", "c4"]), ERD_ALL, "x", 1.5)
        # Deliberately do NOT write an entry for ["c1","c2","c3","c4"] —
        # this is the "pruned, never recursed into" subgroup.

        import types
        soln = types.SimpleNamespace(current_words=words, cache=cache)

        scores = _erd_solve_scores(soln, score_cache=mc, policy=ERD_ALL)
        self.assertIsNotNone(
            scores, "a fully-cached candidate must still produce a ranking")
        result_words = [w for w, _ in scores]
        self.assertNotIn("beta", result_words,
                         "candidate with an uncached subgroup must be skipped")
        for w in ["alpha", "c1", "c2", "c3", "c4"]:
            self.assertIn(w, result_words)


class TestERDSolverKeepsWorking(unittest.TestCase):

    @staticmethod
    def _word(i):
        return f"w{i:04d}"  # 5 ASCII chars, satisfies encode_subset's slicing

    def test_no_ready_message_when_superseded_mid_root_computation(self):
        """If stop() fires while the (slow, uninterruptible) root computation
        is in flight, the solver must not announce a result on the way out —
        a superseded solver racing a fresh one to print the same value is
        exactly what produces duplicate "[ERD ready]" lines at the prompt."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

            @staticmethod
            def group_words(word, current_words):
                return {}

            @staticmethod
            def group_counts(word, current_words):
                return {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')

        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)

        def cancel_during_root(remaining, cache, sc, deadline=None,
                                guesses=None, policy=None,
                                progress_callback=None,
                                cancel_check=None, heartbeat=None,
                                budget=None, pattern_matrix=None):
            solver.stop()  # e.g. the user moved on; a fresh solver supersedes this one
            return 1.8

        printed = []
        solver._rcache = FakeResponseCache()
        with mock.patch('wordle.min_expected_guesses', side_effect=cancel_during_root), \
             mock.patch('builtins.print', side_effect=lambda *a, **k: printed.append(a)):
            solver._scan(score_cache)

        self.assertFalse(
            any('ERD ready' in str(a) for a in printed),
            "a superseded solver must not print a stale [ERD ready] announcement")

    def test_ranking_uses_word_scores_cache(self):
        """If 's' (cmd_solve / compute_scores_multi) already populated
        word_scores for this exact position, _ranked_root_guesses must use
        those cached MAX_GROUP_SIZE/ENTROPY_GAIN scores directly instead of
        recomputing via rcache.group_counts."""
        words = [self._word(i) for i in range(10)]

        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            score_cache = ScoreCache(tmp.name, words)

            # Simulate what cmd_solve does: compute_scores_multi populates
            # and persists word_scores for this position.
            cache = ResponseCache(words, score_cache)
            soln = Solution(words, words, cache=cache, score_cache=score_cache)
            soln.compute_scores_multi(
                words, [ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN])

            class ExplodingResponseCache:
                @staticmethod
                def group_counts(word, subset):
                    raise AssertionError(
                        "ranking should use cached word_scores, not recompute")

            solver = ERDSolver(words, words, words, None,
                               policy=ERD_ALL, persist=True)
            solver._rcache = ExplodingResponseCache()

            ranked = solver._ranked_root_guesses(score_cache)
            self.assertEqual(set(ranked), set(words))
        finally:
            score_cache.close()
            os.unlink(tmp.name)

    def test_operational_error_in_pre_loop_read_is_caught(self):
        """sqlite3.OperationalError from score_cache.read() at the top of
        _scan must be caught and printed rather than crashing the thread."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

        class ErrorScoreCache(MemoryScoreCache):
            def read(self, branch_key, policy):
                raise sqlite3.OperationalError("disk I/O error")

        score_cache = ErrorScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)
        solver._rcache = FakeResponseCache()

        printed = []
        with mock.patch('builtins.print', side_effect=lambda *a, **k: printed.append(a)):
            solver._scan(score_cache)  # must not raise

        self.assertFalse(
            any('OperationalError' in str(a) for a in printed),
            "OperationalError must be swallowed silently — no print to stderr")

    def test_run_sets_rcache_from_thread_private_connection(self):
        """run() must replace self._rcache with a thread-private ResponseCache
        before calling _scan, so _ranked_root_guesses and min_expected_guesses
        never use the main thread's SQLite connection."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)

        self.assertIsNone(solver._rcache,
            "_rcache should be None before run() — it is set inside run()")

        rcaches_seen = []
        original_scan = solver._scan
        def spy_scan(sc):
            rcaches_seen.append(solver._rcache)
            # Don't actually run the scan.
        solver._scan = spy_scan

        solver.run()
        self.assertEqual(len(rcaches_seen), 1)
        self.assertIsInstance(rcaches_seen[0], ResponseCache,
            "run() must set _rcache to a thread-private ResponseCache before calling _scan")

    def test_cumulative_time_survives_pause_restart(self):
        """word_stats (and root_done) reset to empty at the top of every
        while-loop pass — a pause/resume mid-scan throws away the previous
        pass's per-word table. cumulative_cpu_s/cumulative_wall_s must NOT
        be reset, so they keep growing across passes even though word_stats
        only ever shows the current pass."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

            @staticmethod
            def group_words(word, current_words):
                return {}

            @staticmethod
            def group_counts(word, current_words):
                return {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)
        solver._rcache = FakeResponseCache()

        # Monotonically increasing fake clocks so wall_elapsed/cpu_elapsed
        # are always positive, regardless of how many times each is called.
        wall_counter = [0.0]
        cpu_counter = [0.0]

        def fake_time():
            wall_counter[0] += 1.0
            return wall_counter[0]

        def fake_thread_time():
            cpu_counter[0] += 1.0
            return cpu_counter[0]

        snapshot = {}

        def fake_min_expected(remaining, cache, sc, deadline=None,
                               guesses=None, policy=None,
                               progress_callback=None, cancel_check=None,
                               heartbeat=None, budget=None,
                               pattern_matrix=None):
            if 'pass' not in snapshot:
                snapshot['pass'] = 1
                progress_callback(1, len(words), words[0], 1.5)
                return None  # simulate pause: caller waits and retries
            snapshot['pass'] = 2
            snapshot['cpu_before_pass2'] = solver.cumulative_cpu_s
            snapshot['wall_before_pass2'] = solver.cumulative_wall_s
            progress_callback(1, len(words), words[0], 1.5)
            progress_callback(2, len(words), words[1], 1.4)
            return 1.4  # final result: while-loop exits

        with mock.patch('wordle.time.time', side_effect=fake_time), \
             mock.patch('wordle.time.thread_time', side_effect=fake_thread_time), \
             mock.patch('wordle.min_expected_guesses', side_effect=fake_min_expected), \
             mock.patch('builtins.print'):
            solver._scan(score_cache)

        # Pass 1's single progress callback already accumulated time before
        # pass 2 started — proving cumulative totals aren't reset alongside
        # word_stats/root_done at the top of the while loop.
        self.assertGreater(snapshot['cpu_before_pass2'], 0)
        self.assertGreater(snapshot['wall_before_pass2'], 0)

        # word_stats only reflects pass 2 (reset wiped pass 1's entry).
        self.assertEqual(len(solver.word_stats), 2)

        # But the cumulative totals include both passes' contributions.
        self.assertGreater(solver.cumulative_cpu_s, snapshot['cpu_before_pass2'])
        self.assertGreater(solver.cumulative_wall_s, snapshot['wall_before_pass2'])

    def test_targeted_scan_periodic_report(self):
        """When constructed with last_guess, _scan must periodically print a
        'Targeted scan of WORD <pattern>' report — same shape as
        BranchPrecacheSolver's 'Root-word scan' report — driven by the
        heartbeat callback, every 30s."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

            @staticmethod
            def group_words(word, current_words):
                return {}

            @staticmethod
            def group_counts(word, current_words):
                return {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache,
                           last_guess=("salet", ["gray"] * 5))
        solver._rcache = FakeResponseCache()

        counter = itertools.count()

        def fake_min_expected(remaining, cache, sc, deadline=None,
                               guesses=None, policy=None,
                               progress_callback=None, cancel_check=None,
                               heartbeat=None, budget=None,
                               pattern_matrix=None):
            heartbeat()
            return 1.5

        out = io.StringIO()
        with mock.patch('wordle.time.time', side_effect=lambda: next(counter) * 31), \
             mock.patch('wordle.min_expected_guesses', side_effect=fake_min_expected):
            with redirect_stdout(out):
                solver._scan(score_cache)

        text = out.getvalue()
        self.assertIn("Targeted scan of SALET -----", text)
        self.assertIn(f"{len(words)} words | ERD: computing...", text)
        self.assertIn("0/0, 0 culled", text)
        self.assertRegex(text, r'\d+s, \d+ hit/\d+ miss')

    def test_no_targeted_scan_report_without_last_guess(self):
        """last_guess defaults to None (e.g. tests that construct an
        ERDSolver directly) — _maybe_print must be a no-op in that case,
        even when 30+ seconds appear to have elapsed."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words

            def __init__(self):
                self._cache = {}

            @staticmethod
            def group_words(word, current_words):
                return {}

            @staticmethod
            def group_counts(word, current_words):
                return {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)
        solver._rcache = FakeResponseCache()

        counter = itertools.count()

        def fake_min_expected(remaining, cache, sc, deadline=None,
                               guesses=None, policy=None,
                               progress_callback=None, cancel_check=None,
                               heartbeat=None, budget=None,
                               pattern_matrix=None):
            heartbeat()
            return 1.5

        out = io.StringIO()
        with mock.patch('wordle.time.time', side_effect=lambda: next(counter) * 31), \
             mock.patch('wordle.min_expected_guesses', side_effect=fake_min_expected):
            with redirect_stdout(out):
                solver._scan(score_cache)

        self.assertNotIn("Targeted scan", out.getvalue())


class TestERDSolverBudget(unittest.TestCase):
    """ERDSolver must thread its remaining-guess budget into
    min_expected_guesses (issue #79): a recommendation is only announced if
    it is guaranteed to finish within the guesses actually remaining, and a
    proven-unsolvable position is reported instead of silently announcing an
    unconstrained (possibly game-losing) recommendation."""

    # Same fixture as TestDepthLimitedERD: 8 words sharing suffix "ound",
    # each distinguished by a first letter absent from "ound" — answers-only
    # guessing isolates one word per guess, worst-case line 8, ERD 4.5.
    LINEAR = ["bound", "found", "hound", "mound",
              "pound", "round", "sound", "wound"]

    def _solver(self, budget):
        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')
        solver = ERDSolver(self.LINEAR, self.LINEAR, self.LINEAR, None,
                           policy=ERD_ALL, persist=False,
                           seed_mem_cache=score_cache, budget=budget)
        solver._rcache = ResponseCache(self.LINEAR)
        return solver, score_cache

    def test_feasible_budget_announces_ready(self):
        """budget=8 exactly fits LINEAR's worst-case line: the recommendation
        is announced normally, unlabeled as best-effort."""
        solver, score_cache = self._solver(budget=8)
        out = io.StringIO()
        with redirect_stdout(out):
            solver._scan(score_cache)
        text = out.getvalue()
        self.assertIn("[ERD ready: 4.500", text)
        self.assertNotIn("no guaranteed finish", text)
        self.assertNotIn("best-effort", text)

    def test_infeasible_budget_reports_loss_and_best_effort_fallback(self):
        """budget=7 is one short of LINEAR's worst-case line 8: no strategy
        wins every game, so the solver must report the loss instead of an
        unconstrained recommendation, then fall back to a clearly labeled
        best-effort (unconstrained) value so the player isn't left with
        nothing."""
        solver, score_cache = self._solver(budget=7)
        out = io.StringIO()
        with redirect_stdout(out):
            solver._scan(score_cache)
        text = out.getvalue()
        self.assertIn("[ERD: no guaranteed finish within 7 guesses]", text)
        self.assertIn("[ERD best-effort (unconstrained): 4.500", text)
        self.assertNotIn("[ERD ready:", text)

        # The exhaustive disproof at budget=7 is persisted, not just announced.
        root_key = ScoreCache.encode_subset(self.LINEAR)
        self.assertEqual(score_cache.read_loss(root_key, ERD_ALL), 7)

    def test_unconstrained_solver_never_reports_loss(self):
        """budget=None (legacy/unconstrained callers) must behave exactly as
        before: no loss branch is reachable since min_expected_guesses never
        returns None except on cancel."""
        solver, score_cache = self._solver(budget=None)
        out = io.StringIO()
        with redirect_stdout(out):
            solver._scan(score_cache)
        text = out.getvalue()
        self.assertIn("[ERD ready: 4.500", text)
        self.assertNotIn("no guaranteed finish", text)

    def test_pause_resume_race_does_not_misreport_budget_loss(self):
        """min_expected_guesses returns None for two different reasons: an
        abort (cancel/pause fired cancel_check mid-search) and a genuine
        budget floor. The main thread's pause()/resume() wrap tightly
        around each command handler (main()), so self._paused can already
        be set again by the time _scan inspects it — even right after a
        real pause aborted the search. _scan must not infer "proven loss"
        from that racy flag; it must consult the persisted loss record,
        which a merely-aborted call never wrote."""
        solver, score_cache = self._solver(budget=8)  # a feasible budget

        real_min_expected = min_expected_guesses
        calls = {'n': 0}

        def flaky_first_call(*args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 1:
                # No loss is written for this call — it is a bare abort,
                # not an exhausted, proven-infeasible search. self._paused
                # is left set (as if resume() already fired), reproducing
                # the race even without real threads.
                return None
            return real_min_expected(*args, **kwargs)

        out = io.StringIO()
        with mock.patch('wordle.min_expected_guesses', side_effect=flaky_first_call), \
             redirect_stdout(out):
            solver._scan(score_cache)

        text = out.getvalue()
        self.assertNotIn("no guaranteed finish", text,
            "an aborted (not exhausted) search must not be reported as a "
            "proven budget floor")
        self.assertIn("[ERD ready: 4.500", text,
            "the retried search must still deliver the real recommendation")
        self.assertEqual(calls['n'], 2)


# ---------------------------------------------------------------------------
# BranchPrecacheSolver: precache ERD for sibling branches of a guess
# ---------------------------------------------------------------------------

class TestBranchPrecacheSolver(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        os.unlink(self.db)

    @staticmethod
    def _branches():
        # Three tiny, disjoint synthetic branches (3 words each), paired
        # with arbitrary valid response codes (used to format the branch
        # pattern in the periodic report).
        return [
            (0, ANSWERS[0:3]),
            (1, ANSWERS[3:6]),
            (2, ANSWERS[6:9]),
        ]

    def test_run_populates_erd_for_every_branch(self):
        branches = self._branches()
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))

        with redirect_stdout(io.StringIO()):
            solver.run()

        self.assertEqual(solver.branches_done, solver.branches_total)
        self.assertEqual(solver.branches_done, 3)
        self.assertEqual(solver.branches_skipped, 0)

        sc = ScoreCache(self.db, ANSWERS)
        try:
            for _, words in branches:
                hit = sc.read(ScoreCache.encode_subset(words), ERD_ALL)
                self.assertIsNotNone(hit)
        finally:
            sc.close()

    def test_tracks_root_total_culled(self):
        """root_total is set to len(ranked) (the candidate vocabulary)
        before min_expected_guesses starts — so the status line has
        something to show even before the first top-level candidate
        fully resolves — and culled (driven by the progress callback) ends
        up populated for the last branch processed."""
        branches = self._branches()
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))

        with redirect_stdout(io.StringIO()):
            solver.run()

        self.assertEqual(solver.root_total, len(GUESSES))
        self.assertGreaterEqual(solver.culled, 0)

    def test_periodic_print_reports_culled_and_current_candidate(self):
        """The periodic print (driven by either the per-candidate progress
        callback or the per-subgroup heartbeat) must report how many
        top-level candidates were culled by ERD-lower-bound pruning and the
        in-progress candidate's elapsed time and cache hit/miss counts —
        the signal that's missing while a single top-level candidate's deep
        recursion is still running."""
        branches = self._branches()[:1]
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))

        # Force every _maybe_print() check ("now - last_print >= 30") to
        # see 30+ seconds elapsed on every call.
        counter = itertools.count()
        out = io.StringIO()
        with mock.patch('wordle.time.time', side_effect=lambda: next(counter) * 31), \
             redirect_stdout(out):
            solver.run()

        text = out.getvalue()
        self.assertIn("culled", text)
        self.assertRegex(text, r'\d+s, \d+ hit/\d+ miss')

    def test_skips_already_cached_branch_and_fills_others(self):
        """Simulates the live ERDSolver having already filled one branch's
        ERD entry (e.g. in an earlier game): precache must skip it
        (resumability/meshing) while still filling the other branch."""
        branches = self._branches()[:2]
        live_solved_words = branches[0][1]
        other_words = branches[1][1]

        sc = ScoreCache(self.db, ANSWERS)
        cache = ResponseCache(ANSWERS, sc)
        min_expected_guesses(live_solved_words, cache, sc,
                              guesses=GUESSES, policy=ERD_ALL)
        sc.close()

        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))

        with redirect_stdout(io.StringIO()):
            solver.run()

        self.assertEqual(solver.branches_skipped, 1)
        self.assertEqual(solver.branches_done, 2)

        sc = ScoreCache(self.db, ANSWERS)
        try:
            self.assertIsNotNone(
                sc.read(ScoreCache.encode_subset(other_words), ERD_ALL))
        finally:
            sc.close()

    def test_precached_keys_seed_counts_immediately_without_double_counting(self):
        """cmd_precache's upfront scan passes already-cached branch keys in;
        the status line (branches_done/branches_skipped) must reflect them
        before run() ever starts, and run() must not re-count them."""
        branches = self._branches()[:2]
        live_solved_words = branches[0][1]
        other_words = branches[1][1]

        sc = ScoreCache(self.db, ANSWERS)
        cache = ResponseCache(ANSWERS, sc)
        min_expected_guesses(live_solved_words, cache, sc,
                              guesses=GUESSES, policy=ERD_ALL)
        sc.close()

        precached_keys = {ScoreCache.encode_subset(live_solved_words)}
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS), precached_keys=precached_keys)

        # Accurate immediately, before run() processes anything.
        self.assertEqual(solver.branches_done, 1)
        self.assertEqual(solver.branches_skipped, 1)

        with redirect_stdout(io.StringIO()):
            solver.run()

        self.assertEqual(solver.branches_done, 2)
        self.assertEqual(solver.branches_skipped, 1)  # not double-counted

        sc = ScoreCache(self.db, ANSWERS)
        try:
            self.assertIsNotNone(
                sc.read(ScoreCache.encode_subset(other_words), ERD_ALL))
        finally:
            sc.close()

    def test_stopped_before_run_prints_nothing(self):
        """If stop() fires before run() is even scheduled (e.g. the user
        made a guess immediately after 'p'), run() must not print a stale
        '[Precache: starting ...]' announcement for work it never starts —
        same principle as ERDSolver's no-stale-ready-message guarantee."""
        branches = self._branches()
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))
        solver.stop()

        out = io.StringIO()
        with redirect_stdout(out):
            solver.run()

        self.assertEqual(out.getvalue(), '')
        self.assertEqual(solver.branches_done, 0)

    def test_stop_mid_run_returns_promptly(self):
        branches = self._branches()
        solver = BranchPrecacheSolver(
            "heart", branches, ANSWERS, GUESSES, self.db,
            anchor_word_count=len(ANSWERS))

        from wordle_engine import min_expected_guesses as real_min_expected
        call_count = [0]

        def fake_min_expected(*args, **kwargs):
            call_count[0] += 1
            solver.stop()  # e.g. the user made a guess mid-branch
            return real_min_expected(*args, **kwargs)

        with mock.patch('wordle.min_expected_guesses', side_effect=fake_min_expected), \
             redirect_stdout(io.StringIO()):
            solver.run()

        self.assertEqual(call_count[0], 1)
        self.assertEqual(solver.branches_done, 0)


# ---------------------------------------------------------------------------
# _platform_label: distinguish iPhone/iPad/macOS/Linux/Windows for the banner
# ---------------------------------------------------------------------------

class TestPlatformLabel(unittest.TestCase):

    @staticmethod
    def _uname(system, release, machine):
        return platform.uname_result(
            system=system, node='', release=release, version='', machine=machine)

    def test_linux(self):
        with mock.patch('wordle.platform.uname',
                         return_value=self._uname('Linux', '6.18.5', 'x86_64')):
            self.assertEqual(_platform_label(), 'Linux 6.18.5')

    def test_windows(self):
        with mock.patch('wordle.platform.uname',
                         return_value=self._uname('Windows', '10', 'AMD64')):
            self.assertEqual(_platform_label(), 'Windows 10')

    def test_macos(self):
        with mock.patch('wordle.platform.uname',
                         return_value=self._uname('Darwin', '25.5.0', 'arm64')):
            self.assertEqual(_platform_label(), 'macOS 25.5.0')

    def test_iphone(self):
        with mock.patch('wordle.platform.uname',
                         return_value=self._uname('Darwin', '25.5.0', 'iPhone14,2')):
            self.assertEqual(_platform_label(), 'iPhone 25.5.0')

    def test_ipad(self):
        with mock.patch('wordle.platform.uname',
                         return_value=self._uname('Darwin', '25.5.0', 'iPad13,4')):
            self.assertEqual(_platform_label(), 'iPad 25.5.0')


# ---------------------------------------------------------------------------
# Shared scan-progress formatting helpers
# ---------------------------------------------------------------------------

class TestFormatCacheTimestamp(unittest.TestCase):

    def test_none_is_na(self):
        self.assertEqual(_format_cache_timestamp(None), "n/a")

    def test_zero_is_na(self):
        self.assertEqual(_format_cache_timestamp(0), "n/a")

    def test_formats_epoch_seconds(self):
        ts = datetime(2026, 6, 11, 14, 2, 18).timestamp()
        self.assertEqual(_format_cache_timestamp(ts), "2026-06-11 14:02:18")


class TestCurrentCandidateTag(unittest.TestCase):

    def test_none_word_returns_none(self):
        self.assertIsNone(_current_candidate_tag(MemoryScoreCache(), None, time.time()))

    def test_none_start_time_returns_none(self):
        self.assertIsNone(_current_candidate_tag(MemoryScoreCache(), "arise", None))

    def test_none_score_cache_returns_none(self):
        self.assertIsNone(_current_candidate_tag(None, "arise", time.time()))

    def test_returns_word_elapsed_hits_misses(self):
        sc = MemoryScoreCache()
        sc.read_hits = 12
        sc.read_misses = 3
        word, elapsed, hits, misses = _current_candidate_tag(sc, "arise", time.time() - 5)
        self.assertEqual(word, "arise")
        self.assertGreaterEqual(elapsed, 5)
        self.assertEqual(hits, 12)
        self.assertEqual(misses, 3)


class TestFormatScanProgress(unittest.TestCase):

    def test_basic_no_best_no_current(self):
        lines = _format_scan_progress(234, 12972, None, 4772, None)
        self.assertEqual(lines, ["234/12,972, 4,772 culled"])

    def test_with_best(self):
        lines = _format_scan_progress(234, 12972, ("arise", 3.142), 4772, None)
        self.assertEqual(lines, ["234/12,972, 4,772 culled, best: ARISE 3.142"])

    def test_with_current_candidate(self):
        lines = _format_scan_progress(234, 12972, ("arise", 3.142), 4772,
                                        ("grind", 23.4, 12, 3))
        self.assertEqual(lines, [
            "234/12,972, 4,772 culled, best: ARISE 3.142",
            "  GRIND: 23s, 12 hit/3 miss",
        ])

    def test_indent_and_suffix(self):
        lines = _format_scan_progress(234, 12972, None, 4772, None,
                                        indent=2, suffix=' cands')
        self.assertEqual(lines, ["  234/12,972 cands, 4,772 culled"])

    def test_indent_applies_to_current_candidate_line_too(self):
        lines = _format_scan_progress(234, 12972, None, 4772,
                                        ("grind", 23.4, 12, 3), indent=2)
        self.assertEqual(lines, [
            "  234/12,972, 4,772 culled",
            "    GRIND: 23s, 12 hit/3 miss",
        ])


class TestFormatBranchHeader(unittest.TestCase):

    def test_computing(self):
        self.assertEqual(_format_branch_header(315),
                         "315 words | ERD: computing...")


class TestPrintLineWithPattern(unittest.TestCase):

    def test_prefix_pattern_suffix(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        out = io.StringIO()
        with redirect_stdout(out):
            print_line_with_pattern(
                '  Branch ', response, ' | 315 words | ERD: computing...')
        text = out.getvalue()
        # Pattern characters appear, in order, between prefix and suffix —
        # any ANSI color codes from SUPPORTS_COLOR are stripped out.
        stripped = re.sub(r'\033\[\d*m', '', text)
        self.assertEqual(stripped,
                         '  Branch -ygy- | 315 words | ERD: computing...\n')


# ---------------------------------------------------------------------------
# Color output: characterization tests across plain / ANSI / Pythonista.
#
# These pin down CURRENT behavior so a future markup/renderer refactor can't
# silently change what gets printed. SUPPORTS_COLOR, IS_PYTHONISTA, and
# console are module-level constants computed once at import time, but every
# color function below re-reads them from the module namespace on each call,
# so mock.patch.multiple('wordle', ...) is enough to exercise all 3 modes.
# ---------------------------------------------------------------------------

class FakeConsole:
    """Records set_color() calls, mimicking Pythonista's `console` module."""

    def __init__(self):
        self.calls = []

    def set_color(self, *args):
        self.calls.append(args)


def _capture_stdout(fn):
    out = io.StringIO()
    with redirect_stdout(out):
        fn()
    return out.getvalue()


class TestMark(unittest.TestCase):
    """mark() is pure string-building — sentinel insertion is independent
    of platform/rendering mode."""

    def test_wraps_text_in_color_sentinel_and_reset(self):
        self.assertEqual(mark('red', 'X'), f'{MARK_RED}X{MARK_RESET}')

    def test_each_color_has_its_own_sentinel(self):
        self.assertEqual(mark('green', 'g'), f'{MARK_GREEN}g{MARK_RESET}')
        self.assertEqual(mark('yellow', 'y'), f'{MARK_YELLOW}y{MARK_RESET}')
        self.assertEqual(mark('gray', '-'), f'{MARK_GRAY}-{MARK_RESET}')

    def test_sentinels_are_distinct(self):
        sentinels = {MARK_RESET, MARK_RED, MARK_GREEN, MARK_YELLOW, MARK_GRAY}
        self.assertEqual(len(sentinels), 5)


class TestColorOutputPlain(unittest.TestCase):
    """SUPPORTS_COLOR=False, IS_PYTHONISTA=False — the worst case to guard:
    plain output must be exactly the plain text, with no escape codes or
    other markup mixed in."""

    def setUp(self):
        patcher = mock.patch.multiple('wordle', SUPPORTS_COLOR=False,
                                       IS_PYTHONISTA=False, console=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_print_error(self):
        self.assertEqual(_capture_stdout(lambda: print_error("bad")), "bad\n")

    def test_print_success(self):
        self.assertEqual(_capture_stdout(lambda: print_success("ok")), "ok\n")

    def test_colored_text_is_no_op(self):
        def fn():
            with colored_text("red"):
                print("X", end='')
        self.assertEqual(_capture_stdout(fn), "X")

    def test_reset_color_prints_nothing(self):
        self.assertEqual(_capture_stdout(reset_color), "")

    def test_print_colored_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        self.assertEqual(
            _capture_stdout(lambda: print_colored_pattern(response)), "-ygy-")

    def test_print_colored_word(self):
        response = ['green', 'yellow', 'gray']
        self.assertEqual(
            _capture_stdout(lambda: print_colored_word("cat", response)), "CAT")

    def test_print_line_with_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        out = _capture_stdout(lambda: print_line_with_pattern(
            '  Branch ', response, ' | 315 words'))
        self.assertEqual(out, '  Branch -ygy- | 315 words\n')

    def test_render_markup_strips_all_sentinels(self):
        text = 'A' + mark('red', 'B') + 'C' + mark('green', 'D') + MARK_RESET
        self.assertEqual(
            _capture_stdout(lambda: render_markup(text, end='')), 'ABCD')

    def test_render_markup_default_end_is_newline(self):
        self.assertEqual(_capture_stdout(lambda: render_markup('hi')), 'hi\n')


class TestColorOutputANSI(unittest.TestCase):
    """SUPPORTS_COLOR=True, IS_PYTHONISTA=False — Linux terminal output."""

    def setUp(self):
        patcher = mock.patch.multiple('wordle', SUPPORTS_COLOR=True,
                                       IS_PYTHONISTA=False, console=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_print_error(self):
        self.assertEqual(_capture_stdout(lambda: print_error("bad")),
                         f"{ANSI_COLORS['red']}bad\n{ANSI_RESET}")

    def test_print_success(self):
        self.assertEqual(_capture_stdout(lambda: print_success("ok")),
                         f"{ANSI_COLORS['green']}ok\n{ANSI_RESET}")

    def test_colored_text_wraps_with_escapes(self):
        def fn():
            with colored_text("red"):
                print("X", end='')
        self.assertEqual(_capture_stdout(fn),
                         f"{ANSI_COLORS['red']}X{ANSI_RESET}")

    def test_reset_color_prints_reset_code(self):
        self.assertEqual(_capture_stdout(reset_color), ANSI_RESET)

    def test_print_colored_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        out = _capture_stdout(lambda: print_colored_pattern(response))
        expected = (
            ANSI_COLORS['gray'] + "-" + ANSI_RESET
            + ANSI_COLORS['yellow'] + "y" + ANSI_RESET
            + ANSI_COLORS['green'] + "g" + ANSI_RESET
            + ANSI_COLORS['yellow'] + "y" + ANSI_RESET
            + ANSI_COLORS['gray'] + "-" + ANSI_RESET
        )
        self.assertEqual(out, expected)

    def test_print_colored_word(self):
        response = ['green', 'yellow', 'gray']
        out = _capture_stdout(lambda: print_colored_word("cat", response))
        expected = (
            ANSI_COLORS['green'] + "C" + ANSI_RESET
            + ANSI_COLORS['yellow'] + "A" + ANSI_RESET
            + ANSI_COLORS['gray'] + "T" + ANSI_RESET
        )
        self.assertEqual(out, expected)

    def test_print_line_with_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        out = _capture_stdout(lambda: print_line_with_pattern(
            '  Branch ', response, ' | 315 words'))
        expected = (
            "  Branch "
            + ANSI_COLORS['gray'] + "-" + ANSI_RESET
            + ANSI_COLORS['yellow'] + "y" + ANSI_RESET
            + ANSI_COLORS['green'] + "g" + ANSI_RESET
            + ANSI_COLORS['yellow'] + "y" + ANSI_RESET
            + ANSI_COLORS['gray'] + "-" + ANSI_RESET
            + " | 315 words\n"
        )
        self.assertEqual(out, expected)

    def test_render_markup_translates_sentinels_to_ansi(self):
        text = 'A' + mark('red', 'B') + 'C'
        out = _capture_stdout(lambda: render_markup(text, end=''))
        expected = 'A' + ANSI_COLORS['red'] + 'B' + ANSI_RESET + 'C'
        self.assertEqual(out, expected)


class TestColorOutputPythonista(unittest.TestCase):
    """IS_PYTHONISTA=True — color comes from console.set_color() calls;
    stdout itself stays plain text (no escape codes on this path)."""

    def setUp(self):
        self.console = FakeConsole()
        patcher = mock.patch.multiple('wordle', SUPPORTS_COLOR=True,
                                       IS_PYTHONISTA=True, console=self.console)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_print_error(self):
        text = _capture_stdout(lambda: print_error("bad"))
        self.assertEqual(text, "bad\n")
        self.assertEqual(self.console.calls, [(1, 0, 0), ()])

    def test_print_success(self):
        text = _capture_stdout(lambda: print_success("ok"))
        self.assertEqual(text, "ok\n")
        self.assertEqual(self.console.calls, [(0, 0.6, 0), ()])

    def test_reset_color_calls_set_color_with_no_args(self):
        text = _capture_stdout(reset_color)
        self.assertEqual(text, "")
        self.assertEqual(self.console.calls, [()])

    def test_print_colored_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        text = _capture_stdout(lambda: print_colored_pattern(response))
        self.assertEqual(text, "-ygy-")
        self.assertEqual(self.console.calls, [
            (0.5, 0.5, 0.5), (),
            (0.6, 0.6, 0), (),
            (0, 0.6, 0), (),
            (0.6, 0.6, 0), (),
            (0.5, 0.5, 0.5), (),
        ])

    def test_print_colored_word(self):
        response = ['green', 'yellow', 'gray']
        text = _capture_stdout(lambda: print_colored_word("cat", response))
        self.assertEqual(text, "CAT")
        self.assertEqual(self.console.calls, [
            (0, 0.6, 0), (),
            (0.6, 0.6, 0), (),
            (0.5, 0.5, 0.5), (),
        ])

    def test_print_line_with_pattern(self):
        response = ['gray', 'yellow', 'green', 'yellow', 'gray']
        text = _capture_stdout(lambda: print_line_with_pattern(
            '  Branch ', response, ' | 315 words'))
        self.assertEqual(text, "  Branch -ygy- | 315 words\n")
        self.assertEqual(self.console.calls, [
            (0.5, 0.5, 0.5), (),
            (0.6, 0.6, 0), (),
            (0, 0.6, 0), (),
            (0.6, 0.6, 0), (),
            (0.5, 0.5, 0.5), (),
        ])

    def test_render_markup_calls_set_color(self):
        text = 'A' + mark('red', 'B') + 'C'
        out = _capture_stdout(lambda: render_markup(text, end=''))
        self.assertEqual(out, 'ABC')
        self.assertEqual(self.console.calls, [(1, 0, 0), ()])


# ---------------------------------------------------------------------------
# BranchPrecacheSolver.branches_line / _branches_starting_line
# ---------------------------------------------------------------------------

class TestBranchesLine(unittest.TestCase):

    @staticmethod
    def _solver(branches_done, branches_total, branches_skipped):
        s = BranchPrecacheSolver.__new__(BranchPrecacheSolver)
        s.branches_done = branches_done
        s.branches_total = branches_total
        s.branches_skipped = branches_skipped
        return s

    def test_starting_line(self):
        s = self._solver(0, 124, 42)
        self.assertEqual(s._branches_starting_line(), "Branches: 0/124, 42 cached")

    def test_active_line_shows_hit_and_miss(self):
        s = self._solver(5, 124, 4)
        self.assertEqual(s.branches_line(), "Branches: 5/124, 4 hit/1 miss")

    def test_done_line(self):
        s = self._solver(124, 124, 42)
        self.assertEqual(s.branches_line(done=True),
                         "Branches: 124/124 done, 42 hit/82 miss")


# ---------------------------------------------------------------------------
# current_word_tag: 4-tuple (word, elapsed, hits, misses) for both solvers
# ---------------------------------------------------------------------------

class TestCurrentWordTag(unittest.TestCase):

    @staticmethod
    def _word(i):
        return f"w{i:04d}"

    def test_erd_solver_current_word_tag(self):
        words = [self._word(i) for i in range(10)]
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False)
        sc = MemoryScoreCache()
        solver._score_cache = sc
        solver._start_word(sc, words, 0)
        sc.read_hits = 5
        sc.read_misses = 2

        word, elapsed, hits, misses = solver.current_word_tag()
        self.assertEqual(word, words[0])
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(hits, 5)
        self.assertEqual(misses, 2)

    def test_erd_solver_no_current_word_returns_none(self):
        words = [self._word(i) for i in range(10)]
        solver = ERDSolver(words, words, words, None,
                           policy=ERD_ALL, persist=False)
        self.assertIsNone(solver.current_word_tag())

    def test_branch_precache_solver_current_word_tag(self):
        words = [self._word(i) for i in range(10)]
        solver = BranchPrecacheSolver(
            "heart", [(0, words)], words, words, None,
            anchor_word_count=len(words))
        sc = MemoryScoreCache()
        solver._score_cache = sc
        solver._start_word(sc, words, 1)
        sc.read_hits = 7
        sc.read_misses = 1

        word, elapsed, hits, misses = solver.current_word_tag()
        self.assertEqual(word, words[1])
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(hits, 7)
        self.assertEqual(misses, 1)


# ---------------------------------------------------------------------------
# print_status (single-board): idle line, ERD hit, ordering/scanning, precache
# ---------------------------------------------------------------------------

class TestPrintStatusSingleBoard(unittest.TestCase):

    @staticmethod
    def _gs(soln, precache_solver=None):
        return types.SimpleNamespace(
            single=True,
            solutions=[soln],
            universe=GuessUniverse.ALL_WORDS,
            compliance=ComplianceFilter.UNFILTERED,
            constrained_erd_cache=None,
            precache_solver=precache_solver,
        )

    def test_idle_root_single_line(self):
        soln = make_solution()
        gs = self._gs(soln)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(gs)
        text = out.getvalue()
        self.assertIn(f"{len(ANSWERS)} words left | 0 guesses so far", text)
        self.assertNotIn("Branches:", text)

    def test_root_with_precache_shows_branches_line(self):
        soln = make_solution()
        precache = types.SimpleNamespace(
            is_alive=lambda: True,
            branches_line=lambda: "Branches: 5/124, 4 hit/1 miss",
        )
        gs = self._gs(soln, precache_solver=precache)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(gs)
        text = out.getvalue()
        self.assertIn(f"{len(ANSWERS)} words left | 0 guesses so far", text)
        self.assertIn("Branches: 5/124, 4 hit/1 miss", text)

    def test_solved_shows_guess_count(self):
        soln = make_solution()
        soln.guesses = [["salet", ["yellow", "gray", "gray", "yellow", "yellow"]],
                        ["tenor", ["green", "green", "green", "green", "green"]]]
        soln.current_words = ["tenor"]

        gs = self._gs(soln)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(gs)
        text = out.getvalue()
        self.assertIn("Solved: tenor | 2 guesses", text)

    def test_solved_singular_guess(self):
        soln = make_solution()
        soln.guesses = [["tenor", ["green", "green", "green", "green", "green"]]]
        soln.current_words = ["tenor"]

        gs = self._gs(soln)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(gs)
        text = out.getvalue()
        self.assertIn("Solved: tenor | 1 guess", text)
        self.assertNotIn("1 guesses", text)

    def test_solved_by_deduction_counts_the_unplayed_final_guess(self):
        # Narrowed to one candidate by a non-winning guess: the player
        # still has to play that candidate as a real Wordle guess, so the
        # reported count must include it even though it isn't in
        # soln.guesses yet.
        soln = make_solution()
        soln.guesses = [["salet", ["yellow", "gray", "gray", "yellow", "yellow"]],
                        ["trite", ["green", "green", "gray", "gray", "yellow"]],
                        ["metro", ["gray", "yellow", "green", "green", "gray"]]]
        soln.current_words = ["entry"]

        gs = self._gs(soln)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(gs)
        text = out.getvalue()
        self.assertIn("Solved: entry | 4 guesses", text)

    def test_non_root_with_erd_hit(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            cache = ResponseCache(ANSWERS, sc)
            soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
            soln.guesses = [["salet", ["gray"] * 5]]
            soln.current_words = ANSWERS[:3]
            sc.write(ScoreCache.encode_subset(soln.current_words), ERD_ALL,
                     "arise", 3.142)

            gs = self._gs(soln)
            out = io.StringIO()
            with redirect_stdout(out):
                print_status(gs)
            text = out.getvalue()
            self.assertIn("3 words left | 1 guess so far | 3.142 ARISE", text)
        finally:
            sc.close()
            os.unlink(tmp.name)

    def test_non_root_solver_ordering(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            cache = ResponseCache(ANSWERS, sc)
            soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
            soln.guesses = [["salet", ["gray"] * 5]]
            soln.current_words = ANSWERS[:3]

            solver = ERDSolver.__new__(ERDSolver)
            solver._words = soln.current_words
            solver.root_total = 0

            gs = self._gs(soln)
            out = io.StringIO()
            with redirect_stdout(out):
                print_status(gs, solver)
            text = out.getvalue()
            self.assertIn("3 words left | 1 guess so far", text)
            self.assertIn("ERD:", text)
            self.assertIn("ordering candidates...", text)
        finally:
            sc.close()
            os.unlink(tmp.name)

    def test_non_root_solver_scanning(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        try:
            sc = ScoreCache(tmp.name, ANSWERS)
            cache = ResponseCache(ANSWERS, sc)
            soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
            soln.guesses = [["salet", ["gray"] * 5]]
            soln.current_words = ANSWERS[:3]

            solver = ERDSolver.__new__(ERDSolver)
            solver._words = soln.current_words
            solver.root_total = 12972
            solver.root_done = 234
            solver.root_best = ("arise", 3.142)
            solver.culled = 4772
            solver._score_cache = None
            solver.current_word = None
            solver.current_word_start = None

            gs = self._gs(soln)
            out = io.StringIO()
            with redirect_stdout(out):
                print_status(gs, solver)
            text = out.getvalue()
            self.assertIn("3 words left | 1 guess so far", text)
            self.assertIn("234/12,972 cands, 4,772 culled, best: ARISE 3.142", text)
        finally:
            sc.close()
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# _compare_words display: answer-set marker and column spacing
# ---------------------------------------------------------------------------

class TestCompareWordsDisplay(unittest.TestCase):

    @staticmethod
    def _fake_stats(word, *args, **kwargs):
        return dict(step1=4.0, step2=2.0, step3=1.0,
                    wt_avg=2.5, max_group_size=10, prob_finish=0.5,
                    buckets=[1, 2, 3, 0, 0], erd=None)

    def _render(self, words):
        soln = make_solution()
        set_display_context(soln)
        buf = io.StringIO()
        with mock.patch('wordle._multistep_stats', side_effect=self._fake_stats), \
             redirect_stdout(buf):
            _compare_words(words, soln)
        return buf.getvalue()

    def test_answer_set_words_marked_with_asterisk(self):
        soln = make_solution()
        self.assertIn("crane", soln.answer_set)
        self.assertNotIn("brain", soln.answer_set)

        out = self._render(["crane", "brain"])
        header = next(l for l in out.splitlines()
                      if l.strip().startswith('CRANE') and 'BRAIN' in l)
        self.assertIn('CRANE*', header)
        self.assertIn('BRAIN ', header)
        self.assertNotIn('BRAIN*', header)


if __name__ == "__main__":
    unittest.main(verbosity=2)
