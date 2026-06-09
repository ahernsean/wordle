"""
Tests for Wordle engine correctness and cache behavior.
Run with:  python test_wordle.py
"""
import io
import math
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wordle_engine import (
    Solution, ScoringMethod, ResponseCache,
    calculate_response, score_groups, score_groups_multi, calculate_group_counts,
    score_word, score_word_multi, load_word_list, max_entropy,
    Restriction, answer_to_restriction, apply_guess, _encode_response,
    min_expected_guesses, ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED,
    ERD_ANSWERS_UNFILTERED, cache_all_scores,
)
from cache_sqlite import ScoreCache, MemoryScoreCache
from wordle import (
    _multistep_stats, _erd_solve_scores, ERDWarmer,
    _compare_words, set_display_context,
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


class TestScoringMethodDisplay(unittest.TestCase):

    def test_label_is_distinct_per_method(self):
        labels = {m.label for m in ScoringMethod}
        self.assertEqual(len(labels), len(list(ScoringMethod)))

    def test_format_score_max_group_size_is_integer(self):
        self.assertEqual(ScoringMethod.MAX_GROUP_SIZE.format_score(7.0), "7")

    def test_format_score_prob_finish_is_percentage(self):
        self.assertEqual(ScoringMethod.PROB_FINISH.format_score(0.25), "25.0%")

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
        subset_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write_scores(subset_key, [("crane", 3.14159), ("slate", 2.71828)],
                        "entropy_gain")
        result = dict(sc.read_scores(subset_key, "entropy_gain"))
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
        subset_key = ScoreCache.encode_subset(["crane", "slate", "trace"])
        sc.write(subset_key, "full", "heart", 2.5)
        word, ent = sc.read(subset_key, "full")
        self.assertEqual(word, "heart")
        self.assertAlmostEqual(ent, 2.5)

    def test_read_miss_returns_none(self):
        sc = ScoreCache(self.db, ANSWERS)
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        self.assertIsNone(sc.read(subset_key, "full"))
        self.assertIsNone(sc.read_scores(subset_key, "entropy_gain"))

    def test_policy_separation(self):
        sc = ScoreCache(self.db, ANSWERS)
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write(subset_key, "full", "heart", 2.5)
        sc.write(subset_key, "hard", "earth", 1.8)
        self.assertEqual(sc.read(subset_key, "full")[0], "heart")
        self.assertEqual(sc.read(subset_key, "hard")[0], "earth")

    def test_different_universe_no_cross_contamination(self):
        alt_answers = ["brain", "stove", "cloud"]
        sc1 = ScoreCache(self.db, ANSWERS)
        sc2 = ScoreCache(self.db, alt_answers)
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        sc1.write_scores(subset_key, [("crane", 3.14)], "entropy_gain")
        self.assertIsNone(sc2.read_scores(subset_key, "entropy_gain"))

    def test_overwrite_replaces_value(self):
        sc = ScoreCache(self.db, ANSWERS)
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write_scores(subset_key, [("crane", 1.0)], "entropy_gain")
        sc.write_scores(subset_key, [("crane", 9.9)], "entropy_gain")
        result = dict(sc.read_scores(subset_key, "entropy_gain"))
        self.assertAlmostEqual(result["crane"], 9.9)

    def test_encode_subset_is_compact(self):
        # Key length = 5 * number of words, no separators
        words = ["crane", "slate", "trace"]
        key = ScoreCache.encode_subset(words)
        self.assertEqual(len(key), 15)
        self.assertNotIn(b"\x00", key)

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

        # Opening ScoreCache should rename the table/columns to
        # subgroup_best_by_policy/best_word/best_score AND delete the
        # old-format row
        ScoreCache(self.db, ANSWERS)
        conn2 = _sqlite3.connect(self.db)
        rows = conn2.execute(
            "SELECT COUNT(*) FROM subgroup_best_by_policy "
            "WHERE instr(subset_key, char(0)) > 0"
        ).fetchone()[0]
        conn2.close()
        self.assertEqual(rows, 0)

    def test_old_lookahead_result_table_is_renamed(self):
        # Simulate a row persisted under the original pre-rename table/column
        # names, WITHOUT the legacy key encoding — i.e. data written between
        # the subset-key-encoding migration and the table/column rename.
        # Built entirely without a ScoreCache so neither subgroup_pick nor
        # subgroup_best_by_policy exists yet — otherwise the rename-in-place
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
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        conn.execute(
            "INSERT OR REPLACE INTO lookahead_result VALUES (?,?,?,?,?,?)",
            (subset_key, "hard", universe_id, "heart", 2.5, 0),
        )
        conn.commit()
        conn.close()

        sc = ScoreCache(self.db, ANSWERS)
        hit = sc.read(subset_key, "hard")
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
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        conn.execute(
            "INSERT OR REPLACE INTO subgroup_pick VALUES (?,?,?,?,?,?)",
            (subset_key, "hard", universe_id, "heart", 2.5, 0),
        )
        conn.commit()
        conn.close()

        sc = ScoreCache(self.db, ANSWERS)
        hit = sc.read(subset_key, "hard")
        self.assertIsNotNone(hit)
        self.assertEqual(hit, ("heart", 2.5))

    def test_old_minimax_method_key_is_migrated(self):
        # Simulate rows persisted under the pre-rename method key.
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(self.db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS word_scores (
                subset_hash TEXT NOT NULL, word TEXT NOT NULL,
                method TEXT NOT NULL, score REAL NOT NULL,
                universe_id TEXT NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, universe_id, word)
            )
        """)
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        subset_hash = ScoreCache._subset_hash(subset_key)
        universe_id = ScoreCache(self.db, ANSWERS).universe_id
        conn.execute(
            "INSERT OR REPLACE INTO word_scores VALUES (?,?,?,?,?,?)",
            (subset_hash, "crane", "minimax", 4.0, universe_id, 0),
        )
        conn.commit()
        conn.close()

        # Re-opening ScoreCache should migrate the old method key forward —
        # the data must remain reachable under its new name.
        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read_scores(subset_key, "minimax"))
        migrated = sc.read_scores(subset_key, "max_group_size")
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
        subset_key = ScoreCache.encode_subset(["crane", "slate"])
        sc.write_scores(subset_key, [("crane", 1.0)], "entropy_gain")

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
                sc.write_scores(subset_key, [("slate", 2.0)], "entropy_gain")
        finally:
            sc._conn = real_conn

        # The pre-existing row must survive untouched, and no partial
        # row from the failed write should have leaked in.
        result = dict(sc.read_scores(subset_key, "entropy_gain"))
        self.assertEqual(result, {"crane": 1.0})


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
        subset_key = ScoreCache.encode_subset(s.current_words)
        cached = sc.read_scores(subset_key, "entropy_gain")
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
        expected = dict(cache1._cache["crane"])

        sc2 = ScoreCache(self.db, ANSWERS)
        cache2 = ResponseCache(ANSWERS, score_cache=sc2)
        with mock.patch('wordle_engine.calculate_response') as fake_calc:
            cache2._ensure("crane")
            fake_calc.assert_not_called()
        self.assertEqual(dict(cache2._cache["crane"]), expected)

    def test_decomposition_reload_matches_freshly_computed(self):
        sc1 = ScoreCache(self.db, ANSWERS)
        cache1 = ResponseCache(ANSWERS, score_cache=sc1)
        cache1._ensure("crane")
        expected = dict(cache1._cache["crane"])

        sc2 = ScoreCache(self.db, ANSWERS)
        cache2 = ResponseCache(ANSWERS, score_cache=sc2)
        cache2._ensure("crane")
        self.assertEqual(dict(cache2._cache["crane"]), expected)


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
        subset_key = ScoreCache.encode_subset(subgroup)

        cache_all_scores("heart", subgroup, sc, subset_key, cache=cache)

        for method in ScoringMethod:
            cached = sc.read_scores(subset_key, method.name.lower())
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
            subset_key = ScoreCache.encode_subset(subgroup)
            hit = sc.read(subset_key, "hard")
            self.assertIsNotNone(hit)
            best_word, _best_score = hit

            # Every method besides the ranking criterion (entropy, already
            # captured in subgroup_best_by_policy) should be persisted too — they
            # all come from the same group-count partition, so there's no
            # principled reason to single any of them out.
            for method in ScoringMethod:
                if method == ScoringMethod.ENTROPY_GAIN:
                    continue
                cached = sc.read_scores(subset_key, method.name.lower())
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
                subset_key = ScoreCache.encode_subset(subgroup)
                hit = s.score_cache.read(subset_key, "full")
                self.assertIsNotNone(hit)
                self.assertIn(hit[0], second_step_words,
                              "full-mode winner must come from second_step_words")
                self.assertIsNone(
                    s.score_cache.read(subset_key, "hard"),
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

            subset_key = ScoreCache.encode_subset(subset)
            for method in ScoringMethod:
                cached = sc.read_scores(subset_key, method.name.lower())
                self.assertIsNotNone(
                    cached, f"{method.name} should be persisted for the ERD winner")
                expected = score_word(best_word, subset, method, cache=self.cache)
                self.assertEqual(dict(cached)[best_word], expected)
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

        with mock.patch('wordle_engine.time.time', side_effect=fake_time):
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
        subset_key = ScoreCache.encode_subset(soln.current_words)
        rows = sc.read_scores(subset_key, "entropy_gain")
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
        subset_key = ScoreCache.encode_subset(soln.current_words)
        cached = sc.read_scores(subset_key, "entropy_gain")
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

    def test_possible_answers_warmer_readable_by_solve(self):
        """
        Results written with policy=ERD_ANSWERS (POSSIBLE_ANSWERS warmer) are
        readable by _erd_solve_scores under policy=ERD_ANSWERS.

        With the bug, the warmer calls min_expected_guesses(guesses=words) which
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

        # Simulate POSSIBLE_ANSWERS warmer: guesses=current_words, policy=ERD_ANSWERS
        min_expected_guesses(words, soln.cache, sc,
                             guesses=words, policy=ERD_ANSWERS)

        scores = _erd_solve_scores(soln, score_cache=sc, policy=ERD_ANSWERS)
        self.assertIsNotNone(scores,
                             "POSSIBLE_ANSWERS ERD must be readable after warmer completes")

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
# That cost belongs solely to the background ERDWarmer, which uses its own
# short deadlines to detect when it has bitten off more than it can chew.
# The foreground must instead simply *surface* whatever the warmer has
# already cached: an instant, recursion-free cache read per subgroup. If a
# subgroup isn't cached yet, ERD is reported as unavailable (None) rather
# than computed live — exactly mirroring how print_status's ERD tag already
# works. As the warmer (running continuously in the background) populates
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
                "ERD computation belongs to the background warmer only"))

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
        long-lived MemoryScoreCache shared with the warmer) without ever
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
        connection — close() must be safe to call (ERDWarmer's run() calls
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
        even for the identical (subset_key, policy) — preventing false hits
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
    words outside current_words (non-answers) that have pre-warmed subgroup ERDs
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
        A non-answer word with all subgroup ERDs pre-warmed appears in the
        ranking when guesses= is passed to _erd_solve_scores.
        """
        answers = ANSWERS[:5]
        non_answer = next(w for w in GUESSES if w not in ANSWERS)

        with tempfile.TemporaryDirectory() as d:
            sc = ScoreCache(os.path.join(d, 'test.sqlite3'), answers)
            soln = self._soln(answers, sc)

            # Pre-warm all subgroup ERDs using the full GUESSES vocabulary
            min_expected_guesses(answers, soln.cache, sc,
                                 guesses=GUESSES, policy=ERD_ALL)

            scores = _erd_solve_scores(soln, score_cache=sc,
                                       policy=ERD_ALL, guesses=GUESSES)
            self.assertIsNotNone(scores)
            result_words = [w for w, _ in scores]
            self.assertIn(non_answer, result_words,
                          f"non-answer '{non_answer}' must appear in ERD ranking "
                          f"when guesses= is supplied")

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


class TestERDWarmerKeepsWorking(unittest.TestCase):

    @staticmethod
    def _word(i):
        return f"w{i:04d}"  # 5 ASCII chars, satisfies encode_subset's slicing

    def test_no_ready_message_when_superseded_mid_root_computation(self):
        """If stop() fires while the (slow, uninterruptible) root computation
        is in flight, the warmer must not announce a result on the way out —
        a superseded warmer racing a fresh one to print the same value is
        exactly what produces duplicate "[ERD ready]" lines at the prompt."""
        words = [self._word(i) for i in range(10)]

        class FakeResponseCache:
            answer_words = words
            _cache = {}

            @staticmethod
            def group_words(word, current_words):
                return {}

            @staticmethod
            def group_counts(word, current_words):
                return {}

        score_cache = MemoryScoreCache()
        score_cache.set_scope('test-scope')

        warmer = ERDWarmer(words, words, words, None, FakeResponseCache(),
                           policy=ERD_ALL, persist=False, seed_mem_cache=score_cache)

        def cancel_during_root(remaining, cache, sc, deadline=None,
                                guesses=None, policy=None,
                                progress_callback=None,
                                cancel_check=None):
            warmer.stop()  # e.g. the user moved on; a fresh warmer supersedes this one
            return 1.8

        printed = []
        warmer._rcache = FakeResponseCache()
        with mock.patch('wordle.min_expected_guesses', side_effect=cancel_during_root), \
             mock.patch('builtins.print', side_effect=lambda *a, **k: printed.append(a)):
            warmer._warm(score_cache)

        self.assertFalse(
            any('ERD ready' in str(a) for a in printed),
            "a superseded warmer must not print a stale [ERD ready] announcement")


# ---------------------------------------------------------------------------
# _compare_words display: answer-set marker and column spacing
# ---------------------------------------------------------------------------

class TestCompareWordsDisplay(unittest.TestCase):

    @staticmethod
    def _fake_stats(word, *args, **kwargs):
        return dict(step1=4.0, step2=2.0, step3=1.0,
                    wt_avg=2.5, max_grp=10, prob_finish=0.5,
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
