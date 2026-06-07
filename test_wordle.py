"""
Tests for Wordle engine correctness and cache behavior.
Run with:  python test_wordle.py
"""
import math
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wordle_engine import (
    Solution, ScoringMethod, ResponseCache,
    calculate_response, score_groups, calculate_group_counts,
    min_expected_guesses,
)
from cache_sqlite import ScoreCache
from wordle import _multistep_stats, _erd_solve_scores


# Small deterministic word sets used across all tests.
ANSWERS = ["crane", "slate", "trace", "stale", "tales",
           "least", "heart", "earth", "share", "rates"]
GUESSES = ANSWERS + ["brain", "stove", "cloud", "piano", "train"]


def make_solution(db_path=None):
    cache = ResponseCache(ANSWERS)
    sc = ScoreCache(db_path, ANSWERS) if db_path else None
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

    def test_symmetric_known_pair(self):
        # crane vs trace and trace vs crane differ (C/T swap yellow vs gray)
        r1 = calculate_response("crane", "trace")
        r2 = calculate_response("trace", "crane")
        self.assertNotEqual(r1, r2)


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

    def test_minimax(self):
        groups = {0: 5, 1: 3, 2: 7, 3: 2}
        self.assertEqual(score_groups(groups, ScoringMethod.MINIMAX), 7)

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
        methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MINIMAX]
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

    def test_scores_sorted_minimax_ascending(self):
        self.soln.compute_scores(GUESSES, ScoringMethod.MINIMAX)
        scores = [s for _, s in self.soln.scores]
        self.assertEqual(scores, sorted(scores))


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
        sc.write_scores([("crane", 3.14159), ("slate", 2.71828)], "entropy_gain")
        result = dict(sc.read_scores("entropy_gain"))
        self.assertAlmostEqual(result["crane"], 3.14159)
        self.assertAlmostEqual(result["slate"], 2.71828)

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
        self.assertIsNone(sc.read_scores("entropy_gain"))

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
        sc1.write_scores([("crane", 3.14)], "entropy_gain")
        self.assertIsNone(sc2.read_scores("entropy_gain"))

    def test_overwrite_replaces_value(self):
        sc = ScoreCache(self.db, ANSWERS)
        sc.write_scores([("crane", 1.0)], "entropy_gain")
        sc.write_scores([("crane", 9.9)], "entropy_gain")
        result = dict(sc.read_scores("entropy_gain"))
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
        # Simulate a row written with the old encoding (null-separated)
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

        # Opening ScoreCache should delete the old-format row
        ScoreCache(self.db, ANSWERS)
        conn2 = _sqlite3.connect(self.db)
        rows = conn2.execute(
            "SELECT COUNT(*) FROM lookahead_result WHERE instr(subset_key, char(0)) > 0"
        ).fetchone()[0]
        conn2.close()
        self.assertEqual(rows, 0)


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

    def test_second_session_is_all_cache_hits(self):
        make_solution(db_path=self.db).compute_scores(
            GUESSES, ScoringMethod.ENTROPY_GAIN)

        s2 = make_solution(db_path=self.db)
        computed = [0]
        orig = s2.score_cache.read_scores  # pre-load will populate word_scores

        # After first compute_scores call, _db_loaded_methods should be set
        s2.compute_scores(GUESSES, ScoringMethod.ENTROPY_GAIN)
        # If loaded from DB, all words should already be in word_scores
        # and _db_loaded_methods should contain ENTROPY_GAIN
        self.assertIn(ScoringMethod.ENTROPY_GAIN, s2._db_loaded_methods)

    def test_mid_game_not_persisted(self):
        s = make_solution(db_path=self.db)
        pattern = calculate_response("crane", "slate")
        s.apply_guess("crane", pattern)
        self.assertFalse(s._is_full_game())

        s.compute_scores(s.current_words, ScoringMethod.ENTROPY_GAIN)

        # Nothing should have been written to DB
        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read_scores("entropy_gain"))

    def test_multi_method_persist_and_reload(self):
        methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MINIMAX]
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
            hit = sc.read(ScoreCache.encode_subset(subset), 'erd_answers')
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit[1], result, places=10)
        finally:
            os.unlink(tmp.name)

    def test_expired_deadline_returns_none(self):
        already_past = time.time() - 1
        result = min_expected_guesses(ANSWERS, self.cache, None,
                                      deadline=already_past)
        self.assertIsNone(result)

    def test_result_is_at_least_one(self):
        result = min_expected_guesses(ANSWERS[:5], self.cache, None)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 1.0)

    def test_larger_set_costs_more_than_smaller(self):
        cost3 = min_expected_guesses(ANSWERS[:3], self.cache, None)
        cost5 = min_expected_guesses(ANSWERS[:5], self.cache, None)
        self.assertLess(cost3, cost5)


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
                hit = sc.read(ScoreCache.encode_subset([w]), 'erd_all')
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
            root_hit = sc.read(ScoreCache.encode_subset(words), 'erd_all')
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
        rows = sc.read_scores("entropy_gain")
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

    def test_mid_game_scores_not_persisted(self):
        soln = make_solution(db_path=self.db)
        pattern = calculate_response("crane", "slate")
        soln.apply_guess("crane", pattern)
        self.assertFalse(soln._is_full_game())

        _multistep_stats("slate", soln)

        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNone(sc.read_scores("entropy_gain"))


# ---------------------------------------------------------------------------
# Bug: min_expected_guesses derives policy from guesses-is-not-None, ignoring
# the caller's intended policy.  The fix adds an explicit policy= parameter.
# ---------------------------------------------------------------------------

class TestERDPolicyParameter(unittest.TestCase):
    """
    min_expected_guesses must accept and honour an explicit policy= argument
    so callers can separate 'erd_answers', 'erd_all', 'erd_constrained' in
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
        """Passing policy='erd_answers' with guesses= stores under 'erd_answers', not 'erd_all'."""
        sc = self._sc()
        subset = ANSWERS[:4]
        key = ScoreCache.encode_subset(subset)

        min_expected_guesses(subset, self.cache, sc,
                             guesses=subset, policy='erd_answers')

        self.assertIsNotNone(sc.read(key, 'erd_answers'),
                             "result must be stored under the explicit policy")
        self.assertIsNone(sc.read(key, 'erd_all'),
                          "result must NOT be stored under the derived policy")

    def test_possible_answers_warmer_readable_by_solve(self):
        """
        Results written with policy='erd_answers' (POSSIBLE_ANSWERS warmer) are
        readable by _erd_solve_scores under policy='erd_answers'.

        With the bug, the warmer calls min_expected_guesses(guesses=words) which
        stores under 'erd_all'; _erd_solve_scores looks under 'erd_answers' → miss.
        """
        words = ANSWERS[:5]
        sc = self._sc()
        import types
        soln = types.SimpleNamespace(
            current_words=list(words),
            score_cache=sc,
            cache=ResponseCache(words),
        )

        # Simulate POSSIBLE_ANSWERS warmer: guesses=current_words, policy='erd_answers'
        min_expected_guesses(words, soln.cache, sc,
                             guesses=words, policy='erd_answers')

        scores = _erd_solve_scores(soln, score_cache=sc, policy='erd_answers')
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
        min_expected_guesses(subset, self.cache, sc, policy='erd_answers')
        self.assertIsNotNone(sc.read(key, 'erd_answers'),
                             "erd_answers must be stored after first call")
        self.assertIsNone(sc.read(key, 'erd_all'),
                          "erd_all must be untouched after erd_answers-only call")

        # Now write erd_all (guesses=GUESSES, different vocabulary)
        min_expected_guesses(subset, self.cache, sc,
                             guesses=GUESSES, policy='erd_all')
        self.assertIsNotNone(sc.read(key, 'erd_all'),
                             "erd_all must be stored after second call")
        self.assertIsNotNone(sc.read(key, 'erd_answers'),
                             "erd_answers must still be present — independent slot")


# ---------------------------------------------------------------------------
# Bug: _multistep_stats calls min_expected_guesses without guesses=, so it
# always uses policy='erd_answers' regardless of all_words mode.
# After the fix it passes guesses=all_words and policy='erd_all'.
# ---------------------------------------------------------------------------

class TestMultistepStatsERDPolicy(unittest.TestCase):
    """
    When all_words is provided, _multistep_stats should cache subgroup ERDs
    under 'erd_all'.  Currently the guesses= arg is omitted so results land
    under 'erd_answers' regardless of mode.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmpdir.name, 'test.sqlite3')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mid_game_soln(self):
        """Apply 'piano' → leaves 6 words including subgroup-producing 'heart'."""
        soln = make_solution(db_path=self.db)
        # 'piano' shares no a/e/i/o/u overlap with crane/rates/tales/trace/earth;
        # guessing it against 'slate' leaves: slate, trace, stale, least, heart, share.
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

    def test_all_words_mode_caches_under_erd_all(self):
        """
        After _multistep_stats("heart", soln, all_words=GUESSES), at least one
        subgroup ERD must be stored under 'erd_all', not 'erd_answers'.
        ('heart' vs the 6-word remaining set creates a 2-word subgroup.)
        """
        soln = self._mid_game_soln()
        key = self._find_subgroup_key("heart", soln.current_words)
        self.assertIsNotNone(key, "setup must produce a subgroup with k>=2")

        _multistep_stats("heart", soln, all_words=GUESSES)

        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNotNone(sc.read(key, 'erd_all'),
                             "ERD subgroups must be cached under 'erd_all' in any-word mode")

    def test_default_mode_caches_under_erd_answers(self):
        """
        Without all_words, _multistep_stats uses 'erd_answers' — this is the
        existing correct behaviour for POSSIBLE_ANSWERS / default mode.
        """
        soln = self._mid_game_soln()
        key = self._find_subgroup_key("heart", soln.current_words)
        self.assertIsNotNone(key, "setup must produce a subgroup with k>=2")

        _multistep_stats("heart", soln)  # all_words=None

        sc = ScoreCache(self.db, ANSWERS)
        self.assertIsNotNone(sc.read(key, 'erd_answers'),
                             "ERD subgroups must be cached under 'erd_answers' in default mode")


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
                                 guesses=GUESSES, policy='erd_all')

            scores = _erd_solve_scores(soln, score_cache=sc,
                                       policy='erd_all', guesses=GUESSES)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
