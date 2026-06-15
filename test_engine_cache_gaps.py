"""
Targeted tests closing specific coverage gaps in wordle_engine.py and
cache_sqlite.py.  Mirrors the conventions of test_wordle.py (small ANSWERS/
GUESSES word sets, ScoreCache(db_path, ANSWERS), MemoryScoreCache(), temp
files via tempfile).
"""
import logging
import os
import sqlite3
import tempfile
import unittest
from collections import defaultdict
from unittest import mock

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wordle_engine import (
    Solution, ScoringMethod, ResponseCache,
    calculate_response, score_groups, _encode_response, _ALL_GREEN_PATTERN,
    _perform_restriction, rank_guesses_by_group_then_entropy,
    evaluate_guess, _solve_subset, min_expected_guesses, verify_erd_cache,
    cache_all_scores, ERD_ANSWERS,
)
from cache_sqlite import ScoreCache, MemoryScoreCache, _is_disk_io_error


ANSWERS = ["crane", "slate", "trace", "stale", "tales",
           "least", "heart", "earth", "share", "rates"]
GUESSES = ANSWERS + ["brain", "stove", "cloud", "piano", "train"]


def make_cache(db_path=None):
    sc = ScoreCache(db_path, ANSWERS) if db_path else None
    cache = ResponseCache(ANSWERS, score_cache=sc)
    return cache, sc


class _ConnProxy:
    """Wraps a real sqlite3 connection, overriding execute/executemany.

    sqlite3.Connection.execute is a read-only C attribute, so it can't be
    patched in place; swap the cache's whole connection for this proxy
    instead.  Any callable overrides take precedence; everything else
    delegates to the real connection.
    """

    def __init__(self, real, execute=None, executemany=None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_execute_override", execute)
        object.__setattr__(self, "_executemany_override", executemany)

    def execute(self, *args, **kwargs):
        ov = object.__getattribute__(self, "_execute_override")
        if ov is not None:
            return ov(*args, **kwargs)
        return object.__getattribute__(self, "_real").execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        ov = object.__getattribute__(self, "_executemany_override")
        if ov is not None:
            return ov(*args, **kwargs)
        return object.__getattribute__(self, "_real").executemany(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


# ===========================================================================
# cache_sqlite.py gaps
# ===========================================================================

class TestCacheSQLiteDiskIOSwallow(unittest.TestCase):
    """Disk-I/O error handlers that swallow sqlite3.OperationalError, plus the
    re-raise branch for non-disk OperationalErrors."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    # Sanity: confirm which messages _is_disk_io_error matches so the
    # re-raise tests pick a message it does NOT match.
    def test_is_disk_io_error_matcher(self):
        self.assertTrue(_is_disk_io_error(
            sqlite3.OperationalError("disk I/O error")))
        self.assertFalse(_is_disk_io_error(
            sqlite3.OperationalError("database is locked")))

    # ---- write() ----
    def test_write_swallows_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])

        def boom(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        sc._conn = _ConnProxy(sc._conn, execute=boom)
        with self.assertLogs("wordle", level="WARNING") as cm:
            sc.write(key, "full", "heart", 2.5)   # must not raise
        self.assertTrue(any("failed" in m for m in cm.output))
        # _mem_cache is still populated despite the failed disk write.
        self.assertEqual(sc._mem_cache[(key, "full")], ("heart", 2.5, None, None))

    def test_write_reraises_non_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        sc._conn = _ConnProxy(sc._conn, execute=boom)
        with self.assertRaises(sqlite3.OperationalError):
            sc.write(key, "full", "heart", 2.5)

    # ---- write_decomposition() ----
    def test_write_decomposition_swallows_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)

        def boom(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        sc._conn = _ConnProxy(sc._conn, execute=boom)
        with self.assertLogs("wordle", level="WARNING") as cm:
            sc.write_decomposition("crane", b"\x00" * len(ANSWERS))
        self.assertTrue(any("write_decomposition" in m for m in cm.output))

    def test_write_decomposition_reraises_non_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        sc._conn = _ConnProxy(sc._conn, execute=boom)
        with self.assertRaises(sqlite3.OperationalError):
            sc.write_decomposition("crane", b"\x00" * len(ANSWERS))

    # ---- write_scores() incl. ROLLBACK path ----
    def test_write_scores_swallows_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])

        # BEGIN (execute) succeeds; the bulk executemany raises the disk-I/O
        # error; the subsequent ROLLBACK (execute) succeeds — exercising the
        # OperationalError handler and its inner ROLLBACK try/except.
        def boom(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        sc._conn = _ConnProxy(sc._conn, executemany=boom)
        with self.assertLogs("wordle", level="WARNING") as cm:
            sc.write_scores(key, [("crane", 1.0)], "entropy_gain")
        self.assertTrue(any("write_scores" in m for m in cm.output))

    def test_write_scores_swallows_disk_io_with_failing_rollback(self):
        """ROLLBACK itself raising OperationalError is swallowed (pass)."""
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])
        real = sc._conn

        def execute_side_effect(sql, *args, **kwargs):
            if str(sql).strip().upper().startswith("ROLLBACK"):
                raise sqlite3.OperationalError("cannot rollback")
            return real.execute(sql, *args, **kwargs)

        def boom_many(*a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        sc._conn = _ConnProxy(real, execute=execute_side_effect,
                              executemany=boom_many)
        with self.assertLogs("wordle", level="WARNING") as cm:
            sc.write_scores(key, [("crane", 1.0)], "entropy_gain")
        self.assertTrue(any("write_scores" in m for m in cm.output))

    def test_write_scores_reraises_non_disk_io_error(self):
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        sc._conn = _ConnProxy(sc._conn, executemany=boom)
        with self.assertRaises(sqlite3.OperationalError):
            sc.write_scores(key, [("crane", 1.0)], "entropy_gain")

    def test_write_scores_non_operational_error_rolls_back_and_raises(self):
        """A non-OperationalError (e.g. a programming error) in the bulk write
        hits the bare `except Exception` ROLLBACK+raise path (537-539)."""
        sc = ScoreCache(self.db, ANSWERS)
        key = ScoreCache.encode_subset(["crane", "slate"])

        def boom(*a, **k):
            raise ValueError("unexpected")

        sc._conn = _ConnProxy(sc._conn, executemany=boom)
        with self.assertRaises(ValueError):
            sc.write_scores(key, [("crane", 1.0)], "entropy_gain")


class TestCacheSQLiteCloseAndMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def test_close_without_checkpoint(self):
        """checkpoint_on_close=False skips checkpoint() on close (294->296)."""
        sc = ScoreCache(self.db, ANSWERS, checkpoint_on_close=False)
        self.assertFalse(sc.checkpoint_on_close)
        with mock.patch.object(sc, "checkpoint") as cp:
            sc.close()
        cp.assert_not_called()

    def test_close_with_checkpoint_default(self):
        """Default checkpoint_on_close=True does invoke checkpoint()."""
        sc = ScoreCache(self.db, ANSWERS)
        self.assertTrue(sc.checkpoint_on_close)
        with mock.patch.object(sc, "checkpoint") as cp:
            sc.close()
        cp.assert_called_once()

    def test_memory_cache_read_miss(self):
        """MemoryScoreCache read of an unwritten key returns None and bumps
        read_misses (604-605 via read_with_depth, and read())."""
        mc = MemoryScoreCache()
        key = ScoreCache.encode_subset(["crane", "slate"])
        self.assertIsNone(mc.read_with_depth(key, ERD_ANSWERS))
        self.assertEqual(mc.read_misses, 1)
        self.assertIsNone(mc.read(key, ERD_ANSWERS))
        self.assertEqual(mc.read_misses, 2)


# ===========================================================================
# wordle_engine.py gaps
# ===========================================================================

class TestPerformRestrictionUnknownColor(unittest.TestCase):
    def test_unknown_color_keeps_nothing(self):
        """An answer color that is not green/gray/yellow leaves keep False
        (265->267): no word survives."""
        words = ["crane", "slate", "trace"]
        out = _perform_restriction(words, 0, "c", 0, "purple")
        self.assertEqual(out, [])


class TestScoreGroupsEntropyZeroGroup(unittest.TestCase):
    def test_entropy_skips_zero_size_group(self):
        """A zero-size group makes the `if p > 0` guard False (428->426)."""
        groups = {0: 0, 1: 5, 2: 5}   # one zero-count group
        val = score_groups(groups, ScoringMethod.ENTROPY_GAIN)
        # Entropy of a 50/50 split over the two real groups = 1.0 bit.
        self.assertAlmostEqual(val, 1.0)


class TestRankGuessesPartialMethodCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def test_partial_method_cache_hit(self):
        """When one of the two methods is already cached for a word but the
        other isn't, only the missing method is queued to write (535->534:
        the `m not in cached` guard is False for the cached method)."""
        cache, sc = make_cache(self.db)
        words = ["crane", "slate", "trace"]
        subset_key = ScoreCache.encode_subset(words)

        # Pre-seed ONLY MAX_GROUP_SIZE for "crane"; ENTROPY_GAIN missing.
        from wordle_engine import score_word_multi
        scores = score_word_multi(
            "crane", words,
            [ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN],
            cache=cache)
        sc.write_scores(
            subset_key,
            [("crane", scores[ScoringMethod.MAX_GROUP_SIZE])],
            ScoringMethod.MAX_GROUP_SIZE.name.lower())

        ranked = rank_guesses_by_group_then_entropy(
            words, ["crane"], cache, sc)
        self.assertEqual(ranked, ["crane"])
        # ENTROPY_GAIN (the previously-missing method) was now written.
        eg = dict(sc.read_scores(
            subset_key, ScoringMethod.ENTROPY_GAIN.name.lower()))
        self.assertIn("crane", eg)


class TestPersistScoresEmpty(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def test_persist_scores_no_scores(self):
        """_persist_scores with no scores for the method skips the write
        (625->exit)."""
        cache, sc = make_cache(self.db)
        sol = Solution(ANSWERS, GUESSES, cache=cache, score_cache=sc)
        # word_scores is empty after reset, so nothing to persist.
        with mock.patch.object(sc, "write_scores") as ws:
            sol._persist_scores(ScoringMethod.ENTROPY_GAIN)
        ws.assert_not_called()


class TestSolutionFallback(unittest.TestCase):
    def test_apply_guess_fallback_to_full_vocabulary(self):
        """When the answer list is exhausted but the full vocabulary still
        has survivors, fall back to it (659-664)."""
        # answer_words excludes the true answer "brain"; all_words includes it.
        answers = ["crane", "slate", "trace"]
        all_words = answers + ["brain"]
        sol = Solution(answers, all_words=all_words)
        # Response of "crane" vs "brain": filters out every answer-list word
        # but keeps "brain" in the full vocabulary.
        resp = calculate_response("crane", "brain")
        remaining = sol.apply_guess("crane", resp)
        self.assertTrue(sol.fallback_active)
        self.assertGreaterEqual(remaining, 1)
        self.assertIn("brain", sol.current_words)


class TestEvaluateGuessBranches(unittest.TestCase):
    """Branches inside evaluate_guess / _solve_subset."""

    def test_evaluate_guess_n_none_and_no_cache(self):
        """n=None is derived from remaining (948) and cache=None walks the
        manual decomposition loop (957-960)."""
        remaining = ["crane", "slate", "trace", "stale"]
        status, cost, md, floor = evaluate_guess(
            remaining, "crane", None, None,
            n=None, best_erd=float('inf'),
            guesses=remaining, policy=ERD_ANSWERS, budget=None)
        self.assertIn(status, ('ok', 'useless', 'cutoff', 'pruned'))

    def test_evaluate_guess_matches_cached_decomposition(self):
        """cache=None path produces the same partition as the cache path."""
        remaining = ["crane", "slate", "trace", "stale", "tales"]
        cache, _ = make_cache(None)
        s1 = evaluate_guess(remaining, "crane", cache, None,
                            guesses=remaining, policy=ERD_ANSWERS)
        s2 = evaluate_guess(remaining, "crane", None, None,
                            guesses=remaining, policy=ERD_ANSWERS)
        self.assertEqual(s1[0], s2[0])
        if s1[1] is not None and s2[1] is not None:
            self.assertAlmostEqual(s1[1], s2[1])

    def test_solve_subset_depth_observer_called(self):
        """depth_observer is invoked (1058)."""
        remaining = ["crane", "slate", "trace"]
        cache, _ = make_cache(None)
        seen = []
        _solve_subset(remaining, cache, None, None, None, remaining,
                      ERD_ANSWERS, None, None, 0,
                      depth_observer=lambda d, n: seen.append((d, n)),
                      progress_callback=None)
        self.assertTrue(seen)
        self.assertEqual(seen[0], (0, len(remaining)))

    def test_solve_subset_budget_below_one(self):
        """budget < 1 returns infinite cost immediately (1060)."""
        remaining = ["crane", "slate", "trace"]
        cache, _ = make_cache(None)
        cost, md, floor, cutoff = _solve_subset(
            remaining, cache, None, 0, None, remaining,
            ERD_ANSWERS, None, None, 0, None, None)
        self.assertEqual(cost, float('inf'))
        self.assertTrue(floor)
        self.assertFalse(cutoff)

    def test_solve_subset_subgroup_cutoff(self):
        """A tight ceiling forces a subgroup-level cutoff so evaluate_guess
        returns ('cutoff', ...) from the sub_cutoff branch (1015), and
        _solve_subset reports cutoff with best_word None (1148)."""
        remaining = ["crane", "slate", "trace", "stale", "tales",
                     "least", "heart", "earth"]
        cache, _ = make_cache(None)
        # A ceiling well below any achievable ERD: every candidate prices out
        # at >= ceiling, so the search reports a cutoff lower bound.
        cost, md, floor, cutoff = _solve_subset(
            remaining, cache, None, None, None, remaining,
            ERD_ANSWERS, None, None, 0, None, None, ceiling=0.5)
        self.assertTrue(cutoff)
        self.assertEqual(cost, 0.5)

    def test_solve_subset_tie_not_below_best(self):
        """Exercises the 1135->1140 branch: a candidate with cost not strictly
        below best_erd still runs the progress_callback without updating the
        best.  Hit by a full small solve where later candidates tie/exceed."""
        remaining = ["crane", "slate", "trace", "stale"]
        cache, _ = make_cache(None)
        calls = []
        cost = min_expected_guesses(
            remaining, cache, None, guesses=remaining, policy=ERD_ANSWERS,
            progress_callback=lambda *a: calls.append(a))
        self.assertIsNotNone(cost)
        # progress_callback fires once per fully-evaluated top-level candidate.
        self.assertTrue(calls)


class TestVerifyErdCacheZeroGroup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def test_verify_skips_empty_group(self):
        """verify_erd_cache skips a zero-length response group (1249).

        group_words never yields an empty list, so inject a cache whose
        group_words returns one to reach the defensive k==0 continue.
        """
        cache, sc = make_cache(self.db)
        words = ["crane", "slate", "trace"]
        root_key = ScoreCache.encode_subset(words)
        sc.write(root_key, ERD_ANSWERS, "crane", 2.0)

        class GroupWithEmpty:
            def group_words(self, guess, subset):
                g = defaultdict(list)
                # Real partition plus one spurious empty group.
                for w in subset:
                    pat = _encode_response(calculate_response(guess, w))
                    g[pat].append(w)
                g[999] = []   # the empty group that triggers k==0 -> continue
                return g

        report = verify_erd_cache(words, GroupWithEmpty(), sc, ERD_ANSWERS)
        self.assertTrue(report)
        self.assertEqual(report[0]['n'], len(words))


class TestCacheAllScoresMemoryCacheSkip(unittest.TestCase):
    def test_memory_cache_has_no_write_scores(self):
        """cache_all_scores is a no-op for a cache lacking write_scores."""
        mc = MemoryScoreCache()
        self.assertFalse(hasattr(mc, "write_scores"))
        cache, _ = make_cache(None)
        key = ScoreCache.encode_subset(["crane", "slate"])
        # Should simply return without error.
        cache_all_scores("crane", ["crane", "slate"], mc, key, cache=cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
