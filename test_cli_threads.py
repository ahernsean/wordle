"""Targeted tests for the scattered uncovered branches of the two background
solver threads (ERDSolver, BranchPrecacheSolver) in wordle.py.

These drive the thread methods synchronously (calling run()/_scan() directly,
or just the small setters) rather than starting real threads, so they are
deterministic. The happy-path scan behavior is already covered by the
TestERDSolver*/TestBranchPrecacheSolver suites in test_wordle.py; this file
fills in the pause/resume, trivial-position, cancellation, and persist
open/close edges.
"""
import os
import tempfile
import unittest
from unittest import mock

import wordle
from wordle import ERDSolver, BranchPrecacheSolver
from wordle_engine import ERD_ALL
from cache_sqlite import ScoreCache, MemoryScoreCache


def _w(i):
    return f"w{i:04d}"  # 5 ASCII chars, valid for encode_subset


class _FakeRCache:
    """Minimal ResponseCache stand-in for ranking/scan calls."""
    def __init__(self, words):
        self.answer_words = words

    @staticmethod
    def group_words(word, current_words):
        return {}

    @staticmethod
    def group_counts(word, current_words):
        return {}


class TestERDSolverEdges(unittest.TestCase):
    def setUp(self):
        self.words = [_w(i) for i in range(10)]

    def _solver(self, words=None, **kw):
        words = self.words if words is None else words
        kw.setdefault("policy", ERD_ALL)
        kw.setdefault("persist", False)
        kw.setdefault("seed_mem_cache", MemoryScoreCache())
        return ERDSolver(words, words, words, None, **kw)

    def test_pause_resume_setters(self):
        s = self._solver()
        s.pause()
        self.assertFalse(s._paused.is_set())
        s.resume()
        self.assertTrue(s._paused.is_set())
        s.stop()
        self.assertTrue(s._cancel.is_set())

    def test_run_trivial_position(self):
        s = self._solver(words=[_w(0), _w(1)])
        s.run()  # <= 2 words: sentinel, no scan
        self.assertEqual(s.root_total, -1)

    def test_run_persist_opens_and_closes_cache(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        opened = {}

        s = ERDSolver(self.words, self.words, self.words, tmp.name,
                      policy=ERD_ALL, persist=True)
        # Replace the heavy scan with a probe so run() just exercises the
        # persist open + finally-close wiring.
        def fake_scan(score_cache):
            opened["cache"] = score_cache
        s._scan = fake_scan
        s.run()
        self.assertIn("cache", opened)

    def test_on_root_progress_respects_cancel(self):
        s = self._solver()
        s.stop()
        s._on_root_progress(5, 10, "crane", 1.2)
        self.assertEqual(s.root_done, 0)  # unchanged: cancelled
        s._cancel.clear()
        s._on_root_progress(5, 10, "crane", 1.2)
        self.assertEqual(s.root_done, 5)
        self.assertEqual(s.root_best, ("crane", 1.2))

    def test_ranked_root_guesses_short_circuits_on_cancel(self):
        s = self._solver()
        s.stop()
        self.assertEqual(s._ranked_root_guesses(MemoryScoreCache()), self.words)

    def test_scan_skips_when_root_already_cached(self):
        s = self._solver()
        s._rcache = _FakeRCache(self.words)
        sc = MemoryScoreCache()
        sc.set_scope("scope")
        root_key = ScoreCache.encode_subset(self.words)
        sc.write(root_key, ERD_ALL, "crane", 1.5, max_depth=1, solve_budget=None)
        with mock.patch("wordle.min_expected_guesses",
                        side_effect=AssertionError("must not scan a cached root")):
            s._scan(sc)  # returns at the cache hit, never scans

    def test_scan_progress_and_ready_announcement(self):
        # last_guess set exercises the periodic "Targeted scan" heartbeat
        # branch (its 30s guard returns early on the immediate call).
        s = self._solver(last_guess=("crane", wordle.parse_response("ggggg")))
        s._rcache = _FakeRCache(self.words)
        s._ranked_root_guesses = lambda sc: self.words
        sc = MemoryScoreCache()
        sc.set_scope("scope")

        def fake_meg(remaining, cache, score_cache, guesses=None, policy=None,
                     progress_callback=None, cancel_check=None, heartbeat=None,
                     budget=None):
            cancel_check()  # exercise the _cancel_or_paused closure
            progress_callback(1, len(guesses), guesses[0], 1.5)
            if heartbeat:
                heartbeat()
            return 1.80

        printed = []
        with mock.patch("wordle.min_expected_guesses", side_effect=fake_meg), \
             mock.patch("builtins.print",
                        side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))):
            s._scan(sc)
        self.assertTrue(any("ERD ready" in p for p in printed))
        self.assertTrue(s.word_stats)

    def test_scan_cancelled_after_ranking(self):
        s = self._solver()
        s._rcache = _FakeRCache(self.words)
        # Ranking itself observes the cancellation, so _scan bails right after.
        def cancel_during_ranking(sc):
            s.stop()
            return self.words
        s._ranked_root_guesses = cancel_during_ranking
        with mock.patch("wordle.min_expected_guesses",
                        side_effect=AssertionError("must not scan after cancel")):
            s._scan(MemoryScoreCache())

    def test_scan_cancelled_mid_scan(self):
        s = self._solver()
        s._rcache = _FakeRCache(self.words)
        s._ranked_root_guesses = lambda sc: self.words
        sc = MemoryScoreCache()
        sc.set_scope("scope")

        def fake_meg(*a, **k):
            s.stop()      # superseded mid-scan
            return None    # aborted

        with mock.patch("wordle.min_expected_guesses", side_effect=fake_meg):
            s._scan(sc)   # hits the "cancelled mid-scan" return

    def test_scan_swallows_operational_error(self):
        s = self._solver()
        s._rcache = _FakeRCache(self.words)
        s._ranked_root_guesses = lambda sc: self.words
        sc = MemoryScoreCache()
        sc.set_scope("scope")
        import sqlite3
        with mock.patch("wordle.min_expected_guesses",
                        side_effect=sqlite3.OperationalError("disk I/O error")):
            s._scan(sc)   # logged and swallowed, no raise


class TestBranchPrecacheSolverEdges(unittest.TestCase):
    def setUp(self):
        self.words = [_w(i) for i in range(8)]
        self.branches = [(0, self.words)]

    def _solver(self, path, **kw):
        kw.setdefault("anchor_word_count", len(self.words))
        return BranchPrecacheSolver(
            "crane", self.branches, self.words, self.words, path, **kw)

    def test_pause_resume_setters(self):
        s = self._solver(None)
        s.pause()
        self.assertFalse(s._paused.is_set())
        s.resume()
        self.assertTrue(s._paused.is_set())

    def test_run_returns_immediately_when_cancelled(self):
        s = self._solver(None)
        s.stop()
        s.run()  # cancel set before opening the cache: returns at the top
        self.assertEqual(s.branches_done, 0)

    def test_branches_line_formatting(self):
        s = self._solver(None)
        self.assertIn("Branches:", s.branches_line())
        self.assertIn("done", s.branches_line(done=True))
        self.assertIn("cached", s._branches_starting_line())

    def test_run_cancel_inside_loop(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        s = self._solver(tmp.name)
        # Cancel as soon as ranking begins so the per-branch loop takes its
        # cancellation exit after opening the real cache.
        def cancel_then_raise(*a, **k):
            s.stop()
            return []
        with mock.patch("wordle.rank_candidates_by_max_group_size_then_entropy_gain",
                        side_effect=cancel_then_raise):
            s.run()

    def test_run_swallows_operational_error(self):
        import sqlite3
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        s = self._solver(tmp.name)
        with mock.patch("wordle.rank_candidates_by_max_group_size_then_entropy_gain",
                        return_value=list(self.words)), \
             mock.patch("wordle.min_expected_guesses",
                        side_effect=sqlite3.OperationalError("disk I/O error")):
            s.run()  # OperationalError is logged and swallowed; finally closes


if __name__ == "__main__":
    unittest.main()
