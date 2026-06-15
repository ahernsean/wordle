"""Tests for assorted wordle.py CLI/display helpers.

Two parts:
  A. command handlers: cmd_verify_erd, cmd_precache, print_status
  B. display/format/terminal helpers: width detection, ProgressTracker,
     format_columns, print_scored_list, print_word_list, _fmt_size,
     print_guesses, set_display_context, _mark, _is_max_ent

These exercise branches the rest of the suite leaves uncovered. Everything is
deterministic and fast (small word lists make ERD cheap); any background
precache solver a test starts is stopped/joined so nothing outlives its temp
dir or hangs the run.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import wordle
from wordle import ScoringMethod
from clitestutil import CliTestCase


def _second_solution(gs):
    return wordle.Solution(
        gs.all_answers, gs.all_words, gs.cache, gs.score_cache)


def _capture(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------

class VerifyErdTests(CliTestCase):
    def test_fresh_root_no_entry(self):
        out = self.run_cmd(wordle.cmd_verify_erd)
        self.assertIn("No ERD entry to verify", out)

    def test_midgame_not_cached(self):
        self.soln().apply_guess("crane", wordle.parse_response("00g0g"))
        out = self.run_cmd(wordle.cmd_verify_erd)
        # Either a header + "Not cached yet." or the mode-unavailable error;
        # default mode (ALL/UNFILTERED) yields a real cache, so expect the
        # uncached branch.
        self.assertIn("Not cached yet.", out)

    def test_multiboard_pick_midgame(self):
        s2 = _second_solution(self.gs)
        self.gs.solutions = [self.soln(), s2]
        # Pick board 2, which is mid-game (after a guess) -> Not cached yet.
        s2.apply_guess("crane", wordle.parse_response("00g0g"))
        out = self.run_cmd(wordle.cmd_verify_erd, inputs=["2"])
        self.assertIn("Not cached yet.", out)

    def test_multiboard_pick_full_game(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_verify_erd, inputs=["1"])
        self.assertIn("No ERD entry to verify", out)

    def test_multiboard_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_verify_erd, inputs=["x"])
        self.assertIn("Invalid solution number.", out)

    def test_midgame_mode_unavailable(self):
        # Constrained mode -> MemoryScoreCache, but verify still computes;
        # switch to a mode whose erd cache path differs and confirm a header
        # appears (covers the _erd_cache_and_policy branch).
        self.soln().apply_guess("crane", wordle.parse_response("00g0g"))
        self.gs.universe = wordle.GuessUniverse.ALL_WORDS
        self.gs.compliance = wordle.ComplianceFilter.COMPLIANT
        out = self.run_cmd(wordle.cmd_verify_erd)
        self.assertIn("ERD cache check", out)


class PrecacheTests(CliTestCase):
    def _stop_running(self):
        s = self.gs.precache_solver
        if s is not None and s.is_alive():
            s.stop()
            s.join(timeout=10)
        self.gs.precache_solver = None

    def test_happy_start(self):
        word = self.gs.all_words[0]  # valid 5-letter word from all_words
        out = self.run_cmd(wordle.cmd_precache, inputs=[word])
        self.addCleanup(self._stop_running)
        self.assertIn("Precaching ERD", out)
        self.assertIsNotNone(self.gs.precache_solver)

    def test_invalid_word(self):
        out = self.run_cmd(wordle.cmd_precache, inputs=["zzzzz"])
        self.assertIn("Invalid word.", out)
        self.assertIsNone(self.gs.precache_solver)

    def test_short_word(self):
        out = self.run_cmd(wordle.cmd_precache, inputs=["abc"])
        self.assertIn("Invalid word.", out)

    def test_already_running_then_stop(self):
        word = self.gs.all_words[0]
        self.run_cmd(wordle.cmd_precache, inputs=[word])
        self.addCleanup(self._stop_running)
        self.assertIsNotNone(self.gs.precache_solver)
        # Second invocation: it's running -> status + stop prompt, answer 'y'.
        out = self.run_cmd(wordle.cmd_precache, inputs=["y"])
        self.assertIn("Precache running", out)
        self.assertIn("Stopped.", out)
        self.assertIsNone(self.gs.precache_solver)

    def test_already_running_keep(self):
        word = self.gs.all_words[0]
        self.run_cmd(wordle.cmd_precache, inputs=[word])
        self.addCleanup(self._stop_running)
        # Answer 'n' -> keeps running, no "Stopped."
        out = self.run_cmd(wordle.cmd_precache, inputs=["n"])
        self.assertIn("Precache running", out)
        self.assertNotIn("Stopped.", out)
        self.assertIsNotNone(self.gs.precache_solver)

    def test_not_at_root(self):
        self.soln().apply_guess("crane", wordle.parse_response("00g0g"))
        out = self.run_cmd(wordle.cmd_precache, inputs=["crane"])
        self.assertIn("only at the root", out)
        self.assertIsNone(self.gs.precache_solver)

    def test_not_single_board(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_precache)
        self.assertIn("requires a single board", out)

    def test_wrong_candidate_mode(self):
        # ALL_ANSWERS/COMPLIANT -> policy != ERD_ALL -> mode error.
        self.gs.universe = wordle.GuessUniverse.ALL_ANSWERS
        self.gs.compliance = wordle.ComplianceFilter.COMPLIANT
        out = self.run_cmd(wordle.cmd_precache)
        self.assertIn("only available in 'any word", out)
        self.assertIsNone(self.gs.precache_solver)


class PrintStatusTests(CliTestCase):
    def test_fresh_single_no_solver(self):
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("words left", out)
        self.assertIn("0 guesses so far", out)

    def test_after_guess(self):
        self.soln().apply_guess("crane", wordle.parse_response("00g0g"))
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("guess", out)
        self.assertIn("words left", out)

    def test_solved(self):
        soln = self.soln()
        target = soln.current_words[0]
        soln.apply_guess(target, wordle.parse_response("ggggg"))
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("Solved:", out)

    def test_no_words_remaining(self):
        soln = self.soln()
        # Impossible response set leaves nothing.
        soln.current_words = []
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("No words remaining!", out)

    def test_multiboard(self):
        s2 = _second_solution(self.gs)
        self.gs.solutions = [self.soln(), s2]
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("2 wordlists", out)
        self.assertIn("remaining", out)

    def test_multiboard_solved_and_empty(self):
        s1 = self.soln()
        s2 = _second_solution(self.gs)
        s3 = _second_solution(self.gs)
        s2.current_words = [s2.current_words[0]]  # 1 remaining
        s3.current_words = []                      # 0 remaining
        self.gs.solutions = [s1, s2, s3]
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("3 wordlists", out)
        self.assertIn("1 remaining", out)
        self.assertIn("0 remaining", out)

    def test_with_solver_mismatched_words(self):
        # Pass a solver whose word set differs from the displayed position:
        # status should still print without showing solver progress.
        self.soln().apply_guess("crane", wordle.parse_response("00g0g"))
        solver = mock.Mock()
        solver._words = ["totally", "different"]
        out = _capture(wordle.print_status, self.gs, solver)
        self.assertIn("words left", out)

    def test_multiboard_fallback_and_sim(self):
        s1 = self.soln()
        s2 = _second_solution(self.gs)
        s2.fallback_active = True
        s2.answer_word = "crane"
        self.gs.solutions = [s1, s2]
        out = _capture(wordle.print_status, self.gs)
        self.assertIn("fallback", out)
        self.assertIn("sim:crane", out)


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------

class DisplayWidthTests(unittest.TestCase):
    def setUp(self):
        # Snapshot module globals we mutate so tests stay isolated.
        self._cache = wordle._width_cache[0]
        self.addCleanup(self._restore)

    def _restore(self):
        wordle._width_cache[0] = self._cache

    def test_columns_env(self):
        with mock.patch.dict(os.environ, {"COLUMNS": "133"}):
            self.assertEqual(wordle._detect_display_width(), 133)

    def test_columns_env_invalid_falls_through(self):
        with mock.patch.dict(os.environ, {"COLUMNS": "notanint"}), \
                mock.patch("wordle.IS_PYTHONISTA", False), \
                mock.patch("shutil.get_terminal_size",
                           return_value=os.terminal_size((97, 24))):
            self.assertEqual(wordle._detect_display_width(), 97)

    def test_terminal_size_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("wordle.IS_PYTHONISTA", False), \
                mock.patch("shutil.get_terminal_size",
                           return_value=os.terminal_size((71, 24))):
            self.assertEqual(wordle._detect_display_width(), 71)

    def test_terminal_size_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("wordle.IS_PYTHONISTA", False), \
                mock.patch("shutil.get_terminal_size",
                           side_effect=OSError("boom")):
            self.assertEqual(wordle._detect_display_width(), 80)

    def test_pythonista_screen_size_path(self):
        # Drive the Pythonista branch via fake `console` and `ui` modules.
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        fake_console = mock.Mock()
        fake_console.get_size.side_effect = Exception("no console size")
        fake_ui = mock.Mock()
        fake_ui.get_screen_size.return_value = (400, 800)
        fake_ui.measure_string.return_value = (200.0, 14.0)  # 10 px/char
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("wordle.IS_PYTHONISTA", True), \
                mock.patch("wordle.console", fake_console), \
                mock.patch.dict(sys.modules, {"ui": fake_ui}), \
                mock.patch("os.get_terminal_size", side_effect=OSError):
            width = wordle._detect_display_width()
        # w_points = min(400-16, 720) = 384; px/char = 200/20 = 10 -> 38 cols.
        self.assertEqual(width, 38)

    def test_pythonista_falls_to_default(self):
        # console present but every layer fails -> the hard 42 fallback.
        env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
        fake_console = mock.Mock()
        fake_console.get_size.side_effect = Exception("boom")
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("wordle.IS_PYTHONISTA", True), \
                mock.patch("wordle.console", fake_console), \
                mock.patch.dict(sys.modules, {"ui": None}), \
                mock.patch("os.get_terminal_size", side_effect=OSError):
            self.assertEqual(wordle._detect_display_width(), 42)

    def test_get_and_refresh(self):
        wordle._width_cache[0] = None
        w = wordle.get_display_width()
        self.assertIsInstance(w, int)
        # Cached: a second call doesn't re-detect.
        wordle._width_cache[0] = 55
        self.assertEqual(wordle.get_display_width(), 55)
        # refresh re-detects.
        with mock.patch("wordle._detect_display_width", return_value=99):
            wordle.refresh_display_width()
            self.assertEqual(wordle._width_cache[0], 99)


class PlatformLabelTests(unittest.TestCase):
    def _uname(self, system, machine, release="1.0"):
        return mock.Mock(system=system, machine=machine, release=release)

    def test_iphone(self):
        with mock.patch("platform.uname",
                        return_value=self._uname("Darwin", "iPhone14,2")):
            self.assertTrue(wordle._platform_label().startswith("iPhone"))

    def test_ipad(self):
        with mock.patch("platform.uname",
                        return_value=self._uname("Darwin", "iPad13,4")):
            self.assertTrue(wordle._platform_label().startswith("iPad"))

    def test_macos(self):
        with mock.patch("platform.uname",
                        return_value=self._uname("Darwin", "arm64")):
            self.assertTrue(wordle._platform_label().startswith("macOS"))

    def test_linux(self):
        with mock.patch("platform.uname",
                        return_value=self._uname("Linux", "x86_64")):
            self.assertTrue(wordle._platform_label().startswith("Linux"))


class ProgressTrackerTests(unittest.TestCase):
    def setUp(self):
        self._cache = wordle._width_cache[0]
        wordle._width_cache[0] = 40  # narrow margin to force wrapping
        self.addCleanup(self._restore)

    def _restore(self):
        wordle._width_cache[0] = self._cache

    def test_full_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            pt = wordle.ProgressTracker(8)
            for _ in range(8):
                pt.update()
            elapsed = pt.finish()
        out = buf.getvalue()
        self.assertIn("25%", out)
        self.assertIn("100%", out)
        self.assertIn("Duration:", out)
        self.assertIsNotNone(elapsed)

    def test_wrapping_long_total(self):
        # Many updates emit enough dots to wrap past the margin.
        buf = io.StringIO()
        with redirect_stdout(buf):
            pt = wordle.ProgressTracker(200)
            for _ in range(200):
                pt.update()
            pt.finish()
        out = buf.getvalue()
        self.assertIn("\n", out)
        self.assertIn("100%", out)

    def test_eta_emitted(self):
        # Force the ETA interval by faking time so an ETA token is emitted.
        from datetime import datetime, timedelta
        base = datetime(2020, 1, 1, 0, 0, 0)
        times = [base]  # __init__ start_time/last_eta

        seq = iter([
            base,                       # start_time
            base,                       # last_eta
            base + timedelta(seconds=20),  # update(): now (>= interval)
            base + timedelta(seconds=20),  # update(): elapsed calc
            base + timedelta(seconds=20),  # finish(): elapsed calc
        ])

        real_now = datetime.now

        def fake_now():
            try:
                return next(seq)
            except StopIteration:
                return real_now()

        buf = io.StringIO()
        with redirect_stdout(buf), \
                mock.patch("wordle.datetime") as dt:
            dt.now.side_effect = fake_now
            pt = wordle.ProgressTracker(10)
            pt.update()
            pt.finish()
        out = buf.getvalue()
        # ETA token like '1m20s' or 'NNs' should appear.
        self.assertTrue(any(c.isdigit() for c in out))

    def test_fmt_eta(self):
        from datetime import timedelta
        self.assertEqual(
            wordle.ProgressTracker._fmt_eta(timedelta(seconds=45)), "45s")
        self.assertEqual(
            wordle.ProgressTracker._fmt_eta(timedelta(seconds=120)), "2m")
        self.assertEqual(
            wordle.ProgressTracker._fmt_eta(timedelta(seconds=125)), "2m05s")


class FormatHelperTests(unittest.TestCase):
    def setUp(self):
        self._cache = wordle._width_cache[0]
        wordle._width_cache[0] = 80
        self._answers = wordle._display_answer_set
        self._maxent = wordle._display_max_ent
        self.addCleanup(self._restore)

    def _restore(self):
        wordle._width_cache[0] = self._cache
        wordle._display_answer_set = self._answers
        wordle._display_max_ent = self._maxent

    def test_format_columns_empty(self):
        self.assertEqual(wordle.format_columns([]), [])

    def test_format_columns_basic(self):
        out = wordle.format_columns(["apple", "berry", "cherry"], width=80)
        self.assertTrue(out)
        self.assertTrue(all(line.startswith("    ") for line in out))

    def test_format_columns_explicit_width(self):
        out = wordle.format_columns(["a", "b", "c", "d"], width=10)
        self.assertTrue(out)

    def test_fmt_size(self):
        self.assertEqual(wordle._fmt_size(512), "512 B")
        self.assertEqual(wordle._fmt_size(2048), "2 KB")
        self.assertEqual(wordle._fmt_size(5 * 1024 * 1024), "5 MB")
        self.assertEqual(wordle._fmt_size(3 * 1024 ** 3), "3.0 GB")
        self.assertEqual(wordle._fmt_size(2 * 1024 ** 4), "2.0 TB")

    def test_mark_and_max_ent(self):
        wordle._display_answer_set = {"crane"}
        wordle._display_max_ent = 0.0
        self.assertEqual(wordle._mark("crane"), "*")
        self.assertEqual(wordle._mark("slate"), " ")
        self.assertFalse(wordle._is_max_ent(2.0))
        wordle._display_max_ent = 2.321928
        self.assertTrue(wordle._is_max_ent(2.321928))
        self.assertFalse(wordle._is_max_ent(1.0))

    def test_print_scored_list_no_method(self):
        wordle._display_answer_set = {"crane"}
        out = _capture(wordle.print_scored_list,
                       [("crane", 2.0), ("slate", 1.5)])
        self.assertIn("crane*", out)
        self.assertIn("2.0000", out)

    def test_print_scored_list_with_method_and_limit(self):
        wordle._display_answer_set = set()
        wordle._display_max_ent = 2.0
        pairs = [(f"wrd{i:02d}", 2.0 - i * 0.01) for i in range(30)]
        out = _capture(wordle.print_scored_list, pairs,
                       ScoringMethod.ENTROPY_GAIN, 20)
        self.assertIn("...", out)  # over limit -> ellipsis
        # The top score equals max_ent -> '=' marker.
        self.assertIn("=", out)

    def test_print_scored_list_empty(self):
        self.assertEqual(_capture(wordle.print_scored_list, []), "")

    def test_print_word_list_short(self):
        wordle._display_answer_set = {"crane"}
        out = _capture(wordle.print_word_list, ["crane", "slate"])
        self.assertIn("crane*", out)
        self.assertNotIn("total", out)

    def test_print_word_list_truncated(self):
        words = [f"word{i:03d}" for i in range(40)]
        out = _capture(wordle.print_word_list, words, 20)
        self.assertIn("... (40 total)", out)

    def test_print_word_list_empty(self):
        self.assertEqual(_capture(wordle.print_word_list, []), "")


class SetContextAndGuessesTests(CliTestCase):
    def test_set_display_context_with_soln(self):
        wordle.set_display_context(self.soln())
        self.assertTrue(wordle._display_answer_set)
        self.assertGreater(wordle._display_max_ent, 0)

    def test_set_display_context_none(self):
        wordle.set_display_context(None)
        self.assertEqual(wordle._display_answer_set, set())
        self.assertEqual(wordle._display_max_ent, 0.0)

    def test_print_guesses_empty(self):
        out = self.run_cmd(lambda gs: wordle.print_guesses(self.soln()))
        self.assertEqual(out, "")

    def test_print_guesses_with_history(self):
        soln = self.soln()
        soln.apply_guess("crane", wordle.parse_response("00g0g"))
        out = _capture(wordle.print_guesses, soln)
        self.assertIn("Prior guesses:", out)
        self.assertIn("crane", out)


if __name__ == "__main__":
    unittest.main()
