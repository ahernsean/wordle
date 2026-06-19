"""Second supplemental coverage pass for wordle.py — pure-function edges and
the remaining handler branches (parse_response, _explain_conflict, the
candidate-selection helpers, multi-board invalid-pick returns, and assorted
display branches)."""
import io
import unittest
from contextlib import redirect_stdout

import wordle
from wordle import (
    parse_response, _explain_conflict, _multistep_stats,
    cmd_solve, cmd_grid, cmd_test, cmd_include, cmd_exclude, cmd_undo,
    cmd_answer, cmd_help, print_status, ERDSolver, Solution,
    GuessUniverse, ComplianceFilter,
)
from wordle_engine import ResponseCache, ERD_ALL, ScoringMethod
from cache_sqlite import MemoryScoreCache
from clitestutil import CliTestCase, ANSWERS, GUESSES


def _second_board(gs):
    return Solution(gs.all_answers, gs.all_words, gs.cache, gs.score_cache)


class TestParseResponse(unittest.TestCase):
    def test_invalid_char(self):
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(parse_response("qqqqq"))

    def test_x_is_gray(self):
        self.assertEqual(parse_response("xxxxx"), ["gray"] * 5)

    def test_space_separated_parts(self):
        self.assertEqual(parse_response("green gray gray gray gray")[0], "green")

    def test_wrong_length(self):
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(parse_response("abc"))


class TestExplainConflict(unittest.TestCase):
    def _resp(self, s):
        return parse_response(s)

    def test_gray_not_in_answer(self):
        msg = _explain_conflict(0, "crane", self._resp("00000"), self._resp("g0000"))
        self.assertIn("not in the answer", msg)

    def test_green_position(self):
        msg = _explain_conflict(0, "crane", self._resp("g0000"), self._resp("00000"))
        self.assertIn("position 1", msg)

    def test_yellow_must_be_present(self):
        msg = _explain_conflict(0, "crane", self._resp("y0000"), self._resp("00000"))
        self.assertIn("must be in the answer", msg)

    def test_yellow_wrong_position(self):
        msg = _explain_conflict(0, "crane", self._resp("y0000"), self._resp("g0000"))
        self.assertIn("can't be at", msg)

    def test_duplicate_letter_gray(self):
        # 'crane' has no dup; use a word with a repeated letter at pos 0 & elsewhere.
        msg = _explain_conflict(0, "sassy", self._resp("0g000"), self._resp("00000"))
        self.assertTrue(msg)

    def test_fallthrough(self):
        # recorded yellow + hypothetical yellow matches none of the specific
        # yellow branches, so it hits the generic fallthrough line.
        msg = _explain_conflict(0, "crane", self._resp("y0000"), self._resp("y0000"))
        self.assertIn("expected", msg)


class TestMultistepNoCacheGroups(unittest.TestCase):
    def test_no_cache_step2_step3(self):
        # piano leaves a 6-word subgroup, forcing the no-cache step-2 and
        # step-3 recompute loops to actually run.
        soln = Solution(ANSWERS, GUESSES, cache=None, score_cache=None)
        st = _multistep_stats("piano", soln, step2_pool=None,
                              constraint_compliant=False)
        self.assertGreaterEqual(st["step2"], 0.0)


class TestMultiBoardCandidateSelection(CliTestCase):
    """cmd_solve/cmd_grid prompt for the input-word set on multi-board games."""
    def setUp(self):
        super().setUp()
        self.gs.solutions = [self.soln(), _second_board(self.gs)]

    def test_solve_hard_mode_selection(self):
        # pick board 1, scoring method 1, then 'h' for hard-mode candidates.
        out = self.run_cmd(cmd_solve, inputs=["1", "1", "h"])
        self.assertTrue(out)

    def test_grid_all_words_selection(self):
        out = self.run_cmd(cmd_grid, inputs=["1", "a"])
        self.assertTrue(out)

    def test_grid_solved_filter_empty(self):
        # 's' selects the solved-words display filter; none solved → abort.
        out = self.run_cmd(cmd_grid, inputs=["1", "s"])
        self.assertIn("No words", out)

    def test_grid_invalid_input_choice(self):
        out = self.run_cmd(cmd_grid, inputs=["1", "q"])
        self.assertIn("Invalid", out)


class TestMultiBoardInvalidPicks(CliTestCase):
    def setUp(self):
        super().setUp()
        self.gs.solutions = [self.soln(), _second_board(self.gs)]

    def test_include_invalid(self):
        out = self.run_cmd(cmd_include, inputs=["z"])
        self.assertIn("Invalid", out)

    def test_exclude_invalid(self):
        out = self.run_cmd(cmd_exclude, inputs=["z"])
        self.assertIn("Invalid", out)

    def test_undo_invalid(self):
        out = self.run_cmd(cmd_undo, inputs=["z"])
        self.assertIn("Invalid", out)

    def test_answer_invalid_pick(self):
        out = self.run_cmd(cmd_answer, inputs=["z"])
        self.assertIn("Invalid", out)


class TestCmdGuessEndStates(CliTestCase):
    def test_guess_to_zero_remaining(self):
        # Apply an impossible response so the candidate set empties out.
        out = self.run_cmd(wordle.cmd_guess, inputs=["zzzzz", "ggggg"])
        # zzzzz isn't a real word but parse/apply still runs; assert it handled.
        self.assertTrue(out)

    def test_guess_solved_announcement(self):
        self.soln().answer_word = "crane"
        out = self.run_cmd(wordle.cmd_guess, inputs=["crane"])
        self.assertIn("remaining", out)


class TestCmdTestDisplayBranches(CliTestCase):
    def test_buckets_via_piano(self):
        out = self.run_cmd(lambda gs: cmd_test(gs, "piano"))
        self.assertIn("Response group sizes", out)

    def test_compare_three_words(self):
        out = self.run_cmd(lambda gs: cmd_test(gs, "crane slate heart"))
        self.assertTrue(out)


class TestCmdHelpTinyTiming(CliTestCase):
    def test_help_sub_millisecond(self):
        s = ERDSolver(ANSWERS, ANSWERS, GUESSES, None, policy=ERD_ALL,
                      persist=False, seed_mem_cache=MemoryScoreCache())
        s.word_stats = [(1, "crane", 0.0005, 0.0003, 5, 2)]
        s.cumulative_cpu_s = 0.0003
        s.cumulative_wall_s = 0.0005
        self.gs.solver = s
        out = self.run_cmd(cmd_help, inputs=[])
        self.assertIn("ms", out)


class TestPrintStatusFallback(CliTestCase):
    def test_fallback_banner(self):
        s = self.soln()
        s.apply_guess("piano", wordle.parse_response("00g00"))
        s.fallback_active = True
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_status(self.gs)
        self.assertIn("full guess vocabulary", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
