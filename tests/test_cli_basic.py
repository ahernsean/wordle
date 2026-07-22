"""Tests for the simpler wordle.py command handlers (single- and multi-board).

Covers: cmd_guess, cmd_include, cmd_exclude, cmd_reset, cmd_undo, cmd_answer,
cmd_wordcount, cmd_candidates, cmd_help, cmd_display, and the pick_one/
pick_one_or_all branch_targets they share.
"""
import unittest

import wordle
from wordle import (
    cmd_guess, cmd_include, cmd_exclude, cmd_reset, cmd_undo, cmd_answer,
    cmd_wordcount, cmd_candidates, cmd_help, cmd_display,
    GuessUniverse, ComplianceFilter,
)
from clitestutil import CliTestCase


class TestPickers(CliTestCase):
    def test_pick_one_autoselects_single(self):
        idx, soln = wordle.pick_one(self.gs)
        self.assertEqual(idx, 0)
        self.assertIs(soln, self.soln())

    def test_pick_one_invalid_number(self):
        self.gs.solutions = [self.soln(), self.soln()]  # force multi-board
        out = self.run_cmd(lambda gs: self.assertIsNone(wordle.pick_one(gs)),
                           inputs=["nope"])
        self.assertIn("Invalid", out)

    def test_pick_one_or_all_all(self):
        self.gs.solutions = [self.soln(), self.soln()]
        out = self.run_cmd(
            lambda gs: self.assertEqual(wordle.pick_one_or_all(gs)[0], 'all'),
            inputs=["a"])
        self.assertEqual(out.count("Which"), 1)

    def test_pick_one_or_all_invalid(self):
        self.gs.solutions = [self.soln(), self.soln()]
        out = self.run_cmd(
            lambda gs: self.assertIsNone(wordle.pick_one_or_all(gs)),
            inputs=["zz"])
        self.assertIn("Invalid", out)


class TestCmdGuess(CliTestCase):
    def test_guess_with_manual_response(self):
        out = self.run_cmd(cmd_guess, inputs=["crane", "gy0y0"])
        self.assertIn("words remaining", out)

    def test_guess_rejects_wrong_length(self):
        out = self.run_cmd(cmd_guess, inputs=["cat"])
        self.assertIn("5 letters", out)

    def test_guess_with_simulated_answer(self):
        self.soln().answer_word = "crane"
        out = self.run_cmd(cmd_guess, inputs=["slate"])
        # No response prompt is consumed because answer_word drives it.
        self.assertIn("words remaining", out)

    def test_guess_stop_breaks(self):
        out = self.run_cmd(cmd_guess, inputs=["crane", "stop"])
        self.assertIn("Stopped", out)

    def test_guess_already_solved(self):
        s = self.soln()
        s.apply_guess("crane", wordle.parse_response("ggggg"))
        out = self.run_cmd(cmd_guess, inputs=["slate"])
        self.assertIn("Already solved", out)


class TestIncludeExclude(CliTestCase):
    def test_include_letters(self):
        self.run_cmd(cmd_include, inputs=["ae"])
        for w in self.soln().current_words:
            self.assertTrue('a' in w and 'e' in w)

    def test_exclude_letters(self):
        self.run_cmd(cmd_exclude, inputs=["z"])
        # Excluding an absent letter leaves the list unchanged but exercises path.
        self.assertTrue(self.soln().current_words)


class TestReset(CliTestCase):
    def test_reset_confirm(self):
        self.soln().apply_guess("crane", wordle.parse_response("00000"))
        out = self.run_cmd(cmd_reset, inputs=["y"])
        self.assertIn("Reset.", out)
        self.assertEqual(len(self.soln().guesses), 0)

    def test_reset_cancel(self):
        out = self.run_cmd(cmd_reset, inputs=["n"])
        self.assertIn("Cancelled", out)

    def test_reset_multiboard_single(self):
        self.gs.solutions = [self.soln(), wordle.Solution(
            self.gs.all_answers, self.gs.all_words, self.gs.cache,
            self.gs.score_cache)]
        out = self.run_cmd(cmd_reset, inputs=["2"])
        self.assertIn("reset", out.lower())


class TestUndo(CliTestCase):
    def test_undo_nothing(self):
        out = self.run_cmd(cmd_undo, inputs=[])
        self.assertIn("Nothing to undo", out)

    def test_undo_one(self):
        self.soln().apply_guess("crane", wordle.parse_response("00000"))
        out = self.run_cmd(cmd_undo, inputs=[])
        self.assertIn("Undid", out)


class TestAnswer(CliTestCase):
    def test_set_answer(self):
        out = self.run_cmd(cmd_answer, inputs=["crane"])
        self.assertEqual(self.soln().answer_word, "crane")
        self.assertIn("Simulation on", out)

    def test_set_answer_bad_length(self):
        out = self.run_cmd(cmd_answer, inputs=["ab"])
        self.assertIn("5 letters", out)

    def test_clear_answer(self):
        self.soln().answer_word = "crane"
        out = self.run_cmd(cmd_answer, inputs=["y"])
        self.assertIsNone(self.soln().answer_word)
        self.assertIn("Simulation off", out)

    def test_replace_answer(self):
        self.soln().answer_word = "crane"
        out = self.run_cmd(cmd_answer, inputs=["n", "slate"])
        self.assertEqual(self.soln().answer_word, "slate")


class TestWordcount(CliTestCase):
    def test_single(self):
        out = self.run_cmd(cmd_wordcount, inputs=["1"])
        self.assertIn("Set up 1 game", out)
        self.assertEqual(len(self.gs.solutions), 1)

    def test_multi(self):
        out = self.run_cmd(cmd_wordcount, inputs=["4", "2"])
        self.assertEqual(len(self.gs.solutions), 4)
        self.assertEqual(self.gs.columns, 2)

    def test_invalid(self):
        out = self.run_cmd(cmd_wordcount, inputs=["0"])
        self.assertIn("Invalid", out)


class TestCandidates(CliTestCase):
    def test_set_answers_compliant(self):
        out = self.run_cmd(cmd_candidates, inputs=["2", "2"])
        self.assertIs(self.gs.universe, GuessUniverse.ALL_ANSWERS)
        self.assertIs(self.gs.compliance, ComplianceFilter.COMPLIANT)

    def test_bad_universe(self):
        out = self.run_cmd(cmd_candidates, inputs=["9"])
        self.assertIn("Invalid", out)

    def test_bad_compliance(self):
        out = self.run_cmd(cmd_candidates, inputs=["1", "9"])
        self.assertIn("Invalid", out)


class TestDisplay(CliTestCase):
    def test_display_words(self):
        out = self.run_cmd(cmd_display, inputs=[])
        self.assertIn("words remaining", out)


class TestHelp(CliTestCase):
    def test_help_single(self):
        out = self.run_cmd(cmd_help, inputs=[])
        self.assertIn("Guess a word", out)
        self.assertIn("Cache:", out)

    def test_help_multiboard(self):
        self.gs.solutions = [self.soln(), self.soln()]
        out = self.run_cmd(cmd_help, inputs=[])
        self.assertIn("set", out)


if __name__ == "__main__":
    unittest.main()
