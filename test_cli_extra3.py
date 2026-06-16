"""Third supplemental pass: display-bucket branches (needing larger group
sizes than the default 10-word fixture provides), plus a few remaining
multi-board and precache branches."""
import io
import itertools
import unittest
from contextlib import redirect_stdout

import wordle
from wordle import (
    cmd_test, cmd_display, cmd_reset, cmd_precache, cmd_lookahead, Solution,
    GuessUniverse, ComplianceFilter,
)
from clitestutil import CliTestCase


def _second_board(gs):
    return Solution(gs.all_answers, gs.all_words, gs.cache, gs.score_cache)


# 60 distinct 5-letter words sharing no 'z', so a 'zzzzz' guess collapses them
# all into a single all-gray group (size 60 → the "50+" bucket).
_BIG_ANSWERS = ["mn" + "".join(t)
                for t in itertools.product("abcde", repeat=3)][:60]


class TestBucketBranchesDefault(CliTestCase):
    def test_ten_word_group_bucket(self):
        # 'zzzzz' vs the 10-word fixture → one all-gray group of 10 (10-49 bucket).
        out = self.run_cmd(lambda gs: cmd_test(gs, "zzzzz"))
        self.assertIn("Subgroup sizes", out)


class TestBucketBranchesLarge(CliTestCase):
    ANSWERS = _BIG_ANSWERS
    GUESSES = _BIG_ANSWERS

    def test_fifty_plus_group_bucket(self):
        out = self.run_cmd(lambda gs: cmd_test(gs, "zzzzz"))
        self.assertIn("Subgroup sizes", out)


class TestRemainingMultiBoard(CliTestCase):
    def setUp(self):
        super().setUp()
        self.gs.solutions = [self.soln(), _second_board(self.gs)]

    def test_display_invalid_pick(self):
        out = self.run_cmd(cmd_display, inputs=["z"])
        self.assertIn("Invalid", out)

    def test_reset_invalid_pick(self):
        out = self.run_cmd(cmd_reset, inputs=["z"])
        self.assertIn("Invalid", out)


class TestPrecacheNothing(CliTestCase):
    def test_nothing_to_precache(self):
        # 'slate' fully discriminates the 10-word fixture → no 2+ branch.
        out = self.run_cmd(cmd_precache, inputs=["slate"])
        self.assertIn("Nothing to precache", out)


class TestLookaheadPossibleAnswers(CliTestCase):
    def test_possible_answers_mode(self):
        self.gs.universe = GuessUniverse.ALL_ANSWERS
        self.gs.compliance = ComplianceFilter.COMPLIANT
        out = self.run_cmd(cmd_lookahead, inputs=[""])
        self.assertIn("lookahead", out.lower())


class TestGuessAndVerifyEdges(CliTestCase):
    def test_verify_multiboard_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_board(self.gs)]
        out = self.run_cmd(wordle.cmd_verify_erd, inputs=["z"])
        self.assertIn("Invalid", out)

    def test_guess_invalid_response_continues(self):
        # An unparseable response (bad char, not 'stop') is rejected and the
        # loop continues without applying anything.
        out = self.run_cmd(wordle.cmd_guess, inputs=["crane", "qqqqq"])
        self.assertNotIn("words remaining", out)

    def test_guess_when_no_words_remain(self):
        s = self.soln()
        s.apply_guess("zzzzz", wordle.parse_response("ggggg"))  # matches nothing
        self.assertEqual(len(s.current_words), 0)
        out = self.run_cmd(wordle.cmd_guess, inputs=["crane"])
        self.assertIn("No remaining words", out)

    def test_test_command_reraises_internal_error(self):
        from unittest import mock
        with mock.patch("wordle.score_groups", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.run_cmd(lambda gs: cmd_test(gs, "crane"))


if __name__ == "__main__":
    unittest.main()
