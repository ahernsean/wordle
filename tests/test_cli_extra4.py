"""Final margin tests: a guess that leaves several candidates, possible-answers
candidate resolution, and the MemoryScoreCache read-miss counter."""
import unittest

import wordle
from wordle import cmd_solve, cmd_guess, GuessUniverse, ComplianceFilter
from wordle_engine import ERD_ALL
from cache_sqlite import MemoryScoreCache
from clitestutil import CliTestCase


class TestGuessLeavesMany(CliTestCase):
    def test_simulated_guess_keeps_multiple(self):
        s = self.soln()
        s.answer_word = "slate"
        out = self.run_cmd(cmd_guess, inputs=["piano"])  # 00g00 → 6 remain
        self.assertIn("words remaining", out)
        self.assertGreater(len(s.current_words), 1)


class TestSolvePossibleAnswers(CliTestCase):
    def test_possible_answers_candidate_list(self):
        self.gs.universe = GuessUniverse.ALL_ANSWERS
        self.gs.compliance = ComplianceFilter.COMPLIANT
        out = self.run_cmd(cmd_solve, inputs=["1"])  # scoring method 1
        self.assertTrue(out)


class TestMemoryCacheMiss(unittest.TestCase):
    def test_read_miss_counts(self):
        mc = MemoryScoreCache()
        mc.set_scope("scope")
        before = mc.read_misses
        self.assertIsNone(mc.read("nope0", ERD_ALL))
        self.assertEqual(mc.read_misses, before + 1)

    def test_read_with_depth_hit(self):
        mc = MemoryScoreCache()
        mc.set_scope("scope")
        mc.write("abcde", ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        row = mc.read_with_depth("abcde", ERD_ALL)
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "crane")


if __name__ == "__main__":
    unittest.main()
