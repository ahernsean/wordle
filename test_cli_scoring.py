"""Tests for the cmd_solve, cmd_grid, and cmd_lookahead command handlers."""
import unittest

import wordle
from clitestutil import CliTestCase


def _second_solution(gs):
    """Build a fresh Solution sharing the GameState's word lists/caches."""
    return wordle.Solution(
        gs.all_answers, gs.all_words, gs.cache, gs.score_cache)


class TestCmdSolve(CliTestCase):
    # Menu (single board): 1-4 are scoring methods, 5 is ERD.

    def test_single_each_scoring_method(self):
        for i, m in enumerate(wordle.ScoringMethod):
            with self.subTest(method=m.name):
                out = self.run_cmd(wordle.cmd_solve, inputs=[str(i + 1)])
                self.assertIn(m.label, out)
                self.assertIn("Best guesses:", out)

    def test_cached_fast_path(self):
        # First run computes and caches; second run with same method hits the
        # "(cached)" fast path.
        self.run_cmd(wordle.cmd_solve, inputs=["2"])  # ENTROPY_GAIN
        out = self.run_cmd(wordle.cmd_solve, inputs=["2"])
        self.assertIn("(cached)", out)
        self.assertIn("Best guesses:", out)

    def test_invalid_choice_non_numeric(self):
        out = self.run_cmd(wordle.cmd_solve, inputs=["xyz"])
        self.assertIn("Invalid choice.", out)

    def test_invalid_choice_out_of_range(self):
        # 99 is above the valid scoring-method range and not the ERD index,
        # so methods[98] raises IndexError -> "Invalid choice."
        out = self.run_cmd(wordle.cmd_solve, inputs=["99"])
        self.assertIn("Invalid choice.", out)

    def test_erd_not_ready(self):
        # ERD index = number of methods + 1; the ERD tree is not cached, so
        # choosing it prints the "not ready" error.
        erd_idx = len(list(wordle.ScoringMethod)) + 1
        out = self.run_cmd(wordle.cmd_solve, inputs=[str(erd_idx)])
        self.assertIn("ERD tree not ready", out)

    def test_multi_board_single_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        # pick solution 1, scoring method 2, then candidate input set.
        out = self.run_cmd(wordle.cmd_solve, inputs=["1", "2", "h"])
        self.assertIn("Best guesses:", out)

    def test_multi_board_all_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        # 'a' -> Solution.join, scoring method 2, then candidate input set.
        out = self.run_cmd(wordle.cmd_solve, inputs=["a", "2", "h"])
        self.assertIn("Best guesses:", out)

    def test_multi_board_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_solve, inputs=["zzz"])
        self.assertIn("Invalid response.", out)

    def test_multi_board_invalid_input_set_aborts(self):
        # Valid pick + valid method, then a bad candidate-input choice makes
        # _resolve_candidate_wordlist return None and cmd_solve aborts.
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_solve, inputs=["1", "2", "zzz"])
        self.assertIn("Invalid choice.", out)
        self.assertNotIn("Best guesses:", out)

    def test_menu_progress_scanning(self):
        # Drive the alternate ERD progress branches in the menu print.
        self.gs.last_erd_root_progress = (3, 10, ("crane", 4.2))
        out = self.run_cmd(wordle.cmd_solve, inputs=["1"])
        self.assertIn("scanning root", out)
        self.assertIn("best so far", out)

    def test_menu_progress_no_best(self):
        self.gs.last_erd_root_progress = (3, 10, None)
        out = self.run_cmd(wordle.cmd_solve, inputs=["1"])
        self.assertIn("scanning root", out)

    def test_menu_progress_not_ready(self):
        # root_total < 0 -> the 'not ready' progress label.
        self.gs.last_erd_root_progress = (0, -1, None)
        out = self.run_cmd(wordle.cmd_solve, inputs=["1"])
        self.assertIn("not ready", out)

    def test_cached_entropy_shows_max_ent(self):
        # ENTROPY_GAIN cached fast path prints the max-entropy annotation.
        self.run_cmd(wordle.cmd_solve, inputs=["2"])
        out = self.run_cmd(wordle.cmd_solve, inputs=["2"])
        self.assertIn("max entropy", out)

    def test_cached_non_entropy_no_max_ent(self):
        # Cached fast path with a non-entropy method skips the max-ent line.
        self.run_cmd(wordle.cmd_solve, inputs=["1"])  # WEIGHTED_AVG
        out = self.run_cmd(wordle.cmd_solve, inputs=["1"])
        self.assertIn("(cached)", out)
        self.assertNotIn("max entropy", out)


class TestCmdGrid(CliTestCase):
    def test_single_happy_path(self):
        out = self.run_cmd(wordle.cmd_grid)
        self.assertIn("Entropy vs Max Group Size", out)

    def test_multi_board_all_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        # 'a' -> join, then 'h' for hard-mode input set.
        out = self.run_cmd(wordle.cmd_grid, inputs=["a", "h"])
        self.assertIn("Entropy vs Max Group Size", out)

    def test_multi_board_single_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_grid, inputs=["1", "h"])
        self.assertIn("Entropy vs Max Group Size", out)

    def test_multi_board_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_grid, inputs=["zzz"])
        self.assertIn("Invalid response.", out)

    def test_multi_board_invalid_input_set(self):
        # Valid pick, then an invalid candidate-input choice aborts.
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_grid, inputs=["1", "zzz"])
        self.assertIn("Invalid choice.", out)


class TestComputePareto(unittest.TestCase):
    def test_frontier(self):
        data = [
            ("aaaaa", 1.0, 5),
            ("bbbbb", 2.0, 5),   # better entropy at mx 5
            ("ccccc", 3.0, 3),   # smaller mx, higher entropy -> dominates
            ("ddddd", 0.5, 1),   # dominated by ccccc (lower entropy, but smaller mx)
        ]
        frontier = wordle._compute_pareto(data)
        # mx levels whose best entropy beats all smaller-mx levels.
        self.assertIsInstance(frontier, set)
        self.assertIn(3, frontier)


class TestCmdLookahead(CliTestCase):
    def test_single_happy_path_default_count(self):
        # Empty input uses the default LOOKAHEAD_N.
        out = self.run_cmd(wordle.cmd_lookahead, inputs=[""])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_single_explicit_count(self):
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["3"])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_invalid_count_non_numeric(self):
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["abc"])
        self.assertIn("Invalid number.", out)

    def test_invalid_count_negative(self):
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["0"])
        self.assertIn("Invalid number.", out)

    def test_two_or_fewer_words(self):
        # Narrow the solution down to <=2 words; lookahead short-circuits.
        self.soln().apply_guess("crane", wordle.parse_response("ggggg"))
        out = self.run_cmd(wordle.cmd_lookahead, inputs=[""])
        self.assertIn("Two or fewer words remain", out)

    def test_cached_entropy_reused(self):
        # Pre-compute entropy so lookahead skips the "Computing entropy" step.
        self.run_cmd(wordle.cmd_solve, inputs=["2"])  # ENTROPY_GAIN
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["3"])
        self.assertNotIn("Computing entropy ranking", out)
        self.assertIn("Two-step entropy lookahead:", out)

    def test_possible_answers_mode(self):
        # (ALL_ANSWERS, COMPLIANT) drives the possible-answers branch where
        # step-2 candidates are the live subgroup.
        self.gs.universe = wordle.GuessUniverse.ALL_ANSWERS
        self.gs.compliance = wordle.ComplianceFilter.COMPLIANT
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["3"])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_rerun_uses_score_cache(self):
        # A second lookahead reuses score_cache entries written by the first,
        # exercising the cache-hit skip in the work-counting loop.
        self.run_cmd(wordle.cmd_lookahead, inputs=["3"])
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["3"])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_multi_board_all_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["a", "3"])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_multi_board_single_selection(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["1", "3"])
        self.assertIn("Two-step entropy lookahead:", out)

    def test_multi_board_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_solution(self.gs)]
        out = self.run_cmd(wordle.cmd_lookahead, inputs=["zzz"])
        self.assertIn("Invalid response.", out)


if __name__ == "__main__":
    unittest.main()
