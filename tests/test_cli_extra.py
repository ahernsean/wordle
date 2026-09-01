"""Supplemental coverage for wordle.py: handler branches and helpers not
reached by the other CLI test modules — multi-board paths, the ERD-ready
paths (with a cache built on the fly), status/help display branches, and a
handful of pure-function edge cases.
"""
import io
import unittest
from contextlib import redirect_stdout

import wordle
from wordle import (
    ProgressTracker, _erd_solve_scores,
    _solver_branch_key, _multistep_stats, cmd_guess, cmd_solve, cmd_display,
    cmd_test, cmd_include, cmd_exclude, cmd_reset, cmd_undo, cmd_answer,
    cmd_verify_erd, cmd_help, print_status, ERDSolver,
    GuessUniverse, ComplianceFilter, Solution,
)
from wordle_engine import min_expected_guesses, ERD_ALL, ScoringMethod
from cache_sqlite import ScoreCache, MemoryScoreCache
from clitestutil import CliTestCase, ANSWERS, GUESSES


def _second_board(gs):
    return Solution(gs.all_answers, gs.all_words, gs.cache, gs.score_cache)


class TestProgressTrackerEdges(unittest.TestCase):
    def test_wrap_and_eta(self):
        import datetime as _dt
        with mock_width(24):
            buf = io.StringIO()
            with redirect_stdout(buf):
                t = ProgressTracker(50)
                # Force the ETA branch: pretend the last ETA was long ago and
                # the run started a while back so a remaining estimate exists.
                t.start_time = _dt.datetime.now() - _dt.timedelta(seconds=20)
                t.last_eta = _dt.datetime.now() - _dt.timedelta(seconds=60)
                for _ in range(50):
                    t.update()
                t.finish()
            self.assertIn("100%", buf.getvalue())


class _MockWidth:
    def __init__(self, w):
        self.w = w
    def __enter__(self):
        self._orig = wordle._width_cache[0]
        wordle._width_cache[0] = self.w
        return self
    def __exit__(self, *a):
        wordle._width_cache[0] = self._orig


def mock_width(w):
    return _MockWidth(w)


class TestERDHelpers(unittest.TestCase):
    def setUp(self):
        self.sc = MemoryScoreCache()
        self.sc.set_scope("s")
        from wordle_engine import ResponseCache
        cache = ResponseCache(ANSWERS, None)
        self.soln = Solution(ANSWERS, GUESSES, cache=cache, score_cache=None)

    def test_erd_solve_scores_returns_none_when_uncached(self):
        # No subgroup cached → every candidate skipped → None.
        self.assertIsNone(
            _erd_solve_scores(self.soln, self.sc, ERD_ALL, guesses=["crane"]))

    def test_solver_branch_key_hard_mode_fingerprints(self):
        from wordle import GameState  # avoid import cycle at top
        gs = _FakeGS(self.soln)
        gs.universe = GuessUniverse.ALL_WORDS
        gs.compliance = ComplianceFilter.COMPLIANT  # hard mode
        key = _solver_branch_key(gs, self.soln, GUESSES)
        self.assertIsNotNone(key[2])  # vocabulary fingerprint present


class _FakeGS:
    def __init__(self, soln):
        self.solutions = [soln]
        self.universe = GuessUniverse.ALL_WORDS
        self.compliance = ComplianceFilter.UNFILTERED
    @property
    def single(self):
        return len(self.solutions) == 1


class TestMultistepNoCache(unittest.TestCase):
    def test_no_response_cache_paths(self):
        soln = Solution(ANSWERS, GUESSES, cache=None, score_cache=None)
        st = _multistep_stats("brain", soln, step2_pool=None,
                              constraint_compliant=False)
        self.assertIn("step1", st)
        self.assertIn("step2", st)
        self.assertIn("step3", st)


class TestCmdGuessMultiBoard(CliTestCase):
    def setUp(self):
        super().setUp()
        self.gs.solutions = [self.soln(), _second_board(self.gs)]
        self.gs.columns = 1

    def test_guess_all_boards_manual(self):
        out = self.run_cmd(cmd_guess, inputs=["a", "crane", "gy0y0", "gy0y0"])
        self.assertIn("Solution", out)

    def test_guess_one_board(self):
        out = self.run_cmd(cmd_guess, inputs=["1", "crane", "00000"])
        self.assertIn("words remaining", out)

    def test_guess_invalid_pick(self):
        out = self.run_cmd(cmd_guess, inputs=["x"])
        self.assertIn("Invalid", out)


class TestCmdSolveERDReady(CliTestCase):
    def _build_erd_root(self):
        s = self.soln()
        s.apply_guess("piano", wordle.parse_response("00g00"))
        from wordle_engine import ResponseCache
        rc = ResponseCache(self.gs.all_answers, self.gs.score_cache)
        min_expected_guesses(
            s.current_words, rc, self.gs.score_cache,
            guesses=self.gs.all_words, policy=ERD_ALL)
        return s

    def test_erd_option_ranks_when_ready(self):
        self._build_erd_root()
        # Menu: methods then ERD option (last). Choose the ERD index.
        n_methods = len(list(ScoringMethod))
        out = self.run_cmd(cmd_solve, inputs=[str(n_methods + 1)])
        self.assertIn("ERD", out)

    def test_verify_erd_mid_game(self):
        self._build_erd_root()
        out = self.run_cmd(cmd_verify_erd, inputs=[])
        self.assertIn("ERD cache check", out)


class TestBudgetScopedERDResultVisibleAtOneGuess(CliTestCase):
    """Issue #311: a budget-specific exact ERD result (imported, or written
    by an earlier search under a remaining-depth cap) must be visible from
    the status line and the ERD menu even when no *unrestricted* result is
    cached for the branch — both must read through read_for_budget at the
    branch's own budget, not the unbudgeted read().
    """

    def _one_guess_branch(self):
        s = self.soln()
        s.apply_guess("piano", wordle.parse_response("00g00"))
        return s

    def _write_budget_specific_root(self, soln, guess, cost, max_depth):
        branch_key = ScoreCache.encode_subset(soln.current_words)
        budget = wordle.GAME_GUESSES - len(soln.guesses)
        self.gs.score_cache.write(branch_key, ERD_ALL, guess, cost,
                                   max_depth=max_depth, solve_budget=budget)
        return branch_key, budget

    def test_no_unrestricted_result_is_cached(self):
        """Sanity check on the fixture: the plain unrestricted read must
        miss, so the assertions below are exercising the budget-aware path
        and not an unrestricted hit that would pass either way."""
        s = self._one_guess_branch()
        branch_key, _ = self._write_budget_specific_root(
            s, "tarse", 3.036, max_depth=5)
        self.assertIsNone(self.gs.score_cache.read(branch_key, ERD_ALL))

    def test_status_line_shows_budget_specific_result(self):
        s = self._one_guess_branch()
        self._write_budget_specific_root(s, "tarse", 3.036, max_depth=5)
        out = io.StringIO()
        with redirect_stdout(out):
            print_status(self.gs)
        self.assertIn("3.036 TARSE", out.getvalue())

    def test_erd_menu_available_with_budget_specific_result(self):
        s = self._one_guess_branch()
        self._write_budget_specific_root(s, "tarse", 3.036, max_depth=5)
        n_methods = len(list(ScoringMethod))
        out = self.run_cmd(cmd_solve, inputs=[str(n_methods + 1)])
        self.assertNotIn("ERD tree not ready", out)
        self.assertIn("3.036", out)


class TestCmdDisplayScores(CliTestCase):
    def test_display_with_scores(self):
        s = self.soln()
        s.compute_scores(self.gs.all_answers, ScoringMethod.ENTROPY_GAIN)
        out = self.run_cmd(cmd_display, inputs=[])
        self.assertIn("words remaining", out)


class TestCmdTestBranches(CliTestCase):
    def test_in_answer_set_and_buckets(self):
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("CRANE", out)

    def test_conflict_path(self):
        s = self.soln()
        s.apply_guess("crane", wordle.parse_response("ggggg"))
        # 'slate' conflicts with an all-green 'crane'.
        out = self.run_cmd(lambda gs: cmd_test(gs, "slate"))
        self.assertIn("Conflicts", out)

    def test_compare_two_words(self):
        out = self.run_cmd(lambda gs: cmd_test(gs, "crane slate"))
        self.assertTrue(out)

    def test_bad_length(self):
        out = self.run_cmd(cmd_test, inputs=["abc"])
        self.assertIn("5 letters", out)

    def test_multiboard_invalid_pick(self):
        self.gs.solutions = [self.soln(), _second_board(self.gs)]
        out = self.run_cmd(cmd_test, inputs=["z"])
        self.assertIn("Invalid", out)


class TestSimpleHandlersMultiBoard(CliTestCase):
    def setUp(self):
        super().setUp()
        self.gs.solutions = [self.soln(), _second_board(self.gs)]

    def test_include_pick(self):
        out = self.run_cmd(cmd_include, inputs=["1", "ae"])
        self.assertEqual(out.count("Invalid"), 0)

    def test_exclude_pick(self):
        self.run_cmd(cmd_exclude, inputs=["2", "z"])

    def test_reset_all(self):
        out = self.run_cmd(cmd_reset, inputs=["a"])
        self.assertIn("reset", out.lower())

    def test_undo_pick_nothing(self):
        out = self.run_cmd(cmd_undo, inputs=["1"])
        self.assertIn("Nothing to undo", out)

    def test_answer_all(self):
        out = self.run_cmd(cmd_answer, inputs=["a", "crane", "slate"])
        self.assertTrue(out)


class TestCmdHelpWordStats(CliTestCase):
    def _solver_with_stats(self, with_cpu=True):
        s = ERDSolver(ANSWERS, ANSWERS, GUESSES, None,
                      policy=ERD_ALL, persist=False,
                      seed_mem_cache=MemoryScoreCache())
        cpu = 0.4 if with_cpu else None
        s.word_stats = [(1, "crane", 0.5, cpu, 100, 50),
                        (2, "slate", 12.0, (8.0 if with_cpu else None), 9, 3)]
        s.cumulative_cpu_s = 8.4 if with_cpu else 0.0
        s.cumulative_wall_s = 12.5
        return s

    def test_help_with_cpu_stats(self):
        self.gs.solver = self._solver_with_stats(with_cpu=True)
        out = self.run_cmd(cmd_help, inputs=[])
        self.assertIn("ERD root scan", out)

    def test_help_without_cpu_stats(self):
        self.gs.solver = self._solver_with_stats(with_cpu=False)
        out = self.run_cmd(cmd_help, inputs=[])
        self.assertIn("ERD root scan", out)


class TestPrintStatusBranches(CliTestCase):
    def _out(self, gs=None, solver=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_status(gs or self.gs, solver)
        return buf.getvalue()

    def test_sim_banner(self):
        self.soln().answer_word = "crane"
        self.assertIn("Sim:", self._out())

    def test_solved(self):
        self.soln().apply_guess("crane", wordle.parse_response("ggggg"))
        self.assertIn("Solved", self._out())

    def test_midgame_erd_hit_and_scores(self):
        s = self.soln()
        s.apply_guess("piano", wordle.parse_response("00g00"))
        from wordle_engine import ResponseCache
        rc = ResponseCache(self.gs.all_answers, self.gs.score_cache)
        min_expected_guesses(s.current_words, rc, self.gs.score_cache,
                             guesses=self.gs.all_words, policy=ERD_ALL)
        s.compute_scores(s.current_words, ScoringMethod.ENTROPY_GAIN)
        out = self._out()
        self.assertIn("words left", out)

    def test_multiboard(self):
        self.gs.solutions = [self.soln(), _second_board(self.gs)]
        self.gs.solutions[1].apply_guess("crane", wordle.parse_response("ggggg"))
        self.assertIn("wordlists", self._out())


if __name__ == "__main__":
    unittest.main()
