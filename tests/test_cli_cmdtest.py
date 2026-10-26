"""Tests for the "Test a word" command (cmd_test) and its display helpers.

Covers cmd_test, _compare_words, _multistep_stats, and _explain_conflict in
wordle.py.  Drives the helpers through cmd_test (interactive and inline paths)
across the candidate modes (possible-answers, hard mode, unfiltered), fresh and
mid-game, plus comparison of 2-5 words and conflict explanations.
"""
import contextlib
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import wordle
from wordle import cmd_test, GuessUniverse, ComplianceFilter
from clitestutil import CliTestCase


def _set_mode(gs, universe, compliance):
    gs.universe = universe
    gs.compliance = compliance


class TestCmdTestSingle(CliTestCase):
    def test_single_word_fresh(self):
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("CRANE", out)
        self.assertIn("words", out)
        # Score lines and group/subgroup display are produced.
        self.assertIn("Groups:", out)
        self.assertIn("Response group sizes:", out)

    def test_single_word_multistep_lookahead(self):
        # n = 10 (> 2) so the multi-step lookahead block runs.
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("Multi-step lookahead", out)
        self.assertIn("Entropy 1:", out)
        self.assertIn("Total:", out)

    def test_in_answer_set_marker(self):
        # "crane" is in ANSWERS, so the "(in answer set)" label appears.
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("in answer set", out)

    def test_bad_length_word(self):
        out = self.run_cmd(cmd_test, inputs=["cat"])
        self.assertIn("5 letters", out)

    def test_bad_length_word_via_compare_branch(self):
        out = self.run_cmd(cmd_test, inputs=["crane sla"])
        self.assertIn("Not 5 letters: SLA.", out)
        self.assertNotIn("Computing", out)

    def test_every_wrong_length_word_is_named(self):
        out = self.run_cmd(cmd_test, inputs=["cat crane brains"])
        self.assertIn("Not 5 letters: CAT, BRAINS.", out)

    def test_too_many_words_names_the_limit_not_the_length(self):
        out = self.run_cmd(
            cmd_test, inputs=["crane slate trace stale tales least"])
        self.assertIn(
            f"Test takes 1 to {wordle.MAX_COMPARE_WORDS} words, not 6.", out)
        self.assertNotIn("5 letters", out)
        self.assertNotIn("Computing", out)

    def test_no_words(self):
        out = self.run_cmd(cmd_test, inputs=[""])
        self.assertIn("Test takes 1 to", out)
        self.assertNotIn("5 letters", out)

    def test_solver_assertion_is_not_reported_as_bad_input(self):
        def failing(*_args, **_kwargs):
            raise AssertionError("engine invariant")

        for line, target in (("crane", "_multistep_stats"),
                             ("crane slate", "_compare_words")):
            with self.subTest(line=line), \
                    mock.patch.object(wordle, target, failing):
                buf = io.StringIO()
                with redirect_stdout(buf), \
                        self.assertRaisesRegex(AssertionError,
                                               "engine invariant"):
                    wordle.cmd_test(self.gs, line)
                self.assertNotIn("5 letters", buf.getvalue())


class TestCmdTestInline(CliTestCase):
    def test_inline_word(self):
        # main() passes the word inline; no input() is consumed.
        out = self.run_cmd(lambda gs: wordle.cmd_test(gs, "crane"))
        self.assertIn("CRANE", out)
        self.assertIn("Multi-step lookahead", out)

    def test_inline_compare(self):
        out = self.run_cmd(lambda gs: wordle.cmd_test(gs, "crane slate"))
        self.assertIn("CRANE", out)
        self.assertIn("SLATE", out)
        self.assertIn("words:", out)


class TestCmdTestCompare(CliTestCase):
    def test_compare_two(self):
        out = self.run_cmd(cmd_test, inputs=["crane slate"])
        self.assertIn("Computing", out)
        self.assertIn("Ent. 1", out)
        self.assertIn("Wt avg", out)
        self.assertIn("max-grp", out)
        self.assertIn("Solve%", out)
        # n == 2 path: the "+ ent. 2/3"/"Tot. ent" rows are gated on n > 2,
        # which holds here (n = 10), so they should appear.
        self.assertIn("Tot. ent", out)
        self.assertIn("+ ent. 2", out)

    def test_compare_three(self):
        out = self.run_cmd(cmd_test, inputs=["crane slate trace"])
        self.assertIn("[1/3]", out)
        self.assertIn("[3/3]", out)
        self.assertIn("Tot. ent", out)

    def test_compare_four(self):
        out = self.run_cmd(cmd_test, inputs=["crane slate trace stale"])
        self.assertIn("[4/4]", out)
        self.assertIn("TRACE", out)

    def test_compare_five(self):
        out = self.run_cmd(
            cmd_test, inputs=["crane slate trace stale tales"])
        self.assertIn("[5/5]", out)
        self.assertIn("TALES", out)

    def test_five_widest_columns_fit_the_phone_console(self):
        # The widest value each row can print over the full 3,209-word
        # answer list: every word in one response group, maximal entropy.
        widest = {'wt_avg': 3209.0, 'max_group_size': 3209,
                  'prob_finish': 1.0, 'step1': 9.999, 'step2': 9.999,
                  'step3': 9.999, 'buckets': [3209] * 5, 'erd': 9.999}
        line = " ".join(["crane", "slate", "trace", "stale", "tales"])
        with mock.patch.object(wordle, '_multistep_stats',
                               lambda *_a, **_k: dict(widest)), \
                mock.patch.object(wordle, 'SUPPORTS_COLOR', False):
            out = self.run_cmd(lambda gs: wordle.cmd_test(gs, line))
        self.assertIn("Tot. ent", out)
        self.assertIn("ERD", out)
        table = out.split("words:", 1)[1]
        self.assertEqual(max(len(row) for row in table.splitlines()), 49)

    def test_green_follows_full_precision_not_the_printed_digits(self):
        # 1.50004 and 1.50001 both print 1.500; only the larger is best.
        stats = iter([{'wt_avg': 1.0, 'max_group_size': 1,
                       'prob_finish': 0.5, 'step1': step1,
                       'step2': 0.0, 'step3': 0.0,
                       'buckets': [1, 0, 0, 0, 0]}
                      for step1 in (1.50001, 1.50004)])
        greens = []

        def record(color):
            greens.append(color)
            return contextlib.nullcontext()

        with mock.patch.object(wordle, '_multistep_stats',
                               lambda *_a, **_k: next(stats)), \
                mock.patch.object(wordle, '_erd_cache_and_policy',
                                  return_value=(None, None)), \
                mock.patch.object(wordle, 'colored_text', record), \
                mock.patch.object(wordle, 'SUPPORTS_COLOR', False):
            out = self.run_cmd(lambda gs: wordle.cmd_test(gs, "crane slate"))
        # Ent. 1 and Tot. ent, which repeats it: one green cell in each.
        self.assertEqual(len(greens), 2)
        ent_row = next(l for l in out.splitlines() if l.startswith("  Ent. 1"))
        self.assertEqual(ent_row.split()[2:], ["1.500", "1.500"])

    def test_header_words_end_at_the_column_edge(self):
        # A word's last character, its letter or its answer-set marker,
        # sits where the values beneath it end.
        out = self.run_cmd(lambda gs: wordle.cmd_test(gs, "brain crane"))
        lines = out.split("words:", 1)[1].splitlines()
        header = next(l for l in lines if "CRANE" in l)
        ent_row = next(l for l in lines if l.startswith("  Ent. 1"))
        self.assertTrue(header.endswith("CRANE*"))
        self.assertEqual(len(header), len(ent_row))
        brain_end = header.index("BRAIN") + len("BRAIN")
        self.assertNotEqual(ent_row[brain_end - 1], " ")
        self.assertEqual(ent_row[brain_end], " ")

    def test_compare_buckets(self):
        # Bucket rows (1:, 2-4:, etc.) print when any bucket is non-empty.
        out = self.run_cmd(cmd_test, inputs=["crane slate"])
        self.assertTrue(any(lbl in out for lbl in ("1:", "2-4:", "5-9:")))

    def test_compare_columns_stay_aligned(self):
        """Every data row is built from the same fixed-width label field
        plus values right-justified to a shared column width, so every
        row's printed line comes out the same total length. A label
        longer than the reserved width (lw) isn't truncated — it just
        widens that one row's prefix, throwing its line length, and
        therefore its columns, out of step with every other row. This is
        the exact defect a too-long "Max group size" label caused."""
        out = self.run_cmd(cmd_test, inputs=["crane slate"])
        labels = ["Wt avg", "max-grp", "Solve%", "Ent. 1", "Tot. ent"]
        lengths = {}
        for line in out.splitlines():
            for label in labels:
                if line.startswith('  ' + label):
                    lengths[label] = len(line)
                    break
        self.assertEqual(set(lengths), set(labels))
        self.assertEqual(len(set(lengths.values())), 1,
                         f"columns misaligned, row lengths: {lengths}")


class TestCmdTestModes(CliTestCase):
    def test_possible_answers_mode(self):
        _set_mode(self.gs, GuessUniverse.ALL_ANSWERS, ComplianceFilter.COMPLIANT)
        self.assertTrue(wordle._is_possible_answers_mode(self.gs))
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("possible answers", out)

    def test_hard_mode(self):
        _set_mode(self.gs, GuessUniverse.ALL_WORDS, ComplianceFilter.COMPLIANT)
        self.assertTrue(wordle._is_hard_mode(self.gs))
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("hard mode", out)

    def test_hard_mode_compare(self):
        _set_mode(self.gs, GuessUniverse.ALL_WORDS, ComplianceFilter.COMPLIANT)
        out = self.run_cmd(cmd_test, inputs=["crane slate"])
        self.assertIn("CRANE", out)
        self.assertIn("SLATE", out)

    def test_unfiltered_mode(self):
        # Default GameState mode is ALL_WORDS / UNFILTERED -> step2_pool path
        # using soln.all_answers (scores not yet updated).
        self.assertFalse(wordle._is_hard_mode(self.gs))
        self.assertFalse(wordle._is_possible_answers_mode(self.gs))
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("Multi-step lookahead", out)
        # mode label reflects either "top N" or "possible answers" fallback.
        self.assertTrue("top " in out or "possible answers" in out)

    def test_unfiltered_top_pool(self):
        # Populate entropy scores so the step2_pool top-200 branch is taken.
        soln = self.soln()
        soln.compute_scores(soln.all_answers, wordle.ScoringMethod.ENTROPY_GAIN)
        self.assertTrue(soln.scores_updated)
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("top ", out)


class TestCmdTestAnswerSet(CliTestCase):
    def test_answer_word_shows_pattern(self):
        self.soln().answer_word = "slate"
        out = self.run_cmd(cmd_test, inputs=["crane"])
        # The "vs SLATE:" pattern line is printed when an answer is simulated.
        self.assertIn("vs SLATE", out)


class TestCmdTestMidGame(CliTestCase):
    def test_consistent_after_guess(self):
        soln = self.soln()
        # Record crane all-gray-ish then test a still-valid candidate.
        soln.apply_guess("brain", wordle.parse_response("00000"))
        # Pick a remaining word so it is consistent.
        word = soln.current_words[0]
        out = self.run_cmd(cmd_test, inputs=[word])
        self.assertIn("Consistent with all guesses.", out)

    def test_midgame_lookahead_runs(self):
        # "cloud" all-gray leaves 4 candidates (n > 2) and a non-full game,
        # so the multi-step lookahead + ERD surfacing block executes.
        soln = self.soln()
        soln.apply_guess("cloud", wordle.parse_response("00000"))
        self.assertGreater(len(soln.current_words), 2)
        self.assertFalse(soln._is_full_game())
        word = soln.current_words[0]
        out = self.run_cmd(cmd_test, inputs=[word])
        self.assertIn("Multi-step lookahead", out)
        self.assertIn("Consistent with all guesses.", out)

    def test_midgame_compare_lookahead(self):
        soln = self.soln()
        soln.apply_guess("cloud", wordle.parse_response("00000"))
        words = " ".join(soln.current_words[:3])
        out = self.run_cmd(cmd_test, inputs=[words])
        self.assertIn("words:", out)

    def test_midgame_answers_mode_erd_branch(self):
        # Possible-answers mode mid-game exercises the answers-only ERD policy
        # branch of _multistep_stats.
        _set_mode(self.gs, GuessUniverse.ALL_ANSWERS, ComplianceFilter.COMPLIANT)
        soln = self.soln()
        soln.apply_guess("cloud", wordle.parse_response("00000"))
        word = soln.current_words[0]
        out = self.run_cmd(cmd_test, inputs=[word])
        self.assertIn("Multi-step lookahead (possible answers)", out)

    def test_second_call_uses_cached_scores(self):
        # Calling cmd_test twice on the same word/soln: the first call computes
        # and caches all four step-1 method scores; the second takes the
        # all-cached fast path in _multistep_stats.
        self.run_cmd(cmd_test, inputs=["crane"])
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("Multi-step lookahead", out)

    def test_midgame_small_pool_skips_lookahead(self):
        # Drive remaining words to <= 2 so the n > 2 lookahead block is skipped.
        soln = self.soln()
        soln.apply_guess("crane", wordle.parse_response("ggggg"))
        # Now only "crane" remains (n == 1).
        self.assertLessEqual(len(soln.current_words), 2)
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertNotIn("Multi-step lookahead", out)
        self.assertIn("Groups:", out)

    def test_incompatible_patterns_do_not_crash_stats(self):
        soln = self.soln()
        soln.apply_guess("crane", wordle.parse_response("ggggg"))
        soln.apply_guess("slate", wordle.parse_response("ggggg"))
        soln.apply_guess("trace", wordle.parse_response("ggggg"))
        self.assertEqual(len(soln.current_words), 0)

        out = self.run_cmd(cmd_test, inputs=["crane"])

        self.assertIn("No words remaining", out)


class TestExplainConflict(CliTestCase):
    def test_conflict_all_branches(self):
        soln = self.soln()
        # Manually record a guess pattern engineered so testing "crane"
        # produces conflicts hitting the green / yellow+gray / yellow+green /
        # gray explanation branches of _explain_conflict.
        # hyp = calculate_response("stale", "crane") = [gray,gray,green,gray,green]
        recorded = ["green", "yellow", "yellow", "gray", "gray"]
        soln.guesses.append(["stale", recorded])
        out = self.run_cmd(cmd_test, inputs=["crane"])
        self.assertIn("Conflicts:", out)
        self.assertIn("Not a valid candidate.", out)
        self.assertIn("position 1 must be S", out)       # green branch
        self.assertIn("T must be in the answer", out)    # yellow + gray
        self.assertIn("A can't be at position 3", out)   # yellow + green
        self.assertIn("E is not in the answer", out)     # gray, no dup

    def test_conflict_gray_has_other(self):
        soln = self.soln()
        # Use a guess with a doubled letter ('e' at pos2 and pos3) to exercise
        # the gray "no extra E beyond those found" branch (has_other True).
        # hyp = calculate_response("sweet","rates")
        #     = [yellow(s), gray(w), gray(e2), green(e3), yellow(t)]
        # recorded[2]=yellow (conflict vs gray -> "E must be in the answer")
        # recorded[3]=gray   (conflict vs green); its sibling 'e' at pos2 is
        #   recorded yellow (non-gray) -> has_other True ->
        #   "no extra E beyond those found".
        recorded = ["yellow", "gray", "yellow", "gray", "yellow"]
        soln.guesses.append(["sweet", recorded])
        out = self.run_cmd(cmd_test, inputs=["rates"])
        self.assertIn("Conflicts:", out)
        self.assertIn("no extra E beyond", out)


if __name__ == "__main__":
    unittest.main()
