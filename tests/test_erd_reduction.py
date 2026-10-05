"""Reducing response-group results to an ERD, and storing an opener's."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from cache_sqlite import ScoreCache
from erd_reduction import (
    ALL_GREEN_PATTERN_TEXT,
    reduce_candidate_erd,
    reduce_opener,
    response_group_is_solved,
)
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache

ANSWERS = ["crane", "slate"]
GROUP_BUDGET = GAME_GUESSES - 1


def _group(pattern, answer_count, best_erd, max_remaining_depth,
           cache_state="exact"):
    return {"pattern": pattern, "answer_count": answer_count,
            "best_erd": best_erd, "max_remaining_depth": max_remaining_depth,
            "cache_state": cache_state}


class ReduceCandidateERDTest(unittest.TestCase):
    def test_a_group_with_a_result_and_a_lone_survivor_reduce_to_their_mean(self):
        summary = reduce_candidate_erd([
            _group(ALL_GREEN_PATTERN_TEXT, 1, None, None),
            _group("-y---", 3, 1.0, 1),
        ], GROUP_BUDGET)

        self.assertEqual(summary["state"], "complete")
        self.assertAlmostEqual(summary["erd"], 1.0 + (3 * 1.0) / 4)
        self.assertEqual(summary["max_remaining_depth"], 2)

    def test_a_proven_loss_makes_the_candidate_infeasible(self):
        summary = reduce_candidate_erd(
            [_group("-y---", 3, None, None, cache_state="loss")], GROUP_BUDGET)

        self.assertEqual(summary["state"], "infeasible")
        self.assertIsNone(summary["erd"])

    def test_an_unsolved_group_leaves_the_candidate_pending(self):
        summary = reduce_candidate_erd(
            [_group("-y---", 3, None, None, cache_state="missing")],
            GROUP_BUDGET)

        self.assertEqual(summary["state"], "pending")

    def test_a_group_is_solved_only_with_a_result_and_its_worst_case(self):
        self.assertTrue(response_group_is_solved(
            _group("-y---", 3, 1.0, 1), GROUP_BUDGET))
        self.assertFalse(response_group_is_solved(
            _group("-y---", 3, 1.0, None), GROUP_BUDGET))


class ReduceOpenerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = ScoreCache(os.path.join(self._tmp.name, "cache.sqlite3"),
                                ANSWERS, checkpoint_on_close=False)
        self.addCleanup(self.cache.close)
        self.response_cache = ResponseCache(ANSWERS, score_cache=None)
        self.both = ScoreCache.encode_subset(ANSWERS)

    def _reduce(self, opener):
        return reduce_opener(opener, ANSWERS, self.response_cache, self.cache,
                             ERD_ALL, GROUP_BUDGET)

    def test_an_opener_that_splits_the_answers_into_singletons_needs_no_result(self):
        # CRANE answers itself and leaves SLATE alone: both groups are solved
        # by playing the survivor, so no branch result is read.
        reduction = self._reduce("crane")

        self.assertEqual(reduction["state"], "complete")
        self.assertAlmostEqual(reduction["erd"], 1.5)
        self.assertEqual(reduction["max_remaining_depth"], 2)
        self.assertEqual(reduction["response_group_count"], 2)

    def test_an_opener_whose_group_has_no_result_is_pending(self):
        # HOWDY gives both answers one response, so the pair is a branch the
        # cache has to have solved.
        self.assertEqual(self._reduce("howdy")["state"], "pending")

    def test_an_opener_whose_group_is_solved_reduces_over_that_result(self):
        self.cache.write(self.both, ERD_ALL, "crane", 1.5, max_depth=2,
                         solve_budget=None)

        reduction = self._reduce("howdy")

        self.assertEqual(reduction["state"], "complete")
        self.assertAlmostEqual(reduction["erd"], 2.5)
        self.assertEqual(reduction["max_remaining_depth"], 3)

    def test_an_opener_whose_group_is_a_proven_loss_is_infeasible(self):
        self.cache.write_loss(self.both, ERD_ALL, GROUP_BUDGET)

        self.assertEqual(self._reduce("howdy")["state"], "infeasible")


class WriteOpenerERDTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = ScoreCache(os.path.join(self._tmp.name, "cache.sqlite3"),
                                ANSWERS, checkpoint_on_close=False)
        self.addCleanup(self.cache.close)

    def _stored(self):
        return [tuple(row) for row in self.cache._conn.execute(
            "SELECT opener, erd, max_remaining_depth, response_group_count "
            "FROM opener_erd_by_policy")]

    def test_the_opener_is_stored_lowercased_with_its_reduction(self):
        self.cache.write_opener_erd("SALET", ERD_ALL, 3.5, 6, 140)

        self.assertEqual(self._stored(), [("salet", 3.5, 6, 140)])

    def test_storing_the_same_opener_again_replaces_it(self):
        self.cache.write_opener_erd("salet", ERD_ALL, 3.5, 6, 140)
        self.cache.write_opener_erd("salet", ERD_ALL, 3.5, 6, 140)

        self.assertEqual(self._stored(), [("salet", 3.5, 6, 140)])

    def test_a_disk_error_propagates_rather_than_being_swallowed(self):
        # The opener is marked done on the strength of this row, so a write
        # that failed quietly would record an opener done with no ERD.
        failing = mock.Mock()
        failing.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        with mock.patch.object(self.cache, "_conn", failing):
            with self.assertRaises(sqlite3.OperationalError):
                self.cache.write_opener_erd("salet", ERD_ALL, 3.5, 6, 140)


if __name__ == "__main__":
    unittest.main()
