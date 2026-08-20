"""Tests for the shared ERD swarm report model."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
import erd_search
import report_model
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportSources,
    WORKER_LIVENESS_SECONDS,
    _candidate_erd_summary,
    _source_word_group_key,
    _resolved_candidate_erd,
    _grouped_response_groups,
    _response_group_key,
    _response_group_rollup,
    _root_progress_estimate,
    validate_report_request,
    branch_reference,
    collect_overview_report,
    collect_report,
    collect_source_report,
    collect_workers_report,
    normalize_worker_descent,
    parse_rich_spine,
    parse_report_branch_target,
    resolve_branch_reference,
)
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache


ANSWERS = ["salet", "crane", "nurdy", "khaki"]


class ReportModelTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = self.temporary_directory.name
        self.queue_path = os.path.join(directory, "queue.sqlite3")
        self.telemetry_path = os.path.join(directory, "telemetry.sqlite3")
        self.cache_path = os.path.join(directory, "cache.sqlite3")
        self.answer_list_path = os.path.join(directory, "answers.txt")
        self.candidate_list_path = os.path.join(directory, "candidates.txt")
        with open(self.answer_list_path, "w") as answer_file:
            answer_file.write("\n".join(ANSWERS) + "\n")
        with open(self.candidate_list_path, "w") as candidate_file:
            candidate_file.write("\n".join(ANSWERS + ["raise"]) + "\n")
        self.sources = ReportSources(
            queue_path=self.queue_path,
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=self.telemetry_path,
        )
        self._open_queue().close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _open_queue(self):
        return ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)

    @staticmethod
    def _group(pattern, answer_count, best_erd, max_remaining_depth,
               cache_state="exact"):
        return {
            "pattern": pattern,
            "answer_count": answer_count,
            "best_erd": best_erd,
            "max_remaining_depth": max_remaining_depth,
            "cache_state": cache_state,
        }

    def test_candidate_erd_summary_folds_solved_groups_with_the_candidate_guess(self):
        summary = _candidate_erd_summary([
            self._group("-----", 8, 2.1, 3),
            self._group("----g", 2, 1.5, 2),
            self._group("ggggg", 1, None, None),
        ], 5)
        # 1 (the candidate's own guess) + weighted mean of remaining depth; the
        # all-green group is the candidate itself, contributing zero.
        self.assertEqual(summary["state"], "complete")
        self.assertAlmostEqual(summary["erd"], 1.0 + (8 * 2.1 + 2 * 1.5) / 11)
        self.assertEqual(summary["max_remaining_depth"], 4)
        self.assertEqual(summary["resolved_group_count"], 3)
        self.assertEqual(summary["response_group_count"], 3)

    def test_candidate_erd_summary_is_pending_while_a_group_is_unsolved(self):
        summary = _candidate_erd_summary([
            self._group("-----", 8, 2.1, 3),
            self._group("y----", 5, None, None, cache_state="missing"),
            self._group("ggggg", 1, None, None),
        ], 5)
        self.assertEqual(summary["state"], "pending")
        self.assertIsNone(summary["erd"])
        self.assertIsNone(summary["max_remaining_depth"])
        self.assertEqual(summary["resolved_group_count"], 2)
        self.assertEqual(summary["infeasible_group_count"], 0)
        self.assertEqual(summary["response_group_count"], 3)

    def test_candidate_erd_summary_is_infeasible_when_a_group_is_a_proven_loss(self):
        summary = _candidate_erd_summary([
            self._group("-----", 8, 2.1, 3),
            self._group("yy---", 5, None, None, cache_state="loss"),
            self._group("-y---", 3, None, None, cache_state="missing"),
        ], 5)
        # A proven loss has no finite line: infeasible, not pending, and it
        # dominates a still-unsolved group.
        self.assertEqual(summary["state"], "infeasible")
        self.assertIsNone(summary["erd"])
        self.assertIsNone(summary["max_remaining_depth"])
        self.assertEqual(summary["infeasible_group_count"], 1)

    def test_candidate_erd_summary_solves_a_lone_survivor_in_one_more_guess(self):
        summary = _candidate_erd_summary([self._group("----y", 1, None, None)], 5)
        self.assertEqual(summary["state"], "complete")
        self.assertEqual(summary["erd"], 2.0)
        self.assertEqual(summary["max_remaining_depth"], 2)

    def test_candidate_erd_summary_treats_erd_without_worst_case_as_pending(self):
        # An ERD present but no proven worst-case line cannot complete the fold;
        # it must not crash the max() and must not read as complete.
        summary = _candidate_erd_summary([self._group("-----", 8, 2.1, None)], 5)
        self.assertEqual(summary["state"], "pending")
        self.assertIsNone(summary["max_remaining_depth"])

    def test_candidate_erd_summary_lone_survivor_is_infeasible_with_no_guess_left(self):
        # A lone survivor needs one guess to play; at group_budget 0 there is no
        # guess left, so it is a proven loss — matching evaluate_candidate's
        # budget floor, checked before its n == 1 shortcut.
        summary = _candidate_erd_summary([self._group("----y", 1, None, None)], 0)
        self.assertEqual(summary["state"], "infeasible")
        self.assertEqual(summary["infeasible_group_count"], 1)
        # The all-green group was already solved by the guess that reached it,
        # so it stays complete even with no budget.
        solved = _candidate_erd_summary([self._group("ggggg", 1, None, None)], 0)
        self.assertEqual(solved["state"], "complete")
        self.assertEqual(solved["erd"], 1.0)

    # SALET against the 4-word fixture list splits into its own all-green
    # group plus three lone survivors: 1 + (0+1+1+1)/4 = 1.75.
    _SALET_GROUPS = [
        {"pattern": "ggggg", "answer_count": 1, "best_erd": None,
         "max_remaining_depth": None, "cache_state": "exact"},
        {"pattern": "-----", "answer_count": 1, "best_erd": None,
         "max_remaining_depth": None, "cache_state": "exact"},
        {"pattern": "-----", "answer_count": 1, "best_erd": None,
         "max_remaining_depth": None, "cache_state": "exact"},
        {"pattern": "-----", "answer_count": 1, "best_erd": None,
         "max_remaining_depth": None, "cache_state": "exact"},
    ]

    def test_resolved_candidate_erd_persists_a_complete_fold(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        branch_key = ScoreCache.encode_subset(ANSWERS)
        summary = _resolved_candidate_erd(
            cache, branch_key, "salet", ERD_ALL, self._SALET_GROUPS, 5
        )
        self.assertEqual(summary["state"], "complete")
        self.assertAlmostEqual(summary["erd"], 1.75)
        stored = cache.read_candidate_erd(branch_key, "salet", ERD_ALL)
        self.assertEqual(stored["erd"], summary["erd"])
        self.assertEqual(stored["response_group_count"], 4)
        cache.close()

    def test_resolved_candidate_erd_does_not_persist_a_pending_fold(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        branch_key = ScoreCache.encode_subset(ANSWERS)
        summary = _resolved_candidate_erd(
            cache, branch_key, "nurdy", ERD_ALL,
            [self._group("-----", 2, None, None, cache_state="missing")], 5,
        )
        self.assertEqual(summary["state"], "pending")
        self.assertIsNone(cache.read_candidate_erd(branch_key, "nurdy", ERD_ALL))
        cache.close()

    def test_resolved_candidate_erd_reads_the_stored_row_without_refolding(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        branch_key = ScoreCache.encode_subset(ANSWERS)
        cache.write_candidate_erd(branch_key, "salet", ERD_ALL, 1.75, 1, 4)
        with patch("report_model._candidate_erd_summary") as folded:
            summary = _resolved_candidate_erd(
                cache, branch_key, "salet", ERD_ALL, self._SALET_GROUPS, 5
            )
        folded.assert_not_called()
        self.assertEqual(summary, {
            "state": "complete", "erd": 1.75, "max_remaining_depth": 1,
            "resolved_group_count": 4, "infeasible_group_count": 0,
            "response_group_count": 4,
        })
        cache.close()

    def test_resolved_candidate_erd_refolds_when_the_stored_group_count_is_stale(self):
        # A changed vocabulary reshapes a candidate's own grouping, so a row
        # stored for the old shape must not answer for the new one.
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        branch_key = ScoreCache.encode_subset(ANSWERS)
        cache.write_candidate_erd(branch_key, "salet", ERD_ALL, 1.75, 1, 4)
        summary = _resolved_candidate_erd(
            cache, branch_key, "salet", ERD_ALL,
            [self._group("ggggg", 1, None, None),
             self._group("-----", 2, None, None, cache_state="missing")],
            5,
        )
        self.assertEqual(summary["state"], "pending")
        self.assertEqual(summary["response_group_count"], 2)
        cache.close()

    def test_resolved_candidate_erd_without_a_cache_still_folds(self):
        summary = _resolved_candidate_erd(
            None, ScoreCache.encode_subset(ANSWERS), "salet", ERD_ALL,
            self._SALET_GROUPS, 5,
        )
        self.assertEqual(summary["state"], "complete")
        self.assertAlmostEqual(summary["erd"], 1.75)

    def test_resolved_candidate_erd_stored_map_path_matches_direct_lookup(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        branch_key = ScoreCache.encode_subset(ANSWERS)
        cache.write_candidate_erd(branch_key, "salet", ERD_ALL, 1.75, 1, 4)
        stored_map = cache.candidate_erd_map(ERD_ALL)
        with patch("report_model._candidate_erd_summary") as folded:
            summary = _resolved_candidate_erd(
                cache, branch_key, "salet", ERD_ALL, self._SALET_GROUPS, 5,
                stored_map,
            )
        folded.assert_not_called()
        self.assertEqual(summary["erd"], 1.75)
        cache.close()

    def test_leaderboard_persists_and_then_reuses_each_complete_erd(self):
        sources = self._leaderboard_sources(
            ["crane", "slate"], ["crane", "slate", "raise", "howdy"]
        )
        first = collect_report(sources, ReportRequest(report_kind="leaderboard"))
        cache = ScoreCache(sources.cache_path, ["crane", "slate"],
                            checkpoint_on_close=False)
        stored = cache.candidate_erd_map(ERD_ALL)
        cache.close()
        # crane/slate/raise complete and are persisted; howdy never completes
        # in this fixture (both answers collide in one unsolved group).
        self.assertEqual(len(stored), 3)
        with patch(
            "report_model._candidate_erd_summary", wraps=_candidate_erd_summary,
        ) as folded:
            second = collect_report(
                sources, ReportRequest(report_kind="leaderboard")
            )
        self.assertEqual(folded.call_count, 1)   # howdy alone
        self.assertEqual(second["data"]["rows"], first["data"]["rows"])

    def test_collect_word_report_populates_candidate_erd_summary(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
        )
        report = collect_report(self.sources, request)
        summary = report["data"]["erd_summary"]
        self.assertEqual(set(summary), {
            "state", "erd", "max_remaining_depth", "resolved_group_count",
            "infeasible_group_count", "response_group_count",
        })
        self.assertIn(summary["state"], {"complete", "pending", "infeasible"})
        # The fold walks every response group, not the filtered/limited view.
        self.assertEqual(
            summary["response_group_count"],
            report["data"]["response_group_counts"]["response_group_count"],
        )

    def test_collect_word_report_folds_candidate_erd_below_the_root(self):
        # A one-step spine: SALET/-y--- leaves {khaki}, and CRANE is the
        # candidate folded at guess_depth 1.  The fold measures depth from this
        # branch, not from the root — CRANE splits {khaki} into a lone survivor,
        # so its ERD is 2.0 (play CRANE, then khaki), not the 3.0 a
        # depth-from-root fold would give (SALET, CRANE, khaki).
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet -y--- crane"),
        )
        report = collect_report(self.sources, request)
        context = report["data"]["context"]
        self.assertEqual(context["guess_depth"], 1)
        # The branch answer set is smaller than the full list.
        self.assertLess(context["answer_count"], len(ANSWERS))
        summary = report["data"]["erd_summary"]
        self.assertEqual(summary["state"], "complete")
        self.assertEqual(summary["erd"], 2.0)
        self.assertLessEqual(summary["erd"], GAME_GUESSES - context["guess_depth"])
        self.assertEqual(
            summary["response_group_count"],
            report["data"]["response_group_counts"]["response_group_count"],
        )

    def test_reports_live_owner_priority_for_stale_promoted_branch(self):
        response_groups = ResponseCache(ANSWERS).group_words("salet", ANSWERS)
        pattern, branch_words = next(iter(response_groups.items()))
        branch_key = ScoreCache.encode_subset(branch_words)
        queue = self._open_queue()
        queue.add_pending_many([
            (branch_key, len(branch_words), 9, "salet", pattern),
        ])
        claimed = queue.claim_next("worker-0")
        queue.create_branch(
            branch_key, len(branch_words), 5, priority=1_000_001,
            source_word="salet", source_pattern=pattern,
            source_work_id=claimed["source_work_id"],
        )
        queue.close()

        overview = collect_overview_report(self.sources)
        word = collect_report(self.sources, ReportRequest(
            branch_target=parse_report_branch_target("salet"),
        ))
        branch = collect_report(self.sources, ReportRequest(
            report_kind="branch",
            branch_target=parse_report_branch_target(
                "@" + branch_reference(branch_key)),
        ))

        self.assertEqual(overview["data"]["branches"][0]["priority"], 9)
        self.assertEqual(
            next(row for row in word["data"]["response_groups"]
                 if row["priority"] is not None)["priority"], 9)
        self.assertEqual(branch["data"]["queue"]["priority"], 9)

    def test_response_group_key_orders_status_by_lifecycle(self):
        order = [
            _response_group_key({"branch_status": status}, "status")[0]
            for status in ("active", "pending", "done", "unqueued")
        ]
        self.assertEqual(order, sorted(order))

    def test_response_group_key_buckets_answer_count(self):
        labels = {
            answer_count: _response_group_key(
                {"answer_count": answer_count}, "answer_count"
            )[1]
            for answer_count in (1, 2, 9, 10, 29, 30, 99, 100, 500)
        }
        self.assertEqual(labels, {
            1: "1", 2: "2–9", 9: "2–9", 10: "10–29", 29: "10–29",
            30: "30–99", 99: "30–99", 100: "100+", 500: "100+",
        })

    def test_response_group_key_orders_cache_state_by_urgency(self):
        order = [
            _response_group_key({"cache_state": cache_state}, "cache_state")[0]
            for cache_state in ("missing", "loss", "exact", "not_applicable")
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(
            _response_group_key({"cache_state": "not_applicable"}, "cache_state")[1],
            "trivial",
        )

    def test_response_group_key_splits_by_worker_presence(self):
        self.assertEqual(
            _response_group_key({"worker_count": 2}, "worker_presence"),
            (0, "has worker"),
        )
        self.assertEqual(
            _response_group_key({"worker_count": 0}, "worker_presence"),
            (1, "no worker"),
        )

    def test_response_group_key_orders_priority_high_first_with_unset_last(self):
        order = [
            _response_group_key({"priority": priority}, "priority")[0]
            for priority in (1, 0, None)
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(
            _response_group_key({"priority": 1}, "priority")[1], "priority 1"
        )
        self.assertEqual(
            _response_group_key({"priority": None}, "priority")[1], "no priority"
        )

    def test_response_group_rollup_partitions_by_cache_state(self):
        rows = [
            {"answer_count": 1, "cache_state": "not_applicable"},
            {"answer_count": 1, "cache_state": "not_applicable"},
            {"answer_count": 5, "cache_state": "exact"},
            {"answer_count": 8, "cache_state": "loss"},
            {"answer_count": 12, "cache_state": "missing"},
        ]
        rollup = _response_group_rollup(rows)
        self.assertEqual(rollup, {
            "answer_count": 27, "branch_count": 5, "trivial_count": 2,
            "exact_count": 1, "loss_count": 1, "missing_count": 1,
        })
        # trivial/exact/loss/missing partition cache_state exhaustively.
        self.assertEqual(
            rollup["trivial_count"] + rollup["exact_count"]
            + rollup["loss_count"] + rollup["missing_count"],
            rollup["branch_count"],
        )

    def test_grouped_response_groups_orders_groups_and_sums_rollups(self):
        rows = [
            {"branch_status": "done", "answer_count": 3, "cache_state": "exact"},
            {"branch_status": "unqueued", "answer_count": 10, "cache_state": "missing"},
            {"branch_status": "active", "answer_count": 1, "cache_state": "not_applicable"},
        ]
        groups = _grouped_response_groups(rows, "status")
        self.assertEqual(
            [group["label"] for group in groups], ["active", "done", "unqueued"]
        )
        self.assertEqual(
            sum(group["rollup"]["answer_count"] for group in groups),
            sum(row["answer_count"] for row in rows),
        )

    def test_collect_word_report_computes_group_by_status(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
            filters=ReportFilters(group_by="status"),
        )
        data = collect_report(self.sources, request)["data"]
        summary = data["response_group_summary"]
        self.assertEqual(
            summary["trivial_count"] + summary["exact_count"]
            + summary["loss_count"] + summary["missing_count"],
            summary["branch_count"],
        )
        groups = data["response_group_groups"]
        self.assertEqual(
            sum(group["rollup"]["branch_count"] for group in groups),
            summary["branch_count"],
        )
        self.assertEqual(
            sum(group["rollup"]["answer_count"] for group in groups),
            summary["answer_count"],
        )

    def test_collect_word_report_omits_groups_when_ungrouped(self):
        request = ReportRequest(branch_target=parse_report_branch_target("salet"))
        data = collect_report(self.sources, request)["data"]
        self.assertNotIn("response_group_groups", data)
        self.assertIn("response_group_summary", data)

    def test_response_group_summary_reflects_active_filters_not_all_groups(self):
        # The grand summary describes what's currently filtered to, like the
        # response_groups list itself — not every response group regardless of
        # filter (that invariant belongs to erd_summary, which must fold every
        # group to compute one true word ERD).
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
            filters=ReportFilters(minimum_answer_count=0),
        )
        data = collect_report(self.sources, request)["data"]
        self.assertEqual(
            data["response_group_summary"]["branch_count"], data["matched_rows"]
        )

    def test_response_group_summary_ignores_display_limit(self):
        # A limit truncates what's shown, not what's summarized.
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
            filters=ReportFilters(limit=1),
        )
        data = collect_report(self.sources, request)["data"]
        self.assertGreaterEqual(
            data["response_group_summary"]["branch_count"],
            len(data["response_groups"]),
        )

    def test_response_group_groups_respect_display_limit(self):
        # response_group_groups must be built from the same limited set as
        # the flat response_groups list — otherwise the card grid renders
        # more rows than "Shown N of M matched" claims (PR #231 review).
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
            filters=ReportFilters(group_by="status", limit=1),
        )
        data = collect_report(self.sources, request)["data"]
        total_grouped_rows = sum(
            len(group["rows"]) for group in data["response_group_groups"]
        )
        self.assertEqual(total_grouped_rows, len(data["response_groups"]))
        self.assertLessEqual(total_grouped_rows, 1)

    def _leaderboard_sources(self, answers, candidates):
        directory = self.temporary_directory.name
        answer_path = os.path.join(directory, "lb_answers.txt")
        candidate_path = os.path.join(directory, "lb_candidates.txt")
        with open(answer_path, "w") as answer_file:
            answer_file.write("\n".join(answers) + "\n")
        with open(candidate_path, "w") as candidate_file:
            candidate_file.write("\n".join(candidates) + "\n")
        return ReportSources(
            queue_path=self.queue_path,
            cache_path=self.cache_path,
            answer_list_path=answer_path,
            candidate_list_path=candidate_path,
            telemetry_path=self.telemetry_path,
        )

    def test_leaderboard_ranks_complete_openers_by_erd(self):
        # With two answers, an opener that separates them into singletons is
        # complete with no cache reads at all: crane/slate are also answers
        # (all-green self group contributes 0 → ERD 1.5), raise is not
        # (two size-1 groups → ERD 2.0), and howdy shares no letters so both
        # answers collide in one unsolved group → pending.
        sources = self._leaderboard_sources(
            ["crane", "slate"], ["crane", "slate", "raise", "howdy"]
        )
        report = collect_report(sources, ReportRequest(report_kind="leaderboard"))
        self.assertEqual(report["report_kind"], "leaderboard")
        data = report["data"]
        self.assertEqual(data["candidate_count"], 4)
        self.assertEqual(
            data["counts"], {"complete": 3, "pending": 1, "infeasible": 0}
        )
        self.assertEqual([row["word"] for row in data["rows"]],
                         ["crane", "slate", "raise"])
        self.assertEqual([row["rank"] for row in data["rows"]], [1, 2, 3])
        self.assertAlmostEqual(data["rows"][0]["erd"], 1.5)
        self.assertAlmostEqual(data["rows"][2]["erd"], 2.0)
        self.assertTrue(data["rows"][0]["word_is_answer"])
        self.assertFalse(data["rows"][2]["word_is_answer"])
        self.assertEqual(data["rows"][0]["answer_count"], 2)
        self.assertEqual(
            [group["answer_count"] for group in data["rows"][0]["response_groups"]],
            [1, 1],
        )
        self.assertEqual(data["response_pattern_count"], 243)

    def test_leaderboard_report_counts_partition_the_candidate_list(self):
        report = collect_report(
            self.sources, ReportRequest(report_kind="leaderboard")
        )
        data = report["data"]
        counts = data["counts"]
        self.assertEqual(
            counts["complete"] + counts["pending"] + counts["infeasible"],
            data["candidate_count"],
        )
        self.assertEqual(len(data["rows"]), counts["complete"])
        erds = [row["erd"] for row in data["rows"]]
        self.assertEqual(erds, sorted(erds))
        self.assertEqual([row["rank"] for row in data["rows"]],
                         list(range(1, len(data["rows"]) + 1)))

    def test_leaderboard_builds_matrix_beside_the_cache_not_the_cwd(self):
        # load_or_build derives the matrix directory from the cache *path*;
        # passing a directory would apply dirname twice and strand a .npy in the
        # cwd while never finding the swarm's matrix.
        import glob
        import report_model
        report_model._candidate_skeleton_memo = None
        before = set(glob.glob("*.npy"))
        sources = self._leaderboard_sources(["crane", "slate"], ["crane", "slate"])
        collect_report(sources, ReportRequest(report_kind="leaderboard"))
        self.assertEqual(set(glob.glob("*.npy")), before)  # nothing stray in cwd
        cache_directory = os.path.dirname(sources.cache_path)
        self.assertTrue(glob.glob(os.path.join(cache_directory, "*.npy")))

    def test_leaderboard_reports_honest_empty_on_cache_error(self):
        # A mid-build cache failure must not publish a truncated ranking that
        # reads as complete; the report is empty with the error on the source.
        sources = self._leaderboard_sources(
            ["crane", "slate"], ["crane", "slate", "raise"]
        )
        with patch.object(
            ScoreCache, "report_branch_row_maps",
            side_effect=sqlite3.OperationalError("cache read failed"),
        ):
            report = collect_report(
                sources, ReportRequest(report_kind="leaderboard")
            )
        self.assertFalse(report["sources"]["cache"]["ok"])
        self.assertIn("cache read failed", report["sources"]["cache"]["error"])
        data = report["data"]
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["total_rows"], 0)
        self.assertEqual(
            data["counts"], {"complete": 0, "pending": 0, "infeasible": 0}
        )
        self.assertEqual(data["candidate_count"], 3)

    def test_rich_spine_parser_preserves_legacy_tuple_contract(self):
        path = "3:KHAKI:--y--/33→4:NURDY:---y-/17"
        expected = [
            (3, "KHAKI", "--y--", "33"),
            (4, "NURDY", "---y-", "17"),
        ]
        self.assertEqual(parse_rich_spine(path), expected)
        self.assertEqual(erd_search._parse_spine(path), expected)

    def test_worker_descent_is_normalized_without_changing_parser(self):
        parsed = parse_rich_spine("3:KHAKI:--y--/33→4:RAISE:---y-/17")
        descent = normalize_worker_descent(parsed, set(ANSWERS))
        self.assertEqual(descent[0], {
            "guess_depth": 3,
            "word": "khaki",
            "pattern": "--y--",
            "answer_count_text": "33",
            "word_is_answer": True,
        })
        self.assertFalse(descent[1]["word_is_answer"])
        self.assertIsInstance(parsed[0], tuple)

    def test_branch_reference_is_stable_long_and_distinct(self):
        reference = branch_reference(b"branch-one")
        self.assertEqual(reference, branch_reference(b"branch-one"))
        self.assertEqual(len(reference), 12)
        self.assertNotEqual(reference, branch_reference(b"branch-two"))
        self.assertNotIn("@", reference)

    def test_empty_sources_produce_complete_json_report(self):
        report = collect_overview_report(self.sources)
        self.assertEqual(json.loads(json.dumps(report)), report)
        self.assertEqual(set(report), {
            "schema_version", "report_kind", "generated_at", "branch_target",
            "filters", "tree", "sources", "data",
        })
        self.assertFalse(report["tree"])
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertTrue(report["sources"]["telemetry"]["ok"])
        self.assertTrue(report["sources"]["cache"]["ok"])
        self.assertEqual(report["data"]["branches"], [])
        self.assertEqual(report["data"]["workers"], [])

    def test_queue_and_cache_fail_independently(self):
        missing_directory = os.path.join(self.temporary_directory.name, "missing")
        unavailable_queue = ReportSources(
            queue_path=os.path.join(missing_directory, "queue.sqlite3"),
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=os.path.join(missing_directory, "telemetry.sqlite3"),
        )
        report = collect_overview_report(unavailable_queue)
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertFalse(report["sources"]["telemetry"]["ok"])
        self.assertTrue(report["sources"]["cache"]["ok"])

        unavailable_cache = ReportSources(
            queue_path=self.queue_path,
            cache_path=os.path.join(missing_directory, "cache.sqlite3"),
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=self.telemetry_path,
        )
        report = collect_overview_report(unavailable_cache)
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertFalse(report["sources"]["cache"]["ok"])

    def test_missing_queue_source_does_not_create_database_files(self):
        missing_queue_path = os.path.join(
            self.temporary_directory.name, "missing-queue.sqlite3"
        )
        missing_telemetry_path = os.path.join(
            self.temporary_directory.name, "missing-telemetry.sqlite3"
        )
        unavailable_queue = ReportSources(
            queue_path=missing_queue_path,
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=missing_telemetry_path,
        )

        report = collect_overview_report(unavailable_queue)

        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertFalse(report["sources"]["telemetry"]["ok"])
        self.assertFalse(os.path.exists(missing_queue_path))
        self.assertFalse(os.path.exists(missing_telemetry_path))

    def test_answer_list_failure_skips_queue_and_cache_normalization(self):
        with (
            patch("report_model.load_word_list", side_effect=OSError("missing answers")),
            patch("report_model.ERDQueue") as queue_class,
        ):
            report = collect_overview_report(self.sources)
        queue_class.assert_not_called()
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertIn("missing answers", report["sources"]["queue"]["error"])
        self.assertFalse(report["sources"]["cache"]["ok"])
        self.assertIn("missing answers", report["sources"]["cache"]["error"])
        self.assertEqual(report["data"]["branches"], [])
        self.assertEqual(report["data"]["workers"], [])

    def test_answer_list_failure_only_degrades_the_answer_asterisk(self):
        # Queue/tree/hotspot reports use the answer list purely to decorate
        # words with word_is_answer/best_guess_is_answer -- they are fully
        # meaningful without it. A missing or unreadable answer list must
        # cost only that decoration, not the report, and must never be
        # blamed on the (perfectly healthy) queue database.
        branch_key = ScoreCache.encode_subset(ANSWERS[:3])
        queue = self._open_queue()
        queue.create_branch(
            branch_key, 3, 5, source_word="salet", source_pattern=0,
            budget=4, spine="salet ----- crane y----",
        )
        queue._conn.execute(
            "UPDATE active_branches SET best_guess = 'crane', best_erd = 2.25 "
            "WHERE branch_id = ?",
            (queue._intern_branch(branch_key),),
        )
        queue.close()

        with patch("report_model.load_word_list", side_effect=OSError("missing answers")):
            queue_report = collect_report(self.sources, ReportRequest(report_kind="queue"))
            tree_report = collect_report(self.sources, ReportRequest(tree=True))
            hotspot_report = collect_report(self.sources, ReportRequest(report_kind="hotspots"))

        for report in (queue_report, tree_report, hotspot_report):
            self.assertTrue(report["sources"]["queue"]["ok"])
            self.assertIsNone(report["sources"]["queue"]["error"])

        # crane is a real answer word, but with the answer list unreadable
        # neither the spine step nor the best guess can be starred.
        row = queue_report["data"]["rows"][0]
        self.assertEqual(row["spine"][1]["word"], "crane")
        self.assertFalse(row["spine"][1]["word_is_answer"])
        self.assertEqual(row["best_guess"], "crane")
        self.assertFalse(row["best_guess_is_answer"])

        root_step = tree_report["data"]["nodes"][0]["step"]
        self.assertEqual(root_step["word"], "salet")
        self.assertFalse(root_step["word_is_answer"])

        self.assertEqual(hotspot_report["data"]["rows"][0]["best_guess"], "crane")
        self.assertFalse(hotspot_report["data"]["rows"][0]["best_guess_is_answer"])

    def test_queue_collection_error_does_not_relabel_attached_telemetry(self):
        with patch.object(
            ERDQueue, "counts_by_status",
            side_effect=sqlite3.OperationalError("queue read failed"),
        ):
            report = collect_overview_report(self.sources)
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertIn("queue read failed", report["sources"]["queue"]["error"])
        self.assertTrue(report["sources"]["telemetry"]["ok"])
        self.assertIsNone(report["sources"]["telemetry"]["error"])

    def test_programming_error_in_queue_collection_is_not_masked(self):
        with patch.object(ERDQueue, "counts_by_status", side_effect=KeyError("bug")):
            with self.assertRaises(KeyError):
                collect_overview_report(self.sources)

    def test_epoch_metadata_is_queue_source_metadata(self):
        queue = self._open_queue()
        queue.set_epoch(7, label="bounded-claims", git_sha="abcdef12")
        queue.close()
        report = collect_overview_report(self.sources)
        self.assertEqual(report["sources"]["queue"]["epoch"], 7)
        self.assertEqual(report["sources"]["queue"]["label"], "bounded-claims")
        self.assertEqual(report["sources"]["queue"]["git_sha"], "abcdef12")
        self.assertEqual(set(report["sources"]["telemetry"]), {
            "path", "ok", "error",
        })

    def test_custom_paths_are_retained_without_erd_search_ownership(self):
        self.assertEqual(self.sources.answer_list_path, self.answer_list_path)
        self.assertEqual(
            self.sources.candidate_list_path, self.candidate_list_path
        )
        report = collect_overview_report(self.sources)
        self.assertEqual(report["sources"]["queue"]["path"], self.queue_path)
        self.assertEqual(report["sources"]["cache"]["path"], self.cache_path)

    def test_active_branch_and_worker_normalize_domain_fields(self):
        now = int(time.time())
        branch_key = ScoreCache.encode_subset(ANSWERS[:3])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 9, "SALET", 0)])
        queue.claim_next("worker-2")
        queue.create_branch(
            branch_key, 3, 10, priority=9, source_word="SALET",
            source_pattern=0, budget=4, spine="SALET ----- CRANE y----",
        )
        queue._conn.execute(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done, done_at) "
            "VALUES (?, 0, 'worker-2', ?, 1, ?)",
            (queue._intern_branch(branch_key), now, now),
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 2, "
            "one_level_erd_pruned_candidates = 1, "
            "two_level_erd_pruned_candidates = 1, "
            "best_guess = 'CRANE', best_erd = 2.25, best_max_depth = 3, "
            "nodes_spent = 1234 WHERE branch_id = ?",
            (queue._intern_branch(branch_key),),
        )
        queue.heartbeat(
            "worker-2", pid=42, current_branch_key=branch_key, n_words=3,
            started_at=now, claims_done=7, claim_idx=4,
            claim_started_at=now, cache_hits=5, cache_misses=2, n_ok=3,
            n_cutoff=4, n_pruned=1, best_guess="CRANE", best_erd=2.25,
            bound_erd=2.5, cur_candidate="NURDY", cur_max_depth=5,
            cur_nodes=900, node_rate=45.5,
            cur_path="3:KHAKI:--y--/3→4:NURDY:---y-/2",
        )
        queue.close()

        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        branch = report["data"]["branches"][0]
        worker = report["data"]["workers"][0]
        self.assertEqual(branch["branch_status"], "active")
        self.assertEqual(branch["branch_phase"], "evaluating")
        self.assertEqual(branch["raw_status"], "in_progress")
        self.assertFalse(branch["is_cooperative"])
        self.assertEqual(branch["guess_depth"], 2)
        self.assertEqual(branch["source_pattern"], "-----")
        self.assertEqual(branch["completed_candidate_count"], 1)
        self.assertEqual(branch["bulk_completed_candidate_count"], 2)
        self.assertEqual(branch["one_level_erd_pruned_candidate_count"], 1)
        self.assertEqual(branch["two_level_erd_pruned_candidate_count"], 1)
        self.assertEqual(branch["best_max_remaining_depth"], 3)
        self.assertNotIn("best_max_depth", branch)
        self.assertTrue(branch["best_guess_is_answer"])
        self.assertEqual(worker["candidate_index"], 4)
        self.assertNotIn("claim_idx", worker)
        self.assertEqual(worker["current_max_guess_depth"], 5)
        self.assertNotIn("cur_max_depth", worker)
        self.assertTrue(worker["current_candidate_is_answer"])
        self.assertTrue(worker["best_guess_is_answer"])
        self.assertTrue(worker["descent"][0]["word_is_answer"])
        self.assertEqual(report["data"]["worker_totals"]["cache_hit_count"], 5)
        self.assertGreater(report["data"]["disk"]["total_bytes"], 0)
        self.assertGreaterEqual(report["data"]["disk"]["queue_wal_bytes"], 0)

    def test_all_gray_legacy_spine_fallback_has_one_guess(self):
        branch_key = b"legacy-gray"
        queue = self._open_queue()
        queue.create_branch(
            branch_key, 2, 4, source_word="SALET", source_pattern=0
        )
        queue.close()
        report = collect_overview_report(self.sources)
        branch = report["data"]["branches"][0]
        self.assertEqual(branch["guess_depth"], 1)
        self.assertEqual(branch["spine"][0]["pattern"], "-----")

    def test_cooperative_branch_is_active_and_counted(self):
        queue = self._open_queue()
        queue.create_branch(b"cooperative", 3, 5)
        queue.close()
        report = collect_overview_report(self.sources)
        self.assertTrue(report["data"]["branches"][0]["is_cooperative"])
        self.assertEqual(
            report["data"]["queue_counts"]["evaluating_cooperative_branch_count"], 1
        )

    def test_finalizing_user_branch_is_not_counted_as_evaluating(self):
        branch_key = b"finalizing-user"
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 5, "SALET", 0)])
        queue.claim_next("worker-1")
        queue.create_branch(branch_key, 3, 5)
        queue.try_finalize_branch(branch_key)
        queue.close()

        report = collect_overview_report(self.sources)
        queue_counts = report["data"]["queue_counts"]
        self.assertEqual(queue_counts["evaluating_user_branch_count"], 0)
        self.assertEqual(queue_counts["finalizing_branch_count"], 1)

    def test_overview_uses_one_queue_snapshot_and_preserves_worker_ramps(self):
        now = int(time.time())
        branch_key = b"ramp-branch"
        queue = self._open_queue()
        queue.create_branch(branch_key, 3, 5)
        branch_id = queue._intern_branch(branch_key)
        queue._conn.executemany(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done, done_at) "
            "VALUES (?, ?, 'worker-1', ?, 1, ?)",
            [(branch_id, index, now, now) for index in (1, 3)],
        )
        queue.heartbeat("worker-1", 1, branch_key, 3, now, 2)
        queue.close()

        with (
            patch.object(
                ERDQueue, "report_queue_rows", autospec=True,
                side_effect=ERDQueue.report_queue_rows,
            ) as report_queue_rows,
            patch("report_model.time.time", return_value=now),
        ):
            report = collect_overview_report(self.sources)

        self.assertEqual(report_queue_rows.call_count, 1)
        self.assertEqual(
            report["data"]["branches"][0]["completed_candidate_indexes"], [1, 3]
        )

    def test_active_report_rows_ignore_inactive_claim_history(self):
        now = int(time.time())
        active_branch_key = b"active-branch"
        inactive_branch_key = b"inactive-branch"
        queue = self._open_queue()
        queue.create_branch(active_branch_key, 3, 5)
        queue.heartbeat("worker-1", 1, active_branch_key, 3, now, 0)
        queue.create_branch(inactive_branch_key, 3, 100_000)
        inactive_branch_id = queue._intern_branch(inactive_branch_key)
        queue._conn.execute("BEGIN")
        queue._conn.executemany(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done, done_at) "
            "VALUES (?, ?, 'worker-2', ?, 1, ?)",
            (
                (inactive_branch_id, candidate_index, now, now)
                for candidate_index in range(100_000)
            ),
        )
        queue._conn.execute("COMMIT")

        started_at = time.monotonic()
        result = queue.report_queue_rows({"branch_statuses": ("active",)})
        elapsed_seconds = time.monotonic() - started_at
        queue.close()

        self.assertEqual(
            [row["branch_key"] for row in result["rows"]], [active_branch_key]
        )
        self.assertLess(elapsed_seconds, 1.0)

    def test_branch_report_lists_live_off_branch_claim_holders(self):
        now = int(time.time())
        parent_key = b"parent-branch"
        child_key = b"child-branch"
        queue = self._open_queue()
        queue.create_branch(
            parent_key, 3, 10, source_word="SALET", source_pattern=0, budget=5
        )
        queue.create_branch(
            child_key, 2, 10, source_word="SALET", source_pattern=6, budget=4
        )
        parent_id = queue._intern_branch(parent_key)
        queue._conn.execute(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done) "
            "VALUES (?, 17, 'worker-5', ?, 0)",
            (parent_id, now),
        )
        queue.heartbeat(
            "worker-5", pid=55, current_branch_key=child_key, n_words=2,
            started_at=now, claims_done=9, claim_idx=3, claim_started_at=now,
            cur_candidate="BEEFY", cur_max_depth=5,
        )
        queue.close()
        report = collect_report(self.sources, ReportRequest(
            report_kind="auto",
            branch_target=parse_report_branch_target(
                "@" + branch_reference(parent_key)
            ),
            filters=ReportFilters(),
        ))
        ownership = report["data"]["branch_ownership"]
        self.assertEqual(ownership["live_workers"], [])
        self.assertEqual(
            [worker["worker_id"] for worker in ownership["claim_holders_off_branch"]],
            ["worker-5"],
        )
        off_branch_holder = ownership["claim_holders_off_branch"][0]
        self.assertEqual(off_branch_holder["branch_reference"], branch_reference(child_key))
        self.assertEqual(off_branch_holder["in_flight_claim_count"], 1)
        self.assertEqual(off_branch_holder["branch_context"][0]["word"], "salet")
        self.assertEqual(off_branch_holder["branch_context"][0]["pattern"], "---g-")

    def test_finalizing_branch_remains_visible_after_worker_departure(self):
        now = int(time.time())
        live_key = b"live-finalizing"
        dead_key = b"dead-finalizing"
        queue = self._open_queue()
        for key, worker_id in ((live_key, "worker-1"), (dead_key, "worker-3")):
            queue.create_branch(key, 2, 1)
            queue.try_finalize_branch(key)
            queue.heartbeat(worker_id, 10, key, 2, now, 0)
        queue._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = 'worker-3'",
            (now - 31,),
        )
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        self.assertEqual(len(report["data"]["branches"]), 2)
        branches = {
            row["branch_key_hex"]: row for row in report["data"]["branches"]
        }
        self.assertEqual(branches[live_key.hex()]["branch_status"], "active")
        self.assertEqual(branches[live_key.hex()]["branch_phase"], "finalizing")
        self.assertEqual(branches[dead_key.hex()]["branch_status"], "pending")
        self.assertEqual(branches[dead_key.hex()]["branch_phase"], "finalizing")
        self.assertEqual(
            set(branches), {live_key.hex(), dead_key.hex()}
        )
        self.assertEqual(report["data"]["queue_counts"]["finalizing_branch_count"], 2)
        self.assertEqual(len(report["data"]["workers"]), 2)
        self.assertEqual(sum(worker["is_live"] for worker in report["data"]["workers"]), 1)

    def test_worker_on_removed_branch_is_not_on_a_live_branch(self):
        now = int(time.time())
        live_key = b"live-branch-xx"
        removed_key = b"gone-branch-xx"
        queue = self._open_queue()
        queue.create_branch(live_key, 2, 4)
        queue.heartbeat("worker-1", 1, live_key, 2, now, 0)
        # worker-2 still names a branch that has no pending or active row, as
        # happens for one heartbeat interval after a branch is finalized and
        # removed while the worker moves on to its next claim.
        queue.heartbeat("worker-2", 2, removed_key, 2, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        branch_keys = {
            row["branch_key_hex"] for row in report["data"]["branches"]
        }
        self.assertEqual(branch_keys, {live_key.hex()})
        workers = {
            worker["worker_id"]: worker for worker in report["data"]["workers"]
        }
        self.assertTrue(workers["worker-1"]["on_active_branch"])
        self.assertFalse(workers["worker-2"]["on_active_branch"])

    def test_idle_worker_is_not_on_a_live_branch(self):
        now = int(time.time())
        queue = self._open_queue()
        queue.heartbeat("worker-1", 1, None, 0, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        worker = report["data"]["workers"][0]
        self.assertIsNone(worker["branch_key_hex"])
        self.assertFalse(worker["on_active_branch"])

    def test_branch_worker_without_candidate_is_coordinating(self):
        now = int(time.time())
        key = b"coord-branch-x"
        queue = self._open_queue()
        queue.create_branch(key, 2, 4)
        queue.heartbeat("worker-1", 1, key, 2, now, 0, cur_candidate="crane")
        queue.heartbeat("worker-2", 2, key, 2, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        workers = {
            worker["worker_id"]: worker for worker in report["data"]["workers"]
        }
        self.assertEqual(workers["worker-1"]["state"], "working")
        self.assertEqual(workers["worker-2"]["state"], "coordinating")

    def test_overview_status_filter_tracks_worker_arrival_and_departure(self):
        now = int(time.time())
        working_key = b"working-branch"
        waiting_key = b"waiting-branch"
        queue = self._open_queue()
        queue.create_branch(working_key, 2, 4)
        queue.create_branch(waiting_key, 3, 4)
        queue.heartbeat("worker-1", 1, working_key, 2, now, 0)
        queue.close()

        active_request = ReportRequest(filters=ReportFilters(
            branch_statuses=("active",)
        ))
        with patch("report_model.time.time", return_value=now):
            active_report = collect_report(self.sources, active_request)
        self.assertEqual(
            [row["branch_key_hex"] for row in active_report["data"]["branches"]],
            [working_key.hex()],
        )

        queue = self._open_queue()
        queue._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = ?",
            (now - WORKER_LIVENESS_SECONDS - 1, "worker-1"),
        )
        queue.close()
        with patch("report_model.time.time", return_value=now):
            active_report = collect_report(self.sources, active_request)
            pending_report = collect_report(self.sources, ReportRequest(
                filters=ReportFilters(branch_statuses=("pending",), limit=1)
            ))
        self.assertEqual(active_report["data"]["branches"], [])
        self.assertEqual(len(pending_report["data"]["branches"]), 1)
        self.assertEqual(
            pending_report["data"]["branches"][0]["branch_status"], "pending"
        )

    def test_pending_overview_includes_scheduled_branch_before_evaluation(self):
        branch_key = b"queued-branch"
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 5, "raise", 0)])
        queue.close()

        report = collect_report(self.sources, ReportRequest(
            filters=ReportFilters(branch_statuses=("pending",))
        ))
        self.assertEqual(len(report["data"]["branches"]), 1)
        branch = report["data"]["branches"][0]
        self.assertEqual(branch["branch_key_hex"], branch_key.hex())
        self.assertEqual(branch["branch_status"], "pending")
        self.assertEqual(branch["branch_phase"], "queued")
        self.assertIsNone(branch["candidate_count"])

    def test_candidate_progress_batch_handles_all_states(self):
        first_key = b"first"
        second_key = b"second"
        missing_key = b"missing"
        queue = self._open_queue()
        queue.create_branch(first_key, 2, 4)
        queue.create_branch(second_key, 2, 4)
        queue._conn.execute(
            "INSERT INTO candidate_claims (branch_id, idx, done) VALUES (?, 0, 1)",
            (queue._intern_branch(first_key),),
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 3, "
            "one_level_erd_pruned_candidates = 3 "
            "WHERE branch_id = ?", (queue._intern_branch(second_key),),
        )
        progress = queue.candidate_progress_by_branch_keys(
            [first_key, second_key, missing_key]
        )
        self.assertEqual(queue.candidate_progress_by_branch_keys([]), {})
        self.assertEqual(progress[first_key]["completed_candidate_count"], 1)
        self.assertEqual(progress[second_key]["bulk_completed_candidate_count"], 3)
        self.assertEqual(progress[missing_key], {
            "completed_candidate_count": 0,
            "bulk_completed_candidate_count": 0,
            "one_level_erd_pruned_candidate_count": 0,
            "two_level_erd_pruned_candidate_count": 0,
        })
        queue.close()

    def test_cache_summary_and_dispatch(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        now = int(time.time())
        cache._conn.execute(
            "INSERT INTO branch_best_by_policy "
            "(branch_key, policy, answer_list_id, best_guess, best_score, updated_at) "
            "VALUES (?, ?, ?, 'salet', 2.0, ?)",
            (b"exact", ERD_ALL, cache.answer_list_id, now),
        )
        cache._conn.execute(
            "INSERT INTO branch_loss_by_policy "
            "(branch_key, policy, answer_list_id, loss_budget, updated_at) "
            "VALUES (?, ?, ?, 3, ?)",
            (b"loss", ERD_ALL, cache.answer_list_id, now),
        )
        self.assertEqual(cache.erd_report_summary(ERD_ALL, now - 1), {
            "exact_branch_count": 1,
            "recent_exact_branch_count": 1,
            "loss_branch_count": 1,
        })
        cache.close()
        report = collect_report(self.sources, ReportRequest())
        self.assertEqual(report["data"]["cache_summary"]["exact_branch_count"], 1)
        self.assertEqual(ReportRequest().report_kind, "auto")
        with self.assertRaisesRegex(ValueError, "unsupported report kind: hotspot"):
            collect_report(self.sources, ReportRequest("hotspot"))

    def test_cache_reference_resolves_finalized_branch_without_a_spine(self):
        branch_key = b"finalized branch"
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(branch_key, ERD_ALL, "salet", 2.0)
        reference = branch_reference(branch_key)
        queue = self._open_queue()
        resolved = resolve_branch_reference(queue, reference[:8], cache)
        self.assertEqual(bytes(resolved["branch_key"]), branch_key)
        self.assertIsNone(resolved["spine"])
        queue.close()
        cache.close()

    def test_branch_reference_migration_backfills_once(self):
        branch_key = b"pre-migration branch"
        connection = sqlite3.connect(self.cache_path)
        connection.execute(
            "CREATE TABLE schema_migrations "
            "(name TEXT PRIMARY KEY, completed_at INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE branch_best_by_policy "
            "(branch_key BLOB NOT NULL, policy TEXT NOT NULL, "
            "answer_list_id TEXT NOT NULL, best_guess TEXT NOT NULL, "
            "best_score REAL NOT NULL, updated_at INTEGER NOT NULL, "
            "PRIMARY KEY (branch_key, policy, answer_list_id))"
        )
        connection.execute(
            "CREATE TABLE branch_loss_by_policy "
            "(branch_key BLOB NOT NULL, policy TEXT NOT NULL, "
            "answer_list_id TEXT NOT NULL, loss_budget INTEGER NOT NULL, "
            "updated_at INTEGER NOT NULL, "
            "PRIMARY KEY (branch_key, policy, answer_list_id))"
        )
        connection.execute(
            "INSERT INTO branch_best_by_policy VALUES (?, 'policy', 'answers', "
            "'salet', 2.0, 1)", (branch_key,)
        )
        connection.commit()
        connection.close()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        migration_count = cache._conn.execute(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = 'add_branch_references'"
        ).fetchone()[0]
        indexes = {row[1] for row in cache._conn.execute(
            "PRAGMA index_list(branch_best_by_policy)"
        )}
        self.assertEqual(migration_count, 1)
        self.assertIn("idx_branch_best_by_policy_reference", indexes)
        self.assertEqual(cache._conn.execute(
            "SELECT branch_reference FROM branch_best_by_policy"
        ).fetchone()[0], branch_reference(branch_key))
        cache.close()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        self.assertEqual(cache._conn.execute(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE name = 'add_branch_references'"
        ).fetchone()[0], 1)
        cache.close()

    def test_cache_reference_reports_ambiguous_prefix_without_a_spine(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        first_key = second_key = None
        keys_by_prefix = {}
        for index in range(10000):
            branch_key = f"branch {index}".encode()
            prefix = branch_reference(branch_key)[:4]
            if prefix in keys_by_prefix:
                first_key, second_key = keys_by_prefix[prefix], branch_key
                break
            keys_by_prefix[prefix] = branch_key
        self.assertIsNotNone(first_key)
        cache.write(first_key, ERD_ALL, "salet", 2.0)
        cache.write_loss(second_key, ERD_ALL, 3)
        queue = unittest.mock.Mock()
        queue.branch_rows_for_reference_prefix.return_value = [{
            "branch_key": first_key,
            "spine": "salet -----",
        }]
        queue.row_spine_text.return_value = "salet -----"
        with self.assertRaisesRegex(ValueError, "ambiguous") as raised:
            resolve_branch_reference(queue, branch_reference(first_key)[:4], cache)
        self.assertEqual(len(raised.exception.candidates), 2)
        self.assertIsNone(raised.exception.candidates[1]["spine"])
        cache.close()


class SourceReportTest(unittest.TestCase):
    setUp = ReportModelTest.setUp
    tearDown = ReportModelTest.tearDown
    _open_queue = ReportModelTest._open_queue

    def _source_rows_for(self, word):
        report = collect_report(self.sources, ReportRequest(
            report_kind="sources",
            branch_target=parse_report_branch_target([word]),
        ))
        return {row["source_word"]: row for row in report["data"]["rows"]}

    def test_collapsed_report_is_one_row_per_request_whatever_the_branch_count(self):
        # The report's unit is the request: a root that spawned a hundred
        # branches is one row, not a hundred, and no branch rows are emitted
        # until a request is named.
        queue = self._open_queue()
        queue.add_pending_many([
            (ScoreCache.encode_subset(ANSWERS[:2] + [f"w{index:04d}"]), 3, 5,
             "salet", index)
            for index in range(40)
        ])
        queue.add_pending_many([
            (ScoreCache.encode_subset(ANSWERS[:2] + [f"x{index:04d}"]), 3, 3,
             "crane", index)
            for index in range(15)
        ])
        queue.close()

        report = collect_report(
            self.sources, ReportRequest(report_kind="sources"))

        data = report["data"]
        self.assertEqual(len(data["summary"]), 2)
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["matched_rows"], 0)
        rollups = {row["source_word"]: row for row in data["summary"]}
        self.assertEqual(rollups["salet"]["branch_count"], 40)
        self.assertEqual(rollups["salet"]["open_branch_count"], 40)
        self.assertEqual(rollups["salet"]["done_branch_count"], 0)
        self.assertEqual(rollups["crane"]["branch_count"], 15)
        # Naming one request is what opens its branches.
        self.assertEqual(len(self._source_rows_for("crane")), 1)

    def _queue_words(self, *rows):
        queue = self._open_queue()
        for index, (word, priority, count) in enumerate(rows):
            queue.add_pending_many([
                (ScoreCache.encode_subset(
                    ANSWERS[:2] + [f"{word}{item:04d}"]), 3, priority, word,
                 index * 100 + item)
                for item in range(count)
            ])
        queue.close()

    def _source_words(self, **filters):
        report = collect_report(self.sources, ReportRequest(
            report_kind="sources", filters=ReportFilters(**filters)))
        return report["data"]

    def test_source_state_filter_narrows_and_reports_what_it_hid(self):
        self._queue_words(("salet", 5, 3), ("crane", 1, 2))
        queue = self._open_queue()
        # CRANE's branches all finish, which resolves its memberships and
        # completes its only request, so the word reads complete.
        for item in range(2):
            queue.mark_done(
                ScoreCache.encode_subset(ANSWERS[:2] + [f"crane{item:04d}"]))
        queue.close()

        every = self._source_words()
        self.assertEqual(every["total_source_word_count"], 2)
        complete = self._source_words(source_states=("complete",))
        self.assertEqual(
            [row["source_word"] for row in complete["summary"]], ["crane"])
        queued = self._source_words(source_states=("queued",))
        self.assertEqual(
            [row["source_word"] for row in queued["summary"]], ["salet"])
        # The total is the unfiltered count, so a filtered report can say how
        # much it is hiding rather than looking like the whole queue.
        self.assertEqual(complete["total_source_word_count"], 2)
        self.assertEqual(
            complete["matched_source_word_count"], len(complete["summary"]))

    def test_source_sorts_order_by_the_column_named(self):
        self._queue_words(("salet", 5, 3), ("crane", 9, 1), ("nurdy", 1, 7))
        queue = self._open_queue()
        queue._conn.execute(
            "UPDATE source_work SET requested_at = CASE source_word "
            "WHEN 'salet' THEN 100 WHEN 'crane' THEN 300 "
            "WHEN 'nurdy' THEN 200 END"
        )
        queue._conn.commit()
        queue.close()

        # The default is the explicit ERD ordering.
        self.assertEqual(
            [row["source_word"] for row in self._source_words()["summary"]],
            [row["source_word"] for row in
             self._source_words(sort="erd")["summary"]])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="default")["summary"]],
            [row["source_word"] for row in
             self._source_words(sort="erd")["summary"]])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="priority")["summary"]],
            ["crane", "salet", "nurdy"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="word")["summary"]],
            ["crane", "nurdy", "salet"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="branches")["summary"]],
            ["nurdy", "salet", "crane"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="open")["summary"]],
            ["nurdy", "salet", "crane"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="requested")["summary"]],
            ["crane", "nurdy", "salet"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="age")["summary"]],
            ["salet", "nurdy", "crane"])

    def test_source_timing_fields_and_sorts_roll_up_finalized_branches(self):
        self._queue_words(("salet", 5, 1), ("crane", 3, 1), ("nurdy", 1, 1))
        queue = self._open_queue()
        salet_key = ScoreCache.encode_subset(ANSWERS[:2] + ["salet0000"])
        crane_key = ScoreCache.encode_subset(ANSWERS[:2] + ["crane0000"])
        queue.add_branch_finalize_log(
            salet_key, "SALET -----", 3, 3, 10, 40, 100, 1,
            total_bundle_wall_millis=2_000,
        )
        queue.add_branch_finalize_log(
            crane_key, "CRANE -----", 3, 3, 20, 30, 100, 1,
            total_bundle_wall_millis=5_000,
        )
        queue.mark_done(salet_key)
        queue.mark_done(crane_key)
        queue.close()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write_completed_source_summary("salet", ERD_ALL, 40, 30_000, 2_000)
        cache.write_completed_source_summary("crane", ERD_ALL, 30, 10_000, 5_000)
        cache.close()

        rows = {row["source_word"]: row for row in self._source_words()["summary"]}
        self.assertEqual(rows["salet"]["elapsed_millis"], 30_000)
        self.assertEqual(rows["salet"]["worker_millis"], 2_000)
        self.assertEqual(rows["crane"]["elapsed_millis"], 10_000)
        self.assertEqual(rows["crane"]["worker_millis"], 5_000)
        self.assertEqual(rows["crane"]["completed_at"], 30)
        self.assertIsNone(rows["nurdy"]["elapsed_millis"])
        self.assertIsNone(rows["nurdy"]["worker_millis"])
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="elapsed")["summary"]],
            ["salet", "crane", "nurdy"],
        )
        self.assertEqual(
            [row["source_word"] for row in
             self._source_words(sort="worker_time")["summary"]],
            ["crane", "salet", "nurdy"],
        )

    def test_source_report_keeps_erd_summary_shape_when_cache_unavailable(self):
        self._queue_words(("salet", 5, 1))
        unavailable_sources = replace(
            self.sources, answer_list_path="unused-answers"
        )

        report = collect_source_report(
            unavailable_sources, ReportRequest(report_kind="sources")
        )

        self.assertIn("error", report["sources"]["cache"])
        self.assertIn("erd_summary", report["data"]["summary"][0])
        self.assertIsNone(report["data"]["summary"][0]["erd_summary"])

    def test_requeued_source_hides_completed_run_timing(self):
        self._queue_words(("salet", 5, 1))
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write_completed_source_summary("salet", ERD_ALL, 160, 60_000, 2_000)
        cache.close()

        row = self._source_words()["summary"][0]

        self.assertEqual(row["state"], "queued")
        self.assertIsNone(row["completed_at"])
        self.assertIsNone(row["elapsed_millis"])
        self.assertIsNone(row["worker_millis"])

    def test_source_erd_summary_cache_invalidates_for_wal_writes(self):
        self.addCleanup(report_model._SOURCE_ERD_SUMMARY_CACHE.clear)
        self._queue_words(("nurdy", 5, 1))
        first = self._source_words()["summary"][0]["erd_summary"]
        self.assertEqual(first["state"], "pending")

        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        response_cache = ResponseCache(ANSWERS, score_cache=None)
        for _pattern_code, words in response_cache.group_words(
                "nurdy", ANSWERS).items():
            if words:
                cache.write(
                    ScoreCache.encode_subset(words), ERD_ALL, "salet", 1.0,
                    max_depth=1, solve_budget=GAME_GUESSES - 1,
                )
        cache._conn.commit()

        second = self._source_words()["summary"][0]["erd_summary"]

        self.assertEqual(second["state"], "complete")
        self.assertNotEqual(first, second)
        cache.close()

    def test_each_source_word_carries_its_own_erd(self):
        # A word's ERD is why it was queued.  It is derived from the word's
        # cached response groups, and kept once the whole tree is solved.
        # NURDY is the one of these that leaves a two-answer group, so it is
        # the one whose ERD needs the cache; the others partition this answer
        # list into singletons, which are solved by playing them.
        self._queue_words(("nurdy", 5, 1), ("crane", 3, 1))
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        response_cache = ResponseCache(ANSWERS, score_cache=None)
        now = int(time.time())
        for _pattern_code, words in response_cache.group_words(
                "crane", ANSWERS).items():
            if not words:
                continue
            cache._conn.execute(
                "INSERT OR REPLACE INTO branch_best_by_policy "
                "(branch_key, policy, answer_list_id, best_guess, best_score, "
                " max_depth, solve_budget, updated_at) "
                "VALUES (?, ?, ?, 'salet', 1.0, 1, ?, ?)",
                (ScoreCache.encode_subset(words), ERD_ALL,
                 cache.answer_list_id, GAME_GUESSES - 1, now),
            )
        cache._conn.commit()
        cache.close()

        rows = {row["source_word"]: row
                for row in self._source_words()["summary"]}

        # Every group of CRANE is solved, so its ERD is exact: the guess
        # itself, plus the mean of its groups -- and the all-green group costs
        # nothing, since that guess was the answer.  1 + (0+1+1+1)/4.
        crane = rows["crane"]["erd_summary"]
        self.assertEqual(crane["state"], "complete")
        self.assertEqual(crane["erd"], 1.75)
        self.assertEqual(crane["max_remaining_depth"], 2)
        # NURDY's two-answer group has nothing cached, so it reports how far
        # along it is rather than a number that would move under the reader.
        nurdy = rows["nurdy"]["erd_summary"]
        self.assertEqual(nurdy["state"], "pending")
        self.assertIsNone(nurdy["erd"])
        # Its two singleton groups are solved by playing them; the
        # two-answer group is the one still outstanding.
        self.assertEqual(nurdy["resolved_group_count"], 2)
        self.assertEqual(nurdy["response_group_count"], 3)

        # The solved word's fold is kept, so a later page reads it back
        # instead of refolding; the unsolved one is not, since its value can
        # still change.
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        root_branch_key = ScoreCache.encode_subset(ANSWERS)
        stored = cache.read_candidate_erd(root_branch_key, "crane", ERD_ALL)
        self.assertEqual(stored["erd"], 1.75)
        self.assertEqual(stored["max_remaining_depth"], 2)
        self.assertIsNone(
            cache.read_candidate_erd(root_branch_key, "nurdy", ERD_ALL))
        cache.close()

    def test_branch_totals_count_a_shared_branch_once(self):
        # Two different words owning one branch is the case the report exists
        # to show, and it is exactly where summing per-word counts goes wrong.
        shared_key = ScoreCache.encode_subset(ANSWERS[:2])
        solo_key = ScoreCache.encode_subset(ANSWERS[2:4])
        queue = self._open_queue()
        queue.add_pending_many([(shared_key, 2, 5, "salet", 0)])
        queue.add_pending_many([(shared_key, 2, 3, "crane", 0),
                                (solo_key, 2, 3, "crane", 1)])
        queue.close()

        data = self._source_words()

        self.assertEqual(
            sum(row["branch_count"] for row in data["summary"]), 3)
        self.assertEqual(data["matched_branch_count"], 2)
        self.assertEqual(data["matched_open_branch_count"], 2)
        # The totals follow the filter, so they describe the words on screen.
        narrowed = self._source_words(source_states=("complete",))
        self.assertEqual(narrowed["matched_branch_count"], 0)
        self.assertEqual(narrowed["matched_open_branch_count"], 0)

    def test_source_offset_pages_through_the_words(self):
        self._queue_words(*[(word, 5, 1) for word in
                            ("salet", "crane", "nurdy", "khaki", "scope")])

        pages = [
            self._source_words(sort="word", limit=2, source_offset=offset)
            for offset in (0, 2, 4, 6)
        ]

        self.assertEqual(
            [[row["source_word"] for row in page["summary"]] for page in pages],
            [["crane", "khaki"], ["nurdy", "salet"], ["scope"], []],
        )
        # Every page reports the same matched total and its own start, which
        # is what lets a client know it has more to page through.
        for offset, page in zip((0, 2, 4, 6), pages):
            self.assertEqual(page["matched_source_word_count"], 5)
            self.assertEqual(page["source_word_offset"], offset)
        # Paging is over the matched words, so a state filter repaginates
        # rather than leaving holes where the filtered-out words were.
        filtered = self._source_words(
            sort="word", limit=2, source_offset=0, source_states=("queued",))
        self.assertEqual(filtered["matched_source_word_count"], 5)

    def test_group_rollups_count_a_shared_branch_once(self):
        # Same defect as the report's own totals, one level down: two words in
        # the same group owning one branch must not count it twice.
        shared_key = ScoreCache.encode_subset(ANSWERS[:2])
        solo_key = ScoreCache.encode_subset(ANSWERS[2:4])
        queue = self._open_queue()
        queue.add_pending_many([(shared_key, 2, 5, "salet", 0)])
        queue.add_pending_many([(shared_key, 2, 5, "crane", 0),
                                (solo_key, 2, 5, "crane", 1)])
        queue.close()

        data = self._source_words(group_by="state")

        group = data["summary_groups"][0]
        self.assertEqual(group["rollup"]["source_word_count"], 2)
        self.assertEqual(
            sum(row["branch_count"] for row in group["rows"]), 3)
        self.assertEqual(group["rollup"]["branch_count"], 2)
        self.assertEqual(group["rollup"]["open_branch_count"], 2)
        self.assertEqual(group["rollup"]["done_branch_count"], 0)

    def test_branch_rows_page_like_the_words_do(self):
        queue = self._open_queue()
        queue.add_pending_many([
            (ScoreCache.encode_subset(ANSWERS[:2] + [f"w{index:04d}"]), 3, 5,
             "salet", index)
            for index in range(5)
        ])
        queue.close()
        salet = parse_report_branch_target(["salet"])

        pages = [
            collect_report(self.sources, ReportRequest(
                report_kind="sources", branch_target=salet,
                filters=ReportFilters(limit=2, branch_row_offset=offset),
            ))["data"]
            for offset in (0, 2, 4, 6)
        ]

        self.assertEqual([len(page["rows"]) for page in pages], [2, 2, 1, 0])
        for offset, page in zip((0, 2, 4, 6), pages):
            # Every page names the same matched total and its own start, so a
            # client knows there is more to page through.
            self.assertEqual(page["matched_rows"], 5)
            self.assertEqual(page["branch_row_offset"], offset)
        # The pages partition the branch rows: no row on two pages, none lost.
        seen = [row["branch_key_hex"] for page in pages for row in page["rows"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 5)

    def test_source_grouping_buckets_words_with_their_own_rollup(self):
        self._queue_words(("salet", 5, 3), ("crane", 5, 1), ("nurdy", 1, 7))

        groups = self._source_words(group_by="priority")["summary_groups"]

        self.assertEqual([group["label"] for group in groups],
                         ["priority 5", "priority 1"])
        self.assertEqual(groups[0]["rollup"]["source_word_count"], 2)
        # The rollup sums the group's rows, so a collapsed group still says
        # how much work it holds.
        self.assertEqual(groups[0]["rollup"]["branch_count"], 4)
        self.assertEqual(groups[0]["rollup"]["open_branch_count"], 4)
        self.assertEqual(groups[1]["rollup"]["branch_count"], 7)
        self.assertEqual(
            [row["source_word"] for row in groups[1]["rows"]], ["nurdy"])
        # Every word lands in exactly one group.
        self.assertEqual(
            sum(len(group["rows"]) for group in groups),
            len(self._source_words()["summary"]))

    def test_source_time_grouping_boundaries(self):
        generated_at = 1_787_270_400  # 21 Aug 2026 00:00 UTC (Friday)
        base_row = {"state": "complete", "worker_count": 0,
                    "requested_priority": 0, "completed_at": generated_at,
                    "elapsed_millis": 0, "worker_millis": 0,
                    "requested_at": generated_at}
        completed = lambda seconds_ago: _source_word_group_key(
            {**base_row, "completed_at": generated_at - seconds_ago},
            "completed", generated_at)[1]
        duration = lambda field, millis: _source_word_group_key(
            {**base_row, field: millis},
            "elapsed" if field == "elapsed_millis" else "worker_time",
            generated_at)[1]

        self.assertEqual(completed(0), "today")
        self.assertEqual(completed(24 * 60 * 60), "earlier this week")
        self.assertEqual(completed(7 * 24 * 60 * 60), "earlier this month")
        self.assertEqual(completed(31 * 24 * 60 * 60), "earlier this year")
        self.assertEqual(completed(366 * 24 * 60 * 60), "older")
        self.assertEqual(
            [duration("elapsed_millis", millis) for millis in
             (0, 60 * 60 * 1000, 24 * 60 * 60 * 1000,
              7 * 24 * 60 * 60 * 1000, 30 * 24 * 60 * 60 * 1000)],
            ["[0, 1 hour)", "[1 hour, 1 day)", "[1 day, 1 week)",
             "[1 week, 1 month)", "[1 month, ∞)"],
        )
        self.assertEqual(
            _source_word_group_key(
                {**base_row, "requested_at": generated_at - 24 * 60 * 60},
                "requested", generated_at)[1],
            "[1 day, 1 week)",
        )
        self.assertEqual(
            _source_word_group_key(
                {**base_row, "completed_at": None, "elapsed_millis": None,
                 "worker_millis": None},
                "completed", generated_at)[1],
            "not completed",
        )

    def test_source_only_filters_and_sorts_are_rejected_elsewhere(self):
        for filters, message in (
            ({"source_states": ("queued",)}, "source_state requires"),
            ({"source_offset": 2}, "source_offset requires"),
            ({"sort": "branches"}, "requires a source report"),
        ):
            with self.subTest(filters=filters):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report_request(ReportRequest(
                        report_kind="queue", filters=ReportFilters(**filters)))
        with self.assertRaisesRegex(ValueError, "source reports must be"):
            validate_report_request(ReportRequest(
                report_kind="sources", filters=ReportFilters(sort="nodes")))
        validate_report_request(ReportRequest(
            report_kind="sources", filters=ReportFilters(sort="age")))
        for group_by in ("completed", "elapsed", "worker_time", "requested"):
            with self.subTest(group_by=group_by):
                validate_report_request(ReportRequest(
                    report_kind="sources", filters=ReportFilters(group_by=group_by)))
        with self.assertRaisesRegex(ValueError, "source reports must be"):
            validate_report_request(ReportRequest(
                report_kind="sources",
                filters=ReportFilters(group_by="cache_state")))

    def test_one_word_queued_twice_merges_into_one_row(self):
        # Source work is keyed by (word, priority), so the same word queued at
        # a second priority is a second request.  The report is per word: the
        # two fold into one row, the branch they both own is counted once, and
        # the priority reported is the one that actually schedules.
        shared_key = ScoreCache.encode_subset(ANSWERS[:2])
        solo_key = ScoreCache.encode_subset(ANSWERS[2:4])
        queue = self._open_queue()
        queue.add_pending_many([(shared_key, 2, 1, "salet", 0)])
        queue.add_pending_many([(shared_key, 2, 7, "salet", 0),
                                (solo_key, 2, 7, "salet", 1)])
        queue.close()

        report = collect_report(
            self.sources, ReportRequest(report_kind="sources"))

        self.assertEqual(len(report["data"]["summary"]), 1)
        row = report["data"]["summary"][0]
        self.assertEqual(row["source_word"], "salet")
        self.assertEqual(row["request_count"], 2)
        self.assertEqual(row["requested_priority"], 7)
        self.assertEqual(row["branch_count"], 2)
        self.assertEqual(row["open_branch_count"], 2)
        self.assertEqual(row["worker_count"], 0)

    def test_source_report_exposes_multiple_owners_with_requested_and_effective_priority(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        queue.add_pending_many([(branch_key, 2, 9, "crane", 1)])
        queue.close()

        report = collect_report(
            self.sources, ReportRequest(report_kind="sources"))

        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertEqual(len(report["data"]["summary"]), 2)
        # Each owner's own row comes from naming that owner's word; the shared
        # branch is still never reduced to one owner's claim.
        rows = {**self._source_rows_for("salet"), **self._source_rows_for("crane")}
        self.assertEqual(set(rows), {"salet", "crane"})
        self.assertEqual(rows["salet"]["requested_priority"], 1)
        self.assertEqual(rows["crane"]["requested_priority"], 9)
        # Both rows report the branch's own effective (MAX) priority, not
        # each request's own requested priority — a shared branch is never
        # reduced to a single owner's claim.
        self.assertEqual(rows["salet"]["branch_effective_priority"], 9)
        self.assertEqual(rows["crane"]["branch_effective_priority"], 9)
        self.assertTrue(rows["salet"]["is_shared"])
        self.assertEqual(rows["salet"]["owner_count"], 2)
        self.assertEqual(rows["salet"]["branch_status"], "pending")
        self.assertEqual(rows["salet"]["branch_phase"], "queued")

    def test_source_report_filters_by_word_but_keeps_global_shared_detection(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        solo_key = ScoreCache.encode_subset(ANSWERS[2:4])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        queue.add_pending_many([(branch_key, 2, 9, "crane", 1)])
        queue.add_pending_many([(solo_key, 2, 1, "salet", 2)])
        queue.close()

        report = collect_source_report(
            self.sources,
            ReportRequest(
                report_kind="sources",
                branch_target=parse_report_branch_target(["salet"]),
            ),
        )

        rows = report["data"]["rows"]
        self.assertEqual({row["source_word"] for row in rows}, {"salet"})
        shared_row = next(r for r in rows
                          if bytes.fromhex(r["branch_key_hex"]) == branch_key)
        # The word filter drops crane's own row, but shared detection still
        # reflects crane's (unfiltered) ownership of the same branch.
        self.assertTrue(shared_row["is_shared"])
        self.assertEqual(shared_row["owner_count"], 2)
        solo_row = next(r for r in rows
                        if bytes.fromhex(r["branch_key_hex"]) == solo_key)
        self.assertFalse(solo_row["is_shared"])

    def test_shared_branch_count_spans_matched_rows_not_the_limited_window(self):
        shared_key = ScoreCache.encode_subset(ANSWERS[:2])
        solo_key = ScoreCache.encode_subset(ANSWERS[2:4])
        queue = self._open_queue()
        # SALET owns a solo branch and one shared with NURDY, in that order, so
        # a limit of one leaves the shared branch outside the returned rows.
        queue.add_pending_many([(solo_key, 2, 1, "salet", 0),
                                (shared_key, 2, 1, "salet", 1)])
        queue.add_pending_many([(shared_key, 2, 1, "nurdy", 2)])
        queue.close()

        salet = ReportRequest(
            report_kind="sources",
            branch_target=parse_report_branch_target(["salet"]),
        )
        full = collect_report(self.sources, salet)
        limited = collect_report(self.sources, ReportRequest(
            report_kind="sources", branch_target=salet.branch_target,
            filters=ReportFilters(limit=1),
        ))

        self.assertEqual(full["data"]["matched_rows"], 2)
        self.assertEqual(full["data"]["shared_branch_count"], 1)
        # The limit truncates the returned rows only.  Both counts describe
        # every matched row, so a shared branch whose rows all fall past the
        # limit is still counted as shared.
        self.assertEqual(len(limited["data"]["rows"]), 1)
        self.assertFalse(limited["data"]["rows"][0]["is_shared"])
        self.assertEqual(limited["data"]["matched_rows"], 2)
        self.assertEqual(limited["data"]["shared_branch_count"], 1)

    def test_workers_report_exposes_scheduling_role_and_source_work_id(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        source_work_id = queue.source_work_rows()[0]["source_work_id"]
        now = int(time.time())
        queue.heartbeat("worker-1", 1, branch_key, 2, now, 0,
                        source_work_id=source_work_id,
                        scheduling_role="preferred")
        queue.close()

        report = collect_report(
            self.sources, ReportRequest(report_kind="workers"))

        worker = report["data"]["rows"][0]
        self.assertEqual(worker["answer_count"], 2)
        self.assertEqual(worker["source_work_id"], source_work_id)
        self.assertEqual(worker["scheduling_role"], "preferred")


class RootProgressReportTest(unittest.TestCase):
    setUp = ReportModelTest.setUp
    tearDown = ReportModelTest.tearDown
    _open_queue = ReportModelTest._open_queue

    NOW = 1_800_000_000

    def _finalize(self, queue, spine, n_words, nodes, wall_millis,
                  created_at, finalized_at, epoch=0):
        queue.add_branch_finalize_log(
            ScoreCache.encode_subset(ANSWERS[:1]), spine, n_words, 3,
            created_at, finalized_at, nodes, 1,
            total_bundle_wall_millis=wall_millis)
        queue._conn.execute(
            "UPDATE telemetry.branch_finalize_log SET epoch = ? "
            "WHERE spine = ?", (epoch, spine))

    def _open_branch(self, queue, spine, n_words, created_at,
                     answer_slice=slice(0, 2)):
        queue.create_branch(
            ScoreCache.encode_subset(ANSWERS[answer_slice]), n_words, 5,
            source_word="salet", spine=spine)
        queue._conn.execute(
            "UPDATE active_branches SET created_at = ? WHERE spine = ?",
            (created_at, spine))

    def _request(self, epoch=None, branch_target=("salet",)):
        return ReportRequest(
            report_kind="root_progress",
            branch_target=parse_report_branch_target(list(branch_target)),
            epoch=epoch)

    def test_rollup_attributes_descendant_cost_to_its_top_level_group(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 100, 1000, 10, 20)
        self._finalize(queue, "SALET -y--- CRANE -----", 1, 900, 9000, 12, 30)
        self._finalize(queue, "SALET ----- NURDY -----", 1, 7, 70, 15, 25)
        queue.close()

        report = collect_report(self.sources, self._request())

        rows = {row["pattern"]: row for row in report["data"]["response_groups"]}
        # Both SALET -y--- rows fold into the one group, deepest included.
        self.assertEqual(rows["-y---"]["search_node_count"], 1000)
        self.assertEqual(rows["-y---"]["branch_count"], 2)
        self.assertEqual(rows["-----"]["search_node_count"], 7)
        # Every response group appears, including ones never worked.
        self.assertEqual(len(rows), 4)
        self.assertFalse(rows["-y-y-"]["started"])
        self.assertEqual(rows["-y-y-"]["search_node_count"], 0)
        self.assertIsNone(rows["-y-y-"]["elapsed_millis"])

    def test_rollup_reports_elapsed_and_worker_time_separately(self):
        queue = self._open_queue()
        # One branch spanning 100s of wall clock that consumed 400s of
        # worker-time: four workers on it, not a 400s stretch of the clock.
        self._finalize(queue, "SALET -y---", 1, 5, 400_000, 1_000, 1_100)
        queue.close()

        report = collect_report(self.sources, self._request())

        row = next(row for row in report["data"]["response_groups"]
                   if row["pattern"] == "-y---")
        self.assertEqual(row["elapsed_millis"], 100_000)
        self.assertEqual(row["wall_millis"], 400_000)

    def test_rollup_is_fenced_by_epoch(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 100, 10, 10, 20, epoch=0)
        self._finalize(queue, "SALET -----", 1, 500, 10, 10, 20, epoch=1)
        queue.close()

        rows = {row["pattern"]: row for row
                in collect_report(self.sources,
                                  self._request(epoch=1))["data"]["response_groups"]}
        self.assertTrue(rows["-----"]["started"])
        self.assertFalse(rows["-y---"]["started"])

    def test_work_started_is_distinct_from_when_the_word_was_requested(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        requested_at = queue.source_work_rows()[0]["requested_at"]
        # Work opens well after the request: the root waits behind others.
        self._finalize(queue, "SALET -y---", 1, 100, 10,
                       requested_at + 86_400, requested_at + 90_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        self.assertEqual(data["totals"]["requested_at"], requested_at)
        self.assertEqual(data["work_started_at"], requested_at + 86_400)
        self.assertNotEqual(data["work_started_at"],
                            data["totals"]["requested_at"])

    def test_request_time_stamped_after_the_work_it_asked_for_is_dropped(self):
        # A queue rebuild restamps every source_work row with its own clock
        # while the branches keep their true creation times, which leaves the
        # request looking later than the work.  Reporting that stamp would
        # claim the swarm started before the word was asked for.
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        requested_at = queue.source_work_rows()[0]["requested_at"]
        self._finalize(queue, "SALET -y---", 1, 100, 10,
                       requested_at - 86_400, requested_at - 80_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        self.assertEqual(data["work_started_at"], requested_at - 86_400)
        self.assertIsNone(data["totals"]["requested_at"])

    def test_estimate_excludes_stalled_branches_from_both_rate_and_remainder(self):
        estimate = _root_progress_estimate([
            # Progressing: 400 of 1,000 candidates left, 100/day observed.
            {"candidate_count": 1000, "done_candidate_count": 600,
             "recent_done_candidate_count": 100, "created_at": 0},
            # Stalled at 99%: waiting on published sub-branches, not on its
            # own candidates.  Its remainder must not be charged to the
            # rate above, which it is not producing.
            {"candidate_count": 1000, "done_candidate_count": 990,
             "recent_done_candidate_count": 0, "created_at": 0},
        ], 86400, 86400)

        self.assertEqual(estimate["remaining_candidate_count"], 400)
        self.assertEqual(estimate["candidates_per_day"], 100)
        self.assertEqual(estimate["estimated_seconds"], 4 * 86400)
        self.assertEqual(estimate["stalled_branch_count"], 1)
        self.assertEqual(estimate["stalled_remaining_candidate_count"], 10)

    def test_estimate_uses_the_available_sample_span_until_the_window_fills(self):
        estimate = _root_progress_estimate([
            {"candidate_count": 1000, "done_candidate_count": 100,
             "recent_done_candidate_count": 100, "created_at": 1000},
        ], 86400, 1600)

        self.assertEqual(estimate["sample_duration_seconds"], 600)
        self.assertTrue(estimate["provisional"])
        self.assertEqual(estimate["candidates_per_day"], 14_400)
        self.assertEqual(estimate["estimated_seconds"], 5400)

    def test_estimate_is_absent_when_nothing_completed_in_the_window(self):
        self.assertIsNone(_root_progress_estimate([
            {"candidate_count": 1000, "done_candidate_count": 10,
             "recent_done_candidate_count": 0, "created_at": 0},
        ], 86400, 86400))

    def test_root_progress_requires_a_word_target(self):
        with self.assertRaises(ValueError):
            validate_report_request(ReportRequest(
                report_kind="root_progress",
                branch_target=parse_report_branch_target(None)))

    def test_root_progress_accepts_a_spine_of_more_than_one_guess(self):
        # The rollup scopes telemetry by spine prefix and a longer spine is
        # simply a longer prefix, so "why is CRANE --g-- SALET taking so long"
        # is the same question at a greater guess_depth.
        validate_report_request(ReportRequest(
            report_kind="root_progress",
            branch_target=parse_report_branch_target(
                ["crane", "--g--", "salet"])))

    def test_deeper_spine_scopes_the_rollup_to_its_own_subtree(self):
        # SALET played as a root and SALET played after CRANE reach different
        # branches.  Sharing a trailing word must not merge their work.
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "crane", 0)])
        self._finalize(queue, "CRANE --g-- SALET -y---", 1, 700, 70, 10, 20)
        self._finalize(queue, "SALET -y---", 1, 100, 10, 10, 20)
        queue.close()

        deeper = collect_report(self.sources, self._request(
            branch_target=["crane", "--g--", "salet"]))["data"]
        root = collect_report(self.sources, self._request())["data"]

        self.assertEqual(deeper["spine_prefix"], "CRANE --g-- SALET")
        self.assertEqual(deeper["totals"]["search_node_count"], 700)
        self.assertEqual(root["totals"]["search_node_count"], 100)

    def test_group_with_only_open_branches_counts_as_started(self):
        # The group is being worked right now and has finalized nothing.
        # Reading it as untouched would hide exactly the state this report
        # exists to show.
        queue = self._open_queue()
        self._open_branch(queue, "SALET -y---", 1, 5_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        row = next(row for row in data["response_groups"]
                   if row["pattern"] == "-y---")
        self.assertTrue(row["started"])
        self.assertEqual(row["open_branch_count"], 1)
        # Nothing has finalized, so cost is genuinely zero-so-far and the
        # finalize-derived span is unknown, not zero.
        self.assertEqual(row["branch_count"], 0)
        self.assertEqual(row["search_node_count"], 0)
        self.assertIsNone(row["elapsed_millis"])
        self.assertEqual(data["totals"]["started_response_group_count"], 1)

    def test_open_branches_can_start_the_clock_before_any_finalization(self):
        queue = self._open_queue()
        self._open_branch(queue, "SALET -y---", 1, 5_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        self.assertEqual(data["work_started_at"], 5_000)

    def test_open_branch_predating_a_finalization_moves_work_start_earlier(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -----", 1, 100, 10, 9_000, 9_500)
        self._open_branch(queue, "SALET -y---", 1, 5_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        self.assertEqual(data["work_started_at"], 5_000)

    def test_open_and_finalized_branches_fold_into_one_group(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 400, 40, 6_000, 6_500)
        self._open_branch(queue, "SALET -y--- CRANE -----", 1, 7_000)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]

        row = next(row for row in data["response_groups"]
                   if row["pattern"] == "-y---")
        self.assertEqual(row["branch_count"], 1)
        self.assertEqual(row["open_branch_count"], 1)
        self.assertEqual(row["search_node_count"], 400)
        self.assertEqual(data["totals"]["started_response_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
