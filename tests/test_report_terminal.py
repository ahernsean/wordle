"""Tests for terminal rendering and refresh of shared swarm reports."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
import random
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cache_sqlite import ScoreCache, branch_reference
import erd_search
from erd_queue import ERDQueue, encode_subset
import report_terminal
from report_model import ReportFilters, parse_report_branch_target
from report_terminal import DisplayOrder, WatchSession, render_overview, render_report
from wordle_engine import ERD_ALL


def overview_report():
    return {
        "schema_version": 3,
        "report_kind": "overview",
        "generated_at": 1000,
        "branch_target": None,
        "filters": {},
        "sources": {
            "queue": {
                "path": "queue.sqlite3", "ok": True, "error": None,
                "epoch": 4, "label": "packed", "git_sha": "abcdef12",
            },
            "telemetry": {
                "path": "telemetry.sqlite3", "ok": True, "error": None,
            },
            "cache": {
                "path": "cache.sqlite3", "ok": True, "error": None,
            },
        },
        "data": {
            "disk": {
                "total_bytes": 400 * 2 ** 30,
                "used_bytes": 50 * 2 ** 30,
                "available_bytes": 350 * 2 ** 30,
                "used_fraction": 0.125,
                "queue_wal_bytes": 2 ** 30,
                "fill_rate_bytes_per_second": None,
                "warning_fraction": 0.8,
                "stop_fraction": 0.9,
            },
            "queue_counts": {
                "pending_branch_count": 12,
                "evaluating_user_branch_count": 1,
                "evaluating_cooperative_branch_count": 0,
                "finalizing_branch_count": 0,
                "done_branch_count": 20,
            },
            "cache_summary": {
                "exact_branch_count": 200,
                "recent_exact_branch_count": 5,
                "loss_branch_count": 3,
            },
            "worker_totals": {
                "cache_hit_count": 80,
                "cache_miss_count": 20,
                "solved_evaluation_count": 40,
                "erd_cutoff_evaluation_count": 50,
                "remaining_depth_pruned_evaluation_count": 10,
                "hint_lookup_count": 0,
                "hint_hit_count": 0,
                "hint_accepted_count": 0,
                "hint_rejected_count": 0,
                "hint_inline_placement_count": 0,
                "hint_inline_win_count": 0,
            },
            "branches": [{
                "branch_reference": "0123456789ab",
                "branch_key_hex": "010203",
                "branch_status": "evaluating",
                "branch_worker_status": "active",
                "raw_status": "in_progress",
                "answer_count": 33,
                "candidate_count": 100,
                "completed_candidate_count": 25,
                "bulk_completed_candidate_count": 5,
                "one_level_erd_pruned_candidate_count": 4,
                "two_level_erd_pruned_candidate_count": 1,
                "priority": 10,
                "is_cooperative": False,
                "source_word": "salet",
                "source_pattern": "-----",
                "best_guess": "crane",
                "best_guess_is_answer": True,
                "best_erd": 2.25,
                "best_max_remaining_depth": 3,
                "budget": 4,
                "guess_depth": 2,
                "spine": [
                    {"word": "salet", "pattern": "-----", "word_is_answer": True},
                    {"word": "crane", "pattern": "y----", "word_is_answer": True},
                ],
                "worker_count": 1,
                "created_at": 900,
                "search_node_count": 12345,
                "ceiling": None,
            }],
            "workers": [{
                "worker_id": "worker-2",
                "worker_number": "2",
                "pid": 42,
                "updated_at": 995,
                "is_live": True,
                "branch_reference": "0123456789ab",
                "branch_key_hex": "010203",
                "candidate_index": 7,
                "claim_started_at": 990,
                "completed_claim_count": 12,
                "current_candidate": "nurdy",
                "current_candidate_is_answer": True,
                "current_max_guess_depth": 5,
                "current_node_count": 900,
                "nodes_per_second": 45.5,
                "descent": [],
                "cache_hit_count": 80,
                "cache_miss_count": 20,
                "solved_evaluation_count": 40,
                "erd_cutoff_evaluation_count": 50,
                "remaining_depth_pruned_evaluation_count": 10,
                "best_guess": "crane",
                "best_erd": 2.25,
                "bound_erd": 2.5,
            }],
        },
    }


def view_args(**overrides):
    values = {
        "watch": None,
        "format": "text",
        "no_color": False,
        "queue_path": "queue.sqlite3",
        "cache_path": "cache.sqlite3",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeInput(io.StringIO):
    def __init__(self, text="", tty=False):
        super().__init__(text)
        self.tty = tty

    def isatty(self):
        return self.tty

    def fileno(self):
        return 10


class FakeOutput(io.StringIO):
    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class OverviewRendererTest(unittest.TestCase):
    def test_text_contains_report_semantics_without_ansi(self):
        output = render_overview(overview_report(), color=False, width=100)
        self.assertIn("sources ok", output)
        self.assertIn("epoch=4 packed revision=abcdef12", output)
        self.assertIn("Disk: 50.0G/400G (12%)  queue WAL 1.0G", output)
        self.assertIn("Cache: exact 200", output)
        self.assertIn("Queue: pending 12", output)
        self.assertIn("@0123", output)
        self.assertNotIn("@0123456789ab", output)
        self.assertIn("GuessD", output)
        self.assertIn("25/100", output)
        self.assertNotIn("30/100", output)
        self.assertIn("MaxRD", output)
        self.assertIn("w2", output)
        self.assertNotIn("worker-2", output)
        self.assertIn("NURDY*", output)
        self.assertIn("CRANE*/2.250", output)
        self.assertNotIn("guess_depth=", output)
        self.assertNotIn("worker=", output)
        self.assertNotIn("\033", output)

    def test_header_compacts_healthy_sources_to_one_line(self):
        output = render_overview(overview_report(), color=False, width=100)
        lines = output.splitlines()
        blank_index = lines.index("")
        header_lines = lines[:blank_index]
        self.assertEqual(len(header_lines), 3)
        self.assertNotIn("queue: ok", output)
        self.assertNotIn("telemetry", output)

    def test_worker_rows_have_no_repeated_headers(self):
        report = overview_report()
        second_branch = deepcopy(report["data"]["branches"][0])
        second_branch.update({
            "branch_reference": "bbbbbbbbbbbb", "branch_key_hex": "bbbb",
        })
        second_worker = deepcopy(report["data"]["workers"][0])
        second_worker.update({
            "worker_id": "worker-3", "worker_number": "3",
            "branch_reference": "bbbbbbbbbbbb", "branch_key_hex": "bbbb",
        })
        report["data"]["branches"].append(second_branch)
        report["data"]["workers"].append(second_worker)
        output = render_overview(report, color=False, width=100)
        self.assertEqual(output.count("GuessD"), 1)
        self.assertNotIn("Worker", output)
        self.assertNotIn("Current", output)
        self.assertIn("w2", output)
        self.assertIn("w3", output)

    def test_renders_the_model_worker_state_in_other_workers(self):
        # The terminal renders whatever state the model set; it no longer
        # re-derives it, so a transitioning worker shows as "trans".
        report = overview_report()
        stray = deepcopy(report["data"]["workers"][0])
        stray.update({
            "worker_id": "worker-7", "worker_number": "7",
            "branch_reference": "cccccccccccc", "branch_key_hex": "cccc",
            "state": "transitioning",
        })
        report["data"]["workers"].append(stray)
        output = render_overview(report, color=False, width=100)
        self.assertIn("Other workers", output)
        worker_line = next(
            line for line in output.splitlines() if "w7" in line
        )
        self.assertIn("trans", worker_line)

    def test_working_worker_under_its_branch_shows_model_state(self):
        report = overview_report()
        report["data"]["workers"][0]["state"] = "working"
        output = render_overview(report, color=False, width=100)
        worker_line = next(
            line for line in output.splitlines() if "w2" in line
        )
        self.assertIn("working", worker_line)
        self.assertNotIn("trans", worker_line)

    def test_narrow_rendering_respects_width(self):
        output = render_overview(overview_report(), color=False, width=50)
        self.assertTrue(all(len(line) <= 50 for line in output.splitlines()))
        self.assertIn("@0123", output)
        self.assertIn("w2", output)
        self.assertIn("Ref", output)
        self.assertIn("Status", output)

    def test_branch_table_shows_status_abbreviated(self):
        report = overview_report()
        branch = report["data"]["branches"][0]
        branch["branch_status"] = "finalizing"
        branch["branch_worker_status"] = "active"
        output = render_overview(report, color=False, width=100)
        branch_line = next(
            line for line in output.splitlines() if "@0123" in line
        )
        self.assertIn("final", branch_line)
        self.assertNotIn("finalizing", branch_line)

    def test_progress_is_highlighted_by_cell_rules(self):
        previous = overview_report()
        current = deepcopy(previous)
        current["data"]["branches"][0]["completed_candidate_count"] += 1
        current["data"]["workers"][0]["current_candidate"] = "slate"
        output = render_overview(
            current, previous_report=previous, color=True, width=100
        )
        self.assertIn(report_terminal.GREEN + "26/100", output)
        self.assertIn(report_terminal.GREEN + "SLATE", output)
        self.assertNotIn(report_terminal.RED, output)

    def test_improved_best_erd_highlights_best_cell(self):
        previous = overview_report()
        current = deepcopy(previous)
        current["data"]["branches"][0]["best_erd"] = 2.125
        output = render_overview(
            current, previous_report=previous, color=True, width=100
        )
        self.assertIn(report_terminal.GREEN + "CRANE*/2.125", output)
        self.assertNotIn(report_terminal.GREEN + "26/100", output)

    def test_stalled_worker_rate_is_highlighted_red(self):
        previous = overview_report()
        current = deepcopy(previous)
        current["data"]["workers"][0]["nodes_per_second"] = 0
        output = render_overview(
            current, previous_report=previous, color=True, width=100
        )
        self.assertIn(report_terminal.RED + "0/s", output)

    def test_ticking_timestamps_alone_are_not_highlighted(self):
        previous = overview_report()
        current = deepcopy(previous)
        current["generated_at"] += 1
        current["data"]["workers"][0]["updated_at"] += 1
        output = render_overview(
            current, previous_report=previous, color=True, width=100,
            display_order=DisplayOrder(),
        )
        self.assertNotIn(report_terminal.GREEN, output)
        self.assertNotIn(report_terminal.RED, output)

    def test_stale_and_dead_workers_have_persistent_warning_colors(self):
        stale = overview_report()
        stale["data"]["workers"][0]["updated_at"] = 979
        stale_output = render_overview(stale, color=True, width=100)
        self.assertIn(
            report_terminal.AMBER + "    w2", stale_output
        )

        dead = deepcopy(stale)
        dead["data"]["workers"][0]["is_live"] = False
        dead_output = render_overview(dead, color=True, width=100)
        self.assertIn(
            report_terminal.RED + "  w2", dead_output
        )

    def test_adaptive_columns_cover_phone_and_wide_widths(self):
        report = overview_report()
        branch = report["data"]["branches"][0]
        branch["candidate_count"] = 12972
        branch["completed_candidate_count"] = 12616
        branch["spine"] = branch["spine"] * 5
        report["data"]["workers"][0].update({
            "worker_id": "worker-12", "worker_number": "12",
        })

        # The Ref column holds an eight-character reference plus its "@", so
        # each threshold below is the narrowest width at which the columns
        # listed still fit.
        expected_branch_headings = {
            54: ("Ref", "GuessD", "Status", "Done", "W", "Ans"),
            59: ("Ref", "GuessD", "Status", "Done", "W", "Ans"),
            63: ("Ref", "GuessD", "Status", "Done", "W", "Ans", "ERD1/2"),
            64: ("Ref", "GuessD", "Status", "Done", "W", "Ans", "ERD1/2"),
            83: (
                "Ref", "GuessD", "Status", "Done", "W", "Ans", "ERD1/2",
                "Best", "MaxRD",
            ),
            84: (
                "Ref", "GuessD", "Status", "Done", "W", "Ans", "ERD1/2",
                "Best", "MaxRD",
            ),
            124: (
                "Ref", "GuessD", "Status", "Done", "W", "Ans", "ERD1/2",
                "Best", "MaxRD", "ETA",
            ),
        }
        for width, expected_headings in expected_branch_headings.items():
            with self.subTest(width=width):
                output = render_overview(report, color=False, width=width)
                self.assertTrue(
                    all(len(line) <= width for line in output.splitlines())
                )
                self.assertIn("@0123", output)
                self.assertIn("12,616/12,972", output)
                self.assertIn("w12", output)
                self.assertNotIn("guess_depth=", output)
                self.assertNotIn("candidate=", output)
                branch_header = next(
                    line for line in output.splitlines()
                    if "Ref" in line and "GuessD" in line
                )
                self.assertEqual(tuple(branch_header.split()), expected_headings)

        narrow = render_overview(report, color=False, width=50)
        narrow_branch_header = next(
            line for line in narrow.splitlines() if "Ref" in line and "GuessD" in line
        )
        self.assertIn("Status", narrow_branch_header)
        self.assertIn("Done", narrow_branch_header)
        self.assertNotIn("Spine", narrow_branch_header)

        wide = render_overview(report, color=False, width=120)
        wide_branch_header = next(
            line for line in wide.splitlines() if "Ref" in line and "GuessD" in line
        )
        self.assertIn("Best", wide_branch_header)
        self.assertIn("4/1", wide)

    def test_reordered_input_keeps_prior_identity_order(self):
        first = overview_report()
        second_branch = deepcopy(first["data"]["branches"][0])
        second_branch.update({
            "branch_reference": "bbbbbbbbbbbb",
            "branch_key_hex": "bbbb",
        })
        first["data"]["branches"].append(second_branch)
        display_order = DisplayOrder()
        render_overview(first, width=100, display_order=display_order)
        reordered = deepcopy(first)
        reordered["data"]["branches"].reverse()
        output = render_overview(
            reordered, previous_report=first, width=100,
            display_order=display_order,
        )
        self.assertLess(output.index("@0123"), output.index("@bbbb"))

    def test_unavailable_queue_still_renders_cache(self):
        report = overview_report()
        report["sources"]["queue"].update({"ok": False, "error": "locked"})
        output = render_overview(report, width=100)
        self.assertIn("queue unavailable: locked", output)
        self.assertNotIn("sources ok", output)
        self.assertIn("exact 200", output)

    def _word_report(self, erd_summary):
        report = overview_report()
        report["report_kind"] = "word"
        report["data"] = {
            "word": "salet",
            "word_is_answer": False,
            "context": {
                "branch_reference": "rootrootroot", "branch_key_hex": "root",
                "spine": [], "guess_depth": 0, "answer_count": 20,
            },
            "response_group_counts": {
                "response_group_count": erd_summary["response_group_count"],
                "trivial_response_group_count": 0,
                "queued_response_group_count": 0,
                "active_response_group_count": 0,
                "exact_response_group_count": 0,
                "loss_response_group_count": 0,
                "missing_response_group_count": 0,
            },
            "erd_summary": erd_summary,
            "response_groups": [],
        }
        return report

    def test_word_report_renders_complete_erd_and_rounds(self):
        report = self._word_report({
            "state": "complete", "erd": 3.564102564102564,
            "max_remaining_depth": 6, "resolved_group_count": 4,
            "infeasible_group_count": 0, "response_group_count": 4,
        })
        output = render_report(report, width=100)
        self.assertIn("ERD 3.564  max-d=6", output)
        self.assertNotIn("3.564102564102564", output)

    def test_word_report_renders_pending_erd(self):
        report = self._word_report({
            "state": "pending", "erd": None, "max_remaining_depth": None,
            "resolved_group_count": 2, "infeasible_group_count": 0,
            "response_group_count": 4,
        })
        output = render_report(report, width=100)
        self.assertIn("ERD pending: 2 of 4 response groups solved", output)

    def test_word_report_renders_infeasible_erd(self):
        report = self._word_report({
            "state": "infeasible", "erd": None, "max_remaining_depth": None,
            "resolved_group_count": 2, "infeasible_group_count": 1,
            "response_group_count": 4,
        })
        output = render_report(report, width=100)
        self.assertIn(
            "ERD ∞: 1 of 4 response groups unsolvable within budget", output
        )

    @staticmethod
    def _leaderboard_report(rows, counts):
        report = overview_report()
        report["report_kind"] = "leaderboard"
        report["data"] = {
            "candidate_count": 14855,
            "counts": counts,
            "total_rows": len(rows),
            "matched_rows": len(rows),
            "rows": rows,
        }
        return report

    def test_leaderboard_report_renders_ranked_table_aligned(self):
        report = self._leaderboard_report(
            [
                {"word": "salet", "word_is_answer": False, "erd": 3.5643502648,
                 "max_remaining_depth": 6, "rank": 1},
                {"word": "crane", "word_is_answer": True, "erd": 3.712,
                 "max_remaining_depth": 5, "rank": 2},
            ],
            {"complete": 2, "pending": 14852, "infeasible": 1},
        )
        output = render_report(report, width=100)
        self.assertIn("Opener leaderboard", output)
        self.assertIn("complete 2", output)
        self.assertIn("3.564", output)
        self.assertNotIn("3.5643502648", output)
        self.assertIn("CRANE*", output)  # word_is_answer renders the asterisk
        # The worst-case header column sits directly over its values.
        lines = output.splitlines()
        header = next(line for line in lines if "Worst-case guesses" in line)
        row = next(line for line in lines if line.strip().startswith("1")).rstrip()
        value_column = len(row) - len(row.split()[-1])
        self.assertEqual(header.index("Worst-case guesses"), value_column)

    def test_leaderboard_report_renders_empty_fallback(self):
        report = self._leaderboard_report(
            [], {"complete": 0, "pending": 14855, "infeasible": 0}
        )
        output = render_report(report, width=100)
        self.assertIn("none complete yet", output)

    def test_watched_word_groups_preserve_full_identity_order(self):
        first = overview_report()
        first["report_kind"] = "word"
        first["data"] = {
            "word": "raise",
            "word_is_answer": False,
            "context": {
                "branch_reference": "rootrootroot",
                "branch_key_hex": "root",
                "spine": [],
                "guess_depth": 0,
                "answer_count": 20,
            },
            "response_group_counts": {
                "response_group_count": 2,
                "trivial_response_group_count": 0,
                "queued_response_group_count": 0,
                "active_response_group_count": 0,
                "exact_response_group_count": 0,
                "loss_response_group_count": 0,
                "missing_response_group_count": 2,
            },
            "erd_summary": {
                "state": "pending", "erd": None, "max_remaining_depth": None,
                "resolved_group_count": 0, "infeasible_group_count": 0,
                "response_group_count": 2,
            },
            "response_groups": [
                {
                    "pattern": "-----", "answer_count": 8,
                    "branch_reference": "aaaaaaaaaaaa", "branch_key_hex": "aa",
                    "branch_status": "unqueued", "branch_worker_status": None,
                    "priority": None, "worker_count": 0,
                    "cache_state": "missing", "best_guess": None,
                    "best_erd": None, "max_remaining_depth": None,
                    "updated_at": None,
                },
                {
                    "pattern": "y----", "answer_count": 4,
                    "branch_reference": "bbbbbbbbbbbb", "branch_key_hex": "bb",
                    "branch_status": "unqueued", "branch_worker_status": None,
                    "priority": None, "worker_count": 0,
                    "cache_state": "missing", "best_guess": None,
                    "best_erd": None, "max_remaining_depth": None,
                    "updated_at": None,
                },
            ],
        }
        display_order = DisplayOrder()
        render_report(first, width=100, display_order=display_order)
        second = deepcopy(first)
        second["data"]["response_groups"].reverse()
        output = render_report(
            second, previous_report=first, width=100,
            display_order=display_order,
        )
        self.assertLess(output.index("@aaaa"), output.index("@bbbb"))

    def test_watched_branch_workers_preserve_worker_identity_order(self):
        first = overview_report()
        worker_two = first["data"]["workers"][0]
        worker_one = deepcopy(worker_two)
        worker_one.update({"worker_id": "worker-1", "worker_number": "1"})
        first["report_kind"] = "branch"
        first["data"] = {
            "branch": {
                "branch_reference": "0123456789ab",
                "branch_key_hex": "010203",
                "spine": [{"word": "raise", "pattern": "-----"}],
                "guess_depth": 1,
                "answer_count": 8,
                "budget": 5,
                "branch_status": "evaluating",
                "branch_worker_status": "active",
            },
            "queue": None,
            "cache": {
                "cache_state": "missing", "best_guess": None,
                "best_erd": None, "max_remaining_depth": None,
            },
            "workers": [worker_two, worker_one],
            "republished_candidates": [],
            "claims": None,
            "provenance_unknown": False,
        }
        display_order = DisplayOrder()
        render_report(first, width=100, display_order=display_order)
        second = deepcopy(first)
        second["data"]["workers"].reverse()
        output = render_report(
            second, previous_report=first, width=100,
            display_order=display_order,
        )
        self.assertLess(output.index("w2"), output.index("w1"))

    def test_candidate_state_renders_bounded_claim_summary(self):
        report = overview_report()
        report["report_kind"] = "branch"
        report["data"] = {
            "branch": {
                "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                "spine": [], "guess_depth": 0, "answer_count": 3, "budget": 6,
                "branch_status": "evaluating", "branch_worker_status": "active",
            },
            "queue": {
                "branch_status": "evaluating", "branch_worker_status": "active",
                "priority": 0, "budget": 6, "best_guess": None,
                "ceiling": None, "search_node_count": 0,
                "candidate_count": 3, "completed_candidate_count": 0,
                "bulk_completed_candidate_count": 0,
                "one_level_erd_pruned_candidate_count": 0,
                "two_level_erd_pruned_candidate_count": 0,
            },
            "cache": {
                "cache_state": "missing", "best_guess": None,
                "best_erd": None, "max_remaining_depth": None,
            },
            "workers": [],
            "republished_candidates": [],
            "claims": None,
            "claim_summary": {
                "total_claim_count": 12972, "done_count": 12819,
                "in_flight_count": 5, "evaluated_count": 11200,
                "one_level_erd_pruned_count": 1500,
                "two_level_erd_pruned_count": 119,
                "provenance_unknown_count": 0,
                "worker_contributions": [
                    {"worker_id": "worker-0", "done_count": 6484},
                    {"worker_id": "worker-2", "done_count": 6335},
                ],
            },
            "provenance_unknown": False,
        }
        output = render_report(report, width=100)
        self.assertNotIn("12,819 done", output)
        self.assertIn("candidates 0/3 =", output)
        self.assertNotIn("=  + evaluated", output)
        self.assertIn("evaluated 11,200", output)
        self.assertNotIn("1,500 one-level ERD prunes", output)
        self.assertNotIn("119 two-level ERD prunes", output)
        self.assertIn("in flight 5", output)
        self.assertIn("worker evals w0:6,484 w2:6,335", output)
        self.assertNotIn("idx=", output)

    def test_selected_branch_detail_survives_parent_status_filter(self):
        report = overview_report()
        report["report_kind"] = "branch"
        report["filters"] = {
            "branch_statuses": [], "branch_worker_statuses": ["active"],
        }
        report["data"] = {
            "branch": {
                "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                "spine": [], "guess_depth": 0, "answer_count": 3, "budget": 6,
                "branch_status": "done", "branch_worker_status": None,
            },
            "queue": None,
            "cache": {
                "cache_state": "exact", "best_guess": "crane",
                "best_erd": 2.0, "max_remaining_depth": 3,
            },
            "workers": [], "republished_candidates": [], "claims": None,
            "provenance_unknown": False,
        }
        output = render_report(report, width=100)
        self.assertIn("Branch @0123", output)
        self.assertIn("status=done worker=—", output)
        self.assertNotIn("no longer matches the parent filter", output)

    def test_pending_queued_overview_renders_without_candidate_total(self):
        report = overview_report()
        report["filters"] = {
            "branch_statuses": ["queued"], "branch_worker_statuses": [],
        }
        branch = report["data"]["branches"][0]
        branch.update({
            "branch_status": "queued",
            "branch_worker_status": None,
            "candidate_count": None,
            "completed_candidate_count": 0,
            "created_at": None,
            "worker_count": 0,
        })
        report["data"]["workers"] = []
        output = render_report(report, width=120)
        self.assertIn("Branches (status=queued)", output)
        self.assertIn("0/—", output)

    def test_hotspot_render_rounds_erd_and_ceiling_bounds(self):
        report = overview_report()
        report.update({"report_kind": "hotspots", "tree": False})
        report["data"] = {
            "field": "nodes", "population": "current_queue_branches",
            "epoch": 4, "since_seconds": 3600, "sample_size": 50000,
            "sampled_row_count": 1, "sample_truncated": False,
            "rows": [{
                "branch_reference": "abcd1234ef00", "answer_count": 33,
                "best_erd": 2.793103449275866, "ceiling": 2.0000000009999996,
                "spine": "RAISE ----- CRANE y----",
            }],
        }
        output = render_report(report, width=120)
        self.assertIn("best_erd=2.793", output)
        self.assertIn("ceiling=2.000", output)
        self.assertIn("spine=RAISE ----- CRANE y----", output)
        self.assertNotIn("2.793103449275866", output)

    def test_hotspot_render_labels_population_window_and_truncation(self):
        report = overview_report()
        report.update({"report_kind": "hotspots", "tree": False})
        report["data"] = {
            "field": "coordination",
            "population": "recent_claim_coordination_buckets",
            "epoch": 3,
            "since_seconds": 3600,
            "window_started_at": 100,
            "sample_size": 50000,
            "sampled_row_count": 50000,
            "sample_truncated": True,
            "rows": [{
                "row_id": "coordination:20:4",
                "answer_count": 20,
                "worker_count": 4,
                "coordination_millis": 900,
            }],
        }
        output = render_report(report, width=100)
        self.assertIn("Population: recent_claim_coordination_buckets", output)
        self.assertIn("epoch=3", output)
        self.assertIn("since-seconds=3600", output)
        self.assertIn("sample-size=50000", output)
        self.assertIn("truncated=true", output)

    def test_accuracy_render_distinguishes_requested_and_achieved_samples(self):
        report = overview_report()
        report.update({"report_kind": "accuracy", "tree": False})
        report["data"] = {
            "epoch": 4, "population_row_count": None,
            "requested_sample_size": 50_000, "sampled_row_count": 2,
            "erd_pruned_row_count": 1, "non_erd_pruned_row_count": 1,
            "no_prediction_row_count": 1,
            "calibration": {"row_count": 1,
                            "actual_predicted_ratio": {
                                "p1": 0.1, "p10": 0.2, "p50": 1.0,
                                "p90": 5.0, "p99": 10.0}},
            "largest_under_predicted": [], "rows": [],
        }
        output = render_report(report, width=120)
        self.assertIn("population not counted", output)
        self.assertIn("random sample 2/50,000 requested", output)
        self.assertIn("non-pruned calibration rows 1", output)
        self.assertIn("p1=0.10", output)


class CandidateSweepBarTest(unittest.TestCase):
    def test_block_heights_scale_with_cell_completion(self):
        bar = report_terminal.candidate_sweep_bar(80, range(0, 40), (), width=8)
        self.assertEqual(len(bar), 8)
        self.assertEqual(bar[:4], "████")
        self.assertEqual(bar[4:], "    ")

    def test_partial_cells_use_intermediate_blocks(self):
        bar = report_terminal.candidate_sweep_bar(80, range(0, 5), (), width=8)
        self.assertEqual(len(bar), 8)
        self.assertNotEqual(bar[0], " ")
        self.assertNotEqual(bar[0], "█")

    def test_first_completion_lifts_cell_off_baseline(self):
        bar = report_terminal.candidate_sweep_bar(80, [0], (), width=8)
        self.assertEqual(bar[0], "▁")

    def test_full_block_requires_entirely_completed_cell(self):
        nearly_full = report_terminal.candidate_sweep_bar(
            80, range(0, 9), (), width=8
        )
        self.assertEqual(nearly_full[0], "▇")
        full = report_terminal.candidate_sweep_bar(80, range(0, 10), (), width=8)
        self.assertEqual(full[0], "█")

    def test_worker_positions_overlay_digits(self):
        bar = report_terminal.candidate_sweep_bar(
            100, range(0, 50), [(75, "2")], width=10
        )
        self.assertEqual(len(bar), 10)
        self.assertIn("2", bar)
        self.assertEqual(bar.index("2"), 7)

    def test_adjacent_worker_digits_are_preserved(self):
        bar = report_terminal.candidate_sweep_bar(
            100, (), [(70, "1"), (70, "2")], width=10
        )
        self.assertIn("1", bar)
        self.assertIn("2", bar)

    def test_empty_branch_renders_empty_bar(self):
        self.assertEqual(report_terminal.candidate_sweep_bar(0, (), ()), "")

    def test_branch_report_renders_sweep_with_worker_position(self):
        report = {
            "schema_version": 3,
            "report_kind": "branch",
            "generated_at": 1000,
            "branch_target": None,
            "filters": {},
            "sources": {
                "queue": {"path": "q", "ok": True, "error": None},
                "telemetry": {"path": "t", "ok": True, "error": None},
                "cache": {"path": "c", "ok": True, "error": None},
            },
            "data": {
                "branch": {
                    "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                    "spine": [], "guess_depth": 0, "answer_count": 8, "budget": 6,
                    "branch_status": "evaluating", "branch_worker_status": "active",
                },
                "queue": {
                    "branch_status": "evaluating", "branch_worker_status": "active",
                    "priority": 1, "candidate_count": 80,
                    "completed_candidate_count": 40,
                    "bulk_completed_candidate_count": 0,
                    "best_guess": "crane", "best_guess_is_answer": True,
                    "best_erd": 2.0, "best_max_remaining_depth": 3,
                    "ceiling": None, "search_node_count": 100,
                    "created_at": 900, "finalized_at": None,
                },
                "cache": {
                    "cache_state": "missing", "best_guess": None,
                    "best_erd": None, "max_remaining_depth": None,
                },
                "workers": [{
                    "worker_id": "worker-2", "worker_number": "2",
                    "updated_at": 999, "is_live": True,
                    "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                    "candidate_index": 60, "current_candidate": "slate",
                    "current_candidate_is_answer": True,
                    "current_max_guess_depth": 2, "nodes_per_second": 10.0,
                }],
                "republished_candidates": [],
                "completed_candidate_indexes": list(range(0, 40)),
                "claims": None,
                "provenance_unknown": False,
            },
        }
        output = render_report(report, width=80)
        sweep_line = next(
            line for line in output.splitlines()
            if line.strip().startswith("[") and "█" in line
        )
        self.assertIn("2", sweep_line)
        self.assertLessEqual(len(sweep_line), 80)

        unswept = deepcopy(report)
        unswept["data"]["completed_candidate_indexes"] = []
        unswept["data"]["workers"] = []
        unswept_output = render_report(unswept, width=80)
        self.assertFalse(any(
            line.strip().startswith("[") for line in unswept_output.splitlines()
        ))

    def test_branch_report_names_ceiling_proven_loss(self):
        report = overview_report()
        report.update({"report_kind": "branch", "branch_target": None})
        branch = deepcopy(report["data"]["branches"][0])
        queue = deepcopy(branch)
        report["data"] = {
            "branch": branch,
            "queue": queue,
            "cache": {"cache_state": "missing", "best_guess": None,
                      "best_erd": None, "max_remaining_depth": None},
            "workers": [],
            "republished_candidates": [], "completed_candidate_indexes": [],
            "claims": None, "provenance_unknown": False,
            "recent_finalizations": [{
                "spine": "CRANE -----", "outcome": "loss",
                "loss_proof": "ceiling_above_budget", "epoch": 11,
                "search_node_count": 100, "evaluated_candidate_count": 8,
                "bulk_completed_candidate_count": 0,
            }],
        }
        self.assertIn("loss unsolvable-within-budget", render_report(report, width=100))

    def test_branch_report_shows_best_first_scheduling_evidence(self):
        report = overview_report()
        report.update({"report_kind": "branch", "branch_target": None})
        branch = deepcopy(report["data"]["branches"][0])
        report["data"] = {
            "branch": branch,
            "queue": deepcopy(branch),
            "cache": {"cache_state": "missing", "best_guess": None,
                      "best_erd": None, "max_remaining_depth": None},
            "workers": [],
            "republished_candidates": [], "completed_candidate_indexes": [],
            "claims": None, "provenance_unknown": False,
            "recent_finalizations": [{
                "spine": "CRANE -----", "outcome": "exact", "epoch": 11,
                "search_node_count": 100, "evaluated_candidate_count": 8,
                "bulk_completed_candidate_count": 0,
                "winner_best_first_rank": 46,
                "winner_republish_count": 1,
                "candidates_completed_before_winner": 4900,
                "weakest_best_first_rank_before_winner": 7795,
                "republished_candidate_count": 5305,
                "max_candidate_republish_count": 3,
            }],
        }
        output = render_report(report, width=200)
        self.assertIn("winner ranked 46", output)
        self.assertIn("4,900 candidates completed first", output)
        self.assertIn("weakest of them ranked 7,795", output)
        self.assertIn("winner republished 1x", output)
        self.assertIn("5,305 candidates republished (up to 3x each)", output)

    def test_branch_report_omits_scheduling_evidence_when_unrecorded(self):
        report = overview_report()
        report.update({"report_kind": "branch", "branch_target": None})
        branch = deepcopy(report["data"]["branches"][0])
        report["data"] = {
            "branch": branch,
            "queue": deepcopy(branch),
            "cache": {"cache_state": "missing", "best_guess": None,
                      "best_erd": None, "max_remaining_depth": None},
            "workers": [],
            "republished_candidates": [], "completed_candidate_indexes": [],
            "claims": None, "provenance_unknown": False,
            "recent_finalizations": [{
                "spine": "CRANE -----", "outcome": "exact", "epoch": 11,
                "search_node_count": 100, "evaluated_candidate_count": 8,
                "bulk_completed_candidate_count": 0,
                "winner_best_first_rank": None,
                "republished_candidate_count": 0,
            }],
        }
        self.assertNotIn("best-first order:", render_report(report, width=200))

    def test_branch_report_omits_comparisons_without_a_winner_rank(self):
        # "weakest of them ranked M" has no referent without the winner's rank.
        report = overview_report()
        report.update({"report_kind": "branch", "branch_target": None})
        branch = deepcopy(report["data"]["branches"][0])
        report["data"] = {
            "branch": branch,
            "queue": deepcopy(branch),
            "cache": {"cache_state": "missing", "best_guess": None,
                      "best_erd": None, "max_remaining_depth": None},
            "workers": [],
            "republished_candidates": [], "completed_candidate_indexes": [],
            "claims": None, "provenance_unknown": False,
            "recent_finalizations": [{
                "spine": "CRANE -----", "outcome": "exact", "epoch": 11,
                "search_node_count": 100, "evaluated_candidate_count": 8,
                "bulk_completed_candidate_count": 0,
                "winner_best_first_rank": None,
                "candidates_completed_before_winner": 4900,
                "weakest_best_first_rank_before_winner": 7795,
                "republished_candidate_count": 12,
                "max_candidate_republish_count": 2,
            }],
        }
        output = render_report(report, width=200)
        self.assertIn("12 candidates republished", output)
        self.assertNotIn("candidates completed first", output)
        self.assertNotIn("weakest of them ranked", output)


class CollectionRendererTest(unittest.TestCase):
    def _report(self, report_kind, data, tree=False):
        report = overview_report()
        report.update({
            "report_kind": report_kind,
            "branch_target": {
                "kind": "root", "steps": [], "trailing_word": None,
                "branch_reference": None, "input_text": "",
            },
            "filters": {},
            "tree": tree,
            "data": data,
        })
        return report

    def test_tree_groups_base_patterns_under_their_word(self):
        def node(node_id, parent_node_id, guess_depth, word=None, pattern=None):
            return {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "step": (
                    {"word": word, "pattern": pattern}
                    if word is not None else None
                ),
                "branch_key_hex": node_id if word is not None else None,
                "branch_reference": node_id[:12] if word is not None else None,
                "branch_status": "evaluating" if word is not None else None,
                "branch_worker_status": "active" if word is not None else None,
                "answer_count": 2 if word is not None else None,
                "guess_depth": guess_depth,
                "worker_count": 0,
                "completed_candidate_count": 0,
                "candidate_count": 4,
                "is_context": False,
            }

        report = self._report("queue", {
            "tree_available": True,
            "unavailable_reason": None,
            "nodes": [
                node("raise:-----", None, 1, "raise", "-----"),
                node("raise:y----", None, 1, "raise", "y----"),
                node("stink:g----", None, 1, "stink", "g----"),
                node(
                    "raise:-----/crane:y----", "raise:-----", 2,
                    "crane", "y----",
                ),
                node(
                    "stink:g----/mount:-y---", "stink:g----", 2,
                    "mount", "-y---",
                ),
            ],
        }, tree=True)
        output = render_report(report, width=120)
        # A word is named once, by its group, at every level; the rows beneath
        # carry only the response pattern.  One pattern still gets a group.
        self.assertIn("RAISE  2 branches", output)
        self.assertIn("STINK  1 branch", output)
        self.assertIn("CRANE  1 branch", output)
        self.assertNotIn("RAISE -----", output)
        self.assertNotIn("CRANE y----", output)
        indents = {
            line.strip(): len(line) - len(line.lstrip())
            for line in output.splitlines() if line.strip()
        }

        def indent_of(prefix):
            return indents[next(key for key in indents if key.startswith(prefix))]

        # One space per level: word group, its patterns, their word groups.
        self.assertEqual(indents["RAISE  2 branches"], 0)
        self.assertEqual(indents["STINK  1 branch"], 0)
        self.assertEqual(indent_of("-----"), 1)
        self.assertEqual(indents["CRANE  1 branch"], 2)
        # RAISE y---- is a base pattern; CRANE y---- is a pattern of a group one
        # level down, and they render the same but for their indent.
        self.assertEqual(sorted(
            len(line) - len(line.lstrip())
            for line in output.splitlines() if line.strip().startswith("y----")
        ), [1, 3])
        self.assertLess(output.index("RAISE  2 branches"), output.index("CRANE  1 branch"))
        self.assertLess(output.index("CRANE  1 branch"), output.index("STINK  1 branch"))
        self.assertLess(output.index("STINK  1 branch"), output.index("MOUNT  1 branch"))

    def test_queue_worker_and_cache_collections_are_semantically_formatted(self):
        queue_report = self._report("queue", {
            "summary": {
                "branch_count_by_status": {"evaluating": 1},
                "branch_count_by_worker_status": {"active": 1},
            },
            "matched_rows": 1,
            "rows": [{
                "branch_reference": "0123456789ab",
                "branch_status": "evaluating",
                "branch_worker_status": "active",
                "answer_count": 2,
                "spine": [
                    {"word": "raise", "pattern": "-----", "word_is_answer": False},
                    {"word": "crane", "pattern": "y----", "word_is_answer": True},
                ],
                "priority": 7,
                "worker_count": 1,
            }],
        })
        queue_output = render_report(queue_report, width=120)
        self.assertIn("guess_depth=2", queue_output)
        self.assertIn("spine=RAISE ----- CRANE y----", queue_output)
        self.assertNotIn(" d=2", queue_output)
        narrow_queue_output = render_report(queue_report, width=40)
        self.assertTrue(all(
            len(line) <= 40 for line in narrow_queue_output.splitlines()
        ))

        fallback_queue_report = self._report("queue", {
            "summary": {
                "branch_count_by_status": {"evaluating": 1},
                "branch_count_by_worker_status": {"waiting": 1},
            },
            "matched_rows": 1,
            "rows": [{
                "branch_reference": "fedcba987654",
                "branch_status": "evaluating",
                "branch_worker_status": "waiting",
                "answer_count": 2,
                "source_word": "raise",
                "source_pattern": "-----",
                "priority": 7,
                "worker_count": 0,
            }],
        })
        fallback_queue_output = render_report(fallback_queue_report, width=120)
        self.assertIn("guess_depth=1", fallback_queue_output)
        self.assertIn("spine=RAISE -----", fallback_queue_output)

        worker = deepcopy(overview_report()["data"]["workers"][0])
        worker["state"] = "stale"
        workers_report = self._report("workers", {
            "summary": {"worker_count_by_state": {"stale": 1}},
            "matched_rows": 1,
            "rows": [worker],
        })
        self.assertIn("w2  stale", render_report(workers_report, width=120))

        cache_report = self._report("cache", {
            "summary": {
                "exact_branch_count": 2,
                "loss_branch_count": 1,
                "recent_exact_branch_count": 1,
            },
            "distributions": {
                "state_branch_counts": {
                    "exact_branch_count": 2, "loss_branch_count": 1,
                },
                "exact_branch_count_by_max_remaining_depth": {"3": 2},
                "exact_branch_count_by_solve_budget": {"unbounded": 2},
                "exact_branch_count_by_taint": {
                    "untainted": 2, "tainted": 0,
                },
                "loss_branch_count_by_loss_budget": {"4": 1},
            },
        })
        cache_output = render_report(cache_report, width=120)
        self.assertIn("max remaining depth: 3=2", cache_output)
        self.assertIn("loss budget: 4=1", cache_output)
        self.assertNotIn("{'", cache_output)


class ViewSessionTest(unittest.TestCase):
    def test_json_output_round_trips_exact_report(self):
        report = overview_report()
        output = io.StringIO()
        with patch("report_terminal.collect_report", return_value=report):
            WatchSession(
                view_args(format="json"), FakeInput(), output, io.StringIO()
            ).run()
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_one_shot_jsonl_is_compact(self):
        report = overview_report()
        output = io.StringIO()
        with patch("report_terminal.collect_report", return_value=report):
            WatchSession(
                view_args(format="jsonl"), FakeInput(), output, io.StringIO()
            ).run()
        self.assertEqual(json.loads(output.getvalue()), report)
        self.assertNotIn('": ', output.getvalue())

    def test_jsonl_watch_emits_parseable_lines(self):
        first = overview_report()
        second = deepcopy(first)
        second["generated_at"] += 1
        output = io.StringIO()
        with (
            patch("report_terminal.collect_report", side_effect=[first, second]),
            patch("report_terminal.time.sleep", side_effect=[None, KeyboardInterrupt]),
        ):
            WatchSession(
                view_args(format="jsonl", watch=1.0), FakeInput(), output,
                io.StringIO(),
            ).run()
        lines = output.getvalue().splitlines()
        self.assertEqual([json.loads(line) for line in lines], [first, second])
        self.assertNotIn("\033", output.getvalue())

    def test_non_tty_watch_has_separators_and_no_control_codes(self):
        report = overview_report()
        output = io.StringIO()
        with (
            patch("report_terminal.collect_report", return_value=report),
            patch("report_terminal.time.sleep", side_effect=KeyboardInterrupt),
        ):
            WatchSession(
                view_args(watch=1.0), FakeInput(), output, io.StringIO()
            ).run()
        self.assertIn("--- generated_at=1000 ---", output.getvalue())
        self.assertNotIn("\033", output.getvalue())

    def test_redirected_output_uses_non_tty_watch_even_with_tty_input(self):
        report = overview_report()
        output = FakeOutput(tty=False)
        with (
            patch("report_terminal.collect_report", return_value=report),
            patch("report_terminal.time.sleep", side_effect=KeyboardInterrupt),
        ):
            WatchSession(
                view_args(watch=1.0), FakeInput(tty=True), output, io.StringIO()
            ).run()
        self.assertIn("--- generated_at=1000 ---", output.getvalue())
        self.assertNotIn("\033", output.getvalue())

    def test_non_tty_watch_lists_ambiguity_candidates_on_error(self):
        first_key = encode_subset(["salet", "crane"])
        second_key = encode_subset(["nurdy", "khaki"])
        error = ValueError("branch reference @abcd is ambiguous")
        error.candidates = [
            {"branch_reference": branch_reference(first_key),
             "branch_key": first_key, "spine": "salet -----"},
            {"branch_reference": branch_reference(second_key),
             "branch_key": second_key, "spine": None},
        ]
        output = io.StringIO()
        with (
            patch("report_terminal.collect_report", side_effect=error),
            patch("report_terminal.time.sleep", side_effect=KeyboardInterrupt),
        ):
            WatchSession(
                view_args(watch=1.0), FakeInput(), output, io.StringIO()
            ).run()
        text = output.getvalue()
        self.assertIn("ambiguous", text)
        self.assertIn("@" + branch_reference(first_key), text)
        self.assertIn("@" + branch_reference(second_key), text)
        self.assertIn("spine=SALET -----", text)

    def test_tty_watch_lists_ambiguity_candidates_on_error(self):
        first_key = encode_subset(["salet", "crane"])
        second_key = encode_subset(["nurdy", "khaki"])
        error = ValueError("branch reference @abcd is ambiguous")
        error.candidates = [
            {"branch_reference": branch_reference(first_key),
             "branch_key": first_key, "spine": "salet -----"},
            {"branch_reference": branch_reference(second_key),
             "branch_key": second_key, "spine": None},
        ]
        output = FakeOutput(tty=True)
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), output, io.StringIO()
        )
        session._collect = Mock(side_effect=error)
        session._wait_for_refresh = Mock(return_value=False)
        with (
            patch("report_terminal.termios.tcgetattr", return_value=[0, 0, 0, 0]),
            patch("report_terminal.termios.tcsetattr"),
        ):
            session.run()
        text = output.getvalue()
        self.assertIn("ambiguous", text)
        self.assertIn("@" + branch_reference(first_key), text)
        self.assertIn("@" + branch_reference(second_key), text)

    def test_error_lines_falls_back_when_candidate_rendering_fails(self):
        """Enrichment failure (e.g. an unreadable answer list) must not
        replace the original error -- it degrades to the bare message."""
        session = WatchSession(
            view_args(watch=1.0), FakeInput(), io.StringIO(), io.StringIO()
        )
        error = ValueError("branch reference @abcd is ambiguous")
        error.candidates = []
        with patch(
            "report_terminal.collect_ambiguous_branch_reference_report",
            side_effect=RuntimeError("answer list unreadable"),
        ):
            self.assertEqual(session._error_lines(error), [f"view: {error}"])

    def test_shrinking_section_clears_old_lines(self):
        output = io.StringIO()
        session = WatchSession(view_args(watch=1.0), FakeInput(tty=True), output)
        session.previous_sections = [
            ("header", ["one", "two", "three"]),
            ("cache", ["cache"]),
        ]
        session._refresh_sections([
            ("header", ["one"]),
            ("cache", ["cache"]),
        ])
        self.assertGreaterEqual(output.getvalue().count(report_terminal.CLEAR_LINE), 3)
        self.assertIn("\033[J", output.getvalue())

    def test_growing_section_clears_shifted_separator_row(self):
        output = io.StringIO()
        session = WatchSession(view_args(watch=1.0), FakeInput(tty=True), output)
        session.previous_sections = [
            ("first", ["one"]),
            ("second", ["two", "three"]),
        ]
        session._refresh_sections([
            ("first", ["one", "new two", "new three"]),
            ("second", ["two", "three"]),
        ])
        self.assertIn("\033[4;1H" + report_terminal.CLEAR_LINE, output.getvalue())

    def test_stdin_eof_exits_refresh_wait(self):
        session = WatchSession(
            view_args(watch=1.0), FakeInput("", tty=True), FakeOutput(tty=True)
        )
        with patch("report_terminal.select.select", return_value=([session.input_stream], [], [])):
            self.assertFalse(session._wait_for_refresh())

    def test_tty_watch_shows_loading_notice_before_first_report(self):
        report = overview_report()
        output = FakeOutput(tty=True)
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), output, io.StringIO()
        )

        def collect_after_notice():
            self.assertIn("Collecting report…", output.getvalue())
            self.assertIn("queue: queue.sqlite3", output.getvalue())
            self.assertIn("cache: cache.sqlite3", output.getvalue())
            return report

        session._collect = Mock(side_effect=collect_after_notice)
        session._wait_for_refresh = Mock(return_value=False)
        with (
            patch("report_terminal.termios.tcgetattr", return_value=[0, 0, 0, 0]),
            patch("report_terminal.termios.tcsetattr"),
        ):
            session.run()
        self.assertIn("ERD swarm overview", output.getvalue())

    def test_navigation_reset_shows_loading_notice(self):
        output = FakeOutput(tty=True)
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), output, io.StringIO()
        )
        session._update_navigation_targets(overview_report())
        session._select_branch("010203")
        self.assertIn("Collecting report…", output.getvalue())

    def test_tty_failure_retries_and_restores_terminal(self):
        report = overview_report()
        output = FakeOutput(tty=True)
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), output, io.StringIO()
        )
        session._collect = Mock(side_effect=[RuntimeError("temporary"), report])
        session._wait_for_refresh = Mock(side_effect=[True, False])
        terminal_values = [0, 0, 0, 0]
        with (
            patch("report_terminal.termios.tcgetattr", return_value=terminal_values),
            patch("report_terminal.termios.tcsetattr") as set_attributes,
        ):
            session.run()
        self.assertIn("view: temporary", output.getvalue())
        self.assertIn("ERD swarm overview", output.getvalue())
        self.assertIn("\033[?25h", output.getvalue())
        self.assertGreaterEqual(set_attributes.call_count, 2)

    def test_tty_branch_and_worker_hotkeys_push_report_requests(self):
        report = overview_report()
        output = io.StringIO()
        session = WatchSession(
            view_args(watch=1.0), FakeInput("a2", tty=True), output
        )
        session._update_navigation_targets(report)
        with patch("report_terminal.select.select", return_value=([session.input_stream], [], [])):
            self.assertTrue(session._wait_for_refresh())
        branch_request = session.current_request
        self.assertEqual(branch_request.report_kind, "auto")
        self.assertEqual(branch_request.branch_target.kind, "branch")
        self.assertEqual(len(branch_request.branch_target.steps), 2)

        session._navigate_back()
        session._update_navigation_targets(report)
        with patch("report_terminal.select.select", return_value=([session.input_stream], [], [])):
            self.assertTrue(session._wait_for_refresh())
        worker_request = session.current_request
        self.assertEqual(worker_request.report_kind, "workers")
        self.assertEqual(worker_request.worker_id, "worker-2")

    def test_tty_multi_digit_worker_hotkey_pushes_report_request(self):
        report = overview_report()
        worker_twelve = deepcopy(report["data"]["workers"][0])
        worker_twelve.update({
            "worker_id": "worker-12",
            "worker_number": "12",
        })
        report["data"]["workers"].append(worker_twelve)
        session = WatchSession(
            view_args(watch=1.0), FakeInput("12", tty=True), io.StringIO()
        )
        session._update_navigation_targets(report)
        with patch(
            "report_terminal.select.select",
            return_value=([session.input_stream], [], []),
        ):
            self.assertTrue(session._wait_for_refresh())
        self.assertEqual(session.current_request.report_kind, "workers")
        self.assertEqual(session.current_request.worker_id, "worker-12")

    def test_navigation_legend_is_compact_and_fits_width(self):
        report = overview_report()
        workers = []
        for worker_number in range(1, 16):
            worker = deepcopy(report["data"]["workers"][0])
            worker.update({
                "worker_id": f"worker-{worker_number}",
                "worker_number": str(worker_number),
            })
            workers.append(worker)
        report["data"]["workers"] = workers
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), io.StringIO()
        )
        session._width = Mock(return_value=48)
        session._update_navigation_targets(report)
        _name, lines = session._navigation_section()[0]
        self.assertTrue(all(len(line) <= 48 for line in lines))
        self.assertLessEqual(len(lines), 2)
        rendered = "\n".join(lines)
        self.assertIn("[a-z] branch", rendered)
        self.assertIn("[0-9] worker", rendered)
        self.assertIn("[space] refresh", rendered)
        self.assertNotIn("worker-12", rendered)

    def test_branch_hotkey_letters_render_inline(self):
        report = overview_report()
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), io.StringIO()
        )
        session._update_navigation_targets(report)
        letter = session.branch_letter_by_key["010203"]
        output = render_overview(
            report, width=100, display_order=session.display_order
        )
        self.assertIn("Key", output)
        self.assertIn(f"[{letter}]  @0123", output)

    def test_back_restores_complete_prior_request(self):
        filters = ReportFilters(
            branch_statuses=("evaluating",), minimum_answer_count=10,
            sort="size", limit=4,
        )
        session = WatchSession(view_args(
            watch=1.0,
            branch_target=parse_report_branch_target("CRANE"),
            report_kind="queue",
            tree=True,
            filters=filters,
        ), FakeInput(tty=True), io.StringIO())
        original_request = session.current_request
        session._update_navigation_targets(overview_report())
        session._select_branch("010203")
        self.assertNotEqual(session.current_request, original_request)
        session._navigate_back()
        self.assertEqual(session.current_request, original_request)

    def test_branch_hotkey_stays_pinned_by_full_identity_during_finalization(self):
        report = overview_report()
        session = WatchSession(
            view_args(watch=1.0), FakeInput(tty=True), io.StringIO()
        )
        session._update_navigation_targets(report)
        letter = session.branch_letter_by_key["010203"]
        finalizing = deepcopy(report)
        finalizing["data"]["branches"][0]["branch_status"] = "finalizing"
        session._update_navigation_targets(finalizing)
        self.assertEqual(session.branch_hotkeys[letter], "010203")


class SourcesCommandEndToEndTest(unittest.TestCase):
    """`view --openers` end to end against a real temp queue.  An earlier
    `--sources` attempt raised TypeError on every invocation and was backed
    out during #203 review because no test ever actually ran it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        queue = ERDQueue(self.queue_path)
        branch_key = encode_subset(["crane", "slate"])
        queue.add_pending_many([(branch_key, 2, 1, "slate", 0)])
        # A second request with several branches: the report must render it as
        # one row, not one row per branch.
        queue.add_pending_many([
            (encode_subset(["crane", "slate", f"w{index:04d}"]), 3, 4, "raise",
             index)
            for index in range(9)
        ])
        queue.close()

    def _run(self, *args):
        output = io.StringIO()
        with (
            patch("sys.argv", [
                "erd_search.py", "view", "--openers",
                "--queue-path", self.queue_path,
                "--cache-path", self.cache_path,
                *args,
            ]),
            redirect_stdout(output),
        ):
            erd_search.main()
        return output.getvalue()

    def test_text_output_is_one_row_per_word_not_per_branch(self):
        text = self._run()
        self.assertIn("SLATE", text)
        self.assertIn("RAISE", text)
        # Two words, ten branches between them: two rows.
        self.assertEqual(text.count("SLATE"), 1)
        self.assertEqual(len([
            line for line in text.splitlines()
            if line.startswith("  ") and line.split()[0] in ("SLATE", "RAISE")
        ]), 2)
        self.assertIn("Openers: 2", text)
        self.assertIn("requests: 2", text)
        # The branch rows are not printed until a word is named, and the report
        # says which command opens them.
        self.assertNotIn("Ownership:", text)
        self.assertIn("view --openers", text)

    def test_a_word_queued_twice_is_one_row_counting_its_branches_once(self):
        # Source work is keyed by (word, priority), so queueing RAISE again at
        # a new priority makes a second request that shares a branch with the
        # first.  The report merges them without counting that branch twice.
        queue = ERDQueue(self.queue_path)
        queue.add_pending_many([
            (encode_subset(["crane", "slate", "w0000"]), 3, 8, "raise", 0),
            (encode_subset(["crane", "slate", "fresh"]), 3, 8, "raise", 1),
        ])
        queue.close()

        report = json.loads(self._run("--format", "json"))

        rollups = {row["source_word"]: row for row in report["data"]["summary"]}
        self.assertEqual(rollups["raise"]["request_count"], 2)
        # Nine branches from the first request plus one new one; the branch
        # both requests own is counted once.
        self.assertEqual(rollups["raise"]["branch_count"], 10)
        self.assertEqual(rollups["raise"]["open_branch_count"], 10)
        # The merged priority is the one that actually schedules.
        self.assertEqual(rollups["raise"]["requested_priority"], 8)
        self.assertIn("Reqs", self._run())

    def test_a_named_word_with_no_live_branches_says_so(self):
        # Naming a word whose branches have all finished must not print the
        # "name a source word" hint naming the word already named.
        queue = ERDQueue(self.queue_path)
        queue.mark_done(encode_subset(["crane", "slate"]))
        queue.close()

        text = self._run("slate")

        self.assertIn("SLATE owns no live branches", text)
        self.assertNotIn("view --openers SLATE", text)
        # The unnamed report still points the way in.
        self.assertIn("Name an opener", self._run())

    def test_source_state_filter_and_sort_reach_the_rendered_table(self):
        # --opener-state and --sort are the terminal's half of the same
        # filtering the browser gets; grouping is browser-only.
        sorted_by_word = self._run("--sort", "word")
        rows = [line.split()[0] for line in sorted_by_word.splitlines()
                if line.startswith("  ") and line.split()[0] in
                ("SLATE", "RAISE")]
        self.assertEqual(rows, ["RAISE", "SLATE"])
        # Nothing is complete yet, so filtering to complete empties the table
        # and says so against the unfiltered total rather than reading as an
        # empty queue.
        complete = self._run("--opener-state", "complete")
        self.assertIn("Openers: 0 of 2", complete)
        self.assertNotIn("SLATE", complete)
        self.assertIn("Openers: 2", self._run("--opener-state", "queued"))

    def test_naming_a_word_opens_that_request_s_branches(self):
        text = self._run("raise")
        self.assertIn("Ownership:", text)
        self.assertEqual(text.count("@"), 9)
        self.assertNotIn("SLATE", text)

    def test_json_output_round_trips_the_rolled_up_summary(self):
        report = json.loads(self._run("--format", "json"))
        self.assertEqual(report["report_kind"], "openers")
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertEqual(report["data"]["rows"], [])
        rollups = {row["source_word"]: row for row in report["data"]["summary"]}
        self.assertEqual(rollups["slate"]["requested_priority"], 1)
        self.assertEqual(rollups["raise"]["branch_count"], 9)
        self.assertEqual(rollups["raise"]["open_branch_count"], 9)
        self.assertEqual(rollups["raise"]["done_branch_count"], 0)

    def test_jsonl_output_is_one_line(self):
        text = self._run("--format", "jsonl")
        lines = text.splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["report_kind"], "openers")

    def test_word_filter_narrows_to_the_matching_request(self):
        report = json.loads(self._run("slate", "--format", "json"))
        self.assertEqual(
            [row["source_word"] for row in report["data"]["summary"]], ["slate"]
        )
        self.assertEqual(
            [row["source_word"] for row in report["data"]["rows"]], ["slate"]
        )

    def test_mutually_exclusive_with_other_view_kinds(self):
        with (
            patch("sys.argv", [
                "erd_search.py", "view", "--openers", "--workers",
                "--queue-path", self.queue_path,
                "--cache-path", self.cache_path,
            ]),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            erd_search.main()
        self.assertEqual(raised.exception.code, 2)


_REAL_WORD_POOL = [
    "salet", "crane", "nurdy", "khaki", "fuzzy", "raise", "slate", "adieu",
    "mango", "brisk", "vapid", "zesty", "gloom", "humid", "joker", "witty",
    "ovate", "plumb", "quirk", "xenon",
]


class BranchReferenceRoundTripTest(unittest.TestCase):
    """A handle printed by `view --queue` must resolve, unmodified, through a
    later `view @handle` -- even when a shorter prefix would have collided
    against another branch elsewhere in the cache (#212)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")

    def _run(self, *args):
        output = io.StringIO()
        with (
            patch("sys.argv", [
                "erd_search.py", "view",
                "--queue-path", self.queue_path,
                "--cache-path", self.cache_path,
                *args,
            ]),
            redirect_stdout(output),
        ):
            erd_search.main()
        return output.getvalue()

    def _seed_four_character_collision(self):
        """Two branch keys whose references share a 4-character prefix --
        the width the printer used to truncate to -- found by the birthday
        approach: over enough samples in a 65536-slot space, an internal
        collision is all but certain."""
        keys_by_four_char_prefix = {}
        for index in range(10000):
            branch_key = f"branch {index}".encode()
            prefix = branch_reference(branch_key)[:4]
            if prefix in keys_by_four_char_prefix:
                return keys_by_four_char_prefix[prefix], branch_key
            keys_by_four_char_prefix[prefix] = branch_key
        self.fail("no 4-character reference collision found")

    def _seed_four_character_collision_with_real_words(self):
        """Two branch keys, each five real words encoded via
        ScoreCache.encode_subset(), whose references share a 4-character
        prefix -- found the same way as _seed_four_character_collision, but
        from real words so decoding produces a real answer preview, and five
        words each so a preview truncates and exercises the ellipsis."""
        keys_by_four_char_prefix = {}
        for index in range(10000):
            words = random.Random(index).sample(_REAL_WORD_POOL, 5)
            branch_key = encode_subset(words)
            prefix = branch_reference(branch_key)[:4]
            existing = keys_by_four_char_prefix.get(prefix)
            if existing is not None and existing != branch_key:
                return existing, branch_key
            keys_by_four_char_prefix[prefix] = branch_key
        self.fail("no 4-character reference collision found")

    def test_queue_printed_handle_resolves_despite_a_four_character_collision(self):
        queued_key, cache_only_key = self._seed_four_character_collision()

        queue = ERDQueue(self.queue_path)
        queue.add_pending_many([(queued_key, 2, 1, "salet", 0)])
        queue.close()

        cache = ScoreCache(self.cache_path, ["salet"], checkpoint_on_close=False)
        cache.write_loss(cache_only_key, ERD_ALL, 3)
        cache.close()

        displayed_handle = "@" + branch_reference(queued_key)[:8]
        queue_text = self._run("--queue")
        self.assertIn(displayed_handle, queue_text)

        branch_text = self._run(displayed_handle)
        self.assertIn(f"Branch {displayed_handle}", branch_text)
        self.assertNotIn("ambiguous", branch_text)

    def test_hand_typed_short_prefix_lists_every_candidate_in_full(self):
        queued_key, cache_only_key = self._seed_four_character_collision_with_real_words()

        queue = ERDQueue(self.queue_path)
        queue.add_pending_many([(queued_key, 5, 1, "salet", 0)])
        queue.close()

        cache = ScoreCache(self.cache_path, ["salet"], checkpoint_on_close=False)
        cache.write_loss(cache_only_key, ERD_ALL, 3)
        cache.close()

        short_prefix = "@" + branch_reference(queued_key)[:4]
        error_output = io.StringIO()
        with (
            patch("sys.argv", [
                "erd_search.py", "view",
                "--queue-path", self.queue_path,
                "--cache-path", self.cache_path,
                short_prefix,
            ]),
            redirect_stdout(io.StringIO()),
            patch("sys.stderr", error_output),
            self.assertRaises(SystemExit) as raised,
        ):
            erd_search.main()
        self.assertEqual(raised.exception.code, 1)
        text = error_output.getvalue()
        self.assertIn("ambiguous", text)
        self.assertIn("@" + branch_reference(queued_key), text)
        self.assertIn("@" + branch_reference(cache_only_key), text)
        # Each candidate holds five words, one more than the three-word
        # preview -- covers the truncated-preview ellipsis.
        self.assertIn("n=5", text)
        self.assertEqual(text.count("…"), 2)


class ViewParserTest(unittest.TestCase):
    def test_every_swarm_guide_command_example_parses(self):
        with open("SWARM.md") as guide_file:
            physical_lines = guide_file.readlines()
        logical_lines = []
        pending = ""
        for physical_line in physical_lines:
            stripped = physical_line.strip()
            pending += stripped[:-1] + " " if stripped.endswith("\\") else stripped
            if not stripped.endswith("\\"):
                logical_lines.append(pending)
                pending = ""
        commands = [
            shlex.split(line, comments=True)[2:]
            for line in logical_lines
            if line.startswith("python3.13 erd_search.py ")
        ]
        handler_names = (
            "cmd_start", "cmd_stop", "cmd_restart", "cmd_run", "cmd_view",
            "cmd_queue_add", "cmd_queue_clear", "cmd_queue_remove",
            "cmd_queue_priority", "cmd_queue_source_priority",
            "cmd_reset_stale", "cmd_queue_clear_disk_stop",
            "cmd_queue_set_disk_stop", "cmd_queue_reconcile_orphaned_ownership",
            "cmd_epoch_show", "cmd_epoch_set",
        )
        patches = {name: patch.object(erd_search, name)
                   for name in handler_names}
        handlers = {name: handler_patch.start()
                    for name, handler_patch in patches.items()}
        self.addCleanup(lambda: [handler_patch.stop()
                                 for handler_patch in patches.values()])
        self.assertTrue(commands)
        for arguments in commands:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", ["erd_search.py", *arguments]):
                    erd_search.main()
        self.assertTrue(any(handler.called for handler in handlers.values()))
        self.assertTrue(handlers["cmd_queue_source_priority"].called)
        self.assertTrue(handlers["cmd_queue_reconcile_orphaned_ownership"].called)

    def test_removed_read_commands_fail_argparse(self):
        removed_commands = [
            ["status"],
            ["cache-status", "--word", "raise"],
        ] + [
            ["queue", command]
            for command in ("ls", "tree", "show", "summary", "top", "coverage")
        ]
        for arguments in removed_commands:
            with self.subTest(arguments=arguments):
                with (
                    patch("sys.argv", ["erd_search.py", *arguments]),
                    patch("sys.stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    erd_search.main()
                self.assertEqual(raised.exception.code, 2)

    def test_set_disk_stop_requires_reason(self):
        with (
            patch("sys.argv", ["erd_search.py", "queue", "set-disk-stop"]),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            erd_search.main()
        self.assertEqual(raised.exception.code, 2)

    def test_lifecycle_and_queue_mutation_commands_still_dispatch(self):
        cases = [
            (["start"], "cmd_start"),
            (["stop"], "cmd_stop"),
            (["restart"], "cmd_restart"),
            (["run"], "cmd_run"),
            (["queue", "add", "--word", "raise"], "cmd_queue_add"),
            (["queue", "clear", "--yes"], "cmd_queue_clear"),
            (["queue", "remove", "--word", "raise", "--pattern", "....."],
             "cmd_queue_remove"),
            (["queue", "priority", "--word", "raise", "--pattern", ".....",
              "--priority", "3"], "cmd_queue_priority"),
            (["queue", "reset-stale"], "cmd_reset_stale"),
            (["queue", "set-disk-stop", "--reason", "maintenance hold"],
             "cmd_queue_set_disk_stop"),
        ]
        for arguments, handler_name in cases:
            with self.subTest(arguments=arguments):
                with (
                    patch("sys.argv", ["erd_search.py", *arguments]),
                    patch.object(erd_search, handler_name) as handler,
                ):
                    erd_search.main()
                handler.assert_called_once()

    def test_invalid_json_watch_and_short_interval_are_argparse_errors(self):
        for arguments in (
            ["erd_search.py", "view", "--format", "json", "--watch"],
            ["erd_search.py", "view", "--watch", "0.1"],
        ):
            with self.subTest(arguments=arguments):
                with patch("sys.argv", arguments), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        erd_search.main()
                self.assertEqual(raised.exception.code, 2)

    def test_view_cli_delegates_valid_arguments(self):
        with (
            patch("sys.argv", ["erd_search.py", "view", "--format", "jsonl"]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        args = run_view.call_args.args[0]
        self.assertEqual(args.format, "jsonl")
        self.assertIsNone(args.watch)

    def test_queue_and_cache_positionals_are_word_branch_targets(self):
        for word in ("QUEUE", "CACHE"):
            with self.subTest(word=word):
                with (
                    patch("sys.argv", ["erd_search.py", "view", word]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                branch_target = run_view.call_args.args[0].branch_target
                self.assertEqual(branch_target.kind, "word")
                self.assertEqual(branch_target.trailing_word, word.lower())

    def test_collection_options_dispatch_explicit_kinds(self):
        cases = [
            (["--queue"], "queue", None),
            (["--workers"], "workers", None),
            (["--worker", "2"], "workers", "2"),
            (["--cache"], "cache", None),
            (["--accuracy"], "accuracy", None),
        ]
        for options, report_kind, worker_id in cases:
            with self.subTest(options=options):
                with (
                    patch("sys.argv", ["erd_search.py", "view", *options]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                args = run_view.call_args.args[0]
                self.assertEqual(args.report_kind, report_kind)
                self.assertEqual(args.worker, worker_id)
                if report_kind == "accuracy":
                    self.assertEqual(args.limit, 20)
                    self.assertIsNone(args.since_seconds)

    def test_accuracy_offset_is_forwarded_to_the_report_request(self):
        with (
            patch("sys.argv", ["erd_search.py", "view", "--accuracy",
                              "--limit", "5", "--accuracy-offset", "10"]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        self.assertEqual(run_view.call_args.args[0].accuracy_offset, 10)

    def test_accuracy_since_seconds_is_forwarded_to_the_report_request(self):
        with (
            patch("sys.argv", ["erd_search.py", "view", "--accuracy",
                              "--since-seconds", "60"]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        args = run_view.call_args.args[0]
        self.assertEqual(args.since_seconds, 60)
        self.assertEqual(args.limit, 20)

    def test_incompatible_report_options_and_invalid_branch_filters_are_rejected(self):
        invalid_arguments = [
            ["erd_search.py", "view", "--cache", "--tree"],
            ["erd_search.py", "view", "--branch-status", "evaluating,evaluating"],
            ["erd_search.py", "view", "--branch-status", "all,done"],
            ["erd_search.py", "view", "--branch-worker-status", "unknown"],
            ["erd_search.py", "view", "--queue", "--claims"],
            ["erd_search.py", "view", "--tree", "--answers"],
            ["erd_search.py", "view", "--hotspots", "--tree"],
            ["erd_search.py", "view", "--hotspots", "--by", "cut-reuse",
             "--branch-status", "evaluating"],
            ["erd_search.py", "view", "--hotspots", "--by", "coordination",
             "RAISE"],
            ["erd_search.py", "view", "--by", "nodes"],
            ["erd_search.py", "view", "--epoch", "2"],
            ["erd_search.py", "view", "RAISE -----", "--tree", "--claims"],
            ["erd_search.py", "view", "RAISE", "--sort", "nodes"],
            ["erd_search.py", "view", "--openers", "RAISE", "-----"],
            ["erd_search.py", "view", "--openers", "@abcd"],
            ["erd_search.py", "view", "--branch-status", "unqueued"],
            ["erd_search.py", "view", "--queue", "--branch-status", "unqueued"],
            ["erd_search.py", "view", "RAISE", "--tree",
             "--branch-status", "unqueued"],
            ["erd_search.py", "view", "RAISE -----",
             "--branch-status", "unqueued"],
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", arguments), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        erd_search.main()
                self.assertEqual(raised.exception.code, 2)

    def test_unqueued_branch_status_is_accepted_on_a_word_report(self):
        with (
            patch("sys.argv", ["erd_search.py", "view", "RAISE",
                               "--branch-status", "unqueued,done"]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        self.assertEqual(
            run_view.call_args.args[0].filters.branch_statuses,
            ("unqueued", "done"),
        )

    def test_worker_status_filter_is_dropped_when_no_status_carries_one(self):
        cases = [
            (["--queue", "--branch-status", "queued",
              "--branch-worker-status", "active"], ()),
            (["--queue", "--branch-status", "done",
              "--branch-worker-status", "active,waiting"], ()),
            (["--queue", "--branch-status", "evaluating",
              "--branch-worker-status", "active"], ("active",)),
            (["--queue", "--branch-status", "queued,finalizing",
              "--branch-worker-status", "waiting"], ("waiting",)),
            (["--queue", "--branch-worker-status", "waiting"], ("waiting",)),
        ]
        for options, worker_statuses in cases:
            with self.subTest(options=options):
                with (
                    patch("sys.argv", ["erd_search.py", "view", *options]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                self.assertEqual(
                    run_view.call_args.args[0].filters.branch_worker_statuses,
                    worker_statuses,
                )

    def test_branch_filters_are_comma_separated_and_overview_defaults_active(self):
        cases = [
            ([], (), ("active",)),
            (["--branch-status", "evaluating,finalizing"],
             ("evaluating", "finalizing"), ("active",)),
            (["--branch-worker-status", "all"], (), ()),
            (["--branch-status", "all"], (), ("active",)),
            (["--queue"], (), ()),
        ]
        for options, statuses, worker_statuses in cases:
            with self.subTest(options=options):
                with (
                    patch("sys.argv", ["erd_search.py", "view", *options]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                filters = run_view.call_args.args[0].filters
                self.assertEqual(filters.branch_statuses, statuses)
                self.assertEqual(filters.branch_worker_statuses, worker_statuses)

    def test_hotspot_defaults_and_sample_cap_are_normalized(self):
        with (
            patch("sys.argv", [
                "erd_search.py", "view", "--hotspots", "--by", "coordination",
                "--sample-size", "2000000",
            ]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        args = run_view.call_args.args[0]
        self.assertEqual(args.report_kind, "hotspots")
        self.assertEqual(args.hotspot_field, "coordination")
        self.assertEqual(args.since_seconds, 3600)
        self.assertEqual(args.sample_size, 1_000_000)
        self.assertEqual(args.limit, 10)

    def test_claims_and_answers_reach_watch_session_request(self):
        with (
            patch("sys.argv", [
                "erd_search.py", "view", "RAISE -----",
                "--claims", "--answers",
            ]),
            patch("report_terminal.run_view") as run_view,
        ):
            erd_search.main()
        session = WatchSession(run_view.call_args.args[0])
        with patch("report_terminal.collect_report") as collect_report:
            session._collect()
        request = collect_report.call_args.args[1]
        self.assertTrue(request.include_claims)
        self.assertTrue(request.include_answers)


def root_progress_report(estimate=None, requested_at=1_798_000_000):
    return {
        "report_kind": "root_progress",
        "schema_version": 3,
        "generated_at": 1_800_000_000,
        "tree": False,
        "sources": {
            "queue": {"path": "queue.sqlite3", "ok": True, "error": None,
                      "epoch": 11, "label": "packed", "git_sha": "abcdef12"},
            "telemetry": {"path": "telemetry.sqlite3", "ok": True,
                          "error": None},
            "cache": {"path": "cache.sqlite3", "ok": True, "error": None},
        },
        "filters": {},
        "branch_target": {"kind": "word", "word": "scope"},
        "data": {
            "word": "scope",
            "word_is_answer": False,
            "spine_prefix": "SCOPE",
            "context": {"spine": []},
            "epoch": 11,
            "selected_telemetry_epoch": None,
            "telemetry_epochs": [9, 11],
            "work_started_at": 1_799_000_000,
            "work_latest_at": 1_799_900_000,
            "estimate": estimate,
            "response_groups": [
                {"pattern": "-y---", "state": "working",
                 "answer_count": 502, "started": True,
                 "branch_count": 524_184, "open_branch_count": 8_584,
                 "search_node_count": 129_900_000_000,
                 "search_node_share": 0.989, "wall_millis": 3_400_000_000,
                 "elapsed_millis": 968_000_000,
                 "first_created_at": 1, "last_finalized_at": 2},
                # Open, nothing finalized: started, with no cost yet.
                {"pattern": "-gg--", "state": "working",
                 "answer_count": 16, "started": True,
                 "branch_count": 0, "open_branch_count": 1,
                 "search_node_count": 0,
                 "search_node_share": 0.0, "wall_millis": 0,
                 "elapsed_millis": None,
                 "first_created_at": 3, "last_finalized_at": None},
                {"pattern": "--y--", "state": "waiting",
                 "answer_count": 126, "started": False,
                 "branch_count": 0, "open_branch_count": 0,
                 "search_node_count": 0,
                 "search_node_share": 0.0, "wall_millis": 0,
                 "elapsed_millis": None,
                 "first_created_at": None, "last_finalized_at": None},
            ],
            "totals": {
                "response_group_count": 117,
                "started_response_group_count": 38,
                "answer_count": 3209,
                "state_counts": {"waiting": 79, "working": 34, "solved": 3,
                                 "loss": 1},
                "branch_count": 550_292,
                "search_node_count": 131_367_632_458,
                "wall_millis": 3_471_277_846,
                "open_branch_count": 8632,
                "counted_branch_count": 10,
                "requested_at": requested_at,
                "recent_window_seconds": 86400,
            },
        },
    }


class RootProgressRendererTest(unittest.TestCase):
    def test_reports_request_time_and_work_start_as_separate_facts(self):
        output = render_report(root_progress_report(), width=120)
        self.assertIn("active-epoch=11  telemetry-epochs=9,11", output)
        self.assertIn("Requested", output)
        self.assertIn("work began", output)
        # The two differ by ~11 days here; collapsing them would report the
        # queue wait as though it were search time.
        self.assertIn("2026-", output)

    def test_shows_worst_group_share_and_both_time_bases(self):
        output = render_report(root_progress_report(), width=120)
        self.assertIn("-y---", output)
        self.assertIn("129.9G", output)
        self.assertIn("98.9%", output)
        # elapsed 11.2d of clock against 39.4d of summed worker-time.
        self.assertIn("11.2d", output)
        self.assertIn("39.4d", output)

    def test_unstarted_group_shows_no_measured_cost(self):
        output = render_report(root_progress_report(), width=120)
        unstarted = next(line for line in output.splitlines()
                         if line.startswith("--y--"))
        self.assertIn("—", unstarted)
        self.assertNotIn("0.0%", unstarted)

    def test_open_group_with_nothing_finalized_reads_as_started(self):
        # Distinct from unstarted: its zeros are measured, and only the
        # figures that exist solely at finalize stay unknown.
        output = render_report(root_progress_report(), width=120)
        fresh = next(line for line in output.splitlines()
                     if line.startswith("-gg--"))
        self.assertIn("0.0%", fresh)
        # One open branch, no finalized ones, and no worker-time yet.
        self.assertRegex(
            fresh, r"-gg--\s+working\s+16\s+0\s+1\s+0\s+0\.0%\s+—\s+—")

    def test_cumulative_branch_total_is_left_out_of_the_summary(self):
        # Carving work into a sub-branch raises the count without any answer
        # being closer to solved, so the absolute total tracks scheduling as
        # much as progress.
        output = render_report(root_progress_report(), width=120)
        summary = output.split("Pattern")[0]
        self.assertNotIn("550,292", summary)
        self.assertIn("nodes", summary)
        self.assertIn("worker-time", summary)

    def test_open_branch_count_is_its_own_column(self):
        output = render_report(root_progress_report(), width=120)
        # Named for the lifecycle phase the branches are in, so the column
        # beside it can be named for the phase they reach.
        self.assertIn("Evaluating", output.splitlines()[7])
        self.assertIn("Done", output.splitlines()[7])
        hot = next(line for line in output.splitlines()
                   if line.startswith("-y---"))
        self.assertIn("8,584", hot)

    def test_estimate_states_what_it_excludes(self):
        output = render_report(root_progress_report(estimate={
            "remaining_candidate_count": 7171,
            "recent_candidate_count": 576,
            "candidates_per_day": 576.0,
            "estimated_seconds": 1_075_650.0,
            "stalled_branch_count": 9,
            "stalled_remaining_candidate_count": 4256,
        }), width=120)
        self.assertIn("estimate ~12.4d", output)
        # Only groups still waiting hold work the estimate cannot see.
        self.assertIn("79 waiting groups", output)
        self.assertIn("9 stalled branches", output)

    def test_provisional_estimate_states_its_sample_duration(self):
        output = render_report(root_progress_report(estimate={
            "remaining_candidate_count": 7171,
            "recent_candidate_count": 576,
            "candidates_per_day": 576.0,
            "estimated_seconds": 1_075_650.0,
            "sample_duration_seconds": 600,
            "provisional": True,
            "stalled_branch_count": 0,
            "stalled_remaining_candidate_count": 0,
        }), width=120)

        self.assertIn("provisional; 10m sample", output)

    def test_absent_estimate_is_stated_rather_than_guessed(self):
        output = render_report(root_progress_report(), width=120)
        self.assertIn("estimate unavailable", output)

    def test_absent_request_time_is_omitted_rather_than_shown_as_unknown(self):
        # The queue has no trustworthy request time for this root, so the line
        # reports only what is known instead of printing a placeholder that
        # would read as a measurement.
        output = render_report(root_progress_report(requested_at=None),
                               width=120)
        self.assertNotIn("Requested", output)
        self.assertIn("work began", output)

    def test_counts_are_rendered_with_thousands_separators(self):
        output = render_report(root_progress_report(), width=120)
        self.assertIn("of 117 response groups", output)
        self.assertIn("branches evaluating 8,632", output)
        # Per-pattern branch counts stay: comparing patterns is what they are
        # for, unlike the cumulative total, which grows with every promotion.
        self.assertIn("524,184", output)


if __name__ == "__main__":
    unittest.main()
