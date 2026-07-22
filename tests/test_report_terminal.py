"""Tests for terminal rendering and refresh of shared swarm reports."""

from copy import deepcopy
import io
import json
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import erd_search
import report_terminal
from report_model import ReportFilters, parse_report_branch_target
from report_terminal import DisplayOrder, WatchSession, render_overview, render_report


def overview_report():
    return {
        "schema_version": 2,
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
            },
            "branches": [{
                "branch_reference": "0123456789ab",
                "branch_key_hex": "010203",
                "branch_status": "active",
                "branch_phase": "evaluating",
                "raw_status": "in_progress",
                "answer_count": 33,
                "candidate_count": 100,
                "completed_candidate_count": 25,
                "bulk_completed_candidate_count": 5,
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
        self.assertIn("State", output)

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

        expected_branch_headings = {
            50: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            55: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            59: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            60: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            79: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD", "ETA",
            ),
            80: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD", "ETA",
            ),
            120: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD", "ETA", "Spine",
            ),
        }
        for width, expected_headings in expected_branch_headings.items():
            with self.subTest(width=width):
                output = render_overview(report, color=False, width=width)
                self.assertTrue(
                    all(len(line) <= width for line in output.splitlines())
                )
                self.assertIn("@0123", output)
                self.assertIn("12616/12972", output)
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
        self.assertIn("State", narrow_branch_header)
        self.assertIn("Done", narrow_branch_header)
        self.assertNotIn("Spine", narrow_branch_header)

        wide = render_overview(report, color=False, width=120)
        wide_branch_header = next(
            line for line in wide.splitlines() if "Ref" in line and "GuessD" in line
        )
        self.assertIn("Best", wide_branch_header)

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
            "response_groups": [
                {
                    "pattern": "-----", "answer_count": 8,
                    "branch_reference": "aaaaaaaaaaaa", "branch_key_hex": "aa",
                    "branch_status": "unqueued", "branch_phase": None,
                    "priority": None, "worker_count": 0,
                    "cache_state": "missing", "best_guess": None,
                    "best_erd": None, "max_remaining_depth": None,
                    "updated_at": None,
                },
                {
                    "pattern": "y----", "answer_count": 4,
                    "branch_reference": "bbbbbbbbbbbb", "branch_key_hex": "bb",
                    "branch_status": "unqueued", "branch_phase": None,
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
                "branch_status": "active",
                "branch_phase": "evaluating",
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
                "branch_status": "active", "branch_phase": "evaluating",
            },
            "queue": None,
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
                "bulk_eliminated_count": 1619, "provenance_unknown_count": 0,
                "worker_contributions": [
                    {"worker_id": "worker-0", "done_count": 6484},
                    {"worker_id": "worker-2", "done_count": 6335},
                ],
            },
            "provenance_unknown": False,
        }
        output = render_report(report, width=100)
        self.assertIn("12,819 done", output)
        self.assertIn("1,619 bulk proofs", output)
        self.assertIn("5 in flight", output)
        self.assertIn("by worker: w0 6,484", output)
        self.assertNotIn("idx=", output)

    def test_selected_branch_detail_survives_parent_status_filter(self):
        report = overview_report()
        report["report_kind"] = "branch"
        report["filters"] = {
            "branch_statuses": ["active"], "branch_phases": [],
        }
        report["data"] = {
            "branch": {
                "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                "spine": [], "guess_depth": 0, "answer_count": 3, "budget": 6,
                "branch_status": "done", "branch_phase": "complete",
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
        self.assertIn("status=done phase=complete", output)
        self.assertNotIn("no longer matches the parent filter", output)

    def test_pending_queued_overview_renders_without_candidate_total(self):
        report = overview_report()
        report["filters"] = {
            "branch_statuses": ["pending"], "branch_phases": [],
        }
        branch = report["data"]["branches"][0]
        branch.update({
            "branch_status": "pending",
            "branch_phase": "queued",
            "candidate_count": None,
            "completed_candidate_count": 0,
            "created_at": None,
            "worker_count": 0,
        })
        report["data"]["workers"] = []
        output = render_report(report, width=120)
        self.assertIn("Branches (status=pending)", output)
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
            }],
        }
        output = render_report(report, width=120)
        self.assertIn("best_erd=2.793", output)
        self.assertIn("ceiling=2.000", output)
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
            "schema_version": 2,
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
                    "branch_status": "active", "branch_phase": "evaluating",
                },
                "queue": {
                    "branch_status": "active", "branch_phase": "evaluating",
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

    def test_tree_renders_each_child_beneath_its_parent(self):
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
                "branch_status": "active" if word is not None else None,
                "branch_phase": "evaluating" if word is not None else None,
                "answer_count": 2 if word is not None else None,
                "guess_depth": guess_depth,
                "worker_count": 0,
                "completed_candidate_count": 0,
                "candidate_count": 4,
                "is_context": node_id == "root",
            }

        report = self._report("queue", {
            "tree_available": True,
            "unavailable_reason": None,
            "nodes": [
                node("root", None, 0),
                node("raise:-----", "root", 1, "raise", "-----"),
                node("stink:g----", "root", 1, "stink", "g----"),
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
        self.assertLess(output.index("RAISE -----"), output.index("CRANE y----"))
        self.assertLess(output.index("CRANE y----"), output.index("STINK g----"))
        self.assertLess(output.index("STINK g----"), output.index("MOUNT -y---"))

    def test_queue_worker_and_cache_collections_are_semantically_formatted(self):
        queue_report = self._report("queue", {
            "summary": {
                "branch_count_by_status": {"active": 1},
                "branch_count_by_phase": {"evaluating": 1},
            },
            "matched_rows": 1,
            "rows": [{
                "branch_reference": "0123456789ab",
                "branch_status": "active",
                "branch_phase": "evaluating",
                "answer_count": 2,
                "spine": "RAISE ----- CRANE y----",
                "priority": 7,
                "worker_count": 1,
            }],
        })
        queue_output = render_report(queue_report, width=120)
        self.assertIn("guess_depth=2", queue_output)
        self.assertNotIn(" d=2", queue_output)

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
            branch_statuses=("active",), minimum_answer_count=10,
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
        finalizing["data"]["branches"][0]["branch_phase"] = "finalizing"
        session._update_navigation_targets(finalizing)
        self.assertEqual(session.branch_hotkeys[letter], "010203")


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
            "cmd_queue_priority", "cmd_reset_stale",
        )
        patches = [patch.object(erd_search, name) for name in handler_names]
        started_patches = [handler_patch.start() for handler_patch in patches]
        self.addCleanup(lambda: [handler_patch.stop() for handler_patch in patches])
        self.assertTrue(commands)
        for arguments in commands:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", ["erd_search.py", *arguments]):
                    erd_search.main()
        self.assertTrue(any(handler.called for handler in started_patches))

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

    def test_incompatible_report_options_and_invalid_branch_filters_are_rejected(self):
        invalid_arguments = [
            ["erd_search.py", "view", "--cache", "--tree"],
            ["erd_search.py", "view", "--branch-status", "active,active"],
            ["erd_search.py", "view", "--branch-status", "all,pending"],
            ["erd_search.py", "view", "--branch-phase", "unknown"],
            ["erd_search.py", "view", "--queue", "--claims"],
            ["erd_search.py", "view", "--tree", "--answers"],
            ["erd_search.py", "view", "--hotspots", "--tree"],
            ["erd_search.py", "view", "--hotspots", "--by", "cut-reuse",
             "--branch-status", "active"],
            ["erd_search.py", "view", "--hotspots", "--by", "coordination",
             "RAISE"],
            ["erd_search.py", "view", "--by", "nodes"],
            ["erd_search.py", "view", "--epoch", "2"],
            ["erd_search.py", "view", "RAISE -----", "--tree", "--claims"],
            ["erd_search.py", "view", "RAISE", "--sort", "nodes"],
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", arguments), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        erd_search.main()
                self.assertEqual(raised.exception.code, 2)

    def test_branch_filters_are_comma_separated_and_overview_defaults_active(self):
        cases = [
            ([], ("active",), ()),
            (["--branch-status", "active,pending"], ("active", "pending"), ()),
            (["--branch-status", "all"], (), ()),
            (["--branch-phase", "evaluating,finalizing"],
             ("active",), ("evaluating", "finalizing")),
            (["--queue"], (), ()),
        ]
        for options, statuses, phases in cases:
            with self.subTest(options=options):
                with (
                    patch("sys.argv", ["erd_search.py", "view", *options]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                filters = run_view.call_args.args[0].filters
                self.assertEqual(filters.branch_statuses, statuses)
                self.assertEqual(filters.branch_phases, phases)

    def test_view_help_explains_status_phase_and_lifecycle(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["erd_search.py", "view", "--help"]),
            patch("sys.stdout", output),
            self.assertRaises(SystemExit) as raised,
        ):
            erd_search.main()
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn(
            "Status describes the branch's relationship to current work",
            help_text,
        )
        self.assertIn("Phase describes the durable search milestone", help_text)
        self.assertIn("The axes are related", help_text)
        self.assertIn("pending / queued", help_text)
        self.assertIn("active / evaluating <-> pending / evaluating", help_text)
        self.assertIn("done / complete", help_text)

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


if __name__ == "__main__":
    unittest.main()
