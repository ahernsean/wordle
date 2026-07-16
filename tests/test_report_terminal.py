"""Tests for terminal rendering and refresh of shared swarm reports."""

from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import erd_search
import report_terminal
from report_terminal import DisplayOrder, WatchSession, render_overview, render_report


def overview_report():
    return {
        "schema_version": 1,
        "report_kind": "overview",
        "generated_at": 1000,
        "selector": None,
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
            "queue_counts": {
                "pending_branch_count": 12,
                "active_user_branch_count": 1,
                "active_cooperative_branch_count": 0,
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
                "lifecycle": "active",
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
        self.assertIn("queue: ok", output)
        self.assertIn("pending 12", output)
        self.assertIn("@0123456789ab", output)
        self.assertIn("GuessD", output)
        self.assertIn("25/100", output)
        self.assertNotIn("30/100", output)
        self.assertIn("MaxRD", output)
        self.assertIn("worker-2", output)
        self.assertNotIn("guess_depth=", output)
        self.assertNotIn("worker=", output)
        self.assertNotIn("\033", output)

    def test_narrow_rendering_respects_width(self):
        output = render_overview(overview_report(), color=False, width=50)
        self.assertTrue(all(len(line) <= 50 for line in output.splitlines()))
        self.assertIn("@0123456789ab", output)
        self.assertIn("worker-2", output)
        self.assertIn("Ref", output)
        self.assertIn("State", output)

    def test_progress_and_worker_changes_are_semantically_colored(self):
        previous = overview_report()
        current = deepcopy(previous)
        current["data"]["branches"][0]["completed_candidate_count"] += 1
        current["data"]["workers"][0]["candidate_index"] += 1
        output = render_overview(
            current, previous_report=previous, color=True, width=100
        )
        self.assertIn(report_terminal.GREEN + "  @0123456789ab", output)
        self.assertIn(report_terminal.RED + "    worker-2", output)

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
            report_terminal.AMBER + "    worker-2", stale_output
        )

        dead = deepcopy(stale)
        dead["data"]["workers"][0]["is_live"] = False
        dead_output = render_overview(dead, color=True, width=100)
        self.assertIn(
            report_terminal.RED + "  worker-2", dead_output
        )

    def test_adaptive_columns_cover_phone_and_wide_widths(self):
        report = overview_report()
        branch = report["data"]["branches"][0]
        branch["candidate_count"] = 12972
        branch["completed_candidate_count"] = 12616
        branch["spine"] = branch["spine"] * 5
        report["data"]["workers"][0]["worker_id"] = "worker-12"

        expected_branch_headings = {
            50: ("Ref", "GuessD", "State", "Done", "W"),
            55: ("Ref", "GuessD", "State", "Done", "W", "Ans"),
            59: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            60: ("Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk"),
            79: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD",
            ),
            80: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD",
            ),
            120: (
                "Ref", "GuessD", "State", "Done", "W", "Ans", "Bulk",
                "Best", "MaxRD", "ETA",
            ),
        }
        for width, expected_headings in expected_branch_headings.items():
            with self.subTest(width=width):
                output = render_overview(report, color=False, width=width)
                self.assertTrue(
                    all(len(line) <= width for line in output.splitlines())
                )
                self.assertIn("@0123456789ab", output)
                self.assertIn("12616/12972", output)
                self.assertIn("worker-12", output)
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
        self.assertLess(output.index("@0123456789ab"), output.index("@bbbbbbbbbbbb"))

    def test_unavailable_queue_still_renders_cache(self):
        report = overview_report()
        report["sources"]["queue"].update({"ok": False, "error": "locked"})
        output = render_overview(report, width=100)
        self.assertIn("queue: unavailable: locked", output)
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
                    "lifecycle": "unqueued", "priority": None, "worker_count": 0,
                    "cache_state": "missing", "best_guess": None,
                    "best_erd": None, "max_remaining_depth": None,
                    "updated_at": None,
                },
                {
                    "pattern": "y----", "answer_count": 4,
                    "branch_reference": "bbbbbbbbbbbb", "branch_key_hex": "bb",
                    "lifecycle": "unqueued", "priority": None, "worker_count": 0,
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
        self.assertLess(output.index("@aaaaaaaaaaaa"), output.index("@bbbbbbbbbbbb"))

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
        self.assertLess(
            output.index("worker-2"), output.index("worker-1")
        )

    def test_watched_branch_claims_compare_by_candidate_index(self):
        report = overview_report()
        report["report_kind"] = "branch"
        report["data"] = {
            "branch": {
                "branch_reference": "0123456789ab", "branch_key_hex": "010203",
                "spine": [], "guess_depth": 0, "answer_count": 3, "budget": 6,
            },
            "queue": None,
            "cache": {
                "cache_state": "missing", "best_guess": None,
                "best_erd": None, "max_remaining_depth": None,
            },
            "workers": [],
            "republished_candidates": [],
            "claims": [{
                "candidate_index": 4, "state": "in_flight",
                "completion_kind": None, "worker_id": "worker-2",
                "bundle_id": None, "claimed_at": 900, "done_at": None,
                "republish_count": 0,
            }],
            "provenance_unknown": False,
        }
        changed = deepcopy(report)
        changed["data"]["claims"][0]["state"] = "done"
        changed["data"]["claims"][0]["completion_kind"] = "evaluated"
        output = render_report(
            changed, previous_report=report, color=True, width=100
        )
        self.assertIn(report_terminal.RED + "  idx=4 done", output)

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


class CollectionRendererTest(unittest.TestCase):
    def _report(self, report_kind, data, tree=False):
        report = overview_report()
        report.update({
            "report_kind": report_kind,
            "selector": {
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
                "lifecycle": "active" if word is not None else None,
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
            "summary": {"branch_count_by_lifecycle": {"active": 1}},
            "matched_rows": 1,
            "rows": [{
                "branch_reference": "0123456789ab",
                "lifecycle": "active",
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
        workers_output = render_report(workers_report, width=120)
        self.assertIn("Worker", workers_output)
        self.assertIn("worker-2", workers_output)
        self.assertIn("stale", workers_output)

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


class ViewParserTest(unittest.TestCase):
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

    def test_queue_and_cache_positionals_are_word_selectors(self):
        for word in ("QUEUE", "CACHE"):
            with self.subTest(word=word):
                with (
                    patch("sys.argv", ["erd_search.py", "view", word]),
                    patch("report_terminal.run_view") as run_view,
                ):
                    erd_search.main()
                selector = run_view.call_args.args[0].selector
                self.assertEqual(selector.kind, "word")
                self.assertEqual(selector.trailing_word, word.lower())

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

    def test_tree_cache_and_conflicting_lifecycle_filters_are_rejected(self):
        invalid_arguments = [
            ["erd_search.py", "view", "--cache", "--tree"],
            ["erd_search.py", "view", "--active-only", "--status", "active"],
            ["erd_search.py", "view", "--queue", "--claims"],
            ["erd_search.py", "view", "--tree", "--answers"],
            ["erd_search.py", "view", "--hotspots", "--tree"],
            ["erd_search.py", "view", "--hotspots", "--by", "cut-reuse",
             "--active-only"],
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
