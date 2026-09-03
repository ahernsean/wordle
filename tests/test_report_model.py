"""Tests for the shared ERD swarm report model."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from cache_sqlite import ScoreCache
from pattern_matrix import PatternMatrix
from erd_queue import ERDQueue
import erd_search
import report_model
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportSources,
    WORKER_LIVENESS_SECONDS,
    _candidate_erd_summary,
    _candidate_eta,
    _source_word_group_key,
    _grouped_response_groups,
    _response_group_is_solved,
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
    def test_disk_fill_rate_uses_fresh_samples_and_rejects_flat_time(self):
        self.assertIsNone(report_model.disk_fill_rate([(0, 100)], 0))
        self.assertIsNone(report_model.disk_fill_rate([(0, 100), (0, 90)], 0))
        self.assertEqual(
            report_model.disk_fill_rate([(0, 100), (10, 80), (-1000, 1)], 10),
            2.0,
        )

    def test_request_validation_rejects_invalid_operator_combinations(self):
        root = parse_report_branch_target(None)
        cases = (
            (ReportRequest(report_kind="cache", tree=True), "cannot be used"),
            (ReportRequest(tree_parent="RAISE -----"), "require tree"),
            (ReportRequest(raw_row_offset=1), "requires an accuracy"),
            (ReportRequest(report_kind="accuracy", raw_row_offset=-1), "cannot be negative"),
            (ReportRequest(worker_id="w1"), "workers report"),
            (ReportRequest(report_kind="root_progress", branch_target=root), "requires a target"),
        )
        for request, message in cases:
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report_request(request)

    def test_rich_spine_and_row_helpers_preserve_legacy_shapes(self):
        self.assertEqual(parse_rich_spine(None), [])
        parsed = parse_rich_spine("2:RAISE:-----/4→-y---/2")
        self.assertEqual(parsed, [(2, "RAISE", "-----", "4"), (None, None, "-y---", "2")])
        normalized = normalize_worker_descent(parsed, {"raise"})
        self.assertTrue(normalized[0]["word_is_answer"])
        self.assertFalse(normalized[1]["word_is_answer"])
        self.assertEqual(report_model._row_value(None, "missing", "fallback"), "fallback")
        self.assertEqual(report_model._row_value({}, "missing", "fallback"), "fallback")

    def test_branch_target_parser_rejects_invalid_reference_and_word_tokens(self):
        for target, message in (
            ("@xyz", "expected @"),
            ("toolong", "five-letter word"),
            ("raise wrong", "response pattern"),
        ):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, message):
                    parse_report_branch_target(target)

    def test_request_validation_rejects_filter_scope_mismatches(self):
        word = parse_report_branch_target("raise")
        branch = parse_report_branch_target("raise -----")
        cases = (
            (ReportRequest(filters=ReportFilters(source_states=("queued",))), "opener_state"),
            (ReportRequest(filters=ReportFilters(source_offset=0)), "source_offset"),
            (ReportRequest(filters=ReportFilters(sort="word")), "requires an opener"),
            (ReportRequest(branch_target=branch, filters=ReportFilters(branch_statuses=("unqueued",))), "unqueued"),
            (ReportRequest(report_kind="hotspots", hotspot_field="coordination", branch_target=word), "coordination"),
            (ReportRequest(report_kind="openers", branch_target=branch), "accepts only"),
        )
        for request, message in cases:
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report_request(request)

    def test_filter_and_lifecycle_helpers_cover_normalization_and_errors(self):
        self.assertEqual(report_model.parse_branch_filter(" queued , done ", "status", ("queued", "done")), ("queued", "done"))
        self.assertEqual(report_model.parse_branch_filter("all", "status", ("queued",)), ())
        for value, message in (("", "comma"), ("all,queued", "combined"), ("nope", "unknown"), ("queued,queued", "duplicate")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    report_model.parse_branch_filter(value, "status", ("queued",))
        filters = ReportFilters(branch_statuses=("done",), branch_worker_statuses=("active",))
        self.assertEqual(report_model.applied_branch_filters(filters).branch_worker_statuses, ())
        active = report_model.applied_branch_filters(
            replace(filters, branch_statuses=("evaluating",)))
        self.assertEqual(active.branch_worker_statuses, ("active",))
        self.assertEqual(report_model.branch_status_and_worker_status("pending", None, 0), ("queued", None))
        self.assertEqual(report_model.branch_status_and_worker_status(None, "open", 1), ("evaluating", "active"))
        self.assertEqual(report_model._response_group_key({}, "none"), (0, "all"))
        self.assertEqual(report_model._root_progress_group_state(
            {"answer_count": 3, "started": False},
            {"best_erd": 2, "max_remaining_depth": 3, "cache_state": "exact"}, 2), "solved")
        self.assertEqual(report_model._root_progress_group_state(
            {"answer_count": 3, "started": False},
            {"best_erd": None, "max_remaining_depth": None, "cache_state": "loss"}, 2), "loss")
        self.assertEqual(report_model.branch_status_and_worker_status(None, "finalized", 0), ("finalizing", "waiting"))

    def test_branch_reference_resolution_and_ambiguity_report(self):
        key = ScoreCache.encode_subset(["salet", "crane"])
        queue = Mock()
        queue.branch_rows_for_reference_prefix.return_value = []
        cache = Mock()
        cache.branch_keys_for_reference_prefix.return_value = []
        with self.assertRaisesRegex(ValueError, "No queued"):
            resolve_branch_reference(queue, "abcd", cache)
        cache.branch_keys_for_reference_prefix.return_value = [key, b"other"]
        with self.assertRaisesRegex(ValueError, "ambiguous") as raised:
            resolve_branch_reference(queue, "abcd", cache)
        self.assertEqual(len(raised.exception.candidates), 2)

        request = ReportRequest(branch_target=parse_report_branch_target("@abcd"))
        error = ValueError("ambiguous")
        error.candidates = [{"branch_reference": "abcd", "branch_key": key, "spine": None}]
        base = {"sources": {"queue": {}, "cache": {}}}
        with (
            patch("report_model._decorative_answer_set", return_value={"salet", "crane"}),
            patch("report_model._semantic_report", side_effect=lambda *_args: {
                **base, "data": _args[4],
            }),
        ):
            report = report_model.collect_ambiguous_branch_reference_report(
                self.sources, request, error)
        self.assertEqual(report["data"]["candidates"][0]["answer_count"], 2)
        self.assertTrue(report["sources"]["queue"]["ok"])

    def test_openers_filter_offsets_and_sort_validation(self):
        for filters, message in (
            (ReportFilters(source_offset=-1), "cannot be negative"),
            (ReportFilters(branch_row_offset=-1), "cannot be negative"),
            (ReportFilters(sort="nope"), "must be one"),
        ):
            with self.subTest(filters=filters):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report_request(ReportRequest(report_kind="openers", filters=filters))
        allowed = ReportFilters(source_offset=0, branch_row_offset=0, sort="word")
        validate_report_request(ReportRequest(report_kind="openers", filters=allowed))

    def test_tree_layout_handles_empty_legacy_and_paged_topology(self):
        request = ReportRequest(
            report_kind="queue", tree=True,
            filters=ReportFilters(limit=1),
        )
        self.assertFalse(report_model._tree_layout(
            [], request, "", [], set())["tree_available"])
        legacy_key = ScoreCache.encode_subset(["salet"])
        rows = [
            {
                "branch_key": legacy_key, "branch_key_hex": legacy_key.hex(),
                "budget": 4, "branch_status": "evaluating",
                "branch_worker_status": "active", "answer_count": 1,
                "worker_count": 2, "priority": 4,
                "completed_candidate_count": 2, "candidate_count": 3,
            },
            {
                "branch_key": ScoreCache.encode_subset(["crane"]),
                "branch_key_hex": ScoreCache.encode_subset(["crane"]).hex(),
                "spine": "RAISE ----- CRANE y----", "branch_status": "queued",
                "branch_worker_status": None, "answer_count": 1,
                "worker_count": 0, "priority": 2,
                "completed_candidate_count": 0, "candidate_count": 3,
            },
        ]
        layout = report_model._tree_layout(rows, request, "", rows, set(ANSWERS))
        self.assertTrue(layout["tree_available"])
        self.assertEqual(layout["paging"]["next_cursor"], "unknown:1:" + legacy_key.hex())
        self.assertEqual(layout["nodes"][0]["guess_depth"], 1)
        second_page = report_model._tree_layout(
            rows, replace(request, tree_cursor=layout["paging"]["next_cursor"]),
            "", rows, set(ANSWERS))
        self.assertEqual(second_page["paging"]["returned_group_count"], 1)

    def test_accuracy_report_normalizes_each_calibration_collection(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        queue = Mock(epoch=9)
        queue.report_candidate_accuracy.return_value = {
            "rows": [{"branch_key": branch_key}],
            "largest_under_predicted": [{"branch_key": branch_key}],
            "largest_over_predicted": [{"branch_key": branch_key}],
            "requested_sample_size": 2,
        }
        request = ReportRequest(report_kind="accuracy", sample_size=2)
        with patch("report_model._open_report_queue", return_value=queue):
            report = report_model.collect_accuracy_report(self.sources, request)
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertEqual(report["data"]["rows"][0]["branch_key_hex"], branch_key.hex())
        self.assertIn("branch_reference", report["data"]["largest_under_predicted"][0])
        queue.close.assert_called_once()

    def test_accuracy_and_historical_hotspot_reports_surface_queue_errors(self):
        queue = Mock(epoch=9)
        queue.report_candidate_accuracy.side_effect = sqlite3.OperationalError("offline")
        with patch("report_model._open_report_queue", return_value=queue):
            accuracy = report_model.collect_accuracy_report(
                self.sources, ReportRequest(report_kind="accuracy"))
        self.assertEqual(accuracy["sources"]["queue"]["error"], "offline")

        branch_key = ScoreCache.encode_subset(["salet"])
        queue = Mock(epoch=9)
        queue.report_hotspots.return_value = {
            "population": "candidate_claims", "epoch": 9, "since": 4,
            "sample_size": 10, "sampled_row_count": 1, "sample_truncated": False,
            "rows": [{"branch_key": branch_key, "best_guess": "salet"}],
        }
        request = ReportRequest(report_kind="hotspots", hotspot_field="cut-reuse")
        with patch("report_model._open_report_queue", return_value=queue):
            hotspots = report_model.collect_hotspot_report(self.sources, request)
        self.assertEqual(hotspots["data"]["rows"][0]["branch_key_hex"], branch_key.hex())

    def test_workers_report_filters_normalized_worker_rows(self):
        queue = Mock()
        queue.report_queue_rows.return_value = {"rows": [{
            "branch_key_hex": "a", "branch_status": "evaluating",
        }]}
        queue.heartbeats_with_branch.return_value = [{"worker_id": "worker-2"}]
        worker = {
            "worker_id": "worker-2", "worker_number": "2",
            "branch_key_hex": "a", "is_live": True, "updated_at": 1,
        }
        request = ReportRequest(report_kind="workers", worker_id="2")
        with (
            patch("report_model._open_report_queue", return_value=queue),
            patch("report_model._normalize_worker", return_value=worker),
            patch("report_model.worker_state", return_value="working"),
            patch("report_model.load_word_list", return_value=ANSWERS),
        ):
            report = collect_workers_report(self.sources, request)
        self.assertEqual(report["data"]["summary"]["worker_count"], 1)
        self.assertEqual(report["data"]["rows"][0]["state"], "working")

    def test_cache_report_collects_recent_cache_rows_without_a_live_queue(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        cache = Mock()
        cache.report_recent_rows.return_value = [{
            "branch_key": branch_key, "best_guess": "salet",
        }]
        cache.erd_report_summary.return_value = {"exact_branch_count": 1}
        cache.report_cache_distributions.return_value = {"answer_count": []}
        with (
            patch("report_model._open_report_queue", side_effect=sqlite3.OperationalError("queue offline")),
            patch("report_model.ScoreCache", return_value=cache),
            patch("report_model.load_word_list", return_value=ANSWERS),
        ):
            report = report_model.collect_cache_report(
                self.sources, ReportRequest(report_kind="cache"))
        self.assertEqual(report["sources"]["queue"]["error"], "queue offline")
        self.assertTrue(report["sources"]["cache"]["ok"])
        self.assertEqual(report["data"]["recent_rows"][0]["branch_reference"], branch_reference(branch_key))
        cache.close.assert_called_once()

    def test_cache_report_keeps_a_queue_report_when_cache_open_fails(self):
        queue = Mock()
        with (
            patch("report_model._open_report_queue", return_value=queue),
            patch("report_model.ScoreCache", side_effect=sqlite3.OperationalError("cache offline")),
            patch("report_model.load_word_list", return_value=ANSWERS),
        ):
            report = report_model.collect_cache_report(
                self.sources, ReportRequest(report_kind="cache"))
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertEqual(report["sources"]["cache"]["error"], "cache offline")
        queue.close.assert_called_once()

    def test_cache_word_report_lists_response_group_cache_states(self):
        report = report_model.collect_cache_report(
            self.sources, ReportRequest(
                report_kind="cache", branch_target=parse_report_branch_target("RAISE")))
        self.assertTrue(report["sources"]["cache"]["ok"])
        self.assertGreater(report["data"]["summary"]["response_group_count"], 0)
        self.assertIn("branch_reference", report["data"]["rows"][0])

    def test_cache_branch_report_uses_the_spine_budget(self):
        report = report_model.collect_cache_report(
            self.sources, ReportRequest(
                report_kind="cache", branch_target=parse_report_branch_target("RAISE -----")))
        self.assertIn("cache", report["data"])
        self.assertIn("branch_reference", report["data"])

    def test_branch_report_collects_an_unqueued_spine_without_creating_work(self):
        request = ReportRequest(
            report_kind="branch", branch_target=parse_report_branch_target("RAISE -----"),
            include_answers=True,
        )
        report = report_model.collect_branch_report(self.sources, request)
        self.assertIn(report["data"]["branch"]["branch_status"], ("done", "unqueued"))
        self.assertIsNone(report["data"]["queue"])
        self.assertTrue(report["sources"]["queue"]["ok"])

    def test_branch_reference_report_falls_back_to_cache_when_queue_is_unavailable(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        cache = Mock()
        cache.report_branch_state.return_value = {
            "cache_state": "exact", "best_guess": "salet",
            "best_erd": 1.0, "max_remaining_depth": 1,
        }
        cache_class = Mock(return_value=cache)
        cache_class.encode_subset = ScoreCache.encode_subset
        cache_class.report_branch_state_without_rows.return_value = {
            "cache_state": "missing", "best_guess": None,
            "best_erd": None, "max_remaining_depth": None,
        }
        referenced_row = {"branch_key": branch_key}
        request = ReportRequest(
            report_kind="branch",
            branch_target=parse_report_branch_target("@" + branch_reference(branch_key)),
        )
        with (
            patch("report_model.ScoreCache", cache_class),
            patch("report_model._open_report_queue", side_effect=sqlite3.OperationalError("offline")),
            patch("report_model.resolve_branch_reference", return_value=referenced_row),
        ):
            report = report_model.collect_branch_report(self.sources, request)
        self.assertEqual(report["sources"]["queue"]["error"], "offline")
        self.assertEqual(report["data"]["cache"]["cache_state"], "exact")

    def test_branch_report_preserves_queue_data_when_cache_read_fails(self):
        request = ReportRequest(
            report_kind="branch", branch_target=parse_report_branch_target("RAISE -----"),
        )
        cache = Mock()
        cache.report_branch_state.side_effect = sqlite3.OperationalError("cache offline")
        cache_class = Mock(return_value=cache)
        cache_class.encode_subset = ScoreCache.encode_subset
        cache_class.report_branch_state_without_rows.return_value = {
            "cache_state": "missing", "best_guess": None,
            "best_erd": None, "max_remaining_depth": None,
        }
        with patch("report_model.ScoreCache", cache_class):
            report = report_model.collect_branch_report(self.sources, request)
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertEqual(report["sources"]["cache"]["error"], "cache offline")

    def test_root_progress_report_handles_an_opener_without_started_work(self):
        report = report_model.collect_root_progress_report(
            self.sources, ReportRequest(
                report_kind="root_progress",
                branch_target=parse_report_branch_target("RAISE")))
        self.assertEqual(report["data"]["word"], "raise")
        self.assertEqual(report["data"]["estimate"], None)

    def test_opener_report_persists_missing_completed_timing(self):
        queue = Mock()
        queue.source_word_rows.return_value = [{"source_word": "raise", "state": "done"}]
        queue.source_membership_rows.return_value = []
        queue.completed_source_timing.return_value = {
            "completed_at": 20, "first_created_at": 10,
            "worker_millis": 4, "telemetry_epochs": "2,4",
        }
        queue.distinct_branch_count_for_words.return_value = 0
        timing_cache = Mock()
        timing_cache.completed_source_summary_map.return_value = {}
        payload = {"source_word": "raise", "state": "complete", "branch_count": 0,
                   "completed_at": 20, "requested_priority": 0}
        with (
            patch("report_model._open_report_queue", return_value=queue),
            patch("report_model.ScoreCache", return_value=timing_cache),
            patch("report_model.load_word_list", return_value=ANSWERS),
            patch("report_model._source_rollups", return_value={"raise": {}}),
            patch("report_model._source_summary_payload", return_value=payload),
            patch("report_model._source_word_erd_summaries", return_value={}),
        ):
            report = collect_source_report(self.sources, ReportRequest(report_kind="openers"))
        timing_cache.write_completed_source_summary.assert_called_once()
        self.assertEqual(report["data"]["summary"], [payload])

    def test_response_group_scale_skips_empty_and_unbuilt_matrices(self):
        self.assertIsNone(report_model._maximum_response_group_count(
            self.sources, ANSWERS, [], b"key", None))
        with patch("report_model.PatternMatrix.load_if_built", return_value=None):
            self.assertIsNone(report_model._maximum_response_group_count(
               self.sources, ANSWERS, ANSWERS, b"other", None))

    def test_source_summary_and_group_helpers_cover_active_and_empty_states(self):
        row = {"source_word": "raise", "branch_count": 3,
               "requested_priority": 4, "request_count": 1,
               "direct_branch_count": 2, "direct_done_branch_count": 1,
               "has_incomplete_request": 1, "has_active_request": 1,
               "started_at": 10}
        payload = report_model._source_summary_payload(
            row, {"open_branch_count": 2, "worker_count": 1},
            {"completed_at": None, "elapsed_millis": None, "worker_millis": None}, 20)
        self.assertEqual(payload["state"], "active")
        self.assertEqual(payload["elapsed_millis"], 10_000)
        self.assertEqual(report_model._source_erd_sort_key({"source_word": "raise"})[0], 3)
        self.assertEqual(report_model._duration_group_key(31 * 24 * 60 * 60 * 1000)[1], "[1 month, ∞)")
        self.assertEqual(report_model._source_word_group_key(
            {"worker_count": 0}, "worker_presence", 0), (1, "no workers"))
        self.assertEqual(report_model._completed_at_group_key(None, 0), (5, "not completed"))
        self.assertEqual(report_model._source_word_group_key(
            {"elapsed_millis": None}, "elapsed", 0), (5, "not completed"))

    def test_source_rollups_and_membership_payload_preserve_shared_branch_context(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        parent_key = ScoreCache.encode_subset(["crane"])
        row = {
            "branch_id": 3, "branch_key": branch_key, "parent_branch_key": parent_key,
            "source_work_id": 7, "source_word": "raise", "requested_priority": 4,
            "source_state": "active", "root_pattern": 0,
            "pending_status": "in_progress", "active_status": "open", "worker_count": 2,
            "branch_effective_priority": 6,
        }
        rollup = report_model._source_rollups([row, row])["raise"]
        self.assertEqual(rollup["open_branch_count"], 1)
        payload = report_model._source_membership_payload(row, 2)
        self.assertTrue(payload["is_shared"])
        self.assertEqual(payload["parent_branch_reference"], branch_reference(parent_key))

    def test_source_erd_summary_reuses_the_current_cache_generation(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        cache = Mock(answer_list_id="answers")
        cache.report_branch_states.return_value = {
            branch_key: {"best_erd": 1.0, "max_remaining_depth": 1, "cache_state": "exact"},
        }
        response_cache = Mock()
        response_cache.group_words.return_value = {0: ["salet"]}
        report_model._SOURCE_ERD_SUMMARY_CACHE = {}
        report = {"sources": {"cache": {"ok": False, "error": None}}}
        with (
            patch("report_model.ResponseCache", return_value=response_cache),
            patch("report_model._score_cache_file_signature", return_value=((1, 1), None)),
        ):
            first = report_model._source_word_erd_summaries(
                self.sources, ["raise"], report, cache, ANSWERS)
            second = report_model._source_word_erd_summaries(
                self.sources, ["raise"], report, cache, ANSWERS)
        self.assertTrue(report["sources"]["cache"]["ok"])
        self.assertEqual(first, second)
        self.assertEqual(response_cache.group_words.call_count, 1)

    def test_queue_and_current_hotspot_reports_normalize_rows(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        queue = Mock(epoch=12)
        queue.report_queue_rows.return_value = {"rows": [{
            "branch_key": branch_key, "branch_status": "evaluating",
            "branch_worker_status": "active", "spine": "RAISE -----",
            "best_guess": "salet",
        }]}
        with patch("report_model._open_report_queue", return_value=queue):
            queue_report = report_model.collect_queue_report(
                self.sources, ReportRequest(report_kind="queue"))
        row = queue_report["data"]["rows"][0]
        self.assertEqual(row["branch_reference"], branch_reference(branch_key))
        self.assertTrue(row["best_guess_is_answer"])

        queue = Mock(epoch=12)
        queue.report_queue_rows.return_value = {"rows": [{
            "branch_key": branch_key, "branch_key_hex": branch_key.hex(),
            "branch_status": "evaluating", "branch_worker_status": "active",
            "spine": "RAISE -----", "best_guess": "salet",
        }]}
        with patch("report_model._open_report_queue", return_value=queue):
            hotspots = report_model.collect_hotspot_report(
                self.sources, ReportRequest(report_kind="hotspots", hotspot_field="nodes"))
        self.assertEqual(hotspots["data"]["population"], "current_queue_branches")
        self.assertEqual(len(hotspots["data"]["rows"]), 1)

    def test_tree_report_limits_rows_to_the_selected_worker(self):
        selected_key = ScoreCache.encode_subset(["salet"])
        other_key = ScoreCache.encode_subset(["crane"])
        queue = Mock()
        queue.report_queue_rows.return_value = {"rows": [
            {"branch_key_hex": selected_key.hex()},
            {"branch_key_hex": other_key.hex()},
        ]}
        queue.heartbeats_with_branch.return_value = [{
            "worker_id": "worker-2", "current_branch_key": selected_key,
        }]
        layout = {"tree_available": True, "nodes": ["selected"]}
        request = ReportRequest(report_kind="queue", tree=True, worker_id="2")
        with (
            patch("report_model._open_report_queue", return_value=queue),
            patch("report_model._tree_layout", return_value=layout) as tree_layout,
        ):
            report = report_model.collect_tree_report(self.sources, request)
        self.assertEqual(report["data"]["nodes"], ["selected"])
        self.assertEqual(len(tree_layout.call_args.args[0]), 1)

    def test_branch_target_queue_scopes_and_row_matching_cover_each_target_kind(self):
        branch_key = ScoreCache.encode_subset(["salet"])
        queue = Mock()
        queue.branch_rows_for_reference_prefix.return_value = [{
            "branch_key": branch_key, "spine": "RAISE -----",
        }]
        reference_target = parse_report_branch_target("@" + branch_reference(branch_key))
        scope, prefix = report_model._branch_target_queue_scope(reference_target, queue)
        self.assertEqual(scope["branch_key"], branch_key)
        self.assertEqual(prefix, "RAISE -----")
        word_target = parse_report_branch_target("RAISE")
        scope, prefix = report_model._branch_target_queue_scope(word_target)
        self.assertEqual(scope["source_word"], "raise")
        self.assertEqual(prefix, "RAISE")
        self.assertTrue(report_model._row_matches_branch_target(
            {"spine": "RAISE ----- CRANE y----"}, word_target, prefix))
        branch_target = parse_report_branch_target("RAISE -----")
        self.assertTrue(report_model._row_matches_branch_target(
            {"spine": "RAISE ----- CRANE y----"}, branch_target, "RAISE -----"))
        self.assertTrue(report_model._row_matches_branch_target(
            {"branch_key": branch_key}, reference_target, ""))

    def test_report_dispatches_each_explicit_collection_kind(self):
        request = ReportRequest(report_kind="queue")
        with patch("report_model.collect_queue_report", return_value={"kind": "queue"}):
            self.assertEqual(collect_report(self.sources, request), {"kind": "queue"})
        for kind, function_name in (
            ("workers", "collect_workers_report"), ("cache", "collect_cache_report"),
            ("hotspots", "collect_hotspot_report"), ("accuracy", "collect_accuracy_report"),
            ("leaderboard", "collect_leaderboard_report"), ("openers", "collect_source_report"),
            ("root_progress", "collect_root_progress_report"),
        ):
            with self.subTest(kind=kind), patch("report_model." + function_name,
                                                return_value={"kind": kind}):
                self.assertEqual(collect_report(
                    self.sources, ReportRequest(report_kind=kind)), {"kind": kind})
        with self.assertRaisesRegex(ValueError, "tree layout"):
            collect_report(self.sources, ReportRequest(report_kind="cache", tree=True))
        with patch("report_model.collect_overview_report", return_value={"kind": "overview"}):
            self.assertEqual(collect_report(self.sources, ReportRequest()), {"kind": "overview"})
        with patch("report_model.collect_word_report", return_value={"kind": "word"}):
            self.assertEqual(collect_report(self.sources, ReportRequest(
                branch_target=parse_report_branch_target("RAISE"))), {"kind": "word"})

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

    def test_candidate_eta_estimates_prune_checks_and_full_evaluations(self):
        eta = _candidate_eta(
            {"candidate_count": 1_000, "completed_candidate_count": 400},
            {
                "best_updated_at": 900,
                "window_started_at": 900,
                "inspected_candidate_count": 200,
                "pruned_candidate_count": 150,
                "inspection_worker_millis": 20_000,
                "evaluated_candidate_count": 50,
                "evaluation_worker_millis": 50_000,
            },
            live_worker_count=2,
            now=1_600,
        )
        self.assertEqual(eta["state"], "ready")
        self.assertEqual(eta["remaining_inspection_count"], 600)
        self.assertEqual(eta["expected_full_evaluation_count"], 150)
        self.assertEqual(eta["estimated_seconds"], 105)

    def test_candidate_eta_withholds_an_estimate_it_cannot_support(self):
        base = {
            "best_updated_at": 900, "window_started_at": 900,
            "inspected_candidate_count": 200, "pruned_candidate_count": 150,
            "inspection_worker_millis": 20_000,
            "evaluated_candidate_count": 50,
            "evaluation_worker_millis": 50_000,
        }
        work = {"candidate_count": 1_000, "completed_candidate_count": 400}

        def eta(payload=None, **sample):
            return _candidate_eta(
                payload or work, {**base, **sample},
                live_worker_count=2, now=1_600,
            )

        # Nothing left to evaluate is not a slow estimate, it is no estimate.
        self.assertIsNone(
            eta({"candidate_count": 400, "completed_candidate_count": 400})
        )
        # A timing the sample never observed would divide by zero, so the
        # estimate is withheld instead.
        for sample in (
            {"inspection_worker_millis": 0},
            {"evaluated_candidate_count": 0},
            {"inspected_candidate_count": 0, "pruned_candidate_count": 0,
             "evaluated_candidate_count": 0},
        ):
            with self.subTest(sample=sample):
                self.assertNotEqual(eta(**sample)["state"], "ready")

    def test_candidate_eta_expects_no_evaluations_when_every_check_prunes(self):
        eta = _candidate_eta(
            {"candidate_count": 1_000, "completed_candidate_count": 400},
            {
                "best_updated_at": 900, "window_started_at": 900,
                "inspected_candidate_count": 200,
                "pruned_candidate_count": 200,
                "inspection_worker_millis": 20_000,
                "evaluated_candidate_count": 50,
                "evaluation_worker_millis": 50_000,
            },
            live_worker_count=2, now=1_600,
        )
        self.assertEqual(eta["expected_full_evaluation_count"], 0)
        self.assertEqual(eta["state"], "ready")

    def test_candidate_eta_learns_after_a_new_best_erd(self):
        eta = _candidate_eta(
            {"candidate_count": 1_000, "completed_candidate_count": 400},
            {
                "best_updated_at": 1_000,
                "window_started_at": 1_000,
                "inspected_candidate_count": 99,
                "pruned_candidate_count": 99,
                "inspection_worker_millis": 9_900,
                "evaluated_candidate_count": 0,
                "evaluation_worker_millis": 0,
            },
            live_worker_count=2,
            now=1_119,
        )
        self.assertEqual(eta["state"], "learning")

    def test_candidate_eta_estimates_without_a_best_erd(self):
        eta = _candidate_eta(
            {"candidate_count": 100, "completed_candidate_count": 10},
            {
                "best_updated_at": None,
                "window_started_at": 1_000,
                "inspected_candidate_count": 0,
                "pruned_candidate_count": 0,
                "inspection_worker_millis": 0,
                "evaluated_candidate_count": 10,
                "evaluation_worker_millis": 20_000,
            },
            live_worker_count=2,
            now=1_200,
        )
        self.assertEqual(eta["state"], "rough")
        self.assertEqual(eta["remaining_inspection_count"], 0)
        self.assertEqual(eta["expected_full_evaluation_count"], 90)
        self.assertEqual(eta["estimated_seconds"], 90)

    def test_candidate_eta_scales_a_rough_sample_to_new_workers(self):
        eta = _candidate_eta(
            {"candidate_count": 100, "completed_candidate_count": 0},
            {
                "best_updated_at": 1_000,
                "window_started_at": 1_000,
                "inspected_candidate_count": 0,
                "pruned_candidate_count": 0,
                "inspection_worker_millis": 0,
                "evaluated_candidate_count": 10,
                "evaluation_worker_millis": 24_000,
                "evaluation_worker_count": 1,
                "evaluation_worker_count_min": 1,
                "evaluation_worker_count_max": 1,
            },
            live_worker_count=4,
            now=1_200,
        )
        self.assertEqual(eta["state"], "rough")
        self.assertAlmostEqual(eta["estimated_seconds"], 240 / 1.97)
        self.assertTrue(eta["worker_count_changed"])
        self.assertEqual(eta["sample_worker_count"], 1)

    def test_candidate_eta_does_not_scale_legacy_global_worker_counts(self):
        eta = _candidate_eta(
            {"candidate_count": 100, "completed_candidate_count": 0},
            {
                "best_updated_at": None,
                "window_started_at": 1_000,
                "inspected_candidate_count": 0,
                "pruned_candidate_count": 0,
                "inspection_worker_millis": 0,
                "evaluated_candidate_count": 10,
                "evaluation_worker_millis": 24_000,
                "evaluation_unknown_worker_count": 10,
            },
            live_worker_count=2,
            now=1_700,
        )
        self.assertEqual(eta["state"], "rough")
        self.assertFalse(eta["worker_count_changed"])
        self.assertFalse(eta["worker_count_sample_complete"])

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

    def test_candidate_erd_summary_folds_the_groups_it_is_handed(self):
        summary = _candidate_erd_summary(self._SALET_GROUPS, 5)
        self.assertEqual(summary["state"], "complete")
        self.assertAlmostEqual(summary["erd"], 1.75)
        self.assertEqual(summary["resolved_group_count"], 4)
        self.assertEqual(summary["response_group_count"], 4)

    def test_candidate_erd_summary_holds_no_state_between_calls(self):
        # The fold is a pure function of the groups handed to it.  Folding a
        # candidate complete and then folding the same candidate again with one
        # group unresolved must report the second answer, not the first: that
        # is the whole difference between deriving the value and memoising it.
        # The changed group holds two answers, so its state genuinely turns on
        # a cached branch result rather than on playing a lone survivor.
        groups = [self._group("ggggg", 1, None, None),
                  self._group("-----", 2, 1.5, 2)]
        complete = _candidate_erd_summary(groups, 5)
        self.assertEqual(complete["state"], "complete")
        self.assertAlmostEqual(complete["erd"], 2.0)
        degraded = _candidate_erd_summary(
            [groups[0],
             self._group("-----", 2, None, None, cache_state="missing")],
            5,
        )
        self.assertEqual(degraded["state"], "pending")
        self.assertIsNone(degraded["erd"])
        self.assertEqual(degraded["resolved_group_count"], 1)
        self.assertEqual(degraded["response_group_count"], 2)

    def test_leaderboard_refolds_every_candidate_on_every_build(self):
        # No candidate's ERD survives a build.  Each row is folded from the
        # branch results as they stand when the leaderboard is asked for, so a
        # cache that has not changed gives the same ranking by recomputing it,
        # never by reading a stored fold back.
        sources = self._leaderboard_sources(
            ["crane", "slate"], ["crane", "slate", "raise", "howdy"]
        )
        first = collect_report(sources, ReportRequest(report_kind="leaderboard"))
        with patch(
            "report_model._candidate_erd_summary", wraps=_candidate_erd_summary,
        ) as folded:
            second = collect_report(
                sources, ReportRequest(report_kind="leaderboard")
            )
        self.assertEqual(folded.call_count, 4)   # every candidate, every build
        self.assertEqual(second["data"]["rows"], first["data"]["rows"])
        self.assertEqual(second["data"]["counts"], first["data"]["counts"])

    def test_leaderboard_drops_a_candidate_whose_child_result_is_deleted(self):
        # HOWDY shares no letters with either answer, so both collide in one
        # two-answer group -- the only candidate here whose completeness turns
        # on a cached branch result rather than on playing a lone survivor.
        sources = self._leaderboard_sources(
            ["crane", "slate"], ["crane", "slate", "raise", "howdy"]
        )
        answers = ["crane", "slate"]
        collided_key = ScoreCache.encode_subset(answers)
        pending = collect_report(
            sources, ReportRequest(report_kind="leaderboard"))["data"]
        self.assertEqual(pending["counts"]["complete"], 3)
        self.assertNotIn("howdy", {row["word"] for row in pending["rows"]})

        cache = ScoreCache(sources.cache_path, answers,
                           checkpoint_on_close=False)
        cache.write(collided_key, ERD_ALL, "crane", 1.5,
                    max_depth=2, solve_budget=None)
        cache.close()
        complete = collect_report(
            sources, ReportRequest(report_kind="leaderboard"))["data"]
        self.assertEqual(complete["counts"]["complete"], 4)
        self.assertIn("howdy", {row["word"] for row in complete["rows"]})

        # Nothing names HOWDY when this row goes, and nothing needs to: the
        # next build folds HOWDY from the group that no longer resolves.
        cache = ScoreCache(sources.cache_path, answers,
                           checkpoint_on_close=False)
        cache.delete(collided_key, ERD_ALL)
        cache.close()

        second = collect_report(
            sources, ReportRequest(report_kind="leaderboard"))["data"]
        self.assertEqual(second["counts"]["complete"], 3)
        self.assertEqual(second["counts"]["pending"], 1)
        self.assertNotIn("howdy", {row["word"] for row in second["rows"]})
        self.assertEqual(second["rows"], pending["rows"])

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
            for status in ("unqueued", "queued", "evaluating", "finalizing", "done")
        ]
        self.assertEqual(order, sorted(order))

    def test_overview_includes_recent_finalization_after_active_row_is_gone(self):
        branch_key = ScoreCache.encode_subset(ANSWERS)
        now = 1_000
        queue = self._open_queue()
        with patch("erd_queue.time.time", return_value=now - 1):
            queue.add_branch_finalize_log(
                branch_key, "SALET -----", len(ANSWERS), 5,
                now - 20, now - 1, 123, 4, outcome="exact",
                best_guess="crane", best_erd=2.5,
            )
        queue.close()

        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)

        self.assertEqual(report["data"]["branches"], [])
        completed = report["data"]["recently_completed_branches"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["branch_reference"], branch_reference(branch_key))
        self.assertEqual(completed[0]["branch_status"], "done")
        self.assertIsNone(completed[0]["branch_worker_status"])
        self.assertTrue(completed[0]["recently_completed"])
        self.assertEqual(completed[0]["finalized_at"], now - 1)

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
            {"branch_status": "evaluating", "answer_count": 1, "cache_state": "not_applicable"},
        ]
        groups = _grouped_response_groups(rows, "status")
        self.assertEqual(
            [group["label"] for group in groups], ["unqueued", "evaluating", "done"]
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

    def test_word_and_root_progress_reports_need_a_target_ending_in_a_word(self):
        branch = parse_report_branch_target("salet -----")
        with self.assertRaisesRegex(ValueError, "ending in a word"):
            report_model.collect_word_report(
                self.sources, ReportRequest(branch_target=branch)
            )
        with self.assertRaisesRegex(ValueError, "ending in a word"):
            report_model.collect_root_progress_report(
                self.sources,
                ReportRequest(report_kind="root_progress", branch_target=branch),
            )

    def test_root_progress_report_degrades_when_a_store_is_unavailable(self):
        request = ReportRequest(
            report_kind="root_progress",
            branch_target=parse_report_branch_target("salet"),
        )
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            report = collect_report(self.sources, request)
        self.assertFalse(report["sources"]["queue"]["ok"])
        with patch.object(report_model, "ScoreCache",
                          side_effect=sqlite3.OperationalError("cache locked")):
            report = collect_report(self.sources, request)
        self.assertFalse(report["sources"]["cache"]["ok"])

    def test_a_root_progress_group_is_working_only_while_a_worker_holds_it(self):
        row = {"answer_count": 5, "pattern": "-y---", "started": 1}
        self.assertEqual(
            report_model._root_progress_group_state(row, None, 4), "working"
        )
        self.assertEqual(
            report_model._root_progress_group_state({**row, "started": 0},
                                                    None, 4),
            "waiting",
        )

    def test_parsing_helpers_cover_their_degenerate_inputs(self):
        # A trailing descent token carrying no answer count is kept as text.
        self.assertEqual(
            parse_rich_spine("RAISE:-----/4\u2192pending")[-1],
            (None, None, None, "pending"),
        )
        # A row that cannot be subscripted at all yields the default.
        self.assertEqual(
            report_model._row_value([], "missing", "fallback"), "fallback"
        )
        # A worker whose number is not a digit sorts after the numbered ones.
        self.assertEqual(
            report_model._worker_sort_key(
                {"worker_number": "x", "worker_id": "worker-x"}
            ),
            (1, "worker-x"),
        )

    def test_root_progress_is_not_a_tree_and_a_reference_needs_the_queue(self):
        with self.assertRaisesRegex(ValueError, "tree cannot be used"):
            validate_report_request(
                ReportRequest(report_kind="root_progress", tree=True)
            )
        with self.assertRaisesRegex(ValueError, "requires queue resolution"):
            report_model.resolve_branch_target(
                parse_report_branch_target("@abcdef12"), ANSWERS
            )

    def test_opener_sort_orders_complete_pending_infeasible_then_unknown(self):
        def rank(state, erd=None):
            summary = None if state is None else {"state": state, "erd": erd}
            return report_model._source_erd_sort_key(
                {"source_word": "salet", "erd_summary": summary}
            )[0]

        self.assertEqual(
            [rank("complete", 3.5), rank("pending"),
             rank("infeasible"), rank(None)],
            [0, 1, 2, 3],
        )

    def test_a_cache_file_that_is_not_there_has_no_signature(self):
        missing = os.path.join(self.temporary_directory.name, "absent.sqlite3")
        self.assertEqual(
            report_model._score_cache_file_signature(missing), (None, None)
        )

    def test_hotspot_and_accuracy_reports_scope_and_survive_queue_errors(self):
        requests = (
            ReportRequest(report_kind="hotspots", hotspot_field="nodes"),
            ReportRequest(
                report_kind="accuracy",
                branch_target=parse_report_branch_target("salet -----"),
            ),
            ReportRequest(
                report_kind="accuracy",
                branch_target=parse_report_branch_target("salet"),
            ),
        )
        for request in requests:
            with self.subTest(kind=request.report_kind,
                              target=request.branch_target.kind):
                self.assertTrue(
                    collect_report(self.sources, request)["sources"]["queue"]["ok"]
                )
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            report = collect_report(self.sources, requests[0])
        self.assertFalse(report["sources"]["queue"]["ok"])

    def test_tree_report_reads_a_reference_as_a_branch_and_survives_errors(self):
        request = ReportRequest(
            report_kind="auto", tree=True,
            branch_target=parse_report_branch_target("@abcdef12"),
        )
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            report = collect_report(self.sources, request)
        self.assertEqual(report["report_kind"], "branch")
        self.assertFalse(report["sources"]["queue"]["ok"])

    def test_workers_report_scopes_to_a_branch_and_survives_a_queue_error(self):
        request = ReportRequest(
            report_kind="workers",
            branch_target=parse_report_branch_target("salet -----"),
        )
        scoped = collect_report(self.sources, request)
        self.assertTrue(scoped["sources"]["queue"]["ok"])
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            report = collect_report(self.sources, request)
        self.assertFalse(report["sources"]["queue"]["ok"])

    def test_branch_target_scoping_and_row_matching_cover_each_kind(self):
        root = parse_report_branch_target("")
        reference = parse_report_branch_target("@abcdef12")
        word = parse_report_branch_target("salet")
        branch = parse_report_branch_target("salet -----")

        # A reference names a branch only the queue can resolve, so without one
        # it scopes nothing rather than guessing.
        self.assertEqual(
            report_model._branch_target_queue_scope(reference, None), ({}, "")
        )
        # A resolved reference whose row carries no spine scopes by key alone.
        with patch.object(
            report_model, "resolve_branch_reference",
            return_value={"branch_key": b"salet", "spine": None},
        ):
            scope, prefix = report_model._branch_target_queue_scope(
                reference, Mock()
            )
        self.assertEqual(prefix, "")
        self.assertNotIn("spine_prefix", scope)

        self.assertTrue(report_model._row_matches_branch_target({}, root, ""))
        self.assertTrue(report_model._row_matches_branch_target(
            {"spine": None, "source_word": "SALET"}, word, "SALET"
        ))
        self.assertFalse(report_model._row_matches_branch_target(
            {"spine": None, "source_word": "crane"}, word, "SALET"
        ))
        # A spine-scoped target cannot match a row that has no spine.
        self.assertFalse(report_model._row_matches_branch_target(
            {"spine": None}, branch, ""
        ))

    def test_queue_report_scopes_to_a_tree_parent_and_survives_a_queue_error(self):
        tree_request = ReportRequest(
            report_kind="queue", tree=True, tree_parent="SALET -----",
        )
        collect_report(self.sources, tree_request)
        flat_request = ReportRequest(report_kind="queue")
        collect_report(self.sources, flat_request)
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            for request in (tree_request, flat_request):
                with self.subTest(tree=request.tree):
                    report = collect_report(self.sources, request)
                    self.assertFalse(report["sources"]["queue"]["ok"])

    def test_branch_report_survives_a_cache_that_will_not_open(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet -----")
        )
        with patch.object(
            report_model, "ScoreCache",
            side_effect=sqlite3.OperationalError("cache locked"),
        ):
            report = collect_report(self.sources, request)
        self.assertFalse(report["sources"]["cache"]["ok"])

    def test_branch_report_pages_finalizations_in_both_directions(self):
        for direction in ("after", "before"):
            with self.subTest(direction=direction):
                request = ReportRequest(
                    branch_target=parse_report_branch_target("salet -----"),
                    filters=ReportFilters(
                        finalization_cursor_direction=direction,
                        finalization_cursor_recorded_at=1_700_000_000,
                        finalization_cursor_id=42,
                    ),
                )
                collect_report(self.sources, request)

    def test_a_branch_reference_needs_a_queue_or_a_cache_to_resolve(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("@abcdef12")
        )
        with (
            patch.object(report_model, "ScoreCache",
                         side_effect=sqlite3.OperationalError("cache locked")),
            patch.object(report_model, "_open_report_queue",
                         side_effect=sqlite3.OperationalError("queue locked")),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                collect_report(self.sources, request)

    def test_a_branch_report_resolves_from_the_answer_list_without_a_queue(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet -----")
        )
        with patch.object(report_model, "_open_report_queue",
                          side_effect=sqlite3.OperationalError("queue locked")):
            report = collect_report(self.sources, request)
        self.assertFalse(report["sources"]["queue"]["ok"])

    def test_word_report_filters_and_sorts_its_response_groups(self):
        target = parse_report_branch_target("salet")

        def groups(**filter_values):
            request = ReportRequest(
                branch_target=target, filters=ReportFilters(**filter_values)
            )
            return collect_report(
                self.sources, request
            )["data"]["response_groups"]

        everything = groups()
        self.assertTrue(everything)
        self.assertLessEqual(
            len(groups(branch_statuses=("evaluating",))), len(everything)
        )
        self.assertLessEqual(
            len(groups(branch_worker_statuses=("active",))), len(everything)
        )
        self.assertTrue(all(
            row["answer_count"] <= 1 for row in groups(maximum_answer_count=1)
        ))
        self.assertTrue(all(
            row["priority"] == 4 for row in groups(priority=4)
        ))
        # A budget other than the one these groups are solved at matches none.
        self.assertEqual(groups(budget=2), [])
        for sort, rank in (
            ("size", lambda row: -row["answer_count"]),
            ("workers", lambda row: -row["worker_count"]),
            ("priority", lambda row: -(row["priority"] or 0)),
        ):
            with self.subTest(sort=sort):
                ranks = [rank(row) for row in groups(sort=sort)]
                self.assertEqual(ranks, sorted(ranks))

    def test_word_report_includes_answer_words_on_request(self):
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"),
            include_answers=True,
        )
        groups = collect_report(self.sources, request)["data"]["response_groups"]
        self.assertTrue(any("answer_words" in row for row in groups))

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

    def _build_pattern_matrix(self):
        """Put this vocabulary's matrix on disk, as a swarm run would leave it."""
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        PatternMatrix.load_or_build(
            self.cache_path, ANSWERS + ["raise"], ANSWERS, cache)
        cache.close()

    def test_word_report_scale_is_the_best_split_available_on_its_branch(self):
        # Not one constant for the whole vocabulary: a branch of two answers
        # cannot be split more than two ways, and scaling it against the root's
        # best would draw every deep branch as a stub.
        self._build_pattern_matrix()
        response_cache = ResponseCache(ANSWERS, score_cache=None)
        best_at_root = max(
            len([words for words in response_cache.group_words(
                candidate, list(ANSWERS)).values() if words])
            for candidate in ANSWERS + ["raise"]
        )
        root = collect_report(self.sources, ReportRequest(
            branch_target=parse_report_branch_target("salet"),
        ))["data"]
        self.assertEqual(root["maximum_response_group_count"], best_at_root)

        # BEBOP splits these answers two ways, so its first branch holds two.
        deeper = collect_report(self.sources, ReportRequest(
            branch_target=parse_report_branch_target("bebop ----- salet"),
        ))["data"]
        self.assertEqual(deeper["context"]["answer_count"], 2)
        self.assertEqual(deeper["maximum_response_group_count"], 2)
        self.assertLess(
            deeper["maximum_response_group_count"],
            root["maximum_response_group_count"],
            "the scale must follow the branch, not the vocabulary",
        )

    def test_word_report_never_builds_a_pattern_matrix(self):
        # A cold build walks the whole vocabulary and takes minutes.  A report
        # answers without the scale instead of blocking on one.
        with patch.object(PatternMatrix, "build") as build:
            data = collect_report(self.sources, ReportRequest(
                branch_target=parse_report_branch_target("salet"),
            ))["data"]
        build.assert_not_called()
        self.assertNotIn("maximum_response_group_count", data)
        self.assertTrue(data["response_group_breakdown"],
                        "the graph is still drawn, just unscaled")

    def test_word_report_measures_each_branch_scale_once(self):
        # The root branch costs ~1.5s of NumPy to measure and the report is
        # polled every couple of seconds.
        self._build_pattern_matrix()
        request = ReportRequest(
            branch_target=parse_report_branch_target("salet"))
        first = collect_report(self.sources, request)["data"]
        with patch.object(
            PatternMatrix, "counts_for_all_candidates"
        ) as measured:
            second = collect_report(self.sources, request)["data"]
        measured.assert_not_called()
        self.assertEqual(second["maximum_response_group_count"],
                         first["maximum_response_group_count"])

    def test_response_group_breakdown_covers_every_group_largest_first(self):
        # AUDIO's groups arrive in pattern order as 1, 2, 1, so a breakdown
        # that merely echoed the row order would not lead with the largest.
        data = collect_report(self.sources, ReportRequest(
            branch_target=parse_report_branch_target("audio"),
        ))["data"]
        breakdown = data["response_group_breakdown"]
        self.assertEqual(len(breakdown), data["total_rows"])
        self.assertEqual(
            [set(entry) for entry in breakdown],
            [{"pattern", "answer_count", "solved"}] * len(breakdown),
        )
        self.assertEqual(
            [entry["answer_count"] for entry in breakdown], [2, 1, 1]
        )
        self.assertEqual(
            [entry["pattern"] for entry in breakdown],
            ["y----", "-gy--", "y--y-"],
        )
        self.assertEqual(
            {(entry["pattern"], entry["answer_count"]) for entry in breakdown},
            {
                (row["pattern"], row["answer_count"])
                for row in data["response_groups"]
            },
        )

    def test_response_group_breakdown_marks_solved_groups(self):
        # BEBOP splits these answers into two groups of two, so caching one
        # branch leaves a matched pair: same size, one solved and one not.
        target = parse_report_branch_target("bebop")
        groups = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]["response_groups"]
        cached, uncached = groups[0], groups[1]
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(
            bytes.fromhex(cached["branch_key_hex"]), ERD_ALL, "salet", 1.5,
            max_depth=2, solve_budget=None,
        )
        cache.close()
        data = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]
        solved = {
            entry["pattern"]: entry["solved"]
            for entry in data["response_group_breakdown"]
        }
        self.assertTrue(solved[cached["pattern"]])
        self.assertFalse(solved[uncached["pattern"]])
        self.assertEqual(
            sum(solved.values()), data["erd_summary"]["resolved_group_count"],
            "the outlined groups must be the groups the ERD line counts",
        )

    def test_response_group_breakdown_solves_a_lone_survivor_only_with_budget(self):
        # A one-answer group is solved by playing the survivor, which costs a
        # guess -- so at the last guess it is a proven loss, not a solved group.
        data = collect_report(self.sources, ReportRequest(
            branch_target=parse_report_branch_target("audio"),
        ))["data"]
        solved = {
            entry["pattern"]: entry["solved"]
            for entry in data["response_group_breakdown"]
        }
        lone = [
            row for row in data["response_groups"] if row["answer_count"] == 1
        ]
        self.assertTrue(lone)
        self.assertTrue(all(solved[row["pattern"]] for row in lone))
        self.assertEqual(
            sum(solved.values()), data["erd_summary"]["resolved_group_count"]
        )
        self.assertFalse(_response_group_is_solved(
            {"best_erd": None, "max_remaining_depth": None,
             "answer_count": 1, "pattern": "-----"},
            0,
        ))
        self.assertTrue(_response_group_is_solved(
            {"best_erd": None, "max_remaining_depth": None,
             "answer_count": 1, "pattern": "ggggg"},
            0,
        ))

    def test_response_group_is_solved_requires_a_worst_case_line(self):
        # An ERD with no proven worst-case line cannot complete the fold, so it
        # is not a solved group either.
        self.assertFalse(_response_group_is_solved(
            {"best_erd": 2.0, "max_remaining_depth": None,
             "answer_count": 9, "pattern": "-----"},
            5,
        ))
        self.assertTrue(_response_group_is_solved(
            {"best_erd": 2.0, "max_remaining_depth": 3,
             "answer_count": 9, "pattern": "-----"},
            5,
        ))

    def _solve_bebop_groups(self):
        """Give BEBOP's every response group an exact result, and report it.

        Returns (branch_keys, report data) with the word reading `complete` —
        the state every deletion regression below starts from.
        """
        target = parse_report_branch_target("bebop")
        groups = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]["response_groups"]
        branch_keys = [
            bytes.fromhex(row["branch_key_hex"]) for row in groups
        ]
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        for branch_key in branch_keys:
            cache.write(branch_key, ERD_ALL, "salet", 1.5,
                        max_depth=2, solve_budget=None)
        cache.close()
        complete = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]
        self.assertEqual(complete["erd_summary"]["state"], "complete")
        self.assertTrue(all(entry["solved"]
                            for entry in complete["response_group_breakdown"]))
        self.assertEqual(complete["erd_summary"]["resolved_group_count"], 2)
        return branch_keys, complete

    def _bebop_report(self):
        return collect_report(
            self.sources,
            ReportRequest(branch_target=parse_report_branch_target("bebop")),
        )["data"]

    def _assert_bebop_reads_pending(self, resolved_group_count):
        data = self._bebop_report()
        solved = [entry["solved"]
                  for entry in data["response_group_breakdown"]]
        self.assertEqual(data["erd_summary"]["state"], "pending")
        self.assertIsNone(data["erd_summary"]["erd"])
        self.assertIsNone(data["erd_summary"]["max_remaining_depth"])
        self.assertEqual(sum(solved), resolved_group_count)
        self.assertEqual(
            data["erd_summary"]["resolved_group_count"], resolved_group_count,
            "the ERD line and the outlined groups must count the same groups",
        )
        self.assertEqual(data["erd_summary"]["response_group_count"], 2,
                         "every group is still counted, resolved or not")
        return data

    def test_a_complete_candidate_goes_pending_when_one_child_is_deleted(self):
        """One deleted response group is enough to un-complete the word.

        This is the failure #288 describes, reduced to its smallest form: no
        caller announces the deletion, and nothing above the branch is
        invalidated.  A word reads `complete` only because its groups are read
        and folded afresh, so deleting one of them shows up on the next report
        with no invalidation step in between to get right.
        """
        branch_keys, _ = self._solve_bebop_groups()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.delete(branch_keys[0], ERD_ALL)
        cache.close()
        self._assert_bebop_reads_pending(resolved_group_count=1)

    def test_a_word_goes_pending_after_a_spot_check_deletes_a_branch(self):
        # verify_erd_cache's reverification and wordle.py's "this root
        # contradicts its own cached subtree" both reach the report through a
        # bare ScoreCache.delete of a branch whose parents they cannot name.
        branch_keys, _ = self._solve_bebop_groups()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        for branch_key in branch_keys:
            cache.delete(branch_key, ERD_ALL)
        cache.close()
        self._assert_bebop_reads_pending(resolved_group_count=0)

    def test_a_word_goes_pending_after_a_repair_withdraws_a_child(self):
        # verify_branch_depths repairs max_depth in place.  Raising a child's
        # worst case past the budget it is read at withdraws its reuse, so the
        # parent's fold must lose that group -- the case a memo keyed on an
        # unchanged response-group count could not see at all.
        branch_keys, _ = self._solve_bebop_groups()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        group_budget = GAME_GUESSES - 1
        self.assertTrue(cache.repair_max_depth(
            branch_keys[0], ERD_ALL, group_budget + 1, None))
        cache.close()
        self._assert_bebop_reads_pending(resolved_group_count=1)

    def test_a_word_goes_pending_after_the_recompute_path_deletes_its_groups(self):
        # The real recompute sequence, not a hand-made approximation of it.
        branch_keys, _ = self._solve_bebop_groups()
        queue = self._open_queue()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        erd_search.invalidate_branches_for_recompute(queue, cache, branch_keys)
        cache.close()
        queue.close()
        self._assert_bebop_reads_pending(resolved_group_count=0)

    def test_a_word_folds_a_childs_budget_specific_result_at_its_own_budget(self):
        """A child solved at exactly this budget completes the fold; one solved
        at another budget is not available to it.

        Both facts can be stored for one branch, and only the scope the parent
        would actually reuse may be folded -- `report_branch_states` picks it
        with the same rule `read_for_budget` applies.
        """
        target = parse_report_branch_target("bebop")
        groups = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]["response_groups"]
        branch_keys = [
            bytes.fromhex(row["branch_key_hex"]) for row in groups
        ]
        group_budget = GAME_GUESSES - 1
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        # One child has only a result solved at the wrong budget ...
        cache.write(branch_keys[0], ERD_ALL, "salet", 1.5,
                    max_depth=2, solve_budget=group_budget - 1)
        cache.write(branch_keys[1], ERD_ALL, "salet", 1.5,
                    max_depth=2, solve_budget=None)
        cache.close()
        self._assert_bebop_reads_pending(resolved_group_count=1)

        # ... and completes the fold once it holds one at this budget.
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(branch_keys[0], ERD_ALL, "salet", 1.5,
                    max_depth=2, solve_budget=group_budget)
        cache.close()
        data = self._bebop_report()
        self.assertEqual(data["erd_summary"]["state"], "complete")
        self.assertEqual(data["erd_summary"]["resolved_group_count"], 2)

    def test_response_group_breakdown_ignores_filters_and_the_display_limit(self):
        # The graph draws the whole decomposition; narrowing the rows listed
        # beneath it must not redraw the picture of the word.
        target = parse_report_branch_target("audio")
        unfiltered = collect_report(
            self.sources, ReportRequest(branch_target=target)
        )["data"]["response_group_breakdown"]
        limited = collect_report(self.sources, ReportRequest(
            branch_target=target, filters=ReportFilters(limit=1),
        ))["data"]
        self.assertEqual(len(limited["response_groups"]), 1)
        self.assertEqual(limited["response_group_breakdown"], unfiltered)
        excluded = collect_report(self.sources, ReportRequest(
            branch_target=target, filters=ReportFilters(minimum_answer_count=3),
        ))["data"]
        self.assertEqual(excluded["response_groups"], [])
        self.assertEqual(excluded["response_group_breakdown"], unfiltered)

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
        self.assertEqual(branch["branch_status"], "evaluating")
        self.assertEqual(branch["branch_worker_status"], "active")
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
        result = queue.report_queue_rows({"branch_worker_statuses": ("active",)})
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
        self.assertEqual(branches[live_key.hex()]["branch_status"], "finalizing")
        self.assertEqual(branches[live_key.hex()]["branch_worker_status"], "active")
        self.assertEqual(branches[dead_key.hex()]["branch_status"], "finalizing")
        self.assertEqual(branches[dead_key.hex()]["branch_worker_status"], "waiting")
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
            branch_worker_statuses=("active",)
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
            waiting_report = collect_report(self.sources, ReportRequest(
                filters=ReportFilters(branch_worker_statuses=("waiting",), limit=1)
            ))
        self.assertEqual(active_report["data"]["branches"], [])
        self.assertEqual(len(waiting_report["data"]["branches"]), 1)
        self.assertEqual(
            waiting_report["data"]["branches"][0]["branch_worker_status"], "waiting"
        )

    def test_pending_overview_includes_scheduled_branch_before_evaluation(self):
        branch_key = b"queued-branch"
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 5, "raise", 0)])
        queue.close()

        report = collect_report(self.sources, ReportRequest(
            filters=ReportFilters(branch_statuses=("queued",))
        ))
        self.assertEqual(len(report["data"]["branches"]), 1)
        branch = report["data"]["branches"][0]
        self.assertEqual(branch["branch_key_hex"], branch_key.hex())
        self.assertEqual(branch["branch_status"], "queued")
        self.assertIsNone(branch["branch_worker_status"])
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
            "budgeted_result_count": 0,
            "budgeted_branch_count": 0,
            "recent_budgeted_result_count": 0,
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
            report_kind="openers",
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
            self.sources, ReportRequest(report_kind="openers"))

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
            report_kind="openers", filters=ReportFilters(**filters)))
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

        # The default shows the newest completed work first.  With no
        # completed sources, its word tie-breaker gives a stable order.
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
        self.assertEqual(
            [row["source_word"] for row in self._source_words()["summary"]],
            ["salet", "crane", "nurdy"],
        )

    def test_active_source_elapsed_time_starts_when_work_is_claimed(self):
        self._queue_words(("salet", 5, 1))
        queue = self._open_queue()
        queue._conn.execute(
            "UPDATE source_work SET state = 'active', started_at = 100 "
            "WHERE source_word = 'salet'"
        )
        queue._conn.commit()
        queue.close()

        with patch("report_model.time.time", return_value=130):
            row = self._source_words()["summary"][0]

        self.assertEqual(row["started_at"], 100)
        self.assertEqual(row["elapsed_millis"], 30_000)
        self.assertIsNone(row["worker_millis"])

    def test_source_report_keeps_erd_summary_shape_when_cache_unavailable(self):
        self._queue_words(("salet", 5, 1))
        unavailable_sources = replace(
            self.sources, answer_list_path="unused-answers"
        )

        report = collect_source_report(
            unavailable_sources, ReportRequest(report_kind="openers")
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
        # cached response groups on every read.  NURDY is the one of these that
        # leaves a two-answer group, so it is the one whose ERD needs the
        # cache; the others partition this answer list into singletons, which
        # are solved by playing them.
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

        # Neither word's fold is persisted anywhere: the report's numbers came
        # from the branch tables and nothing else was written.
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        self.assertEqual(
            [row["name"] for row in cache._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'candidate_erd_by_policy'")],
            [],
        )
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
                report_kind="openers", branch_target=salet,
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
            ({"source_states": ("queued",)}, "opener_state requires"),
            ({"source_offset": 2}, "source_offset requires"),
            ({"sort": "branches"}, "requires an opener report"),
        ):
            with self.subTest(filters=filters):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report_request(ReportRequest(
                        report_kind="queue", filters=ReportFilters(**filters)))
        with self.assertRaisesRegex(ValueError, "opener reports must be"):
            validate_report_request(ReportRequest(
                report_kind="openers", filters=ReportFilters(sort="nodes")))
        validate_report_request(ReportRequest(
            report_kind="openers", filters=ReportFilters(sort="age")))
        for group_by in ("completed", "elapsed", "worker_time", "requested"):
            with self.subTest(group_by=group_by):
                validate_report_request(ReportRequest(
                    report_kind="openers", filters=ReportFilters(group_by=group_by)))
        with self.assertRaisesRegex(ValueError, "opener reports must be"):
            validate_report_request(ReportRequest(
                report_kind="openers",
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
            self.sources, ReportRequest(report_kind="openers"))

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
            self.sources, ReportRequest(report_kind="openers"))

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
        self.assertEqual(rows["salet"]["branch_status"], "queued")
        self.assertIsNone(rows["salet"]["branch_worker_status"])

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
                report_kind="openers",
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
            report_kind="openers",
            branch_target=parse_report_branch_target(["salet"]),
        )
        full = collect_report(self.sources, salet)
        limited = collect_report(self.sources, ReportRequest(
            report_kind="openers", branch_target=salet.branch_target,
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

    def test_completed_root_reports_its_finalization_time(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 5, 1_000, 100, 200)
        queue.close()

        with patch("report_model._root_progress_group_state",
                   return_value="solved"):
            data = collect_report(self.sources, self._request())["data"]

        self.assertEqual(data["completed_at"], 200)
        self.assertIsNone(data["estimate"])

    def test_completed_root_keeps_its_completion_time_across_epoch_rollover(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:2])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 2, 1, "salet", 0)])
        with patch("erd_queue.time.time", return_value=300):
            queue.mark_done(branch_key)
        queue.set_epoch(1)
        queue.close()

        with patch("report_model._root_progress_group_state",
                   return_value="solved"):
            data = collect_report(self.sources, self._request(epoch=1))["data"]

        self.assertIsNone(data["work_latest_at"])
        self.assertEqual(data["completed_at"], 300)

    def test_rollup_can_be_fenced_to_an_explicit_epoch(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 100, 10, 10, 20, epoch=0)
        self._finalize(queue, "SALET -----", 1, 500, 10, 10, 20, epoch=1)
        queue.close()

        rows = {row["pattern"]: row for row
                in collect_report(self.sources,
                                  self._request(epoch=1))["data"]["response_groups"]}
        self.assertTrue(rows["-----"]["started"])
        self.assertFalse(rows["-y---"]["started"])

    def test_default_rollup_includes_prior_epochs_and_records_them(self):
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 100, 10, 10, 20, epoch=0)
        self._finalize(queue, "SALET -----", 1, 500, 10, 10, 20, epoch=1)
        queue.set_epoch(2)
        queue.close()

        data = collect_report(self.sources, self._request())["data"]
        rows = {row["pattern"]: row for row in data["response_groups"]}

        self.assertEqual(data["epoch"], 2)
        self.assertIsNone(data["selected_telemetry_epoch"])
        self.assertEqual(data["telemetry_epochs"], [0, 1])
        self.assertTrue(rows["-y---"]["started"])
        self.assertEqual(rows["-y---"]["search_node_count"], 100)
        self.assertEqual(rows["-y---"]["telemetry_epochs"], [0])
        self.assertEqual(rows["-----"]["search_node_count"], 500)
        self.assertEqual(rows["-----"]["telemetry_epochs"], [1])

    def test_root_progress_prefers_queue_telemetry_to_cached_summary(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.add_root_response_group_summary(
            "salet", "-y---", ERD_ALL, 900, 9_000, 1, 2, 0)
        cache.close()
        queue = self._open_queue()
        self._finalize(queue, "SALET -y---", 1, 100, 1_000, 10, 20)
        queue.close()

        rows = {row["pattern"]: row for row in
                collect_report(self.sources, self._request())["data"]["response_groups"]}
        self.assertEqual(rows["-y---"]["branch_count"], 1)
        self.assertEqual(rows["-y---"]["search_node_count"], 100)
        self.assertEqual(rows["-y---"]["open_branch_count"], 0)

    def test_root_progress_uses_cached_summary_without_queue_telemetry(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.add_root_response_group_summary(
            "salet", "-y---", ERD_ALL, 100, 1_000, 10, 20, 0)
        cache.add_root_response_group_summary(
            "salet", "-y---", ERD_ALL, 900, 9_000, 12, 30, 1)
        cache.close()

        rows = {row["pattern"]: row for row in
                collect_report(self.sources, self._request())["data"]["response_groups"]}
        self.assertEqual(rows["-y---"]["branch_count"], 2)
        self.assertEqual(rows["-y---"]["search_node_count"], 1_000)
        self.assertEqual(rows["-y---"]["telemetry_epochs"], [0, 1])

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


class TreePageCursorTest(unittest.TestCase):
    """Paging a tree page whose groups are not all named by a guess word.

    A node with no spine has no word to group by, so its group is keyed by a
    synthetic node id that sorts ahead of every worded group.  A cursor names
    one or the other, and the two do not order against each other as plain
    strings.
    """

    def _row(self, word, spine=None):
        branch_key = ScoreCache.encode_subset([word])
        row = {
            "branch_key": branch_key,
            "branch_key_hex": branch_key.hex(),
            "branch_status": "queued",
            "branch_worker_status": None,
            "answer_count": 1,
            "worker_count": 0,
            "priority": 1,
            "completed_candidate_count": 0,
            "candidate_count": 3,
        }
        if spine is not None:
            row["spine"] = spine
        return row

    def _worded_rows(self):
        return [
            self._row("crane", "CRANE y----"),
            self._row("nurdy", "NURDY -y---"),
            self._row("salet", "SALET --g--"),
        ]

    def _page(self, rows, cursor=None, limit=1):
        request = ReportRequest(
            report_kind="queue",
            tree=True,
            filters=ReportFilters(limit=limit),
            tree_cursor=cursor,
        )
        return report_model._tree_layout(rows, request, "", rows, set(ANSWERS))

    def _words(self, page):
        return [(node["step"] or {}).get("word") for node in page["nodes"]]

    def test_paging_past_a_spineless_group_keeps_every_worded_group(self):
        spineless = self._row("khaki")
        rows = [spineless] + self._worded_rows()
        first = self._page(rows)
        cursor = first["paging"]["next_cursor"]
        self.assertEqual(cursor, "unknown:1:" + spineless["branch_key_hex"])
        # Each worded group sorts after the spineless one in the page order and
        # below it as a plain string, which is what a string cursor gets wrong.
        self.assertTrue(all(word < cursor for word in ("crane", "nurdy", "salet")))
        second = self._page(rows, cursor=cursor)
        self.assertEqual(second["paging"]["total_group_count"], 4)
        self.assertEqual(second["paging"]["returned_group_count"], 1)
        self.assertEqual(self._words(second), ["crane"])

    def test_walking_every_page_visits_each_group_exactly_once(self):
        rows = [self._row("khaki")] + self._worded_rows()
        visited, cursor = [], None
        for _ in range(len(rows) + 1):
            page = self._page(rows, cursor=cursor)
            visited.extend(
                (node["step"] or {}).get("word") or node["node_id"]
                for node in page["nodes"]
            )
            cursor = page["paging"]["next_cursor"]
            if cursor is None:
                break
        self.assertIsNone(cursor)
        self.assertEqual(len(visited), len(set(visited)))
        self.assertEqual(visited[1:], ["crane", "nurdy", "salet"])

    def test_a_cursor_whose_group_has_gone_resumes_where_it_stood(self):
        rows = self._worded_rows()
        after_word = self._page(rows, cursor="khaki", limit=10)
        self.assertEqual(self._words(after_word), ["nurdy", "salet"])
        after_spineless = self._page(rows, cursor="unknown:1:" + "ab" * 5, limit=10)
        self.assertEqual(
            self._words(after_spineless), ["crane", "nurdy", "salet"]
        )
