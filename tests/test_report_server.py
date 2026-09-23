"""Tests for the read-only HTTP report adapter."""

from contextlib import contextmanager
import errno
from http.server import ThreadingHTTPServer
from threading import Event, Lock, Thread
import io
import copy
import json
import threading
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from erd_queue import ERDQueue, encode_subset
import report_server
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportOpeners,
    _tree_layout,
    collect_report,
)
from report_server import (
    FIXTURE_FILENAMES,
    InvalidRequest,
    ServerConfiguration,
    build_configuration,
    load_fixtures,
    main,
    make_handler,
    parse_report_request,
)


ROOT = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIRECTORY = os.path.join(ROOT, "tests", "fixtures", "reports")
CLIENT_PATH = os.path.join(ROOT, "report_client.html")


@contextmanager
def running_server(configuration):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(configuration)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(base_url, path, method="GET", headers=None):
    try:
        with urlopen(Request(base_url + path, method=method,
                             headers=headers or {}), timeout=3) as response:
            body = response.read()
            return response.status, response.headers, body
    except HTTPError as error:
        return error.code, error.headers, error.read()


def fixture_configuration():
    sources = ReportOpeners("unused-queue", "unused-cache", "unused-answers", "unused-guesses")
    return ServerConfiguration(
        sources, CLIENT_PATH, FIXTURE_DIRECTORY, load_fixtures(FIXTURE_DIRECTORY)
    )


class ReportServerTest(unittest.TestCase):
    def test_request_parser_rejects_invalid_boolean_and_tree_parent(self):
        with self.assertRaisesRegex(InvalidRequest, "must be 1, 0, true, or false"):
            parse_report_request("/api/view", "tree=maybe")
        with self.assertRaisesRegex(InvalidRequest, "complete spine"):
            parse_report_request("/api/view", "tree=1&tree_parent=RAISE")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        directory = self.temporary_directory.name
        self.answer_list_path = os.path.join(directory, "answers.txt")
        self.guess_list_path = os.path.join(directory, "guesses.txt")
        with open(self.answer_list_path, "w") as answer_file:
            answer_file.write("cigar\nrebut\nsissy\nhumph\nawake\n")
        with open(self.guess_list_path, "w") as guess_file:
            guess_file.write("cigar\nrebut\nsissy\nhumph\nawake\nraise\n")
        self.sources = ReportOpeners(
            os.path.join(directory, "queue.sqlite3"),
            os.path.join(directory, "cache.sqlite3"),
            self.answer_list_path,
            self.guess_list_path,
            os.path.join(directory, "telemetry.sqlite3"),
        )
        queue = ERDQueue(
            self.sources.queue_path, telemetry_path=self.sources.telemetry_path
        )
        queue.close()
        self.live_configuration = ServerConfiguration(self.sources, CLIENT_PATH)

    def test_everyday_reports_return_within_one_second(self):
        configuration = ServerConfiguration(
            self.sources, CLIENT_PATH, None, None
        )
        with running_server(configuration) as base_url:
            for path in ("/api/view", "/api/view/queue", "/api/view/workers"):
                started_at = time.monotonic()
                status, _, _ = request(base_url, path)
                self.assertEqual(status, 200)
                self.assertLess(time.monotonic() - started_at, 1.0, path)

    def test_live_root_contract_matches_direct_collection_shape(self):
        disk = {
            "total_bytes": 1_000,
            "used_bytes": 250,
            "avail_bytes": 750,
            "used_fraction": 0.25,
        }
        with (
            patch("report_model.disk_stats", return_value=disk),
            patch("report_model.ERDQueue.wal_size_bytes", return_value=128),
            running_server(self.live_configuration) as base_url,
        ):
            status, _headers, body = request(base_url, "/api/view")
            direct = collect_report(
                self.sources, parse_report_request("/api/view", "")
            )
        self.assertEqual(status, 200)
        served = json.loads(body)
        served.pop("generated_at")
        direct.pop("generated_at")
        self.assertEqual(served, direct)

    def test_branch_target_inference_uses_one_endpoint(self):
        with running_server(fixture_configuration()) as base_url:
            word = json.loads(request(base_url, "/api/view?branch_target=CACHE")[2])
            branch = json.loads(request(
                base_url, "/api/view?branch_target=RAISE%20....."
            )[2])
        self.assertEqual(word["report_kind"], "word")
        self.assertEqual(branch["report_kind"], "branch")

    def test_tree_and_branch_filters_reach_normalized_request(self):
        report_request = parse_report_request(
            "/api/view/queue",
            "tree=true&branch_status=evaluating,finalizing&"
            "branch_worker_status=active&limit=4",
        )
        self.assertTrue(report_request.tree)
        self.assertEqual(
            report_request.filters.branch_statuses, ("evaluating", "finalizing")
        )
        self.assertEqual(
            report_request.filters.branch_worker_statuses, ("active",)
        )
        self.assertEqual(report_request.filters.limit, 4)

    def test_finalization_cursor_is_parsed_into_filters(self):
        default_request = parse_report_request("/api/view", "")
        after_request = parse_report_request(
            "/api/view",
            "branch_target=RAISE%20.....&finalization_cursor=after:1700000000:42",
        )
        before_request = parse_report_request(
            "/api/view",
            "branch_target=RAISE%20.....&finalization_cursor=before:1700000000:42",
        )
        self.assertIsNone(default_request.filters.finalization_cursor_direction)
        self.assertEqual(after_request.filters.finalization_cursor_direction, "after")
        self.assertEqual(
            after_request.filters.finalization_cursor_recorded_at, 1700000000
        )
        self.assertEqual(after_request.filters.finalization_cursor_id, 42)
        self.assertEqual(before_request.filters.finalization_cursor_direction, "before")

    def test_tree_page_parameters_require_tree_and_complete_parent_spine(self):
        request = parse_report_request(
            "/api/view", "tree=1&tree_parent=RAISE%20.....&tree_cursor=raise",
        )
        self.assertEqual(request.tree_parent, "RAISE -----")
        self.assertEqual(request.tree_cursor, "raise")
        for query, message in (
            ("tree_cursor=raise", "require tree"),
            ("tree=1&tree_parent=RAISE", "complete spine"),
            ("tree=1&tree_cursor=not-a-word", "tree page group"),
            ("tree=1&tree_cursor=unknown:1:zzzz", "tree page group"),
        ):
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, message):
                    parse_report_request("/api/view", query)
        spineless = parse_report_request(
            "/api/view", "tree=1&tree_cursor=unknown:1:abcdef",
        )
        self.assertEqual(spineless.tree_cursor, "unknown:1:abcdef")

    def test_every_tree_page_cursor_the_model_emits_parses_as_a_request(self):
        # The client hands paging.next_cursor straight back as tree_cursor, so
        # a cursor the model emits and the parser refuses breaks paging.
        spineless_key = encode_subset(["khaki"])
        worded_key = encode_subset(["crane"])
        rows = [
            {
                "branch_key": spineless_key,
                "branch_key_hex": spineless_key.hex(),
                "branch_status": "queued",
                "branch_worker_status": None,
                "answer_count": 1,
                "worker_count": 0,
                "priority": 1,
                "completed_candidate_count": 0,
                "candidate_count": 3,
            },
            {
                "branch_key": worded_key,
                "branch_key_hex": worded_key.hex(),
                "spine": "CRANE y----",
                "branch_status": "queued",
                "branch_worker_status": None,
                "answer_count": 1,
                "worker_count": 0,
                "priority": 1,
                "completed_candidate_count": 0,
                "candidate_count": 3,
            },
        ]
        request = ReportRequest(
            report_kind="queue", tree=True, filters=ReportFilters(limit=1),
        )
        layout = _tree_layout(rows, request, "", rows, {"khaki", "crane"})
        cursor = layout["paging"]["next_cursor"]
        self.assertEqual(cursor, "unknown:1:" + spineless_key.hex())
        parsed = parse_report_request(
            "/api/view", f"tree=1&tree_cursor={cursor}",
        )
        self.assertEqual(parsed.tree_cursor, cursor)

    def test_a_boolean_parameter_accepts_both_spellings_of_false(self):
        for query in ("tree=0", "tree=false", "tree=FALSE"):
            with self.subTest(query=query):
                self.assertFalse(parse_report_request("/api/view", query).tree)

    def test_a_tree_parent_that_is_not_a_branch_target_is_an_invalid_request(self):
        # parse_report_branch_target raises ValueError on a malformed token,
        # which reaches the reader as a bad request rather than a crash.
        with self.assertRaisesRegex(InvalidRequest, "hexadecimal"):
            parse_report_request("/api/view", "tree=1&tree_parent=@zz")

    def test_root_overview_defaults_to_worked_branches_and_all_disables_filter(self):
        default_request = parse_report_request("/api/view", "")
        all_request = parse_report_request("/api/view", "branch_worker_status=all")
        all_statuses = parse_report_request("/api/view", "branch_status=all")
        word_request = parse_report_request("/api/view", "branch_target=RAISE")
        self.assertEqual(
            default_request.filters.branch_statuses, ("evaluating", "finalizing")
        )
        self.assertEqual(default_request.filters.branch_worker_statuses, ("active",))
        self.assertEqual(all_request.filters.branch_worker_statuses, ())
        self.assertEqual(all_statuses.filters.branch_statuses, ())
        # Only the overview opens on a filter; every other report starts unfiltered.
        self.assertEqual(word_request.filters.branch_statuses, ())
        self.assertEqual(word_request.filters.branch_worker_statuses, ())
        queue_request = parse_report_request("/api/view/queue", "")
        self.assertEqual(queue_request.filters.branch_statuses, ())
        self.assertEqual(queue_request.filters.branch_worker_statuses, ())

    def test_http_and_terminal_compatibility_validation_is_shared(self):
        invalid_requests = (
            ("/api/view", "branch_target=RAISE%20.....&tree=1&claims=1"),
            ("/api/view", "branch_target=RAISE&sort=nodes"),
        )
        for path, query in invalid_requests:
            with self.subTest(query=query), self.assertRaises(InvalidRequest):
                parse_report_request(path, query)

    def test_unqueued_branch_status_selects_only_on_a_word_report(self):
        word_request = parse_report_request(
            "/api/view", "branch_target=RAISE&branch_status=unqueued,done"
        )
        self.assertEqual(
            word_request.filters.branch_statuses, ("unqueued", "done")
        )
        refused = (
            ("/api/view", "branch_status=unqueued"),
            ("/api/view/queue", "branch_status=unqueued"),
            ("/api/view/workers", "branch_status=unqueued"),
            ("/api/view/hotspots", "branch_status=unqueued"),
            ("/api/view", "branch_target=RAISE&tree=1&branch_status=unqueued"),
            ("/api/view", "branch_target=RAISE%20.....&branch_status=unqueued"),
            ("/api/view", "branch_status=queued,unqueued"),
        )
        for path, query in refused:
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, "unqueued"):
                    parse_report_request(path, query)

    def test_worker_status_filter_is_dropped_when_no_status_carries_one(self):
        # A queued/done/unqueued branch has no worker status at all, so a
        # worker filter applied alongside them would match nothing.
        for query in (
            "branch_status=queued&branch_worker_status=active",
            "branch_status=done&branch_worker_status=waiting",
            "branch_status=queued,done&branch_worker_status=active,waiting",
        ):
            with self.subTest(query=query):
                request = parse_report_request("/api/view/queue", query)
                self.assertEqual(request.filters.branch_worker_statuses, ())
        kept = (
            ("branch_status=evaluating&branch_worker_status=active", ("active",)),
            ("branch_status=finalizing&branch_worker_status=waiting", ("waiting",)),
            ("branch_status=queued,evaluating&branch_worker_status=active",
             ("active",)),
            ("branch_worker_status=waiting", ("waiting",)),
        )
        for query, expected in kept:
            with self.subTest(query=query):
                request = parse_report_request("/api/view/queue", query)
                self.assertEqual(request.filters.branch_worker_statuses, expected)

    def test_every_explicit_endpoint_returns_its_kind(self):
        with running_server(fixture_configuration()) as base_url:
            for kind in ("queue", "workers", "cache", "hotspots", "openers"):
                with self.subTest(kind=kind):
                    status, _headers, body = request(base_url, f"/api/view/{kind}")
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["report_kind"], kind)

    def test_queue_and_cache_branch_targets_are_words(self):
        with running_server(fixture_configuration()) as base_url:
            for branch_target in ("QUEUE", "CACHE"):
                report = json.loads(request(
                    base_url, f"/api/view?branch_target={branch_target}"
                )[2])
                self.assertEqual(report["report_kind"], "word")

    def test_tree_fixture_selection_and_invalid_tree_kinds(self):
        with running_server(fixture_configuration()) as base_url:
            inferred = json.loads(request(
                base_url, "/api/view?branch_target=RAISE%20.....&tree=1"
            )[2])
            queue = json.loads(request(base_url, "/api/view/queue?tree=1")[2])
            workers = json.loads(request(base_url, "/api/view/workers?tree=1")[2])
            cache_status = request(base_url, "/api/view/cache?tree=1")[0]
            hotspot_status = request(base_url, "/api/view/hotspots?tree=1")[0]
        self.assertEqual(inferred["report_kind"], "branch")
        self.assertEqual(queue["report_kind"], "queue")
        self.assertEqual(workers["report_kind"], "workers")
        self.assertEqual((cache_status, hotspot_status), (400, 400))

    def test_comma_separated_status_is_accepted_but_parameters_do_not_repeat(self):
        report_request = parse_report_request(
            "/api/view/queue", "branch_status=evaluating,done"
        )
        self.assertEqual(
            report_request.filters.branch_statuses, ("evaluating", "done")
        )
        for query in (
            "branch_status=evaluating&branch_status=done",
            "limit=2&limit=3",
            "unknown=1",
        ):
            with self.subTest(query=query), self.assertRaises(InvalidRequest):
                parse_report_request("/api/view", query)

    def test_invalid_values_and_overlong_target_return_400(self):
        invalid_queries = (
            "tree=yes", "limit=x", "branch_target=BAD", "limit=0",
            "sample_size=0", "minimum_answer_count=5&maximum_answer_count=2",
            "branch_status=evaluating,evaluating", "branch_status=all,done",
            "branch_worker_status=active,", "branch_worker_status=working",
            "finalization_cursor=sideways:1:2", "finalization_cursor=after:1",
        )
        with running_server(fixture_configuration()) as base_url:
            for query in invalid_queries:
                with self.subTest(query=query):
                    path = "/api/view/hotspots?" + query if "sample" in query else "/api/view?" + query
                    self.assertEqual(request(base_url, path)[0], 400)
            self.assertEqual(
                request(base_url, "/api/view?branch_target=" + "A" * 8200)[0], 400
            )

    def test_unknown_branch_reference_returns_404(self):
        with running_server(self.live_configuration) as base_url:
            status, _headers, body = request(
                base_url, "/api/view?branch_target=%401234"
            )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["kind"], "not_found")

    def test_fixture_startup_validates_all_files_and_uses_no_database(self):
        fixtures = load_fixtures(FIXTURE_DIRECTORY)
        self.assertEqual(set(fixtures), set(FIXTURE_FILENAMES))
        with patch("report_server.collect_report", side_effect=AssertionError):
            with running_server(fixture_configuration()) as base_url:
                self.assertEqual(request(base_url, "/api/view")[0], 200)
        bad_directory = os.path.join(self.temporary_directory.name, "bad-fixtures")
        os.mkdir(bad_directory)
        for filename in FIXTURE_FILENAMES:
            with open(os.path.join(bad_directory, filename), "w") as fixture_file:
                json.dump({"schema_version": 1}, fixture_file)
        with self.assertRaisesRegex(ValueError, "schema version"):
            load_fixtures(bad_directory)

    def test_cache_fixture_uses_live_root_cache_shape(self):
        fixture = load_fixtures(FIXTURE_DIRECTORY)["cache.json"]
        live = collect_report(
            self.sources, parse_report_request("/api/view/cache", "")
        )
        self.assertEqual(set(fixture["data"]), set(live["data"]))
        self.assertEqual(
            set(fixture["data"]["summary"]), set(live["data"]["summary"])
        )
        self.assertEqual(
            set(fixture["data"]["distributions"]),
            set(live["data"]["distributions"]),
        )
        self.assertIn("recent_rows", fixture["data"])
        expected_recent_row_keys = {
            "branch_key_hex", "branch_reference", "best_guess", "best_erd",
            "max_remaining_depth", "solve_budget", "tainted", "updated_at",
        }
        self.assertEqual(
            set(fixture["data"]["recent_rows"][0]), expected_recent_row_keys
        )

    def test_build_configuration_uses_candidate_list_path(self):
        configuration = build_configuration("queue.sqlite3", "cache.sqlite3")
        defaults = ReportOpeners.defaults()
        self.assertEqual(
            configuration.sources.candidate_list_path,
            defaults.candidate_list_path,
        )

    def test_static_client_and_unknown_path(self):
        with running_server(fixture_configuration()) as base_url:
            status, headers, body = request(base_url, "/")
            deep_link = request(base_url, "/?branch_target=CRANE&tree=1")
            missing = request(base_url, "/report_client.html")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/html")
        self.assertIn(b"<title>ERD swarm reports</title>", body)
        self.assertEqual(deep_link[0], 200)
        self.assertEqual(deep_link[2], body)
        self.assertEqual(missing[0], 404)

    def test_mutation_methods_are_rejected(self):
        with running_server(fixture_configuration()) as base_url:
            for method in ("POST", "PUT", "DELETE"):
                with self.subTest(method=method):
                    status, headers, _body = request(
                        base_url, "/api/view", method=method
                    )
                    self.assertEqual(status, 405)
                    self.assertEqual(headers["Allow"], "GET")

    def test_api_headers_and_content_length(self):
        with running_server(fixture_configuration()) as base_url:
            status, headers, body = request(base_url, "/api/view")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_two_servers_do_not_share_configuration(self):
        first_fixtures = load_fixtures(FIXTURE_DIRECTORY)
        second_fixtures = load_fixtures(FIXTURE_DIRECTORY)
        second_fixtures["overview.json"] = dict(
            second_fixtures["overview.json"], generated_at=2000
        )
        first = ServerConfiguration(self.sources, CLIENT_PATH, fixtures=first_fixtures)
        second = ServerConfiguration(self.sources, CLIENT_PATH, fixtures=second_fixtures)
        with running_server(first) as first_url, running_server(second) as second_url:
            first_report = json.loads(request(first_url, "/api/view")[2])
            second_report = json.loads(request(second_url, "/api/view")[2])
        self.assertEqual(first_report["generated_at"], 1000)
        self.assertEqual(second_report["generated_at"], 2000)

    def test_partial_source_failure_remains_200(self):
        missing_sources = ReportOpeners(
            os.path.join(self.temporary_directory.name, "missing", "queue.sqlite3"),
            self.sources.cache_path,
            self.answer_list_path,
            self.guess_list_path,
        )
        configuration = ServerConfiguration(missing_sources, CLIENT_PATH)
        with running_server(configuration) as base_url:
            status, _headers, body = request(base_url, "/api/view")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["sources"]["queue"]["ok"])

    def test_unexpected_collector_failure_is_sanitized(self):
        with patch("report_server.collect_report", side_effect=RuntimeError("/secret/path")):
            with running_server(self.live_configuration) as base_url:
                status, _headers, body = request(base_url, "/api/view")
        self.assertEqual(status, 500)
        self.assertNotIn("secret", body.decode())

    def test_missing_browser_client_and_ambiguous_reference_have_normal_responses(self):
        missing_client = ServerConfiguration(self.sources, "missing-client.html")
        with running_server(missing_client) as base_url:
            status, _headers, body = request(base_url, "/")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"]["kind"], "server_error")

        class AmbiguousReference(ValueError):
            candidates = [object()]

        report = {"report_kind": "branch_reference_matches"}
        with (
            patch("report_server.collect_report", side_effect=AmbiguousReference("ambiguous")),
            patch("report_server.collect_ambiguous_branch_reference_report", return_value=report),
            running_server(self.live_configuration) as base_url,
        ):
            status, _headers, body = request(base_url, "/api/view?branch_target=%401234")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), report)

    def test_leaderboard_collection_failure_is_shared_and_returned_as_server_error(self):
        with patch("report_server.collect_report", side_effect=RuntimeError("failed")):
            with running_server(self.live_configuration) as base_url:
                status, _headers, body = request(base_url, "/api/view/leaderboard")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"]["kind"], "server_error")

    def test_concurrent_leaderboard_requests_share_one_collection(self):
        started = Event()
        release = Event()
        call_lock = Lock()
        call_count = 0
        report = load_fixtures(FIXTURE_DIRECTORY)["leaderboard.json"]

        def collect_slowly(_sources, _request):
            nonlocal call_count
            with call_lock:
                call_count += 1
            started.set()
            release.wait(2)
            return report

        with patch("report_server.collect_report", side_effect=collect_slowly):
            with running_server(self.live_configuration) as base_url:
                responses = []
                first = Thread(
                    target=lambda: responses.append(
                        request(base_url, "/api/view/leaderboard")
                    )
                )
                second = Thread(
                    target=lambda: responses.append(
                        request(base_url, "/api/view/leaderboard")
                    )
                )
                first.start()
                self.assertTrue(started.wait(1))
                second.start()
                time.sleep(0.05)
                release.set()
                first.join(2)
                second.join(2)
        self.assertEqual(call_count, 1)
        self.assertEqual(
            [status for status, _headers, _body in responses], [200, 200]
        )


class RevalidatedReportCacheTest(ReportServerTest):
    """When a rebuilt leaderboard is served, and when a cached one is."""

    def setUp(self):
        super().setUp()
        self.report = load_fixtures(FIXTURE_DIRECTORY)["leaderboard.json"]
        self.calls = 0

    def collect_counting(self, _sources, _request):
        self.calls += 1
        return self.report

    def get_leaderboard(self, base_url, times=1):
        for _ in range(times):
            status, _headers, body = request(base_url, "/api/view/leaderboard")
            self.assertEqual(status, 200)
        return body

    def test_an_unchanged_completion_signal_serves_the_cached_report(self):
        # The signal is what the whole cache turns on: while no opener has
        # finished, a rebuild cannot produce a different ranking, so asking for
        # one is pure cost against a client that polls every two seconds.
        with patch("report_server.collect_report", side_effect=self.collect_counting), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)):
            with running_server(self.live_configuration) as base_url:
                first = self.get_leaderboard(base_url)
                self.get_leaderboard(base_url, times=4)
                last = self.get_leaderboard(base_url)
        self.assertEqual(self.calls, 1)
        self.assertEqual(first, last)

    def test_a_moved_completion_signal_rebuilds(self):
        signals = iter([(3, 3), (3, 3), (4, 4), (4, 4)])
        with patch("report_server.collect_report", side_effect=self.collect_counting), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda _sources: next(signals)):
            with running_server(self.live_configuration) as base_url:
                self.get_leaderboard(base_url, times=4)
        self.assertEqual(self.calls, 2)

    def test_an_unreadable_signal_rebuilds_every_time(self):
        # None means the queue could not be read, so nothing is known about
        # whether an opener finished.  Serving a cached report on no
        # information would be asserting freshness the server cannot support.
        with patch("report_server.collect_report", side_effect=self.collect_counting), \
             patch("report_server.opener_completion_signal", return_value=None):
            with running_server(self.live_configuration) as base_url:
                self.get_leaderboard(base_url, times=3)
        self.assertEqual(self.calls, 3)

    def test_a_cached_report_expires_even_though_the_signal_stands_still(self):
        # A repair or an import changes the cache without completing any opener
        # work, so the signal cannot see it.  The age bound is what keeps such a
        # change from going unnoticed indefinitely.
        # The server's own machinery reads the clock too, so the fake advances
        # on the cache's reads rather than replacing time everywhere.
        now = [1000.0]

        def collect_and_age(sources, request):
            now[0] += report_server.REPORT_CACHE_MAX_AGE_SECONDS + 1
            return self.collect_counting(sources, request)

        with patch("report_server.collect_report", side_effect=collect_and_age), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)), \
             patch.object(report_server.time, "time", lambda: now[0]):
            with running_server(self.live_configuration) as base_url:
                self.get_leaderboard(base_url, times=2)
        self.assertEqual(self.calls, 2)

    def test_a_queue_backed_report_is_never_served_from_the_cache(self):
        # The queue reports exist to say what the swarm is doing now.  Serving
        # one a minute old would make a liveness dashboard report a liveness it
        # no longer has, so they are collected on every request however cheap
        # caching them would be.
        self.assertNotIn("queue", report_server.REVALIDATED_REPORT_KINDS)
        queue_report = load_fixtures(FIXTURE_DIRECTORY)["queue.json"]
        with patch("report_server.collect_report",
                   side_effect=lambda _s, _r: (
                       setattr(self, "calls", self.calls + 1) or queue_report)), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)):
            with running_server(self.live_configuration) as base_url:
                for _ in range(3):
                    status, _headers, _body = request(base_url, "/api/view/queue")
                    self.assertEqual(status, 200)
        self.assertEqual(self.calls, 3)


    def test_a_waiter_does_not_file_the_builders_body_under_its_own_token(self):
        # A waiter joins a build that started before its own signal read, so
        # the body it receives can predate the token it holds.  Filing under
        # that token would publish a body older than the state the token names,
        # and every later poll matching it would be served the stale ranking --
        # which inverts the guarantee that a stale signal costs freshness and
        # never correctness.
        started, release = Event(), Event()
        bodies = iter(["first", "second"])

        def collect_slowly(_sources, _request):
            started.set()
            release.wait(2)
            self.calls += 1
            return {"data": next(bodies), "sources": {}}

        # The builder reads (1, 1); the waiter that joins mid-build reads (2, 2).
        signals = iter([(1, 1), (2, 2), (2, 2), (2, 2)])
        with patch("report_server.collect_report", side_effect=collect_slowly), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda _sources: next(signals)):
            with running_server(self.live_configuration) as base_url:
                responses = []
                builder = Thread(target=lambda: responses.append(
                    request(base_url, "/api/view/leaderboard")))
                builder.start()
                self.assertTrue(started.wait(1))
                waiter = Thread(target=lambda: responses.append(
                    request(base_url, "/api/view/leaderboard")))
                waiter.start()
                time.sleep(0.05)
                release.set()
                builder.join(3)
                waiter.join(3)
                # A third request holding (2, 2) must not be handed the body
                # built while the signal still read (1, 1).
                _status, _headers, body = request(
                    base_url, "/api/view/leaderboard")
        self.assertEqual(json.loads(body)["data"], "second")

    def test_a_waiter_is_released_only_once_the_body_is_published(self):
        # Releasing waiters before publishing leaves a window in which the
        # build is finished, the in-flight marker is gone, and the entry is not
        # yet stored.  A request landing there finds neither and rebuilds a
        # report already in hand.
        #
        # That window is a dict write, and reaching it takes an HTTP round
        # trip, so it is not reachable by timing alone -- this test widens it
        # by slowing the release of the lock the marker is dropped under, which
        # is the last thing the old ordering did before publishing.  Under the
        # correct ordering the entry is already stored by then, so widening
        # changes nothing.
        started, release = Event(), Event()
        widen = [False]

        class SlowReleaseLock:
            def __init__(self):
                self._lock = threading.Lock()

            def __enter__(self):
                return self._lock.__enter__()

            def __exit__(self, *details):
                result = self._lock.__exit__(*details)
                if widen[0]:
                    time.sleep(0.4)
                return result

        def collect_slowly(_sources, _request):
            self.calls += 1
            started.set()
            release.wait(2)
            return self.report

        followup = []
        with patch("report_server.Lock", SlowReleaseLock), \
             patch("report_server.collect_report", side_effect=collect_slowly), \
             patch("report_server.opener_completion_signal", return_value=(5, 5)):
            with running_server(self.live_configuration) as base_url:
                builder = Thread(target=lambda: request(
                    base_url, "/api/view/leaderboard"))
                builder.start()
                self.assertTrue(started.wait(1))

                def wait_then_ask():
                    request(base_url, "/api/view/leaderboard")
                    followup.append(request(base_url, "/api/view/leaderboard"))

                waiter = Thread(target=wait_then_ask)
                waiter.start()
                time.sleep(0.05)
                widen[0] = True
                release.set()
                builder.join(5)
                waiter.join(5)
        self.assertEqual([status for status, _h, _b in followup], [200])
        self.assertEqual(self.calls, 1)

    def test_a_build_that_recorded_a_source_error_is_not_cached(self):
        # collect_leaderboard_report catches its own SQLite errors and returns
        # an ordinary report with an empty ranking and the error on the source,
        # so a degraded build is a normal-looking 200.  Caching one pins an
        # empty leaderboard until it expires, where every poll used to recover
        # on the next request.
        degraded = {"data": {"rows": []},
                    "sources": {"cache": {"ok": False, "error": "disk I/O error"}}}
        with patch("report_server.collect_report",
                   side_effect=lambda _s, _r: (
                       setattr(self, "calls", self.calls + 1) or degraded)), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)):
            with running_server(self.live_configuration) as base_url:
                self.get_leaderboard(base_url, times=3)
        self.assertEqual(self.calls, 3)

    def test_a_source_that_was_never_consulted_does_not_block_caching(self):
        # The leaderboard never opens the queue, so its queue source reports
        # ok=false with no error.  That is "not consulted", not a failure, and
        # must not be read as one -- it would disable the cache entirely.
        intact = {"data": {"rows": []},
                  "sources": {"queue": {"ok": False, "error": None},
                              "cache": {"ok": True, "error": None}}}
        with patch("report_server.collect_report",
                   side_effect=lambda _s, _r: (
                       setattr(self, "calls", self.calls + 1) or intact)), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)):
            with running_server(self.live_configuration) as base_url:
                self.get_leaderboard(base_url, times=3)
        self.assertEqual(self.calls, 1)

    def test_a_limit_evicted_by_newer_ones_is_rebuilt_rather_than_retained(self):
        # The cache key is the request, and a request carries a user-controlled
        # limit, so each distinct ?limit= value is a separate multi-megabyte
        # body.  Expiry alone only stops an entry being reused; without a cap
        # the process grows until it is restarted.  Eviction is observable as a
        # rebuild: the earliest limit is gone once enough newer ones arrive.
        cap = report_server.REPORT_CACHE_MAX_ENTRIES
        with patch("report_server.collect_report", side_effect=self.collect_counting), \
             patch("report_server.opener_completion_signal", return_value=(3, 3)):
            with running_server(self.live_configuration) as base_url:
                for limit in range(1, cap + 1):
                    request(base_url, f"/api/view/leaderboard?limit={limit}")
                self.assertEqual(self.calls, cap)
                # Still held: re-asking for one of them rebuilds nothing.
                request(base_url, "/api/view/leaderboard?limit=1")
                self.assertEqual(self.calls, cap)
                # Push past the cap, which evicts the oldest entries.
                for limit in range(cap + 1, cap + 4):
                    request(base_url, f"/api/view/leaderboard?limit={limit}")
                self.assertEqual(self.calls, cap + 3)
                request(base_url, "/api/view/leaderboard?limit=2")
        self.assertEqual(self.calls, cap + 4)


class ReportServerMainTest(unittest.TestCase):
    def test_port_collision_exits_cleanly_instead_of_crashing(self):
        with running_server(fixture_configuration()) as base_url:
            port = base_url.rsplit(":", 1)[1]
            argv = [
                "report_server.py",
                "--port", port,
                "--queue-path", "unused-queue",
                "--cache-path", "unused-cache",
            ]
            with patch("sys.argv", argv), \
                 patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("already in use", stderr.getvalue())

    def test_main_closes_server_after_keyboard_interrupt(self):
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with (
            patch("sys.argv", ["report_server.py"]),
            patch("report_server.ensure_runtime_dir"),
            patch("report_server.build_configuration", return_value=fixture_configuration()),
            patch("report_server.ThreadingHTTPServer", return_value=server),
        ):
            main()
        server.server_close.assert_called_once_with()


class ReportServerStartupTest(unittest.TestCase):
    def test_a_startup_failure_that_is_not_a_port_collision_is_raised(self):
        refused = OSError(errno.EACCES, "permission denied")
        with (
            patch("sys.argv", ["report_server.py"]),
            patch("report_server.ensure_runtime_dir"),
            patch(
                "report_server.build_configuration",
                return_value=fixture_configuration(),
            ),
            patch("report_server.ThreadingHTTPServer", side_effect=refused),
        ):
            with self.assertRaises(OSError) as raised:
                main()
        self.assertEqual(raised.exception.errno, errno.EACCES)


class SourcesRequestTest(unittest.TestCase):
    def test_request_validation_rejects_each_endpoint_specific_option(self):
        cases = (
            ("/api/view", "tree_parent=RAISE+-----", "require tree"),
            ("/api/view", "tree=1&tree_cursor=oops", "tree page group"),
            ("/api/view", "limit=0", "at least 1"),
            ("/api/view", "minimum_answer_count=3&maximum_answer_count=2", "cannot exceed"),
            ("/api/view", "sort=nope", "invalid sort"),
            ("/api/view", "group_by=nope", "invalid group_by"),
            ("/api/view", "by=nodes", "by requires the hotspots"),
            ("/api/view", "epoch=3",
             "require the hotspots or work-distribution"),
            ("/api/view/work-distribution", "by=nodes",
             "by requires the hotspots"),
            ("/api/view/work-distribution", "tree=1", "tree cannot be used"),
            ("/api/view/work-distribution", "answers=1", "answers requires"),
            ("/api/view/work-distribution", "sample_size=1000",
             "sample_size requires the hotspots endpoint"),
            ("/api/view/work-distribution", "finalization_cursor=after:1:1",
             "cannot use finalization_cursor"),
            ("/api/view/work-distribution", "priority=0", "cannot use --priority"),
            ("/api/view/hotspots", "by=nope", "invalid hotspot"),
            ("/api/view", "claims=1", "singular branch"),
            ("/api/view", "answers=1", "answers requires"),
        )
        for path, query, message in cases:
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, message):
                    parse_report_request(path, query)

    def test_request_validation_handles_cursor_and_boolean_failures(self):
        for query, message in (
            ("tree=maybe", "must be 1"),
            ("limit=one", "must be an integer"),
            ("finalization_cursor=after:bad:1", "finalization_cursor"),
            ("since_seconds=0", "at least 1"),
            ("sample_size=0", "at least 1"),
        ):
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, message):
                    parse_report_request("/api/view", query)

    def test_hotspots_default_and_tree_parent_are_normalized(self):
        hotspots = parse_report_request("/api/view/hotspots", "")
        self.assertEqual(hotspots.hotspot_field, "nodes")
        self.assertEqual(hotspots.filters.limit, 10)
        request = parse_report_request(
            "/api/view", "tree=1&tree_parent=raise+-----"
        )
        self.assertEqual(request.tree_parent, "RAISE -----")

    def test_configuration_uses_telemetry_only_for_the_default_queue(self):
        defaults = ReportOpeners.defaults()
        with patch("report_server.load_fixtures", return_value={}) as load:
            configured = build_configuration("another.sqlite3", "cache.sqlite3", "fixtures")
        self.assertIsNone(configured.sources.telemetry_path)
        load.assert_called_once_with("fixtures")
        self.assertEqual(defaults.queue_path, ReportOpeners.defaults().queue_path)

    def test_bare_endpoint_and_word_target_are_accepted(self):
        rooted = parse_report_request("/api/view/openers", "")
        worded = parse_report_request("/api/view/openers", "branch_target=SALET")
        self.assertEqual(rooted.report_kind, "openers")
        self.assertEqual(rooted.branch_target.kind, "root")
        self.assertEqual(worded.report_kind, "openers")
        self.assertEqual(worded.branch_target.trailing_word, "salet")

    def test_limit_reaches_the_filters(self):
        request = parse_report_request("/api/view/openers", "limit=2")
        self.assertEqual(request.filters.limit, 2)

    def test_branch_target_and_tree_are_rejected(self):
        # A membership row names a branch already; a spine ending in a pattern
        # selects one branch, which the opener report has no view of, and there
        # is no topology to lay out as a tree.
        for query, message in (
            ("branch_target=RAISE+-----", "trailing word"),
            ("tree=1", "tree cannot be used with openers"),
            ("worker=worker-1", "worker requires the workers endpoint"),
            ("answers=1", "answers requires"),
        ):
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, message):
                    parse_report_request("/api/view/openers", query)

    def test_fixture_shapes_match_a_live_opener_report(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path = os.path.join(directory, "queue.sqlite3")
            queue = ERDQueue(queue_path)
            shared = encode_subset(["cigar", "rebut"])
            queue.add_pending_many([(shared, 2, 3, "salet", "-y---")])
            queue.add_pending_many([(shared, 2, 5, "raise", "-----")])
            queue.close()
            sources = ReportOpeners(
                queue_path, os.path.join(directory, "cache.sqlite3"),
                "unused-answers", "unused-guesses",
            )
            # The collapsed fixture stands in for the browser's default
            # request, which groups by state, so the live comparison must ask
            # for the same shape.
            live = collect_report(
                sources, parse_report_request("/api/view/openers", "group_by=state")
            )
            live_word = collect_report(
                sources,
                parse_report_request("/api/view/openers", "branch_target=SALET"),
            )
        fixtures = load_fixtures(FIXTURE_DIRECTORY)
        collapsed, worded = fixtures["openers.json"], fixtures["openers-word.json"]
        self.assertEqual(set(collapsed["data"]), set(live["data"]))
        self.assertEqual(
            set(collapsed["data"]["summary"][0]), set(live["data"]["summary"][0])
        )
        # The collapsed report is requests only: branch rows wait for a word.
        self.assertEqual(collapsed["data"]["rows"], [])
        self.assertEqual(live["data"]["rows"], [])
        self.assertEqual(
            set(worded["data"]["rows"][0]), set(live_word["data"]["rows"][0])
        )
        # The word fixture must carry the shared-ownership case the branch rows
        # exist to show: one branch, two owning requests, one effective priority.
        self.assertTrue(any(row["is_shared"] for row in worded["data"]["rows"]))

    def test_word_target_selects_the_word_scoped_fixture(self):
        with running_server(fixture_configuration()) as base_url:
            collapsed = json.loads(request(base_url, "/api/view/openers")[2])
            worded = json.loads(
                request(base_url, "/api/view/openers?branch_target=SALET")[2]
            )
        self.assertEqual(collapsed["data"]["rows"], [])
        self.assertTrue(worded["data"]["rows"])


class RootProgressRequestTest(unittest.TestCase):
    def test_bare_word_target_is_accepted(self):
        request = parse_report_request(
            "/api/view/root-progress", "branch_target=SCOPE")
        self.assertEqual(request.report_kind, "root_progress")
        self.assertEqual(request.branch_target.trailing_word, "scope")
        self.assertEqual(request.branch_target.steps, ())

    def test_spine_with_more_than_one_guess_is_accepted(self):
        # "Why is RAISE ----- SALET taking so long" is the same question the
        # report answers for a root, at a greater guess_depth.
        request = parse_report_request(
            "/api/view/root-progress",
            "branch_target=RAISE+-----+SALET")
        self.assertEqual(request.report_kind, "root_progress")
        self.assertEqual(request.branch_target.trailing_word, "salet")

    def test_target_without_a_trailing_word_is_rejected(self):
        # A spine ending in a pattern names a branch, which has no response
        # groups of its own to report.
        with self.assertRaisesRegex(InvalidRequest, "ending in a word"):
            parse_report_request(
                "/api/view/root-progress",
                "branch_target=RAISE+-----")

    def test_word_view_display_state_is_rejected_not_ignored(self):
        # Each of these is carried by the word view's state query. The client
        # must not forward them, and this pins why: the report rejects them,
        # so a forwarded parameter fails the whole panel rather than being
        # dropped.
        for query in (
            "branch_target=SCOPE&group_by=worker_presence",
            "branch_target=SCOPE&by=nodes",
            "branch_target=SCOPE&worker=worker-1",
        ):
            with self.subTest(query=query):
                with self.assertRaises(InvalidRequest):
                    parse_report_request("/api/view/root-progress", query)


if __name__ == "__main__":
    unittest.main()


class ConditionalLeaderboardRequestTest(ReportServerTest):
    """A poll that would receive the same ranking receives nothing instead.

    The leaderboard's answer moves when an opener completes -- about every 27
    minutes -- and the client polls every two seconds, so an unchanged answer
    is the ordinary case.  The signal that decides whether to rebuild is the
    same one that decides whether to send, so it is spent as an ETag.
    """

    def setUp(self):
        super().setUp()
        self.report = load_fixtures(FIXTURE_DIRECTORY)["leaderboard.json"]
        self.tokens = ["opener-signal-1"]

    def test_a_ranking_carries_the_signal_it_was_revalidated_against(self):
        with patch("report_server.collect_report", return_value=self.report), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda *_: self.tokens[0]):
            with running_server(self.live_configuration) as base_url:
                status, headers, body = request(base_url, "/api/view/leaderboard")
        self.assertEqual(status, 200)
        self.assertTrue(headers.get("ETag"), "no ETag to revalidate against")
        self.assertTrue(body)

    def test_an_unchanged_ranking_answers_304_with_no_body(self):
        with patch("report_server.collect_report", return_value=self.report), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda *_: self.tokens[0]):
            with running_server(self.live_configuration) as base_url:
                _status, headers, first = request(
                    base_url, "/api/view/leaderboard")
                tag = headers["ETag"]
                status, _headers, body = request(
                    base_url, "/api/view/leaderboard",
                    headers={"If-None-Match": tag})
        self.assertEqual(status, 304)
        self.assertEqual(body, b"", "a 304 must carry no body")
        self.assertTrue(first, "the first response should have carried one")

    def _ranking(self, first_word):
        report = copy.deepcopy(self.report)
        columns = report["data"]["columns"]
        columns["words"] = first_word + columns["words"][5:]
        return report

    def test_a_changed_ranking_returns_a_body_and_a_new_tag(self):
        rankings = [self._ranking("salet")]
        with patch("report_server.collect_report",
                   side_effect=lambda *_: rankings[0]), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda *_: self.tokens[0]):
            with running_server(self.live_configuration) as base_url:
                _status, headers, _body = request(
                    base_url, "/api/view/leaderboard")
                stale_tag = headers["ETag"]
                # An opener finished and took the top of the ranking with it.
                self.tokens[0] = "opener-signal-2"
                rankings[0] = self._ranking("tarse")
                status, new_headers, body = request(
                    base_url, "/api/view/leaderboard",
                    headers={"If-None-Match": stale_tag})
        self.assertEqual(status, 200)
        self.assertIn(b"tarse", body)
        self.assertNotEqual(new_headers["ETag"], stale_tag)

    def test_a_rebuild_that_changes_nothing_still_revalidates(self):
        # An opener can finish without moving the ranking the client holds --
        # it lands below the rows being shown.  The body is identical, so the
        # client's copy is current and there is nothing to send.
        with patch("report_server.collect_report",
                   side_effect=lambda *_: copy.deepcopy(self.report)), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda *_: self.tokens[0]):
            with running_server(self.live_configuration) as base_url:
                _status, headers, _body = request(
                    base_url, "/api/view/leaderboard")
                tag = headers["ETag"]
                self.tokens[0] = "opener-signal-2"
                status, _new_headers, body = request(
                    base_url, "/api/view/leaderboard",
                    headers={"If-None-Match": tag})
        self.assertEqual(status, 304)
        self.assertEqual(body, b"")

    def test_a_rebuild_the_signal_cannot_see_still_reaches_the_client(self):
        """The backstop rebuild must be visible, or it accomplishes nothing.

        `opener_completion_signal` is deliberately not exhaustive: a repair, a
        reverification or an import changes the cache while completing no queue
        work.  REPORT_CACHE_MAX_AGE_SECONDS exists to catch exactly that, by
        rebuilding once the entry ages out even though the signal has not
        moved.

        A validator taken from the signal would still match across that
        rebuild, so the client would be told nothing had changed while the
        server held a ranking it had already replaced -- and with the queue
        stopped, no opener ever completes and the client never learns.
        """
        rankings = [self._ranking("salet")]
        with patch("report_server.collect_report",
                   side_effect=lambda *_: rankings[0]), \
             patch("report_server.opener_completion_signal",
                   side_effect=lambda *_: self.tokens[0]), \
             patch.object(report_server, "REPORT_CACHE_MAX_AGE_SECONDS", 0):
            with running_server(self.live_configuration) as base_url:
                _status, headers, _body = request(
                    base_url, "/api/view/leaderboard")
                stale_tag = headers["ETag"]
                # A repair rewrote a branch result.  No opener completed, so
                # the signal stands exactly where it was.
                rankings[0] = self._ranking("crane")
                status, new_headers, body = request(
                    base_url, "/api/view/leaderboard",
                    headers={"If-None-Match": stale_tag})
        self.assertEqual(
            status, 200,
            "a rebuild the signal cannot see was withheld from the client")
        self.assertIn(b"crane", body)
        self.assertNotEqual(new_headers["ETag"], stale_tag)


class RevalidatedReportsDeclareThemselvesTest(ReportServerTest):
    """A report the poll does not refresh has to say when its data is from.

    Every other view is rebuilt on each two-second poll, so the cadence is the
    freshness.  The leaderboard is not: its data is as old as the last opener
    to finish, up to REPORT_CACHE_MAX_AGE_SECONDS.  Without a mark on the
    report the client cannot tell the two apart, and a reader would take the
    poll interval for the age of what is on screen.
    """

    def setUp(self):
        super().setUp()
        self.report = load_fixtures(FIXTURE_DIRECTORY)["leaderboard.json"]

    def test_a_revalidated_report_is_marked_for_the_client(self):
        with patch("report_server.collect_report", return_value=dict(self.report)), \
             patch("report_server.opener_completion_signal", return_value=(1, 1)):
            with running_server(self.live_configuration) as base_url:
                _status, _headers, body = request(base_url, "/api/view/leaderboard")
        self.assertTrue(json.loads(body)["revalidated"])

    def test_a_live_report_is_not_marked(self):
        # The queue report's subject is what the swarm is doing right now, so
        # it is rebuilt every poll and carries no age to declare.
        with running_server(self.live_configuration) as base_url:
            _status, _headers, body = request(base_url, "/api/view/queue")
        self.assertNotIn("revalidated", json.loads(body))
