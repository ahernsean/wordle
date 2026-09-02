"""Tests for the read-only HTTP report adapter."""

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from threading import Event, Lock, Thread
import io
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from erd_queue import ERDQueue, encode_subset
from report_model import ReportSources, collect_report
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


def request(base_url, path, method="GET"):
    try:
        with urlopen(Request(base_url + path, method=method), timeout=3) as response:
            body = response.read()
            return response.status, response.headers, body
    except HTTPError as error:
        return error.code, error.headers, error.read()


def fixture_configuration():
    sources = ReportSources("unused-queue", "unused-cache", "unused-answers", "unused-guesses")
    return ServerConfiguration(
        sources, CLIENT_PATH, FIXTURE_DIRECTORY, load_fixtures(FIXTURE_DIRECTORY)
    )


class ReportServerTest(unittest.TestCase):
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
        self.sources = ReportSources(
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
            ("tree=1&tree_cursor=not-a-word", "five-letter word"),
        ):
            with self.subTest(query=query):
                with self.assertRaisesRegex(InvalidRequest, message):
                    parse_report_request("/api/view", query)

    def test_root_overview_defaults_to_active_worker_and_all_disables_filter(self):
        default_request = parse_report_request("/api/view", "")
        all_request = parse_report_request("/api/view", "branch_worker_status=all")
        word_request = parse_report_request("/api/view", "branch_target=RAISE")
        self.assertEqual(default_request.filters.branch_worker_statuses, ("active",))
        self.assertEqual(all_request.filters.branch_worker_statuses, ())
        self.assertEqual(word_request.filters.branch_worker_statuses, ())
        self.assertEqual(default_request.filters.branch_statuses, ())

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
        defaults = ReportSources.defaults()
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
        missing_sources = ReportSources(
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


class SourcesRequestTest(unittest.TestCase):
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

    def test_fixture_shapes_match_a_live_source_report(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path = os.path.join(directory, "queue.sqlite3")
            queue = ERDQueue(queue_path)
            shared = encode_subset(["cigar", "rebut"])
            queue.add_pending_many([(shared, 2, 3, "salet", "-y---")])
            queue.add_pending_many([(shared, 2, 5, "raise", "-----")])
            queue.close()
            sources = ReportSources(
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
