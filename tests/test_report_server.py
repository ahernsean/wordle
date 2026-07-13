"""Tests for the read-only HTTP report adapter."""

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from threading import Thread
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from report_model import ReportSources, collect_report
from report_server import (
    FIXTURE_FILENAMES,
    InvalidRequest,
    ServerConfiguration,
    build_configuration,
    load_fixtures,
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
        self.live_configuration = ServerConfiguration(self.sources, CLIENT_PATH)

    def test_live_root_contract_matches_direct_collection_shape(self):
        with running_server(self.live_configuration) as base_url:
            status, _headers, body = request(base_url, "/api/view")
        self.assertEqual(status, 200)
        served = json.loads(body)
        direct = collect_report(self.sources, parse_report_request("/api/view", ""))
        served.pop("generated_at")
        direct.pop("generated_at")
        self.assertEqual(served, direct)

    def test_selector_inference_uses_one_endpoint(self):
        with running_server(fixture_configuration()) as base_url:
            word = json.loads(request(base_url, "/api/view?selector=CACHE")[2])
            branch = json.loads(request(
                base_url, "/api/view?selector=RAISE%20....."
            )[2])
        self.assertEqual(word["report_kind"], "word")
        self.assertEqual(branch["report_kind"], "branch")

    def test_tree_and_active_only_reach_normalized_request(self):
        report_request = parse_report_request(
            "/api/view/queue", "tree=true&active_only=1&limit=4"
        )
        self.assertTrue(report_request.tree)
        self.assertTrue(report_request.filters.active_only)
        self.assertEqual(report_request.filters.limit, 4)

    def test_every_explicit_endpoint_returns_its_kind(self):
        with running_server(fixture_configuration()) as base_url:
            for kind in ("queue", "workers", "cache", "hotspots"):
                with self.subTest(kind=kind):
                    status, _headers, body = request(base_url, f"/api/view/{kind}")
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["report_kind"], kind)

    def test_queue_and_cache_selectors_are_words(self):
        with running_server(fixture_configuration()) as base_url:
            for selector in ("QUEUE", "CACHE"):
                report = json.loads(request(
                    base_url, f"/api/view?selector={selector}"
                )[2])
                self.assertEqual(report["report_kind"], "word")

    def test_tree_fixture_selection_and_invalid_tree_kinds(self):
        with running_server(fixture_configuration()) as base_url:
            inferred = json.loads(request(
                base_url, "/api/view?selector=RAISE%20.....&tree=1"
            )[2])
            queue = json.loads(request(base_url, "/api/view/queue?tree=1")[2])
            workers = json.loads(request(base_url, "/api/view/workers?tree=1")[2])
            cache_status = request(base_url, "/api/view/cache?tree=1")[0]
            hotspot_status = request(base_url, "/api/view/hotspots?tree=1")[0]
        self.assertEqual(inferred["report_kind"], "branch")
        self.assertEqual(queue["report_kind"], "queue")
        self.assertEqual(workers["report_kind"], "workers")
        self.assertEqual((cache_status, hotspot_status), (400, 400))

    def test_repeated_status_is_accepted_but_other_duplicates_are_not(self):
        report_request = parse_report_request(
            "/api/view/queue", "status=pending&status=done"
        )
        self.assertEqual(report_request.filters.statuses, ("pending", "done"))
        for query in ("limit=2&limit=3", "unknown=1"):
            with self.subTest(query=query), self.assertRaises(InvalidRequest):
                parse_report_request("/api/view", query)

    def test_invalid_values_and_overlong_target_return_400(self):
        invalid_queries = (
            "tree=yes", "limit=x", "selector=BAD", "limit=0",
            "sample_size=0", "minimum_answer_count=5&maximum_answer_count=2",
        )
        with running_server(fixture_configuration()) as base_url:
            for query in invalid_queries:
                with self.subTest(query=query):
                    path = "/api/view/hotspots?" + query if "sample" in query else "/api/view?" + query
                    self.assertEqual(request(base_url, path)[0], 400)
            self.assertEqual(
                request(base_url, "/api/view?selector=" + "A" * 8200)[0], 400
            )

    def test_unknown_branch_reference_returns_404(self):
        with running_server(self.live_configuration) as base_url:
            status, _headers, body = request(
                base_url, "/api/view?selector=%401234"
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
                json.dump({"schema_version": 2}, fixture_file)
        with self.assertRaisesRegex(ValueError, "schema version"):
            load_fixtures(bad_directory)

    def test_static_client_and_unknown_path(self):
        with running_server(fixture_configuration()) as base_url:
            status, headers, body = request(base_url, "/")
            missing = request(base_url, "/report_client.html")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/html")
        self.assertIn(b"<title>ERD swarm reports</title>", body)
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


if __name__ == "__main__":
    unittest.main()
