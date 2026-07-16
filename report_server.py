#!/usr/bin/env python3
"""Read-only HTTP adapter for shared ERD swarm reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
from urllib.parse import parse_qs, urlsplit

from report_model import (
    BRANCH_PHASES,
    BRANCH_STATUSES,
    SCHEMA_VERSION,
    ReportFilters,
    ReportRequest,
    ReportSources,
    collect_report,
    parse_branch_filter,
    parse_report_selector,
    validate_report_request,
)
from runtime_paths import DEFAULT_CACHE_PATH, DEFAULT_QUEUE_PATH


FIXTURE_FILENAMES = (
    "overview.json",
    "word.json",
    "branch.json",
    "tree.json",
    "queue.json",
    "queue-tree.json",
    "workers.json",
    "workers-tree.json",
    "cache.json",
    "hotspots.json",
)
BOOLEAN_PARAMETERS = {"tree", "claims", "answers"}
INTEGER_PARAMETERS = {
    "minimum_answer_count", "maximum_answer_count", "budget", "priority",
    "limit", "epoch", "since_seconds", "sample_size",
}
SCALAR_PARAMETERS = {
    "selector", "sort", "by", "worker", "branch_status", "branch_phase",
    *BOOLEAN_PARAMETERS,
    *INTEGER_PARAMETERS,
}
ALLOWED_PARAMETERS = SCALAR_PARAMETERS
SORT_FIELDS = {"default", "age", "size", "workers", "priority", "nodes", "slowest"}
HOTSPOT_FIELDS = {
    "nodes", "age", "size", "workers", "priority", "slowest",
    "evaluated-candidates", "bulk-completed-candidates", "cut-reuse",
    "coordination",
}
class InvalidRequest(ValueError):
    pass


@dataclass(frozen=True)
class ServerConfiguration:
    sources: ReportSources
    client_path: str
    fixture_directory: str | None = None
    fixtures: dict[str, dict] | None = None


def _single_value(parameters, name, default=None):
    values = parameters.get(name)
    if values is None:
        return default
    if len(values) != 1:
        raise InvalidRequest(f"parameter {name!r} may not be repeated")
    return values[0]


def _boolean_value(parameters, name, default=False):
    value = _single_value(parameters, name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in ("1", "true"):
        return True
    if normalized in ("0", "false"):
        return False
    raise InvalidRequest(f"parameter {name!r} must be 1, 0, true, or false")


def _integer_value(parameters, name):
    value = _single_value(parameters, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise InvalidRequest(f"parameter {name!r} must be an integer") from error


def parse_report_request(path, query):
    explicit_kind = {
        "/api/view": "auto",
        "/api/view/queue": "queue",
        "/api/view/workers": "workers",
        "/api/view/cache": "cache",
        "/api/view/hotspots": "hotspots",
    }.get(path)
    if explicit_kind is None:
        raise KeyError(path)
    parameters = parse_qs(query, keep_blank_values=True)
    unknown = sorted(set(parameters) - ALLOWED_PARAMETERS)
    if unknown:
        raise InvalidRequest(f"unknown parameter {unknown[0]!r}")
    for name in SCALAR_PARAMETERS:
        _single_value(parameters, name)

    try:
        selector = parse_report_selector(_single_value(parameters, "selector"))
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    tree = _boolean_value(parameters, "tree")
    include_claims = _boolean_value(parameters, "claims")
    include_answers = _boolean_value(parameters, "answers")
    branch_status_value = _single_value(parameters, "branch_status")
    branch_phase_value = _single_value(parameters, "branch_phase")
    try:
        branch_statuses = (
            parse_branch_filter(
                branch_status_value, "branch status", BRANCH_STATUSES
            )
            if branch_status_value is not None
            else (
                ("active",)
                if explicit_kind == "auto" and selector.kind == "root" and not tree
                else ()
            )
        )
        branch_phases = (
            parse_branch_filter(branch_phase_value, "branch phase", BRANCH_PHASES)
            if branch_phase_value is not None else ()
        )
    except ValueError as error:
        raise InvalidRequest(str(error)) from error

    integer_values = {
        name: _integer_value(parameters, name) for name in INTEGER_PARAMETERS
    }
    minimum_answer_count = integer_values["minimum_answer_count"]
    maximum_answer_count = integer_values["maximum_answer_count"]
    limit = integer_values["limit"]
    since_seconds = integer_values["since_seconds"]
    sample_size = integer_values["sample_size"]
    if limit is not None and limit < 1:
        raise InvalidRequest("limit must be at least 1")
    if since_seconds is not None and since_seconds < 1:
        raise InvalidRequest("since_seconds must be at least 1")
    if sample_size is not None and sample_size < 1:
        raise InvalidRequest("sample_size must be at least 1")
    if (
        minimum_answer_count is not None
        and maximum_answer_count is not None
        and minimum_answer_count > maximum_answer_count
    ):
        raise InvalidRequest(
            "minimum_answer_count cannot exceed maximum_answer_count"
        )
    sort = _single_value(parameters, "sort")
    if sort is not None and sort not in SORT_FIELDS:
        raise InvalidRequest(f"invalid sort field {sort!r}")

    hotspot_field = _single_value(parameters, "by")
    historical_options_present = any(
        integer_values[name] is not None
        for name in ("epoch", "since_seconds", "sample_size")
    )
    if explicit_kind != "hotspots" and (hotspot_field or historical_options_present):
        raise InvalidRequest(
            "by, epoch, since_seconds, and sample_size require the hotspots endpoint"
        )
    if hotspot_field is not None and hotspot_field not in HOTSPOT_FIELDS:
        raise InvalidRequest(f"invalid hotspot field {hotspot_field!r}")
    hotspot_field = hotspot_field or ("nodes" if explicit_kind == "hotspots" else None)
    if tree and explicit_kind in ("cache", "hotspots"):
        raise InvalidRequest(f"tree cannot be used with {explicit_kind}")
    worker_id = _single_value(parameters, "worker")
    if worker_id is not None and explicit_kind != "workers":
        raise InvalidRequest("worker requires the workers endpoint")
    if include_claims and (
        explicit_kind != "auto"
        or selector.kind not in ("branch", "branch_reference")
    ):
        raise InvalidRequest("claims requires a singular branch selector")
    if include_answers and (
        tree
        or explicit_kind in ("queue", "workers")
        or (explicit_kind == "auto" and selector.kind == "root")
    ):
        raise InvalidRequest("answers requires a word or branch report without tree")
    if explicit_kind == "hotspots" and limit is None:
        limit = 10

    filters = ReportFilters(
        branch_statuses=branch_statuses,
        branch_phases=branch_phases,
        minimum_answer_count=minimum_answer_count,
        maximum_answer_count=maximum_answer_count,
        budget=integer_values["budget"],
        priority=integer_values["priority"],
        sort=sort,
        limit=limit,
    )
    request = ReportRequest(
        report_kind=explicit_kind,
        selector=selector,
        include_claims=include_claims,
        include_answers=include_answers,
        tree=tree,
        filters=filters,
        worker_id=worker_id,
        hotspot_field=hotspot_field,
        epoch=integer_values["epoch"],
        since_seconds=since_seconds or 3600,
        sample_size=min(sample_size or 50_000, 1_000_000),
    )
    try:
        validate_report_request(request)
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    return request


def fixture_name_for_request(path, request):
    if path == "/api/view":
        if request.tree:
            return "tree.json"
        return {
            "root": "overview.json",
            "word": "word.json",
            "branch": "branch.json",
            "branch_reference": "branch.json",
        }[request.selector.kind]
    kind = request.report_kind
    if request.tree:
        return f"{kind}-tree.json"
    return f"{kind}.json"


def load_fixtures(directory):
    fixtures = {}
    for filename in FIXTURE_FILENAMES:
        with open(os.path.join(directory, filename), encoding="utf-8") as fixture_file:
            report = json.load(fixture_file)
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"fixture {filename!r} has schema version "
                f"{report.get('schema_version')!r}, expected {SCHEMA_VERSION}"
            )
        fixtures[filename] = report
    return fixtures


def make_handler(configuration):
    class ReportHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def _write(self, status, content_type, body, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, value, extra_headers=None):
            body = json.dumps(value, sort_keys=True).encode("utf-8")
            self._write(
                status, "application/json; charset=utf-8", body, extra_headers
            )

        def _error(self, status, kind, message, extra_headers=None):
            self._json(
                status, {"error": {"kind": kind, "message": message}},
                extra_headers,
            )

        def do_GET(self):
            if len(self.path.encode("utf-8")) > 8192:
                self._error(400, "invalid_request", "request target is too long")
                return
            target = urlsplit(self.path)
            if target.path in ("/", "/index.html"):
                try:
                    with open(configuration.client_path, "rb") as client_file:
                        body = client_file.read()
                except OSError:
                    self._error(500, "server_error", "browser client unavailable")
                    return
                self._write(200, "text/html; charset=utf-8", body)
                return
            try:
                request = parse_report_request(target.path, target.query)
            except KeyError:
                self._error(404, "not_found", "route not found")
                return
            except InvalidRequest as error:
                self._error(400, "invalid_request", str(error))
                return
            try:
                if configuration.fixtures is not None:
                    report = configuration.fixtures[
                        fixture_name_for_request(target.path, request)
                    ]
                else:
                    report = collect_report(configuration.sources, request)
            except ValueError as error:
                status = 404 if "not found" in str(error).lower() else 400
                kind = "not_found" if status == 404 else "invalid_request"
                self._error(status, kind, str(error))
                return
            except Exception:
                print("report server: report collection failed", file=sys.stderr)
                self._error(500, "server_error", "report collection failed")
                return
            self._json(200, report)

        def _method_not_allowed(self):
            self._error(
                405, "method_not_allowed", "only GET is allowed", {"Allow": "GET"}
            )

        do_POST = _method_not_allowed
        do_PUT = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_PATCH = _method_not_allowed

    return ReportHandler


def build_configuration(queue_path, cache_path, fixture_directory=None):
    defaults = ReportSources.defaults()
    sources = ReportSources(
        queue_path=queue_path,
        cache_path=cache_path,
        answer_list_path=defaults.answer_list_path,
        candidate_list_path=defaults.candidate_list_path,
        telemetry_path=(
            defaults.telemetry_path if queue_path == defaults.queue_path else None
        ),
    )
    client_path = os.path.join(os.path.dirname(__file__), "report_client.html")
    fixtures = load_fixtures(fixture_directory) if fixture_directory else None
    return ServerConfiguration(
        sources, client_path, fixture_directory, fixtures
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", metavar="ADDRESS")
    parser.add_argument("--port", type=int, default=8765, metavar="PORT")
    parser.add_argument("--queue-path", default=DEFAULT_QUEUE_PATH, metavar="PATH")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, metavar="PATH")
    parser.add_argument("--fixture-directory", metavar="PATH")
    args = parser.parse_args()
    configuration = build_configuration(
        args.queue_path, args.cache_path, args.fixture_directory
    )
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(configuration))
    print(f"Serving ERD reports on http://{args.bind}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
