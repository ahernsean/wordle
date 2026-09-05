#!/usr/bin/env python3
"""Read-only HTTP adapter for shared ERD swarm reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import sys
from threading import Event, Lock
from urllib.parse import parse_qs, urlsplit

from report_model import (
    BRANCH_STATUSES,
    BRANCH_WORKER_STATUSES,
    OVERVIEW_BRANCH_STATUSES,
    OVERVIEW_BRANCH_WORKER_STATUSES,
    applied_branch_filters,
    is_overview_request,
    GROUP_BY_STRATEGIES,
    SCHEMA_VERSION,
    OPENER_GROUP_BY_STRATEGIES,
    OPENER_SORT_FIELDS,
    OPENER_STATES,
    TREE_CURSOR_PATTERN,
    ReportFilters,
    ReportRequest,
    ReportOpeners,
    collect_ambiguous_branch_reference_report,
    collect_report,
    parse_branch_filter,
    parse_report_branch_target,
    validate_report_request,
)
from runtime_paths import DEFAULT_CACHE_PATH, DEFAULT_QUEUE_PATH, ensure_runtime_dir


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
    "leaderboard.json",
    "root_progress.json",
    "root_progress-inherited.json",
    "openers.json",
    "openers-word.json",
)
BOOLEAN_PARAMETERS = {"tree", "claims", "answers", "inherited_cost"}
INTEGER_PARAMETERS = {
    "minimum_answer_count", "maximum_answer_count", "budget", "priority",
    "limit", "epoch", "since_seconds", "sample_size", "opener_offset",
    "branch_row_offset",
}
SCALAR_PARAMETERS = {
    "branch_target", "sort", "group_by", "by", "worker", "branch_status",
    "branch_worker_status", "opener_state", "finalization_cursor", "tree_parent",
    "tree_cursor",
    *BOOLEAN_PARAMETERS,
    *INTEGER_PARAMETERS,
}
FINALIZATION_CURSOR_PATTERN = re.compile(r"(after|before):(\d+):(\d+)")
ALLOWED_PARAMETERS = SCALAR_PARAMETERS
SORT_FIELDS = {
    "default", "age", "size", "workers", "priority", "nodes", "slowest",
    *OPENER_SORT_FIELDS,
}
HOTSPOT_FIELDS = {
    "nodes", "age", "size", "workers", "priority", "slowest",
    "evaluated-candidates", "bulk-completed-candidates",
    "one-level-erd-prunes", "two-level-erd-prunes", "cut-reuse", "coordination",
}
class InvalidRequest(ValueError):
    pass


class InFlightLeaderboardReport:
    """One leaderboard collection shared by concurrent identical requests."""

    def __init__(self):
        self.completed = Event()
        self.report = None
        self.error = None


@dataclass(frozen=True)
class ServerConfiguration:
    sources: ReportOpeners
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
        "/api/view/leaderboard": "leaderboard",
        "/api/view/openers": "openers",
        "/api/view/root-progress": "root_progress",
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
        branch_target = parse_report_branch_target(_single_value(parameters, "branch_target"))
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    tree = _boolean_value(parameters, "tree")
    include_claims = _boolean_value(parameters, "claims")
    include_answers = _boolean_value(parameters, "answers")
    inherited_cost = _boolean_value(parameters, "inherited_cost")
    tree_parent = _single_value(parameters, "tree_parent", "")
    tree_cursor = _single_value(parameters, "tree_cursor")
    if (tree_parent or tree_cursor) and not tree:
        raise InvalidRequest("tree_parent and tree_cursor require tree")
    if tree_parent:
        try:
            tree_parent_target = parse_report_branch_target(tree_parent)
        except ValueError as error:
            raise InvalidRequest(str(error)) from error
        if tree_parent_target.kind != "branch":
            raise InvalidRequest("tree_parent must be a complete spine")
        tree_parent = " ".join(
            value for step in tree_parent_target.steps
            for value in (step.word.upper(), step.pattern)
        )
    if tree_cursor is not None and not TREE_CURSOR_PATTERN.fullmatch(tree_cursor):
        raise InvalidRequest("tree_cursor must name a tree page group")
    branch_status_value = _single_value(parameters, "branch_status")
    branch_worker_status_value = _single_value(parameters, "branch_worker_status")
    opener_state_value = _single_value(parameters, "opener_state")
    try:
        overview = is_overview_request(explicit_kind, branch_target.kind, tree)
        branch_statuses = (
            parse_branch_filter(
                branch_status_value, "branch status", BRANCH_STATUSES
            )
            if branch_status_value is not None
            else (OVERVIEW_BRANCH_STATUSES if overview else ())
        )
        branch_worker_statuses = (
            parse_branch_filter(
                branch_worker_status_value, "branch worker status",
                BRANCH_WORKER_STATUSES,
            )
            if branch_worker_status_value is not None
            else (OVERVIEW_BRANCH_WORKER_STATUSES if overview else ())
        )
        opener_states = (
            parse_branch_filter(opener_state_value, "opener state", OPENER_STATES)
            if opener_state_value is not None else ()
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
    finalization_cursor_value = _single_value(parameters, "finalization_cursor")
    finalization_cursor_direction = None
    finalization_cursor_recorded_at = None
    finalization_cursor_id = None
    if finalization_cursor_value:
        match = FINALIZATION_CURSOR_PATTERN.fullmatch(finalization_cursor_value)
        if not match:
            raise InvalidRequest(
                "finalization_cursor must be '(after|before):<recorded_at>:<id>'"
            )
        finalization_cursor_direction = match.group(1)
        finalization_cursor_recorded_at = int(match.group(2))
        finalization_cursor_id = int(match.group(3))
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
    group_by = _single_value(parameters, "group_by")
    if group_by is not None and group_by not in (
        set(GROUP_BY_STRATEGIES) | set(OPENER_GROUP_BY_STRATEGIES)
    ):
        raise InvalidRequest(f"invalid group_by field {group_by!r}")

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
    if tree and explicit_kind in ("cache", "hotspots", "leaderboard", "openers"):
        raise InvalidRequest(f"tree cannot be used with {explicit_kind}")
    worker_id = _single_value(parameters, "worker")
    if worker_id is not None and explicit_kind != "workers":
        raise InvalidRequest("worker requires the workers endpoint")
    if include_claims and (
        explicit_kind != "auto"
        or branch_target.kind not in ("branch", "branch_reference")
    ):
        raise InvalidRequest("claims requires a singular branch target")
    if include_answers and (
        tree
        or explicit_kind in ("queue", "workers")
        or (explicit_kind == "auto" and branch_target.kind == "root")
    ):
        raise InvalidRequest("answers requires a word or branch report without tree")
    if explicit_kind == "hotspots" and limit is None:
        limit = 10

    filters = ReportFilters(
        branch_statuses=branch_statuses,
        branch_worker_statuses=branch_worker_statuses,
        minimum_answer_count=minimum_answer_count,
        maximum_answer_count=maximum_answer_count,
        budget=integer_values["budget"],
        priority=integer_values["priority"],
        sort=sort,
        group_by=group_by,
        opener_states=opener_states,
        opener_offset=integer_values["opener_offset"],
        branch_row_offset=integer_values["branch_row_offset"],
        limit=limit,
        finalization_cursor_direction=finalization_cursor_direction,
        finalization_cursor_recorded_at=finalization_cursor_recorded_at,
        finalization_cursor_id=finalization_cursor_id,
    )
    request = ReportRequest(
        report_kind=explicit_kind,
        branch_target=branch_target,
        include_claims=include_claims,
        include_answers=include_answers,
        tree=tree,
        filters=applied_branch_filters(filters),
        worker_id=worker_id,
        hotspot_field=hotspot_field,
        epoch=integer_values["epoch"],
        tree_parent=tree_parent,
        tree_cursor=tree_cursor,
        since_seconds=since_seconds or 3600,
        sample_size=min(sample_size or 50_000, 1_000_000),
        inherited_cost=inherited_cost,
    )
    try:
        validate_report_request(request)
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    return request


# A report that fills itself in progressively answers one request cheaply and
# a second one with the expensive part added.  Both shapes are real responses
# the client renders, so each needs its own fixture, named by suffixing the
# stage.  Add a report here when it grows a second stage; the client half is
# `attachProgressiveStage`.
PROGRESSIVE_STAGE_FLAGS = {"root_progress": (("inherited_cost", "inherited"),)}


def progressive_stage_suffix(request):
    """The fixture-name suffix naming the enrichments a request asked for."""
    return "".join(
        f"-{suffix}"
        for attribute, suffix in PROGRESSIVE_STAGE_FLAGS.get(request.report_kind, ())
        if getattr(request, attribute, False)
    )


def fixture_name_for_request(path, request):
    if path == "/api/view":
        if request.tree:
            return "tree.json"
        return {
            "root": "overview.json",
            "word": "word.json",
            "branch": "branch.json",
            "branch_reference": "branch.json",
        }[request.branch_target.kind]
    kind = request.report_kind
    if request.tree:
        return f"{kind}-tree.json"
    # The opener report lists a request's branches only once that request is
    # named, so the two shapes need two fixtures.
    if kind == "openers" and request.branch_target.kind == "word":
        return "openers-word.json"
    return f"{kind}{progressive_stage_suffix(request)}.json"


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
    leaderboard_reports = {}
    leaderboard_reports_lock = Lock()

    def collect_leaderboard_once(request):
        with leaderboard_reports_lock:
            in_flight = leaderboard_reports.get(request)
            if in_flight is None:
                in_flight = InFlightLeaderboardReport()
                leaderboard_reports[request] = in_flight
                is_builder = True
            else:
                is_builder = False
        if is_builder:
            try:
                in_flight.report = collect_report(configuration.sources, request)
            except Exception as error:
                in_flight.error = error
            finally:
                in_flight.completed.set()
                with leaderboard_reports_lock:
                    leaderboard_reports.pop(request, None)
        else:
            in_flight.completed.wait()
        if in_flight.error is not None:
            raise in_flight.error
        return in_flight.report

    class ReportHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def handle_error(self, request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

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
                elif request.report_kind == "leaderboard":
                    report = collect_leaderboard_once(request)
                else:
                    report = collect_report(configuration.sources, request)
            except ValueError as error:
                if hasattr(error, "candidates"):
                    report = collect_ambiguous_branch_reference_report(
                        configuration.sources, request, error
                    )
                    self._json(200, report)
                    return
                status = 404 if (
                    "not found" in str(error).lower()
                    or str(error).startswith("No queued or cached @")
                ) else 400
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
    defaults = ReportOpeners.defaults()
    sources = ReportOpeners(
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
    ensure_runtime_dir()
    configuration = build_configuration(
        args.queue_path, args.cache_path, args.fixture_directory
    )
    try:
        server = ThreadingHTTPServer(
            (args.bind, args.port), make_handler(configuration)
        )
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(
                f"Cannot start: {args.bind}:{args.port} is already in use by "
                f"another process. Only one report server may run at a time; "
                f"find and stop it before retrying.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    print(f"Serving ERD reports on http://{args.bind}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
