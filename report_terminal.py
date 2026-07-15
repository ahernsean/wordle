"""Terminal presentation and refresh sessions for shared swarm reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import select
import shutil
import sys
import termios
import time

from report_model import ReportRequest, ReportSources, collect_report


GREEN = "\033[32m"
RED = "\033[31m"
AMBER = "\033[33m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"
WORKER_STALE_SECONDS = 20


@dataclass
class DisplayOrder:
    branch_keys: list[str] = field(default_factory=list)
    worker_ids: list[str] = field(default_factory=list)

    @staticmethod
    def _update_identities(existing, incoming, pinned):
        incoming_set = set(incoming)
        retained = [
            identity for identity in existing
            if identity in incoming_set or identity in pinned
        ]
        retained_set = set(retained)
        retained.extend(
            identity for identity in incoming if identity not in retained_set
        )
        return retained

    def update(self, report, pinned_branch_keys=(), pinned_worker_ids=()):
        branch_keys = [branch["branch_key_hex"] for branch in report["data"]["branches"]]
        worker_ids = [worker["worker_id"] for worker in report["data"]["workers"]]
        self.branch_keys = self._update_identities(
            self.branch_keys, branch_keys, set(pinned_branch_keys)
        )
        self.worker_ids = self._update_identities(
            self.worker_ids, worker_ids, set(pinned_worker_ids)
        )

    def ordered_branches(self, branches):
        by_key = {branch["branch_key_hex"]: branch for branch in branches}
        return [by_key[key] for key in self.branch_keys if key in by_key]

    def ordered_workers(self, workers):
        by_id = {worker["worker_id"]: worker for worker in workers}
        return [by_id[worker_id] for worker_id in self.worker_ids if worker_id in by_id]


def _percentage(numerator, denominator):
    if not denominator:
        return "—"
    return f"{100.0 * numerator / denominator:.1f}%"


def _abbreviate_number(value):
    if value is None:
        return "—"
    absolute_value = abs(value)
    for threshold, suffix in ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k")):
        if absolute_value >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _abbreviate_duration(seconds):
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _branch_eta(branch, generated_at):
    completed = branch["completed_candidate_count"]
    created_at = branch.get("created_at")
    candidate_count = branch["candidate_count"]
    if not created_at or completed <= 0 or completed >= candidate_count:
        return None
    elapsed = max(0, generated_at - created_at)
    return elapsed * (candidate_count - completed) / completed


def _fit(line, width):
    if width is None or len(line) <= width:
        return line
    if width <= 1:
        return line[:width]
    return line[:width - 1] + "…"


def _semantic_branch_class(branch, previous_branch):
    if previous_branch is None:
        return "green"
    if (
        branch["completed_candidate_count"]
        > previous_branch["completed_candidate_count"]
        or (
            branch.get("best_erd") is not None
            and (
                previous_branch.get("best_erd") is None
                or branch["best_erd"] < previous_branch["best_erd"]
            )
        )
    ):
        return "green"
    return "red" if branch != previous_branch else None


def _semantic_worker_class(worker, previous_worker, generated_at):
    if not worker["is_live"]:
        return "red"
    if generated_at - worker["updated_at"] > WORKER_STALE_SECONDS:
        return "amber"
    if previous_worker is None:
        return "green"
    ignored = {"updated_at"}
    current_values = {key: value for key, value in worker.items() if key not in ignored}
    previous_values = {
        key: value for key, value in previous_worker.items() if key not in ignored
    }
    return "red" if current_values != previous_values else None


def _colorize(line, semantic_class, color):
    if not color or semantic_class is None:
        return line
    prefix = {"green": GREEN, "red": RED, "amber": AMBER}[semantic_class]
    return f"{prefix}{line}{RESET}"


def _source_line(name, source, width):
    health = "ok" if source["ok"] else f"unavailable: {source['error']}"
    if width >= 80:
        return _fit(f"  {name}: {health}  path={source['path']}", width)
    return _fit(f"  {name}: {health}", width)


def _render_sections(report, previous_report, color, width, display_order):
    display_order.update(report)
    previous_data = previous_report["data"] if previous_report else {
        "branches": [], "workers": [],
    }
    previous_branches = {
        branch["branch_key_hex"]: branch for branch in previous_data["branches"]
    }
    previous_workers = {
        worker["worker_id"]: worker for worker in previous_data["workers"]
    }
    branches = display_order.ordered_branches(report["data"]["branches"])
    workers = display_order.ordered_workers(report["data"]["workers"])

    generated_at = report["generated_at"]
    generated_text = datetime.fromtimestamp(generated_at).strftime("%Y-%m-%d %H:%M:%S")
    queue_source = report["sources"]["queue"]
    header = [_fit(f"ERD swarm overview  {generated_text}  ({generated_at})", width)]
    for source_name in ("queue", "telemetry", "cache"):
        header.append(_source_line(source_name, report["sources"][source_name], width))
    if queue_source.get("epoch") is not None:
        epoch_detail = f"epoch={queue_source['epoch']}"
        if queue_source.get("label"):
            epoch_detail += f" {queue_source['label']}"
        if queue_source.get("git_sha"):
            epoch_detail += f" revision={queue_source['git_sha']}"
        header.append(_fit(f"  {epoch_detail}", width))

    cache_summary = report["data"]["cache_summary"]
    totals = report["data"]["worker_totals"]
    cache_attempts = totals["cache_hit_count"] + totals["cache_miss_count"]
    cache = [
        "Cache",
        _fit(
            f"  exact {cache_summary['exact_branch_count']:,}  "
            f"loss {cache_summary['loss_branch_count']:,}  "
            f"recent {cache_summary['recent_exact_branch_count']:,}  "
            f"live hit rate {_percentage(totals['cache_hit_count'], cache_attempts)}",
            width,
        ),
    ]

    queue_counts = report["data"]["queue_counts"]
    evaluation_count = (
        totals["solved_evaluation_count"]
        + totals["erd_cutoff_evaluation_count"]
        + totals["remaining_depth_pruned_evaluation_count"]
    )
    queue = [
        "Queue",
        _fit(
            f"  pending {queue_counts['pending_branch_count']:,}  "
            f"active {queue_counts['active_user_branch_count']:,} user + "
            f"{queue_counts['active_cooperative_branch_count']:,} cooperative  "
            f"finalizing {queue_counts['finalizing_branch_count']:,}  "
            f"done {queue_counts['done_branch_count']:,}",
            width,
        ),
        _fit(
            f"  ERD cutoff {_percentage(totals['erd_cutoff_evaluation_count'], evaluation_count)}  "
            f"remaining-depth pruned "
            f"{_percentage(totals['remaining_depth_pruned_evaluation_count'], evaluation_count)}",
            width,
        ),
    ]

    branch_lines = ["Active branches"]
    active_branch_keys = set()
    workers_by_branch = {}
    for worker in workers:
        workers_by_branch.setdefault(worker.get("branch_key_hex"), []).append(worker)
    for branch in branches:
        branch_key = branch["branch_key_hex"]
        if branch["lifecycle"] == "active":
            active_branch_keys.add(branch_key)
        completed = branch["completed_candidate_count"]
        best = "—"
        if branch.get("best_guess"):
            best = branch["best_guess"].upper()
            if branch.get("best_erd") is not None:
                best += f"/{branch['best_erd']:.3f}"
        if width < 60:
            line = (
                f"  @{branch['branch_reference']}  "
                f"guess_depth={branch['guess_depth']}  "
                f"{completed}/{branch['candidate_count']}"
            )
        else:
            line = (
                f"  @{branch['branch_reference']}  n={branch['answer_count']} "
                f"guess_depth={branch['guess_depth']}  "
                f"{completed}/{branch['candidate_count']}"
            )
            if branch.get("best_max_remaining_depth") is not None:
                line += (
                    "  max_remaining_depth="
                    f"{branch['best_max_remaining_depth']}"
                )
            if branch["bulk_completed_candidate_count"]:
                line += f" bulk={branch['bulk_completed_candidate_count']}"
            if width >= 100:
                line += f"  best={best}  workers={branch['worker_count']}"
        if width >= 80:
            line += f"  ETA={_abbreviate_duration(_branch_eta(branch, generated_at))}"
            if branch["spine"]:
                spine = " ▸ ".join(
                    f"{step['word'].upper()} {step['pattern']}" for step in branch["spine"]
                )
                line += f"  {spine}"
        branch_lines.append(_colorize(
            _fit(line, width),
            _semantic_branch_class(branch, previous_branches.get(branch_key)),
            color,
        ))
        for worker in workers_by_branch.get(branch_key, []):
            if worker["is_live"] and branch["lifecycle"] == "active":
                branch_lines.append(_render_worker_line(
                    worker, previous_workers.get(worker["worker_id"]), generated_at,
                    color, width, indent="    ",
                ))
    if len(branch_lines) == 1:
        branch_lines.append("  none")

    remaining_worker_lines = ["Other workers"]
    for worker in workers:
        branch_key = worker.get("branch_key_hex")
        if worker["is_live"] and branch_key in active_branch_keys:
            continue
        if not worker["is_live"]:
            state = "dead"
        elif branch_key is None:
            state = "idle"
        else:
            state = "finalizing"
        remaining_worker_lines.append(_render_worker_line(
            worker, previous_workers.get(worker["worker_id"]), generated_at,
            color, width, indent="  ", state=state,
        ))
    if len(remaining_worker_lines) == 1:
        remaining_worker_lines.append("  none")

    return [
        ("header", header),
        ("cache", cache),
        ("queue", queue),
        ("branches", branch_lines),
        ("workers", remaining_worker_lines),
    ]


def _render_worker_line(
    worker, previous_worker, generated_at, color, width, indent, state=None
):
    age = max(0, generated_at - worker["updated_at"])
    label = f"worker={worker['worker_id']}"
    if state:
        label += f" {state}"
    line = f"{indent}{label}  age={_abbreviate_duration(age)}"
    if width >= 60:
        candidate = (worker.get("current_candidate") or "—").upper()
        line += f"  candidate={candidate}"
        line += f"  nodes/s={_abbreviate_number(worker.get('nodes_per_second'))}"
        if worker.get("current_max_guess_depth") is not None:
            line += (
                "  current_max_guess_depth="
                f"{worker['current_max_guess_depth']}"
            )
    semantic_class = _semantic_worker_class(
        worker, previous_worker, generated_at
    )
    return _colorize(_fit(line, width), semantic_class, color)


def render_overview(
    report: dict,
    previous_report: dict | None = None,
    *,
    color: bool = False,
    width: int | None = None,
    display_order: DisplayOrder | None = None,
) -> str:
    width = 80 if width is None else width
    display_order = display_order or DisplayOrder()
    sections = _render_sections(
        report, previous_report, color, width, display_order
    )
    return "\n\n".join("\n".join(lines) for _name, lines in sections)


def _semantic_header(report, title, width):
    header = [_fit(f"{title}  generated_at={report['generated_at']}", width)]
    for source_name in ("queue", "telemetry", "cache"):
        header.append(_source_line(source_name, report["sources"][source_name], width))
    return header


def _render_word_sections(report, previous_report, color, width, display_order):
    data = report["data"]
    response_groups = data["response_groups"]
    incoming_keys = [row["branch_key_hex"] for row in response_groups]
    display_order.branch_keys = DisplayOrder._update_identities(
        display_order.branch_keys, incoming_keys, set()
    )
    by_key = {row["branch_key_hex"]: row for row in response_groups}
    ordered_groups = [
        by_key[key] for key in display_order.branch_keys if key in by_key
    ]
    previous_groups = {
        row["branch_key_hex"]: row
        for row in (previous_report or {}).get("data", {}).get("response_groups", [])
    }
    context = data["context"]
    header = _semantic_header(
        report,
        f"Word {data['word'].upper()}  context=@{context['branch_reference']} "
        f"n={context['answer_count']} d={context['guess_depth']}",
        width,
    )
    counts = data["response_group_counts"]
    summary = [
        "Response groups",
        _fit(
            f"  total {counts['response_group_count']}  "
            f"trivial {counts['trivial_response_group_count']}  "
            f"queued {counts['queued_response_group_count']}  "
            f"active {counts['active_response_group_count']}  "
            f"exact {counts['exact_response_group_count']}  "
            f"loss {counts['loss_response_group_count']}  "
            f"missing {counts['missing_response_group_count']}",
            width,
        ),
    ]
    rows = ["Pattern  Answers  Queue       Cache           Best        Reference"]
    for group in ordered_groups:
        best = "—"
        if group.get("best_guess"):
            best = group["best_guess"].upper()
            if group.get("best_erd") is not None:
                best += f"/{group['best_erd']:.3f}"
        line = (
            f"{group['pattern']:<7}  {group['answer_count']:>7}  "
            f"{group['lifecycle']:<10}  {group['cache_state']:<14}  "
            f"{best:<10}  @{group['branch_reference']}"
        )
        previous_group = previous_groups.get(group["branch_key_hex"])
        semantic_class = (
            "green" if previous_group is None
            else ("red" if group != previous_group else None)
        )
        rows.append(_colorize(_fit(line, width), semantic_class, color))
        if group.get("answer_words"):
            rows.append(_fit("  " + " ".join(group["answer_words"]), width))
    if len(rows) == 1:
        rows.append("none")
    return [("header", header), ("summary", summary), ("response_groups", rows)]


def _render_branch_sections(report, previous_report, color, width, display_order):
    data = report["data"]
    branch = data["branch"]
    header = _semantic_header(
        report,
        f"Branch @{branch['branch_reference']}  n={branch['answer_count']} "
        f"d={branch['guess_depth']} budget={branch['budget']}",
        width,
    )
    if branch["spine"]:
        header.append(_fit(
            "  " + " ▸ ".join(
                f"{step['word'].upper()} {step['pattern']}" for step in branch["spine"]
            ),
            width,
        ))
    if branch.get("answer_words"):
        header.append(_fit("  answers: " + " ".join(branch["answer_words"]), width))
    queue = data["queue"]
    if queue is None:
        queue_lines = ["Queue", "  unqueued"]
    else:
        progress = "—"
        if queue["candidate_count"] is not None:
            completed = queue["completed_candidate_count"]
            progress = f"{completed}/{queue['candidate_count']}"
        queue_lines = [
            "Queue",
            _fit(
                f"  {queue['lifecycle']}  priority={queue['priority']}  "
                f"progress={progress}  bulk={queue['bulk_completed_candidate_count']}  "
                f"best={queue['best_guess'] or '—'}  nodes={_abbreviate_number(queue['search_node_count'])}",
                width,
            ),
        ]
    cache = data["cache"]
    cache_line = f"  {cache['cache_state']}"
    if cache.get("best_guess"):
        cache_line += f"  best={cache['best_guess'].upper()}/{cache['best_erd']:.3f}"
    if cache.get("max_remaining_depth") is not None:
        cache_line += f"  max-d={cache['max_remaining_depth']}"
    cache_lines = ["Cache", _fit(cache_line, width)]

    incoming_worker_ids = [worker["worker_id"] for worker in data["workers"]]
    display_order.worker_ids = DisplayOrder._update_identities(
        display_order.worker_ids, incoming_worker_ids, set()
    )
    workers_by_id = {worker["worker_id"]: worker for worker in data["workers"]}
    previous_workers = {
        worker["worker_id"]: worker
        for worker in (previous_report or {}).get("data", {}).get("workers", [])
    }
    worker_lines = ["Workers"]
    for worker_id in display_order.worker_ids:
        worker = workers_by_id.get(worker_id)
        if worker is not None:
            worker_lines.append(_render_worker_line(
                worker, previous_workers.get(worker_id), report["generated_at"],
                color, width, indent="  ",
                state=None if worker["is_live"] else "dead",
            ))
    if len(worker_lines) == 1:
        worker_lines.append("  none")

    detail_lines = ["Candidate state"]
    detail_lines.append(
        f"  republished candidates: {len(data['republished_candidates'])}"
    )
    if data["claims"] is not None:
        previous_claims = {
            claim["candidate_index"]: claim
            for claim in (previous_report or {}).get("data", {}).get("claims", []) or []
        }
        for claim in sorted(data["claims"], key=lambda row: row["candidate_index"]):
            previous_claim = previous_claims.get(claim["candidate_index"])
            semantic_class = (
                "green" if previous_claim is None
                else ("red" if claim != previous_claim else None)
            )
            detail_lines.append(_colorize(_fit(
                f"  idx={claim['candidate_index']} {claim['state']} "
                f"kind={claim['completion_kind'] or 'unknown'} "
                f"worker={claim['worker_id'] or '—'} "
                f"republished={claim['republish_count']}",
                width,
            ), semantic_class, color))
    return [
        ("header", header), ("queue", queue_lines), ("cache", cache_lines),
        ("workers", worker_lines), ("candidate_state", detail_lines),
    ]


def _report_sections(report, previous_report, color, width, display_order):
    if report["report_kind"] == "overview":
        return _render_sections(
            report, previous_report, color, width, display_order
        )
    if report["report_kind"] == "word":
        return _render_word_sections(
            report, previous_report, color, width, display_order
        )
    if report["report_kind"] == "branch":
        return _render_branch_sections(
            report, previous_report, color, width, display_order
        )
    raise ValueError(f"unsupported report kind: {report['report_kind']}")


def render_report(
    report, previous_report=None, *, color=False, width=None, display_order=None
):
    width = 80 if width is None else width
    display_order = display_order or DisplayOrder()
    sections = _report_sections(
        report, previous_report, color, width, display_order
    )
    return "\n\n".join("\n".join(lines) for _name, lines in sections)


class WatchSession:
    def __init__(self, args, input_stream=None, output_stream=None, error_stream=None):
        self.args = args
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.error_stream = error_stream or sys.stderr
        self.previous_report = None
        self.display_order = DisplayOrder()
        self.selected_branch_key = None
        self.selected_worker_id = None
        self.previous_sections = []
        self.terminal_settings = None
        self.cursor_hidden = False

    def _sources(self):
        defaults = ReportSources.defaults()
        return ReportSources(
            queue_path=self.args.queue_path,
            cache_path=self.args.cache_path,
            answer_list_path=defaults.answer_list_path,
            candidate_list_path=defaults.candidate_list_path,
            telemetry_path=(
                defaults.telemetry_path
                if self.args.queue_path == defaults.queue_path else None
            ),
        )

    def _collect(self):
        selector = getattr(self.args, "selector", None)
        request = ReportRequest(
            selector=selector if selector is not None else ReportRequest().selector,
            include_claims=getattr(self.args, "claims", False),
            include_answers=getattr(self.args, "answers", False),
        )
        return collect_report(self._sources(), request)

    def _width(self):
        return shutil.get_terminal_size((80, 24)).columns

    def _color(self):
        return (
            self.args.format == "text"
            and self.args.watch is not None
            and self.input_stream.isatty()
            and self.output_stream.isatty()
            and not self.args.no_color
        )

    def run(self):
        if self.args.watch is None:
            self._run_once()
        elif self.args.format == "jsonl":
            self._run_jsonl_watch()
        elif self.input_stream.isatty() and self.output_stream.isatty():
            self._run_tty_text()
        else:
            self._run_non_tty_text()

    def _run_once(self):
        try:
            report = self._collect()
        except Exception as error:
            self.error_stream.write(f"view: {error}\n")
            raise SystemExit(1)
        if self.args.format == "jsonl":
            self.output_stream.write(
                json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
            )
        elif self.args.format == "json":
            self.output_stream.write(json.dumps(report, sort_keys=True) + "\n")
        else:
            self.output_stream.write(render_report(
                report, color=False, width=self._width(),
                display_order=self.display_order,
            ) + "\n")
        self.output_stream.flush()

    def _run_jsonl_watch(self):
        try:
            while True:
                try:
                    report = self._collect()
                    self.output_stream.write(
                        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    self.output_stream.flush()
                except Exception as error:
                    self.error_stream.write(f"view: {error}\n")
                    self.error_stream.flush()
                time.sleep(self.args.watch)
        except KeyboardInterrupt:
            return

    def _run_non_tty_text(self):
        try:
            while True:
                try:
                    report = self._collect()
                    self.output_stream.write(
                        f"--- generated_at={report['generated_at']} ---\n"
                    )
                    self.output_stream.write(render_report(
                        report, previous_report=self.previous_report,
                        color=False, width=self._width(),
                        display_order=self.display_order,
                    ) + "\n")
                    self.previous_report = report
                except Exception as error:
                    self.output_stream.write(f"--- error ---\nview: {error}\n")
                self.output_stream.flush()
                time.sleep(self.args.watch)
        except KeyboardInterrupt:
            return

    def _terminal_sections(self, report):
        return _report_sections(
            report, self.previous_report, self._color(), self._width(),
            self.display_order,
        )

    def _write_initial_sections(self, sections):
        text = "\n\n".join("\n".join(lines) for _name, lines in sections)
        self.output_stream.write(text + CLEAR_LINE)
        self.output_stream.flush()

    @staticmethod
    def _section_start_lines(sections):
        starts = []
        line = 1
        for _name, lines in sections:
            starts.append(line)
            line += len(lines) + 1
        return starts

    def _refresh_sections(self, sections):
        old_by_name = dict(self.previous_sections)
        old_starts = self._section_start_lines(self.previous_sections)
        new_starts = self._section_start_lines(sections)
        first_shift = len(sections)
        for index, (name, lines) in enumerate(sections):
            old_lines = old_by_name.get(name, [])
            if len(lines) != len(old_lines):
                first_shift = min(first_shift, index)
        for index, (name, lines) in enumerate(sections):
            old_lines = old_by_name.get(name)
            if old_lines == lines and index < first_shift:
                continue
            start_line = new_starts[index]
            if index > 0:
                self.output_stream.write(
                    f"\033[{start_line - 1};1H{CLEAR_LINE}"
                )
            self.output_stream.write(f"\033[{start_line};1H")
            for line in lines:
                self.output_stream.write(line + CLEAR_LINE + "\n")
            if old_lines is not None and len(old_lines) > len(lines):
                for _ in range(len(old_lines) - len(lines)):
                    self.output_stream.write(CLEAR_LINE + "\n")
        if first_shift < len(sections):
            final_line = new_starts[-1] + len(sections[-1][1])
            self.output_stream.write(f"\033[{final_line};1H\033[J")
        self.output_stream.flush()

    def _wait_for_refresh(self):
        deadline = time.monotonic() + self.args.watch
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            ready, _, _ = select.select(
                [self.input_stream], [], [], min(remaining, 0.2)
            )
            if not ready:
                continue
            character = self.input_stream.read(1)
            if character in ("", "q", "Q", "\x04"):
                return False
            if character == " ":
                return True

    def _run_tty_text(self):
        file_descriptor = self.input_stream.fileno()
        self.terminal_settings = termios.tcgetattr(file_descriptor)
        new_settings = termios.tcgetattr(file_descriptor)
        new_settings[3] &= ~(termios.ICANON | termios.ECHO)
        try:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, new_settings)
            self.output_stream.write("\033[?25l\033[2J\033[H")
            self.output_stream.flush()
            self.cursor_hidden = True
            while True:
                try:
                    report = self._collect()
                    sections = self._terminal_sections(report)
                except Exception as error:
                    sections = [("error", ["Error", f"  view: {error}"])]
                    report = None
                if not self.previous_sections:
                    self._write_initial_sections(sections)
                else:
                    self._refresh_sections(sections)
                self.previous_sections = sections
                if report is not None:
                    self.previous_report = report
                if not self._wait_for_refresh():
                    return
        except KeyboardInterrupt:
            return
        finally:
            if self.cursor_hidden:
                self.output_stream.write("\033[?25h")
                self.output_stream.flush()
                self.cursor_hidden = False
            termios.tcsetattr(
                file_descriptor, termios.TCSADRAIN, self.terminal_settings
            )


def run_view(args) -> None:
    WatchSession(args).run()
