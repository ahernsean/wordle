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


@dataclass(frozen=True)
class TerminalColumn:
    heading: str
    value: object
    required: bool = False
    remove_priority: int = 0
    alignment: str = "left"
    minimum_width: int = 1
    maximum_width: int | None = None
    truncation: str | None = None


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


def _truncate_cell(value, width, mode):
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    if mode == "tail":
        return "…" + value[-(width - 1):]
    return value[:width - 1] + "…"


def _column_value(column, row):
    value = column.value(row) if callable(column.value) else row.get(column.value)
    return "—" if value is None or value == "" else str(value)


def _table_layout(columns, rows, available_width):
    visible_columns = list(columns)

    def measured_width(column):
        width = max(
            len(column.heading),
            column.minimum_width,
            *(_column_value(column, row).__len__() for row in rows),
        )
        if column.maximum_width is not None:
            width = min(width, column.maximum_width)
        return width

    column_widths = {column: measured_width(column) for column in columns}

    def table_width():
        return (
            sum(column_widths[column] for column in visible_columns)
            + 2 * max(0, len(visible_columns) - 1)
        )

    while table_width() > available_width:
        removable = [column for column in visible_columns if not column.required]
        if not removable:
            break
        visible_columns.remove(max(
            removable,
            key=lambda column: (column.remove_priority, columns.index(column)),
        ))

    while table_width() > available_width:
        shrinkable = [
            column for column in visible_columns
            if column.truncation is not None
            and column_widths[column] > max(column.minimum_width, len(column.heading))
        ]
        if not shrinkable:
            break
        column = max(
            shrinkable,
            key=lambda item: column_widths[item]
            - max(item.minimum_width, len(item.heading)),
        )
        column_widths[column] -= 1

    if table_width() > available_width:
        return None
    return visible_columns, column_widths


def _format_table_cell(value, column, width):
    value = _truncate_cell(value, width, column.truncation)
    if column.alignment == "right":
        return value.rjust(width)
    return value.ljust(width)


def _render_stacked_rows(columns, rows, width, indent, row_classes, color):
    lines = []
    for index, row in enumerate(rows):
        semantic_class = row_classes[index] if row_classes else None
        for column in columns:
            value = _column_value(column, row)
            prefix = f"{indent}{column.heading}: "
            available = width - len(prefix)
            if available >= len(value):
                line = prefix + value
            elif column.truncation is not None and available > 0:
                line = prefix + _truncate_cell(value, available, column.truncation)
            else:
                lines.append(_colorize(_fit(prefix.rstrip(), width), semantic_class, color))
                line = _fit(f"{indent}  {value}", width)
            lines.append(_colorize(line, semantic_class, color))
        if index + 1 < len(rows):
            lines.append("")
    return lines


def _render_table(
    columns, rows, width, *, indent="", row_classes=None,
    measurement_rows=None, include_header=True, color=False,
):
    measurement_rows = rows if measurement_rows is None else measurement_rows
    available_width = max(0, width - len(indent))
    layout = _table_layout(columns, measurement_rows, available_width)
    if layout is None:
        return _render_stacked_rows(
            columns, rows, width, indent, row_classes, color
        )
    visible_columns, column_widths = layout
    lines = []
    if include_header:
        lines.append(indent + "  ".join(
            _format_table_cell(column.heading, column, column_widths[column])
            for column in visible_columns
        ).rstrip())
    for index, row in enumerate(rows):
        line = indent + "  ".join(
            _format_table_cell(
                _column_value(column, row), column, column_widths[column]
            )
            for column in visible_columns
        ).rstrip()
        semantic_class = row_classes[index] if row_classes else None
        lines.append(_colorize(line, semantic_class, color))
    return lines


def _wrap_fields(fields, width, indent="  "):
    lines = []
    current = indent
    for field in fields:
        separator = "" if current == indent else "  "
        if len(current) + len(separator) + len(field) <= width:
            current += separator + field
            continue
        if current != indent:
            lines.append(current)
        current = indent + field
        if len(current) > width:
            lines.append(_fit(current, width))
            current = indent
    if current != indent:
        lines.append(current)
    return lines


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
    cache = ["Cache"] + _wrap_fields([
        f"exact {cache_summary['exact_branch_count']:,}",
        f"loss {cache_summary['loss_branch_count']:,}",
        f"recent {cache_summary['recent_exact_branch_count']:,}",
        f"live hit {_percentage(totals['cache_hit_count'], cache_attempts)}",
    ], width)

    queue_counts = report["data"]["queue_counts"]
    evaluation_count = (
        totals["solved_evaluation_count"]
        + totals["erd_cutoff_evaluation_count"]
        + totals["remaining_depth_pruned_evaluation_count"]
    )
    queue = ["Queue"] + _wrap_fields([
        f"pending {queue_counts['pending_branch_count']:,}",
        f"active user {queue_counts['active_user_branch_count']:,}",
        f"active coop {queue_counts['active_cooperative_branch_count']:,}",
        f"finalizing {queue_counts['finalizing_branch_count']:,}",
        f"done {queue_counts['done_branch_count']:,}",
    ], width) + _wrap_fields([
        f"ERD cutoff {_percentage(totals['erd_cutoff_evaluation_count'], evaluation_count)}",
        "remaining-depth pruned "
        f"{_percentage(totals['remaining_depth_pruned_evaluation_count'], evaluation_count)}",
    ], width)

    branch_lines = ["Active branches"]
    active_branch_keys = set()
    workers_by_branch = {}
    for worker in workers:
        workers_by_branch.setdefault(worker.get("branch_key_hex"), []).append(worker)
    branch_rows = []
    branch_classes = []
    branch_columns = [
        TerminalColumn("Ref", lambda row: "@" + row["branch_reference"], required=True),
        TerminalColumn("GuessD", "guess_depth", required=True, alignment="right"),
        TerminalColumn("State", "display_state", required=True),
        TerminalColumn(
            "Done", lambda row: (
                f"{row['completed_candidate_count']}/{row['candidate_count']}"
            ), required=True, alignment="right",
        ),
        TerminalColumn("W", "worker_count", required=True, alignment="right"),
        TerminalColumn(
            "Ans", "answer_count", remove_priority=10, alignment="right"
        ),
        TerminalColumn(
            "Bulk", "bulk_completed_candidate_count", remove_priority=20,
            alignment="right",
        ),
        TerminalColumn("Best", "display_best", remove_priority=30),
        TerminalColumn(
            "MaxRD", "best_max_remaining_depth", remove_priority=40,
            alignment="right",
        ),
        TerminalColumn("ETA", "display_eta", remove_priority=50, alignment="right"),
        TerminalColumn(
            "Spine", "display_spine", remove_priority=60,
            minimum_width=12, maximum_width=40, truncation="tail",
        ),
    ]
    for branch in branches:
        branch_key = branch["branch_key_hex"]
        if branch["lifecycle"] == "active":
            active_branch_keys.add(branch_key)
        best = "—"
        if branch.get("best_guess"):
            best = branch["best_guess"].upper()
            if branch.get("best_erd") is not None:
                best += f"/{branch['best_erd']:.3f}"
        branch_row = dict(branch)
        branch_row["display_state"] = {
            "finalizing": "final",
        }.get(branch["lifecycle"], branch["lifecycle"])
        branch_row["display_best"] = best
        branch_row["display_eta"] = _abbreviate_duration(
            _branch_eta(branch, generated_at)
        )
        branch_row["display_spine"] = " ▸ ".join(
            f"{step['word'].upper()} {step['pattern']}" for step in branch["spine"]
        ) or "—"
        branch_rows.append(branch_row)
        branch_classes.append(
            _semantic_branch_class(branch, previous_branches.get(branch_key))
        )

    header_rendered = False
    for index, branch in enumerate(branches):
        branch_lines.extend(_render_table(
            branch_columns, [branch_rows[index]], width, indent="  ",
            row_classes=[branch_classes[index]], measurement_rows=branch_rows,
            include_header=not header_rendered, color=color,
        ))
        header_rendered = True
        branch_key = branch["branch_key_hex"]
        for worker in workers_by_branch.get(branch_key, []):
            if worker["is_live"] and branch["lifecycle"] == "active":
                branch_workers = [
                    _worker_display_row(item, generated_at, "active")
                    for item in workers_by_branch.get(branch_key, [])
                    if item["is_live"] and branch["lifecycle"] == "active"
                ]
                branch_worker_classes = [
                    _semantic_worker_class(
                        item, previous_workers.get(item["worker_id"]), generated_at
                    )
                    for item in workers_by_branch.get(branch_key, [])
                    if item["is_live"] and branch["lifecycle"] == "active"
                ]
                branch_lines.extend(_render_worker_table(
                    branch_workers, branch_worker_classes, width,
                    indent="    ", color=color,
                ))
                break
    if len(branch_lines) == 1:
        branch_lines.append("  none")

    remaining_worker_lines = ["Other workers"]
    remaining_workers = []
    remaining_worker_classes = []
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
        remaining_workers.append(_worker_display_row(worker, generated_at, state))
        remaining_worker_classes.append(_semantic_worker_class(
            worker, previous_workers.get(worker["worker_id"]), generated_at
        ))
    if remaining_workers:
        remaining_worker_lines.extend(_render_worker_table(
            remaining_workers, remaining_worker_classes, width,
            indent="  ", color=color,
        ))
    else:
        remaining_worker_lines.append("  none")

    return [
        ("header", header),
        ("cache", cache),
        ("queue", queue),
        ("branches", branch_lines),
        ("workers", remaining_worker_lines),
    ]


def _worker_display_row(worker, generated_at, state):
    age = max(0, generated_at - worker["updated_at"])
    row = dict(worker)
    row["display_state"] = {
        "finalizing": "final",
    }.get(state or "active", state or "active")
    row["display_age"] = _abbreviate_duration(age)
    row["display_candidate"] = (worker.get("current_candidate") or "—").upper()
    row["display_rate"] = _abbreviate_number(worker.get("nodes_per_second"))
    return row


def _worker_columns():
    return [
        TerminalColumn("Worker", "worker_id", required=True),
        TerminalColumn("State", "display_state", required=True),
        TerminalColumn("Age", "display_age", required=True, alignment="right"),
        TerminalColumn("Current", "display_candidate", required=True),
        TerminalColumn(
            "MaxGD", "current_max_guess_depth", required=True, alignment="right"
        ),
        TerminalColumn(
            "Rate", "display_rate", remove_priority=10, alignment="right"
        ),
    ]


def _render_worker_table(rows, row_classes, width, *, indent, color):
    return _render_table(
        _worker_columns(), rows, width, indent=indent,
        row_classes=row_classes, color=color,
    )


def _render_worker_line(
    worker, previous_worker, generated_at, color, width, indent, state=None
):
    rows = [_worker_display_row(worker, generated_at, state or "active")]
    row_classes = [
        _semantic_worker_class(worker, previous_worker, generated_at)
    ]
    return _render_worker_table(
        rows, row_classes, width, indent=indent, color=color
    )[-1]


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
        return collect_report(self._sources(), ReportRequest())

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
            self.output_stream.write(render_overview(
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
                    self.output_stream.write(render_overview(
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
        return _render_sections(
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
