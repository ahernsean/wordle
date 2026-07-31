"""Terminal presentation and refresh sessions for shared swarm reports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import select
import shutil
import sys
import termios
import time

from report_model import (
    ReportRequest,
    ReportSources,
    WORKER_STALE_SECONDS,
    collect_report,
    parse_report_branch_target,
)
from wordle_engine import erd_display_numerator


GREEN = "\033[32m"
RED = "\033[31m"
AMBER = "\033[33m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"
BRANCH_HOTKEYS = "abcdefghijklmnoprstuvwxyz"


@dataclass
class DisplayOrder:
    branch_keys: list[str] = field(default_factory=list)
    worker_ids: list[str] = field(default_factory=list)
    hotkey_letters: dict[str, str] = field(default_factory=dict)

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
    # Semantic cell highlighting: called with (row, previous_row) — the
    # previous row is None on a first sighting — and returns a semantic class
    # or None.  Rules read row fields, so thresholds and comparisons stay
    # per-quantity instead of falling back to character diffs.
    highlight_rule: object = None


def _count_increase_rule(field):
    def rule(row, previous_row):
        if previous_row is None:
            return None
        current = row.get(field) or 0
        previous = previous_row.get(field) or 0
        return "green" if current > previous else None
    return rule


def _best_erd_improvement_rule(row, previous_row):
    if previous_row is None or row.get("best_erd") is None:
        return None
    previous_erd = previous_row.get("best_erd")
    if previous_erd is None or row["best_erd"] < previous_erd:
        return "green"
    return None


def _candidate_advance_rule(row, previous_row):
    if previous_row is None:
        return None
    if row.get("current_candidate") != previous_row.get("current_candidate"):
        return "green"
    return None


def _rate_stall_rule(row, _previous_row):
    # A live worker holding search nodes with a zero evaluation rate is
    # stalled, not merely idle.
    if (
        row.get("is_live")
        and (row.get("current_node_count") or 0) > 0
        and not (row.get("nodes_per_second") or 0)
    ):
        return "red"
    return None


def _percentage(numerator, denominator):
    if not denominator:
        return "—"
    return f"{100.0 * numerator / denominator:.1f}%"


# ERD and its bounds are means/ceilings; three decimals is the precision worth
# showing, never a raw float like 2.793103449275866.
BOUND_FIELDS = frozenset(
    {"best_erd", "erd", "ceiling", "bound_erd", "available_bound", "wanted_ceiling"}
)


def _format_metric_value(key, value):
    if key in BOUND_FIELDS and isinstance(value, float):
        return f"{value:.3f}"
    return value


def _format_branch_erd(value, answer_count, *, ceiling=False):
    """Render a branch ERD with its unreduced answer-count denominator."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    numerator = erd_display_numerator(value, answer_count, ceiling=ceiling)
    if numerator is not None:
        text += f" {numerator}/{answer_count}"
    return text


def _display_reference(branch_reference):
    """Four-character display prefix of a branch reference.

    Collisions among the few dozen simultaneously live branches are
    vanishingly unlikely over a 16-bit space, and any prefix remains a valid
    @reference branch_target.
    """
    return branch_reference[:4]


def _hotkey_label(display_order, branch_key_hex):
    letter = display_order.hotkey_letters.get(branch_key_hex)
    return f"[{letter}]" if letter else ""


def _worker_number_label(worker_id):
    if not worker_id:
        return "—"
    number = worker_id.rsplit("-", 1)[-1]
    return f"w{number}" if number.isdigit() else worker_id


def candidate_sweep_bar(candidate_count, completed_candidate_indexes,
                        worker_positions, width=40):
    """Render compressed candidate completion and worker positions.

    Each cell covers a range of candidate indices.  Unicode block heights show
    how much of the cell is done; worker labels overlay that conceptual
    completion map at their current claim positions.
    """
    if candidate_count <= 0 or width <= 0:
        return ""
    done_counts = [0] * width
    seen_indexes = set()
    for index in completed_candidate_indexes or ():
        if index is None or index < 0 or index in seen_indexes:
            continue
        seen_indexes.add(index)
        position = min(width - 1, int(width * index / candidate_count))
        done_counts[position] += 1
    ramp = " ▁▂▃▄▅▆▇█"
    marks = [" "] * width
    for position, done_count in enumerate(done_counts):
        if done_count == 0:
            continue
        start = (position * candidate_count + width - 1) // width
        end = ((position + 1) * candidate_count + width - 1) // width
        cell_size = end - start
        # The full block is reserved for a cell whose every candidate is done
        # — no worker will ever claim inside it again.  Partial completion
        # maps onto the intermediate blocks only, and any progress at all
        # lifts the cell off the baseline.
        if done_count >= cell_size:
            level = len(ramp) - 1
        else:
            intermediate_levels = len(ramp) - 2
            level = min(
                intermediate_levels,
                (done_count * intermediate_levels + cell_size - 1) // cell_size,
            )
        marks[position] = ramp[level]
    for index, label in worker_positions or ():
        if index is None or index < 0:
            continue
        position = min(width - 1, int(width * index / candidate_count))
        # Preserve multiple adjacent worker digits where possible, while still
        # allowing a worker to replace a progress marker in its approximate
        # bucket.
        while position < width and marks[position].isdigit():
            position += 1
        if position >= width:
            position = width - 1
        marks[position] = str(label)[-1]
    return "".join(marks)


def _highlight_changes(new_line, old_line):
    """Return new_line with runs of changed characters highlighted in red."""
    if new_line == old_line:
        return new_line
    result = []
    in_change = False
    for index, character in enumerate(new_line):
        changed = index >= len(old_line) or character != old_line[index]
        if changed and not in_change:
            result.append(RED)
            in_change = True
        elif not changed and in_change:
            result.append(RESET)
            in_change = False
        result.append(character)
    if in_change:
        result.append(RESET)
    return "".join(result)


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


DISK_RATE_FLOOR_BYTES = 10_000


def format_disk_size(byte_count):
    """Format byte counts with adaptive binary units."""
    gibibytes = byte_count / 2 ** 30
    if gibibytes >= 100:
        return f"{gibibytes:,.0f}G"
    if gibibytes >= 1:
        return f"{gibibytes:.1f}G"
    mebibytes = byte_count / 2 ** 20
    if mebibytes >= 1:
        return f"{mebibytes:,.0f}M"
    return f"{byte_count / 2 ** 10:,.0f}K"


def _format_fill_eta(seconds):
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def render_disk_status(disk, *, color=False):
    """Render filesystem fullness, queue WAL size, and a fresh fill trend."""
    if disk.get("used_fraction") is None:
        return "Disk: unavailable"
    capacity = disk["total_bytes"]
    fullness = (
        f"{format_disk_size(disk['used_bytes'])}/{format_disk_size(capacity)} "
        f"({100 * disk['used_fraction']:.0f}%)"
    )
    if color and disk["used_fraction"] >= disk["warning_fraction"]:
        fullness = f"{RED}{fullness}{RESET}"
    line = (
        f"Disk: {fullness}  queue WAL "
        f"{format_disk_size(disk.get('queue_wal_bytes') or 0)}"
    )
    rate = disk.get("fill_rate_bytes_per_second")
    if rate is not None:
        if rate > DISK_RATE_FLOOR_BYTES:
            available_at_stop = (1 - disk["stop_fraction"]) * capacity
            eta = max(
                disk["available_bytes"] - available_at_stop, 0
            ) / rate
            line += (
                f"  filling {format_disk_size(rate)}/s: "
                f"{100 * disk['stop_fraction']:.0f}% in ~{_format_fill_eta(eta)}"
            )
        elif rate < -DISK_RATE_FLOOR_BYTES:
            line += f"  freeing {format_disk_size(-rate)}/s"
        else:
            line += "  steady"
    return line


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


def _table_layout(columns, rows, available_width, *, include_heading_widths=True):
    visible_columns = list(columns)

    def measured_width(column):
        width = max(
            len(column.heading) if include_heading_widths else 0,
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
    columns, rows, width, *, indent="", row_classes=None, previous_rows=None,
    measurement_rows=None, include_header=True, measure_headings=None,
    color=False,
):
    measurement_rows = rows if measurement_rows is None else measurement_rows
    available_width = max(0, width - len(indent))
    layout = _table_layout(
        columns, measurement_rows, available_width,
        include_heading_widths=(
            include_header if measure_headings is None else measure_headings
        ),
    )
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
        semantic_class = row_classes[index] if row_classes else None
        previous_row = previous_rows[index] if previous_rows else None
        cells = []
        for column in visible_columns:
            cell = _format_table_cell(
                _column_value(column, row), column, column_widths[column]
            )
            if color and semantic_class is None and column.highlight_rule:
                cell = _colorize(
                    cell, column.highlight_rule(row, previous_row), color
                )
            cells.append(cell)
        line = (indent + "  ".join(cells)).rstrip()
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
    return "green" if previous_branch is None else None


def _semantic_worker_class(worker, previous_worker, generated_at):
    if not worker["is_live"]:
        return "red"
    if generated_at - worker["updated_at"] > WORKER_STALE_SECONDS:
        return "amber"
    return "green" if previous_worker is None else None


def _worker_state(worker):
    """The model-computed worker state (report_model.worker_state), with a
    fallback for any report that predates or omits it."""
    return worker.get("state") or "working"


def _colorize(line, semantic_class, color):
    if not color:
        return line
    prefix = {"green": GREEN, "red": RED, "amber": AMBER}.get(semantic_class)
    if prefix is None:
        return line
    return f"{prefix}{line}{RESET}"


def _source_problem_lines(report, width):
    lines = []
    for name in ("queue", "telemetry", "cache"):
        source = report["sources"][name]
        if not source["ok"] and source["error"] is not None:
            lines.append(_fit(
                f"  {name} unavailable: {source['error']}  path={source['path']}",
                width,
            ))
    return lines


def _inline_section(label, fields, width):
    """One wrapped section line: 'Label: field  field  …' with continuations
    indented two spaces."""
    lines = []
    current = label
    at_line_start = True
    for field_text in fields:
        separator = " " if at_line_start else "  "
        candidate = current + separator + field_text
        if len(candidate) > width and not at_line_start:
            lines.append(current)
            current = "  " + field_text
        else:
            current = candidate
        at_line_start = False
    lines.append(current)
    return [_fit(line, width) for line in lines]


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
    previous_generated_at = (
        previous_report["generated_at"] if previous_report else generated_at
    )
    queue_source = report["sources"]["queue"]
    header = [_fit(f"ERD swarm overview  {generated_text}", width)]
    source_fields = []
    problem_lines = _source_problem_lines(report, width)
    if not problem_lines:
        source_fields.append("sources ok")
    if queue_source.get("epoch") is not None:
        epoch_detail = f"epoch={queue_source['epoch']}"
        if queue_source.get("label"):
            epoch_detail += f" {queue_source['label']}"
        if queue_source.get("git_sha"):
            epoch_detail += f" revision={queue_source['git_sha']}"
        source_fields.append(epoch_detail)
    header.extend(problem_lines)
    if source_fields:
        header.extend(_wrap_fields(source_fields, width))
    header.append(_fit(
        render_disk_status(report["data"].get("disk", {}), color=color), width
    ))

    cache_summary = report["data"]["cache_summary"]
    totals = report["data"]["worker_totals"]
    cache_attempts = totals["cache_hit_count"] + totals["cache_miss_count"]
    evaluation_count = (
        totals["solved_evaluation_count"]
        + totals["erd_cutoff_evaluation_count"]
        + totals["remaining_depth_pruned_evaluation_count"]
    )
    queue_counts = report["data"]["queue_counts"]
    summary = _inline_section("Cache:", [
        f"exact {cache_summary['exact_branch_count']:,}",
        f"loss {cache_summary['loss_branch_count']:,}",
        f"recent {cache_summary['recent_exact_branch_count']:,}",
        f"live hit {_percentage(totals['cache_hit_count'], cache_attempts)}",
    ], width) + _inline_section("Queue:", [
        f"pending {queue_counts['pending_branch_count']:,}",
        f"user {queue_counts['evaluating_user_branch_count']:,}",
        f"coop {queue_counts['evaluating_cooperative_branch_count']:,}",
        f"finalizing {queue_counts['finalizing_branch_count']:,}",
        f"done {queue_counts['done_branch_count']:,}",
        f"ERD cutoff {_percentage(totals['erd_cutoff_evaluation_count'], evaluation_count)}",
        "remaining-depth pruned "
        f"{_percentage(totals['remaining_depth_pruned_evaluation_count'], evaluation_count)}",
    ], width)

    selected_statuses = report.get("filters", {}).get("branch_statuses") or []
    status_label = ",".join(selected_statuses) if selected_statuses else "all"
    branch_lines = [f"Branches (status={status_label})"]
    # Which active branches are shown here decides the layout: a worker on one
    # is drawn under its branch, everyone else falls to "Other workers".  This
    # is a display concern (it tracks the filtered branch set), independent of
    # each worker's state, which the model computes and both clients render.
    active_branch_keys = set()
    workers_by_branch = {}
    for worker in workers:
        workers_by_branch.setdefault(worker.get("branch_key_hex"), []).append(worker)
    branch_rows = []
    previous_branch_rows = []
    branch_classes = []
    branch_columns = _branch_columns(display_order)
    for branch in branches:
        branch_key = branch["branch_key_hex"]
        if branch["branch_status"] == "active":
            active_branch_keys.add(branch_key)
        branch_rows.append(
            _branch_display_row(branch, generated_at, display_order)
        )
        previous_branch = previous_branches.get(branch_key)
        previous_branch_rows.append(
            _branch_display_row(
                previous_branch, previous_generated_at, display_order
            )
            if previous_branch is not None else None
        )
        branch_classes.append(_semantic_branch_class(branch, previous_branch))

    header_rendered = False
    for index, branch in enumerate(branches):
        branch_lines.extend(_render_table(
            branch_columns, [branch_rows[index]], width, indent="  ",
            row_classes=[branch_classes[index]],
            previous_rows=[previous_branch_rows[index]],
            measurement_rows=branch_rows,
            include_header=not header_rendered, measure_headings=True,
            color=color,
        ))
        header_rendered = True
        branch_key = branch["branch_key_hex"]
        if branch["branch_status"] != "active":
            continue
        live_branch_workers = [
            item for item in workers_by_branch.get(branch_key, [])
            if item["is_live"]
        ]
        if live_branch_workers:
            branch_lines.extend(_worker_lines(
                live_branch_workers, previous_workers, generated_at,
                previous_generated_at, width, indent="    ", color=color,
                state=_worker_state,
            ))
    if len(branch_lines) == 1:
        branch_lines.append("  none")

    remaining_worker_lines = ["Other workers"]
    remaining_workers = [
        worker for worker in workers
        if not (worker["is_live"] and worker.get("branch_key_hex") in active_branch_keys)
    ]

    if remaining_workers:
        remaining_worker_lines.extend(_worker_lines(
            remaining_workers, previous_workers, generated_at,
            previous_generated_at, width, indent="  ", color=color,
            state=_worker_state,
        ))
    else:
        remaining_worker_lines.append("  none")

    return [
        ("header", header),
        ("summary", summary),
        ("branches", branch_lines),
        ("workers", remaining_worker_lines),
    ]


def _branch_columns(display_order):
    columns = []
    if display_order.hotkey_letters:
        columns.append(
            TerminalColumn("Key", "display_hotkey", required=True)
        )
    columns.extend([
        TerminalColumn(
            "Ref",
            lambda row: "@" + _display_reference(row["branch_reference"]),
            required=True,
        ),
        TerminalColumn("GuessD", "guess_depth", required=True, alignment="right"),
        TerminalColumn("Phase", "display_phase", required=True),
        TerminalColumn(
            "Done", _display_done, required=True, alignment="right",
            highlight_rule=_count_increase_rule("completed_candidate_count"),
        ),
        TerminalColumn("W", "worker_count", required=True, alignment="right"),
        TerminalColumn(
            "Ans", "answer_count", remove_priority=10, alignment="right"
        ),
        TerminalColumn(
            "Bulk", "bulk_completed_candidate_count", remove_priority=20,
            alignment="right",
            highlight_rule=_count_increase_rule("bulk_completed_candidate_count"),
        ),
        TerminalColumn(
            "Best", "display_best", remove_priority=30,
            highlight_rule=_best_erd_improvement_rule,
        ),
        TerminalColumn(
            "MaxRD", "best_max_remaining_depth", remove_priority=40,
            alignment="right",
        ),
        TerminalColumn("ETA", "display_eta", remove_priority=50, alignment="right"),
        TerminalColumn(
            "Spine", "display_spine", remove_priority=60,
            minimum_width=12, maximum_width=40, truncation="tail",
        ),
    ])
    return columns


def _display_done(row):
    candidate_count = row["candidate_count"]
    total = f"{candidate_count:,}" if candidate_count is not None else "—"
    return f"{row['completed_candidate_count']:,}/{total}"


def _display_best(row):
    if not row.get("best_guess"):
        return "—"
    best = row["best_guess"].upper()
    if row.get("best_guess_is_answer"):
        best += "*"
    if row.get("best_erd") is not None:
        best += f"/{row['best_erd']:.3f}"
    return best


def _branch_display_row(branch, generated_at, display_order):
    branch_row = dict(branch)
    branch_row["display_hotkey"] = _hotkey_label(
        display_order, branch["branch_key_hex"]
    )
    branch_row["display_phase"] = {
        "finalizing": "final",
    }.get(branch["branch_phase"], branch["branch_phase"] or "—")
    branch_row["display_best"] = _display_best(branch)
    branch_row["display_eta"] = _abbreviate_duration(
        _branch_eta(branch, generated_at)
    )
    branch_row["display_spine"] = " ▸ ".join(
        f"{step['word'].upper()} {step['pattern']}" for step in branch["spine"]
    ) or "—"
    return branch_row


def _worker_lines(
    workers, previous_workers, generated_at, previous_generated_at, width,
    *, indent, color, state,
):
    rows = [
        _worker_display_row(worker, generated_at, state(worker))
        for worker in workers
    ]
    previous_rows = []
    row_classes = []
    for worker in workers:
        previous_worker = previous_workers.get(worker["worker_id"])
        previous_rows.append(
            _worker_display_row(
                previous_worker, previous_generated_at, state(worker)
            )
            if previous_worker is not None else None
        )
        row_classes.append(_semantic_worker_class(
            worker, previous_worker, generated_at
        ))
    return _render_worker_table(
        rows, row_classes, width, indent=indent, color=color,
        previous_rows=previous_rows,
    )


def _worker_display_row(worker, generated_at, state):
    age = max(0, generated_at - worker["updated_at"])
    row = dict(worker)
    worker_number = str(
        worker.get("worker_number")
        or worker["worker_id"].rsplit("-", 1)[-1]
    )
    row["display_worker"] = (
        f"w{worker_number}" if worker_number.isdigit() else worker["worker_id"]
    )
    row["display_state"] = {
        "finalizing": "final",
        "transitioning": "trans",
        "coordinating": "coord",
    }.get(state or "active", state or "active")
    row["display_age"] = _abbreviate_duration(age)
    candidate = worker.get("current_candidate")
    display_candidate = "—"
    if candidate:
        display_candidate = candidate.upper()
        if worker.get("current_candidate_is_answer"):
            display_candidate += "*"
    row["display_candidate"] = display_candidate
    guess_depth = worker.get("current_max_guess_depth")
    row["display_guess_depth"] = (
        f"d{guess_depth}" if guess_depth is not None else "—"
    )
    rate = worker.get("nodes_per_second")
    row["display_rate"] = (
        f"{_abbreviate_number(rate)}/s" if rate is not None else "—"
    )
    return row


def _worker_columns():
    # Rendered headerless: dN, …/s, and …s suffixes label the cells in place.
    return [
        TerminalColumn("Worker", "display_worker", required=True),
        TerminalColumn("State", "display_state", required=True),
        TerminalColumn(
            "Candidate", "display_candidate", required=True,
            highlight_rule=_candidate_advance_rule,
        ),
        TerminalColumn(
            "MaxGD", "display_guess_depth", required=True, alignment="right"
        ),
        TerminalColumn(
            "Rate", "display_rate", remove_priority=10, alignment="right",
            highlight_rule=_rate_stall_rule,
        ),
        TerminalColumn("Age", "display_age", required=True, alignment="right"),
    ]


def _render_worker_table(
    rows, row_classes, width, *, indent, color, previous_rows=None,
):
    return _render_table(
        _worker_columns(), rows, width, indent=indent,
        row_classes=row_classes, previous_rows=previous_rows,
        include_header=False, color=color,
    )


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
    generated_text = datetime.fromtimestamp(
        report["generated_at"]
    ).strftime("%Y-%m-%d %H:%M:%S")
    header = [_fit(f"{title}  {generated_text}", width)]
    header.extend(_source_problem_lines(report, width))
    return header


def _word_erd_line(erd_summary):
    if not erd_summary:
        return "ERD —"
    state = erd_summary.get("state")
    if state == "complete":
        return f"ERD {erd_summary['erd']:.3f}  max-d={erd_summary['max_remaining_depth']}"
    if state == "infeasible":
        return (
            f"ERD ∞ — {erd_summary['infeasible_group_count']} of "
            f"{erd_summary['response_group_count']} response groups "
            "unsolvable within budget"
        )
    return (
        f"ERD pending — {erd_summary['resolved_group_count']} of "
        f"{erd_summary['response_group_count']} response groups solved"
    )


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
        f"Word {data['word'].upper()}{'*' if data.get('word_is_answer') else ''}  "
        f"context=@{_display_reference(context['branch_reference'])} "
        f"n={context['answer_count']} d={context['guess_depth']}",
        width,
    )
    header.append(_fit("  " + _word_erd_line(data.get("erd_summary")), width))
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
    hotkey_width = 5 if display_order.hotkey_letters else 0

    def group_line(group):
        hotkey = _hotkey_label(display_order, group["branch_key_hex"])
        return (
            f"{hotkey:<{hotkey_width}}"
            f"{group['pattern']:<7}  {group['answer_count']:>7}  "
            f"{group['branch_status']:<8}  {str(group['branch_phase'] or '—'):<10}  "
            f"{group['cache_state']:<14}  "
            f"{_display_best(group):<10}  @{_display_reference(group['branch_reference'])}"
        )

    rows = [
        " " * hotkey_width
        + "Pattern  Answers  Status    Phase       Cache           Best        Ref"
    ]
    for group in ordered_groups:
        line = _fit(group_line(group), width)
        previous_group = previous_groups.get(group["branch_key_hex"])
        if previous_group is None:
            line = _colorize(line, "green", color)
        elif group != previous_group and color:
            line = _highlight_changes(line, _fit(group_line(previous_group), width))
        rows.append(line)
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
        f"Branch @{_display_reference(branch['branch_reference'])}  "
        f"n={branch['answer_count']} "
        f"d={branch['guess_depth']} budget={branch['budget']}",
        width,
    )
    header.append(_fit(
        f"  status={branch['branch_status']} "
        f"phase={branch['branch_phase'] or '—'}",
        width,
    ))
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
                f"  status={queue['branch_status']} phase={queue['branch_phase']}  "
                f"priority={queue['priority']}  "
                f"progress={progress}  bulk={queue['bulk_completed_candidate_count']}  "
                f"best={queue['best_guess'] or '—'}  nodes={_abbreviate_number(queue['search_node_count'])}",
                width,
            ),
        ]
        candidate_count = queue["candidate_count"]
        if candidate_count:
            worker_positions = [
                (worker.get("candidate_index"), worker["worker_number"])
                for worker in data["workers"]
                if worker["is_live"] and worker.get("candidate_index") is not None
            ]
            sweep = candidate_sweep_bar(
                candidate_count,
                data.get("completed_candidate_indexes") or (),
                worker_positions,
                width=max(10, min(40, width - 4)),
            )
            if sweep.strip():
                queue_lines.append(_fit(f"  [{sweep}]", width))
    cache = data["cache"]
    cache_line = f"  {cache['cache_state']}"
    if cache.get("best_guess"):
        cache_line += (
            f"  best={cache['best_guess'].upper()}/ERD "
            f"{_format_branch_erd(cache['best_erd'], branch['answer_count'])}"
        )
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
    ordered_workers = [
        workers_by_id[worker_id]
        for worker_id in display_order.worker_ids
        if worker_id in workers_by_id
    ]
    if ordered_workers:
        worker_lines.extend(_worker_lines(
            ordered_workers, previous_workers, report["generated_at"],
            (previous_report or report)["generated_at"], width,
            indent="  ", color=color,
            state=_worker_state,
        ))
    else:
        worker_lines.append("  none")

    detail_lines = ["Candidate state"]
    detail_lines.append(
        f"  republished candidates: {len(data['republished_candidates'])}"
    )
    claim_summary = data.get("claim_summary") or {}
    if claim_summary.get("total_claim_count"):
        completion_fields = [
            f"{claim_summary['done_count']:,} done",
            f"{claim_summary['evaluated_count']:,} evaluated",
            f"{claim_summary['bulk_eliminated_count']:,} bulk proofs",
        ]
        if claim_summary.get("provenance_unknown_count"):
            completion_fields.append(
                f"{claim_summary['provenance_unknown_count']:,} unattributed"
            )
        if claim_summary.get("in_flight_count"):
            completion_fields.append(
                f"{claim_summary['in_flight_count']:,} in flight"
            )
        detail_lines.extend(_inline_section("  completion:", completion_fields, width))
        contributions = claim_summary.get("worker_contributions") or []
        if contributions:
            worker_fields = [
                f"{_worker_number_label(row['worker_id'])} {row['done_count']:,}"
                for row in contributions
            ]
            detail_lines.extend(_inline_section("  by worker:", worker_fields, width))
    telemetry_lines = ["Telemetry"]
    bundle_summary = data.get("bundle_summary")
    if bundle_summary:
        bundle_labels = {
            "bundle_count": "bundles",
            "node_count": "nodes",
            "wall_millis": "wall ms",
            "censored_unit_count": "censored units",
            "maximum_bundle_node_count": "max bundle nodes",
        }
        bundle_fields = [
            bundle_labels.get(key, key.replace("_", " "))
            + " "
            + (f"{value:,}" if isinstance(value, int) else str(value))
            for key, value in bundle_summary.items()
        ]
        telemetry_lines.extend(
            _inline_section("  active bundles:", bundle_fields, width)
        )
    finalizations = data.get("recent_finalizations", [])
    for finalization in finalizations:
        spine = finalization.get("spine") or "(spine unknown)"
        telemetry_lines.append(_fit(
            f"  {spine}  {finalization['outcome']} "
            f"epoch={finalization['epoch']} "
            f"nodes={_abbreviate_number(finalization['search_node_count'])} "
            f"evaluated={finalization['evaluated_candidate_count']} "
            f"bulk={finalization['bulk_completed_candidate_count']}",
            width,
        ))
    finalization_total = data.get("finalization_total_count", len(finalizations))
    if finalization_total > len(finalizations):
        telemetry_lines.append(_fit(
            f"  … and {finalization_total - len(finalizations)} more reaching "
            f"this same answer set; view --limit {finalization_total} to see all",
            width,
        ))
    for miss in data.get("cut_reuse_misses", []):
        available_bound = _format_branch_erd(
            miss["available_bound"], miss["answer_count"], ceiling=True
        )
        telemetry_lines.append(_fit(
            f"  cut reuse miss epoch={miss['epoch']} budget={miss['budget']} "
            f"available ERD bound={available_bound}",
            width,
        ))
    if len(telemetry_lines) == 1:
        telemetry_lines.append("  none")
    return [
        ("header", header), ("queue", queue_lines), ("cache", cache_lines),
        ("workers", worker_lines), ("candidate_state", detail_lines),
        ("telemetry", telemetry_lines),
    ]


def _word_groups(siblings):
    """Sibling nodes in payload order, gathered under the word that was played.

    A node with no recorded guess has no word to gather under and stands alone
    under a None key.
    """
    groups = {}
    for node in siblings:
        step = node["step"]
        word = step["word"] if step is not None else None
        key = node["node_id"] if word is None else word
        groups.setdefault(key, (word, []))[1].append(node)
    return list(groups.values())


def _render_tree_sections(report, width, display_order):
    data = report["data"]
    header = _semantic_header(report, "Live queue tree", width)
    lines = ["Topology: queue"]
    if not data["tree_available"]:
        lines.append(f"  unavailable: {data['unavailable_reason']}")
    else:
        children_by_parent = {}
        for node in data["nodes"]:
            children_by_parent.setdefault(node["parent_node_id"], []).append(node)
        for children in children_by_parent.values():
            children.sort(key=lambda node: (
                (node["step"] or {}).get("word", ""),
                (node["step"] or {}).get("pattern", ""),
                node["branch_key_hex"] or "",
            ))

        visited_node_ids = set()

        def append_node_and_descendants(node, level, carries_word=True):
            if node["node_id"] in visited_node_ids:
                return
            visited_node_ids.add(node["node_id"])
            step = node["step"]
            label = "unknown"
            if step is not None:
                label = (
                    f"{step['word'].upper()} {step['pattern']}" if carries_word
                    else step["pattern"]
                )
            detail = ""
            if node["branch_reference"]:
                detail = (
                    f"  @{_display_reference(node['branch_reference'])} "
                    f"{node['branch_status']}/{node['branch_phase']} "
                    f"n={node['answer_count']} workers={node['worker_count']}"
                )
            if node["is_context"]:
                detail += "  [context]"
            hotkey = _hotkey_label(display_order, node.get("branch_key_hex"))
            hotkey_prefix = f"{hotkey} " if hotkey else ""
            lines.append(_fit(f"{' ' * level}{hotkey_prefix}{label}{detail}", width))
            children = children_by_parent.get(node["node_id"], [])
            if children:
                append_word_groups(children, level + 1)

        # Every word is a group at every level, including the one-pattern case:
        # the word is named once and its response patterns are the rows beneath
        # it.
        def append_word_groups(siblings, level):
            for word, group_nodes in _word_groups(siblings):
                if word is None:
                    for node in group_nodes:
                        append_node_and_descendants(node, level)
                    continue
                branch_count = len(group_nodes)
                lines.append(_fit(
                    f"{' ' * level}{word.upper()}  {branch_count} "
                    f"{'branch' if branch_count == 1 else 'branches'}",
                    width,
                ))
                for node in group_nodes:
                    append_node_and_descendants(node, level + 1, carries_word=False)

        append_word_groups(children_by_parent.get(None, []), 0)
        for node in data["nodes"]:
            if node["node_id"] not in visited_node_ids:
                append_node_and_descendants(node, 0)
    return [("header", header), ("tree", lines)]


def _render_queue_collection_sections(report, width, display_order):
    data = report["data"]
    header = _semantic_header(report, "Queue report", width)
    summary = data.get("summary", {})
    lines = [
        f"Queue rows: {data.get('matched_rows', 0)} matched",
        _fit(f"  status={summary.get('branch_count_by_status', {})}", width),
        _fit(f"  phase={summary.get('branch_count_by_phase', {})}", width),
    ]
    for row in data.get("rows", []):
        hotkey = _hotkey_label(display_order, row.get("branch_key_hex"))
        hotkey_prefix = f"{hotkey} " if hotkey else ""
        lines.append(_fit(
            f"  {hotkey_prefix}@{_display_reference(row['branch_reference'])} "
            f"{row['branch_status']}/{row['branch_phase']} "
            f"n={row['answer_count']} "
            f"guess_depth={len((row.get('spine') or '').split()) // 2} "
            f"priority={row['priority']} workers={row['worker_count']}",
            width,
        ))
    return [("header", header), ("queue_rows", lines)]


def _render_workers_collection_sections(report, previous_report, color, width, display_order):
    data = report["data"]
    header = _semantic_header(report, "Workers report", width)
    workers = data.get("rows", [])
    incoming_ids = [worker["worker_id"] for worker in workers]
    display_order.worker_ids = DisplayOrder._update_identities(
        display_order.worker_ids, incoming_ids, set()
    )
    by_id = {worker["worker_id"]: worker for worker in workers}
    previous_by_id = {
        worker["worker_id"]: worker
        for worker in (previous_report or {}).get("data", {}).get("rows", [])
    }
    lines = [f"Workers: {data.get('matched_rows', 0)} matched"]
    ordered_workers = [
        by_id[worker_id]
        for worker_id in display_order.worker_ids
        if worker_id in by_id
    ]
    if ordered_workers:
        lines.extend(_worker_lines(
            ordered_workers, previous_by_id, report["generated_at"],
            (previous_report or report)["generated_at"], width,
            indent="  ", color=color,
            state=_worker_state,
        ))
    return [("header", header), ("worker_rows", lines)]


def _render_cache_collection_sections(report, width, display_order):
    data = report["data"]
    header = _semantic_header(report, "Cache report", width)
    lines = ["Cache"]
    if "distributions" in data:
        summary = data["summary"]
        lines.append(_fit(
            f"  exact={summary['exact_branch_count']} "
            f"loss={summary['loss_branch_count']} "
            f"recent={summary['recent_exact_branch_count']}",
            width,
        ))
        distributions = data["distributions"]
        for label, key in (
            ("state", "state_branch_counts"),
            (
                "max remaining depth",
                "exact_branch_count_by_max_remaining_depth",
            ),
            ("solve budget", "exact_branch_count_by_solve_budget"),
            ("taint", "exact_branch_count_by_taint"),
            ("loss budget", "loss_branch_count_by_loss_budget"),
        ):
            values = distributions[key]
            formatted_values = ", ".join(
                f"{name}={count}" for name, count in sorted(values.items())
            ) or "none"
            lines.append(_fit(f"  {label}: {formatted_values}", width))
    elif "rows" in data:
        for row in data["rows"]:
            hotkey = _hotkey_label(display_order, row.get("branch_key_hex"))
            hotkey_prefix = f"{hotkey} " if hotkey else ""
            lines.append(_fit(
                f"  {hotkey_prefix}{row['pattern']} n={row['answer_count']} "
                f"{row['cache_state']} @{_display_reference(row['branch_reference'])}",
                width,
            ))
    elif "cache" in data:
        lines.append(_fit(
            f"  @{_display_reference(data['branch_reference'])} "
            f"{data['cache']['cache_state']}",
            width,
        ))
    return [("header", header), ("cache_rows", lines)]


def _render_hotspot_sections(report, width, display_order):
    data = report["data"]
    header = _semantic_header(report, f"Hotspots by {data['field']}", width)
    lines = [
        f"Population: {data['population']}",
        _fit(
            f"  epoch={data['epoch']} since-seconds={data['since_seconds']} "
            f"sample-size={data['sample_size']} sampled={data['sampled_row_count']} "
            f"truncated={str(data['sample_truncated']).lower()}",
            width,
        ),
    ]
    for row in data["rows"]:
        if row.get("branch_reference"):
            identity = "@" + _display_reference(row["branch_reference"])
        else:
            identity = row.get("row_id") or "bucket"
        hotkey = _hotkey_label(display_order, row.get("branch_key_hex"))
        hotkey_prefix = f"{hotkey} " if hotkey else ""
        metrics = ", ".join(
            f"{key}={_format_metric_value(key, value)}" for key, value in row.items()
            if key not in (
                "row_id", "branch_key_hex", "branch_reference", "spine"
            )
        )
        lines.append(_fit(f"  {hotkey_prefix}{identity}  {metrics}", width))
    return [("header", header), ("hotspots", lines)]


def _render_leaderboard_sections(report, width):
    data = report["data"]
    counts = data["counts"]
    header = _semantic_header(
        report,
        f"Opener leaderboard  candidates={data['candidate_count']}",
        width,
    )
    summary = [
        "Ranked by ERD",
        _fit(
            f"  complete {counts['complete']}  "
            f"pending {counts['pending']}  "
            f"infeasible {counts['infeasible']}  "
            f"(showing {len(data['rows'])} of {data['total_rows']})",
            width,
        ),
    ]
    rows = ["Rank  Opener   ERD    MaxRD"]
    for row in data["rows"]:
        word = row["word"].upper() + ("*" if row["word_is_answer"] else "")
        rows.append(_fit(
            f"{row['rank']:>4}  {word:<7}  {row['erd']:.3f}  "
            f"{row['max_remaining_depth']}",
            width,
        ))
    if len(rows) == 1:
        rows.append("none complete yet")
    return [("header", header), ("summary", summary), ("leaderboard", rows)]


def _report_sections(report, previous_report, color, width, display_order):
    if report.get("tree"):
        return _render_tree_sections(report, width, display_order)
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
    if report["report_kind"] == "queue":
        return _render_queue_collection_sections(report, width, display_order)
    if report["report_kind"] == "workers":
        return _render_workers_collection_sections(
            report, previous_report, color, width, display_order
        )
    if report["report_kind"] == "cache":
        return _render_cache_collection_sections(report, width, display_order)
    if report["report_kind"] == "hotspots":
        return _render_hotspot_sections(report, width, display_order)
    if report["report_kind"] == "leaderboard":
        return _render_leaderboard_sections(report, width)
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
        self.navigation_stack = [self._request_from_args()]
        self.branch_hotkeys = {}
        self.worker_hotkeys = {}
        self.branch_letter_by_key = {}
        self.branch_targets = {}
        self.previous_sections = []
        self.terminal_settings = None
        self.cursor_hidden = False
        self.pending_input_character = None

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

    def _request_from_args(self):
        branch_target = getattr(self.args, "branch_target", None)
        return ReportRequest(
            report_kind=getattr(self.args, "report_kind", "auto"),
            branch_target=branch_target if branch_target is not None else ReportRequest().branch_target,
            include_claims=getattr(self.args, "claims", False),
            include_answers=getattr(self.args, "answers", False),
            tree=getattr(self.args, "tree", False),
            filters=getattr(self.args, "filters", ReportRequest().filters),
            worker_id=getattr(self.args, "worker", None),
            hotspot_field=getattr(self.args, "hotspot_field", None),
            epoch=getattr(self.args, "epoch", None),
            since_seconds=getattr(self.args, "since_seconds", None),
            sample_size=getattr(self.args, "sample_size", None),
        )

    @property
    def current_request(self):
        return self.navigation_stack[-1]

    def _collect(self):
        return collect_report(self._sources(), self.current_request)

    @staticmethod
    def _identity_rows(report, identity_key):
        data = report.get("data", {})
        collections = [
            data.get("branches", []),
            data.get("workers", []),
            data.get("response_groups", []),
            data.get("rows", []),
            data.get("nodes", []),
        ]
        branch = data.get("branch")
        if isinstance(branch, dict):
            collections.append([branch])
        rows = []
        seen = set()
        for collection in collections:
            for row in collection or []:
                identity = row.get(identity_key)
                if identity is not None and identity not in seen:
                    rows.append(row)
                    seen.add(identity)
        return rows

    def _update_navigation_targets(self, report):
        branch_rows = self._identity_rows(report, "branch_key_hex")
        branch_keys = [row["branch_key_hex"] for row in branch_rows]
        self.branch_targets = {
            row["branch_key_hex"]: self._branch_target(row)
            for row in branch_rows
        }
        retained_letters = {
            branch_key: letter
            for branch_key, letter in self.branch_letter_by_key.items()
            if branch_key in branch_keys
        }
        used_letters = set(retained_letters.values())
        free_letters = iter(
            letter for letter in BRANCH_HOTKEYS if letter not in used_letters
        )
        for branch_key in branch_keys:
            if branch_key not in retained_letters:
                letter = next(free_letters, None)
                if letter is not None:
                    retained_letters[branch_key] = letter
        self.branch_letter_by_key = retained_letters
        self.branch_hotkeys = {
            letter: branch_key
            for branch_key, letter in retained_letters.items()
        }
        self.display_order.hotkey_letters = dict(retained_letters)

        self.worker_hotkeys = {}
        for worker in self._identity_rows(report, "worker_id"):
            worker_number = str(worker.get("worker_number") or "")
            if worker_number.isdigit():
                self.worker_hotkeys[worker_number] = worker["worker_id"]

    def _branch_target(self, row):
        spine = row.get("spine")
        if isinstance(spine, list):
            branch_target_text = " ".join(
                token
                for step in spine
                for token in (step["word"], step["pattern"])
            )
            if branch_target_text:
                return parse_report_branch_target(branch_target_text)
        if isinstance(spine, str) and spine.strip():
            try:
                branch_target = parse_report_branch_target(spine)
                if branch_target.kind == "branch":
                    return branch_target
            except ValueError:
                pass
        if (
            self.current_request.branch_target.kind == "word"
            and row.get("pattern") is not None
        ):
            parts = [
                token
                for step in self.current_request.branch_target.steps
                for token in (step.word, step.pattern)
            ]
            parts.extend((self.current_request.branch_target.trailing_word, row["pattern"]))
            return parse_report_branch_target(parts)
        branch_digest = hashlib.sha1(
            bytes.fromhex(row["branch_key_hex"])
        ).hexdigest()
        return parse_report_branch_target("@" + branch_digest)

    def _navigation_section(self):
        # Branch hotkey letters appear inline as [x] on their own rows; worker
        # rows are selected by their wN digits.
        fields = []
        if self.branch_hotkeys:
            fields.append("[a-z] branch")
        if self.worker_hotkeys:
            fields.append("[0-9] worker")
        if len(self.navigation_stack) > 1:
            fields.append("[esc] back")
        fields.extend(("[space] refresh", "[q] quit"))
        return [("navigation", _inline_section("keys:", fields, self._width()))]

    def _read_worker_number(self, first_digit):
        worker_number = first_digit
        while any(
            candidate.startswith(worker_number) and candidate != worker_number
            for candidate in self.worker_hotkeys
        ):
            ready, _, _ = select.select([self.input_stream], [], [], 0.2)
            if not ready:
                break
            character = self.input_stream.read(1)
            if not character.isdigit():
                self.pending_input_character = character
                break
            worker_number += character
        return worker_number

    def _reset_navigation_display(self):
        self.previous_report = None
        self.previous_sections = []
        self.display_order = DisplayOrder()
        self._write_loading_notice()

    def _write_loading_notice(self):
        sources = self._sources()
        self.output_stream.write(
            "\033[2J\033[H"
            f"Collecting report…{CLEAR_LINE}\n"
            f"  queue: {sources.queue_path}{CLEAR_LINE}\n"
            f"  cache: {sources.cache_path}{CLEAR_LINE}\n"
        )
        self.output_stream.flush()

    def _select_branch(self, branch_key_hex):
        request = replace(
            self.current_request,
            report_kind="auto",
            branch_target=self.branch_targets[branch_key_hex],
            tree=False,
            worker_id=None,
            hotspot_field=None,
            epoch=None,
            since_seconds=None,
            sample_size=None,
        )
        self.navigation_stack.append(request)
        self._reset_navigation_display()

    def _select_worker(self, worker_id):
        request = replace(
            self.current_request,
            report_kind="workers",
            branch_target=parse_report_branch_target(None),
            tree=False,
            worker_id=worker_id,
            hotspot_field=None,
            epoch=None,
            since_seconds=None,
            sample_size=None,
        )
        self.navigation_stack.append(request)
        self._reset_navigation_display()

    def _navigate_back(self):
        if len(self.navigation_stack) > 1:
            self.navigation_stack.pop()
            self._reset_navigation_display()

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
        # The loading notice is on screen; repaint from the top without a
        # separate clear so the first report draws over it without a flash.
        lines = []
        for index, (_name, section_lines) in enumerate(sections):
            if index:
                lines.append("")
            lines.extend(section_lines)
        self.output_stream.write("\033[H")
        self.output_stream.write(
            (CLEAR_LINE + "\n").join(lines) + CLEAR_LINE + "\033[J"
        )
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
            if self.pending_input_character is not None:
                character = self.pending_input_character
                self.pending_input_character = None
            else:
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
            if character in ("\x08", "\x7f", "\x1b"):
                self._navigate_back()
                return True
            if character in self.branch_hotkeys:
                self._select_branch(self.branch_hotkeys[character])
                return True
            if character.isdigit():
                worker_number = self._read_worker_number(character)
                if worker_number not in self.worker_hotkeys:
                    continue
                self._select_worker(self.worker_hotkeys[worker_number])
                return True

    def _run_tty_text(self):
        file_descriptor = self.input_stream.fileno()
        self.terminal_settings = termios.tcgetattr(file_descriptor)
        new_settings = termios.tcgetattr(file_descriptor)
        new_settings[3] &= ~(termios.ICANON | termios.ECHO)
        try:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, new_settings)
            self.output_stream.write("\033[?25l")
            self.cursor_hidden = True
            self._write_loading_notice()
            while True:
                try:
                    report = self._collect()
                    self._update_navigation_targets(report)
                    sections = (
                        self._terminal_sections(report)
                        + self._navigation_section()
                    )
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
