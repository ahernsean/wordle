"""Presentation-neutral reports for ERD swarm state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import collections
import datetime
import os
import re
import sqlite3
import time
from typing import Optional, Tuple

from cache_sqlite import ScoreCache, branch_reference
from pattern_matrix import PatternMatrix
from erd_queue import (
    DISK_STOP_FRACTION,
    DISK_WARN_FRACTION,
    ERDQueue,
    WORKER_LIVENESS_SECONDS,
    best_first_rank,
    derive_telemetry_path,
    disk_stats,
)
from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_TELEMETRY_PATH,
)
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache, load_word_list
from wordle_ui import fmt_pattern, parse_pattern


SCHEMA_VERSION = 3
WORKER_STALE_SECONDS = 20
DEFAULT_TREE_PAGE_SIZE = 10

# Source ERD summaries are immutable until the score cache changes.  The
# Sources view polls frequently, so retain the completed folds between polls.
_SOURCE_ERD_SUMMARY_CACHE = {}

RichSpineStep = Tuple[Optional[int], Optional[str], Optional[str], str]


def _open_report_queue(sources):
    """Open the initialized queue for report reads without schema writes."""
    return ERDQueue(
        sources.queue_path,
        telemetry_path=sources.telemetry_path,
        initialize_schema=False,
    )


def disk_fill_rate(samples, now):
    """Return the fitted bytes-per-second filesystem fill rate."""
    fresh_samples = [sample for sample in samples if now - sample[0] <= 900]
    if len(fresh_samples) < 2:
        return None
    first_timestamp = fresh_samples[0][0]
    elapsed_seconds = [sample[0] - first_timestamp for sample in fresh_samples]
    available_bytes = [sample[1] for sample in fresh_samples]
    sample_count = len(fresh_samples)
    mean_seconds = sum(elapsed_seconds) / sample_count
    mean_available_bytes = sum(available_bytes) / sample_count
    seconds_variance = sum(
        (value - mean_seconds) ** 2 for value in elapsed_seconds
    )
    if seconds_variance == 0:
        return None
    covariance = sum(
        (seconds - mean_seconds) * (available - mean_available_bytes)
        for seconds, available in zip(elapsed_seconds, available_bytes)
    )
    return -(covariance / seconds_variance)


@dataclass(frozen=True)
class SpineStep:
    word: str
    pattern: str


@dataclass(frozen=True)
class ReportBranchTarget:
    kind: str
    steps: tuple[SpineStep, ...]
    trailing_word: str | None
    branch_reference: str | None
    input_text: str

    @classmethod
    def root(cls) -> "ReportBranchTarget":
        return cls("root", (), None, None, "")


@dataclass(frozen=True)
class ResolvedBranch:
    answer_words: tuple[str, ...]
    branch_key: bytes
    steps: tuple[SpineStep, ...]
    trailing_word: str | None = None


@dataclass(frozen=True)
class ReportFilters:
    branch_statuses: tuple[str, ...] = ()
    branch_phases: tuple[str, ...] = ()
    minimum_answer_count: int | None = None
    maximum_answer_count: int | None = None
    budget: int | None = None
    priority: int | None = None
    sort: str | None = None
    limit: int | None = None
    finalization_cursor_direction: str | None = None
    finalization_cursor_recorded_at: int | None = None
    finalization_cursor_id: int | None = None
    group_by: str | None = None
    source_states: tuple[str, ...] = ()
    source_offset: int | None = None
    branch_row_offset: int | None = None


BRANCH_STATUSES = ("active", "pending", "done", "unqueued")
BRANCH_PHASES = ("queued", "evaluating", "finalizing", "complete")
# A source word's lifecycle is its own: it is queued until a worker picks one
# of its requests up, and complete only once every request behind it is.  These
# are deliberately not the branch statuses, which describe a single branch.
SOURCE_STATES = ("queued", "active", "complete")
GROUP_BY_STRATEGIES = (
    "none", "status", "answer_count", "cache_state", "worker_presence", "priority",
)
SOURCE_GROUP_BY_STRATEGIES = (
    "none", "state", "worker_presence", "priority", "completed", "elapsed",
    "worker_time", "requested",
)
SOURCE_SORT_FIELDS = (
    "default", "age", "word", "priority", "branches", "open", "done", "workers",
    "completed", "erd", "requested", "elapsed", "worker_time",
)
# Sorts that only a source report can serve, so another report asking for one
# is rejected rather than quietly sorted some other way.
SOURCE_ONLY_SORT_FIELDS = (
    "word", "branches", "open", "done", "completed", "erd", "requested",
    "elapsed", "worker_time",
)
_SOURCE_STATE_GROUP_ORDER = {"active": 0, "queued": 1, "complete": 2}
_STATUS_GROUP_ORDER = {"active": 0, "pending": 1, "done": 2, "unqueued": 3}
_ANSWER_COUNT_GROUP_BOUNDARIES = ((1, "1"), (9, "2–9"), (29, "10–29"), (99, "30–99"))
_CACHE_STATE_GROUP_ORDER = {"missing": 0, "loss": 1, "exact": 2, "not_applicable": 3}
_CACHE_STATE_GROUP_LABEL = {
    "missing": "missing", "loss": "loss", "exact": "exact", "not_applicable": "trivial",
}


def parse_branch_filter(value, filter_name, allowed_values):
    """Parse one comma-separated branch filter, with ``all`` meaning no filter."""
    values = tuple(part.strip().lower() for part in value.split(","))
    if not values or any(not part for part in values):
        raise ValueError(f"{filter_name} requires comma-separated values")
    if "all" in values:
        if len(values) != 1:
            raise ValueError(
                f"{filter_name}=all cannot be combined with other values"
            )
        return ()
    unknown_values = [part for part in values if part not in allowed_values]
    if unknown_values:
        raise ValueError(
            f"unknown {filter_name} value: {unknown_values[0]}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{filter_name} contains a duplicate value")
    return values


def branch_status_and_phase(
    pending_status, active_status, worker_count, cache_state=None,
):
    """Derive operational status and monotonic progress from stored state."""
    if pending_status == "done" or cache_state in (
        "exact", "loss", "not_applicable",
    ):
        return "done", "complete"
    if active_status == "finalized":
        phase = "finalizing"
    elif active_status == "open" or pending_status == "in_progress":
        phase = "evaluating"
    elif pending_status == "pending":
        phase = "queued"
    else:
        return "unqueued", None
    return ("active" if worker_count else "pending"), phase


@dataclass(frozen=True)
class ReportRequest:
    report_kind: str = "auto"
    branch_target: ReportBranchTarget = field(default_factory=ReportBranchTarget.root)
    include_claims: bool = False
    include_answers: bool = False
    tree: bool = False
    filters: ReportFilters = field(default_factory=ReportFilters)
    worker_id: str | None = None
    hotspot_field: str | None = None
    epoch: int | None = None
    tree_parent: str = ""
    tree_cursor: str | None = None
    since_seconds: int | None = None
    sample_size: int | None = None


def validate_report_request(request: ReportRequest) -> None:
    """Reject report options that have no meaning for the selected report."""
    report_kind = request.report_kind
    branch_target_kind = request.branch_target.kind
    if request.tree and report_kind in ("cache", "hotspots", "leaderboard", "sources"):
        raise ValueError(f"--tree cannot be used with --{report_kind}")
    if (request.tree_parent or request.tree_cursor) and not request.tree:
        raise ValueError("tree_parent and tree_cursor require tree")
    if request.include_claims and (
        request.tree
        or report_kind != "auto"
        or branch_target_kind not in ("branch", "branch_reference")
    ):
        raise ValueError("--claims requires a singular branch target")
    if request.include_answers and (
        request.tree
        or report_kind in ("queue", "workers", "leaderboard", "sources")
        or (report_kind == "auto" and branch_target_kind == "root")
    ):
        raise ValueError(
            "--answers requires a word or branch report without --tree"
        )
    if (
        report_kind == "auto"
        and branch_target_kind == "word"
        and request.filters.sort is not None
        and request.filters.sort not in ("default", "size", "workers", "priority")
    ):
        raise ValueError(
            "--sort for word reports must be default, size, workers, or priority"
        )
    if request.filters.group_by is not None and report_kind == "sources":
        if request.filters.group_by not in SOURCE_GROUP_BY_STRATEGIES:
            raise ValueError(
                "group_by for source reports must be one of "
                + ", ".join(SOURCE_GROUP_BY_STRATEGIES)
            )
    elif request.filters.group_by is not None and (
        report_kind != "auto"
        or branch_target_kind != "word"
        or request.filters.group_by not in GROUP_BY_STRATEGIES
    ):
        raise ValueError(
            "group_by requires a word or source report and must be one of "
            + ", ".join(GROUP_BY_STRATEGIES)
        )
    if request.filters.source_states and report_kind != "sources":
        raise ValueError("source_state requires a source report")
    for name in ("source_offset", "branch_row_offset"):
        offset = getattr(request.filters, name)
        if offset is None:
            continue
        if report_kind != "sources":
            raise ValueError(f"{name} requires a source report")
        if offset < 0:
            raise ValueError(f"{name} cannot be negative")
    if request.filters.sort is not None:
        if report_kind == "sources":
            if request.filters.sort not in SOURCE_SORT_FIELDS:
                raise ValueError(
                    "--sort for source reports must be one of "
                    + ", ".join(SOURCE_SORT_FIELDS)
                )
        elif request.filters.sort in SOURCE_ONLY_SORT_FIELDS:
            raise ValueError(
                f"--sort {request.filters.sort} requires a source report"
            )
    historical_hotspot = request.hotspot_field in (
        "evaluated-candidates", "bulk-completed-candidates",
        "one-level-erd-prunes", "two-level-erd-prunes",
        "cut-reuse", "coordination",
    )
    if report_kind == "hotspots" and historical_hotspot and (
        request.filters.branch_statuses or request.filters.branch_phases
    ):
        raise ValueError("historical hotspots cannot use branch filters")
    if (
        report_kind == "hotspots"
        and request.hotspot_field == "coordination"
        and branch_target_kind != "root"
    ):
        raise ValueError("coordination hotspots cannot use a branch target")
    if request.worker_id is not None and report_kind != "workers":
        raise ValueError("worker requires a workers report")
    if report_kind == "root_progress":
        if request.tree:
            raise ValueError("--tree cannot be used with --root-progress")
        # Any spine ending in a word works: the rollup scopes telemetry by
        # spine prefix, and a deeper spine is simply a longer prefix.  A target
        # that does not end in a word has no response groups to report.
        if branch_target_kind != "word":
            raise ValueError("--root-progress requires a target ending in a word")
    if report_kind == "sources" and branch_target_kind not in ("root", "word"):
        raise ValueError(
            "--sources accepts only a trailing word or no branch target"
        )


@dataclass(frozen=True)
class ReportSources:
    queue_path: str
    cache_path: str
    answer_list_path: str
    candidate_list_path: str
    telemetry_path: str | None = None

    @classmethod
    def defaults(cls) -> "ReportSources":
        return cls(
            queue_path=DEFAULT_QUEUE_PATH,
            cache_path=DEFAULT_CACHE_PATH,
            answer_list_path=DEFAULT_ANSWER_LIST_PATH,
            candidate_list_path=DEFAULT_CANDIDATE_LIST_PATH,
            telemetry_path=DEFAULT_TELEMETRY_PATH,
        )


def parse_report_branch_target(parts: list[str] | str | None) -> ReportBranchTarget:
    if parts is None:
        input_text = ""
    elif isinstance(parts, str):
        input_text = parts.strip()
    else:
        input_text = " ".join(parts).strip()
    if not input_text:
        return ReportBranchTarget.root()
    tokens = input_text.split()
    if len(tokens) == 1 and tokens[0].startswith("@"):
        digest_prefix = tokens[0][1:]
        if not re.fullmatch(r"[0-9a-fA-F]{4,40}", digest_prefix):
            raise ValueError(
                f"invalid token {tokens[0]!r}: expected @ followed by 4-40 hexadecimal characters"
            )
        return ReportBranchTarget(
            "branch_reference", (), None, digest_prefix.lower(), input_text
        )

    normalized_words = []
    normalized_patterns = []
    for index, token in enumerate(tokens):
        if index % 2 == 0:
            if not re.fullmatch(r"[A-Za-z]{5}", token):
                raise ValueError(
                    f"invalid token {token!r}: expected a five-letter word"
                )
            normalized_words.append(token.lower())
        else:
            try:
                normalized_patterns.append(fmt_pattern(parse_pattern(token)))
            except ValueError as error:
                raise ValueError(
                    f"invalid token {token!r}: expected a five-character response pattern"
                ) from error

    step_count = len(normalized_patterns)
    steps = tuple(
        SpineStep(normalized_words[index], normalized_patterns[index])
        for index in range(step_count)
    )
    if len(normalized_words) == step_count + 1:
        return ReportBranchTarget(
            "word", steps, normalized_words[-1], None, input_text
        )
    if len(normalized_words) == step_count:
        return ReportBranchTarget("branch", steps, None, None, input_text)
    raise ValueError(
        f"invalid branch target {input_text!r}: expected alternating word and response pattern"
    )


def resolve_branch_target(
    branch_target: ReportBranchTarget, all_answers
) -> ResolvedBranch:
    if branch_target.kind == "branch_reference":
        raise ValueError("a branch reference requires queue resolution")
    branch_words = list(all_answers)
    response_cache = ResponseCache(list(all_answers), score_cache=None)
    for step in branch_target.steps:
        groups = response_cache.group_words(step.word, branch_words)
        branch_words = groups.get(parse_pattern(step.pattern), [])
    branch_key = ScoreCache.encode_subset(branch_words)
    return ResolvedBranch(
        tuple(branch_words), branch_key, branch_target.steps, branch_target.trailing_word
    )


def resolve_branch_reference(queue, digest_prefix, cache=None) -> dict:
    matches = [] if queue is None else queue.branch_rows_for_reference_prefix(digest_prefix)
    cache_matches = [] if cache is None else cache.branch_keys_for_reference_prefix(
        digest_prefix
    )
    queue_keys = {bytes(row["branch_key"]) for row in matches}
    cache_only_keys = [key for key in cache_matches if key not in queue_keys]
    if not matches and not cache_only_keys:
        raise ValueError(f"No queued or cached @{digest_prefix} branch found")
    if len(matches) + len(cache_only_keys) > 1:
        candidates = [
            {"branch_reference": branch_reference(bytes(row["branch_key"])),
             "branch_key": bytes(row["branch_key"]), "spine": queue.row_spine_text(row)} for row in matches
        ]
        candidates.extend({"branch_reference": branch_reference(key), "branch_key": key, "spine": None}
                          for key in cache_only_keys)
        error = ValueError(f"branch reference @{digest_prefix} is ambiguous")
        error.candidates = candidates
        raise error
    if matches:
        return matches[0]
    return {"branch_key": cache_only_keys[0], "spine": None}


def collect_ambiguous_branch_reference_report(sources, request, error):
    answer_set = _decorative_answer_set(sources)
    candidates = []
    for candidate in error.candidates:
        answer_words = sorted(_decode_branch_key(candidate["branch_key"]))
        candidates.append({
            "branch_reference": candidate["branch_reference"],
            "spine": [
                _step_payload(step, answer_set)
                for step in _spine_steps(candidate["spine"])
            ],
            "answer_count": len(answer_words),
            "answer_preview": answer_words[:3],
        })
    report = _semantic_report("branch_reference_matches", sources,
                              request.branch_target, int(time.time()), {
                                  "branch_reference": request.branch_target.branch_reference,
                                  "candidates": candidates,
                              }, request)
    report["sources"]["queue"]["ok"] = True
    report["sources"]["cache"]["ok"] = True
    return report


def parse_rich_spine(path: str | None) -> list[RichSpineStep]:
    """Parse legacy and depth-tagged rich worker descent paths."""
    if not path:
        return []
    result = []
    for token in path.split("→"):
        guess_depth = None
        if token and token[0].isdigit():
            colon_position = token.index(":")
            guess_depth = int(token[:colon_position])
            token = token[colon_position + 1:]
        if "/" in token:
            guess_pattern, answer_count_text = token.rsplit("/", 1)
            if ":" in guess_pattern:
                word, pattern = guess_pattern.split(":", 1)
            else:
                word, pattern = None, guess_pattern
            result.append((guess_depth, word, pattern, answer_count_text))
        elif token:
            result.append((guess_depth, None, None, token))
    return result


def normalize_worker_descent(
    parsed_path: list[RichSpineStep], answer_set: set[str]
) -> list[dict]:
    return [
        {
            "guess_depth": guess_depth,
            "word": word.lower() if word is not None else None,
            "pattern": pattern,
            "answer_count_text": answer_count_text,
            "word_is_answer": bool(word and word.lower() in answer_set),
        }
        for guess_depth, word, pattern, answer_count_text in parsed_path
    ]


def _row_value(row, key, default=None):
    if row is None:
        return default
    try:
        if hasattr(row, "keys") and key not in row.keys():
            return default
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _normalized_pattern(pattern):
    if pattern is None:
        return None
    if isinstance(pattern, int):
        return fmt_pattern(pattern)
    return str(pattern).lower()


def _normalized_branch_spine(row, answer_set):
    stored_spine = _row_value(row, "spine")
    if stored_spine:
        tokens = stored_spine.split()
        return [
            {
                "word": tokens[index].lower(),
                "pattern": _normalized_pattern(tokens[index + 1]),
                "word_is_answer": tokens[index].lower() in answer_set,
            }
            for index in range(0, len(tokens) - 1, 2)
        ]
    source_word = _row_value(row, "source_word")
    source_pattern = _row_value(row, "source_pattern")
    if source_word and source_pattern is not None:
        normalized_word = source_word.lower()
        return [{
            "word": normalized_word,
            "pattern": _normalized_pattern(source_pattern),
            "word_is_answer": normalized_word in answer_set,
        }]
    return []


def _normalize_branch(
    row, branch_status, branch_phase, progress, worker_count, answer_set,
):
    branch_key = bytes(row["branch_key"])
    spine = _normalized_branch_spine(row, answer_set)
    best_guess = _row_value(row, "best_guess")
    source_word = _row_value(row, "source_word")
    return {
        "branch_reference": branch_reference(branch_key),
        "branch_key_hex": branch_key.hex(),
        "branch_status": branch_status,
        "branch_phase": branch_phase,
        "raw_status": row["status"],
        "answer_count": row["n_words"],
        "candidate_count": row["n_candidates"],
        "completed_candidate_count": progress["completed_candidate_count"],
        "bulk_completed_candidate_count": progress[
            "bulk_completed_candidate_count"
        ],
        "one_level_erd_pruned_candidate_count": progress.get(
            "one_level_erd_pruned_candidate_count",
            progress["bulk_completed_candidate_count"],
        ),
        "two_level_erd_pruned_candidate_count": progress.get(
            "two_level_erd_pruned_candidate_count", 0
        ),
        "priority": row["priority"],
        "is_cooperative": not bool(_row_value(row, "is_user_queued", False)),
        "source_word": source_word.lower() if source_word else None,
        "source_pattern": _normalized_pattern(_row_value(row, "source_pattern")),
        "best_guess": best_guess.lower() if best_guess else None,
        "best_guess_is_answer": _is_answer(best_guess, answer_set),
        "best_erd": _row_value(row, "best_erd"),
        "best_max_remaining_depth": _row_value(row, "best_max_depth"),
        "budget": _row_value(row, "budget"),
        "guess_depth": len(spine),
        "spine": spine,
        "worker_count": worker_count,
        "created_at": _row_value(row, "created_at"),
        "search_node_count": _row_value(row, "nodes_spent", 0),
        "ceiling": _row_value(row, "ceiling"),
    }


def _worker_number(worker_id):
    return worker_id.rsplit("-", 1)[-1]


def _normalize_worker(row, generated_at, answer_set):
    branch_key_value = _row_value(row, "current_branch_key")
    branch_key = bytes(branch_key_value) if branch_key_value is not None else None
    worker_id = row["worker_id"]
    current_candidate = _row_value(row, "cur_candidate")
    best_guess = _row_value(row, "best_guess")
    return {
        "worker_id": worker_id,
        "worker_number": _worker_number(worker_id),
        "pid": row["pid"],
        "answer_count": _row_value(row, "n_words"),
        "updated_at": row["updated_at"],
        "is_live": generated_at - row["updated_at"] <= WORKER_LIVENESS_SECONDS,
        "branch_reference": branch_reference(branch_key) if branch_key else None,
        "branch_key_hex": branch_key.hex() if branch_key else None,
        "branch_context": _normalized_branch_spine(row, answer_set),
        # True while the worker's branch still has an active row; False once it
        # has been finalized and removed, which lags the heartbeat by up to one
        # interval and leaves the worker naming a branch no report will list.
        "on_active_branch": bool(_row_value(row, "on_active_branch", 1)),
        "candidate_index": _row_value(row, "claim_idx"),
        "claim_started_at": _row_value(row, "claim_started_at"),
        "completed_claim_count": _row_value(row, "claims_done", 0),
        "current_candidate": current_candidate.lower() if current_candidate else None,
        "current_candidate_is_answer": bool(
            current_candidate and current_candidate.lower() in answer_set
        ),
        "current_max_guess_depth": _row_value(row, "cur_max_depth"),
        "current_node_count": _row_value(row, "cur_nodes"),
        "help_recursion_depth": _row_value(row, "cur_help_depth", 0),
        "nodes_per_second": _row_value(row, "node_rate"),
        "descent": normalize_worker_descent(
            parse_rich_spine(_row_value(row, "cur_path")), answer_set
        ),
        "cache_hit_count": _row_value(row, "cache_hits", 0),
        "cache_miss_count": _row_value(row, "cache_misses", 0),
        "solved_evaluation_count": _row_value(row, "n_ok", 0),
        "erd_cutoff_evaluation_count": _row_value(row, "n_cutoff", 0),
        "remaining_depth_pruned_evaluation_count": _row_value(row, "n_pruned", 0),
        "best_guess": best_guess.lower() if best_guess else None,
        "best_guess_is_answer": bool(
            best_guess and best_guess.lower() in answer_set
        ),
        "best_erd": _row_value(row, "best_erd"),
        "bound_erd": _row_value(row, "bound_erd"),
        "source_work_id": _row_value(row, "source_work_id"),
        "scheduling_role": _row_value(row, "scheduling_role"),
        "scheduling_role_reason": scheduling_role_reason(
            _row_value(row, "scheduling_role")
        ),
    }


def scheduling_role_reason(scheduling_role):
    """Human-readable explanation of why a worker is running the branch it is
    on, for the given persisted scheduling role (see erd_queue.SCHEDULING_ROLE_*).

    None (no role recorded, e.g. an idle worker between claims) reports
    unattributed rather than guessing.
    """
    return {
        "preferred": "serving its preferred source work",
        "fallback": "serving fallback work: preferred source(s) had no "
                    "claimable bundle at the last claim boundary",
        "direct": "direct branch work with no live source-work ownership",
    }.get(scheduling_role, "unattributed")


def worker_state(worker, generated_at, branch_phase):
    """The one worker-state classification, shared by every report and both
    clients.

    Computed in the model and stored on each worker so the browser card and
    the terminal render the same label from the same precedence rather than
    each re-deriving it.  branch_phase is the phase of the worker's current
    branch, or None when it has none listed.  A worker silent past
    WORKER_STALE_SECONDS reads as stale regardless of its branch: the order
    below is the single source of truth for how these conditions rank.
    """
    if not worker["is_live"]:
        return "dead"
    if generated_at - worker["updated_at"] > WORKER_STALE_SECONDS:
        return "stale"
    if worker["branch_key_hex"] is None:
        return "idle"
    if not worker["on_active_branch"]:
        return "transitioning"
    if branch_phase == "finalizing":
        return "finalizing"
    # Holding a branch but naming no candidate: between candidates on a
    # coordination wait (all siblings' claims taken, or awaiting a rival's
    # finalize), not evaluating.
    if not worker["current_candidate"]:
        return "coordinating"
    return "working"


def _branch_ownership(branch_workers, all_workers, normalized_claims, branch_key_hex):
    """Summarize who still owns this branch's unfinished work.

    live_workers are current branch heartbeats. claim_holders_off_branch are
    live workers that still own unfinished claims on this branch while their
    current heartbeat names some other branch.
    """
    unfinished_claim_counts = {}
    for claim in normalized_claims:
        if claim["state"] != "in_flight" or not claim["worker_id"]:
            continue
        worker_id = claim["worker_id"]
        unfinished_claim_counts[worker_id] = (
            unfinished_claim_counts.get(worker_id, 0) + 1
        )
    live_workers = [worker for worker in branch_workers if worker["is_live"]]
    live_workers.sort(key=_worker_sort_key)
    off_branch_holders = []
    worker_by_id = {worker["worker_id"]: worker for worker in all_workers}
    for worker_id, in_flight_claim_count in unfinished_claim_counts.items():
        worker = worker_by_id.get(worker_id)
        if worker is None or not worker["is_live"]:
            continue
        if worker["branch_key_hex"] == branch_key_hex:
            continue
        off_branch_holder = dict(worker)
        off_branch_holder["in_flight_claim_count"] = in_flight_claim_count
        off_branch_holders.append(off_branch_holder)
    off_branch_holders.sort(key=_worker_sort_key)
    return {
        "live_workers": live_workers,
        "claim_holders_off_branch": off_branch_holders,
    }


def _worker_sort_key(worker):
    worker_number = worker["worker_number"]
    if worker_number.isdigit():
        return (0, int(worker_number), worker["worker_id"])
    return (1, worker["worker_id"])


def _empty_data():
    return {
        "disk": {
            "total_bytes": None,
            "used_bytes": None,
            "available_bytes": None,
            "used_fraction": None,
            "queue_wal_bytes": None,
            "fill_rate_bytes_per_second": None,
            "warning_fraction": DISK_WARN_FRACTION,
            "stop_fraction": DISK_STOP_FRACTION,
        },
        "queue_counts": {
            "pending_branch_count": 0,
            "evaluating_user_branch_count": 0,
            "evaluating_cooperative_branch_count": 0,
            "finalizing_branch_count": 0,
            "done_branch_count": 0,
        },
        "cache_summary": {
            "exact_branch_count": 0,
            "recent_exact_branch_count": 0,
            "loss_branch_count": 0,
        },
        "worker_totals": {
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "solved_evaluation_count": 0,
            "erd_cutoff_evaluation_count": 0,
            "remaining_depth_pruned_evaluation_count": 0,
        },
        "branches": [],
        "workers": [],
    }


def _queue_overview(sources, generated_at, answer_set, report):
    queue = None
    try:
        queue = _open_report_queue(sources)
        report["sources"]["telemetry"]["ok"] = True
        counts = queue.counts_by_status()
        heartbeats = list(queue.heartbeats_with_branch())
        filters = report.get("filters") or {}
        queue_result = queue.report_queue_rows(
            filters=filters,
            generated_at=generated_at,
        )
        active_branch_keys = [
            bytes(row["branch_key"])
            for row in queue_result["rows"]
            if row["branch_status"] == "active"
        ]
        completed_candidate_indexes = queue.completed_candidate_indexes_by_branch(
            active_branch_keys
        )
        normalized_rows = []
        for row in queue_result["rows"]:
            branch_values = {
                "branch_key": row["branch_key"],
                "status": row["raw_status"],
                "n_words": row["answer_count"],
                "n_candidates": row["candidate_count"],
                "priority": row["priority"],
                "source_word": row["source_word"],
                "source_pattern": row["source_pattern"],
                "best_guess": row["best_guess"],
                "best_erd": row["best_erd"],
                "best_max_depth": row["best_max_remaining_depth"],
                "budget": row["budget"],
                "spine": row["spine"],
                "created_at": row["created_at"],
                "nodes_spent": row["search_node_count"],
                "ceiling": row["ceiling"],
                "is_user_queued": not row["is_cooperative"],
            }
            progress = {
                "completed_candidate_count": row[
                    "completed_candidate_count"
                ],
                "bulk_completed_candidate_count": row[
                    "bulk_completed_candidate_count"
                ],
                "one_level_erd_pruned_candidate_count": row[
                    "one_level_erd_pruned_candidate_count"
                ],
                "two_level_erd_pruned_candidate_count": row[
                    "two_level_erd_pruned_candidate_count"
                ],
            }
            normalized = _normalize_branch(
                branch_values,
                row["branch_status"],
                row["branch_phase"],
                progress,
                row["worker_count"],
                answer_set,
            )
            # Active branches carry their done claim indexes so overview
            # displays can draw the candidate sweep. Other statuses have no
            # live claim rows.
            if normalized["branch_status"] == "active":
                normalized["completed_candidate_indexes"] = (
                    completed_candidate_indexes[bytes(row["branch_key"])]
                )
            normalized_rows.append(normalized)

        workers = [
            _normalize_worker(row, generated_at, answer_set) for row in heartbeats
        ]
        workers.sort(key=_worker_sort_key)
        phase_by_key = {
            branch["branch_key_hex"]: branch["branch_phase"]
            for branch in normalized_rows
        }
        for worker in workers:
            worker["state"] = worker_state(
                worker, generated_at, phase_by_key.get(worker["branch_key_hex"])
            )
        worker_total_keys = tuple(report["data"]["worker_totals"])
        worker_totals = {
            key: sum(worker[key] for worker in workers if worker["is_live"])
            for key in worker_total_keys
        }
        epoch_metadata = queue.epoch_metadata()
        filesystem = disk_stats(sources.queue_path)
        report["data"]["disk"] = {
            "total_bytes": filesystem["total_bytes"],
            "used_bytes": filesystem["used_bytes"],
            "available_bytes": filesystem["avail_bytes"],
            "used_fraction": filesystem["used_fraction"],
            "queue_wal_bytes": queue.wal_size_bytes(),
            "fill_rate_bytes_per_second": disk_fill_rate(
                queue.disk_samples(), generated_at
            ),
            "warning_fraction": DISK_WARN_FRACTION,
            "stop_fraction": DISK_STOP_FRACTION,
        }

        phase_counts = queue.overview_phase_counts()
        report["data"]["queue_counts"] = {
            "pending_branch_count": counts.get("pending", 0),
            **phase_counts,
            "done_branch_count": counts.get("done", 0),
        }
        report["data"]["worker_totals"] = worker_totals
        report["data"]["branches"] = normalized_rows
        report["data"]["workers"] = workers
        report["sources"]["queue"].update({
            "ok": True,
            "epoch": _row_value(epoch_metadata, "epoch"),
            "label": _row_value(epoch_metadata, "label"),
            "git_sha": _row_value(epoch_metadata, "git_sha"),
        })
    except (sqlite3.Error, OSError) as error:
        message = str(error)
        report["sources"]["queue"]["error"] = message
        if queue is None:
            report["sources"]["telemetry"]["error"] = message
    finally:
        if queue is not None:
            queue.close()


def _cache_overview(sources, generated_at, answer_words, report):
    cache = None
    try:
        cache = ScoreCache(
            sources.cache_path, answer_words, checkpoint_on_close=False
        )
        report["data"]["cache_summary"] = cache.erd_report_summary(
            ERD_ALL, generated_at - 300
        )
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()


def _branch_target_payload(branch_target):
    return {
        "kind": branch_target.kind,
        "steps": [
            {"word": step.word, "pattern": step.pattern}
            for step in branch_target.steps
        ],
        "trailing_word": branch_target.trailing_word,
        "branch_reference": branch_target.branch_reference,
        "input_text": branch_target.input_text,
    }


def _filters_payload(filters):
    return {
        "branch_statuses": list(filters.branch_statuses),
        "branch_phases": list(filters.branch_phases),
        "minimum_answer_count": filters.minimum_answer_count,
        "maximum_answer_count": filters.maximum_answer_count,
        "budget": filters.budget,
        "priority": filters.priority,
        "sort": filters.sort,
        "limit": filters.limit,
    }


def _report_envelope(
    report_kind, sources, generated_at, data, branch_target=None, request=None
):
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": report_kind,
        "generated_at": generated_at,
        "branch_target": _branch_target_payload(branch_target) if branch_target is not None else None,
        "filters": _filters_payload(request.filters) if request else {},
        "tree": request.tree if request else False,
        "sources": {
            "queue": {
                "path": sources.queue_path,
                "ok": False,
                "error": None,
                "epoch": None,
                "label": None,
                "git_sha": None,
            },
            "telemetry": {
                "path": (
                    sources.telemetry_path
                    if sources.telemetry_path is not None
                    else derive_telemetry_path(sources.queue_path)
                ),
                "ok": False,
                "error": None,
            },
            "cache": {"path": sources.cache_path, "ok": False, "error": None},
        },
        "data": data,
    }


def _semantic_report(
    report_kind, sources, branch_target, generated_at, data, request=None
):
    return _report_envelope(
        report_kind, sources, generated_at, data, branch_target, request
    )


def _is_answer(word, answer_set):
    return bool(word and word.lower() in answer_set)


def _decorative_answer_set(sources):
    """Answer set for word_is_answer decoration, empty if unreadable.

    Queue/tree/hotspot reports are fully meaningful without an answer list --
    it only decorates words with '*'. A missing or unreadable answer list
    here must cost the decoration, not the report: it is not a queue or
    cache failure and must never be attributed to either source.
    """
    try:
        return set(load_word_list(sources.answer_list_path))
    except OSError:
        return frozenset()


def _step_payload(step, answer_set):
    return {
        "word": step.word,
        "pattern": step.pattern,
        "word_is_answer": _is_answer(step.word, answer_set),
    }


def _resolved_branch_payload(resolved, answer_set):
    return {
        "branch_reference": branch_reference(resolved.branch_key),
        "branch_key_hex": resolved.branch_key.hex(),
        "spine": [_step_payload(step, answer_set) for step in resolved.steps],
        "guess_depth": len(resolved.steps),
        "answer_count": len(resolved.answer_words),
    }


def _with_best_guess_is_answer(cache_state, answer_set):
    """Add best_guess_is_answer to a row that carries a best_guess field.

    ScoreCache and the queue's hotspot rows have no answer-set knowledge of
    their own, so callers that hold one attach this flag themselves rather
    than teaching those layers about answer sets. Rows with no best_guess
    field (most hotspot populations) are returned unchanged rather than
    gaining an always-False flag that never meant anything for them.
    """
    if not cache_state or "best_guess" not in cache_state:
        return cache_state
    return {
        **cache_state,
        "best_guess_is_answer": _is_answer(cache_state.get("best_guess"), answer_set),
    }


def _mark_queue_source_ok(report):
    report["sources"]["queue"]["ok"] = True
    report["sources"]["telemetry"]["ok"] = True


def _mark_queue_source_error(report, error):
    message = str(error)
    report["sources"]["queue"]["error"] = message
    report["sources"]["telemetry"]["error"] = message


_ALL_GREEN_PATTERN_TEXT = fmt_pattern(3 ** 5 - 1)


def _candidate_erd_summary(response_groups, group_budget):
    """Fold a candidate's response groups into its own ERD and worst-case line.

    Playing the candidate spends one guess from this branch's budget; each
    response group is then solved independently.  So the candidate's ERD is the
    answer-weighted mean of the groups' ERDs plus one, and its worst-case line
    is the deepest group line plus one.  A single remaining answer is solved by
    playing it (one more guess) unless the candidate itself was the answer
    (all-green response, zero more guesses) — but that one guess needs a guess
    left, so with `group_budget < 1` a lone survivor is a proven loss, matching
    `wordle_engine.evaluate_candidate`, which checks the budget floor before its
    n == 1 shortcut.

    The fold reports one of three states.  It is `complete` — an exact ERD and
    worst-case line — only once every group is solved.  A group proven
    unsolvable within budget (a loss, or a lone survivor with no guess left)
    makes the candidate `infeasible`: its ERD is unbounded and no further search
    changes that.  A group still being searched leaves the candidate `pending`.
    """
    total_answers = sum(group["answer_count"] for group in response_groups)
    weighted_remaining_depth = 0.0
    max_group_remaining_depth = 0
    resolved_group_count = 0
    infeasible_group_count = 0
    pending_group_count = 0
    for group in response_groups:
        best_erd = group["best_erd"]
        max_remaining_depth = group["max_remaining_depth"]
        if best_erd is None:
            if group["answer_count"] < 2:
                solved_by_candidate = group["pattern"] == _ALL_GREEN_PATTERN_TEXT
                if not solved_by_candidate and group_budget < 1:
                    infeasible_group_count += 1
                    continue
                best_erd = 0.0 if solved_by_candidate else 1.0
                max_remaining_depth = 0 if solved_by_candidate else 1
            elif group["cache_state"] == "loss":
                infeasible_group_count += 1
                continue
            else:
                pending_group_count += 1
                continue
        elif max_remaining_depth is None:
            # An ERD with no proven worst-case line cannot complete the fold.
            pending_group_count += 1
            continue
        resolved_group_count += 1
        weighted_remaining_depth += group["answer_count"] * best_erd
        max_group_remaining_depth = max(max_group_remaining_depth, max_remaining_depth)
    if infeasible_group_count:
        state = "infeasible"
    elif pending_group_count or total_answers == 0:
        state = "pending"
    else:
        state = "complete"
    return {
        "state": state,
        "erd": (
            1.0 + weighted_remaining_depth / total_answers
            if state == "complete" else None
        ),
        "max_remaining_depth": (
            1 + max_group_remaining_depth if state == "complete" else None
        ),
        "resolved_group_count": resolved_group_count,
        "infeasible_group_count": infeasible_group_count,
        "response_group_count": len(response_groups),
    }


def _resolved_candidate_erd(cache, branch_key, candidate_word, policy,
                             response_groups, group_budget, stored_erd_map=None):
    """A candidate's own ERD at a branch — the single place every caller
    (word report, leaderboard, sources view) gets this number.

    Once `_candidate_erd_summary` reports a candidate `complete`, the value is
    an immutable fact: every response group behind it is itself an exact,
    finalized branch row, and those do not change outside a reverification
    pass.  So a `complete` fold is persisted the first time it is seen, and
    every later call for the same (branch, candidate) is one stored-row lookup
    instead of a fresh fold over the candidate's response groups.

    `stored_erd_map`, when given, comes from `cache.candidate_erd_map` and
    serves a whole-vocabulary pass (the leaderboard); omit it for a single
    candidate, which does one direct lookup instead.  `cache` may be None when
    the caller has no cache connection, leaving the fold unchanged but
    unpersisted.
    """
    stored = None
    if cache is not None:
        stored = (
            cache.candidate_erd_from_map(branch_key, candidate_word, stored_erd_map)
            if stored_erd_map is not None
            else cache.read_candidate_erd(branch_key, candidate_word, policy)
        )
    # A changed vocabulary reshapes a candidate's own grouping, so a stored row
    # whose group count no longer matches describes a different partition.
    if stored is not None and stored["response_group_count"] == len(response_groups):
        return {
            "state": "complete",
            "erd": stored["erd"],
            "max_remaining_depth": stored["max_remaining_depth"],
            "resolved_group_count": stored["response_group_count"],
            "infeasible_group_count": 0,
            "response_group_count": stored["response_group_count"],
        }
    summary = _candidate_erd_summary(response_groups, group_budget)
    if cache is not None and summary["state"] == "complete":
        cache.write_candidate_erd(
            branch_key, candidate_word, policy, summary["erd"],
            summary["max_remaining_depth"], summary["response_group_count"],
        )
    return summary


def _response_group_key(row: dict, group_by: str) -> tuple:
    """Map a response-group row to (sort_key, label) for the given strategy."""
    if group_by == "status":
        status = row["branch_status"]
        return (_STATUS_GROUP_ORDER[status], status)
    if group_by == "answer_count":
        for upper, label in _ANSWER_COUNT_GROUP_BOUNDARIES:
            if row["answer_count"] <= upper:
                return (upper, label)
        return (float("inf"), "100+")
    if group_by == "cache_state":
        cache_state = row["cache_state"]
        return (
            _CACHE_STATE_GROUP_ORDER[cache_state],
            _CACHE_STATE_GROUP_LABEL[cache_state],
        )
    if group_by == "worker_presence":
        has_worker = bool(row["worker_count"])
        return (0 if has_worker else 1, "has worker" if has_worker else "no worker")
    if group_by == "priority":
        priority = row["priority"]
        if priority is None:
            return (float("inf"), "no priority")
        return (-priority, f"priority {priority}")
    return (0, "all")


def _response_group_rollup(rows: list) -> dict:
    """Fold response-group rows into the same breakdown used for one group
    or the grand total, so the two are always consistent with each other."""
    return {
        "answer_count": sum(row["answer_count"] for row in rows),
        "branch_count": len(rows),
        "trivial_count": sum(row["cache_state"] == "not_applicable" for row in rows),
        "exact_count": sum(row["cache_state"] == "exact" for row in rows),
        "loss_count": sum(row["cache_state"] == "loss" for row in rows),
        "missing_count": sum(row["cache_state"] == "missing" for row in rows),
    }


def _grouped_response_groups(rows: list, group_by: str) -> list:
    """Bucket rows by `group_by`, each with its own rollup, ordered by strategy."""
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key, label = _response_group_key(row, group_by)
        group = grouped.setdefault(key, {"label": label, "rows": []})
        group["rows"].append(row)
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        groups.append({
            "label": group["label"],
            "rollup": _response_group_rollup(group["rows"]),
            "rows": group["rows"],
        })
    return groups


def collect_word_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    resolved = resolve_branch_target(request.branch_target, all_answers)
    word = resolved.trailing_word
    if word is None:
        raise ValueError("word report requires a branch target ending in a word")
    response_cache = ResponseCache(all_answers, score_cache=None)
    groups = response_cache.group_words(word, list(resolved.answer_words))
    group_rows = []
    branch_keys = []
    for pattern_code, answer_words in sorted(groups.items()):
        if not answer_words:
            continue
        branch_key = ScoreCache.encode_subset(answer_words)
        branch_keys.append(branch_key)
        group_rows.append({
            "pattern": fmt_pattern(pattern_code),
            "answer_count": len(answer_words),
            "branch_reference": branch_reference(branch_key),
            "branch_key_hex": branch_key.hex(),
            "branch_key": branch_key,
            "answer_words": list(answer_words),
        })
    data = {
        "word": word,
        "word_is_answer": word in answer_set,
        "context": _resolved_branch_payload(resolved, answer_set),
        "response_group_counts": {},
        "response_groups": [],
    }
    report = _semantic_report(
        "word", sources, request.branch_target, generated_at, data, request
    )

    pending_rows = {}
    active_rows = {}
    worker_counts = {}
    queue = None
    try:
        queue = _open_report_queue(sources)
        pending_rows = queue.status_by_branch_keys(branch_keys)
        active_rows = queue.active_branches_by_keys(branch_keys)
        worker_counts = queue.worker_counts_by_branch(WORKER_LIVENESS_SECONDS)
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()

    group_budget = GAME_GUESSES - len(resolved.steps) - 1
    cache_states = {
        branch_key: ScoreCache.report_branch_state_without_rows(
            branch_key, group_budget
        )
        for branch_key in branch_keys
    }
    cache = None
    erd_summary = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        cache_states = cache.report_branch_states(
            branch_keys, ERD_ALL, group_budget
        )
        # Resolved here, while the cache is open, and against the branch this
        # word is played from — not the root, since a word report can sit at
        # any spine.
        erd_summary = _resolved_candidate_erd(
            cache, resolved.branch_key, word, ERD_ALL,
            [
                {
                    "pattern": row["pattern"],
                    "answer_count": row["answer_count"],
                    "best_erd": cache_states[row["branch_key"]]["best_erd"],
                    "max_remaining_depth":
                        cache_states[row["branch_key"]]["max_remaining_depth"],
                    "cache_state": cache_states[row["branch_key"]]["cache_state"],
                }
                for row in group_rows
            ],
            group_budget,
        )
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()

    for group_row in group_rows:
        branch_key = group_row.pop("branch_key")
        answer_words = group_row.pop("answer_words")
        pending_row = pending_rows.get(branch_key)
        active_row = active_rows.get(branch_key)
        cache_state = cache_states[branch_key]
        worker_count = worker_counts.get(branch_key, 0)
        branch_status, branch_phase = branch_status_and_phase(
            _row_value(pending_row, "status"),
            _row_value(active_row, "status"),
            worker_count,
            cache_state["cache_state"],
        )
        group_row.update({
            "branch_status": branch_status,
            "branch_phase": branch_phase,
            "priority": _row_value(
                active_row, "effective_priority",
                _row_value(pending_row, "priority"),
            ),
            "worker_count": worker_count,
            "cache_state": cache_state["cache_state"],
            "best_guess": cache_state["best_guess"],
            "best_guess_is_answer": bool(
                cache_state["best_guess"]
                and cache_state["best_guess"].lower() in answer_set
            ),
            "best_erd": cache_state["best_erd"],
            "max_remaining_depth": cache_state["max_remaining_depth"],
            "updated_at": cache_state["updated_at"],
        })
        if request.include_answers:
            group_row["answer_words"] = answer_words
        data["response_groups"].append(group_row)

    response_groups = data["response_groups"]
    all_response_groups = list(response_groups)
    filters = request.filters
    if filters.branch_statuses:
        response_groups = [
            row for row in response_groups
            if row["branch_status"] in filters.branch_statuses
        ]
    if filters.branch_phases:
        response_groups = [
            row for row in response_groups
            if row["branch_phase"] in filters.branch_phases
        ]
    if filters.minimum_answer_count is not None:
        response_groups = [
            row for row in response_groups
            if row["answer_count"] >= filters.minimum_answer_count
        ]
    if filters.maximum_answer_count is not None:
        response_groups = [
            row for row in response_groups
            if row["answer_count"] <= filters.maximum_answer_count
        ]
    if filters.budget is not None and filters.budget != group_budget:
        response_groups = []
    if filters.priority is not None:
        response_groups = [
            row for row in response_groups if row["priority"] == filters.priority
        ]
    if filters.sort == "size":
        response_groups.sort(key=lambda row: (-row["answer_count"], row["pattern"]))
    elif filters.sort == "workers":
        response_groups.sort(key=lambda row: (-row["worker_count"], row["pattern"]))
    elif filters.sort == "priority":
        response_groups.sort(key=lambda row: (-(row["priority"] or 0), row["pattern"]))
    matched_response_groups = list(response_groups)
    data["response_group_summary"] = _response_group_rollup(matched_response_groups)
    displayed_response_groups = (
        matched_response_groups[:filters.limit]
        if filters.limit is not None else matched_response_groups
    )
    if filters.group_by and filters.group_by != "none":
        data["response_group_groups"] = _grouped_response_groups(
            displayed_response_groups, filters.group_by
        )
    data["response_group_counts"] = {
        "response_group_count": len(all_response_groups),
        "trivial_response_group_count": sum(
            row["answer_count"] < 2 for row in all_response_groups
        ),
        "queued_response_group_count": sum(
            row["branch_status"] != "unqueued" for row in all_response_groups
        ),
        "active_response_group_count": sum(
            row["branch_status"] == "active"
            for row in all_response_groups
        ),
        "exact_response_group_count": sum(
            row["cache_state"] == "exact" for row in all_response_groups
        ),
        "loss_response_group_count": sum(
            row["cache_state"] == "loss" for row in all_response_groups
        ),
        "missing_response_group_count": sum(
            row["cache_state"] == "missing" for row in all_response_groups
        ),
    }
    data["erd_summary"] = (
        erd_summary if erd_summary is not None
        else _candidate_erd_summary(all_response_groups, group_budget)
    )
    data["total_rows"] = len(all_response_groups)
    data["matched_rows"] = len(matched_response_groups)
    data["response_groups"] = displayed_response_groups
    return report


ROOT_PROGRESS_RECENT_WINDOW_SECONDS = 86400


def _root_progress_estimate(active_branches, recent_window_seconds, now):
    """A completion estimate for the branches currently carrying the work.

    Only in-flight branches can be estimated: their candidate lists have a
    known width and an observed completion rate.  Response groups the swarm
    has not opened yet have neither, and the cost model cannot supply one --
    it is keyed on (size, budget), and branches of near-identical size differ
    in cost by orders of magnitude -- so they are excluded and counted
    separately rather than guessed at.

    Only branches that completed a candidate inside the window contribute.  A
    branch sitting at 99% with no recent completions is not slow, it is
    waiting on sub-branches it already published; counting its remainder
    against a rate it is not producing inflates the estimate without bound.
    Those branches are reported as stalled instead, so excluding them hides
    nothing.

    A newly opened root has less history than the window.  Its rate therefore
    uses the observed span from its first sampled branch through `now`, rather
    than treating the unseen part of the window as idle time.  Such an estimate
    is explicitly provisional until the full window has elapsed.

    Returns None when nothing has completed inside the window, which is the
    honest answer for an idle or just-restarted root.
    """
    remaining = 0
    recent = 0
    stalled_remaining = 0
    stalled_count = 0
    sample_started_at = None
    for branch in active_branches:
        branch_remaining = max(0, branch["candidate_count"]
                                  - branch["done_candidate_count"])
        if branch["recent_done_candidate_count"]:
            remaining += branch_remaining
            recent += branch["recent_done_candidate_count"]
            created_at = branch["created_at"]
            sample_started_at = (created_at if sample_started_at is None else
                                 min(sample_started_at, created_at))
        elif branch_remaining:
            stalled_remaining += branch_remaining
            stalled_count += 1
    if not recent or not remaining:
        return None
    sample_duration_seconds = min(
        recent_window_seconds, max(1, now - sample_started_at))
    rate_per_second = recent / sample_duration_seconds
    return {
        "remaining_candidate_count": remaining,
        "recent_candidate_count": recent,
        "candidates_per_day": rate_per_second * 86400,
        "estimated_seconds": remaining / rate_per_second,
        "sample_duration_seconds": sample_duration_seconds,
        "provisional": sample_duration_seconds < recent_window_seconds,
        "stalled_branch_count": stalled_count,
        "stalled_remaining_candidate_count": stalled_remaining,
    }


# Lifecycle order: a group waits, is worked, and ends solved or proven lost.
ROOT_PROGRESS_GROUP_STATES = ("waiting", "working", "solved", "loss")


def _root_progress_group_state(row, cache_state, group_budget):
    """Where a response group sits in the work, from the cache's view.

    `solved` and `loss` both mean there is no more work to do here -- one
    because the group has a proven line, the other because it is proven
    unsolvable within the remaining budget.  Collapsing them would report a
    dead end as progress, so they stay apart.

    A group of fewer than two answers needs no search: the guess either was the
    answer, or one more guess plays the survivor -- and that guess needs a
    budget to spend, so with none left the survivor is a proven loss.  This
    mirrors `_candidate_erd_summary`, which the ERD line above the table reads.

    Queue state decides only the two remaining cases.  The cache is the
    authority on whether work is finished, because a group can be solved with
    no branch open and no finalization in this epoch.
    """
    if cache_state is not None:
        if (cache_state["best_erd"] is not None
                and cache_state["max_remaining_depth"] is not None):
            return "solved"
        if cache_state["cache_state"] == "loss":
            return "loss"
    if row["answer_count"] < 2:
        solved_by_guess = row["pattern"] == _ALL_GREEN_PATTERN_TEXT
        return "solved" if solved_by_guess or group_budget >= 1 else "loss"
    return "working" if row["started"] else "waiting"


def collect_root_progress_report(sources: ReportSources,
                                 request: ReportRequest) -> dict:
    """Work totals and a completion estimate for one root word.

    Separate from the word report because the telemetry rollup behind it is a
    seconds-scale scan: the word report stays instant and a client fetches
    this alongside it.
    """
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    resolved = resolve_branch_target(request.branch_target, all_answers)
    word = resolved.trailing_word
    if word is None:
        raise ValueError(
            "root progress report requires a branch target ending in a word")
    # Spines record words uppercase and patterns as written, so the prefix is
    # built rather than upper-cased whole: upper-casing would corrupt the y in
    # a pattern.
    spine_prefix = " ".join(
        [part for step in resolved.steps
         for part in (step.word.upper(), _normalized_pattern(step.pattern))]
        + [word.upper()]
    )
    group_budget = GAME_GUESSES - len(resolved.steps) - 1
    response_cache = ResponseCache(all_answers, score_cache=None)
    groups = response_cache.group_words(word, list(resolved.answer_words))
    group_answer_words = {fmt_pattern(pattern_code): answer_words
                          for pattern_code, answer_words in groups.items()
                          if answer_words}
    answer_counts = {pattern: len(answer_words)
                     for pattern, answer_words in group_answer_words.items()}
    # No branch context here: a root target's branch_key_hex encodes the whole
    # answer subset, which is tens of kilobytes the panel never reads.  Request
    # detail likewise stays in the sources report; only the earliest request
    # time is carried, in totals.
    data = {
        "word": word,
        "word_is_answer": word in answer_set,
        "spine_prefix": spine_prefix,
        "response_groups": [],
        "totals": {},
        "estimate": None,
        "work_started_at": None,
        "work_latest_at": None,
        "epoch": request.epoch,
    }
    report = _semantic_report(
        "root_progress", sources, request.branch_target, generated_at, data,
        request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        epoch = queue.epoch if request.epoch is None else request.epoch
        progress = queue.report_root_progress(
            spine_prefix, epoch, ROOT_PROGRESS_RECENT_WINDOW_SECONDS,
            now=generated_at)
        # Work under a deeper spine is still requested under its root, so the
        # request time is the root's however deep the target sits.
        requests = queue.source_work_requests_for_word(
            resolved.steps[0].word if resolved.steps else word)
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
        return report
    finally:
        if queue is not None:
            queue.close()

    branch_keys_by_pattern = {
        pattern: ScoreCache.encode_subset(answer_words)
        for pattern, answer_words in group_answer_words.items()
    }
    cache = None
    cache_states = {}
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        cache_states = cache.report_branch_states(
            list(branch_keys_by_pattern.values()), ERD_ALL, group_budget)
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()

    worked = progress["groups"]
    rows = []
    for pattern, answer_count in sorted(answer_counts.items()):
        totals = worked.get(pattern)
        # A group is started once any branch has opened on it, finalized or
        # not; the rollup carries open branches for exactly this reason.
        row = {
            "pattern": pattern,
            "answer_count": answer_count,
            "started": totals is not None,
            "branch_count": totals["branch_count"] if totals else 0,
            "open_branch_count": totals["open_branch_count"] if totals else 0,
            "search_node_count": totals["search_node_count"] if totals else 0,
            "wall_millis": totals["wall_millis"] if totals else 0,
            "elapsed_millis": totals["elapsed_millis"] if totals else None,
            "first_created_at": totals["first_created_at"] if totals else None,
            "last_finalized_at": (totals["last_finalized_at"]
                                  if totals else None),
        }
        row["state"] = _root_progress_group_state(
            row, cache_states.get(branch_keys_by_pattern[pattern]),
            group_budget)
        rows.append(row)
    rows.sort(key=lambda row: (-row["search_node_count"], -row["answer_count"]))
    node_total = sum(row["search_node_count"] for row in rows)
    for row in rows:
        row["search_node_share"] = (row["search_node_count"] / node_total
                                    if node_total else 0.0)
    data["response_groups"] = rows
    data["epoch"] = progress["epoch"]
    data["work_started_at"] = progress["work_started_at"]
    data["work_latest_at"] = progress["work_latest_at"]
    data["estimate"] = _root_progress_estimate(
        progress["active_branches"],
        progress["recent_window_seconds"], generated_at)
    data["active_branches"] = progress["active_branches"]
    # A request time later than the work it asked for is not a request time.
    # Rebuilding the queue's source_work rows stamps every one of them with the
    # rebuild's own clock, discarding whatever the original request time was,
    # while the branches themselves keep their true creation times.  Reporting
    # the stamp would place the request after the work it requested, so it is
    # dropped and the report shows only when work began.
    earliest_request = min((entry["requested_at"] for entry in requests),
                           default=None)
    if (earliest_request is not None
            and progress["work_started_at"] is not None
            and earliest_request > progress["work_started_at"]):
        earliest_request = None
    data["totals"] = {
        "response_group_count": len(rows),
        "started_response_group_count": sum(row["started"] for row in rows),
        "answer_count": sum(row["answer_count"] for row in rows),
        "state_counts": collections.Counter(row["state"] for row in rows),
        "branch_count": sum(row["branch_count"] for row in rows),
        "search_node_count": node_total,
        "wall_millis": sum(row["wall_millis"] for row in rows),
        "open_branch_count": progress["open_branch_count"],
        "counted_branch_count": progress["counted_branch_count"],
        "requested_at": earliest_request,
        "recent_window_seconds": progress["recent_window_seconds"],
    }
    return report


def _spine_steps(spine):
    """The guesses a spine records, as (word, pattern) steps.

    Tokens pair up as word then pattern; a trailing unpaired token records no
    guess and is dropped, so a spine of one token yields no steps at all.
    """
    tokens = (spine or "").split()
    return tuple(
        SpineStep(tokens[index].lower(), _normalized_pattern(tokens[index + 1]))
        for index in range(0, len(tokens) - 1, 2)
    )


def _steps_from_queue_row(row):
    if row.get("spine"):
        return _spine_steps(row["spine"])
    source_word = row.get("source_word")
    source_pattern = row.get("source_pattern")
    if source_word and source_pattern is not None:
        return (SpineStep(source_word.lower(), _normalized_pattern(source_pattern)),)
    return ()


def _decode_branch_key(branch_key):
    return tuple(
        branch_key[index:index + 5].decode()
        for index in range(0, len(branch_key), 5)
    )


def _summarize_claims(normalized_claims):
    """Aggregate per-candidate claims into a bounded provenance summary.

    A branch can hold tens of thousands of claims, so the human reports show
    this summary rather than a row per candidate.  Counts and per-worker
    contributions are the useful signal; the raw list stays available only to
    programmatic consumers via include_claims.
    """
    done_count = evaluated_count = provenance_unknown_count = 0
    one_level_erd_pruned_count = two_level_erd_pruned_count = 0
    in_flight_count = 0
    done_by_worker = {}
    for claim in normalized_claims:
        if claim["state"] != "done":
            in_flight_count += 1
            continue
        done_count += 1
        kind = claim["completion_kind"]
        if kind == "evaluated":
            evaluated_count += 1
        elif kind == "one_level_erd_pruned":
            one_level_erd_pruned_count += 1
        elif kind == "two_level_erd_pruned":
            two_level_erd_pruned_count += 1
        else:
            provenance_unknown_count += 1
        worker_id = claim["worker_id"]
        if worker_id:
            done_by_worker[worker_id] = done_by_worker.get(worker_id, 0) + 1
    worker_contributions = sorted(
        (
            {"worker_id": worker_id, "done_count": count}
            for worker_id, count in done_by_worker.items()
        ),
        key=lambda item: (-item["done_count"], item["worker_id"]),
    )
    return {
        "total_claim_count": len(normalized_claims),
        "done_count": done_count,
        "in_flight_count": in_flight_count,
        "evaluated_count": evaluated_count,
        "one_level_erd_pruned_count": one_level_erd_pruned_count,
        "two_level_erd_pruned_count": two_level_erd_pruned_count,
        "provenance_unknown_count": provenance_unknown_count,
        "worker_contributions": worker_contributions,
    }


def _normalize_claim(row, republish_count):
    claimed_by = _row_value(row, "claimed_by")
    done = bool(row["done"])
    one_level_erd_pruned = claimed_by in (
        "bulk-elimination", "one-level-erd-prune")
    two_level_erd_pruned = claimed_by == "two-level-erd-prune"
    sound_worker = bool(claimed_by and re.fullmatch(r"worker-[0-9]+", claimed_by))
    if done and one_level_erd_pruned:
        completion_kind = "one_level_erd_pruned"
    elif done and two_level_erd_pruned:
        completion_kind = "two_level_erd_pruned"
    elif done and sound_worker:
        completion_kind = "evaluated"
    else:
        completion_kind = None
    return {
        "candidate_index": row["idx"],
        "state": "done" if done else "in_flight",
        "completion_kind": completion_kind,
        "worker_id": claimed_by if sound_worker else None,
        "bundle_id": _row_value(row, "bundle_id"),
        "claimed_at": _row_value(row, "claimed_at"),
        "done_at": _row_value(row, "done_at"),
        "republish_count": republish_count,
        # 1-based rank in the branch's best-first candidate order, recorded
        # only for a claim the packer handed out.
        "best_first_rank": best_first_rank(
            _row_value(row, "best_first_position")),
    }


def collect_branch_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    queue = None
    pending_row = None
    active_row = None
    heartbeat_rows = []
    all_heartbeat_rows = []
    claim_rows = []
    republish_rows = []
    branch_telemetry = {
        "bundle_summary": None,
        "recent_finalizations": [],
        "finalization_total_count": 0,
        "cut_reuse_misses": [],
    }
    queue_error = None
    referenced_row = None
    cache = None
    cache_error = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
    except (sqlite3.Error, OSError) as error:
        cache_error = error
    try:
        queue = _open_report_queue(sources)
        if request.branch_target.kind == "branch_reference":
            referenced_row = resolve_branch_reference(
                queue, request.branch_target.branch_reference, cache
            )
            branch_key = bytes(referenced_row["branch_key"])
            resolved = ResolvedBranch(
                _decode_branch_key(branch_key), branch_key,
                _steps_from_queue_row(referenced_row), None,
            )
        else:
            resolved = resolve_branch_target(request.branch_target, all_answers)
            branch_key = resolved.branch_key
        pending_row = queue.get_pending_branch(branch_key)
        active_row = queue.get_active_branch(branch_key)
        all_heartbeat_rows = queue.heartbeats_with_branch()
        heartbeat_rows = [
            row for row in all_heartbeat_rows
            if row["current_branch_key"] is not None
            and bytes(row["current_branch_key"]) == branch_key
        ]
        claim_rows = list(queue.claims_for_branch(branch_key))
        republish_rows = queue.candidate_republish_for_branch(branch_key)
        cursor_after = cursor_before = None
        if request.filters.finalization_cursor_direction == "after":
            cursor_after = (
                request.filters.finalization_cursor_recorded_at,
                request.filters.finalization_cursor_id,
            )
        elif request.filters.finalization_cursor_direction == "before":
            cursor_before = (
                request.filters.finalization_cursor_recorded_at,
                request.filters.finalization_cursor_id,
            )
        branch_telemetry = queue.report_branch_telemetry(
            branch_key, request.filters.limit or 10,
            after=cursor_after, before=cursor_before,
        )
        branch_telemetry["recent_finalizations"] = [
            {
                **row,
                "spine": [
                    _step_payload(step, answer_set)
                    for step in _spine_steps(row["spine"])
                ],
                "best_guess_is_answer": _is_answer(row["best_guess"], answer_set),
            }
            for row in branch_telemetry["recent_finalizations"]
        ]
    except (sqlite3.Error, OSError) as error:
        queue_error = error
        if request.branch_target.kind == "branch_reference":
            if cache is None:
                raise
            referenced_row = resolve_branch_reference(
                None, request.branch_target.branch_reference, cache
            )
            branch_key = bytes(referenced_row["branch_key"])
            resolved = ResolvedBranch(
                _decode_branch_key(branch_key), branch_key, (), None,
            )
        else:
            resolved = resolve_branch_target(request.branch_target, all_answers)
            branch_key = resolved.branch_key
    finally:
        if queue is not None:
            queue.close()

    budget = _row_value(active_row, "budget", GAME_GUESSES - len(resolved.steps))
    branch_payload = _resolved_branch_payload(resolved, answer_set)
    branch_payload["budget"] = budget
    if request.include_answers:
        branch_payload["answer_words"] = list(resolved.answer_words)
    workers = [
        _normalize_worker(row, generated_at, answer_set) for row in heartbeat_rows
    ]
    workers.sort(key=_worker_sort_key)
    live_worker_count = sum(worker["is_live"] for worker in workers)
    initial_cache_state = _with_best_guess_is_answer(
        ScoreCache.report_branch_state_without_rows(branch_key, budget), answer_set
    )
    branch_status, branch_phase = branch_status_and_phase(
        _row_value(pending_row, "status"),
        _row_value(active_row, "status"),
        live_worker_count,
        initial_cache_state["cache_state"],
    )
    branch_payload.update({
        "branch_status": branch_status,
        "branch_phase": branch_phase,
    })
    for worker in workers:
        worker["state"] = worker_state(worker, generated_at, branch_phase)
    progress = {
        "completed_candidate_count": sum(bool(row["done"]) for row in claim_rows),
        "bulk_completed_candidate_count": _row_value(
            active_row, "bulk_done_candidates", 0
        ),
        "one_level_erd_pruned_candidate_count": _row_value(
            active_row, "one_level_erd_pruned_candidates", 0
        ),
        "two_level_erd_pruned_candidate_count": _row_value(
            active_row, "two_level_erd_pruned_candidates", 0
        ),
    }
    queue_payload = None
    if active_row is not None or pending_row is not None:
        queue_payload = {
            "branch_status": branch_status,
            "branch_phase": branch_phase,
            "pending_status": _row_value(pending_row, "status"),
            "active_status": _row_value(active_row, "status"),
            "priority": _row_value(
                active_row, "effective_priority",
                _row_value(pending_row, "priority"),
            ),
            "candidate_count": _row_value(active_row, "n_candidates"),
            "completed_candidate_count": progress["completed_candidate_count"],
            "bulk_completed_candidate_count": progress["bulk_completed_candidate_count"],
            "one_level_erd_pruned_candidate_count": progress[
                "one_level_erd_pruned_candidate_count"
            ],
            "two_level_erd_pruned_candidate_count": progress[
                "two_level_erd_pruned_candidate_count"
            ],
            "best_guess": _row_value(active_row, "best_guess"),
            "best_guess_is_answer": bool(
                _row_value(active_row, "best_guess")
                and str(_row_value(active_row, "best_guess")).lower() in answer_set
            ),
            "best_erd": _row_value(active_row, "best_erd"),
            "best_max_remaining_depth": _row_value(active_row, "best_max_depth"),
            "ceiling": _row_value(active_row, "ceiling"),
            "budget": _row_value(active_row, "budget"),
            "search_node_count": _row_value(active_row, "nodes_spent", 0),
            "pack_cursor": _row_value(active_row, "pack_cursor"),
            "tainted": bool(_row_value(active_row, "tainted", False)),
            "created_at": _row_value(active_row, "created_at"),
            "finalized_at": _row_value(active_row, "finalized_at"),
            "completed_at": _row_value(pending_row, "completed_at"),
        }
    republish_by_index = {row["idx"]: row["count"] for row in republish_rows}
    normalized_claims = [
        _normalize_claim(row, republish_by_index.get(row["idx"], 0))
        for row in claim_rows
    ]
    all_workers = [
        _normalize_worker(row, generated_at, answer_set)
        for row in all_heartbeat_rows
    ]
    ownership = _branch_ownership(
        workers, all_workers, normalized_claims, branch_key.hex()
    )
    provenance_unknown = any(
        claim["state"] == "done" and claim["completion_kind"] is None
        for claim in normalized_claims
    )
    data = {
        "branch": branch_payload,
        "queue": queue_payload,
        "cache": initial_cache_state,
        "workers": workers,
        "republished_candidates": [
            {"candidate_index": row["idx"], "republish_count": row["count"]}
            for row in republish_rows
        ],
        "completed_candidate_indexes": sorted(
            row["idx"] for row in claim_rows if row["done"]
        ),
        "claims": normalized_claims if request.include_claims else None,
        "claim_summary": _summarize_claims(normalized_claims),
        "branch_ownership": ownership,
        "provenance_unknown": provenance_unknown,
        **branch_telemetry,
    }
    report = _semantic_report(
        "branch", sources, request.branch_target, generated_at, data, request
    )
    if queue_error is None:
        _mark_queue_source_ok(report)
    else:
        _mark_queue_source_error(report, queue_error)

    try:
        if cache is None:
            raise cache_error
        data["cache"] = _with_best_guess_is_answer(
            cache.report_branch_state(branch_key, ERD_ALL, budget), answer_set
        )
        branch_status, branch_phase = branch_status_and_phase(
            _row_value(pending_row, "status"),
            _row_value(active_row, "status"),
            live_worker_count,
            data["cache"]["cache_state"],
        )
        data["branch"].update({
            "branch_status": branch_status,
            "branch_phase": branch_phase,
        })
        if data["queue"] is not None:
            data["queue"].update({
                "branch_status": branch_status,
                "branch_phase": branch_phase,
            })
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()
    return report


def _branch_target_queue_scope(branch_target, queue=None):
    if branch_target.kind == "root":
        return {}, ""
    if branch_target.kind == "branch_reference":
        if queue is None:
            return {}, ""
        row = resolve_branch_reference(queue, branch_target.branch_reference)
        prefix = row.get("spine") or ""
        scope = {"branch_key": bytes(row["branch_key"])}
        if prefix:
            scope["spine_prefix"] = prefix
        return scope, prefix
    tokens = []
    for step in branch_target.steps:
        tokens.extend((step.word.upper(), step.pattern))
    if branch_target.kind == "word":
        tokens.append(branch_target.trailing_word.upper())
    prefix = " ".join(tokens)
    scope = {"spine_prefix": prefix}
    if branch_target.kind == "word" and not branch_target.steps:
        scope["source_word"] = branch_target.trailing_word
    return scope, prefix


def _row_matches_branch_target(row, branch_target, prefix):
    if branch_target.kind == "root":
        return True
    spine = row.get("spine") or ""
    if branch_target.kind == "word":
        if spine:
            return spine.startswith(prefix + " ")
        return (
            len(branch_target.steps) == 0
            and (row.get("source_word") or "").lower() == branch_target.trailing_word
        )
    if prefix and spine:
        return spine == prefix or spine.startswith(prefix + " ")
    if branch_target.kind == "branch_reference":
        return branch_reference(bytes(row["branch_key"])).startswith(
            branch_target.branch_reference
        )
    return False


def _collection_summary(rows):
    by_status = {}
    by_phase = {}
    for row in rows:
        status = row["branch_status"]
        phase = row["branch_phase"]
        by_status[status] = by_status.get(status, 0) + 1
        by_phase[phase] = by_phase.get(phase, 0) + 1
    return {
        "branch_count": len(rows),
        "branch_count_by_status": by_status,
        "branch_count_by_phase": by_phase,
    }


def _scoped_queue_rows(queue, request, apply_filters=True):
    filters = request.filters if apply_filters else ReportFilters()
    unbounded_filters = replace(filters, limit=None)
    scope, prefix = _branch_target_queue_scope(request.branch_target, queue)
    query_filters = _filters_payload(unbounded_filters)
    query_filters.update(scope)
    if request.tree_parent:
        query_filters["spine_prefix"] = request.tree_parent
    result = queue.report_queue_rows(query_filters)
    return result["rows"], prefix


def collect_queue_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    answer_set = _decorative_answer_set(sources)
    data = {"summary": {}, "matched_rows": 0, "rows": []}
    report = _semantic_report(
        "queue", sources, request.branch_target, generated_at, data, request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        rows, _prefix = _scoped_queue_rows(queue, request)
        data["summary"] = _collection_summary(rows)
        data["matched_rows"] = len(rows)
        limit = request.filters.limit
        data["rows"] = rows[:limit] if limit is not None else rows
        for row in data["rows"]:
            row["branch_reference"] = branch_reference(bytes(row.pop("branch_key")))
            row["spine"] = _normalized_branch_spine(row, answer_set)
            row["best_guess_is_answer"] = _is_answer(row.get("best_guess"), answer_set)
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


def _tree_node_id(steps):
    return "/".join(f"{step.word}:{step.pattern}" for step in steps)


def _tree_layout(rows, request, prefix, unfiltered_rows, answer_set):
    if not rows:
        return {
            "root": _branch_target_payload(request.branch_target),
            "topology_source": "queue",
            "tree_available": False,
            "unavailable_reason": "no extant queue topology",
            "nodes": [],
        }

    # A node is a played guess.  The response includes one page of immediate
    # children; their descendants are fetched only when a node is expanded.
    nodes = {}
    for row in rows:
        steps = _steps_from_queue_row(row)
        if steps:
            parent_id = None
            for guess_depth_value in range(1, len(steps) + 1):
                step = steps[guess_depth_value - 1]
                node_id = _tree_node_id(steps[:guess_depth_value])
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "step": _step_payload(step, answer_set),
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "branch_status": None,
                    "branch_phase": None,
                    "answer_count": None,
                    "guess_depth": guess_depth_value,
                    "worker_count": 0,
                    "subtree_worker_count": 0,
                    "child_count": 0,
                    "priority": None,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                    "spine": " ".join(
                        value for spine_step in steps[:guess_depth_value]
                        for value in (spine_step.word.upper(), spine_step.pattern)
                    ),
                })
                parent_id = node_id
            final_node = nodes[parent_id]
        else:
            guess_depth = (
                GAME_GUESSES - row["budget"] if row.get("budget") is not None else 1
            )
            parent_id = None
            for guess_depth_value in range(1, max(guess_depth, 1) + 1):
                node_id = f"unknown:{guess_depth_value}:{row['branch_key_hex']}"
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "step": None,
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "branch_status": None,
                    "branch_phase": None,
                    "answer_count": None,
                    "guess_depth": guess_depth_value,
                    "worker_count": 0,
                    "subtree_worker_count": 0,
                    "child_count": 0,
                    "priority": None,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                    "spine": None,
                })
                parent_id = node_id
            final_node = nodes[parent_id]
        final_node.update({
            "branch_key_hex": row["branch_key_hex"],
            "branch_reference": branch_reference(bytes(row["branch_key"])),
            "branch_status": row["branch_status"],
            "branch_phase": row["branch_phase"],
            "answer_count": row["answer_count"],
            "worker_count": row["worker_count"],
            "priority": row["priority"],
            "completed_candidate_count": row["completed_candidate_count"],
            "candidate_count": row["candidate_count"],
            "is_context": bool(row.get("is_context")),
        })
    parent_node_ids = {
        candidate["parent_node_id"] for candidate in nodes.values()
        if candidate["parent_node_id"] is not None
    }
    for node in nodes.values():
        node["has_children"] = node["node_id"] in parent_node_ids
        node["child_count"] = sum(
            candidate["parent_node_id"] == node["node_id"]
            for candidate in nodes.values()
        )
    for node in nodes.values():
        worker_count = node["worker_count"]
        current_node = node
        while current_node is not None:
            current_node["subtree_worker_count"] += worker_count
            parent_node_id = current_node["parent_node_id"]
            current_node = nodes.get(parent_node_id)
    parent_spine = request.tree_parent or prefix
    parent_id = _tree_node_id(_spine_steps(parent_spine)) if parent_spine else None
    direct_nodes = [
        node for node in nodes.values() if node["parent_node_id"] == parent_id
    ]
    grouped_nodes = {}
    for node in sorted(
        direct_nodes,
        key=lambda node: (
            (node["step"] or {}).get("word", ""),
            (node["step"] or {}).get("pattern", ""),
            node["branch_key_hex"] or "",
        ),
    ):
        group_key = (node["step"] or {}).get("word") or node["node_id"]
        grouped_nodes.setdefault(group_key, []).append(node)
    group_keys = list(grouped_nodes)
    if request.tree_cursor is not None:
        group_keys = [key for key in group_keys if key > request.tree_cursor]
    page_size = request.filters.limit or DEFAULT_TREE_PAGE_SIZE
    page_keys = group_keys[:page_size]
    page_nodes = [node for key in page_keys for node in grouped_nodes[key]]
    next_cursor = page_keys[-1] if len(group_keys) > page_size else None
    return {
        "root": _branch_target_payload(request.branch_target),
        "topology_source": "queue",
        "tree_available": True,
        "unavailable_reason": None,
        "nodes": page_nodes,
        "paging": {
            "parent_spine": parent_spine,
            "cursor": request.tree_cursor,
            "page_size": page_size,
            "returned_group_count": len(page_keys),
            "total_group_count": len(grouped_nodes),
            "next_cursor": next_cursor,
        },
    }


def collect_tree_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    answer_set = _decorative_answer_set(sources)
    inferred_kind = request.report_kind
    if inferred_kind == "auto":
        inferred_kind = "queue" if request.branch_target.kind == "root" else request.branch_target.kind
        if inferred_kind == "branch_reference":
            inferred_kind = "branch"
    data = {
        "root": _branch_target_payload(request.branch_target),
        "topology_source": "queue",
        "tree_available": False,
        "unavailable_reason": "no extant queue topology",
        "nodes": [],
    }
    report = _semantic_report(
        inferred_kind, sources, request.branch_target, generated_at, data, request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        filtered_rows, prefix = _scoped_queue_rows(queue, request, True)
        unfiltered_rows = filtered_rows
        if request.branch_target.kind in ("branch", "branch_reference"):
            unfiltered_rows, _ = _scoped_queue_rows(queue, request, False)
        if request.worker_id is not None:
            selected_branch_keys = {
                bytes(row["current_branch_key"]).hex()
                for row in queue.heartbeats_with_branch()
                if row["current_branch_key"] is not None
                and (
                    row["worker_id"] == request.worker_id
                    or _worker_number(row["worker_id"]) == request.worker_id
                )
            }
            filtered_rows = [
                row for row in filtered_rows
                if row["branch_key_hex"] in selected_branch_keys
            ]
            unfiltered_rows = [
                row for row in unfiltered_rows
                if row["branch_key_hex"] in selected_branch_keys
            ]
        data.update(_tree_layout(
            list(filtered_rows), request, prefix, unfiltered_rows, answer_set
        ))
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


def collect_workers_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    data = {"summary": {}, "matched_rows": 0, "rows": []}
    report = _semantic_report(
        "workers", sources, request.branch_target, generated_at, data, request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        filters_have_branch_scope = any((
            request.filters.branch_statuses,
            request.filters.branch_phases,
            request.filters.minimum_answer_count is not None,
            request.filters.maximum_answer_count is not None,
            request.filters.budget is not None,
            request.filters.priority is not None,
        ))
        if request.branch_target.kind == "root" and not filters_have_branch_scope:
            scoped_rows = queue.report_queue_rows({
                "branch_statuses": ("active",),
            })["rows"]
        else:
            scoped_rows, _prefix = _scoped_queue_rows(queue, request)
        scoped_keys = {row["branch_key_hex"] for row in scoped_rows}
        workers = [
            _normalize_worker(row, generated_at, answer_set)
            for row in queue.heartbeats_with_branch()
        ]
        if request.branch_target.kind != "root" or filters_have_branch_scope:
            workers = [
                worker for worker in workers
                if worker["branch_key_hex"] in scoped_keys
            ]
        if request.worker_id is not None:
            workers = [
                worker for worker in workers
                if worker["worker_id"] == request.worker_id
                or worker["worker_number"] == request.worker_id
            ]
        phase_by_key = {
            row["branch_key_hex"]: row["branch_phase"] for row in scoped_rows
        }
        by_state = {
            "working": 0,
            "coordinating": 0,
            "idle": 0,
            "transitioning": 0,
            "finalizing": 0,
            "stale": 0,
            "dead": 0,
        }
        for worker in workers:
            state = worker_state(
                worker, generated_at, phase_by_key.get(worker["branch_key_hex"])
            )
            worker["state"] = state
            by_state[state] += 1
        workers.sort(key=_worker_sort_key)
        data["summary"] = {
            "worker_count": len(workers),
            "worker_count_by_state": by_state,
        }
        data["matched_rows"] = len(workers)
        limit = request.filters.limit
        data["rows"] = workers[:limit] if limit is not None else workers
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


# One vocabulary's skeletons at a time.  For the full vocabulary this holds
# ~0.25 GB — every candidate materializes a branch key per response group — so a
# growing cache in a long-lived report server is a liability; only the most
# recent vocabulary is retained.
_candidate_skeleton_memo = None


def _candidate_group_skeletons(sources, all_answers, all_candidates, cache):
    """Per-candidate top-level response groups as (pattern, count, branch_key).

    Partitioning every candidate against the answer list is the expensive part
    of a leaderboard build (~9s for the full vocabulary) and depends only on the
    vocabulary, not the cache, so it is memoized and reused across builds.  The
    skeletons are large (~0.25 GB for the full vocabulary), so only the most
    recent vocabulary is kept, and the 243 distinct pattern strings are shared
    rather than reformatted per group.  Keyed on the list files' paths and
    mtimes, so a changed list rebuilds without re-hashing the vocabulary.
    """
    global _candidate_skeleton_memo
    memo_key = (
        sources.answer_list_path, os.path.getmtime(sources.answer_list_path),
        sources.candidate_list_path,
        os.path.getmtime(sources.candidate_list_path),
    )
    if _candidate_skeleton_memo is not None and _candidate_skeleton_memo[0] == memo_key:
        return _candidate_skeleton_memo[1]
    matrix = PatternMatrix.load_or_build(
        sources.cache_path, all_candidates, all_answers, cache
    )
    branch_indices = matrix.answer_indices(all_answers)
    branch_words = list(all_answers)
    pattern_text = {code: fmt_pattern(code) for code in range(3 ** 5)}
    skeletons = [
        (
            candidate,
            [
                (pattern_text[pattern], len(words), ScoreCache.encode_subset(words))
                for pattern, words in matrix.group_words(
                    candidate, branch_words, branch_indices
                ).items()
                if words
            ],
        )
        for candidate in all_candidates
    ]
    _candidate_skeleton_memo = (memo_key, skeletons)
    return skeletons


def collect_leaderboard_report(sources: ReportSources, request: ReportRequest) -> dict:
    """Rank every candidate opener by its own ERD.

    Each candidate's ERD is resolved exactly as the word report resolves it
    (`_resolved_candidate_erd`), reusing the cache's reusability gate so the
    numbers agree with `view WORD`.  Only openers whose whole tree is solved
    have a finite ERD and appear ranked; the rest are summarized as pending or
    infeasible.  A candidate found complete keeps its folded ERD, so a later
    build reads one row per solved opener instead of re-folding its response
    groups; an unsolved candidate is re-folded from current cache state every
    time, since its value can still change.
    """
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    all_candidates = load_word_list(sources.candidate_list_path)
    answer_set = set(all_answers)
    data = {"response_pattern_count": 3 ** 5}
    report = _semantic_report(
        "leaderboard", sources, request.branch_target, generated_at, data, request
    )
    group_budget = GAME_GUESSES - 1
    limit = request.filters.limit
    cache = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        skeletons = _candidate_group_skeletons(
            sources, all_answers, all_candidates, cache
        )
        data["maximum_response_group_count"] = max(
            (len(groups) for _, groups in skeletons),
            default=0,
        )
        exact_by_key, loss_by_key = cache.report_branch_row_maps(ERD_ALL)
        stored_erd_map = cache.candidate_erd_map(ERD_ALL)
        # Every candidate here is an opener, folded against the root branch.
        root_branch_key = ScoreCache.encode_subset(all_answers)
        counts = {"complete": 0, "pending": 0, "infeasible": 0}
        ranked_rows = []
        for candidate, groups in skeletons:
            branch_keys = [branch_key for _, _, branch_key in groups]
            states = cache.report_branch_states_from_maps(
                branch_keys, exact_by_key, loss_by_key, group_budget
            )
            response_groups = [
                {
                    "pattern": pattern,
                    "answer_count": answer_count,
                    "best_erd": states[branch_key]["best_erd"],
                    "max_remaining_depth": states[branch_key]["max_remaining_depth"],
                    "cache_state": states[branch_key]["cache_state"],
                }
                for pattern, answer_count, branch_key in groups
            ]
            summary = _resolved_candidate_erd(
                cache, root_branch_key, candidate, ERD_ALL,
                response_groups, group_budget, stored_erd_map,
            )
            counts[summary["state"]] += 1
            if summary["state"] == "complete":
                ranked_rows.append({
                    "word": candidate,
                    "word_is_answer": candidate in answer_set,
                    "erd": summary["erd"],
                    "max_remaining_depth": summary["max_remaining_depth"],
                    "_response_group_skeletons": groups,
                })
        ranked_rows.sort(
            key=lambda row: (row["erd"], row["max_remaining_depth"], row["word"])
        )
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank
        displayed_rows = (
            ranked_rows[:limit] if limit is not None else ranked_rows
        )
        for row in displayed_rows:
            response_group_skeletons = row.pop("_response_group_skeletons")
            row["answer_count"] = sum(
                answer_count
                for _, answer_count, _ in response_group_skeletons
            )
            row["response_groups"] = [
                {
                    "pattern": pattern,
                    "answer_count": answer_count,
                }
                for pattern, answer_count, _ in sorted(
                    response_group_skeletons,
                    key=lambda group: group[1],
                    reverse=True,
                )
            ]
        # Publish only after the whole vocabulary is folded.  A mid-loop cache
        # error must not leave a truncated ranking that reads as complete.
        data.update({
            "candidate_count": len(all_candidates),
            "counts": counts,
            "total_rows": len(ranked_rows),
            "matched_rows": len(ranked_rows),
            "rows": displayed_rows,
        })
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
        data.update({
            "candidate_count": len(all_candidates),
            "counts": {"complete": 0, "pending": 0, "infeasible": 0},
            "total_rows": 0,
            "matched_rows": 0,
            "rows": [],
        })
    finally:
        if cache is not None:
            cache.close()
    return report


def collect_cache_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    data = {}
    report = _semantic_report(
        "cache", sources, request.branch_target, generated_at, data, request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    cache = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        branch_target = request.branch_target
        if branch_target.kind == "root":
            limit = request.filters.limit or 50
            recent_rows = cache.report_recent_rows(
                ERD_ALL, generated_at - 300, limit
            )
            data.update({
                "summary": cache.erd_report_summary(ERD_ALL, generated_at - 300),
                "distributions": cache.report_cache_distributions(ERD_ALL),
                "recent_rows": [
                    {
                        **_with_best_guess_is_answer(
                            {key: value for key, value in row.items() if key != "branch_key"},
                            answer_set,
                        ),
                        "branch_key_hex": row["branch_key"].hex(),
                        "branch_reference": branch_reference(row["branch_key"]),
                    }
                    for row in recent_rows
                ],
            })
        elif branch_target.kind == "word":
            resolved = resolve_branch_target(branch_target, all_answers)
            groups = ResponseCache(all_answers, score_cache=None).group_words(
                branch_target.trailing_word, list(resolved.answer_words)
            )
            keys = [ScoreCache.encode_subset(words) for words in groups.values() if words]
            states = cache.report_branch_states(
                keys, ERD_ALL, GAME_GUESSES - len(branch_target.steps) - 1
            )
            rows = []
            for pattern_code, words in sorted(groups.items()):
                if not words:
                    continue
                key = ScoreCache.encode_subset(words)
                rows.append({
                    "pattern": fmt_pattern(pattern_code),
                    "answer_count": len(words),
                    "branch_key_hex": key.hex(),
                    "branch_reference": branch_reference(key),
                    **_with_best_guess_is_answer(states[key], answer_set),
                })
            data.update({"summary": {"response_group_count": len(rows)}, "rows": rows})
        else:
            if branch_target.kind == "branch_reference":
                if queue is None:
                    raise ValueError("queue unavailable for branch reference")
                referenced_row = resolve_branch_reference(
                    queue, branch_target.branch_reference
                )
                branch_key = bytes(referenced_row["branch_key"])
                steps = _steps_from_queue_row(referenced_row)
                budget = _row_value(
                    referenced_row, "budget", GAME_GUESSES - len(steps)
                )
            else:
                resolved = resolve_branch_target(branch_target, all_answers)
                branch_key = resolved.branch_key
                steps = resolved.steps
                budget = GAME_GUESSES - len(steps)
            data.update({
                "branch_key_hex": branch_key.hex(),
                "branch_reference": branch_reference(branch_key),
                "cache": _with_best_guess_is_answer(
                    cache.report_branch_state(branch_key, ERD_ALL, budget),
                    answer_set,
                ),
            })
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()
        if queue is not None:
            queue.close()
    return report


def collect_hotspot_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    answer_set = _decorative_answer_set(sources)
    field = request.hotspot_field or "nodes"
    since_seconds = request.since_seconds or 3600
    sample_size = min(request.sample_size or 50_000, 1_000_000)
    limit = request.filters.limit or 10
    data = {
        "field": field,
        "population": None,
        "epoch": request.epoch,
        "since_seconds": since_seconds,
        "sample_size": sample_size,
        "sampled_row_count": 0,
        "sample_truncated": False,
        "rows": [],
    }
    report = _semantic_report(
        "hotspots", sources, request.branch_target, generated_at, data, request
    )
    queue = None
    try:
        queue = _open_report_queue(sources)
        epoch = queue.epoch if request.epoch is None else request.epoch
        current_fields = {"nodes", "age", "size", "workers", "priority", "slowest"}
        if field in current_fields:
            hotspot_filters = replace(
                request.filters, sort=field, limit=None
            )
            hotspot_request = replace(request, filters=hotspot_filters)
            rows, _prefix = _scoped_queue_rows(queue, hotspot_request)
            result = {
                "population": "current_queue_branches",
                "epoch": epoch,
                "since": generated_at - since_seconds,
                "sample_size": None,
                "sampled_row_count": len(rows),
                "sample_truncated": len(rows) > limit,
                "rows": rows[:limit],
            }
        else:
            scope, prefix = _branch_target_queue_scope(request.branch_target, queue)
            branch_key = scope.get("branch_key")
            if field == "cut-reuse" and request.branch_target.kind == "branch":
                branch_key = resolve_branch_target(
                    request.branch_target, sorted(answer_set)
                ).branch_key
            result = queue.report_hotspots(
                field, epoch, generated_at - since_seconds,
                sample_size, limit, prefix or None, branch_key,
            )
        normalized_rows = []
        for row in result["rows"]:
            normalized = _with_best_guess_is_answer(dict(row), answer_set)
            branch_key = normalized.pop("branch_key", None)
            if branch_key is not None:
                normalized["branch_key_hex"] = bytes(branch_key).hex()
                normalized["branch_reference"] = branch_reference(bytes(branch_key))
            normalized_rows.append(normalized)
        data.update({
            "population": result["population"],
            "epoch": result["epoch"],
            "window_started_at": result["since"],
            "sample_size": result["sample_size"],
            "sampled_row_count": result["sampled_row_count"],
            "sample_truncated": result["sample_truncated"],
            "rows": normalized_rows,
        })
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


def _source_summary_payload(row, rollup, timing):
    """One source word, with its requests and branches rolled up.

    This is the report's unit: a word that spawned a thousand branches is one
    row, not a thousand, and a word requested more than once is still one row.
    direct_branch_count and branch_count span every branch the word's requests
    have ever owned; the rollup counts only those still owned live, so a branch
    that finalized and released its ownership reads as done.

    direct_branch_count is the branches the word asked for outright — its own
    response groups.  A branch the search promoted under that word's ownership
    later counts only in branch_count, so the two are equal until promotion
    starts.
    """
    source_word = _row_value(row, "source_word")
    branch_count = row["branch_count"] or 0
    open_branch_count = rollup["open_branch_count"]
    state = _merged_source_state(row)
    completed_at = None
    elapsed_millis = worker_millis = None
    if state == "complete":
        completed_at = timing["completed_at"]
        if completed_at is None:
            completed_at = _row_value(row, "completed_at")
        elapsed_millis = timing["elapsed_millis"]
        worker_millis = timing["worker_millis"]
    return {
        "source_word": source_word.lower() if source_word else None,
        "requested_priority": row["requested_priority"],
        "requested_at": _row_value(row, "requested_at"),
        "completed_at": completed_at,
        "request_count": row["request_count"] or 0,
        "state": state,
        "direct_branch_count": row["direct_branch_count"] or 0,
        "direct_done_branch_count": row["direct_done_branch_count"] or 0,
        "branch_count": branch_count,
        "open_branch_count": open_branch_count,
        "done_branch_count": max(0, branch_count - open_branch_count),
        "worker_count": rollup["worker_count"],
        "elapsed_millis": elapsed_millis,
        "worker_millis": worker_millis,
    }


def _merged_source_state(row):
    """The state of a word whose requests may disagree.

    A word is only complete once every one of its requests is, and reads as
    active while any request is being worked.
    """
    if not _row_value(row, "has_incomplete_request", 0):
        return "complete"
    return "active" if _row_value(row, "has_active_request", 0) else "queued"

def _source_erd_sort_key(row):
    summary = row.get("erd_summary")
    if not summary:
        return (3, float("inf"), row["source_word"] or "")
    if summary["state"] == "complete":
        return (0, summary["erd"], row["source_word"] or "")
    if summary["state"] == "pending":
        return (1, float("inf"), row["source_word"] or "")
    return (2, float("inf"), row["source_word"] or "")


_SOURCE_SORT_KEYS = {
    "default": _source_erd_sort_key,
    "age": lambda row: (row["requested_at"] or float("inf"),
                        row["source_word"] or ""),
    "word": lambda row: (row["source_word"] or "",),
    "priority": lambda row: (-(row["requested_priority"] or 0),
                             row["source_word"] or ""),
    "branches": lambda row: (-row["branch_count"], row["source_word"] or ""),
    "open": lambda row: (-row["open_branch_count"], row["source_word"] or ""),
    "done": lambda row: (-row["done_branch_count"], row["source_word"] or ""),
    "workers": lambda row: (-row["worker_count"], row["source_word"] or ""),
    "completed": lambda row: (-(row["completed_at"] or 0)
                               if row["completed_at"] is not None else float("inf"),
                               row["source_word"] or ""),
    "requested": lambda row: (-(row["requested_at"] or 0)
                               if row["requested_at"] is not None else float("inf"),
                               row["source_word"] or ""),
    "erd": _source_erd_sort_key,
    "elapsed": lambda row: (-(row["elapsed_millis"] or 0)
                            if row["elapsed_millis"] is not None else float("inf"),
                            row["source_word"] or ""),
    "worker_time": lambda row: (-(row["worker_millis"] or 0)
                                if row["worker_millis"] is not None else float("inf"),
                                row["source_word"] or ""),
}


def _sorted_source_words(rows, sort):
    """Order collapsed source-word rows by the requested source-report field."""
    key = _SOURCE_SORT_KEYS.get(sort or "default")
    return sorted(rows, key=key) if key is not None else rows


def _duration_group_key(duration_millis, missing_label):
    """Bucket a non-negative duration into the Sources report's time ranges."""
    if duration_millis is None:
        return 5, missing_label
    for index, (maximum_millis, label) in enumerate((
        (60 * 60 * 1000, "0–1 hour"),
        (24 * 60 * 60 * 1000, "under 24 hours"),
        (7 * 24 * 60 * 60 * 1000, "under 1 week"),
        (30 * 24 * 60 * 60 * 1000, "under 1 month"),
    )):
        if duration_millis <= maximum_millis:
            return index, label
    return 4, "1 month or more"


def _completed_at_group_key(completed_at, generated_at):
    """Bucket a completion timestamp by its local calendar date."""
    if completed_at is None:
        return 5, "not completed"
    today = datetime.datetime.fromtimestamp(generated_at).date()
    completed_date = datetime.datetime.fromtimestamp(completed_at).date()
    if completed_date >= today:
        return 0, "today"
    week_start = today - datetime.timedelta(days=today.weekday())
    if completed_date >= week_start:
        return 1, "earlier this week"
    if completed_date.year == today.year and completed_date.month == today.month:
        return 2, "earlier this month"
    if completed_date.year == today.year:
        return 3, "earlier this year"
    return 4, "older"


def _source_word_group_key(row, group_by, generated_at):
    """Map a collapsed source row to (sort_key, label) for the strategy."""
    if group_by == "state":
        state = row["state"]
        return (_SOURCE_STATE_GROUP_ORDER.get(state, 9), state)
    if group_by == "worker_presence":
        return ((0, "with workers") if row["worker_count"]
                else (1, "no workers"))
    if group_by == "completed":
        return _completed_at_group_key(row["completed_at"], generated_at)
    if group_by == "elapsed":
        return _duration_group_key(row["elapsed_millis"], "not completed")
    if group_by == "worker_time":
        return _duration_group_key(row["worker_millis"], "not completed")
    if group_by == "requested":
        requested_at = row["requested_at"]
        duration_millis = (
            max(0, generated_at - requested_at) * 1000
            if requested_at is not None else None
        )
        return _duration_group_key(duration_millis, "not requested")
    priority = row["requested_priority"]
    return (-(priority or 0), f"priority {priority}")


def _grouped_source_words(rows, group_by, branch_totals, generated_at):
    """Bucket the collapsed rows, each group carrying its own rollup.

    A group's branch totals are counted distinctly over the words in it, for
    the same reason the report's own are: two words can own the same branch,
    and summing their per-word counts would count it once per word.
    """
    grouped = {}
    for row in rows:
        sort_key, label = _source_word_group_key(row, group_by, generated_at)
        group = grouped.setdefault(
            (sort_key, label),
            {"label": label, "rows": [],
             "rollup": {"source_word_count": 0, "branch_count": 0,
                        "open_branch_count": 0, "done_branch_count": 0,
                        "worker_count": 0}},
        )
        group["rows"].append(row)
        rollup = group["rollup"]
        rollup["source_word_count"] += 1
        rollup["worker_count"] += row["worker_count"]
    for group in grouped.values():
        branch_count, open_branch_count = branch_totals(
            [row["source_word"] for row in group["rows"]]
        )
        group["rollup"]["branch_count"] = branch_count
        group["rollup"]["open_branch_count"] = open_branch_count
        group["rollup"]["done_branch_count"] = max(
            0, branch_count - open_branch_count
        )
    return [grouped[key] for key in sorted(grouped)]


def _source_word_erd_summaries(sources, source_words, report, cache,
                               all_answers):
    """Resolve each word's own ERD, reusing folds until the cache changes.

    A source word's ERD is the whole point of queueing it.  It is derived from
    the cached result of each of the word's response groups, the same way the
    word report derives it for one word, and a word whose whole tree is solved
    keeps its folded value in `candidate_erd_by_policy` so later pages read one
    row instead of re-folding ~150 groups.  ERD order requires every visible
    source word's summary; unchanged cache files reuse the previous fold.
    """
    summaries = {}
    if not source_words:
        return summaries
    try:
        response_cache = ResponseCache(all_answers, score_cache=None)
        cache_version = _score_cache_file_signature(sources.cache_path)
        cache_identity = (sources.cache_path, cache.answer_list_id)
        cached_version, cached_summaries = _SOURCE_ERD_SUMMARY_CACHE.get(
            cache_identity, (None, {})
        )
        if cached_version != cache_version:
            cached_summaries = {}
            _SOURCE_ERD_SUMMARY_CACHE[cache_identity] = (
                cache_version, cached_summaries)
        # A root word spends the first guess, leaving the rest for its groups.
        group_budget = GAME_GUESSES - 1
        # Source words are always openers, so every one folds against the same
        # branch: the whole answer list, before any guess is played.
        root_branch_key = ScoreCache.encode_subset(all_answers)
        for word in source_words:
            if word is None:
                continue
            cached_summary = cached_summaries.get(word)
            if cached_summary is not None:
                summaries[word] = cached_summary
                continue
            groups = response_cache.group_words(word, all_answers)
            group_rows = []
            branch_keys = []
            for pattern_code, answer_words in sorted(groups.items()):
                if not answer_words:
                    continue
                branch_key = ScoreCache.encode_subset(answer_words)
                branch_keys.append(branch_key)
                group_rows.append({
                    "pattern": fmt_pattern(pattern_code),
                    "answer_count": len(answer_words),
                    "branch_key": branch_key,
                })
            states = cache.report_branch_states(
                branch_keys, ERD_ALL, group_budget
            )
            summary = _resolved_candidate_erd(
                cache, root_branch_key, word, ERD_ALL,
                [
                    {
                        "pattern": row["pattern"],
                        "answer_count": row["answer_count"],
                        "best_erd": states[bytes(row["branch_key"])]["best_erd"],
                        "max_remaining_depth":
                            states[bytes(row["branch_key"])]["max_remaining_depth"],
                        "cache_state":
                            states[bytes(row["branch_key"])]["cache_state"],
                    }
                    for row in group_rows
                ],
                group_budget,
            )
            cached_summaries[word] = summary
            summaries[word] = summary
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    return summaries


def _score_cache_file_signature(cache_path):
    """Identify the cache generation, including uncheckpointed WAL writes."""
    def file_signature(path):
        try:
            stat_result = os.stat(path)
        except OSError:
            return None
        return stat_result.st_mtime_ns, stat_result.st_size

    return file_signature(cache_path), file_signature(cache_path + "-wal")


def _source_rollups(membership_rows):
    """Per-word live-branch totals, keyed by source word.

    A branch owned by two of a word's requests holds two membership rows, so
    branches are counted once per word rather than once per membership.
    """
    rollups = collections.defaultdict(
        lambda: {"open_branch_count": 0, "worker_count": 0, "branch_ids": set()}
    )
    for row in membership_rows:
        source_word = (_row_value(row, "source_word") or "").lower() or None
        rollup = rollups[source_word]
        if row["branch_id"] in rollup["branch_ids"]:
            continue
        rollup["branch_ids"].add(row["branch_id"])
        worker_count = _row_value(row, "worker_count", 0)
        _branch_status, branch_phase = branch_status_and_phase(
            _row_value(row, "pending_status"), _row_value(row, "active_status"),
            worker_count,
        )
        if branch_phase != "complete":
            rollup["open_branch_count"] += 1
        rollup["worker_count"] += worker_count
    return rollups


def _source_membership_payload(row, owner_count):
    branch_key = bytes(row["branch_key"])
    parent_branch_key = _row_value(row, "parent_branch_key")
    source_word = _row_value(row, "source_word")
    worker_count = _row_value(row, "worker_count", 0)
    # cache_state is omitted (unlike the queue/word/branch reports): a shared
    # branch's cache freshness is answer-set-wide, not per-owner, and adding a
    # ScoreCache lookup per membership row here would duplicate what the
    # queue/branch reports already answer more precisely for one branch.
    branch_status, branch_phase = branch_status_and_phase(
        _row_value(row, "pending_status"), _row_value(row, "active_status"),
        worker_count,
    )
    return {
        "source_work_id": row["source_work_id"],
        "source_word": source_word.lower() if source_word else None,
        "requested_priority": row["requested_priority"],
        "source_state": row["source_state"],
        "branch_reference": branch_reference(branch_key),
        "branch_key_hex": branch_key.hex(),
        "root_pattern": _normalized_pattern(_row_value(row, "root_pattern")),
        "parent_branch_reference": (
            branch_reference(bytes(parent_branch_key))
            if parent_branch_key is not None else None
        ),
        "branch_status": branch_status,
        "branch_phase": branch_phase,
        # The branch's own materialized priority: set_source_work_priority
        # keeps it at MAX(owner_priority) across every live owner, so it can
        # exceed this row's own requested_priority on a shared branch — the
        # two are reported side by side rather than presenting one as if it
        # were the branch's only claim on scheduling.
        "branch_effective_priority": _row_value(row, "branch_priority"),
        "worker_count": worker_count,
        "owner_count": owner_count,
        "is_shared": owner_count > 1,
        "resolved_at": _row_value(row, "resolved_at"),
    }


def collect_source_report(sources: ReportSources, request: ReportRequest) -> dict:
    """Report every source word with its requests and branches rolled up —
    the reporting half of #200's operator surface.

    One word is one row: ten queued root words report ten rows, whatever the
    branch count underneath them, and a word requested more than once still
    reports one row.  Naming a word opens that word's branches, and only there
    is a branch with more than one live owner shown as one row per owning
    request, with the branch's own effective (materialized) priority reported
    alongside each owner's individually requested priority."""
    generated_at = int(time.time())
    data = {"summary": [], "total_source_word_count": 0,
            "matched_source_word_count": 0, "matched_branch_count": 0,
            "matched_open_branch_count": 0, "source_word_offset": 0,
            "matched_rows": 0, "shared_branch_count": 0,
            "branch_row_offset": 0, "rows": []}
    report = _semantic_report(
        "sources", sources, request.branch_target, generated_at, data, request
    )
    queue = None
    timing_cache = None
    try:
        queue = _open_report_queue(sources)
        source_word = (
            request.branch_target.trailing_word
            if request.branch_target.kind == "word" else None
        )
        summary_rows = queue.source_word_rows()
        if source_word is not None:
            summary_rows = [row for row in summary_rows
                            if (row["source_word"] or "").lower() == source_word]
        # Rolled up from every live membership, not only the named word's, so
        # each word's own totals are complete whichever word is in scope.
        all_membership_rows = queue.source_membership_rows()
        rollups = _source_rollups(all_membership_rows)
        all_answers = None
        try:
            all_answers = load_word_list(sources.answer_list_path)
            timing_cache = ScoreCache(sources.cache_path, all_answers,
                                      checkpoint_on_close=False)
            timings = timing_cache.completed_source_summary_map(ERD_ALL)
        except (sqlite3.Error, OSError) as error:
            timings = {}
            report["sources"]["cache"]["error"] = str(error)
        collapsed = [
            _source_summary_payload(
                row, rollups[(_row_value(row, "source_word") or "").lower() or None],
                timings.get(
                    (_row_value(row, "source_word") or "").lower() or None,
                    {"completed_at": None, "elapsed_millis": None,
                     "worker_millis": None},
                ),
            )
            for row in summary_rows
        ]
        source_sort = request.filters.sort or "erd"
        data["total_source_word_count"] = len(collapsed)
        source_states = request.filters.source_states
        if source_states:
            collapsed = [row for row in collapsed
                         if row["state"] in source_states]
        if source_sort in ("default", "erd"):
            erd_summaries = (
                _source_word_erd_summaries(
                    sources, [row["source_word"] for row in collapsed], report,
                    timing_cache, all_answers,
                ) if timing_cache is not None else {}
            )
            for row in collapsed:
                row["erd_summary"] = erd_summaries.get(row["source_word"])
        collapsed = _sorted_source_words(collapsed, source_sort)
        data["matched_source_word_count"] = len(collapsed)
        # Counted with each branch counted once: two words can own the same
        # branch, so summing their per-word counts double-counts precisely the
        # shared ownership this report exists to show.
        def branch_totals(words):
            word_set = set(words)
            open_branch_ids = {
                row["branch_id"] for row in all_membership_rows
                if (_row_value(row, "source_word") or "").lower() in word_set
                and branch_status_and_phase(
                    _row_value(row, "pending_status"),
                    _row_value(row, "active_status"),
                    _row_value(row, "worker_count", 0),
                )[1] != "complete"
            }
            return (queue.distinct_branch_count_for_words(words),
                    len(open_branch_ids))

        matched_words = [row["source_word"] for row in collapsed]
        (data["matched_branch_count"],
         data["matched_open_branch_count"]) = branch_totals(matched_words)
        # limit is a page size and source_offset is where that page starts, so
        # the words past the first page stay reachable rather than truncated
        # away.  An offset past the end yields an empty page, not the last one:
        # the count above is what tells the client how far it can page.
        offset = request.filters.source_offset or 0
        limit = request.filters.limit
        data["source_word_offset"] = offset
        data["summary"] = (
            collapsed[offset:offset + limit] if limit is not None
            else collapsed[offset:]
        )
        # Folded for the page only, and attached to the rows it describes.
        if source_sort not in ("default", "erd"):
            erd_summaries = (
                _source_word_erd_summaries(
                    sources, [row["source_word"] for row in data["summary"]], report,
                    timing_cache, all_answers,
                ) if timing_cache is not None else {}
            )
            for row in data["summary"]:
                row["erd_summary"] = erd_summaries.get(row["source_word"])
        group_by = request.filters.group_by
        if group_by is not None and group_by != "none":
            data["summary_groups"] = _grouped_source_words(
                data["summary"], group_by, branch_totals, generated_at
            )
        # Branch rows belong to one named word.  Emitting them for every word
        # would bury ten queued roots under the hundreds of branches they
        # spawned, which is the explosion this report exists to roll up.
        if source_word is not None:
            # Owner counts are computed from every live membership so a branch
            # shared with a request outside the word filter still reports
            # is_shared correctly, rather than only counting the filtered rows.
            owner_counts = collections.Counter(
                row["branch_id"] for row in all_membership_rows
            )
            payload_rows = [
                _source_membership_payload(row, owner_counts[row["branch_id"]])
                for row in all_membership_rows
                if (row["source_word"] or "").lower() == source_word
            ]
            data["matched_rows"] = len(payload_rows)
            # Counted over every matched row, like matched_rows itself: a shared
            # branch whose rows all fall past the row limit is still shared.
            data["shared_branch_count"] = len({
                row["branch_key_hex"] for row in payload_rows if row["is_shared"]
            })
            # The same page size pages the branch rows, so a named word's
            # hundreds of branches are reachable rather than truncated away.
            branch_row_limit = request.filters.limit
            branch_row_offset = request.filters.branch_row_offset or 0
            data["branch_row_offset"] = branch_row_offset
            data["rows"] = (
                payload_rows[branch_row_offset:branch_row_offset + branch_row_limit]
                if branch_row_limit is not None
                else payload_rows[branch_row_offset:]
            )
        _mark_queue_source_ok(report)
    except (sqlite3.Error, OSError) as error:
        _mark_queue_source_error(report, error)
    finally:
        if timing_cache is not None:
            timing_cache.close()
        if queue is not None:
            queue.close()
    return report


def collect_overview_report(
    sources: ReportSources,
    request: ReportRequest | None = None,
) -> dict:
    if request is not None and request.report_kind not in ("auto", "overview"):
        raise ValueError(f"unsupported report kind: {request.report_kind}")
    generated_at = int(time.time())
    report = _report_envelope(
        "overview", sources, generated_at, _empty_data(), request=request
    )
    try:
        answer_words = load_word_list(sources.answer_list_path)
        answer_set = set(answer_words)
    except OSError as error:
        message = str(error)
        report["sources"]["queue"]["error"] = message
        report["sources"]["cache"]["error"] = message
    else:
        _queue_overview(sources, generated_at, answer_set, report)
        _cache_overview(sources, generated_at, answer_words, report)
    return report


def collect_report(sources: ReportSources, request: ReportRequest) -> dict:
    if request.tree:
        if request.report_kind == "cache":
            raise ValueError("tree layout is unavailable for cache reports")
        return collect_tree_report(sources, request)
    report_kind = request.report_kind
    if report_kind == "auto":
        report_kind = {
            "root": "overview",
            "word": "word",
            "branch": "branch",
            "branch_reference": "branch",
        }.get(request.branch_target.kind)
    if report_kind == "overview":
        return collect_overview_report(sources, request)
    if report_kind == "word":
        return collect_word_report(sources, request)
    if report_kind == "branch":
        return collect_branch_report(sources, request)
    if report_kind == "queue":
        return collect_queue_report(sources, request)
    if report_kind == "workers":
        return collect_workers_report(sources, request)
    if report_kind == "cache":
        return collect_cache_report(sources, request)
    if report_kind == "hotspots":
        return collect_hotspot_report(sources, request)
    if report_kind == "leaderboard":
        return collect_leaderboard_report(sources, request)
    if report_kind == "sources":
        return collect_source_report(sources, request)
    if report_kind == "root_progress":
        return collect_root_progress_report(sources, request)
    raise ValueError(f"unsupported report kind: {request.report_kind}")
