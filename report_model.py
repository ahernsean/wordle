"""Presentation-neutral reports for ERD swarm state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re
import sqlite3
import time
from typing import Optional, Tuple

from cache_sqlite import ScoreCache
from erd_queue import (
    ERDQueue,
    WORKER_LIVENESS_SECONDS,
    derive_telemetry_path,
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


SCHEMA_VERSION = 1
WORKER_STALE_SECONDS = 20

RichSpineStep = Tuple[Optional[int], Optional[str], Optional[str], str]


@dataclass(frozen=True)
class SpineStep:
    word: str
    pattern: str


@dataclass(frozen=True)
class ReportSelector:
    kind: str
    steps: tuple[SpineStep, ...]
    trailing_word: str | None
    branch_reference: str | None
    input_text: str

    @classmethod
    def root(cls) -> "ReportSelector":
        return cls("root", (), None, None, "")


@dataclass(frozen=True)
class ResolvedBranch:
    answer_words: tuple[str, ...]
    branch_key: bytes
    steps: tuple[SpineStep, ...]
    trailing_word: str | None = None


@dataclass(frozen=True)
class ReportFilters:
    active_only: bool = False
    statuses: tuple[str, ...] = ()
    minimum_answer_count: int | None = None
    maximum_answer_count: int | None = None
    budget: int | None = None
    priority: int | None = None
    sort: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class ReportRequest:
    report_kind: str = "auto"
    selector: ReportSelector = field(default_factory=ReportSelector.root)
    include_claims: bool = False
    include_answers: bool = False
    tree: bool = False
    filters: ReportFilters = field(default_factory=ReportFilters)
    worker_id: str | None = None


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


def parse_report_selector(parts: list[str] | str | None) -> ReportSelector:
    if parts is None:
        input_text = ""
    elif isinstance(parts, str):
        input_text = parts.strip()
    else:
        input_text = " ".join(parts).strip()
    if not input_text:
        return ReportSelector.root()
    tokens = input_text.split()
    if len(tokens) == 1 and tokens[0].startswith("@"):
        digest_prefix = tokens[0][1:]
        if not re.fullmatch(r"[0-9a-fA-F]{4,40}", digest_prefix):
            raise ValueError(
                f"invalid token {tokens[0]!r}: expected @ followed by 4-40 hexadecimal characters"
            )
        return ReportSelector(
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
        return ReportSelector(
            "word", steps, normalized_words[-1], None, input_text
        )
    if len(normalized_words) == step_count:
        return ReportSelector("branch", steps, None, None, input_text)
    raise ValueError(
        f"invalid selector {input_text!r}: expected alternating word and response pattern"
    )


def resolve_selector_branch(
    selector: ReportSelector, all_answers
) -> ResolvedBranch:
    if selector.kind == "branch_reference":
        raise ValueError("a branch reference requires queue resolution")
    branch_words = list(all_answers)
    response_cache = ResponseCache(list(all_answers), score_cache=None)
    for step in selector.steps:
        groups = response_cache.group_words(step.word, branch_words)
        branch_words = groups.get(parse_pattern(step.pattern), [])
    branch_key = ScoreCache.encode_subset(branch_words)
    return ResolvedBranch(
        tuple(branch_words), branch_key, selector.steps, selector.trailing_word
    )


def resolve_branch_reference(queue, digest_prefix) -> dict:
    matches = queue.branch_rows_for_reference_prefix(digest_prefix)
    if not matches:
        raise ValueError(
            f"branch reference @{digest_prefix} not found; use the branch spine "
            "for cache-only state after queue completion"
        )
    if len(matches) > 1:
        descriptions = []
        for row in matches:
            reference = branch_reference(bytes(row["branch_key"]))
            spine = queue.row_spine_text(row) or "unknown spine"
            descriptions.append(f"@{reference} {spine}")
        raise ValueError(
            f"branch reference @{digest_prefix} is ambiguous: "
            + "; ".join(descriptions)
        )
    return matches[0]


def branch_reference(branch_key: bytes) -> str:
    return hashlib.sha1(bytes(branch_key)).hexdigest()[:12]


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


def _normalize_branch(row, lifecycle, progress, worker_count, answer_set):
    branch_key = bytes(row["branch_key"])
    spine = _normalized_branch_spine(row, answer_set)
    best_guess = _row_value(row, "best_guess")
    source_word = _row_value(row, "source_word")
    return {
        "branch_reference": branch_reference(branch_key),
        "branch_key_hex": branch_key.hex(),
        "lifecycle": lifecycle,
        "raw_status": row["status"],
        "answer_count": row["n_words"],
        "candidate_count": row["n_candidates"],
        "completed_candidate_count": progress["completed_candidate_count"],
        "bulk_completed_candidate_count": progress[
            "bulk_completed_candidate_count"
        ],
        "priority": row["priority"],
        "is_cooperative": not bool(_row_value(row, "is_user_queued", False)),
        "source_word": source_word.lower() if source_word else None,
        "source_pattern": _normalized_pattern(_row_value(row, "source_pattern")),
        "best_guess": best_guess.lower() if best_guess else None,
        "best_guess_is_answer": bool(best_guess and best_guess.lower() in answer_set),
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
        "updated_at": row["updated_at"],
        "is_live": generated_at - row["updated_at"] <= WORKER_LIVENESS_SECONDS,
        "branch_reference": branch_reference(branch_key) if branch_key else None,
        "branch_key_hex": branch_key.hex() if branch_key else None,
        "candidate_index": _row_value(row, "claim_idx"),
        "claim_started_at": _row_value(row, "claim_started_at"),
        "completed_claim_count": _row_value(row, "claims_done", 0),
        "current_candidate": current_candidate.lower() if current_candidate else None,
        "current_candidate_is_answer": bool(
            current_candidate and current_candidate.lower() in answer_set
        ),
        "current_max_guess_depth": _row_value(row, "cur_max_depth"),
        "current_node_count": _row_value(row, "cur_nodes"),
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
        "best_erd": _row_value(row, "best_erd"),
        "bound_erd": _row_value(row, "bound_erd"),
    }


def _worker_sort_key(worker):
    worker_number = worker["worker_number"]
    if worker_number.isdigit():
        return (0, int(worker_number), worker["worker_id"])
    return (1, worker["worker_id"])


def _empty_data():
    return {
        "queue_counts": {
            "pending_branch_count": 0,
            "active_user_branch_count": 0,
            "active_cooperative_branch_count": 0,
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
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        report["sources"]["telemetry"]["ok"] = True
        counts = queue.counts_by_status()
        open_rows = list(queue.branches_in_progress())
        heartbeats = list(queue.heartbeats_with_branch())
        live_heartbeats = [
            row for row in heartbeats
            if generated_at - row["updated_at"] <= WORKER_LIVENESS_SECONDS
        ]
        open_by_key = {bytes(row["branch_key"]): row for row in open_rows}
        detached_keys = {
            bytes(row["current_branch_key"])
            for row in live_heartbeats
            if row["current_branch_key"] is not None
            and bytes(row["current_branch_key"]) not in open_by_key
        }
        detached_rows = queue.active_branches_by_keys(list(detached_keys))
        finalizing_rows = {
            key: row for key, row in detached_rows.items()
            if row["status"] == "finalized"
        }
        retained_keys = list(open_by_key) + list(finalizing_rows)
        pending_rows = queue.status_by_branch_keys(retained_keys)
        progress = queue.candidate_progress_by_branch_keys(retained_keys)
        live_worker_counts = {}
        for heartbeat in live_heartbeats:
            key_value = heartbeat["current_branch_key"]
            if key_value is not None:
                key = bytes(key_value)
                live_worker_counts[key] = live_worker_counts.get(key, 0) + 1

        normalized_rows = []
        for lifecycle, branch_rows in (
            ("active", open_by_key), ("finalizing", finalizing_rows)
        ):
            for key, row in branch_rows.items():
                branch_values = dict(row)
                branch_values["is_user_queued"] = key in pending_rows
                if lifecycle == "active" and key in pending_rows:
                    branch_values["status"] = pending_rows[key]["status"]
                normalized_rows.append(_normalize_branch(
                    branch_values, lifecycle, progress[key],
                    live_worker_counts.get(key, 0), answer_set,
                ))

        workers = [
            _normalize_worker(row, generated_at, answer_set) for row in heartbeats
        ]
        workers.sort(key=_worker_sort_key)
        worker_total_keys = tuple(report["data"]["worker_totals"])
        worker_totals = {
            key: sum(worker[key] for worker in workers if worker["is_live"])
            for key in worker_total_keys
        }
        epoch_metadata = queue.epoch_metadata()

        report["data"]["queue_counts"] = {
            "pending_branch_count": counts.get("pending", 0),
            "active_user_branch_count": sum(
                key in pending_rows for key in open_by_key
            ),
            "active_cooperative_branch_count": sum(
                key not in pending_rows for key in open_by_key
            ),
            "finalizing_branch_count": len(finalizing_rows),
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


def _selector_payload(selector):
    return {
        "kind": selector.kind,
        "steps": [
            {"word": step.word, "pattern": step.pattern}
            for step in selector.steps
        ],
        "trailing_word": selector.trailing_word,
        "branch_reference": selector.branch_reference,
        "input_text": selector.input_text,
    }


def _filters_payload(filters):
    return {
        "active_only": filters.active_only,
        "statuses": list(filters.statuses),
        "minimum_answer_count": filters.minimum_answer_count,
        "maximum_answer_count": filters.maximum_answer_count,
        "budget": filters.budget,
        "priority": filters.priority,
        "sort": filters.sort,
        "limit": filters.limit,
    }


def _report_envelope(
    report_kind, sources, generated_at, data, selector=None, request=None
):
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": report_kind,
        "generated_at": generated_at,
        "selector": _selector_payload(selector) if selector is not None else None,
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
    report_kind, sources, selector, generated_at, data, request=None
):
    return _report_envelope(
        report_kind, sources, generated_at, data, selector, request
    )


def _resolved_branch_payload(resolved):
    return {
        "branch_reference": branch_reference(resolved.branch_key),
        "branch_key_hex": resolved.branch_key.hex(),
        "spine": [
            {"word": step.word, "pattern": step.pattern}
            for step in resolved.steps
        ],
        "guess_depth": len(resolved.steps),
        "answer_count": len(resolved.answer_words),
    }


def _queue_lifecycle(pending_row, active_row, worker_count=0):
    if pending_row is not None:
        raw_status = pending_row["status"]
        if raw_status == "pending":
            return "pending"
        if raw_status == "in_progress":
            return "active"
        if raw_status == "done":
            return "done"
    if active_row is not None:
        if active_row["status"] == "open":
            return "active"
        if active_row["status"] == "finalized" and worker_count:
            return "finalizing"
    return "unqueued"


def _mark_queue_source_ok(report):
    report["sources"]["queue"]["ok"] = True
    report["sources"]["telemetry"]["ok"] = True


def _mark_queue_source_error(report, error):
    message = str(error)
    report["sources"]["queue"]["error"] = message
    report["sources"]["telemetry"]["error"] = message


def collect_word_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    resolved = resolve_selector_branch(request.selector, all_answers)
    word = resolved.trailing_word
    if word is None:
        raise ValueError("word report requires a trailing word selector")
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
        "context": _resolved_branch_payload(resolved),
        "response_group_counts": {},
        "response_groups": [],
    }
    report = _semantic_report(
        "word", sources, request.selector, generated_at, data, request
    )

    pending_rows = {}
    active_rows = {}
    worker_counts = {}
    queue = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
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
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        cache_states = cache.report_branch_states(
            branch_keys, ERD_ALL, group_budget
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
        lifecycle = _queue_lifecycle(
            pending_row, active_row, worker_counts.get(branch_key, 0)
        )
        cache_state = cache_states[branch_key]
        group_row.update({
            "lifecycle": lifecycle,
            "priority": _row_value(active_row, "priority", _row_value(pending_row, "priority")),
            "worker_count": worker_counts.get(branch_key, 0),
            "cache_state": cache_state["cache_state"],
            "best_guess": cache_state["best_guess"],
            "best_erd": cache_state["best_erd"],
            "max_remaining_depth": cache_state["max_remaining_depth"],
            "updated_at": cache_state["updated_at"],
        })
        if request.include_answers:
            group_row["answer_words"] = answer_words
        data["response_groups"].append(group_row)

    response_groups = data["response_groups"]
    filters = request.filters
    if filters.active_only:
        response_groups = [
            row for row in response_groups if row["lifecycle"] == "active"
        ]
    if filters.statuses:
        response_groups = [
            row for row in response_groups if row["lifecycle"] in filters.statuses
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
    data["response_group_counts"] = {
        "response_group_count": len(matched_response_groups),
        "trivial_response_group_count": sum(
            row["answer_count"] < 2 for row in matched_response_groups
        ),
        "queued_response_group_count": sum(
            row["lifecycle"] != "unqueued" for row in matched_response_groups
        ),
        "active_response_group_count": sum(
            row["lifecycle"] in ("active", "finalizing") for row in matched_response_groups
        ),
        "exact_response_group_count": sum(
            row["cache_state"] == "exact" for row in matched_response_groups
        ),
        "loss_response_group_count": sum(
            row["cache_state"] == "loss" for row in matched_response_groups
        ),
        "missing_response_group_count": sum(
            row["cache_state"] == "missing" for row in matched_response_groups
        ),
    }
    data["matched_rows"] = len(matched_response_groups)
    data["response_groups"] = (
        matched_response_groups[:filters.limit]
        if filters.limit is not None else matched_response_groups
    )
    return report


def _steps_from_queue_row(row):
    spine = row.get("spine")
    if spine:
        tokens = spine.split()
        return tuple(
            SpineStep(tokens[index].lower(), _normalized_pattern(tokens[index + 1]))
            for index in range(0, len(tokens) - 1, 2)
        )
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


def _normalize_claim(row, republish_count):
    claimed_by = _row_value(row, "claimed_by")
    done = bool(row["done"])
    bulk_eliminated = claimed_by == "bulk-elimination"
    sound_worker = bool(claimed_by and re.fullmatch(r"worker-[0-9]+", claimed_by))
    if done and bulk_eliminated:
        completion_kind = "bulk_eliminated"
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
    }


def collect_branch_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    answer_set = set(all_answers)
    queue = None
    pending_row = None
    active_row = None
    heartbeat_rows = []
    claim_rows = []
    republish_rows = []
    queue_error = None
    referenced_row = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        if request.selector.kind == "branch_reference":
            referenced_row = resolve_branch_reference(
                queue, request.selector.branch_reference
            )
            branch_key = bytes(referenced_row["branch_key"])
            resolved = ResolvedBranch(
                _decode_branch_key(branch_key), branch_key,
                _steps_from_queue_row(referenced_row), None,
            )
        else:
            resolved = resolve_selector_branch(request.selector, all_answers)
            branch_key = resolved.branch_key
        pending_row = queue.get_pending_branch(branch_key)
        active_row = queue.get_active_branch(branch_key)
        heartbeat_rows = [
            row for row in queue.heartbeats_with_branch()
            if row["current_branch_key"] is not None
            and bytes(row["current_branch_key"]) == branch_key
        ]
        claim_rows = list(queue.claims_for_branch(branch_key))
        republish_rows = queue.candidate_republish_for_branch(branch_key)
    except (sqlite3.Error, OSError) as error:
        queue_error = error
        if request.selector.kind == "branch_reference":
            raise
        resolved = resolve_selector_branch(request.selector, all_answers)
        branch_key = resolved.branch_key
    finally:
        if queue is not None:
            queue.close()

    budget = _row_value(active_row, "budget", GAME_GUESSES - len(resolved.steps))
    branch_payload = _resolved_branch_payload(resolved)
    branch_payload["budget"] = budget
    if request.include_answers:
        branch_payload["answer_words"] = list(resolved.answer_words)
    workers = [
        _normalize_worker(row, generated_at, answer_set) for row in heartbeat_rows
    ]
    workers.sort(key=_worker_sort_key)
    live_worker_count = sum(worker["is_live"] for worker in workers)
    lifecycle = _queue_lifecycle(pending_row, active_row, live_worker_count)
    progress = {
        "completed_candidate_count": sum(bool(row["done"]) for row in claim_rows),
        "bulk_completed_candidate_count": _row_value(
            active_row, "bulk_done_candidates", 0
        ),
    }
    queue_payload = None
    if active_row is not None or pending_row is not None:
        queue_payload = {
            "lifecycle": lifecycle,
            "pending_status": _row_value(pending_row, "status"),
            "active_status": _row_value(active_row, "status"),
            "priority": _row_value(active_row, "priority", _row_value(pending_row, "priority")),
            "candidate_count": _row_value(active_row, "n_candidates"),
            "completed_candidate_count": progress["completed_candidate_count"],
            "bulk_completed_candidate_count": progress["bulk_completed_candidate_count"],
            "best_guess": _row_value(active_row, "best_guess"),
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
    provenance_unknown = any(
        claim["state"] == "done" and claim["completion_kind"] is None
        for claim in normalized_claims
    )
    data = {
        "branch": branch_payload,
        "queue": queue_payload,
        "cache": ScoreCache.report_branch_state_without_rows(branch_key, budget),
        "workers": workers,
        "republished_candidates": [
            {"candidate_index": row["idx"], "republish_count": row["count"]}
            for row in republish_rows
        ],
        "claims": normalized_claims if request.include_claims else None,
        "provenance_unknown": provenance_unknown,
    }
    report = _semantic_report(
        "branch", sources, request.selector, generated_at, data, request
    )
    if queue_error is None:
        _mark_queue_source_ok(report)
    else:
        _mark_queue_source_error(report, queue_error)

    cache = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        data["cache"] = cache.report_branch_state(branch_key, ERD_ALL, budget)
        report["sources"]["cache"]["ok"] = True
    except (sqlite3.Error, OSError) as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()
    return report


def _selector_queue_scope(selector, queue=None):
    if selector.kind == "root":
        return {}, ""
    if selector.kind == "branch_reference":
        if queue is None:
            return {}, ""
        row = resolve_branch_reference(queue, selector.branch_reference)
        prefix = row.get("spine") or ""
        scope = {"branch_key": bytes(row["branch_key"])}
        if prefix:
            scope["spine_prefix"] = prefix
        return scope, prefix
    tokens = []
    for step in selector.steps:
        tokens.extend((step.word.upper(), step.pattern))
    if selector.kind == "word":
        tokens.append(selector.trailing_word.upper())
    prefix = " ".join(tokens)
    scope = {"spine_prefix": prefix}
    if selector.kind == "word" and not selector.steps:
        scope["source_word"] = selector.trailing_word
    return scope, prefix


def _row_matches_selector(row, selector, prefix):
    if selector.kind == "root":
        return True
    spine = row.get("spine") or ""
    if selector.kind == "word":
        if spine:
            return spine.startswith(prefix + " ")
        return (
            len(selector.steps) == 0
            and (row.get("source_word") or "").lower() == selector.trailing_word
        )
    if prefix and spine:
        return spine == prefix or spine.startswith(prefix + " ")
    if selector.kind == "branch_reference":
        return branch_reference(bytes(row["branch_key"])).startswith(
            selector.branch_reference
        )
    return False


def _collection_summary(rows):
    by_lifecycle = {}
    for row in rows:
        lifecycle = row["lifecycle"]
        by_lifecycle[lifecycle] = by_lifecycle.get(lifecycle, 0) + 1
    return {
        "branch_count": len(rows),
        "branch_count_by_lifecycle": by_lifecycle,
    }


def _scoped_queue_rows(queue, request, apply_filters=True):
    filters = request.filters if apply_filters else ReportFilters()
    unbounded_filters = replace(filters, limit=None)
    scope, prefix = _selector_queue_scope(request.selector, queue)
    query_filters = _filters_payload(unbounded_filters)
    query_filters.update(scope)
    result = queue.report_queue_rows(query_filters)
    return result["rows"], prefix


def collect_queue_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    data = {"summary": {}, "matched_rows": 0, "rows": []}
    report = _semantic_report(
        "queue", sources, request.selector, generated_at, data, request
    )
    queue = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        rows, _prefix = _scoped_queue_rows(queue, request)
        data["summary"] = _collection_summary(rows)
        data["matched_rows"] = len(rows)
        limit = request.filters.limit
        data["rows"] = rows[:limit] if limit is not None else rows
        for row in data["rows"]:
            row["branch_reference"] = branch_reference(bytes(row.pop("branch_key")))
        _mark_queue_source_ok(report)
    except Exception as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


def _tree_node_id(steps):
    return "/".join(f"{step.word}:{step.pattern}" for step in steps) or "root"


def _tree_layout(rows, request, prefix, unfiltered_rows):
    context_rows = [
        row for row in unfiltered_rows
        if request.selector.kind in ("branch", "branch_reference")
        and _row_matches_selector(row, request.selector, prefix)
        and ((row.get("spine") or "") == prefix
             or (request.selector.kind == "branch_reference"
                 and branch_reference(bytes(row["branch_key"])).startswith(
                     request.selector.branch_reference)))
    ]
    selected_by_key = {row["branch_key_hex"]: row for row in rows}
    for context_row in context_rows:
        if context_row["branch_key_hex"] not in selected_by_key:
            context_copy = dict(context_row)
            context_copy["is_context"] = True
            rows.append(context_copy)
            selected_by_key[context_copy["branch_key_hex"]] = context_copy
    if not rows:
        return {
            "root": _selector_payload(request.selector),
            "topology_source": "queue",
            "tree_available": False,
            "unavailable_reason": "no extant queue topology",
            "nodes": [],
        }

    nodes = {
        "root": {
            "node_id": "root",
            "parent_node_id": None,
            "step": None,
            "branch_key_hex": None,
            "branch_reference": None,
            "lifecycle": None,
            "answer_count": None,
            "guess_depth": 0,
            "worker_count": 0,
            "completed_candidate_count": None,
            "candidate_count": None,
            "is_context": request.selector.kind == "root",
        }
    }
    for row in rows:
        spine = row.get("spine")
        if spine:
            tokens = spine.split()
            steps = []
            for index in range(0, len(tokens) - 1, 2):
                step = SpineStep(tokens[index].lower(), _normalized_pattern(tokens[index + 1]))
                parent_id = _tree_node_id(tuple(steps))
                steps.append(step)
                node_id = _tree_node_id(tuple(steps))
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id if parent_id != node_id else None,
                    "step": {"word": step.word, "pattern": step.pattern},
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "lifecycle": None,
                    "answer_count": None,
                    "guess_depth": len(steps),
                    "worker_count": 0,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                })
            final_node = nodes[_tree_node_id(tuple(steps))]
        else:
            guess_depth = (
                GAME_GUESSES - row["budget"] if row.get("budget") is not None else 1
            )
            parent_id = "root"
            for guess_depth_value in range(1, max(guess_depth, 1) + 1):
                node_id = f"unknown:{guess_depth_value}:{row['branch_key_hex']}"
                nodes.setdefault(node_id, {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "step": None,
                    "branch_key_hex": None,
                    "branch_reference": None,
                    "lifecycle": None,
                    "answer_count": None,
                    "guess_depth": guess_depth_value,
                    "worker_count": 0,
                    "completed_candidate_count": None,
                    "candidate_count": None,
                    "is_context": False,
                })
                parent_id = node_id
            final_node = nodes[parent_id]
        final_node.update({
            "branch_key_hex": row["branch_key_hex"],
            "branch_reference": branch_reference(bytes(row["branch_key"])),
            "lifecycle": row["lifecycle"],
            "answer_count": row["answer_count"],
            "worker_count": row["worker_count"],
            "completed_candidate_count": row["completed_candidate_count"],
            "candidate_count": row["candidate_count"],
            "is_context": bool(row.get("is_context")),
        })
    ordered_nodes = sorted(
        nodes.values(),
        key=lambda node: (
            node["guess_depth"],
            (node["step"] or {}).get("word", ""),
            (node["step"] or {}).get("pattern", ""),
            node["branch_key_hex"] or "",
        ),
    )
    return {
        "root": _selector_payload(request.selector),
        "topology_source": "queue",
        "tree_available": True,
        "unavailable_reason": None,
        "nodes": ordered_nodes,
    }


def collect_tree_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    inferred_kind = request.report_kind
    if inferred_kind == "auto":
        inferred_kind = "queue" if request.selector.kind == "root" else request.selector.kind
        if inferred_kind == "branch_reference":
            inferred_kind = "branch"
    data = {
        "root": _selector_payload(request.selector),
        "topology_source": "queue",
        "tree_available": False,
        "unavailable_reason": "no extant queue topology",
        "nodes": [],
    }
    report = _semantic_report(
        inferred_kind, sources, request.selector, generated_at, data, request
    )
    queue = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        filtered_rows, prefix = _scoped_queue_rows(queue, request, True)
        unfiltered_rows = filtered_rows
        if request.selector.kind in ("branch", "branch_reference"):
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
            list(filtered_rows), request, prefix, unfiltered_rows
        ))
        _mark_queue_source_ok(report)
    except Exception as error:
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
        "workers", sources, request.selector, generated_at, data, request
    )
    queue = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        scoped_rows, _prefix = _scoped_queue_rows(queue, request)
        scoped_keys = {row["branch_key_hex"] for row in scoped_rows}
        workers = [
            _normalize_worker(row, generated_at, answer_set)
            for row in queue.heartbeats_with_branch()
        ]
        filters_have_branch_scope = any((
            request.filters.active_only,
            request.filters.statuses,
            request.filters.minimum_answer_count is not None,
            request.filters.maximum_answer_count is not None,
            request.filters.budget is not None,
            request.filters.priority is not None,
        ))
        if request.selector.kind != "root" or filters_have_branch_scope:
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
        lifecycle_by_key = {
            row["branch_key_hex"]: row["lifecycle"] for row in scoped_rows
        }
        by_state = {
            "live": 0,
            "idle": 0,
            "finalizing": 0,
            "stale": 0,
            "dead": 0,
        }
        for worker in workers:
            age = generated_at - worker["updated_at"]
            if not worker["is_live"]:
                state = "dead"
            elif age > WORKER_STALE_SECONDS:
                state = "stale"
            elif worker["branch_key_hex"] is None:
                state = "idle"
            elif lifecycle_by_key.get(worker["branch_key_hex"]) == "finalizing":
                state = "finalizing"
            else:
                state = "live"
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
    except Exception as error:
        _mark_queue_source_error(report, error)
    finally:
        if queue is not None:
            queue.close()
    return report


def collect_cache_report(sources: ReportSources, request: ReportRequest) -> dict:
    generated_at = int(time.time())
    all_answers = load_word_list(sources.answer_list_path)
    data = {}
    report = _semantic_report(
        "cache", sources, request.selector, generated_at, data, request
    )
    queue = None
    try:
        queue = ERDQueue(sources.queue_path, telemetry_path=sources.telemetry_path)
        _mark_queue_source_ok(report)
    except Exception as error:
        _mark_queue_source_error(report, error)
    cache = None
    try:
        cache = ScoreCache(
            sources.cache_path, all_answers, checkpoint_on_close=False
        )
        selector = request.selector
        if selector.kind == "root":
            limit = request.filters.limit or 50
            recent_rows = cache.report_recent_rows(
                ERD_ALL, generated_at - 300, limit
            )
            data.update({
                "summary": cache.erd_report_summary(ERD_ALL, generated_at - 300),
                "distributions": cache.report_cache_distributions(ERD_ALL),
                "recent_rows": [
                    {
                        **{key: value for key, value in row.items() if key != "branch_key"},
                        "branch_key_hex": row["branch_key"].hex(),
                        "branch_reference": branch_reference(row["branch_key"]),
                    }
                    for row in recent_rows
                ],
            })
        elif selector.kind == "word":
            resolved = resolve_selector_branch(selector, all_answers)
            groups = ResponseCache(all_answers, score_cache=None).group_words(
                selector.trailing_word, list(resolved.answer_words)
            )
            keys = [ScoreCache.encode_subset(words) for words in groups.values() if words]
            states = cache.report_branch_states(
                keys, ERD_ALL, GAME_GUESSES - len(selector.steps) - 1
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
                    **states[key],
                })
            data.update({"summary": {"response_group_count": len(rows)}, "rows": rows})
        else:
            if selector.kind == "branch_reference":
                if queue is None:
                    raise ValueError("queue unavailable for branch reference")
                referenced_row = resolve_branch_reference(
                    queue, selector.branch_reference
                )
                branch_key = bytes(referenced_row["branch_key"])
                steps = _steps_from_queue_row(referenced_row)
                budget = _row_value(
                    referenced_row, "budget", GAME_GUESSES - len(steps)
                )
            else:
                resolved = resolve_selector_branch(selector, all_answers)
                branch_key = resolved.branch_key
                steps = resolved.steps
                budget = GAME_GUESSES - len(steps)
            data.update({
                "branch_key_hex": branch_key.hex(),
                "branch_reference": branch_reference(branch_key),
                "cache": cache.report_branch_state(
                    branch_key, ERD_ALL, budget
                ),
            })
        report["sources"]["cache"]["ok"] = True
    except Exception as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()
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
        }.get(request.selector.kind)
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
    raise ValueError(f"unsupported report kind: {request.report_kind}")
