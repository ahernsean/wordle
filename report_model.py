"""Presentation-neutral reports for ERD swarm state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Optional, Tuple

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_GUESS_LIST_PATH,
    DEFAULT_QUEUE_PATH,
    DEFAULT_TELEMETRY_PATH,
)
from wordle_engine import ERD_ALL, load_word_list
from wordle_ui import fmt_pattern


SCHEMA_VERSION = 1
WORKER_LIVENESS_SECONDS = 30

RichSpineStep = Tuple[Optional[int], Optional[str], Optional[str], str]


@dataclass(frozen=True)
class ReportRequest:
    report_kind: str = "overview"


@dataclass(frozen=True)
class ReportSources:
    queue_path: str
    cache_path: str
    answer_list_path: str
    guess_list_path: str
    telemetry_path: str | None = None

    @classmethod
    def defaults(cls) -> "ReportSources":
        return cls(
            queue_path=DEFAULT_QUEUE_PATH,
            cache_path=DEFAULT_CACHE_PATH,
            answer_list_path=DEFAULT_ANSWER_LIST_PATH,
            guess_list_path=DEFAULT_GUESS_LIST_PATH,
            telemetry_path=DEFAULT_TELEMETRY_PATH,
        )


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
        epoch_text = queue.get_meta("epoch")
        epoch = int(epoch_text) if epoch_text is not None else None
        epoch_row = None
        if epoch is not None:
            epoch_row = queue._conn.execute(
                "SELECT label, git_sha FROM main.telemetry_epoch WHERE epoch = ?",
                (epoch,),
            ).fetchone()

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
            "epoch": epoch,
            "label": _row_value(epoch_row, "label"),
            "git_sha": _row_value(epoch_row, "git_sha"),
        })
        report["sources"]["telemetry"]["ok"] = True
    except Exception as error:
        message = str(error)
        report["sources"]["queue"]["error"] = message
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
    except Exception as error:
        report["sources"]["cache"]["error"] = str(error)
    finally:
        if cache is not None:
            cache.close()


def collect_overview_report(
    sources: ReportSources,
    request: ReportRequest | None = None,
) -> dict:
    if request is not None and request.report_kind != "overview":
        raise ValueError(f"unsupported report kind: {request.report_kind}")
    generated_at = int(time.time())
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "overview",
        "generated_at": generated_at,
        "selector": None,
        "filters": {},
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
                "path": sources.telemetry_path,
                "ok": False,
                "error": None,
            },
            "cache": {"path": sources.cache_path, "ok": False, "error": None},
        },
        "data": _empty_data(),
    }
    try:
        answer_words = load_word_list(sources.answer_list_path)
        answer_set = set(answer_words)
    except Exception as error:
        answer_words = []
        answer_set = set()
        report["sources"]["cache"]["error"] = str(error)

    _queue_overview(sources, generated_at, answer_set, report)
    if report["sources"]["cache"]["error"] is None:
        _cache_overview(sources, generated_at, answer_words, report)
    return report


def collect_report(sources: ReportSources, request: ReportRequest) -> dict:
    if request.report_kind == "overview":
        return collect_overview_report(sources, request)
    raise ValueError(f"unsupported report kind: {request.report_kind}")
