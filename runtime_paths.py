"""Shared default paths for Wordle runtime data and word lists."""

import os

RUNTIME_DIR = "runtime"

DEFAULT_QUEUE_PATH = os.path.join(RUNTIME_DIR, "erd_queue.sqlite3")
DEFAULT_TELEMETRY_PATH = os.path.join(RUNTIME_DIR, "erd_queue_telemetry.sqlite3")
DEFAULT_CACHE_PATH = os.path.join(RUNTIME_DIR, "wordle_cache.sqlite3")
DEFAULT_SEARCH_LOG_PATH = os.path.join(RUNTIME_DIR, "erd_search.log")
DEFAULT_DEBUG_LOG_PATH = os.path.join(RUNTIME_DIR, "wordle_debug.log")
DEFAULT_VERIFY_CACHE_LOG_PATH = os.path.join(RUNTIME_DIR, "verify_erd_cache.log")
DEFAULT_VERIFY_LOSSES_LOG_PATH = os.path.join(RUNTIME_DIR, "verify_erd_losses.log")

DEFAULT_ANSWER_LIST_PATH = "all_answers.txt"
DEFAULT_CANDIDATE_LIST_PATH = "all_candidates.txt"


def ensure_runtime_dir():
    """Create the runtime directory if it does not already exist."""
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def worker_log_path(worker_id):
    """Per-worker log path: runtime/erd_worker_<worker_id>.log."""
    return os.path.join(RUNTIME_DIR, f'erd_worker_{worker_id}.log')
