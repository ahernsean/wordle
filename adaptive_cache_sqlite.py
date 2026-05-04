"""SQLite-backed persistence adapter for adaptive deep lookahead bounds."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEMANTICS_VERSION = "entropy-lookahead-v1"


@dataclass(frozen=True)
class AdaptiveState:
    """Cached bounds for a (subset, depth, policy, universe) state."""

    lower_bound: float
    upper_bound: float
    best_guess_id: str | None
    is_exact: bool


class AdaptiveCacheSQLite:
    """Persistence adapter for adaptive deep-search states.

    This class encapsulates all SQL details behind a compact read/write API.
    """

    def __init__(
        self,
        db_path,
        ordered_answers,
        semantics_version=DEFAULT_SEMANTICS_VERSION,
        timeout=30.0,
    ):
        self.db_path = Path(db_path)
        self.ordered_answers = list(ordered_answers)
        self.semantics_version = semantics_version
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=timeout,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()
        self.universe_id = self._ensure_universe_id()

    def close(self):
        self._conn.close()

    def _ensure_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_metadata (
                universe_id TEXT PRIMARY KEY,
                answer_list_hash TEXT NOT NULL,
                semantics_version TEXT NOT NULL,
                answer_count INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_state (
                subset_blob BLOB NOT NULL,
                remaining_depth INTEGER NOT NULL,
                policy TEXT NOT NULL,
                universe_id TEXT NOT NULL,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                best_guess_id TEXT,
                is_exact INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (
                    subset_blob,
                    remaining_depth,
                    policy,
                    universe_id
                ),
                FOREIGN KEY (universe_id)
                    REFERENCES universe_metadata(universe_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_adaptive_lookup
            ON adaptive_state(universe_id, policy, remaining_depth)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_partition (
                subset_blob BLOB NOT NULL,
                guess_word TEXT NOT NULL,
                policy TEXT NOT NULL,
                universe_id TEXT NOT NULL,
                partition_blob BLOB NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (
                    subset_blob,
                    guess_word,
                    policy,
                    universe_id
                ),
                FOREIGN KEY (universe_id)
                    REFERENCES universe_metadata(universe_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_adaptive_partition_lookup
            ON adaptive_partition(universe_id, policy, guess_word)
            """
        )

    def _compute_universe_hash(self):
        payload = {
            "answers": self.ordered_answers,
            "semantics_version": self.semantics_version,
        }
        canonical = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _ensure_universe_id(self):
        universe_id = self._compute_universe_hash()
        now = int(time.time())
        answer_list_hash = hashlib.sha256(
            "\n".join(self.ordered_answers).encode("utf-8")
        ).hexdigest()

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO universe_metadata (
                    universe_id,
                    answer_list_hash,
                    semantics_version,
                    answer_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    universe_id,
                    answer_list_hash,
                    self.semantics_version,
                    len(self.ordered_answers),
                    now,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return universe_id

    def read_partition(self, subset_blob, guess_word, policy):
        row = self._conn.execute(
            """
            SELECT partition_blob
            FROM adaptive_partition
            WHERE subset_blob = ?
              AND guess_word = ?
              AND policy = ?
              AND universe_id = ?
            """,
            (subset_blob, guess_word, policy, self.universe_id),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["partition_blob"])

    def write_partition(self, subset_blob, guess_word, policy, payload):
        now = int(time.time())
        serialized = json.dumps(payload, separators=(",", ":"))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adaptive_partition (
                subset_blob,
                guess_word,
                policy,
                universe_id,
                partition_blob,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                subset_blob,
                guess_word,
                policy,
                self.universe_id,
                serialized,
                now,
            ),
        )

    @staticmethod
    def encode_subset(words):
        """Deterministically encode a subset into a blob key."""
        return "\0".join(sorted(words)).encode("utf-8")

    def read_state(self, subset_blob, remaining_depth, policy):
        row = self._conn.execute(
            """
            SELECT lower_bound, upper_bound, best_guess_id, is_exact
            FROM adaptive_state
            WHERE subset_blob = ?
              AND remaining_depth = ?
              AND policy = ?
              AND universe_id = ?
            """,
            (subset_blob, remaining_depth, policy, self.universe_id),
        ).fetchone()
        if not row:
            return None
        return AdaptiveState(
            lower_bound=row["lower_bound"],
            upper_bound=row["upper_bound"],
            best_guess_id=row["best_guess_id"],
            is_exact=bool(row["is_exact"]),
        )

    def write_state(
        self,
        subset_blob,
        remaining_depth,
        policy,
        lower_bound,
        upper_bound,
        best_guess_id=None,
        is_exact=False,
    ):
        """Upsert state with monotonic bound tightening semantics."""
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._conn.execute(
                """
                SELECT lower_bound, upper_bound, best_guess_id, is_exact
                FROM adaptive_state
                WHERE subset_blob = ?
                  AND remaining_depth = ?
                  AND policy = ?
                  AND universe_id = ?
                """,
                (subset_blob, remaining_depth, policy, self.universe_id),
            ).fetchone()

            if existing is None:
                effective_exact = bool(is_exact)
                effective_lower = lower_bound
                effective_upper = upper_bound
                if effective_exact:
                    effective_lower = lower_bound
                    effective_upper = lower_bound
                self._conn.execute(
                    """
                    INSERT INTO adaptive_state (
                        subset_blob,
                        remaining_depth,
                        policy,
                        universe_id,
                        lower_bound,
                        upper_bound,
                        best_guess_id,
                        is_exact,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subset_blob,
                        remaining_depth,
                        policy,
                        self.universe_id,
                        effective_lower,
                        effective_upper,
                        best_guess_id,
                        int(effective_exact),
                        now,
                    ),
                )
            else:
                if bool(existing["is_exact"]):
                    # Authoritative exact rows are immutable by non-exact updates.
                    if not is_exact:
                        self._conn.execute("COMMIT")
                        return

                merged_lower = max(existing["lower_bound"], lower_bound)
                merged_upper = min(existing["upper_bound"], upper_bound)
                merged_exact = bool(existing["is_exact"]) or bool(is_exact)
                merged_guess = (
                    best_guess_id
                    if best_guess_id is not None
                    else existing["best_guess_id"]
                )

                if merged_exact:
                    # Exact value is authoritative.
                    if is_exact:
                        exact_value = lower_bound
                    else:
                        exact_value = existing["lower_bound"]
                    merged_lower = exact_value
                    merged_upper = exact_value

                self._conn.execute(
                    """
                    UPDATE adaptive_state
                    SET lower_bound = ?,
                        upper_bound = ?,
                        best_guess_id = ?,
                        is_exact = ?,
                        updated_at = ?
                    WHERE subset_blob = ?
                      AND remaining_depth = ?
                      AND policy = ?
                      AND universe_id = ?
                    """,
                    (
                        merged_lower,
                        merged_upper,
                        merged_guess,
                        int(merged_exact),
                        now,
                        subset_blob,
                        remaining_depth,
                        policy,
                        self.universe_id,
                    ),
                )

            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
