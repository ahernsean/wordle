"""erd_queue.py — Sidecar SQLite coordination queue for the parallel ERD_ALL precache job.

Lives in erd_queue.sqlite3, separate from wordle_cache.sqlite3, so the
high-frequency coordination writes (claim/heartbeat/done) don't contend with
the high-volume ERD result writes in the main cache.
"""

from __future__ import annotations

import sqlite3
import time

from cache_sqlite import ScoreCache

# Re-export so callers don't need to import cache_sqlite directly.
encode_subset = ScoreCache.encode_subset


def decode_subset(blob: bytes) -> list[str]:
    """Reverse ScoreCache.encode_subset: split fixed-5-byte words."""
    return [blob[i:i + 5].decode() for i in range(0, len(blob), 5)]


_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pending_subgroups (
    subset_key   BLOB    NOT NULL,
    n_words      INTEGER NOT NULL,
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'pending',
    claimed_by   TEXT,
    claimed_at   INTEGER,
    completed_at INTEGER,
    PRIMARY KEY (subset_key)
);

-- priority DESC first so VIP (priority=1) subgroups drain before priority=0,
-- then n_words DESC within each priority tier for maximum recursive fill-in.
CREATE INDEX IF NOT EXISTS idx_pending_status_pri_n
    ON pending_subgroups(status, priority DESC, n_words DESC);

CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id          TEXT    PRIMARY KEY,
    pid                INTEGER NOT NULL,
    current_subset_key BLOB,
    n_words            INTEGER,
    started_at         INTEGER,
    updated_at         INTEGER NOT NULL,
    subgroups_done     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class ErdQueue:
    """SQLite-backed work queue for the parallel ERD_ALL precache job."""

    def __init__(self, db_path: str, timeout: float = 30.0):
        self._conn = sqlite3.connect(db_path, timeout=timeout,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # Bootstrap / populate-queue
    # ------------------------------------------------------------------

    def add_pending_many(self, rows):
        """Insert (subset_key, n_words, priority) rows, ignoring duplicates.

        Dedup is via PRIMARY KEY(subset_key) + INSERT OR IGNORE, so the same
        branch subgroup appearing under multiple root words is enqueued once.
        An existing row keeps its current priority — if it was already inserted
        with priority=1 from a VIP word, a later INSERT OR IGNORE with
        priority=0 leaves it untouched (which is the desired behaviour).
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany("""
                INSERT OR IGNORE INTO pending_subgroups
                    (subset_key, n_words, priority, status)
                VALUES (?, ?, ?, 'pending')
            """, rows)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Worker claim loop
    # ------------------------------------------------------------------

    def claim_next(self, worker_id: str):
        """Atomically claim the highest-priority / largest pending subgroup.

        Returns (subset_key bytes, n_words int) or None if queue is empty.

        Uses BEGIN IMMEDIATE to acquire the write lock before the SELECT,
        eliminating the TOCTOU race where two workers could both read the
        same 'pending' row before either marks it 'in_progress'.  Under
        contention the loser blocks and retries automatically via
        sqlite3_busy_timeout (set by timeout= on connect).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("""
                SELECT subset_key, n_words FROM pending_subgroups
                WHERE status = 'pending'
                ORDER BY priority DESC, n_words DESC
                LIMIT 1
            """).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            now = int(time.time())
            self._conn.execute("""
                UPDATE pending_subgroups
                SET status = 'in_progress', claimed_by = ?, claimed_at = ?
                WHERE subset_key = ?
            """, (worker_id, now, row["subset_key"]))
            self._conn.execute("COMMIT")
            return bytes(row["subset_key"]), row["n_words"]
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def mark_done(self, subset_key: bytes):
        now = int(time.time())
        self._conn.execute("""
            UPDATE pending_subgroups
            SET status = 'done', completed_at = ?
            WHERE subset_key = ?
        """, (now, subset_key))

    def reset_stale_in_progress(self) -> int:
        """Reset any 'in_progress' rows back to 'pending'.

        Called on supervisor startup so a crashed worker's claimed row is
        re-queued rather than stuck forever.  Returns the number of rows reset.
        """
        self._conn.execute("""
            UPDATE pending_subgroups
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE status = 'in_progress'
        """)
        return self._conn.execute("SELECT changes()").fetchone()[0]

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self, worker_id: str, pid: int,
                  current_subset_key, n_words,
                  started_at: int, subgroups_done: int):
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO worker_heartbeat
                (worker_id, pid, current_subset_key, n_words,
                 started_at, updated_at, subgroups_done)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (worker_id, pid, current_subset_key, n_words,
              started_at, now, subgroups_done))

    def clear_heartbeat(self, worker_id: str):
        self._conn.execute(
            "DELETE FROM worker_heartbeat WHERE worker_id = ?", (worker_id,))

    # ------------------------------------------------------------------
    # Status queries (read-only, safe to call concurrently with workers)
    # ------------------------------------------------------------------

    def counts_by_status(self) -> dict:
        return {r["status"]: r["c"] for r in self._conn.execute(
            "SELECT status, COUNT(*) c FROM pending_subgroups GROUP BY status"
        )}

    def words_by_status(self) -> dict:
        """Total answer-words covered by subgroups in each status bucket."""
        rows = self._conn.execute("""
            SELECT status, SUM(n_words) total
            FROM pending_subgroups GROUP BY status
        """).fetchall()
        return {r["status"]: (r["total"] or 0) for r in rows}

    def total_subgroups(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM pending_subgroups"
        ).fetchone()[0]

    def heartbeats(self):
        return self._conn.execute(
            "SELECT * FROM worker_heartbeat ORDER BY worker_id"
        ).fetchall()

    # ------------------------------------------------------------------
    # run_meta key-value store
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            (key, str(value)))

    def get_meta(self, key: str):
        row = self._conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None
