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


def fmt_pattern(code: int) -> str:
    """Format a response code as a 5-char string: g=green, y=yellow, -=gray."""
    chars = {0: '-', 1: 'y', 2: 'g'}
    digits = []
    for _ in range(5):
        digits.append(code % 3)
        code //= 3
    return ''.join(chars[d] for d in reversed(digits))


def parse_pattern(s: str) -> int:
    """Inverse of fmt_pattern: a 5-char response string to its int code.

    Accepts g/green, y/yellow, and -/./gray (any of '-' '.' 'x' for gray).
    Matches _encode_response: leftmost char is the most significant trit.
    """
    vals = {'g': 2, 'green': 2, 'y': 1, 'yellow': 1,
            '-': 0, '.': 0, 'x': 0, 'gray': 0, 'grey': 0}
    s = s.strip().lower()
    if len(s) != 5:
        raise ValueError(f'pattern must be 5 characters, got {s!r}')
    code = 0
    for ch in s:
        if ch not in vals:
            raise ValueError(f'bad pattern char {ch!r} in {s!r}')
        code = code * 3 + vals[ch]
    return code


_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pending_subgroups (
    subset_key     BLOB    NOT NULL,
    n_words        INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    status         TEXT    NOT NULL DEFAULT 'pending',
    claimed_by     TEXT,
    claimed_at     INTEGER,
    completed_at   INTEGER,
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
    subgroups_done     INTEGER NOT NULL DEFAULT 0,
    candidates_done    INTEGER,
    candidates_total   INTEGER,
    best_word          TEXT,
    best_erd           REAL
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- A single subgroup currently being solved cooperatively by many workers,
-- each evaluating a disjoint slice of the candidate guesses (the "split").
-- best_erd is the running minimum cost across all candidates: it only ever
-- decreases, and is a real achieved value, so any worker may read it as a
-- branch-and-bound bound.  status open -> finalized exactly once, by the
-- worker that observes full chunk coverage; that worker writes the top-level
-- ERD entry to the persistent cache.
CREATE TABLE IF NOT EXISTS split_subgroups (
    subset_key     BLOB    PRIMARY KEY,
    n_words        INTEGER NOT NULL,
    n_candidates   INTEGER NOT NULL,
    chunk          INTEGER NOT NULL,
    ranked_blob    BLOB    NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    best_erd       REAL,
    best_word      TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER
);

-- One row per claimed candidate chunk of a split.  A row's existence is an
-- ADVISORY claim ("someone is probably evaluating this slice"); only done=1
-- is AUTHORITATIVE ("this slice is fully evaluated and folded into best_erd").
-- Coverage = all n_chunks rows with done=1.  A crashed worker leaves a
-- done=0 row that stale-reclaim deletes, turning the slice back into an
-- unclaimed gap to be redone — never skipped.
CREATE TABLE IF NOT EXISTS split_chunks (
    subset_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    PRIMARY KEY (subset_key, idx)
);
"""


class ErdQueue:
    """SQLite-backed work queue for the parallel ERD_ALL precache job."""

    def __init__(self, db_path: str, timeout: float = 30.0):
        self._conn = sqlite3.connect(db_path, timeout=timeout,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate()

    def _migrate(self):
        """Add columns introduced after the initial schema, if missing."""
        existing = {r['name'] for r in
                    self._conn.execute('PRAGMA table_info(pending_subgroups)')}
        for col, defn in [('source_word', 'TEXT'),
                          ('source_pattern', 'INTEGER')]:
            if col not in existing:
                self._conn.execute(
                    f'ALTER TABLE pending_subgroups ADD COLUMN {col} {defn}')

        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        for col, defn in [('candidates_done',  'INTEGER'),
                          ('candidates_total', 'INTEGER'),
                          ('best_word',        'TEXT'),
                          ('best_erd',         'REAL')]:
            if col not in existing_hb:
                self._conn.execute(
                    f'ALTER TABLE worker_heartbeat ADD COLUMN {col} {defn}')

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # Bootstrap / populate-queue
    # ------------------------------------------------------------------

    def add_pending_many(self, rows):
        """Insert (subset_key, n_words, priority, source_word, source_pattern) rows.

        Uses an UPSERT so that:
        - A row inserted for the first time is added as 'pending'.
        - A row already present has its priority UPGRADED (never downgraded),
          e.g. a subgroup first inserted at priority=0 by an earlier root word
          is correctly promoted to priority=1 when a VIP word (SALET) is
          bootstrapped later.
        - source_word / source_pattern record the first root word whose branch
          produced this subgroup (kept for display in `status`).
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany("""
                INSERT INTO pending_subgroups
                    (subset_key, n_words, priority, source_word, source_pattern, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(subset_key) DO UPDATE SET
                    priority       = MAX(priority, excluded.priority),
                    source_word    = COALESCE(source_word, excluded.source_word),
                    source_pattern = COALESCE(source_pattern, excluded.source_pattern)
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
                  started_at: int, subgroups_done: int,
                  candidates_done=None, candidates_total=None,
                  best_word=None, best_erd=None):
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO worker_heartbeat
                (worker_id, pid, current_subset_key, n_words,
                 started_at, updated_at, subgroups_done,
                 candidates_done, candidates_total, best_word, best_erd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (worker_id, pid, current_subset_key, n_words,
              started_at, now, subgroups_done,
              candidates_done, candidates_total, best_word, best_erd))

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

    def heartbeats_with_source(self):
        """Return heartbeat rows joined with source_word/pattern/priority."""
        return self._conn.execute("""
            SELECT h.*,
                   p.priority,
                   p.source_word,
                   p.source_pattern
            FROM worker_heartbeat h
            LEFT JOIN pending_subgroups p
                   ON h.current_subset_key = p.subset_key
            ORDER BY h.worker_id
        """).fetchall()

    # ------------------------------------------------------------------
    # Split (candidate-level cooperative solve of one subgroup)
    # ------------------------------------------------------------------

    @staticmethod
    def n_chunks_for(n_candidates: int, chunk: int) -> int:
        return (n_candidates + chunk - 1) // chunk

    @staticmethod
    def chunk_range(idx: int, chunk: int, n_candidates: int):
        lo = idx * chunk
        hi = min(lo + chunk, n_candidates)
        return lo, hi

    def create_split(self, subset_key, n_words, n_candidates, chunk,
                     ranked_blob, priority=0, source_word=None,
                     source_pattern=None) -> bool:
        """Register a subgroup as an open split, if not already present.

        Idempotent: the first worker to reach a subgroup creates the split
        (storing the one-time-ranked candidate order as ranked_blob); later
        workers see it already exists and just join.  Returns True if this
        call created the row, False if it already existed.
        """
        now = int(time.time())
        cur = self._conn.execute("""
            INSERT OR IGNORE INTO split_subgroups
                (subset_key, n_words, n_candidates, chunk, ranked_blob,
                 priority, source_word, source_pattern, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """, (subset_key, n_words, n_candidates, chunk, ranked_blob,
              priority, source_word, source_pattern, now))
        return cur.rowcount == 1

    def get_split(self, subset_key):
        return self._conn.execute(
            "SELECT * FROM split_subgroups WHERE subset_key = ?",
            (subset_key,)).fetchone()

    def claim_split_chunk(self, subset_key, worker_id, n_chunks):
        """Atomically claim the lowest-indexed chunk that has no row yet.

        A chunk with an existing row is either in-flight (done=0, another
        worker) or complete (done=1); either way we don't re-hand it out here.
        Stale done=0 rows are freed separately by reclaim_stale_chunks, which
        deletes them so they reappear as gaps.  Returns the chunk idx, or None
        if every chunk already has a row (the split is fully claimed — the
        worker should look elsewhere, NOT block).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            taken = {r["idx"] for r in self._conn.execute(
                "SELECT idx FROM split_chunks WHERE subset_key = ?",
                (subset_key,))}
            idx = None
            for c in range(n_chunks):
                if c not in taken:
                    idx = c
                    break
            if idx is None:
                self._conn.execute("COMMIT")
                return None
            now = int(time.time())
            self._conn.execute("""
                INSERT INTO split_chunks
                    (subset_key, idx, claimed_by, claimed_at, done)
                VALUES (?, ?, ?, ?, 0)
            """, (subset_key, idx, worker_id, now))
            self._conn.execute("COMMIT")
            return idx
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def complete_split_chunk(self, subset_key, idx):
        """Mark a chunk authoritatively complete (done=1)."""
        now = int(time.time())
        self._conn.execute("""
            UPDATE split_chunks SET done = 1, done_at = ?
            WHERE subset_key = ? AND idx = ?
        """, (now, subset_key, idx))

    def update_split_best(self, subset_key, best_word, best_erd):
        """Lower the split's running best (monotone — never raises it)."""
        self._conn.execute("""
            UPDATE split_subgroups
            SET best_erd = ?, best_word = ?
            WHERE subset_key = ?
              AND (best_erd IS NULL OR ? < best_erd)
        """, (best_erd, best_word, subset_key, best_erd))

    def read_split_best(self, subset_key):
        """Return (best_word, best_erd) or (None, None)."""
        row = self._conn.execute(
            "SELECT best_word, best_erd FROM split_subgroups WHERE subset_key = ?",
            (subset_key,)).fetchone()
        if row is None:
            return (None, None)
        return (row["best_word"], row["best_erd"])

    def split_done_chunks(self, subset_key) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM split_chunks WHERE subset_key = ? AND done = 1",
            (subset_key,)).fetchone()[0]

    def try_finalize_split(self, subset_key) -> bool:
        """Atomically transition open -> finalized, exactly once.

        Returns True only for the single caller that wins the transition; that
        caller is then responsible for writing the top-level ERD entry to the
        persistent cache.  Returns False if another worker already finalized
        (or the split is gone).  Caller must have confirmed full coverage
        first; the WHERE status='open' guard makes the write idempotent.
        """
        now = int(time.time())
        cur = self._conn.execute("""
            UPDATE split_subgroups SET status = 'finalized', finalized_at = ?
            WHERE subset_key = ? AND status = 'open'
        """, (now, subset_key))
        return cur.rowcount == 1

    def reclaim_stale_chunks(self, max_age_seconds: int) -> int:
        """Delete in-flight (done=0) chunk claims older than max_age_seconds.

        A deleted claim turns its slice back into an unclaimed gap, so the work
        is redone rather than silently skipped.  done=1 rows are never touched.
        Returns the number of claims freed.
        """
        cutoff = int(time.time()) - max_age_seconds
        self._conn.execute("""
            DELETE FROM split_chunks
            WHERE done = 0 AND claimed_at < ?
        """, (cutoff,))
        return self._conn.execute("SELECT changes()").fetchone()[0]

    def open_splits(self):
        """Open splits, highest priority first — for swarm scheduling."""
        return self._conn.execute("""
            SELECT * FROM split_subgroups WHERE status = 'open'
            ORDER BY priority DESC, n_words DESC
        """).fetchall()

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
