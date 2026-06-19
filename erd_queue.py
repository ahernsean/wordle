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

CREATE TABLE IF NOT EXISTS pending_branches (
    branch_key     BLOB    NOT NULL,
    n_words        INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    status         TEXT    NOT NULL DEFAULT 'pending',
    claimed_by     TEXT,
    claimed_at     INTEGER,
    completed_at   INTEGER,
    PRIMARY KEY (branch_key)
);

-- priority DESC first so VIP (priority=1) branches drain before priority=0,
-- then n_words DESC within each priority tier for maximum recursive fill-in.
CREATE INDEX IF NOT EXISTS idx_pending_status_pri_n
    ON pending_branches(status, priority DESC, n_words DESC);

-- One row per worker, overwritten each heartbeat.  In the swarm model a
-- worker is a fungible contributor: it reports which branch and chunk it is
-- on purely so the operator can see it is alive and moving (health), not as
-- the unit of progress (that lives in active_branches).  The metric columns
-- (cache_hits/misses, n_pruned/n_ok, cand_rate) let `status` aggregate cache
-- effectiveness and branch-and-bound pruning across all workers.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id          TEXT    PRIMARY KEY,
    pid                INTEGER NOT NULL,
    current_branch_key BLOB,        -- branch key this worker is contributing to
    n_words            INTEGER,
    started_at         INTEGER,
    updated_at         INTEGER NOT NULL,
    chunks_done        INTEGER NOT NULL DEFAULT 0,
    chunk_idx          INTEGER,     -- chunk currently held
    chunk_started_at   INTEGER,     -- when it was claimed (held-time = now - this)
    cand_rate          REAL,        -- candidates/sec, recent
    cache_hits         INTEGER,
    cache_misses       INTEGER,
    n_pruned           INTEGER,     -- candidates eliminated by the shared bound
    n_ok               INTEGER,     -- candidates fully evaluated
    best_guess          TEXT,
    best_erd           REAL,
    cur_candidate      TEXT,        -- candidate word currently under evaluation
    cand_n_seen        INTEGER,     -- candidates evaluated so far in this chunk
    cand_chunk_size    INTEGER,     -- total candidates in this chunk
    cur_max_depth      INTEGER,     -- deepest recursion level in current candidate
    cur_nodes          INTEGER,     -- monotonic node counter (forward-progress)
    node_rate          REAL,        -- nodes/sec since last heartbeat
    cur_path           TEXT         -- live recursion spine: subset sizes by depth
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- A branch currently being solved cooperatively by one or more workers, each
-- evaluating a disjoint chunk (slice) of the ranked candidate guesses.
-- best_erd is the running-minimum cost across all candidates tried so far: it
-- only ever decreases and is a real achieved value, so any worker may read it
-- as a branch-and-bound bound.  status open -> finalized exactly once, by the
-- worker that observes full chunk coverage; that worker writes the branch's
-- ERD entry to the persistent cache, then the row (and its chunks) is deleted.
-- Candidate order is NOT stored: rank_candidates_by_max_group_size_then_entropy_gain is
-- deterministic, so every worker re-ranks locally and agrees on which
-- candidates chunk i covers, sharing the work through the candidate_scores cache.
CREATE TABLE IF NOT EXISTS active_branches (
    branch_key     BLOB    PRIMARY KEY,
    n_words        INTEGER NOT NULL,
    n_candidates   INTEGER NOT NULL,
    chunk_size     INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    best_erd       REAL,
    best_guess      TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
    ON active_branches(status, priority DESC, n_words DESC);

-- One row per claimed candidate chunk of a branch.  A row's existence is an
-- ADVISORY claim ("a worker is probably evaluating this slice"); only done=1
-- is AUTHORITATIVE ("this slice is fully evaluated and folded into best_erd").
-- Coverage = all chunks with done=1.  A crashed worker leaves a done=0 row
-- that stale-reclaim deletes, turning the slice back into an unclaimed gap to
-- be redone — never skipped.
CREATE TABLE IF NOT EXISTS branch_chunks (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    PRIMARY KEY (branch_key, idx)
);
"""


class ERDQueue:
    """SQLite-backed work queue for the parallel ERD_ALL precache job."""

    def __init__(self, db_path: str, timeout: float = 30.0):
        self._conn = sqlite3.connect(db_path, timeout=timeout,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate()

    def _migrate(self):
        """Add columns and rename tables introduced after the initial schema."""
        # Rename pending_subgroups -> pending_branches for databases predating
        # this migration.  The schema creates pending_branches (empty) first;
        # drop that shell before renaming so the old rows survive.
        tables = {r['name'] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'pending_subgroups' in tables:
            self._conn.execute('DROP TABLE IF EXISTS pending_branches')
            self._conn.execute(
                'ALTER TABLE pending_subgroups RENAME TO pending_branches')

        existing = {r['name'] for r in
                    self._conn.execute('PRAGMA table_info(pending_branches)')}
        for col, defn in [('source_word', 'TEXT'),
                          ('source_pattern', 'INTEGER')]:
            if col not in existing:
                self._conn.execute(
                    f'ALTER TABLE pending_branches ADD COLUMN {col} {defn}')

        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        for col, defn in [('chunks_done',      'INTEGER'),
                          ('chunk_idx',        'INTEGER'),
                          ('chunk_started_at', 'INTEGER'),
                          ('cand_rate',        'REAL'),
                          ('cache_hits',       'INTEGER'),
                          ('cache_misses',     'INTEGER'),
                          ('n_pruned',         'INTEGER'),
                          ('n_ok',             'INTEGER'),
                          ('best_guess',        'TEXT'),
                          ('best_erd',         'REAL'),
                          ('cur_candidate',    'TEXT'),
                          ('cand_n_seen',      'INTEGER'),
                          ('cand_chunk_size',  'INTEGER'),
                          ('cur_max_depth',    'INTEGER'),
                          ('cur_nodes',        'INTEGER'),
                          ('node_rate',        'REAL'),
                          ('cur_path',         'TEXT')]:
            if col not in existing_hb:
                self._conn.execute(
                    f'ALTER TABLE worker_heartbeat ADD COLUMN {col} {defn}')

        # Depth-limited ERD: each branch is solved at a guess budget; track the
        # winner's worst-case line length (best_max_depth) and whether the cap
        # excluded any candidate (tainted), aggregated across all workers.
        existing_ab = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(active_branches)')}
        for col, defn in [('budget',         'INTEGER'),
                          ('best_max_depth', 'INTEGER'),
                          ('tainted',        'INTEGER NOT NULL DEFAULT 0')]:
            if col not in existing_ab:
                self._conn.execute(
                    f'ALTER TABLE active_branches ADD COLUMN {col} {defn}')

        # Column renames from earlier terminology alignment: subset_key ->
        # branch_key throughout; best_word -> best_guess; current_subset_key ->
        # current_branch_key in worker_heartbeat.
        existing_ps = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(pending_branches)')}
        if 'subset_key' in existing_ps:
            self._conn.execute(
                'ALTER TABLE pending_branches RENAME COLUMN subset_key TO branch_key')
        existing_ab = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(active_branches)')}
        if 'subset_key' in existing_ab:
            self._conn.execute(
                'ALTER TABLE active_branches RENAME COLUMN subset_key TO branch_key')
        if 'best_word' in existing_ab:
            self._conn.execute(
                'ALTER TABLE active_branches RENAME COLUMN best_word TO best_guess')
        existing_bc = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(branch_chunks)')}
        if 'subset_key' in existing_bc:
            self._conn.execute(
                'ALTER TABLE branch_chunks RENAME COLUMN subset_key TO branch_key')
        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        if 'current_subset_key' in existing_hb:
            self._conn.execute(
                'ALTER TABLE worker_heartbeat '
                'RENAME COLUMN current_subset_key TO current_branch_key')
        if 'best_word' in existing_hb and 'best_guess' not in existing_hb:
            self._conn.execute(
                'ALTER TABLE worker_heartbeat RENAME COLUMN best_word TO best_guess')

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # Bootstrap / populate-queue
    # ------------------------------------------------------------------

    def add_pending_many(self, rows):
        """Insert (branch_key, n_words, priority, source_word, source_pattern) rows.

        Uses an UPSERT so that:
        - A row inserted for the first time is added as 'pending'.
        - A row already present has its priority UPGRADED (never downgraded),
          e.g. a branch first inserted at priority=0 by an earlier root word
          is correctly promoted to priority=1 when a VIP word (SALET) is
          bootstrapped later.
        - source_word / source_pattern record the first root word whose branch
          produced this entry (kept for display in `status`).
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany("""
                INSERT INTO pending_branches
                    (branch_key, n_words, priority, source_word, source_pattern, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(branch_key) DO UPDATE SET
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
        """Atomically claim the highest-priority / largest pending branch.

        Returns a dict {branch_key, n_words, priority, source_word,
        source_pattern} or None if the queue is empty.  In the swarm model the
        claiming worker uses this to PROMOTE a queued branch into an
        active_branches row that other workers can then join; the branch's
        priority and source word/pattern are carried over for display.

        Uses BEGIN IMMEDIATE to acquire the write lock before the SELECT,
        eliminating the TOCTOU race where two workers could both read the same
        'pending' row before either marks it 'in_progress'.  Under contention
        the loser blocks and retries automatically via sqlite3_busy_timeout.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("""
                SELECT branch_key, n_words, priority, source_word, source_pattern
                FROM pending_branches
                WHERE status = 'pending'
                ORDER BY priority DESC, n_words DESC
                LIMIT 1
            """).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            now = int(time.time())
            self._conn.execute("""
                UPDATE pending_branches
                SET status = 'in_progress', claimed_by = ?, claimed_at = ?
                WHERE branch_key = ?
            """, (worker_id, now, row["branch_key"]))
            self._conn.execute("COMMIT")
            return {
                'branch_key': bytes(row["branch_key"]),
                'n_words': row["n_words"],
                'priority': row["priority"],
                'source_word': row["source_word"],
                'source_pattern': row["source_pattern"],
            }
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def mark_done(self, branch_key: bytes):
        now = int(time.time())
        self._conn.execute("""
            UPDATE pending_branches
            SET status = 'done', completed_at = ?
            WHERE branch_key = ?
        """, (now, branch_key))

    def reset_stale_in_progress(self) -> int:
        """Reset any 'in_progress' rows back to 'pending'.

        Called on supervisor startup so a crashed worker's claimed row is
        re-queued rather than stuck forever.  Returns the number of rows reset.
        """
        self._conn.execute("""
            UPDATE pending_branches
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE status = 'in_progress'
        """)
        return self._conn.execute("SELECT changes()").fetchone()[0]

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self, worker_id: str, pid: int,
                  current_branch_key, n_words, started_at: int,
                  chunks_done: int, chunk_idx=None, chunk_started_at=None,
                  cand_rate=None, cache_hits=None, cache_misses=None,
                  n_pruned=None, n_ok=None, best_guess=None, best_erd=None,
                  cur_candidate=None, cand_n_seen=None, cand_chunk_size=None,
                  cur_max_depth=None, cur_nodes=None, node_rate=None,
                  cur_path=None):
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO worker_heartbeat
                (worker_id, pid, current_branch_key, n_words, started_at,
                 updated_at, chunks_done, chunk_idx, chunk_started_at,
                 cand_rate, cache_hits, cache_misses, n_pruned, n_ok,
                 best_guess, best_erd, cur_candidate, cand_n_seen, cand_chunk_size,
                 cur_max_depth, cur_nodes, node_rate, cur_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (worker_id, pid, current_branch_key, n_words, started_at,
              now, chunks_done, chunk_idx, chunk_started_at,
              cand_rate, cache_hits, cache_misses, n_pruned, n_ok,
              best_guess, best_erd, cur_candidate, cand_n_seen, cand_chunk_size,
              cur_max_depth, cur_nodes, node_rate, cur_path))

    def clear_heartbeat(self, worker_id: str):
        self._conn.execute(
            "DELETE FROM worker_heartbeat WHERE worker_id = ?", (worker_id,))

    # ------------------------------------------------------------------
    # Status queries (read-only, safe to call concurrently with workers)
    # ------------------------------------------------------------------

    def counts_by_status(self) -> dict:
        return {r["status"]: r["c"] for r in self._conn.execute(
            "SELECT status, COUNT(*) c FROM pending_branches GROUP BY status"
        )}

    def heartbeats_with_branch(self):
        """Heartbeat rows joined to the branch each worker is contributing to.

        source_word/pattern/priority come from active_branches (the branch the
        worker is on), so the health display can label a worker by the branch
        it's helping rather than by an opaque key.
        """
        return self._conn.execute("""
            SELECT h.*,
                   b.priority,
                   b.source_word,
                   b.source_pattern
            FROM worker_heartbeat h
            LEFT JOIN active_branches b
                   ON h.current_branch_key = b.branch_key
            ORDER BY h.worker_id
        """).fetchall()

    # ------------------------------------------------------------------
    # Branch swarm: cooperative candidate-level solve of one branch
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_size_for(n_words, n_candidates,
                       min_words_per_chunk=3, max_chunk_count=256) -> int:
        """Candidates-per-chunk for a branch, from its difficulty (n_words).

        Candidates are ranked best-first, so the early chunks hold the
        expensive (fully-recursed) candidates and the tail is cheap (pruned by
        the shared bound).  Easy branches (few words) become a single chunk so
        one worker disposes of them without coordination overhead; hard
        branches are cut into many chunks so the expensive head spreads across
        workers.

        n_chunks = clamp(ceil(n_words / min_words_per_chunk), 1, max_chunk_count)
        chunk_size = ceil(n_candidates / n_chunks)

        min_words_per_chunk controls granularity: lower values produce more
        chunks and more worker sharing on hard branches.  max_chunk_count caps
        the total chunk count regardless of branch size.  When both are
        supplied and conflict, max_chunk_count wins (chunks become larger).
        """
        n_chunks = max(1, min(max_chunk_count, -(-n_words // min_words_per_chunk)))
        return -(-n_candidates // n_chunks)

    @staticmethod
    def n_chunks_for(n_candidates: int, chunk_size: int) -> int:
        return (n_candidates + chunk_size - 1) // chunk_size

    @staticmethod
    def chunk_range(idx: int, chunk_size: int, n_candidates: int):
        lo = idx * chunk_size
        hi = min(lo + chunk_size, n_candidates)
        return lo, hi

    def create_branch(self, branch_key, n_words, n_candidates, chunk_size,
                      priority=0, source_word=None, source_pattern=None,
                      budget=None) -> bool:
        """Register a branch as in-progress (status 'open'), if not present.

        Idempotent via INSERT OR IGNORE: the worker that promoted the branch
        from the queue creates it; others that race simply see it exists and
        join.  Returns True if this call created the row.  budget is the guess
        budget the branch is solved under (depth-limited ERD).
        """
        now = int(time.time())
        cur = self._conn.execute("""
            INSERT OR IGNORE INTO active_branches
                (branch_key, n_words, n_candidates, chunk_size,
                 priority, source_word, source_pattern, status, created_at, budget)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
        """, (branch_key, n_words, n_candidates, chunk_size,
              priority, source_word, source_pattern, now, budget))
        return cur.rowcount == 1

    def get_branch(self, branch_key):
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_key = ?",
            (branch_key,)).fetchone()

    def claim_chunk(self, branch_key, worker_id, n_chunks):
        """Atomically claim the lowest-indexed chunk that has no row yet.

        A chunk with an existing row is either in-flight (done=0) or complete
        (done=1); either way it isn't re-handed-out here.  Stale done=0 rows
        are freed by reclaim_stale_chunks, which deletes them so they reappear
        as gaps.  Returns the chunk idx, or None if every chunk already has a
        row (this branch is fully claimed — the worker should look elsewhere,
        NEVER block).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # Never hand out (and thereby re-create) a chunk for a branch that
            # has been finalized and deleted: a worker still looping on it would
            # otherwise redo the whole branch from scratch.  Checked inside the
            # write transaction so it can't race the finalize+delete.
            br = self._conn.execute(
                "SELECT status FROM active_branches WHERE branch_key = ?",
                (branch_key,)).fetchone()
            if br is None or br["status"] != "open":
                self._conn.execute("COMMIT")
                return None
            taken = {r["idx"] for r in self._conn.execute(
                "SELECT idx FROM branch_chunks WHERE branch_key = ?",
                (branch_key,))}
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
                INSERT INTO branch_chunks
                    (branch_key, idx, claimed_by, claimed_at, done)
                VALUES (?, ?, ?, ?, 0)
            """, (branch_key, idx, worker_id, now))
            self._conn.execute("COMMIT")
            return idx
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def complete_chunk(self, branch_key, idx):
        """Mark a chunk authoritatively complete (done=1)."""
        now = int(time.time())
        self._conn.execute("""
            UPDATE branch_chunks SET done = 1, done_at = ?
            WHERE branch_key = ? AND idx = ?
        """, (now, branch_key, idx))

    def update_branch_best(self, branch_key, best_guess, best_erd, max_depth=None):
        """Lower the branch's running best (monotone — never raises it).

        max_depth is the winning candidate's worst-case line length; it is
        stored atomically with the best it belongs to, so best_max_depth always
        describes the current best_guess.
        """
        self._conn.execute("""
            UPDATE active_branches
            SET best_erd = ?, best_guess = ?, best_max_depth = ?
            WHERE branch_key = ?
              AND (best_erd IS NULL OR ? < best_erd)
        """, (best_erd, best_guess, max_depth, branch_key, best_erd))

    def read_branch_best(self, branch_key):
        """Return (best_guess, best_erd) or (None, None)."""
        row = self._conn.execute(
            "SELECT best_guess, best_erd FROM active_branches WHERE branch_key = ?",
            (branch_key,)).fetchone()
        if row is None:
            return (None, None)
        return (row["best_guess"], row["best_erd"])

    def mark_branch_tainted(self, branch_key):
        """Set the branch's taint flag (monotone OR): some candidate, in some
        worker, was excluded by the depth cap, so the branch's ERD is only
        valid at its solve budget."""
        self._conn.execute(
            "UPDATE active_branches SET tainted = 1 WHERE branch_key = ?",
            (branch_key,))

    def read_branch_meta(self, branch_key):
        """Return (best_guess, best_erd, best_max_depth, tainted, budget) or
        None — everything finalize needs to write a depth-limited cache entry."""
        row = self._conn.execute(
            "SELECT best_guess, best_erd, best_max_depth, tainted, budget "
            "FROM active_branches WHERE branch_key = ?", (branch_key,)).fetchone()
        if row is None:
            return None
        return (row["best_guess"], row["best_erd"], row["best_max_depth"],
                bool(row["tainted"]), row["budget"])

    def branch_done_chunks(self, branch_key) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM branch_chunks WHERE branch_key = ? AND done = 1",
            (branch_key,)).fetchone()[0]

    def try_finalize_branch(self, branch_key) -> bool:
        """Atomically transition a branch open -> finalized, exactly once.

        Returns True only for the single caller that wins the transition; that
        caller writes the branch's ERD entry to the persistent cache and then
        calls delete_branch.  Caller must have confirmed full chunk coverage
        first; the WHERE status='open' guard makes the finalize idempotent.
        """
        now = int(time.time())
        cur = self._conn.execute("""
            UPDATE active_branches SET status = 'finalized', finalized_at = ?
            WHERE branch_key = ? AND status = 'open'
        """, (now, branch_key))
        return cur.rowcount == 1

    def delete_branch(self, branch_key):
        """Remove a finished branch and its chunk rows to bound the queue DB."""
        self._conn.execute(
            "DELETE FROM branch_chunks WHERE branch_key = ?", (branch_key,))
        self._conn.execute(
            "DELETE FROM active_branches WHERE branch_key = ?", (branch_key,))

    def reclaim_stale_chunks(self, heartbeat_timeout_seconds: int,
                             min_claim_age_seconds: int = None) -> int:
        """Free in-flight (done=0) chunk claims whose worker is no longer alive.

        Liveness is proved by the worker's heartbeat (worker_heartbeat.updated_at,
        refreshed every couple of seconds while it works).  A done=0 chunk is
        reclaimed only when its claiming worker has NOT heartbeat within
        heartbeat_timeout_seconds — i.e. it has crashed or hung.  Crucially, a
        slow-but-alive worker (one still heartbeating) is never reclaimed: doing
        so would let a second worker re-evaluate the same slice and finalize the
        branch BEFORE the original folds in a better candidate, writing a
        suboptimal ERD to the permanent cache.

        min_claim_age_seconds (default: heartbeat_timeout_seconds) is a floor on
        claim age, so a freshly claimed chunk isn't reclaimed in the brief window
        before its worker's first heartbeat lands.  done=1 rows are never
        touched.  Returns the number of claims freed.
        """
        now = int(time.time())
        hb_cutoff = now - heartbeat_timeout_seconds
        age_floor = now - (min_claim_age_seconds
                           if min_claim_age_seconds is not None
                           else heartbeat_timeout_seconds)
        self._conn.execute("""
            DELETE FROM branch_chunks
            WHERE done = 0
              AND claimed_at < ?
              AND claimed_by NOT IN (
                  SELECT worker_id FROM worker_heartbeat
                  WHERE updated_at >= ?
              )
        """, (age_floor, hb_cutoff))
        return self._conn.execute("SELECT changes()").fetchone()[0]

    def reclaim_chunks_of_worker(self, worker_id: str) -> int:
        """Free all in-flight (done=0) chunk claims held by a specific worker.

        Called by the supervisor when it kills/respawns a worker, so that
        instance's chunks are freed deterministically BEFORE a replacement of
        the same name starts heartbeating (which would otherwise make the dead
        instance's claims look live again).  done=1 rows are never touched.
        """
        self._conn.execute(
            "DELETE FROM branch_chunks WHERE done = 0 AND claimed_by = ?",
            (worker_id,))
        return self._conn.execute("SELECT changes()").fetchone()[0]

    def branches_in_progress(self):
        """Open branches, highest priority first — for swarm scheduling."""
        return self._conn.execute("""
            SELECT * FROM active_branches WHERE status = 'open'
            ORDER BY priority DESC, n_words DESC
        """).fetchall()

    def reset_active_branches(self):
        """Drop all in-progress branch + chunk state (supervisor startup).

        Branch/chunk rows are pure transient coordination: any branch whose
        result didn't reach the persistent cache is still 'pending' (its
        pending_branches row was set in_progress on promotion and reset to
        pending by reset_stale_in_progress()), so clearing these tables just
        discards half-done coordination — the branch is simply re-promoted and
        redone.  Returns (n_branches, n_chunks) cleared.
        """
        nb = self._conn.execute(
            "SELECT COUNT(*) FROM active_branches").fetchone()[0]
        nc = self._conn.execute(
            "SELECT COUNT(*) FROM branch_chunks").fetchone()[0]
        self._conn.execute("DELETE FROM branch_chunks")
        self._conn.execute("DELETE FROM active_branches")
        return nb, nc

    def worker_counts_by_branch(self, timeout_seconds: int = 30) -> dict:
        """{branch_key bytes: number of recent workers on it} for status."""
        cutoff = int(time.time()) - timeout_seconds
        rows = self._conn.execute("""
            SELECT current_branch_key AS k, COUNT(*) AS c
            FROM worker_heartbeat
            WHERE current_branch_key IS NOT NULL AND updated_at > ?
            GROUP BY current_branch_key
        """, (cutoff,)).fetchall()
        return {bytes(r["k"]): r["c"] for r in rows}

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

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def clear(self):
        """Wipe all queue state: pending/done branches, active branches,
        chunk claims, heartbeats, and run_meta.

        The persistent cache (wordle_cache.sqlite3) is not touched — only
        the transient coordination tables in erd_queue.sqlite3.
        """
        self._conn.execute("DELETE FROM branch_chunks")
        self._conn.execute("DELETE FROM active_branches")
        self._conn.execute("DELETE FROM pending_branches")
        self._conn.execute("DELETE FROM worker_heartbeat")
        self._conn.execute("DELETE FROM run_meta")

    def total_branches(self) -> int:
        """Total rows in pending_branches (all statuses)."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM pending_branches").fetchone()[0]

    def get_pending_branch(self, branch_key: bytes):
        """Return the pending_branches row for branch_key, or None."""
        return self._conn.execute(
            "SELECT * FROM pending_branches WHERE branch_key = ?",
            (branch_key,)
        ).fetchone()

    def get_active_branch(self, branch_key: bytes):
        """Return the active_branches row for branch_key, or None."""
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_key = ?",
            (branch_key,)
        ).fetchone()

    def chunks_for_branch(self, branch_key: bytes):
        """Return all branch_chunks rows for branch_key."""
        return self._conn.execute(
            "SELECT * FROM branch_chunks WHERE branch_key = ? ORDER BY idx",
            (branch_key,)
        ).fetchall()

    def cancel_active_branch(self, branch_key: bytes,
                             remove_from_queue: bool = False):
        """Atomically remove a branch's chunk claims and active_branches row.

        All DELETEs run in one transaction so a crash partway through cannot
        leave orphaned branch_chunks rows or a dangling active_branches row.

        With remove_from_queue=True, also deletes the pending_branches row
        (regardless of its status), fully removing the branch from the queue in
        the same transaction.
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "DELETE FROM branch_chunks WHERE branch_key = ?", (branch_key,))
            self._conn.execute(
                "DELETE FROM active_branches WHERE branch_key = ?", (branch_key,))
            if remove_from_queue:
                self._conn.execute(
                    "DELETE FROM pending_branches WHERE branch_key = ?",
                    (branch_key,))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def set_priority(self, branch_key: bytes, priority: int) -> bool:
        """Update the priority of a pending branch.

        Only updates rows with status='pending' — in-progress and done branches
        are ignored (priority is only read at claim time).  Returns True if a
        pending row was updated, False if the branch was not found or is not
        pending.
        """
        self._conn.execute(
            "UPDATE pending_branches SET priority = ? "
            "WHERE branch_key = ? AND status = 'pending'",
            (priority, branch_key))
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    def remove_pending(self, branch_key: bytes) -> bool:
        """Delete a pending (status='pending') branch from the queue.

        Returns True if a row was deleted.  Does not touch active_branches or
        branch_chunks — call reset_active_branches() first if the branch is
        currently in progress.
        """
        self._conn.execute(
            "DELETE FROM pending_branches "
            "WHERE branch_key = ? AND status = 'pending'",
            (branch_key,))
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0
