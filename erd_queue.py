"""erd_queue.py — Sidecar SQLite coordination queue for the parallel ERD_ALL precache job.

Lives in erd_queue.sqlite3, separate from wordle_cache.sqlite3, so the
high-frequency coordination writes (claim/heartbeat/done) don't contend with
the high-volume ERD result writes in the main cache.
"""

from __future__ import annotations

import math
import sqlite3
import time

from cache_sqlite import ScoreCache

# Time-weighted geometric mean EMA: half-life for the cost model.
_COST_MODEL_TAU = 86400.0          # seconds (≈ 1 day)
# Effective-weight below which a cost-model bucket reads cold (no prediction).
_COST_MODEL_MIN_WEIGHT = 1.0

# Geometric size bucketing.  Sub-branch sizes are sparse and heavy-tailed, so a
# bucket per exact word-count would almost never accumulate enough samples to
# leave "cold".  Bucketing by floor(log(n)/log(BASE)) groups nearby sizes into
# one accumulator, keeping samples dense in the heavy small-size region while
# still separating sizes that differ by more than a ~30% step.
_COST_MODEL_BUCKET_BASE = 1.3
_LOG_BUCKET_BASE = math.log(_COST_MODEL_BUCKET_BASE)


def cost_size_bucket(n_words: int) -> int:
    """Map a branch word-count to its geometric cost-model bucket index."""
    if n_words < 1:
        return 0
    return int(math.log(n_words) / _LOG_BUCKET_BASE)

# Re-export so callers don't need to import cache_sqlite directly.
encode_subset = ScoreCache.encode_subset


def decode_subset(blob: bytes) -> list[str]:
    """Reverse ScoreCache.encode_subset: split fixed-5-byte words."""
    return [blob[i:i + 5].decode() for i in range(0, len(blob), 5)]



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
-- worker is a fungible contributor: it reports which branch and claim it is
-- on purely so the operator can see it is alive and moving (health), not as
-- the unit of progress (that lives in active_branches).  The metric columns
-- (cache_hits/misses, n_cutoff/n_pruned/n_ok) let `status` aggregate
-- cache effectiveness and branch-and-bound pruning across all workers.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id          TEXT    PRIMARY KEY,
    pid                INTEGER NOT NULL,
    current_branch_key BLOB,        -- branch key this worker is contributing to
    n_words            INTEGER,
    started_at         INTEGER,
    updated_at         INTEGER NOT NULL,
    claims_done        INTEGER NOT NULL DEFAULT 0,
    claim_idx          INTEGER,     -- candidate index currently held
    claim_started_at   INTEGER,     -- when it was claimed (held-time = now - this)
    cand_rate          REAL,        -- candidates/sec, recent (legacy; see node_rate)
    cache_hits         INTEGER,
    cache_misses       INTEGER,
    n_cutoff           INTEGER,     -- alpha-beta: cost >= best_erd before full eval
    n_pruned           INTEGER,     -- infeasible within budget (depth floor hit)
    n_ok               INTEGER,     -- candidates fully evaluated
    best_guess          TEXT,
    best_erd           REAL,        -- ERD of the locally-found best candidate
    bound_erd          REAL,        -- effective pruning bound: min(local, shared)
    cur_candidate      TEXT,        -- candidate word currently under evaluation
    cand_n_seen        INTEGER,     -- reserved (always 1 with single-candidate claims)
    claim_total        INTEGER,     -- reserved (always 1 with single-candidate claims)
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
-- evaluating a disjoint set of candidates from the policy-canonical list.
-- best_erd is the running-minimum cost across all candidates tried so far: it
-- only ever decreases and is a real achieved value, so any worker may read it
-- as a branch-and-bound bound.  status open -> finalized exactly once, by the
-- worker that observes full candidate coverage; that worker writes the branch's
-- ERD entry to the persistent cache, then the row (and its claims) is deleted.
-- Claim order is by all_words (file) index for ERD_ALL: each worker claims one
-- candidate index at a time from {0..n_candidates-1}; no ranking required.
CREATE TABLE IF NOT EXISTS active_branches (
    branch_key     BLOB    PRIMARY KEY,
    n_words        INTEGER NOT NULL,
    n_candidates   INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    best_erd       REAL,
    best_guess     TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER,
    budget         INTEGER,
    best_max_depth INTEGER,
    tainted        INTEGER NOT NULL DEFAULT 0,
    depth          INTEGER NOT NULL DEFAULT 0,
    nodes_spent    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
    ON active_branches(status, priority DESC, n_words DESC);

-- One row per claimed candidate of a branch.  A row's existence is an
-- ADVISORY claim ("a worker is probably evaluating this candidate"); only
-- done=1 is AUTHORITATIVE ("fully evaluated and folded into best_erd").
-- Coverage = all n_candidates slots with done=1.  A crashed worker leaves a
-- done=0 row that stale-reclaim deletes, turning it back into an unclaimed
-- gap to be redone — never skipped.
-- idx = index into the policy-canonical candidate list (all_words for ERD_ALL).
CREATE TABLE IF NOT EXISTS candidate_claims (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    PRIMARY KEY (branch_key, idx)
);

-- Per-size-bucket online cost model (time-weighted geometric mean of
-- recursion-node cost).  Keyed by (policy, size_bucket) so ERD_ALL and
-- ERD_ANSWERS models never cross-contaminate.
CREATE TABLE IF NOT EXISTS cost_model (
    policy           TEXT    NOT NULL,
    size_bucket      INTEGER NOT NULL,
    weighted_log_sum REAL    NOT NULL DEFAULT 0,
    weight_sum       REAL    NOT NULL DEFAULT 0,
    weighted_log_sq  REAL    NOT NULL DEFAULT 0,
    last_updated     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (policy, size_bucket)
);

-- Raw per-solve samples for offline distribution analysis and threshold tuning.
CREATE TABLE IF NOT EXISTS cost_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    policy      TEXT    NOT NULL,
    n_words     INTEGER NOT NULL,
    nodes       INTEGER NOT NULL,
    wall_millis INTEGER,
    source      TEXT,        -- 'inline' or 'finalize'
    recorded_at INTEGER NOT NULL
);

-- Outbound-only coordination telemetry for offline clustering-decision analysis.
-- Never read by any runtime control path; freely droppable.
CREATE TABLE IF NOT EXISTS claim_telemetry (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words             INTEGER NOT NULL,
    coordination_millis INTEGER NOT NULL,
    work_nodes          INTEGER NOT NULL,
    claim_retries      INTEGER,
    worker_count       INTEGER,
    recorded_at        INTEGER NOT NULL
);

-- One row per wall-clock backstop firing: a frame handed off its remainder
-- because it ran longer than COLD_BACKSTOP_SECONDS rather than because the
-- node-proportionate overrun check tripped.  Exists to tune COLD_BACKSTOP_SECONDS
-- offline: how often the time cap (not the node check) drives a handoff, at what
-- frame sizes, and whether the cost model was cold (predicted_nodes NULL) or warm
-- at the time.  Outbound-only; never read by any runtime control path.
CREATE TABLE IF NOT EXISTS backstop_telemetry (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words              INTEGER NOT NULL,
    depth                INTEGER,
    elapsed_millis       INTEGER NOT NULL,   -- wall time in frame at fire
    nodes                INTEGER NOT NULL,   -- nodes spent in frame at fire
    predicted_nodes      REAL,               -- typical(n) at entry; NULL = cold model
    remaining_candidates INTEGER NOT NULL,   -- candidates handed off
    recorded_at          INTEGER NOT NULL
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
        pass

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # Populate queue
    # ------------------------------------------------------------------

    def add_pending_many(self, rows):
        """Insert (branch_key, n_words, priority, source_word, source_pattern) rows.

        Uses an UPSERT so that:
        - A row inserted for the first time is added as 'pending'.
        - A row already present has its priority UPGRADED (never downgraded),
          e.g. a branch first inserted at priority=0 by an earlier root word
          is correctly promoted to priority=1 when a VIP word (SALET) is
          queued later.
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
        except Exception:  # pragma: no cover
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
        except Exception:  # pragma: no cover
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
                  claims_done: int, claim_idx=None, claim_started_at=None,
                  cand_rate=None, cache_hits=None, cache_misses=None,
                  n_cutoff=None, n_pruned=None, n_ok=None,
                  best_guess=None, best_erd=None, bound_erd=None,
                  cur_candidate=None, cand_n_seen=None, claim_total=None,
                  cur_max_depth=None, cur_nodes=None, node_rate=None,
                  cur_path=None):
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO worker_heartbeat
                (worker_id, pid, current_branch_key, n_words, started_at,
                 updated_at, claims_done, claim_idx, claim_started_at,
                 cand_rate, cache_hits, cache_misses, n_cutoff, n_pruned, n_ok,
                 best_guess, best_erd, bound_erd, cur_candidate, cand_n_seen, claim_total,
                 cur_max_depth, cur_nodes, node_rate, cur_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (worker_id, pid, current_branch_key, n_words, started_at,
              now, claims_done, claim_idx, claim_started_at,
              cand_rate, cache_hits, cache_misses, n_cutoff, n_pruned, n_ok,
              best_guess, best_erd, bound_erd, cur_candidate, cand_n_seen, claim_total,
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

    def create_branch(self, branch_key, n_words, n_candidates,
                      priority=0, source_word=None, source_pattern=None,
                      budget=None, depth=0) -> bool:
        """Register a branch as in-progress (status 'open'), if not present.

        Idempotent via INSERT OR IGNORE: the worker that promoted the branch
        from the queue creates it; others that race simply see it exists and
        join.  Returns True if this call created the row.  n_candidates is the
        total claim slot count (one slot per candidate in the policy-canonical
        list).  budget is the guess budget for depth-limited ERD.  depth is
        the cooperative nesting level: 0 for user-queued branches, 1+ for
        branches promoted inside cooperative_solve.
        """
        now = int(time.time())
        cur = self._conn.execute("""
            INSERT OR IGNORE INTO active_branches
                (branch_key, n_words, n_candidates,
                 priority, source_word, source_pattern, status, created_at, budget, depth)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """, (branch_key, n_words, n_candidates,
              priority, source_word, source_pattern, now, budget, depth))
        return cur.rowcount == 1

    def get_branch(self, branch_key):
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_key = ?",
            (branch_key,)).fetchone()

    def claim_candidate(self, branch_key, worker_id, n_candidates):
        """Atomically claim the lowest-indexed candidate slot that has no row yet.

        A slot with an existing row is either in-flight (done=0) or complete
        (done=1); either way it isn't re-handed-out here.  Stale done=0 rows
        are freed by reclaim_stale_claims, which deletes them so they reappear
        as gaps.  Returns the claimed idx (= index into the policy-canonical
        candidate list), or None if every slot already has a row (fully claimed
        — the worker should look elsewhere, NEVER block).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # Never hand out a claim for a branch that has been finalized and
            # deleted: a worker still looping would otherwise redo it from scratch.
            # Checked inside the write transaction so it can't race finalize+delete.
            br = self._conn.execute(
                "SELECT status FROM active_branches WHERE branch_key = ?",
                (branch_key,)).fetchone()
            if br is None or br["status"] != "open":
                self._conn.execute("COMMIT")
                return None
            taken = {r["idx"] for r in self._conn.execute(
                "SELECT idx FROM candidate_claims WHERE branch_key = ?",
                (branch_key,))}
            idx = None
            for c in range(n_candidates):
                if c not in taken:
                    idx = c
                    break
            if idx is None:
                self._conn.execute("COMMIT")
                return None
            now = int(time.time())
            self._conn.execute("""
                INSERT INTO candidate_claims
                    (branch_key, idx, claimed_by, claimed_at, done)
                VALUES (?, ?, ?, ?, 0)
            """, (branch_key, idx, worker_id, now))
            self._conn.execute("COMMIT")
            return idx
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def complete_candidate(self, branch_key, idx):
        """Mark a candidate claim authoritatively complete (done=1)."""
        now = int(time.time())
        self._conn.execute("""
            UPDATE candidate_claims SET done = 1, done_at = ?
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

    def branch_done_candidates(self, branch_key) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM candidate_claims WHERE branch_key = ? AND done = 1",
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
        """Remove a finished branch and its claim rows to bound the queue DB."""
        self._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_key = ?", (branch_key,))
        self._conn.execute(
            "DELETE FROM active_branches WHERE branch_key = ?", (branch_key,))

    def reclaim_stale_claims(self, heartbeat_timeout_seconds: int,
                             min_claim_age_seconds: int = None) -> int:
        """Free in-flight (done=0) candidate claims whose worker is no longer alive.

        Liveness is proved by the worker's heartbeat (worker_heartbeat.updated_at,
        refreshed every couple of seconds while it works).  A done=0 claim is
        reclaimed only when its claiming worker has NOT heartbeat within
        heartbeat_timeout_seconds — i.e. it has crashed or hung.  A slow-but-alive
        worker (still heartbeating) is never reclaimed: doing so would let a
        second worker re-evaluate the same candidate and finalize the branch BEFORE
        the original folds in a better result, writing a suboptimal ERD to cache.

        min_claim_age_seconds (default: heartbeat_timeout_seconds) is a floor on
        claim age, so a freshly claimed candidate isn't reclaimed in the brief
        window before its worker's first heartbeat lands.  done=1 rows are never
        touched.  Returns the number of claims freed.
        """
        now = int(time.time())
        hb_cutoff = now - heartbeat_timeout_seconds
        age_floor = now - (min_claim_age_seconds
                           if min_claim_age_seconds is not None
                           else heartbeat_timeout_seconds)
        self._conn.execute("""
            DELETE FROM candidate_claims
            WHERE done = 0
              AND claimed_at < ?
              AND claimed_by NOT IN (
                  SELECT worker_id FROM worker_heartbeat
                  WHERE updated_at >= ?
              )
        """, (age_floor, hb_cutoff))
        return self._conn.execute("SELECT changes()").fetchone()[0]

    def reclaim_claims_of_worker(self, worker_id: str) -> int:
        """Free all in-flight (done=0) candidate claims held by a specific worker.

        Called by the supervisor when it kills/respawns a worker, so that
        instance's claims are freed deterministically BEFORE a replacement of
        the same name starts heartbeating (which would otherwise make the dead
        instance's claims look live again).  done=1 rows are never touched.
        """
        self._conn.execute(
            "DELETE FROM candidate_claims WHERE done = 0 AND claimed_by = ?",
            (worker_id,))
        return self._conn.execute("SELECT changes()").fetchone()[0]

    def branches_in_progress(self):
        """Open branches, highest priority first — for swarm scheduling."""
        return self._conn.execute("""
            SELECT * FROM active_branches WHERE status = 'open'
            ORDER BY priority DESC, n_words DESC
        """).fetchall()

    def reset_active_branches(self):
        """Drop in-progress D-0 branch state; free stale cooperative claims.

        D-0 branches have pending_branches rows that reset_stale_in_progress()
        already flipped back to 'pending', so wiping their active_branches and
        claim rows just discards half-done coordination — they re-promote and
        redo cleanly.

        Cooperative branches (depth>0) have NO pending_branches row — they are
        inserted directly into active_branches by cooperative_solve.  Wiping
        them throws away all partial-claim progress with no recovery path.
        Instead, only free their stale in-flight claims (done=0 rows);
        done=1 rows and the branch row itself survive, so the next worker to
        join picks up exactly where the killed worker left off.

        Returns (n_branches, n_claims) cleared for D-0 branches only.
        """
        nb = self._conn.execute(
            "SELECT COUNT(*) FROM active_branches WHERE depth = 0").fetchone()[0]
        nc = self._conn.execute(
            "SELECT COUNT(*) FROM candidate_claims WHERE branch_key IN "
            "(SELECT branch_key FROM active_branches WHERE depth = 0)").fetchone()[0]
        self._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_key IN "
            "(SELECT branch_key FROM active_branches WHERE depth = 0)")
        self._conn.execute("DELETE FROM active_branches WHERE depth = 0")
        # Free stale in-flight claims on cooperative branches so their
        # remaining candidates are reclaimable as gaps.
        self._conn.execute(
            "DELETE FROM candidate_claims WHERE done = 0")
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
        self._conn.execute("DELETE FROM candidate_claims")
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

    def status_by_branch_keys(self, branch_keys) -> dict:
        """Return {branch_key: pending_branches row} for the given keys.

        A branch_key with no row was never queued; it is simply absent from
        the returned dict.
        """
        if not branch_keys:
            return {}
        placeholders = ','.join('?' for _ in branch_keys)
        rows = self._conn.execute(
            f"SELECT * FROM pending_branches WHERE branch_key IN ({placeholders})",
            list(branch_keys)
        ).fetchall()
        return {bytes(r["branch_key"]): r for r in rows}

    def active_branches_by_keys(self, branch_keys) -> dict:
        """Return {branch_key: active_branches row} for the given keys.

        Only open branches appear; finalized branches are deleted from this
        table and will be absent from the returned dict.
        """
        if not branch_keys:
            return {}
        placeholders = ','.join('?' for _ in branch_keys)
        rows = self._conn.execute(
            f"SELECT * FROM active_branches WHERE branch_key IN ({placeholders})",
            list(branch_keys)
        ).fetchall()
        return {bytes(r["branch_key"]): r for r in rows}

    def get_active_branch(self, branch_key: bytes):
        """Return the active_branches row for branch_key, or None."""
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_key = ?",
            (branch_key,)
        ).fetchone()

    def claims_for_branch(self, branch_key: bytes):
        """Return all candidate_claims rows for branch_key."""
        return self._conn.execute(
            "SELECT * FROM candidate_claims WHERE branch_key = ? ORDER BY idx",
            (branch_key,)
        ).fetchall()

    def cancel_active_branch(self, branch_key: bytes,
                             remove_from_queue: bool = False):
        """Atomically remove a branch's chunk claims and active_branches row.

        All DELETEs run in one transaction so a crash partway through cannot
        leave orphaned candidate_claims rows or a dangling active_branches row.

        With remove_from_queue=True, also deletes the pending_branches row
        (regardless of its status), fully removing the branch from the queue in
        the same transaction.
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "DELETE FROM candidate_claims WHERE branch_key = ?", (branch_key,))
            self._conn.execute(
                "DELETE FROM active_branches WHERE branch_key = ?", (branch_key,))
            if remove_from_queue:
                self._conn.execute(
                    "DELETE FROM pending_branches WHERE branch_key = ?",
                    (branch_key,))
            self._conn.execute("COMMIT")
        except Exception:  # pragma: no cover
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
        candidate_claims — call reset_active_branches() first if the branch is
        currently in progress.
        """
        self._conn.execute(
            "DELETE FROM pending_branches "
            "WHERE branch_key = ? AND status = 'pending'",
            (branch_key,))
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    # ------------------------------------------------------------------
    # Mid-loop publisher support
    # ------------------------------------------------------------------

    def mark_claims_done(self, branch_key: bytes, indices):
        """Insert already-evaluated candidates as authoritative done=1 claims.

        Called by the mid-loop publisher to record the candidates evaluated
        inline before overrun fired.  Uses INSERT OR REPLACE so a racing
        in-flight (done=0) claim from another worker is superseded by the
        authoritative done=1 record — the evaluation already happened.
        """
        now = int(time.time())
        self._conn.executemany("""
            INSERT OR REPLACE INTO candidate_claims
                (branch_key, idx, claimed_by, claimed_at, done, done_at)
            VALUES (?, ?, 'publisher', ?, 1, ?)
        """, [(branch_key, idx, now, now) for idx in indices])

    def add_nodes_spent(self, branch_key: bytes, delta: int):
        """Increment nodes_spent on an active branch for cost-model sampling."""
        if delta <= 0:
            return
        self._conn.execute(
            "UPDATE active_branches SET nodes_spent = nodes_spent + ? "
            "WHERE branch_key = ?", (delta, branch_key))

    # ------------------------------------------------------------------
    # Cost model (online time-weighted geometric mean per size bucket)
    # ------------------------------------------------------------------

    def _cost_bucket_row(self, policy: str, n_words: int):
        return self._conn.execute(
            "SELECT weighted_log_sum, weight_sum, weighted_log_sq, last_updated "
            "FROM cost_model WHERE policy = ? AND size_bucket = ?",
            (policy, cost_size_bucket(n_words))).fetchone()

    def get_cost_typical(self, policy: str, n_words: int):
        """Return the geometric-mean node count for n_words' size bucket, or None.

        The estimate is exp(weighted_log_sum / weight_sum).  When weight_sum is
        below _COST_MODEL_MIN_WEIGHT the bucket reads cold and None is returned —
        the caller should fall back to a size-based heuristic.
        """
        row = self._cost_bucket_row(policy, n_words)
        if row is None or row['weight_sum'] < _COST_MODEL_MIN_WEIGHT:
            return None
        return math.exp(row['weighted_log_sum'] / row['weight_sum'])

    def get_cost_spread(self, policy: str, n_words: int):
        """Std-dev of ln(nodes) for n_words' bucket (the log-normal sigma), or None.

        Recovered from the stored second log-moment:
            sigma^2 = weighted_log_sq/weight_sum - mu^2
        Round-off can make this marginally negative when every sample is equal;
        clamp to 0.  Used for the over-promotion shade exp(mu - Z*sigma) and for
        offline distribution analysis.
        """
        row = self._cost_bucket_row(policy, n_words)
        if row is None or row['weight_sum'] < _COST_MODEL_MIN_WEIGHT:
            return None
        mu = row['weighted_log_sum'] / row['weight_sum']
        var = row['weighted_log_sq'] / row['weight_sum'] - mu * mu
        return math.sqrt(var) if var > 0 else 0.0

    def update_cost_model(self, policy: str, n_words: int, nodes: int,
                          weight: float = 1.0, now: int = None):
        """Fold one node-cost sample (value `nodes`, multiplicity `weight`).

        weight > 1 records `weight` identical samples of `nodes` in one call.
        For a batch of *distinct* samples whose individual magnitudes matter to
        the spread, use update_cost_model_logsums so each sample reaches the
        second log-moment without a lossy pre-averaging collapse.
        """
        if nodes <= 0 or weight <= 0:
            return
        log_n = math.log(nodes)
        self._fold_cost_sample(policy, n_words,
                               log_n * weight, log_n * log_n * weight, weight, now)

    def update_cost_model_logsums(self, policy: str, n_words: int,
                                  log_sum: float, log_sq_sum: float,
                                  weight: float, now: int = None):
        """Fold a pre-summed batch of log samples: (Σ ln x, Σ ln²x, count).

        The worker's inline-sample buffer accumulates these sums directly, so the
        batch contributes to weighted_log_sum and weighted_log_sq exactly as if
        each sample had been folded individually — no exp/int/log round-trip.
        """
        if weight <= 0:
            return
        self._fold_cost_sample(policy, n_words, log_sum, log_sq_sum, weight, now)

    def _fold_cost_sample(self, policy, n_words, d_log_sum, d_log_sq, d_weight, now):
        bucket = cost_size_bucket(n_words)
        if now is None:
            now = int(time.time())
        row = self._conn.execute(
            "SELECT weighted_log_sum, weight_sum, weighted_log_sq, last_updated "
            "FROM cost_model WHERE policy = ? AND size_bucket = ?",
            (policy, bucket)).fetchone()
        if row is None:
            self._conn.execute("""
                INSERT INTO cost_model
                    (policy, size_bucket, weighted_log_sum, weight_sum,
                     weighted_log_sq, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (policy, bucket, d_log_sum, d_weight, d_log_sq, now))
        else:
            # Continuous-time EMA: decay every accumulator by the age of the
            # bucket before folding the new contribution.  Clamp elapsed at 0 so
            # an out-of-order timestamp can never amplify (decay > 1).
            decay = math.exp(-max(0, now - row['last_updated']) / _COST_MODEL_TAU)
            self._conn.execute("""
                UPDATE cost_model
                SET weighted_log_sum = ?, weight_sum = ?, weighted_log_sq = ?,
                    last_updated = ?
                WHERE policy = ? AND size_bucket = ?
            """, (decay * row['weighted_log_sum'] + d_log_sum,
                  decay * row['weight_sum'] + d_weight,
                  decay * row['weighted_log_sq'] + d_log_sq,
                  now, policy, bucket))

    def add_cost_sample(self, policy: str, n_words: int, nodes: int, source: str):
        """Append a raw sample to cost_samples for offline analysis."""
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO cost_samples (policy, n_words, nodes, source, recorded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (policy, n_words, nodes, source, now))

    def add_claim_telemetry(self, n_words: int, coordination_millis: int,
                            work_nodes: int, worker_count: int):
        """Append a claim coordination record to claim_telemetry for offline analysis."""
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO claim_telemetry
                (n_words, coordination_millis, work_nodes, worker_count, recorded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (n_words, coordination_millis, work_nodes, worker_count, now))

    def add_backstop_telemetry(self, n_words: int, depth, elapsed_millis: int,
                               nodes: int, predicted_nodes, remaining_candidates: int):
        """Append a wall-clock backstop firing to backstop_telemetry for offline
        tuning of COLD_BACKSTOP_SECONDS.  predicted_nodes is None when the cost
        model was cold for this size at the time the backstop fired."""
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO backstop_telemetry
                (n_words, depth, elapsed_millis, nodes, predicted_nodes,
                 remaining_candidates, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (n_words, depth, elapsed_millis, nodes, predicted_nodes,
              remaining_candidates, now))
