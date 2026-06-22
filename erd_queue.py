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
    chunk_size     INTEGER NOT NULL DEFAULT 1,  -- always 1; retained for migration compat
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    best_erd       REAL,
    best_guess     TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER,
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
    wall_nanos  INTEGER,
    source      TEXT,        -- 'inline', 'finalize', or 'probe'
    recorded_at INTEGER NOT NULL
);

-- Outbound-only coordination telemetry for offline clustering-decision analysis.
-- Never read by any runtime control path; freely droppable.
CREATE TABLE IF NOT EXISTS claim_telemetry (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words            INTEGER NOT NULL,
    coordination_nanos INTEGER NOT NULL,
    work_nodes         INTEGER NOT NULL,
    claim_retries      INTEGER,
    worker_count       INTEGER,
    recorded_at        INTEGER NOT NULL
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
        if 'pending_subgroups' in tables:  # pragma: migration
            self._conn.execute('DROP TABLE IF EXISTS pending_branches')
            self._conn.execute(
                'ALTER TABLE pending_subgroups RENAME TO pending_branches')

        existing = {r['name'] for r in
                    self._conn.execute('PRAGMA table_info(pending_branches)')}
        for col, defn in [('source_word', 'TEXT'),
                          ('source_pattern', 'INTEGER')]:
            if col not in existing:  # pragma: migration
                self._conn.execute(
                    f'ALTER TABLE pending_branches ADD COLUMN {col} {defn}')

        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        # For columns that will be renamed below, skip adding the old name when
        # the new name already exists (fresh schema already has the new name).
        _hb_renamed = {'chunks_done': 'claims_done', 'chunk_idx': 'claim_idx',
                       'chunk_started_at': 'claim_started_at',
                       'cand_chunk_size': 'claim_total'}
        for col, defn in [('chunks_done',      'INTEGER'),
                          ('chunk_idx',        'INTEGER'),
                          ('chunk_started_at', 'INTEGER'),
                          ('cand_rate',        'REAL'),
                          ('cache_hits',       'INTEGER'),
                          ('cache_misses',     'INTEGER'),
                          ('n_cutoff',         'INTEGER'),
                          ('n_pruned',         'INTEGER'),
                          ('n_ok',             'INTEGER'),
                          ('best_guess',        'TEXT'),
                          ('best_erd',         'REAL'),
                          ('bound_erd',        'REAL'),
                          ('cur_candidate',    'TEXT'),
                          ('cand_n_seen',      'INTEGER'),
                          ('cand_chunk_size',  'INTEGER'),
                          ('cur_max_depth',    'INTEGER'),
                          ('cur_nodes',        'INTEGER'),
                          ('node_rate',        'REAL'),
                          ('cur_path',         'TEXT')]:
            new_name = _hb_renamed.get(col)
            if col not in existing_hb and (new_name is None or new_name not in existing_hb):  # pragma: migration
                self._conn.execute(
                    f'ALTER TABLE worker_heartbeat ADD COLUMN {col} {defn}')

        # Depth-limited ERD: each branch is solved at a guess budget; track the
        # winner's worst-case line length (best_max_depth) and whether the cap
        # excluded any candidate (tainted), aggregated across all workers.
        existing_ab = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(active_branches)')}
        for col, defn in [('budget',         'INTEGER'),
                          ('best_max_depth', 'INTEGER'),
                          ('tainted',        'INTEGER NOT NULL DEFAULT 0'),
                          ('depth',          'INTEGER NOT NULL DEFAULT 0')]:
            if col not in existing_ab:  # pragma: migration
                self._conn.execute(
                    f'ALTER TABLE active_branches ADD COLUMN {col} {defn}')

        # Column renames from earlier terminology alignment: subset_key ->
        # branch_key throughout; best_word -> best_guess; current_subset_key ->
        # current_branch_key in worker_heartbeat.
        existing_ps = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(pending_branches)')}
        if 'subset_key' in existing_ps:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE pending_branches RENAME COLUMN subset_key TO branch_key')
        existing_ab = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(active_branches)')}
        if 'subset_key' in existing_ab:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE active_branches RENAME COLUMN subset_key TO branch_key')
        if 'best_word' in existing_ab:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE active_branches RENAME COLUMN best_word TO best_guess')
        existing_bc = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(branch_chunks)')}
        if 'subset_key' in existing_bc:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE branch_chunks RENAME COLUMN subset_key TO branch_key')
        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        if 'current_subset_key' in existing_hb:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE worker_heartbeat '
                'RENAME COLUMN current_subset_key TO current_branch_key')
        if 'best_word' in existing_hb and 'best_guess' not in existing_hb:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE worker_heartbeat RENAME COLUMN best_word TO best_guess')

        # Rename branch_chunks -> candidate_claims.  The schema creates
        # candidate_claims (empty) first; drop that shell before renaming so
        # existing claim rows survive.
        tables = {r['name'] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'branch_chunks' in tables:  # pragma: migration
            self._conn.execute('DROP TABLE IF EXISTS candidate_claims')
            self._conn.execute(
                'ALTER TABLE branch_chunks RENAME TO candidate_claims')

        # Rename worker_heartbeat coordination columns to claim vocabulary.
        existing_hb = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(worker_heartbeat)')}
        for old, new in [('chunks_done',      'claims_done'),
                         ('chunk_idx',        'claim_idx'),
                         ('chunk_started_at', 'claim_started_at'),
                         ('cand_chunk_size',  'claim_total')]:
            if old in existing_hb:  # pragma: migration
                self._conn.execute(
                    f'ALTER TABLE worker_heartbeat RENAME COLUMN {old} TO {new}')

        # Add nodes_spent to active_branches for cost-model sampling.
        existing_ab = {r['name'] for r in
                       self._conn.execute('PRAGMA table_info(active_branches)')}
        if 'nodes_spent' not in existing_ab:  # pragma: migration
            self._conn.execute(
                'ALTER TABLE active_branches ADD COLUMN nodes_spent INTEGER NOT NULL DEFAULT 0')

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
                (branch_key, n_words, n_candidates, chunk_size,
                 priority, source_word, source_pattern, status, created_at, budget, depth)
            VALUES (?, ?, ?, 1, ?, ?, ?, 'open', ?, ?, ?)
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
        """Drop in-progress D-0 branch state; reclaim stale cooperative chunks.

        D-0 branches have pending_branches rows that reset_stale_in_progress()
        already flipped back to 'pending', so wiping their active_branches and
        chunk rows just discards half-done coordination — they re-promote and
        redo cleanly.

        Cooperative branches (depth>0) have NO pending_branches row — they are
        inserted directly into active_branches by cooperative_solve.  Wiping
        them throws away all partial-chunk progress with no recovery path.
        Instead, only free their stale in-flight claims (done=0 chunk rows);
        done=1 rows (completed chunks) and the branch row itself survive, so
        the next worker to join picks up exactly where the killed worker left
        off.

        Returns (n_branches, n_chunks) cleared for D-0 branches only.
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
        leave orphaned branch_chunks rows or a dangling active_branches row.

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
        branch_chunks — call reset_active_branches() first if the branch is
        currently in progress.
        """
        self._conn.execute(
            "DELETE FROM pending_branches "
            "WHERE branch_key = ? AND status = 'pending'",
            (branch_key,))
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0
