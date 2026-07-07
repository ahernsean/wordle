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
# Sentinel budget for the cost_model cell that aggregates samples across all
# budgets — the warm cross-budget fallback for a still-cold (size, budget) cell.
_AGGREGATE_BUDGET = -1

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

# Binary claim packing (adaptive_claim_packing.md §12 tuning dials).  All three
# are constructor/call parameters on the worker side (erd_swarm.py), not buried
# constants; these are only the defaults.
#
# small_count: how many not-yet-eliminated candidates a "small" bundle packs.
# Trades coordination amortization against republish waste — at most
# small_count - 1 candidates can be stranded behind a heavy head before an
# overrun republishes them.  8 is the spec's conservative anchor (the
# epoch-2 claim-count-reduction model already gives ~105x there; larger
# values buy diminishing returns per §1's table).
DEFAULT_SMALL_COUNT = 8
# count_cap: the max size of any bundle (bulk or small).  Bounds a bulk
# bundle's wall time and its crash-reclaim window (a dead worker's whole
# bundle is redone).  A provably-eliminated candidate completes in O(1)
# (§3 monotonicity) but is not exactly free, so 500 keeps a bulk claim's
# worst-case redo bounded to a few hundred near-instant re-evaluations —
# comfortably inside HB_TIMEOUT_SECONDS — while still collapsing a
# multi-thousand-candidate eliminated tail into a handful of claims.
DEFAULT_COUNT_CAP = 500
# republish_limit: how many times the same candidate may be republished
# (adaptive_claim_packing.md §7's bounded-republish-depth guardrail) before
# claim_next_bundle marks it `forced` so the caller stops applying the
# bundle overrun cap to it and lets within-candidate sub-branch promotion
# absorb its cost instead.  3 rounds is enough for a transient pile-up
# (other bundle members happened to be heavy too) to resolve itself without
# letting a genuinely pathological candidate thrash the pool indefinitely.
DEFAULT_REPUBLISH_LIMIT = 3
# Per-attempt busy_timeout (ms) while probing for claim_next_bundle's write
# lock.  Short enough that each failed attempt is a real, countable retry
# (claim_telemetry.claim_retries) rather than one long internal SQLite wait
# indistinguishable from a single slow claim.
_BUNDLE_CLAIM_RETRY_MILLIS = 100

# Re-export so callers don't need to import cache_sqlite directly.
encode_subset = ScoreCache.encode_subset


def decode_subset(blob: bytes) -> list[str]:
    """Reverse ScoreCache.encode_subset: split fixed-5-byte words."""
    return [blob[i:i + 5].decode() for i in range(0, len(blob), 5)]


def guess_depth_from_spine(spine) -> int:
    """Guesses played to reach a branch, from its `active_branches.spine` value.

    A spine is space-joined "GUESS pattern" tokens (two tokens per guess), so the
    guess count is the token count halved.  An empty/NULL spine is the root: 0.
    """
    return len(spine.split()) // 2 if spine else 0


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
    nodes_spent    INTEGER NOT NULL DEFAULT 0,
    -- the guesses played from the root to this branch, space-joined as
    -- "GUESS pattern" (e.g. "SALET -g-g- CRANE bb-y-").  guess_depth is the
    -- count of these guesses.  NULL = no spine recorded (a row predating spine
    -- capture).
    spine          TEXT,
    -- claim_next_bundle's forward cursor: count of best-first positions that
    -- have been packed into a bundle at least once.  Positions
    -- [0, pack_cursor) may hold holes (reclaimed/republished, no current
    -- candidate_claims row) picked up by holes_pass once the cursor reaches
    -- n_candidates; positions [pack_cursor, n_candidates) are virgin and
    -- packed by the O(1)-amortized forward path.  Lives here (not
    -- worker-local memory) so concurrent claim_next_bundle calls never pack
    -- overlapping bundles.
    pack_cursor    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
    ON active_branches(status, priority DESC, n_words DESC);

-- One row per claimed candidate of a branch.  A row's existence is an
-- ADVISORY claim ("a worker is probably evaluating this candidate"); only
-- done=1 is AUTHORITATIVE ("fully evaluated and folded into best_erd").
-- Coverage = all n_candidates slots with done=1.  A crashed worker leaves a
-- done=0 row that stale-reclaim deletes, turning it back into an unclaimed
-- gap to be redone — never skipped.  Republish-on-overrun deletes the
-- done=0 rows of a bundle's unfinished remainder the same way, so a
-- reclaimed hole and a republished hole are indistinguishable to the packer.
-- idx = index into the policy-canonical candidate list (all_words for ERD_ALL).
-- bundle_id groups the rows one claim_next_bundle call hands out together
-- (a worker evaluates its bundle in one best-first sweep); NULL for a row
-- inserted outside the packer (e.g. mark_claims_done's within-candidate
-- overrun promotion).
CREATE TABLE IF NOT EXISTS candidate_claims (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    bundle_id  TEXT,
    PRIMARY KEY (branch_key, idx)
);

CREATE INDEX IF NOT EXISTS idx_candidate_claims_bundle
    ON candidate_claims(branch_key, bundle_id);

-- One row per bundle a worker has reported on: the nodes actually spent and
-- the coordination overhead of handing it out, aggregated into
-- branch_finalize_log at finalize and dropped with the rest of the branch's
-- transient state by delete_branch.  censored=1 marks a bundle that hit its
-- node/wall cap and republished an unfinished remainder (see
-- ERDQueue.republish_remainder) — nodes is then a lower bound on what the
-- bundle's original member set would have cost.
CREATE TABLE IF NOT EXISTS bundle_stats (
    branch_key   BLOB    NOT NULL,
    bundle_id    TEXT    NOT NULL,
    nodes        INTEGER NOT NULL,
    coord_millis INTEGER NOT NULL,
    censored     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_key, bundle_id)
);

-- Per-candidate republish count: how many times a candidate's cross-candidate
-- bundle has overrun before this candidate finished (see
-- adaptive_claim_packing.md §7's bounded-republish-depth guardrail).  Survives
-- the delete/re-insert cycle a republished candidate's candidate_claims row
-- goes through, so the packer can tell a chronically-stranded candidate apart
-- from one republished for the first time.  Dropped with the rest of the
-- branch's transient state by delete_branch.
CREATE TABLE IF NOT EXISTS candidate_republish (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_key, idx)
);

-- Per-size-bucket online cost model (time-weighted geometric mean of
-- recursion-node cost).  Keyed by (policy, size_bucket, budget): the same word
-- count costs very different node totals at different remaining-guess budgets,
-- so budget is part of the key.  budget = -1 is the budget-aggregate accumulator
-- (every sample also folds into it), read as a warm fallback when a specific
-- (size_bucket, budget) bucket is still cold.  policy keeps ERD_ALL and
-- ERD_ANSWERS models from cross-contaminating.
CREATE TABLE IF NOT EXISTS cost_model (
    policy           TEXT    NOT NULL,
    size_bucket      INTEGER NOT NULL,
    budget           INTEGER NOT NULL DEFAULT -1,
    weighted_log_sum REAL    NOT NULL DEFAULT 0,
    weight_sum       REAL    NOT NULL DEFAULT 0,
    weighted_log_sq  REAL    NOT NULL DEFAULT 0,
    last_updated     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (policy, size_bucket, budget)
);

-- Raw per-solve samples for offline distribution analysis and threshold tuning.
-- censored = 1 marks a sample whose unit was handed off at the node/wall cap
-- before it finished: `nodes` is then a LOWER BOUND on the true cost, and an
-- offline survival fit must treat it as such rather than as an exact point.
CREATE TABLE IF NOT EXISTS cost_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    policy      TEXT    NOT NULL,
    n_words     INTEGER NOT NULL,
    nodes       INTEGER NOT NULL,
    wall_millis INTEGER,
    budget      INTEGER,
    censored    INTEGER NOT NULL DEFAULT 0,
    source      TEXT,        -- 'finalize' or 'censored'
    epoch       INTEGER NOT NULL DEFAULT 0,
    recorded_at INTEGER NOT NULL
);

-- Outbound-only coordination telemetry for offline clustering-decision analysis.
-- Never read by any runtime control path; freely droppable.  busy_wait_millis is
-- the wall time spent acquiring the claim's write lock (the direct contention
-- signal); claim_retries counts application-level BEGIN IMMEDIATE retries.
CREATE TABLE IF NOT EXISTS claim_telemetry (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words             INTEGER NOT NULL,
    coordination_millis INTEGER NOT NULL,
    work_nodes          INTEGER NOT NULL,
    claim_retries      INTEGER,
    busy_wait_millis   INTEGER,
    worker_count       INTEGER,
    epoch              INTEGER NOT NULL DEFAULT 0,
    recorded_at        INTEGER NOT NULL
);

-- One row per telemetry epoch: a contiguous run under one claiming regime.
-- epoch 0 is the single-candidate-atom baseline; the packer deploy inserts a new
-- row and bumps run_meta.epoch.  All offline comparisons filter on epoch.
CREATE TABLE IF NOT EXISTS telemetry_epoch (
    epoch      INTEGER PRIMARY KEY,
    label      TEXT,
    git_sha    TEXT,
    started_at INTEGER,
    notes      TEXT
);

-- Durable per-branch timing/cost record, written at finalize BEFORE delete_branch
-- copies the otherwise-destroyed active_branches.created_at/finalized_at/
-- nodes_spent out of the transient row.  Subbranches finalize as first-class
-- branches under recursive promotion, so each gets one row here.  The packer-era
-- columns (n_bundles, max_bundle_nodes, total_coord_millis, censored_units) are
-- NULL under single-candidate claiming and populated once bundling lands.
CREATE TABLE IF NOT EXISTS branch_finalize_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_key         BLOB,
    spine              TEXT,
    n_words            INTEGER,
    budget             INTEGER,
    epoch              INTEGER NOT NULL DEFAULT 0,
    created_at         INTEGER,
    finalized_at       INTEGER,
    nodes_spent        INTEGER,
    n_claims           INTEGER,
    n_bundles          INTEGER,
    max_bundle_nodes   INTEGER,
    total_coord_millis INTEGER,
    censored_units     INTEGER,
    recorded_at        INTEGER NOT NULL
);

-- Per-candidate predicted-vs-actual work, collected under single-candidate
-- claiming (epoch 0) where one claim == one candidate so actual_nodes IS that
-- candidate's true cost.  Validates the packer's work metric: predicted_work is
-- estimate_candidate_work(c | bound_erd); erd_lower_bound_pruned =
-- (candidate_cost_lower_bound >= bound_erd) means the candidate was provably
-- cut for free (predicted_work 0).  bound_erd and candidate_cost_lower_bound
-- are logged so an ERD-pruned candidate's near-zero cost reads as a correct
-- prediction, not a wild miss.  The §10 go/no-go gate is computed from this.
-- group_sizes is the candidate's response-group sizes ('-'-joined), the
-- sufficient statistic to recompute ANY work metric offline (uncut, cutoff-aware,
-- ...) against the logged bound_erd without re-running the swarm.  Written only
-- for non-ERD-pruned rows (an ERD-pruned row's predicted work is exactly 0).
CREATE TABLE IF NOT EXISTS candidate_accuracy (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_key                 BLOB,
    n_words                    INTEGER NOT NULL,
    budget                     INTEGER,
    predicted_work             REAL,
    bound_erd                  REAL,
    candidate_cost_lower_bound REAL,
    erd_lower_bound_pruned     INTEGER NOT NULL,
    actual_nodes               INTEGER NOT NULL,
    group_sizes                TEXT,
    source_word                TEXT,
    epoch                      INTEGER NOT NULL DEFAULT 0,
    recorded_at                INTEGER NOT NULL
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
    budget               INTEGER,            -- frame's remaining-guess budget at fire
    elapsed_millis       INTEGER NOT NULL,   -- wall time in frame at fire
    nodes                INTEGER NOT NULL,   -- nodes spent in frame at fire
    predicted_nodes      REAL,               -- typical(n) at entry; NULL = cold model
    remaining_candidates INTEGER NOT NULL,   -- candidates handed off
    recorded_at          INTEGER NOT NULL
);
"""


def _pack_bundle(order, start, cost_lower_bound, bound, small_count, count_cap):
    """The exact-elimination packer (adaptive_claim_packing.md §5).

    Walks `order` — a sequence of candidate idx values in best-first (Σk²
    ascending) order — from position `start`, classifying each idx by the
    exact test `cost_lower_bound[idx] >= bound`.  If the first candidate is
    eliminated, absorbs consecutive eliminated candidates up to count_cap (a
    bulk bundle).  Otherwise absorbs candidates — including any eliminated
    ones interleaved among them — until small_count not-yet-eliminated
    ("survivor") candidates have been taken or count_cap is reached (a small
    bundle).

    Returns (bundle, next_position): bundle is the list of idx taken, in the
    order they appear in `order`; next_position is the position in `order`
    just past the last one taken (== len(order) if `order` was exhausted).
    start >= len(order) returns ([], start).
    """
    n = len(order)
    if start >= n:
        return [], start
    i = start
    bundle = []
    if cost_lower_bound[order[i]] >= bound:
        while i < n and len(bundle) < count_cap and cost_lower_bound[order[i]] >= bound:
            bundle.append(order[i])
            i += 1
    else:
        survivors = 0
        while i < n and len(bundle) < count_cap and survivors < small_count:
            idx = order[i]
            bundle.append(idx)
            if cost_lower_bound[idx] < bound:
                survivors += 1
            i += 1
    return bundle, i


class ERDQueue:
    """SQLite-backed work queue for the parallel ERD_ALL precache job."""

    def __init__(self, db_path: str, timeout: float = 30.0):
        self._timeout = timeout
        self._conn = sqlite3.connect(db_path, timeout=timeout,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate()
        # Active epoch, read once: workers open the queue after the supervisor has
        # stamped run_meta.epoch, and are restarted across a cutover, so caching
        # here is safe and keeps every telemetry insert off a per-row SELECT.
        self.epoch = int(self.get_meta('epoch') or 0)
        # Per-claim contention metrics, refreshed by the claim path and read by
        # the next add_claim_telemetry without changing claim_* return shapes.
        self._last_claim_busy_millis = 0
        self._last_claim_retries = 0
        # Monotonic per-connection counter for bundle_id generation: paired
        # with worker_id it is unique without a timestamp-collision risk
        # (two bundles claimed by the same worker within the same second).
        self._bundle_seq = 0

    def _migrate(self):
        # active_branches.spine: additive, nullable, no backfill.  Existing rows
        # keep NULL (display falls back to the source word) until re-promoted.
        cols = {r["name"] for r in
                self._conn.execute("PRAGMA table_info(active_branches)")}
        if "spine" not in cols:
            self._conn.execute("ALTER TABLE active_branches ADD COLUMN spine TEXT")
        # active_branches.depth held a cooperative-nesting count.  It is retired:
        # the only thing it distinguished — user-queued vs cooperatively-promoted
        # — is read from pending_branches membership instead.  SQLite 3.34
        # predates ALTER TABLE DROP COLUMN, so rebuild the table without it.
        # active_branches is transient worker state and migrations run with
        # workers stopped, so the rebuild discards nothing recoverable.
        if "depth" in cols:
            self._conn.executescript("""
                CREATE TABLE active_branches_new (
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
                    nodes_spent    INTEGER NOT NULL DEFAULT 0,
                    spine          TEXT
                );
                INSERT INTO active_branches_new
                    SELECT branch_key, n_words, n_candidates, priority,
                           source_word, source_pattern, best_erd, best_guess,
                           status, created_at, finalized_at, budget,
                           best_max_depth, tainted, nodes_spent, spine
                    FROM active_branches;
                DROP TABLE active_branches;
                ALTER TABLE active_branches_new RENAME TO active_branches;
                CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
                    ON active_branches(status, priority DESC, n_words DESC);
            """)
        # backstop_telemetry.depth recorded the engine recursion level; it now
        # records the frame's remaining-guess budget.  RENAME COLUMN (3.25+) is
        # available on this box.
        bt_cols = {r["name"] for r in
                   self._conn.execute("PRAGMA table_info(backstop_telemetry)")}
        if "depth" in bt_cols and "budget" not in bt_cols:
            self._conn.execute(
                "ALTER TABLE backstop_telemetry RENAME COLUMN depth TO budget")

        # cost_model re-key: (policy, size_bucket) -> (policy, size_bucket, budget).
        # The same word count costs very different node totals at different
        # budgets, so the old single-budget bucket conflated them.  Rebuild with
        # budget in the key; existing accumulators carry across as the budget = -1
        # aggregate (every new sample still folds into -1, so the prior model
        # stays warm as a fallback and packing is not cold at cutover).
        cm_cols = {r["name"] for r in
                   self._conn.execute("PRAGMA table_info(cost_model)")}
        if cm_cols and "budget" not in cm_cols:
            self._conn.executescript("""
                CREATE TABLE cost_model_new (
                    policy           TEXT    NOT NULL,
                    size_bucket      INTEGER NOT NULL,
                    budget           INTEGER NOT NULL DEFAULT -1,
                    weighted_log_sum REAL    NOT NULL DEFAULT 0,
                    weight_sum       REAL    NOT NULL DEFAULT 0,
                    weighted_log_sq  REAL    NOT NULL DEFAULT 0,
                    last_updated     INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (policy, size_bucket, budget)
                );
                INSERT INTO cost_model_new
                    (policy, size_bucket, budget, weighted_log_sum, weight_sum,
                     weighted_log_sq, last_updated)
                    SELECT policy, size_bucket, -1, weighted_log_sum, weight_sum,
                           weighted_log_sq, last_updated
                    FROM cost_model;
                DROP TABLE cost_model;
                ALTER TABLE cost_model_new RENAME TO cost_model;
            """)

        # Epoch tagging: stamp every append-only telemetry table with the active
        # epoch.  Existing rows default to epoch 0 (the single-candidate baseline).
        # Additive columns the CREATE shape declares for fresh DBs but that an
        # existing DB needs via ALTER.
        self._add_columns("claim_telemetry", {
            "busy_wait_millis": "INTEGER",
            "epoch": "INTEGER NOT NULL DEFAULT 0",
        })
        self._add_columns("cost_samples", {
            "budget": "INTEGER",
            "censored": "INTEGER NOT NULL DEFAULT 0",
            "epoch": "INTEGER NOT NULL DEFAULT 0",
        })
        self._add_columns("backstop_telemetry", {
            "epoch": "INTEGER NOT NULL DEFAULT 0",
        })
        self._add_columns("candidate_accuracy", {
            "group_sizes": "TEXT",
            "source_word": "TEXT",
        })
        # Binary claim packing (issue #67): the packer's cursor and bundle
        # grouping.  Additive/nullable-default, so an existing (transient,
        # normally-empty-between-deploys) active_branches/candidate_claims row
        # just starts participating in bundled claiming going forward.
        self._add_columns("active_branches", {
            "pack_cursor": "INTEGER NOT NULL DEFAULT 0",
        })
        self._add_columns("candidate_claims", {
            "bundle_id": "TEXT",
        })

        # candidate_accuracy.cost_lb / .gated are legacy column names from
        # before the ERD-lower-bound-pruning rename (§9b): cost_lb was the
        # candidate cost lower bound now called candidate_cost_lower_bound,
        # and gated was the erd_lower_bound_pruned flag.  RENAME COLUMN
        # (3.25+) is available on this box (see the backstop_telemetry
        # depth->budget migration above).
        ca_cols = {r["name"] for r in
                   self._conn.execute("PRAGMA table_info(candidate_accuracy)")}
        if "cost_lb" in ca_cols and "candidate_cost_lower_bound" not in ca_cols:
            self._conn.execute(
                "ALTER TABLE candidate_accuracy "
                "RENAME COLUMN cost_lb TO candidate_cost_lower_bound")
        if "gated" in ca_cols and "erd_lower_bound_pruned" not in ca_cols:
            self._conn.execute(
                "ALTER TABLE candidate_accuracy "
                "RENAME COLUMN gated TO erd_lower_bound_pruned")

        # Baseline epoch 0 and the run_meta pointer, both idempotent.  git_sha is
        # stamped later (set_epoch) when a deploy knows it.
        now = int(time.time())
        self._conn.execute(
            "INSERT OR IGNORE INTO telemetry_epoch (epoch, label, started_at) "
            "VALUES (0, 'single-candidate atom baseline', ?)", (now,))
        self._conn.execute(
            "INSERT OR IGNORE INTO run_meta (key, value) VALUES ('epoch', '0')")

    def _add_columns(self, table: str, columns: dict):
        """Idempotently ALTER TABLE ADD COLUMN for any of `columns` not present.

        columns maps name -> SQL type/constraint.  A NOT NULL column must carry a
        DEFAULT so the add is safe on a populated table.
        """
        have = {r["name"] for r in
                self._conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

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
                      budget=None, spine=None, root_budget=None) -> bool:
        """Register a branch as in-progress (status 'open'), if not present.

        Idempotent via INSERT OR IGNORE: the worker that promoted the branch
        from the queue creates it; others that race simply see it exists and
        join.  Returns True if this call created the row.  n_candidates is the
        total claim slot count (one slot per candidate in the policy-canonical
        list).  budget is the guess budget for depth-limited ERD.  spine is the
        guesses played from the root to this branch (see the active_branches.spine
        column); None leaves the display to fall back to the source word.

        When root_budget is supplied alongside both budget and spine, the
        invariant budget + guess_depth = root_budget is enforced: a spine whose
        guess count contradicts the budget is a composition bug and is rejected
        rather than persisted.
        """
        if root_budget is not None and budget is not None and spine is not None:
            guess_depth = guess_depth_from_spine(spine)
            if budget + guess_depth != root_budget:
                raise ValueError(
                    f"spine guess_depth {guess_depth} + budget {budget} "
                    f"!= root_budget {root_budget} for spine {spine!r}")
        now = int(time.time())
        cur = self._conn.execute("""
            INSERT OR IGNORE INTO active_branches
                (branch_key, n_words, n_candidates,
                 priority, source_word, source_pattern, status, created_at,
                 budget, spine)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """, (branch_key, n_words, n_candidates,
              priority, source_word, source_pattern, now, budget, spine))
        return cur.rowcount == 1

    def get_branch(self, branch_key):
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_key = ?",
            (branch_key,)).fetchone()

    def claim_next_bundle(self, branch_key, worker_id, n_candidates,
                          candidate_order, cost_lower_bound,
                          small_count=DEFAULT_SMALL_COUNT,
                          count_cap=DEFAULT_COUNT_CAP,
                          republish_limit=DEFAULT_REPUBLISH_LIMIT):
        """Atomically pack and claim the next bundle of unclaimed candidates.

        Runs the exact-elimination packer (_pack_bundle,
        adaptive_claim_packing.md §5) inside one BEGIN IMMEDIATE transaction:
        every unclaimed candidate is classified by
        `candidate_cost_lower_bound(c) >= B` against the branch's current real
        best_erd (B), read from active_branches in this same transaction so
        two concurrent callers never pack overlapping bundles (adaptive_claim_
        packing.md §12).  candidate_order is the branch's best-first order (a
        permutation of range(n_candidates), Σk² ascending) and
        cost_lower_bound is indexed by idx (not by position in
        candidate_order); both are supplied by the caller — see
        _BranchWorker._packing_stats — so this module never imports numpy or
        the engine directly.

        The forward cursor (active_branches.pack_cursor) tracks how much of
        candidate_order has been packed at least once; once it reaches
        n_candidates every further call falls back to a holes pass (idx with
        no current candidate_claims row — reclaimed or republished slots),
        packed the same way over the filtered best-first order.

        Returns (bundle_id, indices, forced), or None if nothing is claimable
        (branch not open or missing, or every slot already has a row).
        indices is the claimed idx list in best-first (evaluation) order.
        forced is the frozenset of indices whose candidate_republish count has
        already reached republish_limit: the caller must evaluate those
        without applying the bundle's own overrun cap (adaptive_claim_
        packing.md §7's bounded-republish-depth guardrail), letting
        within-candidate sub-branch promotion absorb the cost instead of
        bouncing the candidate through another republish cycle.

        Every source of work — fresh handout, a dead worker's reclaimed
        slots, and a republished overrun remainder — flows through this one
        method, so no path ever hands out a single candidate alone (a bundle
        of size 1 can still occur, e.g. the last candidate in a branch, but
        only as an emergent packer outcome).
        """
        if len(candidate_order) != n_candidates:
            raise ValueError(
                f"candidate_order length {len(candidate_order)} != "
                f"n_candidates {n_candidates}")
        # Probe for the write lock with a short per-attempt busy_timeout so a
        # failed attempt is a real, countable retry rather than one long
        # internal SQLite wait indistinguishable from a single slow claim;
        # restore the connection's normal timeout once the lock is held (or
        # the overall budget is exhausted).
        _acquire_t0 = time.perf_counter()
        retries = 0
        self._conn.execute(f"PRAGMA busy_timeout = {_BUNDLE_CLAIM_RETRY_MILLIS}")
        try:
            while True:
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError:
                    if time.perf_counter() - _acquire_t0 > self._timeout:
                        raise
                    retries += 1
        finally:
            self._conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        self._last_claim_busy_millis = int((time.perf_counter() - _acquire_t0) * 1e3)
        self._last_claim_retries = retries
        try:
            # Never hand out a claim for a branch that has been finalized and
            # deleted: a worker still looping would otherwise redo it from
            # scratch.  Checked inside the write transaction so it can't race
            # finalize+delete.
            br = self._conn.execute(
                "SELECT status, best_erd, pack_cursor FROM active_branches "
                "WHERE branch_key = ?", (branch_key,)).fetchone()
            if br is None or br["status"] != "open":
                self._conn.execute("COMMIT")
                return None
            bound = br["best_erd"] if br["best_erd"] is not None else float("inf")
            cursor = br["pack_cursor"]
            if cursor < n_candidates:
                bundle, new_cursor = _pack_bundle(
                    candidate_order, cursor, cost_lower_bound, bound,
                    small_count, count_cap)
                if bundle:
                    self._conn.execute(
                        "UPDATE active_branches SET pack_cursor = ? "
                        "WHERE branch_key = ?", (new_cursor, branch_key))
            else:
                claimed = {r["idx"] for r in self._conn.execute(
                    "SELECT idx FROM candidate_claims WHERE branch_key = ?",
                    (branch_key,))}
                holes = [idx for idx in candidate_order if idx not in claimed]
                bundle, _ = _pack_bundle(holes, 0, cost_lower_bound, bound,
                                         small_count, count_cap)
            if not bundle:
                self._conn.execute("COMMIT")
                return None
            bundle_id = f"{worker_id}:{self._bundle_seq}"
            self._bundle_seq += 1
            now = int(time.time())
            self._conn.executemany("""
                INSERT INTO candidate_claims
                    (branch_key, idx, claimed_by, claimed_at, done, bundle_id)
                VALUES (?, ?, ?, ?, 0, ?)
            """, [(branch_key, idx, worker_id, now, bundle_id) for idx in bundle])
            forced = frozenset()
            if republish_limit is not None:
                placeholders = ",".join("?" * len(bundle))
                rows = self._conn.execute(
                    f"SELECT idx FROM candidate_republish "
                    f"WHERE branch_key = ? AND count >= ? "
                    f"AND idx IN ({placeholders})",
                    (branch_key, republish_limit, *bundle)).fetchall()
                forced = frozenset(r["idx"] for r in rows)
            self._conn.execute("COMMIT")
            return (bundle_id, bundle, forced)
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def republish_remainder(self, branch_key, indices):
        """Return an overrun bundle's unfinished remainder to the unclaimed pool.

        adaptive_claim_packing.md §7a: deletes the done=0 candidate_claims
        rows for `indices` so they re-enter as holes, re-packed by the next
        claim_next_bundle against a B that is at least as tight as when they
        were first packed — never re-inserted as `len(indices)` individual
        claims.  Also bumps each candidate's persistent republish count
        (candidate_republish), which survives this delete/re-insert cycle so
        a chronically-stranded candidate is distinguishable from one
        republished for the first time.  Returns {idx: new_count}.
        """
        if not indices:
            return {}
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            placeholders = ",".join("?" * len(indices))
            self._conn.execute(
                f"DELETE FROM candidate_claims WHERE branch_key = ? "
                f"AND done = 0 AND idx IN ({placeholders})",
                (branch_key, *indices))
            self._conn.executemany("""
                INSERT INTO candidate_republish (branch_key, idx, count)
                VALUES (?, ?, 1)
                ON CONFLICT(branch_key, idx) DO UPDATE SET count = count + 1
            """, [(branch_key, idx) for idx in indices])
            rows = self._conn.execute(
                f"SELECT idx, count FROM candidate_republish "
                f"WHERE branch_key = ? AND idx IN ({placeholders})",
                (branch_key, *indices)).fetchall()
            self._conn.execute("COMMIT")
            return {r["idx"]: r["count"] for r in rows}
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def record_bundle_stats(self, branch_key, bundle_id, nodes, coord_millis,
                            censored=False):
        """Record one bundle's actual node cost and coordination overhead.

        Aggregated into branch_finalize_log at finalize (see
        finalize_bundle_stats) and dropped along with the rest of the
        branch's transient state by delete_branch.  censored=1 marks a bundle
        that hit its node/wall cap and republished an unfinished remainder —
        nodes is then a lower bound on what the bundle's original member set
        would have cost.  A republished remainder re-packs under a new
        bundle_id, which gets its own independent row here.
        """
        self._conn.execute("""
            INSERT OR REPLACE INTO bundle_stats
                (branch_key, bundle_id, nodes, coord_millis, censored)
            VALUES (?, ?, ?, ?, ?)
        """, (branch_key, bundle_id, nodes, coord_millis, 1 if censored else 0))

    def finalize_bundle_stats(self, branch_key):
        """Aggregate and clear a branch's bundle_stats rows at finalize.

        Returns (n_bundles, max_bundle_nodes, total_coord_millis,
        censored_units) for branch_finalize_log — all None if no bundle
        recorded any stats (e.g. a branch solved entirely from reused cache
        entries).  Deletes the rows so bundle_stats stays bounded to
        currently-open branches, matching delete_branch's candidate_claims
        cleanup.
        """
        row = self._conn.execute("""
            SELECT COUNT(*) AS n_bundles, MAX(nodes) AS max_bundle_nodes,
                   SUM(coord_millis) AS total_coord_millis,
                   SUM(censored) AS censored_units
            FROM bundle_stats WHERE branch_key = ?
        """, (branch_key,)).fetchone()
        self._conn.execute(
            "DELETE FROM bundle_stats WHERE branch_key = ?", (branch_key,))
        if row["n_bundles"] == 0:
            return (None, None, None, None)
        return (row["n_bundles"], row["max_bundle_nodes"],
                row["total_coord_millis"], row["censored_units"])

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
        calls delete_branch.  Caller must have confirmed all candidates are done
        first; the WHERE status='open' guard makes the finalize idempotent.
        """
        now = int(time.time())
        cur = self._conn.execute("""
            UPDATE active_branches SET status = 'finalized', finalized_at = ?
            WHERE branch_key = ? AND status = 'open'
        """, (now, branch_key))
        return cur.rowcount == 1

    def delete_branch(self, branch_key):
        """Remove a finished branch and its claim/packer rows to bound the queue DB.

        finalize_bundle_stats already clears bundle_stats before this is
        called; candidate_republish is cleared here alongside candidate_claims
        since neither is meaningful once the branch is gone.
        """
        self._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_key = ?", (branch_key,))
        self._conn.execute(
            "DELETE FROM candidate_republish WHERE branch_key = ?", (branch_key,))
        self._conn.execute(
            "DELETE FROM bundle_stats WHERE branch_key = ?", (branch_key,))
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
        """Drop in-progress user-queued branch state; free stale cooperative claims.

        A branch is user-queued iff it has a pending_branches row.  Those rows
        reset_stale_in_progress() already flipped back to 'pending', so wiping
        their active_branches and claim rows just discards half-done coordination
        — they re-promote and redo cleanly.

        Cooperative branches have NO pending_branches row — they are inserted
        directly into active_branches by cooperative_solve.  Wiping them throws
        away all partial-claim progress with no recovery path.  Instead, only
        free their stale in-flight claims (done=0 rows); done=1 rows and the
        branch row itself survive, so the next worker to join picks up exactly
        where the killed worker left off.

        Returns (n_branches, n_claims) cleared for user-queued branches only.
        """
        member = "branch_key IN (SELECT branch_key FROM pending_branches)"
        nb = self._conn.execute(
            f"SELECT COUNT(*) FROM active_branches WHERE {member}").fetchone()[0]
        nc = self._conn.execute(
            f"SELECT COUNT(*) FROM candidate_claims WHERE {member}").fetchone()[0]
        self._conn.execute(f"DELETE FROM candidate_claims WHERE {member}")
        self._conn.execute(f"DELETE FROM candidate_republish WHERE {member}")
        self._conn.execute(f"DELETE FROM bundle_stats WHERE {member}")
        self._conn.execute(f"DELETE FROM active_branches WHERE {member}")
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
        candidate claims, heartbeats, and run_meta.

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
        """Atomically remove a branch's candidate claims and active_branches row.

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
                "DELETE FROM candidate_republish WHERE branch_key = ?", (branch_key,))
            self._conn.execute(
                "DELETE FROM bundle_stats WHERE branch_key = ?", (branch_key,))
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

    def _cost_bucket_row(self, policy: str, n_words: int, budget):
        """The cost_model accumulators for one (policy, size_bucket, budget) cell.

        budget = _AGGREGATE_BUDGET (-1) reads the budget-aggregate accumulator
        every sample also folds into.
        """
        return self._conn.execute(
            "SELECT weighted_log_sum, weight_sum, weighted_log_sq, last_updated "
            "FROM cost_model WHERE policy = ? AND size_bucket = ? AND budget = ?",
            (policy, cost_size_bucket(n_words), budget)).fetchone()

    def _warm_bucket_row(self, policy: str, n_words: int, budget):
        """Warm accumulators for n_words at `budget`, or None.

        Reads the specific (size_bucket, budget) cell when budget is given and
        falls back to the budget-aggregate (-1) cell when that cell is cold.
        budget=None reads the aggregate directly.
        """
        if budget is not None:
            row = self._cost_bucket_row(policy, n_words, budget)
            if row is not None and row['weight_sum'] >= _COST_MODEL_MIN_WEIGHT:
                return row
        row = self._cost_bucket_row(policy, n_words, _AGGREGATE_BUDGET)
        if row is None or row['weight_sum'] < _COST_MODEL_MIN_WEIGHT:
            return None
        return row

    def get_cost_typical(self, policy: str, n_words: int, budget=None):
        """Geometric-mean node count for n_words at `budget`, or None when cold.

        The estimate is exp(weighted_log_sum / weight_sum).  budget keys the model
        on remaining-guess budget; a cold (size_bucket, budget) cell falls back to
        the budget-aggregate, and a cold aggregate returns None so the caller can
        use a size-based heuristic.
        """
        row = self._warm_bucket_row(policy, n_words, budget)
        if row is None:
            return None
        return math.exp(row['weighted_log_sum'] / row['weight_sum'])

    def get_cost_spread(self, policy: str, n_words: int, budget=None):
        """Std-dev of ln(nodes) for n_words at `budget` (log-normal sigma), or None.

        Recovered from the stored second log-moment:
            sigma^2 = weighted_log_sq/weight_sum - mu^2
        Round-off can make this marginally negative when every sample is equal;
        clamp to 0.  Same (size_bucket, budget) -> aggregate fallback as
        get_cost_typical.  Used for the over-promotion shade exp(mu - Z*sigma) and
        for offline distribution analysis.
        """
        row = self._warm_bucket_row(policy, n_words, budget)
        if row is None:
            return None
        mu = row['weighted_log_sum'] / row['weight_sum']
        var = row['weighted_log_sq'] / row['weight_sum'] - mu * mu
        return math.sqrt(var) if var > 0 else 0.0

    def update_cost_model(self, policy: str, n_words: int, nodes: int,
                          weight: float = 1.0, now: int = None, budget=None):
        """Fold one node-cost sample (value `nodes`, multiplicity `weight`).

        weight > 1 records `weight` identical samples of `nodes` in one call.
        For a batch of *distinct* samples whose individual magnitudes matter to
        the spread, use update_cost_model_logsums so each sample reaches the
        second log-moment without a lossy pre-averaging collapse.  budget, when
        given, also keys a (size_bucket, budget) cell in addition to the aggregate.
        """
        if nodes <= 0 or weight <= 0:
            return
        log_n = math.log(nodes)
        self._fold_cost_sample(policy, n_words, log_n * weight,
                               log_n * log_n * weight, weight, now, budget)

    def update_cost_model_logsums(self, policy: str, n_words: int,
                                  log_sum: float, log_sq_sum: float,
                                  weight: float, now: int = None, budget=None):
        """Fold a pre-summed batch of log samples: (Σ ln x, Σ ln²x, count).

        The worker's inline-sample buffer accumulates these sums directly, so the
        batch contributes to weighted_log_sum and weighted_log_sq exactly as if
        each sample had been folded individually — no exp/int/log round-trip.
        """
        if weight <= 0:
            return
        self._fold_cost_sample(policy, n_words, log_sum, log_sq_sum, weight, now,
                               budget)

    def _fold_cost_sample(self, policy, n_words, d_log_sum, d_log_sq, d_weight,
                          now, budget):
        if now is None:
            now = int(time.time())
        bucket = cost_size_bucket(n_words)
        # Every sample feeds the budget-aggregate (-1) cell so it stays a warm
        # cross-budget fallback; a known budget additionally feeds its own cell.
        self._fold_one(policy, bucket, _AGGREGATE_BUDGET,
                       d_log_sum, d_log_sq, d_weight, now)
        if budget is not None and budget != _AGGREGATE_BUDGET:
            self._fold_one(policy, bucket, budget,
                           d_log_sum, d_log_sq, d_weight, now)

    def _fold_one(self, policy, bucket, budget, d_log_sum, d_log_sq, d_weight, now):
        row = self._conn.execute(
            "SELECT weighted_log_sum, weight_sum, weighted_log_sq, last_updated "
            "FROM cost_model WHERE policy = ? AND size_bucket = ? AND budget = ?",
            (policy, bucket, budget)).fetchone()
        if row is None:
            self._conn.execute("""
                INSERT INTO cost_model
                    (policy, size_bucket, budget, weighted_log_sum, weight_sum,
                     weighted_log_sq, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (policy, bucket, budget, d_log_sum, d_weight, d_log_sq, now))
        else:
            # Continuous-time EMA: decay every accumulator by the age of the
            # bucket before folding the new contribution.  Clamp elapsed at 0 so
            # an out-of-order timestamp can never amplify (decay > 1).
            decay = math.exp(-max(0, now - row['last_updated']) / _COST_MODEL_TAU)
            self._conn.execute("""
                UPDATE cost_model
                SET weighted_log_sum = ?, weight_sum = ?, weighted_log_sq = ?,
                    last_updated = ?
                WHERE policy = ? AND size_bucket = ? AND budget = ?
            """, (decay * row['weighted_log_sum'] + d_log_sum,
                  decay * row['weight_sum'] + d_weight,
                  decay * row['weighted_log_sq'] + d_log_sq,
                  now, policy, bucket, budget))

    def add_cost_sample(self, policy: str, n_words: int, nodes: int, source: str,
                        budget=None, wall_millis=None, censored: int = 0):
        """Append a raw sample to cost_samples for offline analysis.

        wall_millis is the per-solve wall span (the only such figure, populated at
        finalize).  censored=1 marks a handed-off unit whose `nodes` is a lower
        bound, so a survival fit must not treat it as exact.
        """
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO cost_samples
                (policy, n_words, nodes, wall_millis, budget, censored, source,
                 epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (policy, n_words, nodes, wall_millis, budget, censored, source,
              self.epoch, now))

    def add_claim_telemetry(self, n_words: int, coordination_millis: int,
                            work_nodes: int, worker_count: int):
        """Append a claim coordination record to claim_telemetry for offline analysis.

        claim_retries / busy_wait_millis come from the most recent claim path on
        this connection (set by claim_next_bundle), the direct lock-contention
        signal; the row is stamped with the active epoch.
        """
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO claim_telemetry
                (n_words, coordination_millis, work_nodes, claim_retries,
                 busy_wait_millis, worker_count, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (n_words, coordination_millis, work_nodes, self._last_claim_retries,
              self._last_claim_busy_millis, worker_count, self.epoch, now))

    def add_branch_finalize_log(self, branch_key, spine, n_words, budget,
                                created_at, finalized_at, nodes_spent, n_claims,
                                n_bundles=None, max_bundle_nodes=None,
                                total_coord_millis=None, censored_units=None):
        """Persist a branch's timing/cost the moment before delete_branch drops it.

        The bundle-diagnostic columns (n_bundles, max_bundle_nodes,
        total_coord_millis, censored_units) come from finalize_bundle_stats;
        they stay NULL when the caller doesn't pass them (a branch solved
        entirely from reused cache entries never claims a bundle).
        """
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO branch_finalize_log
                (branch_key, spine, n_words, budget, epoch, created_at,
                 finalized_at, nodes_spent, n_claims, n_bundles,
                 max_bundle_nodes, total_coord_millis, censored_units,
                 recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (branch_key, spine, n_words, budget, self.epoch, created_at,
              finalized_at, nodes_spent, n_claims, n_bundles, max_bundle_nodes,
              total_coord_millis, censored_units, now))

    def add_candidate_accuracy(self, branch_key, n_words, budget, predicted_work,
                               bound_erd, candidate_cost_lower_bound,
                               erd_lower_bound_pruned, actual_nodes,
                               group_sizes=None, source_word=None):
        """Log one predicted-vs-actual work point for the §10 metric-validation gate.

        Under single-candidate claiming a claim is exactly one candidate, so
        actual_nodes is that candidate's true cost.  group_sizes ('-'-joined
        response-group sizes) is the sufficient statistic for recomputing any work
        metric offline; logged only for non-ERD-pruned rows.  source_word is the
        root opener of the branch's spine, so a multi-day corpus can be
        segmented per opener (different openers reach differently-shaped
        answer sets).
        """
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO candidate_accuracy
                (branch_key, n_words, budget, predicted_work, bound_erd,
                 candidate_cost_lower_bound, erd_lower_bound_pruned,
                 actual_nodes, group_sizes, source_word, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (branch_key, n_words, budget, predicted_work, bound_erd,
              candidate_cost_lower_bound, 1 if erd_lower_bound_pruned else 0,
              actual_nodes, group_sizes, source_word, self.epoch, now))

    def set_epoch(self, epoch: int, label: str = None, git_sha: str = None,
                  notes: str = None):
        """Register a new telemetry epoch and make it active.

        Inserts the telemetry_epoch row (idempotent), points run_meta.epoch at it,
        and updates this connection's cached epoch so subsequent telemetry stamps
        with it.  Called by a deploy that changes the claiming regime.
        """
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO telemetry_epoch (epoch, label, git_sha, started_at, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(epoch) DO UPDATE SET
                label   = COALESCE(excluded.label, label),
                git_sha = COALESCE(excluded.git_sha, git_sha),
                notes   = COALESCE(excluded.notes, notes)
        """, (epoch, label, git_sha, now, notes))
        self.set_meta('epoch', str(epoch))
        self.epoch = epoch

    def add_backstop_telemetry(self, n_words: int, budget, elapsed_millis: int,
                               nodes: int, predicted_nodes, remaining_candidates: int):
        """Append a wall-clock backstop firing to backstop_telemetry for offline
        tuning of COLD_BACKSTOP_SECONDS.  budget is the frame's remaining-guess
        budget at fire; predicted_nodes is None when the cost model was cold for
        this size at the time the backstop fired."""
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO backstop_telemetry
                (n_words, budget, elapsed_millis, nodes, predicted_nodes,
                 remaining_candidates, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (n_words, budget, elapsed_millis, nodes, predicted_nodes,
              remaining_candidates, now))
