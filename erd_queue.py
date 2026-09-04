"""erd_queue.py — Sidecar SQLite coordination queue for the parallel ERD_ALL precache job.

Lives in erd_queue.sqlite3, separate from wordle_cache.sqlite3, so the
high-frequency coordination writes (claim/heartbeat/done) don't contend with
the high-volume ERD result writes in the main cache.

SQLite version floor: this module must run on SQLite 3.34 (the production
box's bundled version) — do not use a feature newer than that (e.g.
RETURNING needs 3.35+; ALTER TABLE DROP COLUMN needs 3.35+, worked around
in _migrate() by rebuilding the table).  RENAME COLUMN (3.25+) and
INSERT ... ON CONFLICT (3.24+) are both safely below the floor and already
in use.  CI's bundled SQLite is newer than 3.34 and will not catch a
version violation — check the target version by hand, or on the box itself.
"""

from __future__ import annotations

import collections
import json
import logging
import hashlib
import math
import os
import random
import sqlite3
import time
from urllib.parse import quote

from cache_sqlite import ScoreCache
from erd_lattice import erd_ge
from wordle_ui import fmt_pattern

logger = logging.getLogger(__name__)

# Workers heartbeat about every two seconds. This threshold governs both claim
# reclamation and report liveness, so a worker inside it is never reclaimed or
# excluded from live-worker totals.
WORKER_LIVENESS_SECONDS = 30

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
# count_cap: a backward-compatible hard upper bound on the survivor count in
# a bundle.  One-level ERD-pruned candidates are completed directly and never
# enter a bundle, so small_count is the effective limit under the defaults.
DEFAULT_COUNT_CAP = 500
# republish_limit: how many times the same candidate may be republished
# (adaptive_claim_packing.md §7's bounded-republish-depth guardrail) before
# claim_next_bundle marks it `forced` so the caller stops applying the
# bundle overrun cap to it and lets within-candidate sub-branch promotion
# absorb its cost instead.  3 rounds is enough for a transient pile-up
# (other bundle members happened to be heavy too) to resolve itself without
# letting a genuinely pathological candidate thrash the pool indefinitely.
DEFAULT_REPUBLISH_LIMIT = 3
# The inclusive range a caller-supplied opener-work priority may occupy.  Every
# writer of opener_work.requested_priority enforces it and
# check_opener_work_invariants reports a row outside it as a violation, so a
# priority a queue can hold is exactly a priority a caller may ask for.
#
# The range seats one rung per opener word: every candidate word laddered at a
# distinct priority, with room left to append further batches beneath and to
# insert above.  The full candidate list is ~15,000 words, so it runs to just
# below LEGACY_PROMOTED_PRIORITY_MIN rather than stopping at a round hundred.
OPENER_PRIORITY_MIN = 0
OPENER_PRIORITY_MAX = 999_999

# Priorities at or above this bound are never allowed to preempt requested work.
LEGACY_PROMOTED_PRIORITY_MIN = 1_000_000

# A worker's scheduling role, recorded alongside its heartbeat so a report can
# explain why the worker is running the branch it is on: PREFERRED is a
# highest-requested-priority eligible opener at the claim boundary that
# selected it (ties at the top priority all count as preferred — none of them
# was skipped in favor of another), FALLBACK is a strictly-lower-priority
# eligible opener claimed because every higher-priority opener had no
# claimable bundle, and DIRECT is a branch with no live opener-work ownership
# (legacy provenance).  Role is a function of ownership and admission order
# alone, never of how a branch was reached — a branch with a live owner is
# always PREFERRED or FALLBACK, one without is always DIRECT.
SCHEDULING_ROLE_PREFERRED = "preferred"
SCHEDULING_ROLE_FALLBACK = "fallback"
SCHEDULING_ROLE_DIRECT = "direct"
SCHEDULING_ROLES = (SCHEDULING_ROLE_PREFERRED, SCHEDULING_ROLE_FALLBACK,
                    SCHEDULING_ROLE_DIRECT)
# Per-attempt busy_timeout (ms) while probing for claim_next_bundle's write
# lock.  Short enough that each failed attempt is a real, countable retry
# (claim_telemetry.claim_retries) rather than one long internal SQLite wait
# indistinguishable from a single slow claim.
_BUNDLE_CLAIM_RETRY_MILLIS = 100

# A checkpoint_pause flag older than this is ignored by workers: it means the
# supervisor that set it died before clearing it, and honouring it forever
# would wedge the swarm.
CHECKPOINT_PAUSE_STALE_SECONDS = 60

# Approximate WAL cost of one candidate_claims row (table row plus the primary
# key and bundle indexes), for the per-table traffic estimate.  Since branch-id
# normalization each row references a small integer branch_id, not the fat
# branch_key blob, so this is a small constant rather than a multiple of the
# key width.  The row COUNT tallied alongside it is exact (from SQLite
# changes() on the delete/update paths); only this width is an estimate.
_CLAIM_ROW_WAL_BYTES = 96

# WAL frames are whole database pages; the traffic estimate's per-commit floor
# is expressed in these.
_WAL_PAGE_BYTES = 4096

# Queue WAL size at which the swarm must stop rather than keep writing: the
# quiesce/TRUNCATE machinery has failed to reclaim the WAL and the disk is at
# risk.  Enforced twice — by the supervisor (which latches the swarm down and
# collects diagnostics) and by each worker as a backstop (a worker that
# outlives the supervisor, or spins without reaching a bundle boundary, must
# still stop writing on its own).  Overridable via QUEUE_WAL_HARD_CEILING_GIB.
QUEUE_WAL_HARD_CEILING_BYTES = int(
    float(os.environ.get('QUEUE_WAL_HARD_CEILING_GIB', '32')) * 1024 ** 3)

# Disk sample ring length for the status display's growth rate (see
# record_disk_sample); at the supervisor's 30s cadence this covers ~10 min.
DISK_SAMPLE_KEEP = 20

# Disk fullness thresholds, as the used fraction df reports.  Above WARN the
# status display draws the disk figure in red; at or above STOP the swarm
# stops and latches down (see set_disk_stop), reserving the remaining space
# for the rest of the OS and for diagnosis.
DISK_WARN_FRACTION = 0.80
DISK_STOP_FRACTION = 0.90

# Re-export so callers don't need to import cache_sqlite directly.
encode_subset = ScoreCache.encode_subset


class _ClaimRetry:
    """claim_next_bundle sentinel: this call's transaction advanced the
    branch's claim state (the pack cursor moved past a prefix another
    caller's mark_claims_done had already claimed) but yielded no bundle for
    THIS caller.  A fresh call against the same branch_key may succeed
    immediately, unlike a plain None — which means the branch is absent,
    not open, foreign to the caller's expected owner, or (rarest) genuinely
    has no claimable candidate left at all.  Callers must not treat this the
    same as None: retrying the same branch is the correct response, not
    falling through to a different one."""
    __slots__ = ()

    def __repr__(self):
        return 'CLAIM_RETRY'


CLAIM_RETRY = _ClaimRetry()


def disk_stats(path: str) -> dict:
    """Fullness of the filesystem holding `path`, in df's terms.

    Returns {'total_bytes', 'used_bytes', 'avail_bytes', 'used_fraction'}.
    used_fraction = used / (used + available-to-unprivileged), matching df's
    Use% — the root-reserved blocks are excluded from capacity, so 100% here
    is where unprivileged writes start failing, not where the platters end.
    """
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    capacity = used + avail
    return {
        "total_bytes": st.f_blocks * st.f_frsize,
        "used_bytes": used,
        "avail_bytes": avail,
        "used_fraction": (used / capacity) if capacity else 1.0,
    }


def decode_subset(blob: bytes) -> list[str]:
    """Reverse ScoreCache.encode_subset: split fixed-5-byte words."""
    return [blob[i:i + 5].decode() for i in range(0, len(blob), 5)]


def guess_depth_from_spine(spine) -> int:
    """Guesses played to reach a branch, from its `active_branches.spine` value.

    A spine is space-joined "GUESS pattern" tokens (two tokens per guess), so the
    guess count is the token count halved.  An empty/NULL spine is the root: 0.
    """
    return len(spine.split()) // 2 if spine else 0


# The queue spans two database files.  The main file (this schema) holds
# operational state: what to work on, who is working on it, and the cost
# model read on the promotion hot path.  Telemetry tables live in a second
# file attached as `telemetry` (_TELEMETRY_SCHEMA_SQL): they outweigh queue
# state by orders of magnitude, are written best-effort (a telemetry failure
# must never affect queue operations), and are archived or pruned by moving
# the file — never by deleting from a live queue.  Separate files also give
# telemetry inserts their own write lock, so they never serialize against
# claim transactions.
#
# journal_size_limit caps how large the -wal file is left after a checkpoint
# completes (500MB); it bounds runaway growth on a run where checkpoint()
# calls are merely infrequent, independent of the periodic checkpoint the
# supervisor and workers each perform (see checkpoint()). It cannot help if
# checkpoints are blocked outright — the limit only applies at the moment a
# checkpoint succeeds and the WAL resets.
_QUEUE_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA journal_size_limit=524288000;

CREATE TABLE IF NOT EXISTS pending_branches (
    branch_id      INTEGER PRIMARY KEY,   -- references branches(branch_id)
    n_words        INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    opener    TEXT,
    opener_pattern INTEGER,
    status         TEXT    NOT NULL DEFAULT 'pending',
    claimed_by     TEXT,
    claimed_at     INTEGER,
    completed_at   INTEGER
);

-- priority DESC first so VIP (priority=1) branches drain before priority=0,
-- then n_words DESC within each priority tier for maximum recursive fill-in.
CREATE INDEX IF NOT EXISTS idx_pending_status_pri_n
    ON pending_branches(status, priority DESC, n_words DESC);

-- A user request to precache the response branches of one opener guess.  A
-- request remains distinct even when it shares roots with an earlier request.
CREATE TABLE IF NOT EXISTS opener_work (
    opener_work_id     INTEGER PRIMARY KEY,
    opener        TEXT,
    requested_priority INTEGER NOT NULL,
    requested_at       INTEGER NOT NULL,
    started_at         INTEGER,
    state              TEXT    NOT NULL DEFAULT 'queued'
);

CREATE INDEX IF NOT EXISTS idx_opener_work_priority_order
    ON opener_work(requested_priority DESC, opener_work_id);

-- Every opener-work request that owns a branch.  parent_branch_id records the
-- opener-local promotion lineage; NULL identifies a requested root.
CREATE TABLE IF NOT EXISTS branch_opener_work (
    branch_id      INTEGER NOT NULL REFERENCES branches(branch_id),
    opener_work_id INTEGER NOT NULL REFERENCES opener_work(opener_work_id),
    parent_branch_id INTEGER REFERENCES branches(branch_id),
    opener_pattern INTEGER,
    resolved_at    INTEGER,
    PRIMARY KEY (branch_id, opener_work_id)
);

CREATE INDEX IF NOT EXISTS idx_branch_opener_work_opener
    ON branch_opener_work(opener_work_id, branch_id);

-- One row per worker, overwritten each heartbeat.  In the swarm model a
-- worker is a fungible contributor: it reports which branch and claim it is
-- on purely so the operator can see it is alive and moving (health), not as
-- the unit of progress (that lives in active_branches).  The metric columns
-- (cache_hits/misses, n_cutoff/n_pruned/n_ok) let `status` aggregate
-- cache effectiveness and branch-and-bound pruning across all workers.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id          TEXT    PRIMARY KEY,
    pid                INTEGER NOT NULL,
    current_branch_id  INTEGER,     -- branches(branch_id) this worker is on
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
    cur_path           TEXT,        -- live recursion spine: subset sizes by depth
    cur_help_depth     INTEGER NOT NULL DEFAULT 0,  -- _help_other_branch nesting depth
    opener_work_id     INTEGER,     -- opener_work(opener_work_id) selected at the
                                    -- last claim boundary; NULL if none was selected
    scheduling_role    TEXT,        -- one of SCHEDULING_ROLES; NULL if opener_work_id
                                    -- is NULL
    -- Hint-artifact accounting for the run (hint_cache.py).  All NULL when the
    -- swarm was started without --hint-cache, which is not the same fact as a
    -- measured zero.  The first four count LOOKUPS, from both hint sites: a
    -- cooperative branch is looked up once per worker that computes its
    -- packing order, so they measure coverage (hits/lookups) and legality
    -- (accepted/hits), never branches.  The inline pair is the separate,
    -- same-population count of hints placed at the front of an inline solver
    -- frame and the ones that won it — a cooperative branch's win is recorded
    -- once, per branch, in branch_finalize_log.hint_was_winner instead.
    hint_lookups           INTEGER,
    hint_hits              INTEGER,
    hint_accepted          INTEGER,
    hint_rejected          INTEGER,
    hint_inline_placements INTEGER,
    hint_inline_wins       INTEGER
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    name         TEXT PRIMARY KEY,
    completed_at INTEGER NOT NULL
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
    branch_id      INTEGER PRIMARY KEY,   -- references branches(branch_id)
    n_words        INTEGER NOT NULL,
    n_candidates   INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    requires_opener_membership INTEGER NOT NULL DEFAULT 0,
    opener    TEXT,
    opener_pattern INTEGER,
    best_erd       REAL,
    best_guess     TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER,
    budget         INTEGER,
    best_max_depth INTEGER,
    tainted        INTEGER NOT NULL DEFAULT 0,
    nodes_spent    INTEGER NOT NULL DEFAULT 0,
    infeasible_candidates INTEGER NOT NULL DEFAULT 0,
    infeasible_nodes INTEGER NOT NULL DEFAULT 0,
    -- the guesses played from the root to this branch, space-joined as
    -- "GUESS pattern" (e.g. "SALET -g-g- CRANE bb-y-").  guess_depth is the
    -- count of these guesses.  NULL = no spine recorded (a row predating spine
    -- capture).
    spine          TEXT,
    -- Alpha-beta ceiling the branch is being solved under (NULL = exact solve).
    -- Set once at create_branch and immutable for the branch's lifetime: every
    -- claim prunes against it, so done=1 claims are only interpretable together
    -- with this value (including across worker restarts, which keep done=1
    -- claims).  A finalize that found no best_erd below the ceiling is a CUT
    -- (lower bound only), never a loss and never written to the score cache.
    ceiling        REAL,
    -- Monotone flag: some candidate priced out at >= the bound rather than
    -- being proven infeasible.  Only meaningful at finalize when best_guess is
    -- NULL, where it distinguishes CUT (>= ceiling) from a proven loss.
    cut_occurred   INTEGER NOT NULL DEFAULT 0,
    -- Legacy combined count for candidates completed by either ERD-prune
    -- method, retained for storage and programmatic compatibility.
    bulk_done_candidates INTEGER NOT NULL DEFAULT 0,
    one_level_erd_pruned_candidates INTEGER NOT NULL DEFAULT 0,
    two_level_erd_pruned_candidates INTEGER NOT NULL DEFAULT 0,
    -- Tightest branch bound against which every candidate slot received a
    -- one-level ERD-prune sweep.  NULL means no finite bound has been swept.
    bulk_done_bound REAL,
    -- Timestamp of the current best ERD.  ETA samples start here because a
    -- tighter best changes the two-level prune/survivor mix.
    best_updated_at INTEGER,
    -- claim_next_bundle's forward cursor: count of best-first positions that
    -- have been covered or packed into a bundle at least once.  Positions
    -- [0, pack_cursor) may hold holes (reclaimed/republished, no current
    -- candidate_claims row); those are tracked by position in candidate_holes
    -- and reissued ahead of any virgin position.  Positions
    -- [pack_cursor, n_candidates) are virgin and packed by the
    -- O(1)-amortized forward path.  Lives here (not worker-local memory) so
    -- concurrent claim_next_bundle calls never pack overlapping bundles.
    pack_cursor    INTEGER NOT NULL DEFAULT 0,
    -- When this branch first acquired an incumbent (best_erd went from NULL to
    -- a value), and the branch's node count at that moment.  Together with
    -- created_at they are the "how much did it cost to get a first bound"
    -- measurement a hint run is judged on.  NULL until the first incumbent.
    first_best_at        INTEGER,
    nodes_at_first_best  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
    ON active_branches(status, priority DESC, n_words DESC);

-- Registry mapping each branch's key (the concatenated word list, up to a few
-- KB) to a small integer branch_id.  The high-volume per-candidate tables
-- (candidate_claims, candidate_republish) and the per-branch cut_results
-- reference branch_id instead of carrying the fat blob in every row and index
-- entry, so bulk claim sweeps and full-branch re-reads move integers, not
-- kilobytes.  Append-only: a branch keeps its id across pending/active/deleted
-- transitions, so claims never dangle against a reassigned id.
CREATE TABLE IF NOT EXISTS branches (
    branch_id  INTEGER PRIMARY KEY,
    branch_key BLOB    NOT NULL UNIQUE
);

-- Result channel for ceilinged (alpha-beta) cooperative solves that CUT: the
-- score cache only carries exact optima, so a cut — "every candidate priced
-- out at >= bound; true ERD of this branch is >= bound" — is delivered to
-- waiting parents through this table instead.  A cut proven at `budget` is
-- valid at any budget <= it (fewer guesses cannot beat the bound), mirroring
-- the score cache's loss-reuse rule.  Keyed by (branch_id, budget, tainted):
-- a bound proven at a lower budget does not dominate one proven at a higher
-- budget (it serves fewer consumers), and a tainted bound does not dominate
-- an untainted one (it constrains more consumers), so neither axis may evict
-- the other's row — each class keeps its own maximum bound instead.  A row is
-- a durable proof and survives a supervisor restart (recover_active_branches
-- only drops in-flight claims, not this table); never synced anywhere.
CREATE TABLE IF NOT EXISTS cut_results (
    branch_id  INTEGER NOT NULL,
    budget     INTEGER NOT NULL,
    bound      REAL    NOT NULL,
    tainted    INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (branch_id, budget, tainted)
);

-- One row per claimed candidate of a branch.  A row's existence is an
-- ADVISORY claim ("a worker is probably evaluating this candidate"); only
-- done=1 is AUTHORITATIVE ("fully evaluated and folded into best_erd").
-- Coverage = all n_candidates slots with done=1.  A crashed worker leaves a
-- done=0 row that stale-reclaim deletes, turning it back into an unclaimed
-- gap to be redone — never skipped.  Republish-on-overrun deletes the
-- done=0 rows of a bundle's unfinished remainder the same way, so a
-- reclaimed hole and a republished hole are indistinguishable to the packer,
-- and both are recorded in candidate_holes so the packer can reissue them in
-- best-first order.
-- idx = index into the policy-canonical candidate list (all_words for ERD_ALL).
-- bundle_id groups the rows one claim_next_bundle call hands out together
-- (a worker evaluates its bundle in one best-first sweep); NULL for a row
-- inserted outside the packer (e.g. mark_claims_done's within-candidate
-- overrun promotion).
CREATE TABLE IF NOT EXISTS candidate_claims (
    branch_id  INTEGER NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    bundle_id  TEXT,
    -- Where the candidate sits in the branch's best-first candidate order,
    -- stamped by the packer when it hands the row out.  NULL on a row the
    -- packer did not create (an ERD-prune completion, mark_claims_done) and on
    -- rows that predate the column.  Carried into candidate_holes when the row
    -- is freed, so the packer can reissue the earliest outstanding position
    -- without knowing the order itself (the order lives in the worker).
    best_first_position INTEGER,
    PRIMARY KEY (branch_id, idx)
);
-- idx_candidate_claims_bundle is created in _migrate(), after the bundle_id
-- column is guaranteed to exist on an upgraded database: this CREATE TABLE
-- is a no-op against a pre-existing (pre-bundle_id) table, so an index on
-- bundle_id here would fail on any database that predates this column.

-- Outstanding holes: best-first positions a reclaim or a republish freed,
-- which therefore have no candidate_claims row and are waiting to be reissued.
-- Every hole sits at a position below active_branches.pack_cursor, so the
-- earliest outstanding candidate of a branch is the lowest-positioned row
-- whenever this table is non-empty, and a virgin position at the cursor
-- otherwise.  That is what lets claim_next_bundle reissue a high-ranked
-- republished candidate ahead of untouched later ones without rescanning the
-- branch's claim rows.  A row whose idx has since acquired a claim row (an
-- ERD-prune sweep, the mid-loop publisher) is stale and is dropped the next
-- time the packer looks at it; the end-of-sweep holes pass over
-- candidate_claims remains the authority on coverage, so a missing row here
-- can only cost priority, never a candidate.
CREATE TABLE IF NOT EXISTS candidate_holes (
    branch_id INTEGER NOT NULL,
    idx       INTEGER NOT NULL,
    best_first_position INTEGER,
    PRIMARY KEY (branch_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_candidate_holes_best_first_position
    ON candidate_holes(branch_id, best_first_position);

-- Per-candidate republish count: how many times a candidate's cross-candidate
-- bundle has overrun before this candidate finished (see
-- adaptive_claim_packing.md §7's bounded-republish-depth guardrail).  Survives
-- the delete/re-insert cycle a republished candidate's candidate_claims row
-- goes through, so the packer can tell a chronically-stranded candidate apart
-- from one republished for the first time.  Dropped with the rest of the
-- branch's transient state by delete_branch.
CREATE TABLE IF NOT EXISTS candidate_republish (
    branch_id INTEGER NOT NULL,
    idx       INTEGER NOT NULL,
    count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_id, idx)
);

-- Per-size-bucket online cost model (time-weighted geometric mean of
-- recursion-node cost).  Keyed by (policy, size_bucket, budget): the same word
-- count costs very different node totals at different remaining-guess budgets,
-- so budget is part of the key.  policy keeps ERD_ALL and ERD_ANSWERS models
-- from cross-contaminating.
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
"""

# Telemetry tables, in the attached `telemetry` database file.  Every table
# here is outbound measurement data: written best-effort, never read by any
# runtime control path (bundle_stats is aggregated once at finalize, then
# dropped).  All statements qualify these names as telemetry.* so a stray
# same-named table in the queue file can never be read or written silently.
_TELEMETRY_SCHEMA_SQL = """
PRAGMA telemetry.journal_mode=WAL;
PRAGMA telemetry.synchronous=NORMAL;
PRAGMA telemetry.journal_size_limit=524288000;

-- One row per bundle a worker has reported on: the nodes actually spent and
-- the wall-clock span of evaluating it (straggler/reclaim-window diagnostic
-- — NOT claim-handout coordination overhead, which is claim_telemetry's
-- busy_wait_millis), aggregated into branch_finalize_log at finalize and
-- dropped with the rest of the branch's transient state by delete_branch.
-- censored=1 marks a bundle that hit its node/wall cap and republished an
-- unfinished remainder (see ERDQueue.republish_remainder) — nodes is then a
-- lower bound on what the bundle's original member set would have cost.
-- Both nodes and wall_millis exclude any `forced` member's own span (see
-- evaluate_bundle): a forced candidate's real cost lands in its promoted
-- sub-branches' own finalize-log rows instead, so this bundle's row is not
-- a straggler diagnostic for bundles containing one.
CREATE TABLE IF NOT EXISTS telemetry.bundle_stats (
    branch_key   BLOB    NOT NULL,
    bundle_id    TEXT    NOT NULL,
    nodes        INTEGER NOT NULL,
    wall_millis  INTEGER NOT NULL,
    censored     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_key, bundle_id)
);

-- One row per worker-side two-level ERD-prune scan.  inspected_candidate_count
-- includes survivors that proceed to ordinary evaluation; pruned_candidate_count
-- counts only the candidates completed by the bound.  The wall span measures
-- the bound checks themselves, before their SQLite completion transaction.
CREATE TABLE IF NOT EXISTS telemetry.two_level_prune_telemetry (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id                 INTEGER NOT NULL,
    inspected_candidate_count INTEGER NOT NULL,
    pruned_candidate_count    INTEGER NOT NULL,
    bound_erd                 REAL,
    worker_count              INTEGER,
    branch_worker_count       INTEGER,
    wall_millis               INTEGER NOT NULL,
    epoch                     INTEGER NOT NULL DEFAULT 0,
    recorded_at               INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS telemetry.idx_two_level_prune_telemetry_branch_time
    ON two_level_prune_telemetry(branch_id, recorded_at);

-- Raw per-solve samples for offline distribution analysis and threshold tuning.
-- censored = 1 marks a sample whose unit was handed off at the node/wall cap
-- before it finished: `nodes` is then a LOWER BOUND on the true cost, and an
-- offline survival fit must treat it as such rather than as an exact point.
CREATE TABLE IF NOT EXISTS telemetry.cost_samples (
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
-- branch_id/spine attribute a row to the branch its claim belonged to (bulk
-- lower-bound proofs and other branch-less callers leave both NULL).
-- branch_id is the branches registry surrogate (see _intern_branch), not the
-- branch_key BLOB itself: the registry is append-only, so a branch_id stays
-- resolvable to its branch_key/word-list forever, even long after the branch
-- itself is finalized and deleted from active_branches.  Resolving it back
-- means joining against branches, which lives in the main queue file, not
-- this attached one — cross-file for a live ERDQueue (branches is already
-- attached on the same connection); an explicit ATTACH when querying the
-- telemetry file standalone.  worker_id/bundle_id/idx identify which worker
-- evaluated which candidate,
-- with bundle_start_idx/bundle_end_idx the claiming bundle's full index range
-- (bundle_id and both range columns are NULL for a claim taken outside a
-- bundle).  claim_transaction_millis + claim_commit_millis split
-- coordination_millis's claim-handout portion into the scan/write phase and
-- the COMMIT itself; busy_wait_millis is the write-lock wait across the claim
-- paths taken while coordinating; scheduling_millis is the work-selection scan
-- that chose this branch (opener-work ordering, pending promotion, joining an
-- in-progress branch) excluding the phases already counted; idle_millis is
-- what is left over.  Those five partition coordination_millis exactly, and
-- idle_millis means genuinely unaccounted wait -- scheduling work is called
-- out separately so a large idle_millis cannot be misread as starved workers
-- when it is really scan cost.  All five span time BETWEEN candidate
-- evaluations.  candidate_evaluation_millis is the elapsed solver work for this
-- candidate.  Queue work done
-- during a candidate's own evaluation (sub-branch promotion taking the write
-- lock) sits inside the evaluation span that coordination_millis excludes and
-- is deliberately not counted, since counting it would make the parts exceed
-- the whole.  Every row here is one candidate evaluation:
-- finalize cost is NOT recorded here (it belongs to a branch, not to any
-- claim, and lands on branch_finalize_log.cache_write_millis instead) so
-- COUNT(*) over this table is a true claim count.
CREATE TABLE IF NOT EXISTS telemetry.claim_telemetry (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words                   INTEGER NOT NULL,
    coordination_millis       INTEGER NOT NULL,
    candidate_evaluation_millis INTEGER,
    work_nodes                INTEGER NOT NULL,
    claim_retries             INTEGER,
    busy_wait_millis          INTEGER,
    worker_count              INTEGER,
    branch_worker_count       INTEGER,
    evaluation_bound_erd      REAL,
    branch_id                 INTEGER,
    spine                     TEXT,
    worker_id                 TEXT,
    bundle_id                 TEXT,
    idx                       INTEGER,
    bundle_start_idx          INTEGER,
    bundle_end_idx            INTEGER,
    claim_transaction_millis  INTEGER,
    claim_commit_millis       INTEGER,
    scheduling_millis         INTEGER,
    idle_millis               INTEGER,
    epoch                     INTEGER NOT NULL DEFAULT 0,
    recorded_at               INTEGER NOT NULL
);

-- Durable per-branch timing/cost record, written at finalize BEFORE delete_branch
-- copies the otherwise-destroyed active_branches.created_at/finalized_at/
-- nodes_spent out of the transient row.  Subbranches finalize as first-class
-- branches under recursive promotion, so each gets one row here.  The packer-era
-- columns (n_bundles, max_bundle_nodes, total_bundle_wall_millis, censored_units)
-- are NULL under single-candidate claiming and populated once bundling lands.
CREATE TABLE IF NOT EXISTS telemetry.branch_finalize_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_key              BLOB,
    spine                   TEXT,
    n_words                 INTEGER,
    budget                  INTEGER,
    epoch                   INTEGER NOT NULL DEFAULT 0,
    created_at              INTEGER,
    finalized_at            INTEGER,
    nodes_spent             INTEGER,
    infeasible_candidates   INTEGER,
    infeasible_nodes        INTEGER,
    -- Candidates completed through worker evaluation; ERD prunes are counted
    -- separately by method.
    n_claims                INTEGER,
    bulk_done_candidates    INTEGER,
    one_level_erd_pruned_candidates INTEGER,
    two_level_erd_pruned_candidates INTEGER,
    n_bundles               INTEGER,
    max_bundle_nodes        INTEGER,
    total_bundle_wall_millis INTEGER,
    censored_units          INTEGER,
    -- Alpha-beta ceiling the branch was solved under; NULL = exact solve.
    ceiling                 REAL,
    -- How the solve ended: 'exact' (true optimum found — the only outcome
    -- written to the score cache), 'cut' (every candidate priced out at
    -- >= ceiling; lower bound only), 'loss' (proven unsolvable within budget).
    -- The exact/cut split per (n_words, budget) is the payoff measurement for
    -- ceiling propagation.
    outcome                 TEXT,
    -- Solved branch result captured at finalize time.  NULL when the finalize
    -- row predates these columns or when the outcome was not an exact solve.
    best_guess              TEXT,
    best_erd                REAL,
    -- Wall time this worker spent publishing the branch's result once it won
    -- the finalize: the score-cache/loss/cut writes and the cost-model fold.
    -- The finalize phase of the issue-197 coordination breakdown, recorded per
    -- branch because it belongs to the branch, not to any one claim.  Covers
    -- the publish work only, since the row carrying it is written immediately
    -- after (a row cannot time its own insert or the queue cleanup past it).
    cache_write_millis      INTEGER,
    -- Best-first scheduling evidence, captured from the branch's own claim
    -- rows the moment before delete_branch drops them.  A winner whose rank
    -- is far below the weakest rank completed ahead of it is a priority
    -- inversion: the branch spent itself on later candidates while a strong
    -- one sat republished.  NULL on a branch that never claimed through the
    -- packer, and on rows written before these columns existed.
    winner_best_first_position INTEGER,
    winner_republish_count  INTEGER,
    candidates_completed_before_winner INTEGER,
    max_best_first_position_before_winner INTEGER,
    republished_candidates  INTEGER,
    max_candidate_republish_count INTEGER,
    -- Hint-artifact payoff (hint_cache.py).  hint_word is the candidate the
    -- read-only historical artifact named for this branch and hint_was_winner
    -- whether the branch's own recomputed winner turned out to be that word;
    -- both NULL on a run with no hint artifact.  first_best_at and
    -- nodes_at_first_best are when the branch acquired any incumbent at all
    -- and what it had spent by then, which against created_at is the cost a
    -- good hint is supposed to remove.
    hint_word               TEXT,
    hint_was_winner         INTEGER,
    first_best_at           INTEGER,
    nodes_at_first_best     INTEGER,
    recorded_at             INTEGER NOT NULL
);

-- One row per re-solve forced by a cut: a consumer needed branch_key but the
-- only known result was a cut whose bound (or budget validity) could not
-- satisfy it.  The cost side of the ceiling-propagation ledger — how often a
-- ceilinged solve's non-cacheability makes someone redo work.  wanted_ceiling
-- NULL = the consumer needed an exact value.
CREATE TABLE IF NOT EXISTS telemetry.cut_reuse_misses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_key      BLOB,
    n_words         INTEGER NOT NULL,
    budget          INTEGER NOT NULL,
    wanted_ceiling  REAL,
    available_bound REAL    NOT NULL,
    available_budget INTEGER NOT NULL,
    epoch           INTEGER NOT NULL DEFAULT 0,
    recorded_at     INTEGER NOT NULL
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
CREATE TABLE IF NOT EXISTS telemetry.candidate_accuracy (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_key                 BLOB,
    -- branch_id/worker_id/bundle_id/idx use the same identity names as
    -- claim_telemetry, so an accuracy point joins to its claim without
    -- reconstructing transient queue state.  candidate_word preserves the
    -- candidate identity after its ordering changes.
    branch_id                  INTEGER,
    candidate_word             TEXT,
    worker_id                  TEXT,
    bundle_id                  TEXT,
    idx                        INTEGER,
    n_words                    INTEGER NOT NULL,
    budget                     INTEGER,
    predicted_work             REAL,
    bound_erd                  REAL,
    candidate_cost_lower_bound REAL,
    erd_lower_bound_pruned     INTEGER NOT NULL,
    actual_nodes               INTEGER NOT NULL,
    group_sizes                TEXT,
    opener                TEXT,
    -- started_at is the beginning of evaluation; recorded_at remains the
    -- completion time so existing readers retain their timestamp contract.
    started_at                 INTEGER,
    evaluation_millis          INTEGER,
    -- exact, cut, loss, or cancelled.  NULL means the row predates outcome
    -- recording rather than any particular outcome.
    outcome                    TEXT,
    republish_count            INTEGER,
    epoch                      INTEGER NOT NULL DEFAULT 0,
    recorded_at                INTEGER NOT NULL
);

-- One row per wall-clock backstop firing: a frame handed off its remainder
-- because it ran longer than COLD_BACKSTOP_SECONDS rather than because the
-- node-proportionate overrun check tripped.  Exists to tune COLD_BACKSTOP_SECONDS
-- offline: how often the time cap (not the node check) drives a handoff, at what
-- frame sizes, and whether the cost model was cold (predicted_nodes NULL) or warm
-- at the time.  Outbound-only; never read by any runtime control path.
CREATE TABLE IF NOT EXISTS telemetry.backstop_telemetry (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    n_words              INTEGER NOT NULL,
    budget               INTEGER,            -- frame's remaining-guess budget at fire
    elapsed_millis       INTEGER NOT NULL,   -- wall time in frame at fire
    nodes                INTEGER NOT NULL,   -- nodes spent in frame at fire
    predicted_nodes      REAL,               -- typical(n) at entry; NULL = cold model
    remaining_candidates INTEGER NOT NULL,   -- candidates handed off
    epoch                INTEGER NOT NULL DEFAULT 0,
    recorded_at          INTEGER NOT NULL
);
"""

# Every table _TELEMETRY_SCHEMA_SQL creates, used to detect a queue file that
# predates the telemetry split (see _absorb_legacy_telemetry_tables).
_TELEMETRY_TABLES = (
    "bundle_stats", "cost_samples", "claim_telemetry",
    "branch_finalize_log", "candidate_accuracy", "backstop_telemetry",
    "cut_reuse_misses",
)


def derive_telemetry_path(db_path: str) -> str:
    """Default telemetry file path for a queue at db_path: the sibling file
    <stem>_telemetry<ext> (erd_queue.sqlite3 -> erd_queue_telemetry.sqlite3).
    ':memory:' maps to ':memory:', which attaches as an independent private
    in-memory database."""
    if db_path == ":memory:":
        return ":memory:"
    root, ext = os.path.splitext(db_path)
    return f"{root}_telemetry{ext}"


def best_first_rank(best_first_position):
    """1-based rank for a 0-based best-first position; None stays None.

    Positions are stored 0-based, matching the index into a branch's
    candidate order; every display of one is a rank counting from 1.
    """
    return None if best_first_position is None else best_first_position + 1


def check_opener_priority_range(priority: int) -> None:
    """Raise ValueError unless priority is within the opener-work range."""
    if not OPENER_PRIORITY_MIN <= priority <= OPENER_PRIORITY_MAX:
        raise ValueError(
            "opener-work priority must be between "
            f"{OPENER_PRIORITY_MIN} and {OPENER_PRIORITY_MAX}")


def read_only_database_uri(db_path: str) -> str:
    """Return a SQLite URI that opens an existing database without writes."""
    absolute_path = os.path.abspath(db_path)
    return f"file:{quote(absolute_path, safe='/')}?mode=ro"


class ERDQueue:
    """SQLite-backed work queue for the parallel ERD_ALL precache job."""

    def __init__(self, db_path: str, timeout: float = 30.0,
                 telemetry_path: str = None, initialize_schema: bool = True):
        self.db_path = db_path
        self._timeout = timeout
        connection_path = (
            db_path if initialize_schema else read_only_database_uri(db_path)
        )
        self._conn = sqlite3.connect(
            connection_path, timeout=timeout, isolation_level=None,
            uri=not initialize_schema,
        )
        self._conn.row_factory = sqlite3.Row
        if telemetry_path is None:
            telemetry_path = derive_telemetry_path(db_path)
        self.telemetry_path = telemetry_path
        attached_path = (
            telemetry_path
            if initialize_schema else read_only_database_uri(telemetry_path)
        )
        self._conn.execute("ATTACH DATABASE ? AS telemetry", (attached_path,))
        if initialize_schema:
            self._conn.executescript(_QUEUE_SCHEMA_SQL)
            self._conn.executescript(_TELEMETRY_SCHEMA_SQL)
            self._absorb_legacy_telemetry_tables()
            self._migrate()
            self._assert_schema()
        else:
            self._conn.execute("PRAGMA query_only = ON")
        # Active epoch, read once: workers open the queue after the supervisor has
        # stamped run_meta.epoch, and are restarted across a cutover, so caching
        # here is safe and keeps every telemetry insert off a per-row SELECT.
        self.epoch = int(self.get_meta('epoch') or 0)
        # Per-claim contention metrics, refreshed by the claim path and read by
        # the next add_claim_telemetry without changing claim_* return shapes.
        self._last_claim_busy_millis = 0
        self._last_claim_retries = 0
        self._last_claim_transaction_millis = 0
        self._last_claim_commit_millis = 0
        # Monotonic per-connection counter for bundle_id generation: paired
        # with worker_id and this process's pid, it is unique without a
        # timestamp-collision risk (two bundles claimed by the same worker
        # within the same second) AND without a cross-restart collision risk
        # (a crashed-and-respawned worker reuses its fixed worker_id slot,
        # but never its old pid, so a fresh process can never re-mint a
        # bundle_id a still-open branch's bundle_stats already used).
        self._pid = os.getpid()
        self._bundle_seq = 0
        # Cumulative WAL read/write attribution for this connection, keyed by
        # (table/operation).  The WAL file records no per-table breakdown of
        # what filled it; these counters do, so a worker can log which traffic
        # it is pouring into the shared WAL.  Cumulative per connection: the
        # worker logs deltas between snapshots as a rate (wal_traffic_snapshot),
        # which needs no cross-process coordination.
        self._wal_traffic_rows = collections.Counter()
        self._wal_traffic_bytes = collections.Counter()
        # branch_key BLOB -> branch_id int, memoized so a hot branch costs no
        # SQL after its first lookup.  Only positive results are cached; the
        # registry is append-only, so a cached id never goes stale.
        self._branch_id_cache = {}

    def __del__(self):
        conn = getattr(self, '_conn', None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.ProgrammingError:
                # conn was created on a different thread than the one
                # finalizing it; SQLite connections are thread-affine, so
                # closing here is impossible.
                pass

    def _tally_wal_traffic(self, category: str, rows: int, approx_bytes: int):
        """Attribute `rows`/`approx_bytes` of WAL traffic to `category`.

        approx_bytes is a key-width estimate, not an exact frame count: the
        point is relative magnitude between categories (which table is the
        firehose), not a byte-accurate WAL model.

        Write categories are floored at a per-commit page estimate: each tally
        call is roughly one autocommit transaction, and a transaction appends
        whole 4 KiB pages to the WAL — a one-row write costs a few pages (leaf,
        index, overhead), never its row width.  Without the floor, a storm of
        tiny transactions reads as ~100 bytes each and the attribution
        under-reports the real WAL fill rate by orders of magnitude.  Read
        categories (suffixed '(read)') append nothing to the WAL and are
        exempt."""
        if rows <= 0:
            return
        if not category.endswith('(read)'):
            approx_bytes = max(approx_bytes,
                               _WAL_PAGE_BYTES * (2 + rows // 8))
        self._wal_traffic_rows[category] += rows
        self._wal_traffic_bytes[category] += max(0, approx_bytes)

    def wal_traffic_snapshot(self):
        """A copy of the cumulative (rows, approx_bytes) traffic counters, for
        a caller computing a rate from the delta between two snapshots."""
        return (collections.Counter(self._wal_traffic_rows),
                collections.Counter(self._wal_traffic_bytes))

    def wal_traffic_report(self, top: int = 6) -> str:
        """One-line-per-category summary of cumulative WAL traffic on this
        connection, largest byte-estimate first.  Empty string when nothing
        has been tallied yet."""
        if not self._wal_traffic_bytes:
            return ''
        lines = []
        for category, approx_bytes in self._wal_traffic_bytes.most_common(top):
            rows = self._wal_traffic_rows[category]
            lines.append(f'    {category}: {rows:,} rows, '
                         f'~{approx_bytes / 2 ** 20:,.1f} MiB')
        return '\n'.join(lines)

    def _absorb_legacy_telemetry_tables(self):
        """Handle a queue file that carries telemetry tables in the queue
        database itself (a file from before the telemetry split).

        Qualified telemetry.* statements would silently bypass such tables,
        so they must not be left in place: an empty one is dropped (with a
        warning); one that still holds rows refuses to open, so its data is
        archived deliberately instead of ignored silently.
        """
        for table in _TELEMETRY_TABLES:
            present = self._conn.execute(
                "SELECT 1 FROM main.sqlite_master "
                "WHERE type = 'table' AND name = ?", (table,)).fetchone()
            if present is None:
                continue
            has_rows = self._conn.execute(
                f"SELECT 1 FROM main.{table} LIMIT 1").fetchone()
            if has_rows is None:
                self._conn.execute(f"DROP TABLE main.{table}")
                logger.warning(
                    "dropped empty pre-split telemetry table %r from the "
                    "queue database (telemetry lives in %s)",
                    table, self.telemetry_path)
            else:
                raise RuntimeError(
                    f"queue database predates the telemetry split: table "
                    f"{table!r} still holds rows in the queue file, where "
                    f"current code would silently ignore them. Archive the "
                    f"queue file (rename it) and start a fresh queue; "
                    f"telemetry now lives in {self.telemetry_path!r}.")

    def _rename_source_to_opener(self):
        """Rename the pre-opener-vocabulary source_* schema to opener_* naming.

        Runs first in _migrate(), before any other step reads or writes the
        renamed tables/columns by their new names.  A fresh database never
        held the old names (the schema script above already creates
        opener_work/branch_opener_work directly), so every step here is
        conditioned on the old name actually being present rather than on
        the schema_migrations guard alone: CREATE TABLE IF NOT EXISTS runs
        before _migrate() and would otherwise have already created an empty
        opener_work/branch_opener_work table alongside an old database's
        source_work/branch_source_work, which a bare ALTER TABLE ... RENAME
        TO would then refuse (the destination name already exists).
        """
        migration_name = "rename_source_to_opener"
        already_applied = self._conn.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (migration_name,)).fetchone() is not None
        existing_tables = {r["name"] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        pending_columns = {r["name"] for r in
                           self._conn.execute("PRAGMA table_info(pending_branches)")}
        active_columns = {r["name"] for r in
                          self._conn.execute("PRAGMA table_info(active_branches)")}
        renaming_anything = (
            "source_work" in existing_tables
            or "branch_source_work" in existing_tables
            or "source_word" in pending_columns
            or "source_pattern" in pending_columns
            or "source_word" in active_columns
            or "source_pattern" in active_columns
            or "requires_source_membership" in active_columns
        )
        if renaming_anything:
            # Both queue views read columns this migration renames below, and
            # SQLite's RENAME COLUMN tries to rewrite any view that references
            # the renamed column in place -- which fails on a definition still
            # shaped around the old table or column.  Drop both before any
            # rename runs; _rebuild_queue_views recreates them from current
            # code once every renamed column exists under its new name.
            self._conn.execute("DROP VIEW IF EXISTS active_branch_owner_rows")
            self._conn.execute("DROP VIEW IF EXISTS live_branch_source_rows")
            self._conn.execute("DROP VIEW IF EXISTS live_branch_opener_rows")
        if "source_work" in existing_tables:
            # ALTER TABLE ... RENAME TO renames the table but leaves any index
            # on it under its old name -- idx_source_work_priority_order does
            # not become idx_opener_work_priority_order, it stays exactly as
            # named and simply now points at opener_work.  Dropping only the
            # new name here is therefore a no-op on a real old-schema database
            # (that name has never existed yet) and leaves the genuinely old
            # one behind, duplicating the index CREATE INDEX below adds.  Both
            # names must be dropped; which happens before or after the rename
            # makes no difference, since DROP INDEX addresses the index by its
            # own name regardless of which table it is currently attached to.
            self._conn.execute("DROP INDEX IF EXISTS idx_source_work_priority_order")
            self._conn.execute("DROP INDEX IF EXISTS idx_opener_work_priority_order")
            self._conn.execute("DROP TABLE IF EXISTS opener_work")
            self._conn.execute("ALTER TABLE source_work RENAME TO opener_work")
            self._conn.execute(
                "ALTER TABLE opener_work RENAME COLUMN source_word TO opener")
            self._conn.execute(
                "ALTER TABLE opener_work RENAME COLUMN source_work_id "
                "TO opener_work_id")
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_opener_work_priority_order
                    ON opener_work(requested_priority DESC, opener_work_id)
            """)
        if "branch_source_work" in existing_tables:
            # Same index-survives-the-rename hazard as opener_work above.
            self._conn.execute("DROP INDEX IF EXISTS idx_branch_source_work_source")
            self._conn.execute("DROP INDEX IF EXISTS idx_branch_opener_work_opener")
            self._conn.execute("DROP TABLE IF EXISTS branch_opener_work")
            self._conn.execute(
                "ALTER TABLE branch_source_work RENAME TO branch_opener_work")
            self._conn.execute(
                "ALTER TABLE branch_opener_work RENAME COLUMN root_pattern "
                "TO opener_pattern")
            self._conn.execute(
                "ALTER TABLE branch_opener_work RENAME COLUMN source_work_id "
                "TO opener_work_id")
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_branch_opener_work_opener
                    ON branch_opener_work(opener_work_id, branch_id)
            """)
        if "source_word" in pending_columns:
            self._conn.execute(
                "ALTER TABLE pending_branches RENAME COLUMN source_word TO opener")
        if "source_pattern" in pending_columns:
            self._conn.execute(
                "ALTER TABLE pending_branches RENAME COLUMN source_pattern "
                "TO opener_pattern")
        if "source_word" in active_columns:
            self._conn.execute(
                "ALTER TABLE active_branches RENAME COLUMN source_word TO opener")
        if "source_pattern" in active_columns:
            self._conn.execute(
                "ALTER TABLE active_branches RENAME COLUMN source_pattern "
                "TO opener_pattern")
        if "requires_source_membership" in active_columns:
            self._conn.execute(
                "ALTER TABLE active_branches RENAME COLUMN "
                "requires_source_membership TO requires_opener_membership")
        heartbeat_columns = {r["name"] for r in
                             self._conn.execute("PRAGMA table_info(worker_heartbeat)")}
        if "source_work_id" in heartbeat_columns:
            self._conn.execute(
                "ALTER TABLE worker_heartbeat RENAME COLUMN "
                "source_work_id TO opener_work_id")
        telemetry_columns = {r["name"] for r in self._conn.execute(
            "PRAGMA telemetry.table_info(candidate_accuracy)")}
        if "source_word" in telemetry_columns:
            self._conn.execute(
                "ALTER TABLE telemetry.candidate_accuracy "
                "RENAME COLUMN source_word TO opener")
        if not already_applied:
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (name, completed_at) "
                "VALUES (?, ?)", (migration_name, int(time.time())))

    def _migrate(self):
        self._rename_source_to_opener()
        # active_branches.spine: additive, nullable, no backfill.  Existing rows
        # keep NULL (display falls back to the opener word) until re-promoted.
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
                    opener    TEXT,
                    opener_pattern INTEGER,
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
                           opener, opener_pattern, best_erd, best_guess,
                           status, created_at, finalized_at, budget,
                           best_max_depth, tainted, nodes_spent, spine
                    FROM active_branches;
                DROP TABLE active_branches;
                ALTER TABLE active_branches_new RENAME TO active_branches;
                CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
                    ON active_branches(status, priority DESC, n_words DESC);
            """)
        # cost_model re-key: (policy, size_bucket) -> (policy, size_bucket, budget).
        # The same word count costs very different node totals at different
        # budgets, so the old single-budget bucket conflated them.  Rebuild with
        # budget in the key; existing accumulators carry across at budget = -1,
        # a placeholder value no longer read by any query — the pre-rekey model
        # is retired and re-warms per budget from fresh samples.
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

        # Alpha-beta ceiling propagation: additive/nullable-default on both
        # files.  Existing active_branches rows keep NULL ceiling (exact solve,
        # the only regime that produced them) and cut_occurred 0; existing
        # finalize-log rows keep NULL ceiling/outcome (all pre-ceiling rows
        # were exact or loss, distinguishable offline via the score cache).
        self._add_columns("active_branches", {
            "ceiling": "REAL",
            "cut_occurred": "INTEGER NOT NULL DEFAULT 0",
            "bulk_done_candidates": "INTEGER NOT NULL DEFAULT 0",
            "one_level_erd_pruned_candidates": "INTEGER NOT NULL DEFAULT 0",
            "two_level_erd_pruned_candidates": "INTEGER NOT NULL DEFAULT 0",
            "bulk_done_bound": "REAL",
            "best_updated_at": "INTEGER",
        })
        self._add_columns("branch_finalize_log", {
            "infeasible_candidates": "INTEGER",
            "infeasible_nodes": "INTEGER",
        }, schema="telemetry")
        # An existing incumbent predates its bound timestamp.  Begin its ETA
        # sample now instead of using work measured against an unknown bound.
        self._conn.execute("""
            UPDATE active_branches
            SET best_updated_at = ?
            WHERE status = 'open' AND best_erd IS NOT NULL
              AND best_updated_at IS NULL
        """, (int(time.time()),))
        self._add_columns("branch_finalize_log", {
            "ceiling": "REAL",
            "outcome": "TEXT",
            "bulk_done_candidates": "INTEGER",
            "one_level_erd_pruned_candidates": "INTEGER",
            "two_level_erd_pruned_candidates": "INTEGER",
            "best_guess": "TEXT",
            "best_erd": "REAL",
        }, schema="telemetry")
        self._add_columns("two_level_prune_telemetry", {
            "bound_erd": "REAL",
            "worker_count": "INTEGER",
            "branch_worker_count": "INTEGER",
        }, schema="telemetry")
        # Every persisted aggregate predating the provenance split came from
        # the one-level candidate bound; worker-side two-level pruning did not
        # yet exist.  Keep bulk_done_candidates as the legacy combined count.
        provenance_migration = "split_erd_prune_provenance"
        if self._conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = ?",
                (provenance_migration,)).fetchone() is None:
            self._conn.execute("""
                UPDATE active_branches
                SET one_level_erd_pruned_candidates = bulk_done_candidates,
                    two_level_erd_pruned_candidates = 0
            """)
            self._conn.execute("""
                UPDATE telemetry.branch_finalize_log
                SET one_level_erd_pruned_candidates =
                        COALESCE(bulk_done_candidates, 0),
                    two_level_erd_pruned_candidates = 0
            """)
            self._conn.execute(
                "INSERT INTO schema_migrations (name, completed_at) "
                "VALUES (?, ?)", (provenance_migration, int(time.time())))

        # Branch-attributed claim telemetry and coordination phase breakdown
        # (issue #197): additive/nullable, so an existing row simply keeps
        # NULL in every new column — it predates per-branch attribution and
        # the phase split, not an attribution failure.  The finalize phase
        # lands on branch_finalize_log, since it belongs to a branch rather
        # than to any single claim.
        self._add_columns("claim_telemetry", {
            "candidate_evaluation_millis": "INTEGER",
            "branch_id": "INTEGER",
            "spine": "TEXT",
            "worker_id": "TEXT",
            "bundle_id": "TEXT",
            "idx": "INTEGER",
            "bundle_start_idx": "INTEGER",
            "bundle_end_idx": "INTEGER",
            "claim_transaction_millis": "INTEGER",
            "claim_commit_millis": "INTEGER",
            "scheduling_millis": "INTEGER",
            "idle_millis": "INTEGER",
            "branch_worker_count": "INTEGER",
            "evaluation_bound_erd": "REAL",
        }, schema="telemetry")

        # Candidate-accuracy identity and lifecycle fields (issue #223).
        # Existing rows retain NULL for every added field: they record a
        # calibration point, but cannot identify its candidate or outcome.
        candidate_accuracy_migration = "candidate_accuracy_identity_lifecycle"
        if self._conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = ?",
                (candidate_accuracy_migration,)).fetchone() is None:
            self._add_columns("candidate_accuracy", {
                "branch_id": "INTEGER",
                "candidate_word": "TEXT",
                "worker_id": "TEXT",
                "bundle_id": "TEXT",
                "idx": "INTEGER",
                "started_at": "INTEGER",
                "evaluation_millis": "INTEGER",
                "outcome": "TEXT",
                "republish_count": "INTEGER",
            }, schema="telemetry")
            self._conn.execute(
                "INSERT INTO schema_migrations (name, completed_at) "
                "VALUES (?, ?)",
                (candidate_accuracy_migration, int(time.time())))
        self._add_columns("branch_finalize_log", {
            "cache_write_millis": "INTEGER",
        }, schema="telemetry")

        # Best-first scheduling evidence per finalized branch (issue #258):
        # additive/nullable, so a row written before these columns simply has
        # no scheduling evidence to show.
        self._add_columns("branch_finalize_log", {
            "winner_best_first_position": "INTEGER",
            "winner_republish_count": "INTEGER",
            "candidates_completed_before_winner": "INTEGER",
            "max_best_first_position_before_winner": "INTEGER",
            "republished_candidates": "INTEGER",
            "max_candidate_republish_count": "INTEGER",
        }, schema="telemetry")

        report_indexes = (
            ("branch_finalize_log", {"branch_key", "recorded_at"},
             "idx_branch_finalize_log_branch_recorded_at",
             "branch_key, recorded_at"),
            ("branch_finalize_log", {"epoch", "recorded_at", "id"},
             "idx_branch_finalize_log_epoch_recorded_id",
             "epoch, recorded_at DESC, id DESC"),
            ("branch_finalize_log", {"finalized_at", "id"},
             "idx_branch_finalize_log_finalized_at",
             "finalized_at DESC, id DESC"),
            ("cut_reuse_misses", {"branch_key", "recorded_at"},
             "idx_cut_reuse_misses_branch_recorded_at",
             "branch_key, recorded_at"),
            ("cut_reuse_misses", {"epoch", "recorded_at", "id"},
             "idx_cut_reuse_misses_epoch_recorded_id",
             "epoch, recorded_at DESC, id DESC"),
            ("claim_telemetry", {"epoch", "id"},
             "idx_claim_telemetry_epoch_id", "epoch, id"),
            ("claim_telemetry", {"epoch", "recorded_at", "id"},
             "idx_claim_telemetry_epoch_recorded_id",
             "epoch, recorded_at DESC, id DESC"),
            ("claim_telemetry", {"branch_id", "recorded_at"},
             "idx_claim_telemetry_branch_recorded_at",
             "branch_id, recorded_at"),
            ("candidate_accuracy", {"epoch", "id"},
             "idx_candidate_accuracy_epoch_id", "epoch, id"),
            ("candidate_accuracy", {"epoch", "recorded_at", "id"},
             "idx_candidate_accuracy_epoch_recorded_id",
             "epoch, recorded_at DESC, id DESC"),
        )
        for table, required_columns, index_name, indexed_columns in report_indexes:
            columns = {row["name"] for row in self._conn.execute(
                f"PRAGMA telemetry.table_info({table})")}
            if required_columns <= columns:
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS telemetry.{index_name} "
                    f"ON {table}({indexed_columns})")

        # Branch-key normalization: the fat branch_key BLOB (the concatenated
        # word list, up to a few KB) used to sit in every candidate_claims,
        # candidate_republish, and cut_results row and be duplicated in the
        # claims bundle index.  Move it to the branches registry (stored once)
        # and rebuild those tables to reference the small integer branch_id.
        # Each table is guarded independently on its own pre-normalization
        # column, so a database mid-normalization (or a partial fixture) is
        # completed table by table rather than assuming all three move
        # together.  Runs with workers stopped, so no writer races the swap.
        def _cols(table):
            return {r["name"] for r in
                    self._conn.execute(f"PRAGMA table_info({table})")}

        def _has_branch_key(table):
            return "branch_key" in _cols(table)

        def _rebuildable(table, required):
            # Only rebuild a table whose full pre-normalization shape is present.
            # A malformed table (e.g. one missing a column) is left untouched so
            # _assert_schema refuses to open with a clear column-drift message,
            # rather than the rebuild failing mid-SELECT.
            return _has_branch_key(table) and required <= _cols(table)

        normalized_tables = [t for t in
                             ("candidate_claims", "candidate_republish",
                              "cut_results", "active_branches",
                              "pending_branches") if _has_branch_key(t)]
        for table in normalized_tables:
            self._conn.execute(
                f"INSERT OR IGNORE INTO branches (branch_key) "
                f"SELECT branch_key FROM {table}")
        hb_cols = {r["name"] for r in
                   self._conn.execute("PRAGMA table_info(worker_heartbeat)")}
        if "current_branch_key" in hb_cols:
            self._conn.execute(
                "INSERT OR IGNORE INTO branches (branch_key) "
                "SELECT current_branch_key FROM worker_heartbeat "
                "WHERE current_branch_key IS NOT NULL")
        if _has_branch_key("candidate_claims"):
            self._conn.executescript("""
                CREATE TABLE candidate_claims_new (
                    branch_id  INTEGER NOT NULL,
                    idx        INTEGER NOT NULL,
                    claimed_by TEXT,
                    claimed_at INTEGER,
                    done       INTEGER NOT NULL DEFAULT 0,
                    done_at    INTEGER,
                    bundle_id  TEXT,
                    PRIMARY KEY (branch_id, idx)
                );
                INSERT INTO candidate_claims_new
                    SELECT b.branch_id, c.idx, c.claimed_by, c.claimed_at,
                           c.done, c.done_at, c.bundle_id
                    FROM candidate_claims c
                    JOIN branches b ON b.branch_key = c.branch_key;
                DROP TABLE candidate_claims;
                ALTER TABLE candidate_claims_new RENAME TO candidate_claims;
            """)
        if _has_branch_key("candidate_republish"):
            self._conn.executescript("""
                CREATE TABLE candidate_republish_new (
                    branch_id INTEGER NOT NULL,
                    idx       INTEGER NOT NULL,
                    count     INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (branch_id, idx)
                );
                INSERT INTO candidate_republish_new
                    SELECT b.branch_id, c.idx, c.count
                    FROM candidate_republish c
                    JOIN branches b ON b.branch_key = c.branch_key;
                DROP TABLE candidate_republish;
                ALTER TABLE candidate_republish_new RENAME TO candidate_republish;
            """)
        if _has_branch_key("cut_results"):
            self._conn.executescript("""
                CREATE TABLE cut_results_new (
                    branch_id  INTEGER PRIMARY KEY,
                    budget     INTEGER NOT NULL,
                    bound      REAL    NOT NULL,
                    created_at INTEGER NOT NULL
                );
                INSERT INTO cut_results_new
                    SELECT b.branch_id, c.budget, c.bound, c.created_at
                    FROM cut_results c
                    JOIN branches b ON b.branch_key = c.branch_key;
                DROP TABLE cut_results;
                ALTER TABLE cut_results_new RENAME TO cut_results;
            """)
        if _rebuildable("active_branches", {
                "branch_key", "n_words", "n_candidates", "priority",
                "opener", "opener_pattern", "best_erd", "best_guess",
                "status", "created_at", "finalized_at", "budget",
                "best_max_depth", "tainted", "nodes_spent", "spine", "ceiling",
                "cut_occurred", "bulk_done_candidates", "bulk_done_bound",
                "pack_cursor"}):
            self._conn.executescript("""
                CREATE TABLE active_branches_norm (
                    branch_id      INTEGER PRIMARY KEY,
                    n_words        INTEGER NOT NULL,
                    n_candidates   INTEGER NOT NULL,
                    priority       INTEGER NOT NULL DEFAULT 0,
                    opener    TEXT,
                    opener_pattern INTEGER,
                    best_erd       REAL,
                    best_guess     TEXT,
                    status         TEXT    NOT NULL DEFAULT 'open',
                    created_at     INTEGER,
                    finalized_at   INTEGER,
                    budget         INTEGER,
                    best_max_depth INTEGER,
                    tainted        INTEGER NOT NULL DEFAULT 0,
                    nodes_spent    INTEGER NOT NULL DEFAULT 0,
                    spine          TEXT,
                    ceiling        REAL,
                    cut_occurred   INTEGER NOT NULL DEFAULT 0,
                    bulk_done_candidates INTEGER NOT NULL DEFAULT 0,
                    one_level_erd_pruned_candidates INTEGER NOT NULL DEFAULT 0,
                    two_level_erd_pruned_candidates INTEGER NOT NULL DEFAULT 0,
                    bulk_done_bound REAL,
                    best_updated_at INTEGER,
                    pack_cursor    INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO active_branches_norm
                    SELECT b.branch_id, a.n_words, a.n_candidates, a.priority,
                           a.opener, a.opener_pattern, a.best_erd,
                           a.best_guess, a.status, a.created_at, a.finalized_at,
                           a.budget, a.best_max_depth, a.tainted, a.nodes_spent,
                           a.spine, a.ceiling, a.cut_occurred,
                           a.bulk_done_candidates,
                           a.one_level_erd_pruned_candidates,
                           a.two_level_erd_pruned_candidates,
                           a.bulk_done_bound,
                           a.best_updated_at,
                           a.pack_cursor
                    FROM active_branches a
                    JOIN branches b ON b.branch_key = a.branch_key;
                DROP TABLE active_branches;
                ALTER TABLE active_branches_norm RENAME TO active_branches;
                CREATE INDEX IF NOT EXISTS idx_active_branches_status_pri
                    ON active_branches(status, priority DESC, n_words DESC);
            """)
        if _rebuildable("pending_branches", {
                "branch_key", "n_words", "priority", "opener",
                "opener_pattern", "status", "claimed_by", "claimed_at",
                "completed_at"}):
            self._conn.executescript("""
                CREATE TABLE pending_branches_norm (
                    branch_id      INTEGER PRIMARY KEY,
                    n_words        INTEGER NOT NULL,
                    priority       INTEGER NOT NULL DEFAULT 0,
                    opener    TEXT,
                    opener_pattern INTEGER,
                    status         TEXT    NOT NULL DEFAULT 'pending',
                    claimed_by     TEXT,
                    claimed_at     INTEGER,
                    completed_at   INTEGER
                );
                INSERT INTO pending_branches_norm
                    SELECT b.branch_id, p.n_words, p.priority, p.opener,
                           p.opener_pattern, p.status, p.claimed_by,
                           p.claimed_at, p.completed_at
                    FROM pending_branches p
                    JOIN branches b ON b.branch_key = p.branch_key;
                DROP TABLE pending_branches;
                ALTER TABLE pending_branches_norm RENAME TO pending_branches;
                CREATE INDEX IF NOT EXISTS idx_pending_status_pri_n
                    ON pending_branches(status, priority DESC, n_words DESC);
            """)
        if "current_branch_key" in hb_cols:
            # worker_heartbeat is transient liveness state, overwritten every
            # couple of seconds and empty between deploys; migrations run with
            # workers stopped, so its rows are stale.  Clear them and rename the
            # branch reference in place rather than rebuild 20+ metric columns.
            self._conn.execute("DELETE FROM worker_heartbeat")
            self._conn.execute(
                "ALTER TABLE worker_heartbeat "
                "RENAME COLUMN current_branch_key TO current_branch_id")

        # Pre-opener-aware queues recorded only a display label on each root.
        # Preserve each surviving root as independently schedulable opener work;
        # its original request grouping cannot be recovered from those labels.
        pending_columns = {row["name"] for row in self._conn.execute(
            "PRAGMA table_info(pending_branches)")}
        if (self._conn.execute("SELECT COUNT(*) FROM opener_work").fetchone()[0] == 0
                and {"branch_id", "opener", "priority", "status"}
                <= pending_columns):
            legacy_roots = self._conn.execute("""
                SELECT branch_id, opener, priority, status
                FROM pending_branches
                ORDER BY branch_id
            """).fetchall()
            now = int(time.time())
            for row in legacy_roots:
                state = ({"done": "complete", "in_progress": "active"}
                         .get(row["status"], "queued"))
                cur = self._conn.execute("""
                    INSERT INTO opener_work
                        (opener, requested_priority, requested_at, state)
                    VALUES (?, ?, ?, ?)
                """, (row["opener"], row["priority"], now, state))
                self._conn.execute("""
                    INSERT OR IGNORE INTO branch_opener_work
                        (branch_id, opener_work_id, parent_branch_id)
                    VALUES (?, ?, NULL)
                """, (row["branch_id"], cur.lastrowid))

        self._add_columns("branch_opener_work", {
            "opener_pattern": "INTEGER",
            "resolved_at": "INTEGER",
        })
        self._add_columns("opener_work", {"started_at": "INTEGER"})
        if "claimed_at" in pending_columns:
            self._conn.execute("""
                UPDATE opener_work AS opener
                SET started_at = (
                    SELECT MIN(pending.claimed_at)
                    FROM branch_opener_work AS membership
                    JOIN pending_branches AS pending
                      ON pending.branch_id = membership.branch_id
                    WHERE membership.opener_work_id = opener.opener_work_id
                      AND membership.resolved_at IS NULL
                      AND pending.claimed_at IS NOT NULL
                )
                WHERE opener.state = 'active' AND opener.started_at IS NULL
            """)
        self._add_columns("active_branches", {
            "requires_opener_membership": "INTEGER NOT NULL DEFAULT 0",
            "infeasible_candidates": "INTEGER NOT NULL DEFAULT 0",
            "infeasible_nodes": "INTEGER NOT NULL DEFAULT 0",
        })
        # worker_heartbeat is transient liveness state (see its table comment): no
        # backfill for existing rows is needed, they are overwritten on the next
        # heartbeat.
        self._add_columns("worker_heartbeat", {
            "opener_work_id": "INTEGER",
            "scheduling_role": "TEXT",
        })
        self._conn.execute("""
            UPDATE active_branches
            SET requires_opener_membership = 1
            WHERE branch_id IN (
                SELECT branch_id FROM branch_opener_work
                WHERE resolved_at IS NULL
            )
        """)
        if {"branch_id", "opener_pattern"} <= pending_columns:
            self._conn.execute("""
                UPDATE branch_opener_work
                SET opener_pattern = (
                    SELECT opener_pattern FROM pending_branches
                    WHERE pending_branches.branch_id = branch_opener_work.branch_id
                )
                WHERE parent_branch_id IS NULL AND opener_pattern IS NULL
            """)
        self._conn.execute("""
            UPDATE branch_opener_work AS membership
            SET resolved_at = ?
            WHERE resolved_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM opener_work AS opener
                  WHERE opener.opener_work_id = membership.opener_work_id
                    AND opener.state = 'complete'
              )
        """, (int(time.time()),))
        self._apply_queue_migration(
            "neutralize_legacy_promoted_priorities",
            (
                "UPDATE active_branches SET priority = 0 "
                "WHERE status = 'open' AND priority >= ?",
                "UPDATE pending_branches SET priority = 0 "
                "WHERE status != 'done' AND priority >= ?",
            ),
            (LEGACY_PROMOTED_PRIORITY_MIN,),
        )
        self._rebuild_queue_views()

        # Claims bundle index on the (post-normalization) branch_id key.  After
        # the ADD COLUMN and rebuild above, so bundle_id and branch_id both
        # exist whether the database is fresh, upgraded, or already migrated.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_claims_bundle "
            "ON candidate_claims(branch_id, bundle_id)")

        # Branch occupancy (_other_claim_holders) runs inside every claim
        # transaction and asks only which workers hold UNFINISHED claims.  The
        # primary key seeks by branch_id but then walks every claim row the
        # branch has ever had — 3-4 ms on a branch with 8,475 of them, against
        # a whole claim costing well under 1 ms.  Partial on done = 0 so the
        # index holds only work in flight, a handful of rows per branch.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidate_claims_open_by_worker "
            "ON candidate_claims(branch_id, claimed_by) WHERE done = 0")

        # candidate_claims.best_first_position: where the claimed candidate
        # sits in the branch's best-first order,
        # which candidate_holes inherits when the row is freed.  After the
        # branch_key rebuild above, whose CREATE TABLE lists its columns
        # explicitly.  Legacy rows keep NULL: their position was never
        # recorded, so a hole made from one is packed after every hole whose
        # position is known.
        self._add_columns("candidate_claims",
                          {"best_first_position": "INTEGER"})

        # cut_results.tainted: after the branch_key rebuild above so the column
        # lands whether the table is fresh, upgraded, or already migrated.
        # Legacy rows default to 0: cut_results is transient (cleared on
        # supervisor restart), so no backfill question arises.
        self._add_columns("cut_results", {
            "tainted": "INTEGER NOT NULL DEFAULT 0"})

        # cut_results re-key (issue #185): branch_id alone -> (branch_id,
        # budget, tainted).  The old single-row-per-branch shape made every
        # add_cut_result an overwrite, discarding a bound already proven
        # whenever a later write for a different budget or taint class
        # landed on the same row.  Detected by budget/tainted not being part
        # of the primary key yet (pk=0); after the branch_key rebuild above
        # and the tainted add above, so this runs whether the database is
        # fresh, upgraded once, or upgraded from both legacy shapes at once.
        cut_results_pk = {r["name"] for r in
                          self._conn.execute("PRAGMA table_info(cut_results)")
                          if r["pk"] > 0}
        if cut_results_pk == {"branch_id"}:
            self._conn.executescript("""
                CREATE TABLE cut_results_rekeyed (
                    branch_id  INTEGER NOT NULL,
                    budget     INTEGER NOT NULL,
                    bound      REAL    NOT NULL,
                    tainted    INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (branch_id, budget, tainted)
                );
                INSERT INTO cut_results_rekeyed
                    SELECT branch_id, budget, bound, tainted, created_at
                    FROM cut_results;
                DROP TABLE cut_results;
                ALTER TABLE cut_results_rekeyed RENAME TO cut_results;
            """)

        if normalized_tables or cut_results_pk == {"branch_id"}:
            # DROP TABLE hands the fat-blob pages to the free-list, not the OS,
            # so the file stays bloated until compacted.  This one-time upgrade
            # runs with workers stopped, so reclaim the space now.  Best-effort:
            # a VACUUM needs scratch space up to the file size, and a correct
            # (if un-compacted) database is fine if that is unavailable.
            try:
                self._conn.execute("VACUUM")
            except sqlite3.OperationalError as exc:  # pragma: no cover
                logger.warning(
                    "post-normalization VACUUM skipped (%s); database is "
                    "correct but not yet compacted", exc)

        # worker_heartbeat.cur_help_depth: nesting depth of _help_other_branch
        # calls on this worker's stack (see erd_swarm.MAX_HELP_RECURSION_DEPTH),
        # so `view worker` can show when a worker is deep in a rescue chain
        # instead of at its normal claim boundary.  Additive on transient
        # liveness state — see the current_branch_key rename above.
        self._add_columns("worker_heartbeat", {
            "cur_help_depth": "INTEGER NOT NULL DEFAULT 0",
        })

        # Hint-artifact accounting (issue #304): additive/nullable everywhere.
        # NULL means "this row predates hints, or the run had no hint
        # artifact" — never a measured zero.  Last in _migrate because the
        # table rebuilds above copy a fixed column list: a column added before
        # one of them would be dropped by it.
        self._add_columns("worker_heartbeat", {
            "hint_lookups": "INTEGER",
            "hint_hits": "INTEGER",
            "hint_accepted": "INTEGER",
            "hint_rejected": "INTEGER",
            "hint_inline_placements": "INTEGER",
            "hint_inline_wins": "INTEGER",
        })
        self._add_columns("active_branches", {
            "first_best_at": "INTEGER",
            "nodes_at_first_best": "INTEGER",
        })
        self._add_columns("branch_finalize_log", {
            "hint_word": "TEXT",
            "hint_was_winner": "INTEGER",
            "first_best_at": "INTEGER",
            "nodes_at_first_best": "INTEGER",
        }, schema="telemetry")

        # Baseline epoch 0 and the run_meta pointer, both idempotent.  git_sha is
        # stamped later (set_epoch) when a deploy knows it.
        now = int(time.time())
        self._conn.execute(
            "INSERT OR IGNORE INTO telemetry_epoch (epoch, label, started_at) "
            "VALUES (0, 'single-candidate atom baseline', ?)", (now,))
        self._conn.execute(
            "INSERT OR IGNORE INTO run_meta (key, value) VALUES ('epoch', '0')")

    def _apply_queue_migration(self, name: str, statements: tuple[str, ...],
                               parameters: tuple = ()) -> None:
        """Apply one named queue-data migration exactly once."""
        self._conn.execute("SAVEPOINT queue_data_migration")
        try:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (name, completed_at) "
                "VALUES (?, ?)", (name, int(time.time())))
            if cur.rowcount:
                for statement in statements:
                    self._conn.execute(statement, parameters)
            self._conn.execute("RELEASE queue_data_migration")
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK TO queue_data_migration")
            self._conn.execute("RELEASE queue_data_migration")
            raise

    def _rebuild_queue_views(self):
        """Restore queue views when their stored definition differs from code."""
        view_definitions = {
            "live_branch_opener_rows": """
                CREATE VIEW live_branch_opener_rows AS
                SELECT membership.branch_id,
                       opener.opener_work_id,
                       opener.opener AS owner_opener,
                       membership.opener_pattern AS owner_opener_pattern,
                       opener.requested_priority AS owner_priority
                FROM branch_opener_work AS membership
                JOIN opener_work AS opener
                  ON opener.opener_work_id = membership.opener_work_id
                WHERE membership.resolved_at IS NULL
                  AND opener.state != 'complete'
            """,
            "active_branch_owner_rows": f"""
                CREATE VIEW active_branch_owner_rows AS
                SELECT active.*,
                       branch.branch_key,
                       owner.opener_work_id,
                       COALESCE(owner.owner_opener, active.opener)
                           AS owner_opener,
                       COALESCE(owner.owner_opener_pattern, active.opener_pattern)
                           AS owner_opener_pattern,
                       COALESCE(
                           owner.owner_priority,
                           CASE WHEN active.priority >= {LEGACY_PROMOTED_PRIORITY_MIN}
                                THEN 0 ELSE active.priority END
                       )
                           AS owner_priority
                FROM active_branches AS active
                JOIN branches AS branch USING (branch_id)
                LEFT JOIN live_branch_opener_rows AS owner USING (branch_id)
                WHERE active.status = 'open'
                  AND (owner.opener_work_id IS NOT NULL
                       OR active.requires_opener_membership = 0)
            """,
        }
        stored_definitions = {
            name: self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = ?",
                (name,)).fetchone()
            for name in view_definitions
        }
        changed = any(
            stored_definitions[name] is None or " ".join(
                stored_definitions[name]["sql"].split()) != " ".join(
                    definition.split())
            for name, definition in view_definitions.items()
        )
        if not changed:
            return
        self._conn.execute("SAVEPOINT rebuild_queue_views")
        try:
            for name in view_definitions:
                self._conn.execute(f"DROP VIEW IF EXISTS {name}")
            for definition in view_definitions.values():
                self._conn.execute(definition)
            self._conn.execute("RELEASE rebuild_queue_views")
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK TO rebuild_queue_views")
            self._conn.execute("RELEASE rebuild_queue_views")
            raise

    def _assert_schema(self):
        """Refuse to run against a database whose migrated schema disagrees
        with the schema SQL, checked per file (queue and telemetry).

        Invariant: after _migrate(), every table must be column-identical to
        what the schema SQL creates on a fresh database.  A table that
        predates the current code makes CREATE TABLE IF NOT EXISTS a no-op,
        so a column _migrate() has no rule for stays wrong silently and
        fails later and less clearly (e.g. mid-finalize, killing the
        worker).  Tests verify the code against databases the current code
        creates; this verifies the actual databases, so drift from any
        origin — including a database touched by code from an unmerged
        commit — refuses at open instead.

        Missing columns raise; extra columns only warn, since they cannot
        break the code's own reads and writes (every statement names its
        columns) but are still drift worth surfacing.
        """
        expected_conn = sqlite3.connect(":memory:")
        try:
            expected_conn.execute("ATTACH DATABASE ':memory:' AS telemetry")
            expected_conn.executescript(_QUEUE_SCHEMA_SQL)
            expected_conn.executescript(_TELEMETRY_SCHEMA_SQL)
            for schema in ("main", "telemetry"):
                tables = [name for (name,) in expected_conn.execute(
                    f"SELECT name FROM {schema}.sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
                for table in tables:
                    expected = {r[1] for r in expected_conn.execute(
                        f"PRAGMA {schema}.table_info({table})")}
                    actual = {r["name"] for r in self._conn.execute(
                        f"PRAGMA {schema}.table_info({table})")}
                    missing = expected - actual
                    if missing:
                        raise RuntimeError(
                            f"queue database schema mismatch: table "
                            f"{schema}.{table} is missing column(s) "
                            f"{sorted(missing)} after migration. The table "
                            f"was likely created by code from a different "
                            f"commit; add a rule to ERDQueue._migrate() for "
                            f"its shape instead of opening it with "
                            f"mismatched code.")
                    extra = actual - expected
                    if extra:
                        logger.warning(
                            "queue database table %s.%s has unexpected "
                            "column(s) %s — schema drift, harmless to "
                            "current statements",
                            schema, table, sorted(extra))
        finally:
            expected_conn.close()

    def _add_columns(self, table: str, columns: dict, schema: str = "main"):
        """Idempotently ALTER TABLE ADD COLUMN for any of `columns` not present.

        columns maps name -> SQL type/constraint.  A NOT NULL column must carry a
        DEFAULT so the add is safe on a populated table.  schema selects the
        database file ('main' = queue, 'telemetry' = the attached telemetry file).
        """
        have = {r["name"] for r in
                self._conn.execute(f"PRAGMA {schema}.table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                self._conn.execute(
                    f"ALTER TABLE {schema}.{table} ADD COLUMN {name} {decl}")

    def close(self):
        self._conn.close()

    def checkpoint(self, mode: str = "TRUNCATE"):
        """Fold the WAL into the main database file (PRAGMA wal_checkpoint).

        Returns the pragma's (busy, log_frames, checkpointed_frames) row, or
        None if the pragma itself errored.  busy=1 means the checkpoint could
        not complete — SQLite reports contention through this row, NOT as an
        exception, so callers that care must inspect the result.

        mode PASSIVE backfills whatever it can without taking the writer lock
        or waiting on readers: it can never stall other connections, but never
        truncates the file either.  mode TRUNCATE additionally takes the
        writer lock and waits (up to the connection busy timeout) for all
        readers to drain past the WAL before truncating it to
        journal_size_limit; under sustained concurrent traffic it loses that
        wait, so it should only be attempted while writers are quiesced (see
        set_checkpoint_pause).
        """
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            raise ValueError(f"unknown checkpoint mode {mode!r}")
        try:
            row = self._conn.execute(
                f"PRAGMA wal_checkpoint({mode})").fetchone()
        except sqlite3.OperationalError as exc:
            logger.warning("wal_checkpoint(%s) failed: %s", mode, exc)
            return None
        busy, log_frames, checkpointed = row[0], row[1], row[2]
        if busy:
            logger.warning(
                "wal_checkpoint(%s) incomplete: busy=1 log=%s checkpointed=%s "
                "wal_bytes=%s", mode, f"{log_frames:,}", f"{checkpointed:,}",
                f"{self.wal_size_bytes():,}")
        return busy, log_frames, checkpointed

    def wal_size_bytes(self) -> int:
        try:
            return os.path.getsize(f"{self.db_path}-wal")
        except OSError:
            return 0

    # ------------------------------------------------------------------
    # Populate queue
    # ------------------------------------------------------------------

    def _intern_branch(self, branch_key, create=False):
        """Map a branch_key BLOB to its integer branch_id in the branches
        registry — the surrogate key the high-volume claim/cut tables reference
        instead of the fat blob.

        create=True registers the branch if absent (write paths that must be
        able to reference it); create=False returns None for an unregistered
        branch (read/delete paths, where absence means the branch simply has no
        rows to read or delete, and must NOT write a registry entry).  Positive
        lookups are memoized per connection.
        """
        cached = self._branch_id_cache.get(branch_key)
        if cached is not None:
            return cached
        if create:
            self._conn.execute(
                "INSERT OR IGNORE INTO branches (branch_key) VALUES (?)",
                (branch_key,))
        row = self._conn.execute(
            "SELECT branch_id FROM branches WHERE branch_key = ?",
            (branch_key,)).fetchone()
        if row is None:
            return None
        self._branch_id_cache[branch_key] = row[0]
        return row[0]

    def add_pending_many(self, rows):
        """Insert (branch_key, n_words, priority, opener, opener_pattern) rows.

        Uses an UPSERT so that:
        - A row inserted for the first time is added as 'pending'.
        - A row already present has its priority UPGRADED (never downgraded),
          e.g. a branch first inserted at priority=0 by an earlier opener
          is correctly promoted to priority=1 when a VIP word (SALET) is
          queued later.
        - opener / opener_pattern record the first opener whose branch
          produced this entry (kept for display in `status`).

        Raises ValueError, before writing anything, if any row's priority lies
        outside the opener-work range.
        """
        rows = list(rows)
        for row in rows:
            check_opener_priority_range(row[2])
        # Intern before the transaction: the branches registry is append-only,
        # so committing ids up front is safe even if the pending insert fails,
        # and keeps the id cache consistent with the database on a rollback.
        prepared = [(self._intern_branch(r[0], create=True), r[1], r[2], r[3],
                     r[4]) for r in rows]
        self._conn.execute("BEGIN")
        try:
            now = int(time.time())
            opener_work_ids = {}
            for _branch_id, _n_words, priority, opener, _opener_pattern in prepared:
                key = (opener, priority)
                if key not in opener_work_ids:
                    cur = self._conn.execute("""
                        INSERT INTO opener_work
                            (opener, requested_priority, requested_at, state)
                        VALUES (?, ?, ?, 'queued')
                    """, (opener, priority, now))
                    opener_work_ids[key] = cur.lastrowid
            self._conn.executemany("""
                INSERT INTO pending_branches
                    (branch_id, n_words, priority, opener, opener_pattern, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(branch_id) DO UPDATE SET
                    priority       = MAX(priority, excluded.priority),
                    opener    = COALESCE(opener, excluded.opener),
                    opener_pattern = COALESCE(opener_pattern, excluded.opener_pattern)
            """, prepared)
            self._conn.executemany("""
                INSERT INTO branch_opener_work
                    (branch_id, opener_work_id, parent_branch_id, opener_pattern)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(branch_id, opener_work_id) DO UPDATE SET
                    parent_branch_id = NULL,
                    opener_pattern = excluded.opener_pattern,
                    resolved_at = NULL
            """, [(branch_id, opener_work_ids[(opener, priority)], opener_pattern)
                  for branch_id, _n_words, priority, opener, opener_pattern
                  in prepared])
            self._conn.execute("COMMIT")
            self._tally_wal_traffic(
                'pending_branches/add', len(prepared),
                len(prepared) * _CLAIM_ROW_WAL_BYTES)
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Worker claim loop
    # ------------------------------------------------------------------

    def opener_work_candidates(self):
        """Return unfinished opener work in opener-first admission order."""
        return self._conn.execute("""
            SELECT s.* FROM opener_work s
            WHERE s.state != 'complete'
              AND EXISTS (
                SELECT 1 FROM branch_opener_work m
                LEFT JOIN pending_branches p ON p.branch_id = m.branch_id
                LEFT JOIN active_branches a ON a.branch_id = m.branch_id
                WHERE m.opener_work_id = s.opener_work_id
                  AND m.resolved_at IS NULL
                  AND (p.status IN ('pending', 'in_progress') OR a.status = 'open')
            )
            ORDER BY s.requested_priority DESC,
                     CASE s.state WHEN 'active' THEN 0 ELSE 1 END,
                     s.opener_work_id
        """).fetchall()

    def has_opener_work(self) -> bool:
        """Whether this queue has opener-aware scheduling provenance."""
        return self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM opener_work)").fetchone()[0] == 1

    def lowest_unfinished_opener_priority(self):
        """The lowest requested priority still owed work, or None if none is.

        A request that has not completed is still owed work whether or not it
        has started, so a caller adding work behind everything already queued
        ranks it below this.
        """
        return self._conn.execute("""
            SELECT MIN(requested_priority) FROM opener_work
            WHERE state != 'complete'
        """).fetchone()[0]

    def max_unfinished_opener_priority(self):
        """The highest requested priority still owed work, or None if none is.

        Paired with lowest_unfinished_opener_priority, this bounds the span
        unfinished opener-work already occupies.  An append whose ladder would
        clamp onto OPENER_PRIORITY_MIN reads this to see how much headroom sits
        above that span before shifting it upward with
        shift_unfinished_opener_priorities, rather than ratcheting its own
        floor below the minimum.
        """
        return self._conn.execute("""
            SELECT MAX(requested_priority) FROM opener_work
            WHERE state != 'complete'
        """).fetchone()[0]

    def shift_unfinished_opener_priorities(self, amount: int) -> int:
        """Raise every unfinished opener-work request's priority by `amount`.

        Reclaims room below the incumbent for an append that would otherwise
        clamp onto OPENER_PRIORITY_MIN: every request the queue still owes
        work on moves up together, so their relative order and every tie is
        preserved, and a batch appending beneath them still ranks below all of
        them.  Mirrors set_opener_work_priority's propagation to owned
        pending/active branches, applied to every unfinished request in one
        transaction instead of one at a time.

        Raises ValueError, touching nothing, if the shift would push the
        highest unfinished priority past OPENER_PRIORITY_MAX.  Returns the
        number of opener-work requests shifted.
        """
        if amount <= 0:
            raise ValueError('shift amount must be positive')
        current_max = self.max_unfinished_opener_priority()
        if current_max is not None and current_max + amount > OPENER_PRIORITY_MAX:
            raise ValueError(
                f'shifting unfinished opener-work up by {amount:,} would '
                f'push priority {current_max:,} past the maximum '
                f'{OPENER_PRIORITY_MAX:,}')
        self._conn.execute("BEGIN")
        try:
            cur = self._conn.execute("""
                UPDATE opener_work SET requested_priority = requested_priority + ?
                WHERE state != 'complete'
            """, (amount,))
            if cur.rowcount:
                self._conn.execute("""
                    UPDATE pending_branches
                    SET priority = (
                        SELECT MAX(owner_priority)
                        FROM live_branch_opener_rows
                        WHERE branch_id = pending_branches.branch_id
                    )
                    WHERE branch_id IN (SELECT branch_id FROM live_branch_opener_rows)
                """)
                self._conn.execute("""
                    UPDATE active_branches
                    SET priority = (
                        SELECT MAX(owner_priority)
                        FROM live_branch_opener_rows
                        WHERE branch_id = active_branches.branch_id
                    )
                    WHERE branch_id IN (SELECT branch_id FROM live_branch_opener_rows)
                """)
            self._conn.execute("COMMIT")
            return cur.rowcount
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def _opener_response_group_is_live(self, opener_work_id, opener_pattern):
        """Whether an opener request still needs work below one direct response group."""
        if opener_pattern is None:
            return self._conn.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM opener_work
                    WHERE opener_work_id = ? AND state != 'complete'
                )
            """, (opener_work_id,)).fetchone()[0] == 1
        return self._conn.execute("""
            SELECT EXISTS(
                SELECT 1
                FROM branch_opener_work AS membership
                JOIN opener_work AS opener
                  ON opener.opener_work_id = membership.opener_work_id
                WHERE membership.opener_work_id = ?
                  AND membership.opener_pattern = ?
                  AND membership.parent_branch_id IS NULL
                  AND membership.resolved_at IS NULL
                  AND opener.state != 'complete'
            )
        """, (opener_work_id, opener_pattern)).fetchone()[0] == 1

    def claim_next(self, worker_id: str, opener_work_id: int = None):
        """Atomically claim the highest-priority / largest pending branch.

        Returns a dict {branch_key, n_words, priority, opener,
        opener_pattern} or None if the queue is empty.  In the swarm model the
        claiming worker uses this to PROMOTE a queued branch into an
        active_branches row that other workers can then join; the branch's
        priority and opener word/pattern are carried over for display.

        Uses BEGIN IMMEDIATE to acquire the write lock before the SELECT,
        eliminating the TOCTOU race where two workers could both read the same
        'pending' row before either marks it 'in_progress'.  Under contention
        the loser blocks and retries automatically via sqlite3_busy_timeout.
        """
        self._begin_immediate_timed()
        try:
            if opener_work_id is None:
                opener_work_rows = self.opener_work_candidates()
                opener_work_id = (opener_work_rows[0]["opener_work_id"]
                                  if opener_work_rows else None)
            if opener_work_id is None:
                self._conn.execute("COMMIT")
                return None
            row = self._conn.execute("""
                SELECT branch.branch_key, pending.branch_id, pending.n_words,
                       owner.owner_priority, owner.owner_opener,
                       owner.owner_opener_pattern, owner.opener_work_id
                FROM pending_branches AS pending
                JOIN branches AS branch USING (branch_id)
                JOIN live_branch_opener_rows AS owner USING (branch_id)
                WHERE pending.status = 'pending'
                  AND owner.opener_work_id = ?
                ORDER BY pending.n_words DESC
                LIMIT 1
            """, (opener_work_id,)).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return None
            now = int(time.time())
            self._conn.execute("""
                UPDATE pending_branches
                SET status = 'in_progress', claimed_by = ?, claimed_at = ?
                WHERE branch_id = ?
            """, (worker_id, now, row["branch_id"]))
            self._conn.execute("""
                UPDATE opener_work
                SET state = 'active', started_at = COALESCE(started_at, ?)
                WHERE opener_work_id = ?
            """, (now, opener_work_id))
            self._conn.execute("COMMIT")
            return {
                'branch_key': bytes(row["branch_key"]),
                'n_words': row["n_words"],
                'priority': row["owner_priority"],
                'opener': row["owner_opener"],
                'opener_pattern': row["owner_opener_pattern"],
                'opener_work_id': row["opener_work_id"],
            }
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def mark_done(self, branch_key: bytes):
        now = int(time.time())
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("""
                UPDATE pending_branches
                SET status = 'done', completed_at = ?
                WHERE branch_id = ?
            """, (now, branch_id))
            active = self._conn.execute(
                "SELECT 1 FROM active_branches WHERE branch_id = ?",
                (branch_id,)).fetchone()
            completed_words = self._retire_exact_direct_response_groups(branch_id)
            if active is None:
                completed_words.extend(
                    self._resolve_branch_memberships(branch_id))
            self._conn.execute("COMMIT")
            return list(dict.fromkeys(completed_words))
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def _complete_finished_opener_work(self):
        """Mark opener requests terminal once every owned branch is complete."""
        completion_predicate = """
            state != 'complete'
              AND NOT EXISTS (
                  SELECT 1 FROM branch_opener_work m
                  LEFT JOIN pending_branches p ON p.branch_id = m.branch_id
                  LEFT JOIN active_branches a ON a.branch_id = m.branch_id
                  WHERE m.opener_work_id = s.opener_work_id
                    AND m.resolved_at IS NULL
                    AND (p.status IN ('pending', 'in_progress') OR a.status = 'open')
              )
        """
        completed_rows = self._conn.execute(
            "SELECT opener FROM opener_work AS s WHERE "
            + completion_predicate
        ).fetchall()
        if not completed_rows:
            return []
        self._conn.execute(
            "UPDATE opener_work AS s SET state = 'complete' WHERE "
            + completion_predicate
        )
        return list({row["opener"] for row in completed_rows})

    def _resolve_branch_memberships(self, branch_id: int = None,
                                    withdraw: bool = False):
        """Make branch ownership unschedulable and update request lifecycle."""
        branch_condition = "" if branch_id is None else " AND branch_id = ?"
        parameters = () if branch_id is None else (branch_id,)
        if withdraw:
            if branch_id is None:
                self._conn.execute("DELETE FROM branch_opener_work")
            else:
                self._conn.execute(
                    "DELETE FROM branch_opener_work "
                    "WHERE resolved_at IS NULL AND branch_id = ?", parameters)
        else:
            self._conn.execute(
                "UPDATE branch_opener_work SET resolved_at = ? "
                "WHERE resolved_at IS NULL" + branch_condition,
                (int(time.time()), *parameters))
        completed_words = self._complete_finished_opener_work()
        self._demote_orphaned_owned_branches()
        return completed_words

    def _retire_exact_direct_response_groups(self, branch_id: int) -> list[str]:
        """Retire work below direct response groups whose exact result is done.

        A opener request needs each direct response group once.  Once one of
        those roots has an exact result, descendant candidate searches under
        that same root cannot change it.  Other direct response groups, and
        branches still owned by another live request, remain runnable.

        The caller holds the queue transaction.  Removing exclusive active
        rows before resolving their opener memberships prevents the normal
        orphan reconciliation from reclassifying cancelled descendant work as
        direct work.
        """
        direct_groups = self._conn.execute("""
            SELECT opener_work_id, opener_pattern
            FROM branch_opener_work
            WHERE branch_id = ?
              AND parent_branch_id IS NULL
              AND resolved_at IS NULL
        """, (branch_id,)).fetchall()
        if not direct_groups:
            return []

        target_branch_ids = set()
        for group in direct_groups:
            target_branch_ids.update(row[0] for row in self._conn.execute("""
                SELECT branch_id
                FROM branch_opener_work
                WHERE opener_work_id = ?
                  AND opener_pattern IS ?
                  AND resolved_at IS NULL
            """, (group["opener_work_id"], group["opener_pattern"])))

        now = int(time.time())
        for group in direct_groups:
            self._conn.execute("""
                UPDATE branch_opener_work
                SET resolved_at = ?
                WHERE opener_work_id = ?
                  AND opener_pattern IS ?
                  AND resolved_at IS NULL
            """, (now, group["opener_work_id"], group["opener_pattern"]))

        descendant_ids = sorted(target_branch_ids - {branch_id})
        if descendant_ids:
            placeholders = ",".join("?" for _ in descendant_ids)
            exclusive_active = self._conn.execute(f"""
                SELECT active.branch_id, branch.branch_key
                FROM active_branches AS active
                JOIN branches AS branch USING (branch_id)
                WHERE active.status = 'open'
                  AND active.branch_id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM live_branch_opener_rows AS owner
                      WHERE owner.branch_id = active.branch_id
                  )
            """, descendant_ids).fetchall()
            active_ids = [row["branch_id"] for row in exclusive_active]
            if active_ids:
                active_placeholders = ",".join("?" for _ in active_ids)
                self._conn.execute(
                    f"DELETE FROM candidate_claims WHERE branch_id IN ({active_placeholders})",
                    active_ids)
                self._conn.execute(
                    f"DELETE FROM candidate_republish WHERE branch_id IN ({active_placeholders})",
                    active_ids)
                self._conn.execute(
                    f"DELETE FROM candidate_holes WHERE branch_id IN ({active_placeholders})",
                    active_ids)
                self._conn.executemany(
                    "DELETE FROM telemetry.bundle_stats WHERE branch_key = ?",
                    [(row["branch_key"],) for row in exclusive_active])
                self._conn.execute(
                    f"DELETE FROM active_branches WHERE branch_id IN ({active_placeholders})",
                    active_ids)

        return self._complete_finished_opener_work()

    def completed_opener_timing(self, opener):
        """Return durable-opener timing from every telemetry epoch.

        Opener ownership can attach a request to an already-existing branch.
        Its finalization record predates that request, so ownership joins are
        not an opener timing boundary.  The opener in each recorded spine is.
        """
        return self._conn.execute("""
            SELECT MIN(log.created_at) AS first_created_at,
                   MAX(log.finalized_at) AS completed_at,
                   SUM(COALESCE(log.total_bundle_wall_millis, 0)) AS worker_millis,
                   GROUP_CONCAT(DISTINCT log.epoch) AS telemetry_epochs
            FROM telemetry.branch_finalize_log AS log
            WHERE lower(substr(log.spine, 1, 5)) = lower(?)
              AND log.created_at IS NOT NULL
              AND log.finalized_at IS NOT NULL
        """, (opener,)).fetchone()

    def _demote_orphaned_owned_branches(self) -> list[int]:
        """Demote open branches whose only opener ownership has been lost.

        A branch promoted under an opener-work request (requires_opener_
        membership = 1) can outlive every membership that justified the
        requirement, in either of two shapes:

        - claim_next_bundle's bulk elimination retroactively retracts the
          in-flight candidate that promoted it once the shared best_erd
          bound tightens past it, and the branch's own membership resolves
          along with the rest of the (now-finished) request while the
          branch itself stays open;
        - create_branch attaches an opener_work_id without checking the
          request is still live (unlike attach_branch_opener_work, which
          gates on opener.state != 'complete'), so a worker's in-flight
          create_branch call can land an unresolved membership against a
          request that finished moments earlier.

        Neither is an error -- both are routine consequences of alpha-beta
        pruning racing a shared bound -- so the branch is never cancelled
        here (its answer set may still be reachable from another spine).
        Instead it is demoted to requires_opener_membership = 0, the state a
        directly dispatched branch already has, so active_branch_owner_rows
        admits it without a live membership and any worker can pick it up
        again.

        The query mirrors active_branch_owner_rows' own visibility test
        (via live_branch_opener_rows) exactly, rather than only the
        resolved_at half of it, so both shapes above are caught. Any
        leftover unresolved membership rows for a demoted branch are also
        resolved here: claim_next_bundle's direct-claim guard requires both
        requires_opener_membership = 0 and the absence of any resolved_at
        IS NULL row, so flipping the flag alone would leave the branch
        visible but still unclaimable.

        Must run inside the caller's transaction (see _resolve_branch_
        memberships): the condition it looks for is only ever created by a
        membership resolution, an opener-work completion, or a create_branch
        call, all of which already hold the write lock here. Returns the
        demoted branch_ids.
        """
        rows = self._conn.execute("""
            SELECT active.branch_id
            FROM active_branches AS active
            WHERE active.status = 'open'
              AND active.requires_opener_membership = 1
              AND NOT EXISTS (
                  SELECT 1 FROM live_branch_opener_rows AS owner
                  WHERE owner.branch_id = active.branch_id
              )
        """).fetchall()
        branch_ids = [row["branch_id"] for row in rows]
        if branch_ids:
            now = int(time.time())
            placeholders = ",".join("?" for _ in branch_ids)
            self._conn.execute(
                "UPDATE branch_opener_work SET resolved_at = ? "
                f"WHERE resolved_at IS NULL AND branch_id IN ({placeholders})",
                (now, *branch_ids))
            self._conn.executemany(
                "UPDATE active_branches SET requires_opener_membership = 0 "
                "WHERE branch_id = ?",
                [(branch_id,) for branch_id in branch_ids])
        return branch_ids

    def reconcile_orphaned_branch_ownership(self) -> list[int]:
        """Demote every currently-orphaned open owned branch, standalone.

        For one-off cleanup of branches stranded before this reconciliation
        existed (or accumulated while it could not run, e.g. the swarm was
        down) -- the CLI's `queue reconcile-orphaned-ownership` command.
        Normal operation self-heals through _resolve_branch_memberships and
        never needs this called directly."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            branch_ids = self._demote_orphaned_owned_branches()
            self._conn.execute("COMMIT")
            return branch_ids
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

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
                  cur_path=None, cur_help_depth=0, opener_work_id=None,
                  scheduling_role=None,
                  hint_lookups=None, hint_hits=None, hint_accepted=None,
                  hint_rejected=None, hint_inline_placements=None,
                  hint_inline_wins=None):
        if scheduling_role is not None and scheduling_role not in SCHEDULING_ROLES:
            raise ValueError(f"unknown scheduling_role {scheduling_role!r}: "
                             f"expected one of {SCHEDULING_ROLES}")
        now = int(time.time())
        current_branch_id = (self._intern_branch(current_branch_key, create=True)
                             if current_branch_key is not None else None)
        self._conn.execute("""
            INSERT OR REPLACE INTO worker_heartbeat
                (worker_id, pid, current_branch_id, n_words, started_at,
                 updated_at, claims_done, claim_idx, claim_started_at,
                 cand_rate, cache_hits, cache_misses, n_cutoff, n_pruned, n_ok,
                 best_guess, best_erd, bound_erd, cur_candidate, cand_n_seen, claim_total,
                 cur_max_depth, cur_nodes, node_rate, cur_path, cur_help_depth,
                 opener_work_id, scheduling_role,
                 hint_lookups, hint_hits, hint_accepted, hint_rejected,
                 hint_inline_placements, hint_inline_wins)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?)
        """, (worker_id, pid, current_branch_id, n_words, started_at,
              now, claims_done, claim_idx, claim_started_at,
              cand_rate, cache_hits, cache_misses, n_cutoff, n_pruned, n_ok,
              best_guess, best_erd, bound_erd, cur_candidate, cand_n_seen, claim_total,
              cur_max_depth, cur_nodes, node_rate, cur_path, cur_help_depth,
              opener_work_id, scheduling_role,
              hint_lookups, hint_hits, hint_accepted, hint_rejected,
              hint_inline_placements, hint_inline_wins))
        # Attributed so a heartbeat write storm is visible in the WAL report: an
        # unthrottled per-candidate heartbeat is a full-page WAL frame each and
        # otherwise hides here, uncategorised.
        self._tally_wal_traffic('worker_heartbeat/heartbeat', 1,
                                _CLAIM_ROW_WAL_BYTES)

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

    def overview_phase_counts(self) -> dict:
        """Return queue-wide evaluating and finalizing branch counts."""
        row = self._conn.execute("""
            WITH branch_ids AS (
                SELECT branch_id FROM pending_branches
                UNION
                SELECT branch_id FROM active_branches
            )
            SELECT
                SUM(CASE
                    WHEN pending.branch_id IS NOT NULL
                     AND pending.status != 'done'
                     AND active.status IS NOT 'finalized'
                     AND (active.status = 'open'
                          OR pending.status = 'in_progress')
                    THEN 1 ELSE 0 END) AS evaluating_user_branch_count,
                SUM(CASE
                    WHEN pending.branch_id IS NULL AND active.status = 'open'
                    THEN 1 ELSE 0 END) AS evaluating_cooperative_branch_count,
                SUM(CASE
                    WHEN pending.status IS NOT 'done'
                     AND active.status = 'finalized'
                    THEN 1 ELSE 0 END) AS finalizing_branch_count
            FROM branch_ids
            LEFT JOIN pending_branches AS pending USING (branch_id)
            LEFT JOIN active_branches AS active USING (branch_id)
        """).fetchone()
        return {
            key: row[key] or 0
            for key in (
                "evaluating_user_branch_count",
                "evaluating_cooperative_branch_count",
                "finalizing_branch_count",
            )
        }

    def completed_candidate_indexes_by_branch(self, branch_keys) -> dict:
        """Return completed candidate indexes for the requested branches."""
        branch_ids = self._branch_ids_for_keys(branch_keys)
        indexes_by_branch = {bytes(branch_key): [] for branch_key in branch_keys}
        if not branch_ids:
            return indexes_by_branch
        placeholders = ",".join("?" for _ in branch_ids)
        rows = self._conn.execute(
            f"""SELECT branches.branch_key, candidate_claims.idx
                FROM candidate_claims
                JOIN branches USING (branch_id)
                WHERE candidate_claims.branch_id IN ({placeholders})
                  AND candidate_claims.done = 1
                ORDER BY branches.branch_key, candidate_claims.idx""",
            branch_ids,
        ).fetchall()
        for row in rows:
            indexes_by_branch[bytes(row["branch_key"])].append(row["idx"])
        return indexes_by_branch

    def heartbeats_with_branch(self):
        """Heartbeat rows joined to the branch each worker is contributing to.

        opener/pattern/priority prefer the OWNER the worker itself selected
        at its last claim boundary (h.opener_work_id, via live_branch_opener_rows)
        and fall back to active_branches' own fields for a direct branch with no
        live opener-work ownership — so a shared branch's display reflects each
        worker's actual selected owner rather than one arbitrary label for every
        worker on the branch.  on_active_branch is 1 when that branch still has
        an active row and 0 once it has been finalized and removed, so a display
        can tell a working worker apart from one between branches.
        """
        return self._conn.execute("""
            SELECT h.*,
                   bk.branch_key AS current_branch_key,
                   COALESCE(owner.owner_priority, b.priority) AS priority,
                   b.spine,
                   COALESCE(owner.owner_opener, b.opener) AS opener,
                   COALESCE(owner.owner_opener_pattern, b.opener_pattern) AS opener_pattern,
                   b.branch_id IS NOT NULL AS on_active_branch
            FROM worker_heartbeat h
            LEFT JOIN active_branches b
                   ON h.current_branch_id = b.branch_id
            LEFT JOIN branches bk
                   ON h.current_branch_id = bk.branch_id
            LEFT JOIN live_branch_opener_rows owner
                   ON owner.branch_id = h.current_branch_id
                  AND owner.opener_work_id = h.opener_work_id
            ORDER BY h.worker_id
        """).fetchall()

    # ------------------------------------------------------------------
    # Branch swarm: cooperative candidate-level solve of one branch
    # ------------------------------------------------------------------

    def create_branch(self, branch_key, n_words, n_candidates,
                      priority=0, opener=None, opener_pattern=None,
                      budget=None, spine=None, root_budget=None,
                      ceiling=None, opener_work_id=None,
                      parent_branch_key=None) -> bool:
        """Register a branch as in-progress (status 'open'), if not present.

        Idempotent via INSERT OR IGNORE: the worker that promoted the branch
        from the queue creates it; others that race simply see it exists and
        join.  Returns True if this call created the row.  n_candidates is the
        total claim slot count (one slot per candidate in the policy-canonical
        list).  budget is the guess budget for depth-limited ERD.  spine is the
        guesses played from the root to this branch (see the active_branches.spine
        column); None leaves the display to fall back to the opener word.

        ceiling is the alpha-beta ceiling the branch is solved under (NULL =
        exact).  Immutable once set: a racing creator whose ceiling differs
        must check the surviving row's ceiling for joinability (see
        cooperative_solve) — the row is never relaxed in place, because done=1
        claims made under a tighter ceiling would be unsound under a looser one.

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
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if (opener_work_id is not None
                    and not self._opener_response_group_is_live(
                        opener_work_id, opener_pattern)):
                self._conn.execute("COMMIT")
                return False
            cur = self._conn.execute("""
                INSERT OR IGNORE INTO active_branches
                    (branch_id, n_words, n_candidates,
                     priority, requires_opener_membership,
                     opener, opener_pattern, status, created_at,
                     budget, spine, ceiling)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
            """, (branch_id, n_words, n_candidates,
                  priority, int(opener_work_id is not None), opener,
                  opener_pattern, now, budget, spine, ceiling))
            created = cur.rowcount == 1
            if opener_work_id is not None and created:
                parent_branch_id = (self._intern_branch(parent_branch_key)
                                    if parent_branch_key is not None else None)
                self._conn.execute("""
                    INSERT INTO branch_opener_work
                        (branch_id, opener_work_id, parent_branch_id, opener_pattern)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(branch_id, opener_work_id) DO UPDATE SET
                        parent_branch_id = excluded.parent_branch_id,
                        opener_pattern = excluded.opener_pattern,
                        resolved_at = NULL
                """, (branch_id, opener_work_id, parent_branch_id,
                      opener_pattern))
            self._conn.execute("COMMIT")
            return created
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def attach_branch_opener_work(self, branch_key, opener_work_id, budget,
                                  ceiling, n_words, opener_pattern,
                                  parent_branch_key=None) -> bool:
        """Attach opener ownership when the surviving branch is joinable."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            branch_id = self._intern_branch(branch_key)
            active = branch_id is not None and self._conn.execute(
                "SELECT budget, ceiling FROM active_branches "
                "WHERE branch_id = ? AND status = 'open'",
                (branch_id,)).fetchone()
            opener_response_group_is_live = self._opener_response_group_is_live(
                opener_work_id, opener_pattern)
            if (not active or not opener_response_group_is_live
                    or active["budget"] != budget
                    or (active["ceiling"] is not None
                        and (ceiling is None
                             or not erd_ge(active["ceiling"], ceiling, n_words)))):
                self._conn.execute("COMMIT")
                return False
            parent_branch_id = (self._intern_branch(parent_branch_key)
                                if parent_branch_key is not None else None)
            self._conn.execute("""
                INSERT INTO branch_opener_work
                    (branch_id, opener_work_id, parent_branch_id, opener_pattern)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(branch_id, opener_work_id) DO UPDATE SET
                    parent_branch_id = excluded.parent_branch_id,
                    opener_pattern = excluded.opener_pattern,
                    resolved_at = NULL
            """, (branch_id, opener_work_id, parent_branch_id, opener_pattern))
            self._conn.execute("""
                UPDATE active_branches
                SET requires_opener_membership = 1
                WHERE branch_id = ?
            """, (branch_id,))
            self._conn.execute("COMMIT")
            return True
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def get_branch(self, branch_key):
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        return self._conn.execute(
            "SELECT * FROM active_branches WHERE branch_id = ?",
            (branch_id,)).fetchone()

    def _begin_immediate_timed(self):
        """BEGIN IMMEDIATE, adding the wait for the write lock to
        _last_claim_busy_millis.

        For the write paths that let SQLite's own busy_timeout do the waiting
        rather than running claim_next_bundle's application-level retry probe:
        the wait is the same contention signal either way, so it belongs in
        the same accumulator.  Those paths make no application-level retries,
        so they leave claim_retries alone.

        Only for locks taken while the worker is coordinating between
        candidates.  A lock taken during candidate evaluation (sub-branch
        promotion, e.g. attach_branch_opener_work) must NOT use this: that
        time sits inside the evaluation span which evaluate_claim subtracts
        from coordination_millis, so counting it would push the phase parts
        above the whole they are supposed to partition.
        """
        _acquire_t0 = time.perf_counter()
        self._conn.execute("BEGIN IMMEDIATE")
        self._last_claim_busy_millis += int(
            (time.perf_counter() - _acquire_t0) * 1e3)

    def _commit_claim_transaction(self, txn_t0):
        """COMMIT a claim_next_bundle transaction, timing scan/write vs COMMIT.

        txn_t0 is a time.perf_counter() reading taken right after the write
        lock was acquired (BEGIN IMMEDIATE succeeded).  Populates
        _last_claim_transaction_millis (scan and write statements before this
        call) and _last_claim_commit_millis (the COMMIT itself), consumed and
        reset by the next add_claim_telemetry the same way as
        _last_claim_busy_millis/_last_claim_retries.
        """
        self._last_claim_transaction_millis = int(
            (time.perf_counter() - txn_t0) * 1e3)
        _commit_t0 = time.perf_counter()
        self._conn.execute("COMMIT")
        self._last_claim_commit_millis = int(
            (time.perf_counter() - _commit_t0) * 1e3)

    def _count_one_level_erd_prunes(self, branch_id, n):
        """Fold n one-level ERD prunes into a branch's completion counters.

        bulk_done_candidates is the legacy combined count and stays the sum of
        both prune provenances.  A branch that has no achieved best yet but
        does have a ceiling is pruning against that ceiling, which makes the
        outcome a CUT rather than a proven loss.
        """
        self._conn.execute("""
            UPDATE active_branches
            SET bulk_done_candidates = bulk_done_candidates + ?,
                one_level_erd_pruned_candidates =
                    one_level_erd_pruned_candidates + ?,
                cut_occurred = CASE
                    WHEN best_erd IS NULL AND ceiling IS NOT NULL THEN 1
                    ELSE cut_occurred END
            WHERE branch_id = ?
        """, (n, n, branch_id))

    def _record_holes(self, where, parameters, label):
        """Record every candidate_claims row matching `where` as a hole.

        MUST run in the caller's write transaction, the same one as the DELETE
        that frees those rows.  A hole row that commits while its claim row
        still exists is indistinguishable from a stale one, so a
        claim_next_bundle landing in that window drops it without packing it
        and the position falls back to the end-of-sweep backstop — the
        deferral this scheduler exists to remove.  `where` is the DELETE's own
        predicate, reused verbatim so the two statements cannot drift.

        REPLACE, not IGNORE: a position freed twice keeps the position from
        the claim row that was just freed, which is the one the packer will
        reissue.
        """
        self._conn.execute(
            f"INSERT OR REPLACE INTO candidate_holes "
            f"(branch_id, idx, best_first_position) "
            f"SELECT branch_id, idx, best_first_position "
            f"FROM candidate_claims "
            f"WHERE {where}", parameters)
        n = self._conn.execute("SELECT changes()").fetchone()[0]
        self._tally_wal_traffic(label, n, n * _CLAIM_ROW_WAL_BYTES)
        return n

    def _free_claims_as_holes(self, where, parameters, hole_label, claim_label):
        """Atomically record the claims matching `where` as holes and free them.

        The reclaim paths run in autocommit, so the pair needs its own
        BEGIN IMMEDIATE — see _record_holes for what a concurrent claim does to
        a hole that commits ahead of its DELETE.  Every caller runs at top
        level, never inside an open transaction.  Returns the number of claims
        freed.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._record_holes(where, parameters, hole_label)
            self._conn.execute(
                f"DELETE FROM candidate_claims WHERE {where}", parameters)
            n = self._conn.execute("SELECT changes()").fetchone()[0]
            self._conn.execute("COMMIT")
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise
        self._tally_wal_traffic(claim_label, n, n * _CLAIM_ROW_WAL_BYTES)
        return n

    def _pack_recorded_holes(self, branch_id, bound, cost_lower_bound,
                             survivor_limit):
        """Take a branch's earliest outstanding holes for one bundle.

        Runs inside claim_next_bundle's write transaction and returns the
        packed [(idx, best_first_position), ...] in best-first order, at most
        survivor_limit of them.  Holes past that limit keep their
        candidate_holes rows and are packed by the next claim, still ahead of
        any virgin position.

        A hole whose idx has since acquired a claim row is stale — a one-level
        ERD-prune sweep (including this transaction's own) or the mid-loop
        publisher landed on it — and is dropped without being packed.  A hole
        the branch's current bound prunes is completed here as a one-level ERD
        prune, exactly as the end-of-sweep coverage backstop does.  Every hole
        this call resolves loses its row; the rest are left for the next claim.

        A hole recorded from a claim row that predates the best_first_position
        column sorts after every positioned hole, since its rank is not
        recoverable.
        """
        hole_rows = self._conn.execute(
            "SELECT idx, best_first_position FROM candidate_holes "
            "WHERE branch_id = ? "
            "ORDER BY best_first_position IS NULL, best_first_position",
            (branch_id,)).fetchall()
        if not hole_rows:
            return []
        hole_indices = [row["idx"] for row in hole_rows]
        placeholders = ",".join("?" * len(hole_indices))
        claimed = {r["idx"] for r in self._conn.execute(
            f"SELECT idx FROM candidate_claims "
            f"WHERE branch_id = ? AND idx IN ({placeholders})",
            (branch_id, *hole_indices))}
        eliminated = []
        packed = []
        for row in hole_rows:
            idx = row["idx"]
            if idx in claimed:
                continue
            if cost_lower_bound[idx] >= bound:
                eliminated.append(idx)
            elif len(packed) < survivor_limit:
                packed.append((idx, row["best_first_position"]))
        if eliminated:
            now = int(time.time())
            self._conn.executemany("""
                INSERT INTO candidate_claims
                    (branch_id, idx, claimed_by, claimed_at, done, done_at,
                     bundle_id)
                VALUES (?, ?, 'one-level-erd-prune', ?, 1, ?, NULL)
            """, [(branch_id, idx, now, now) for idx in eliminated])
            self._tally_wal_traffic(
                'candidate_claims/one-level-erd-prune-hole', len(eliminated),
                len(eliminated) * _CLAIM_ROW_WAL_BYTES)
            self._count_one_level_erd_prunes(branch_id, len(eliminated))
        resolved = ([idx for idx in hole_indices if idx in claimed]
                    + eliminated
                    + [idx for idx, _position in packed])
        resolved_placeholders = ",".join("?" * len(resolved))
        self._conn.execute(
            f"DELETE FROM candidate_holes WHERE branch_id = ? "
            f"AND idx IN ({resolved_placeholders})", (branch_id, *resolved))
        self._tally_wal_traffic(
            'candidate_holes/pack', len(resolved),
            len(resolved) * _CLAIM_ROW_WAL_BYTES)
        return packed

    def claim_next_bundle(self, branch_key, worker_id, n_candidates,
                          candidate_order, cost_lower_bound,
                          small_count=DEFAULT_SMALL_COUNT,
                          count_cap=DEFAULT_COUNT_CAP,
                          republish_limit=DEFAULT_REPUBLISH_LIMIT,
                          expected_opener_work_id=None,
                          expected_opener_priority=None,
                          max_other_workers=None):
        """Atomically complete one-level ERD prunes and claim survivors.

        max_other_workers caps how many other workers may hold work on this
        branch at once, enforced here rather than by the caller because only
        this transaction can decide it: occupancy is counted from live claims,
        which this call is about to create, so two workers that both saw an
        empty branch cannot both pass.  None disables the cap.

        Runs the exact-elimination classification from
        adaptive_claim_packing.md §5 inside one BEGIN IMMEDIATE transaction:
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

        Each call packs the branch's earliest outstanding candidates by
        candidate_order, on whichever side of the forward cursor
        (active_branches.pack_cursor) they lie.  A hole — an idx with no
        current candidate_claims row, freed by a reclaim or a republish —
        carries the best-first position it was claimed at in candidate_holes,
        and every hole lies below the cursor, so holes are drained in
        best-first order first and the forward cursor only tops the bundle up.
        A republished high-ranked candidate is therefore reissued ahead of
        untouched lower-ranked ones rather than waiting for the forward sweep
        to finish.  Once the cursor reaches n_candidates and no recorded hole
        survives, one full pass over the branch's claim rows backstops
        coverage, so a position missing from candidate_holes can cost
        priority but never the candidate.

        Provably eliminated slots are completed authoritatively in this
        transaction and never returned to a worker.  INSERT OR REPLACE
        deliberately supersedes a racing done=0 claim: the admissible lower
        bound is already a complete proof, while preserving the in-flight row
        would spend engine work to establish the same fact.  Aggregate counts
        are retained on the branch instead of emitting per-candidate telemetry.

        Returns (bundle_id, indices, forced); None if nothing is claimable
        (branch not open or missing, wrong owner, or every slot already has
        a row); or the CLAIM_RETRY sentinel if this call's own claim attempt
        was entirely superseded by a race — the branch remains claimable and
        the caller should call again for the same branch_key rather than
        moving on to a different one (see CLAIM_RETRY).
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
                        raise  # pragma: no cover — total lock starvation
                    retries += 1
        finally:
            self._conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        self._last_claim_busy_millis += int(
            (time.perf_counter() - _acquire_t0) * 1e3)
        self._last_claim_retries += retries
        _txn_t0 = time.perf_counter()
        try:
            # Never hand out a claim for a branch that has been finalized and
            # deleted: a worker still looping would otherwise redo it from
            # scratch.  Checked inside the write transaction so it can't race
            # finalize+delete.  An unregistered branch has no active row, so it
            # is treated the same as a missing one.
            branch_id = self._intern_branch(branch_key)
            br = None if branch_id is None else self._conn.execute(
                "SELECT status, best_erd, pack_cursor, ceiling, bulk_done_bound, "
                "requires_opener_membership "
                "FROM active_branches "
                "WHERE branch_id = ?", (branch_id,)).fetchone()
            if br is None or br["status"] != "open":
                self._commit_claim_transaction(_txn_t0)
                return None
            if expected_opener_work_id is None:
                owner_matches = (not br["requires_opener_membership"] and
                    self._conn.execute("""
                    SELECT 1 FROM branch_opener_work
                    WHERE branch_id = ? AND resolved_at IS NULL
                """, (branch_id,)).fetchone() is None)
            else:
                owner_matches = self._conn.execute("""
                    SELECT 1 FROM live_branch_opener_rows
                    WHERE branch_id = ? AND opener_work_id = ?
                """, (branch_id, expected_opener_work_id)).fetchone() is not None
                if owner_matches and expected_opener_priority is not None:
                    owner_matches = self._conn.execute("""
                        SELECT 1 FROM live_branch_opener_rows
                        WHERE branch_id = ? AND opener_work_id = ?
                          AND owner_priority = ?
                    """, (branch_id, expected_opener_work_id,
                          expected_opener_priority)).fetchone() is not None
            if not owner_matches:
                self._commit_claim_transaction(_txn_t0)
                return None
            if (max_other_workers is not None
                    and self._other_claim_holders(branch_id, worker_id)
                        > max_other_workers):
                self._commit_claim_transaction(_txn_t0)
                return None
            # The branch ceiling is a bound like any achieved best: candidates
            # whose lower bound reaches it are provably pruned for free, so the
            # packer classifies against the tighter of the two.
            bound = min(
                br["best_erd"] if br["best_erd"] is not None else float("inf"),
                br["ceiling"] if br["ceiling"] is not None else float("inf"))
            swept_bound = br["bulk_done_bound"]
            bound_tightened = (bound != float("inf")
                               and (swept_bound is None or bound < swept_bound))
            claim_rows = None
            if bound_tightened:
                claim_rows = {row["idx"]: row for row in self._conn.execute(
                    "SELECT idx, done FROM candidate_claims WHERE branch_id = ?",
                    (branch_id,))}
                eliminated_indices = [
                    idx for idx in candidate_order
                    if cost_lower_bound[idx] >= bound
                    and (idx not in claim_rows or not claim_rows[idx]["done"])
                ]
            else:
                eliminated_indices = []
            if eliminated_indices:
                now = int(time.time())
                # best_erd only decreases and ceiling is immutable, so this
                # bound only tightens; an eliminated candidate remains
                # eliminated for the rest of the branch's lifetime.
                self._conn.executemany("""
                    INSERT OR REPLACE INTO candidate_claims
                        (branch_id, idx, claimed_by, claimed_at, done, done_at,
                         bundle_id)
                    VALUES (?, ?, 'one-level-erd-prune', ?, 1, ?, NULL)
                """, [(branch_id, idx, now, now)
                      for idx in eliminated_indices])
                self._tally_wal_traffic(
                    'candidate_claims/one-level-erd-prune',
                    len(eliminated_indices),
                    len(eliminated_indices) * _CLAIM_ROW_WAL_BYTES)
                self._count_one_level_erd_prunes(
                    branch_id, len(eliminated_indices))
                for idx in eliminated_indices:
                    claim_rows[idx] = {"idx": idx, "done": 1}
            if bound_tightened:
                self._conn.execute(
                    "UPDATE active_branches SET bulk_done_bound = ? "
                    "WHERE branch_id = ?", (bound, branch_id))

            cursor = br["pack_cursor"]
            survivor_limit = min(small_count, count_cap)
            # Outstanding holes all sit below the forward cursor, so the
            # branch's earliest outstanding candidate is the lowest-positioned
            # hole whenever any exists.  Draining them first is what keeps a
            # republished high-ranked candidate ahead of untouched later ones.
            packed = self._pack_recorded_holes(
                branch_id, bound, cost_lower_bound, survivor_limit)
            if len(packed) < survivor_limit and cursor < n_candidates:
                new_cursor = cursor
                while new_cursor < n_candidates and len(packed) < survivor_limit:
                    idx = candidate_order[new_cursor]
                    new_cursor += 1
                    if cost_lower_bound[idx] < bound:
                        packed.append((idx, new_cursor - 1))
                self._conn.execute(
                    "UPDATE active_branches SET pack_cursor = ? "
                    "WHERE branch_id = ?", (new_cursor, branch_id))
            elif not packed and cursor >= n_candidates:
                # Coverage backstop: candidate_holes is an ordering index, and
                # a position missing from it (a row written by code that
                # predates the table, or one dropped as stale while its claim
                # was superseded and later freed) would otherwise never be
                # reissued.  Whatever this finds is put back into the index and
                # packed through the same path as any other hole.  Reached only
                # once the forward sweep is exhausted AND no recorded hole
                # survived, so the full re-read costs nothing on a branch that
                # still has work the index knows about.
                if claim_rows is None:
                    claim_rows = {row["idx"]: row for row in self._conn.execute(
                        "SELECT idx, done FROM candidate_claims "
                        "WHERE branch_id = ?", (branch_id,))}
                    # A full per-branch claims re-read holds a WAL snapshot for
                    # its duration, which is what a checkpoint waits on.
                    # Tallied so the report shows read pressure, not just write
                    # pressure.
                    self._tally_wal_traffic(
                        'candidate_claims/holes-scan(read)', len(claim_rows),
                        len(claim_rows) * _CLAIM_ROW_WAL_BYTES)
                unindexed = [(idx, position)
                             for position, idx in enumerate(candidate_order)
                             if idx not in claim_rows]
                if unindexed:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO candidate_holes "
                        "(branch_id, idx, best_first_position) VALUES (?, ?, ?)",
                        [(branch_id, idx, position)
                         for idx, position in unindexed])
                    self._tally_wal_traffic(
                        'candidate_holes/backstop', len(unindexed),
                        len(unindexed) * _CLAIM_ROW_WAL_BYTES)
                    packed = self._pack_recorded_holes(
                        branch_id, bound, cost_lower_bound, survivor_limit)
            if not packed:
                self._commit_claim_transaction(_txn_t0)
                return None
            bundle = [idx for idx, _position in packed]
            bundle_id = f"{worker_id}:{self._pid}:{self._bundle_seq}"
            self._bundle_seq += 1
            now = int(time.time())
            # INSERT OR IGNORE, not a plain INSERT: on the forward path,
            # cursor < n_candidates only proves these positions have never
            # been *packed* before, not that no row exists at that idx yet.
            # mark_claims_done (within-candidate overrun promotion) can
            # insert done=1 rows for a fresh branch's best-first prefix
            # before the packer ever runs over it, landing on exactly the
            # positions the forward path is about to pack. A plain INSERT
            # would collide on the (branch_id, idx) primary key; IGNORE
            # skips those rows, and the SELECT below discovers which of
            # `bundle` were actually claimed by this call, via the
            # bundle_id just stamped on them (idx_candidate_claims_bundle
            # makes this an indexed lookup, not a rescan).
            self._conn.executemany("""
                INSERT OR IGNORE INTO candidate_claims
                    (branch_id, idx, claimed_by, claimed_at, done, bundle_id,
                     best_first_position)
                VALUES (?, ?, ?, ?, 0, ?, ?)
            """, [(branch_id, idx, worker_id, now, bundle_id, position)
                  for idx, position in packed])
            self._tally_wal_traffic(
                'candidate_claims/claim', len(bundle),
                len(bundle) * _CLAIM_ROW_WAL_BYTES)
            actually_claimed = {r["idx"] for r in self._conn.execute(
                "SELECT idx FROM candidate_claims "
                "WHERE branch_id = ? AND bundle_id = ?",
                (branch_id, bundle_id))}
            bundle = [idx for idx in bundle if idx in actually_claimed]
            if not bundle:
                # Every packed position already had a row (e.g. mark_claims_
                # done beat us to the whole prefix): the cursor still
                # advanced past them, so the caller should just retry.
                self._commit_claim_transaction(_txn_t0)
                return CLAIM_RETRY
            placeholders = ",".join("?" * len(bundle))
            rows = self._conn.execute(
                f"SELECT idx FROM candidate_republish "
                f"WHERE branch_id = ? AND count >= ? "
                f"AND idx IN ({placeholders})",
                (branch_id, republish_limit, *bundle)).fetchall()
            forced = frozenset(r["idx"] for r in rows)
            self._commit_claim_transaction(_txn_t0)
            return (bundle_id, bundle, forced)
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def republish_remainder(self, branch_key, bundle_id, indices):
        """Return an overrun bundle's unfinished remainder to the unclaimed pool.

        adaptive_claim_packing.md §7a: deletes the done=0 candidate_claims
        rows for `indices` so they re-enter as holes, re-packed by the next
        claim_next_bundle against a B that is at least as tight as when they
        were first packed — never re-inserted as `len(indices)` individual
        claims.  Each freed position is recorded in candidate_holes, so the
        next claim reissues the strongest of them before any candidate the
        forward cursor has not reached: a remainder keeps the rank it had
        when its bundle overran.  Also bumps each candidate's persistent republish count
        (candidate_republish), which survives this delete/re-insert cycle so
        a chronically-stranded candidate is distinguishable from one
        republished for the first time.

        The delete is scoped to `bundle_id`, not just (branch_key, idx): a
        worker can stall past the heartbeat timeout on a heavy candidate,
        have reclaim_stale_claims free its done=0 rows, and have another
        worker's packer legitimately re-claim some of those idx under a NEW
        bundle_id before this worker finally overruns — an unscoped delete
        would remove that other worker's live claim rows and spuriously
        bump their republish counts.  A SELECT identifies exactly which idx
        are still claimed under THIS bundle_id before the DELETE removes
        them — both run inside the same BEGIN IMMEDIATE transaction, so no
        other writer can change candidate_claims between the two (the write
        lock is held for the whole method, same as a DELETE ... RETURNING
        would give; RETURNING itself needs SQLite 3.35+, and the production
        box runs 3.34.1 — see the module-level SQLite version note).  The
        count bump — and the returned {idx: count} map — covers only the
        idx the SELECT found; one that already got reclaimed under a
        different bundle_id is silently left alone here.
        """
        if not indices:
            return {}
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            branch_id = self._intern_branch(branch_key, create=True)
            placeholders = ",".join("?" * len(indices))
            deleted = [r["idx"] for r in self._conn.execute(
                f"SELECT idx FROM candidate_claims WHERE branch_id = ? "
                f"AND done = 0 AND bundle_id = ? AND idx IN ({placeholders})",
                (branch_id, bundle_id, *indices))]
            if not deleted:
                self._conn.execute("COMMIT")
                return {}
            deleted_placeholders = ",".join("?" * len(deleted))
            republished_where = (
                f"branch_id = ? AND bundle_id = ? "
                f"AND idx IN ({deleted_placeholders})")
            self._record_holes(republished_where,
                               (branch_id, bundle_id, *deleted),
                               'candidate_holes/republish')
            self._conn.execute(
                f"DELETE FROM candidate_claims WHERE {republished_where}",
                (branch_id, bundle_id, *deleted))
            self._conn.executemany("""
                INSERT INTO candidate_republish (branch_id, idx, count)
                VALUES (?, ?, 1)
                ON CONFLICT(branch_id, idx) DO UPDATE SET count = count + 1
            """, [(branch_id, idx) for idx in deleted])
            self._tally_wal_traffic(
                'candidate_claims/republish-delete', len(deleted),
                len(deleted) * _CLAIM_ROW_WAL_BYTES)
            self._tally_wal_traffic(
                'candidate_republish/republish-upsert', len(deleted),
                len(deleted) * _CLAIM_ROW_WAL_BYTES)
            rows = self._conn.execute(
                f"SELECT idx, count FROM candidate_republish "
                f"WHERE branch_id = ? AND idx IN ({deleted_placeholders})",
                (branch_id, *deleted)).fetchall()
            self._conn.execute("COMMIT")
            return {r["idx"]: r["count"] for r in rows}
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def record_bundle_stats(self, branch_key, bundle_id, nodes, wall_millis,
                            censored=False):
        """Record one bundle's actual node cost and evaluation wall span.

        wall_millis is the bundle's own evaluation wall time (straggler/
        reclaim-window diagnostic) — NOT claim-handout coordination overhead;
        that is claim_telemetry's busy_wait_millis, measured separately in
        claim_next_bundle.  Aggregated into branch_finalize_log at finalize
        (see finalize_bundle_stats) and dropped along with the rest of the
        branch's transient state by delete_branch.  censored=1 marks a bundle
        that hit its node/wall cap and republished an unfinished remainder —
        nodes is then a lower bound on what the bundle's original member set
        would have cost.  A republished remainder re-packs under a new
        bundle_id, which gets its own independent row here.

        The bundle's last candidate can be marked done (making
        branch_done_candidates cover the branch) before this call runs, so
        another worker's maybe_finalize can win try_finalize_branch and
        delete_branch/finalize_bundle_stats out from under this INSERT.  The
        WHERE EXISTS guard suppresses the insert in that race.  The guard
        reads active_branches in the queue file while inserting into the
        telemetry file, so it is advisory rather than atomic: a concurrent
        DELETE committing between the read and the write can still leave an
        orphaned row.  An orphan is harmless — the table is a diagnostic
        never read by any control path — and rare enough that finalize/
        delete cleanup keeps the table bounded in practice.
        """
        # bundle_stats lives in the telemetry file and keeps branch_key so the
        # telemetry database stays self-describing for standalone analysis; only
        # the active_branches existence guard uses branch_id.
        branch_id = self._intern_branch(branch_key)
        self._conn.execute("""
            INSERT OR REPLACE INTO telemetry.bundle_stats
                (branch_key, bundle_id, nodes, wall_millis, censored)
            SELECT ?, ?, ?, ?, ?
            WHERE EXISTS (SELECT 1 FROM active_branches WHERE branch_id = ?)
        """, (branch_key, bundle_id, nodes, wall_millis,
              1 if censored else 0, branch_id))

    def finalize_bundle_stats(self, branch_key):
        """Aggregate and clear a branch's bundle_stats rows at finalize.

        Returns (n_bundles, max_bundle_nodes, total_bundle_wall_millis,
        censored_units) for branch_finalize_log — all None if no bundle
        recorded any stats (e.g. a branch solved entirely from reused cache
        entries).  Deletes the rows so bundle_stats stays bounded to
        currently-open branches, matching delete_branch's candidate_claims
        cleanup.
        """
        row = self._conn.execute("""
            SELECT COUNT(*) AS n_bundles, MAX(nodes) AS max_bundle_nodes,
                   SUM(wall_millis) AS total_bundle_wall_millis,
                   SUM(censored) AS censored_units
            FROM telemetry.bundle_stats WHERE branch_key = ?
        """, (branch_key,)).fetchone()
        self._conn.execute(
            "DELETE FROM telemetry.bundle_stats WHERE branch_key = ?",
            (branch_key,))
        if row["n_bundles"] == 0:
            return (None, None, None, None)
        return (row["n_bundles"], row["max_bundle_nodes"],
                row["total_bundle_wall_millis"], row["censored_units"])

    def complete_candidate(self, branch_key, idx):
        """Mark a candidate claim authoritatively complete (done=1)."""
        now = int(time.time())
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute("""
            UPDATE candidate_claims SET done = 1, done_at = ?
            WHERE branch_id = ? AND idx = ?
        """, (now, branch_id, idx))
        n = self._conn.execute("SELECT changes()").fetchone()[0]
        self._tally_wal_traffic(
            'candidate_claims/complete', n, n * _CLAIM_ROW_WAL_BYTES)

    def complete_bundle_two_level_erd_prunes(self, branch_key, bundle_id,
                                             candidate_indices, nodes_spent=0,
                                             wall_millis=0, bound_erd=None,
                                             worker_count=None, worker_id=None):
        """Complete claimed candidates pruned by the two-level ERD bound.

        The worker computes the bounds before entering this transaction.  A
        branch's best ERD only decreases and its ceiling is immutable, so a
        candidate proved unable to beat the worker's earlier bound remains
        unable to beat every bound visible here.

        Only unfinished rows still owned by bundle_id are changed.  A stale
        worker whose bundle was reclaimed therefore cannot overwrite the new
        owner's claim, while a queue-level one-level ERD-prune sweep that
        already completed a row is left untouched and not counted twice.
        nodes_spent folds the bundle preflight's candidate-entry nodes into
        the same transaction as the completion update.
        """
        candidate_indices = list(candidate_indices)
        if not candidate_indices and nodes_spent <= 0:
            return 0
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return 0
        placeholders = ",".join("?" for _ in candidate_indices)
        now = int(time.time())
        branch_worker_count = self._branch_worker_count(
            branch_id, worker_id, now)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if candidate_indices:
                self._conn.execute(f"""
                    UPDATE candidate_claims
                    SET claimed_by = 'two-level-erd-prune', done = 1,
                        done_at = ?, bundle_id = NULL
                    WHERE branch_id = ? AND bundle_id = ? AND done = 0
                      AND idx IN ({placeholders})
                      AND EXISTS (
                          SELECT 1 FROM active_branches
                          WHERE branch_id = ? AND status = 'open'
                      )
                """, (now, branch_id, bundle_id, *candidate_indices, branch_id))
                completed_candidate_count = self._conn.execute(
                    "SELECT changes()").fetchone()[0]
            else:
                completed_candidate_count = 0
            updated_branch_count = 0
            if completed_candidate_count or nodes_spent > 0:
                self._conn.execute("""
                    UPDATE active_branches
                    SET bulk_done_candidates = bulk_done_candidates + ?,
                        two_level_erd_pruned_candidates =
                            two_level_erd_pruned_candidates + ?,
                        nodes_spent = nodes_spent + ?,
                        cut_occurred = CASE
                            WHEN best_erd IS NULL AND ceiling IS NOT NULL THEN 1
                            ELSE cut_occurred END
                    WHERE branch_id = ? AND status = 'open'
                """, (completed_candidate_count, completed_candidate_count,
                      max(0, nodes_spent), branch_id))
                updated_branch_count = self._conn.execute(
                    "SELECT changes()").fetchone()[0]
            if nodes_spent > 0 and updated_branch_count:
                self._conn.execute("""
                    INSERT INTO telemetry.two_level_prune_telemetry
                        (branch_id, inspected_candidate_count,
                         pruned_candidate_count, bound_erd, worker_count,
                         branch_worker_count, wall_millis, epoch, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (branch_id, nodes_spent, completed_candidate_count,
                      bound_erd, worker_count, branch_worker_count,
                      max(0, wall_millis), self.epoch, now))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._tally_wal_traffic(
            'candidate_claims/two-level-erd-prune',
            completed_candidate_count,
            completed_candidate_count * _CLAIM_ROW_WAL_BYTES)
        if nodes_spent > 0:
            self._tally_wal_traffic(
                'active_branches/nodes-spent', updated_branch_count,
                updated_branch_count * _CLAIM_ROW_WAL_BYTES)
        return completed_candidate_count

    def update_branch_best(self, branch_key, best_guess, best_erd, max_depth=None):
        """Lower the branch's running best (monotone — never raises it).

        max_depth is the winning candidate's worst-case line length; it is
        stored atomically with the best it belongs to, so best_max_depth always
        describes the current best_guess.

        The same statement stamps first_best_at/nodes_at_first_best on the
        update that creates the branch's first incumbent, and leaves them alone
        on every later improvement — COALESCE keeps the first value, so "how
        long, and how many nodes, before this branch had any bound at all"
        survives the rest of the solve.  nodes_spent is read from the row being
        updated, which is why this cannot be done by the caller: only the
        statement that wins the transition knows it was the first.
        """
        branch_id = self._intern_branch(branch_key, create=True)
        now = int(time.time())
        self._conn.execute("""
            UPDATE active_branches
            SET best_erd = ?, best_guess = ?, best_max_depth = ?,
                best_updated_at = ?,
                first_best_at = COALESCE(first_best_at, ?),
                nodes_at_first_best = COALESCE(nodes_at_first_best, nodes_spent)
            WHERE branch_id = ?
              AND (best_erd IS NULL OR ? < best_erd)
        """, (best_erd, best_guess, max_depth, now, now, branch_id,
              best_erd))

    def read_branch_best(self, branch_key):
        """Return (best_guess, best_erd, ceiling) or (None, None, None).

        ceiling is the branch's alpha-beta ceiling (None = exact solve): a
        bound source alongside best_erd — a candidate priced out against it is
        a cut, not a loss."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return (None, None, None)
        row = self._conn.execute(
            "SELECT best_guess, best_erd, ceiling FROM active_branches "
            "WHERE branch_id = ?", (branch_id,)).fetchone()
        if row is None:
            return (None, None, None)
        return (row["best_guess"], row["best_erd"], row["ceiling"])

    def mark_branch_tainted(self, branch_key):
        """Set the branch's taint flag (monotone OR): some candidate, in some
        worker, was excluded by the depth cap, so the branch's ERD is only
        valid at its solve budget."""
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute(
            "UPDATE active_branches SET tainted = 1 WHERE branch_id = ?",
            (branch_id,))

    def mark_branch_cut(self, branch_key):
        """Set the branch's cut flag (monotone OR): some candidate priced out
        at >= the bound rather than being proven infeasible.  At finalize with
        no best_guess this distinguishes a CUT (>= ceiling) from a proven loss."""
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute(
            "UPDATE active_branches SET cut_occurred = 1 WHERE branch_id = ?",
            (branch_id,))

    def read_branch_meta(self, branch_key):
        """Return (best_guess, best_erd, best_max_depth, tainted, budget,
        ceiling, cut_occurred) or None — everything finalize needs to triage
        exact / cut / loss and write the exact case to the cache."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        row = self._conn.execute(
            "SELECT best_guess, best_erd, best_max_depth, tainted, budget, "
            "ceiling, cut_occurred "
            "FROM active_branches WHERE branch_id = ?", (branch_id,)).fetchone()
        if row is None:
            return None
        return (row["best_guess"], row["best_erd"], row["best_max_depth"],
                bool(row["tainted"]), row["budget"], row["ceiling"],
                bool(row["cut_occurred"]))

    def add_cut_result(self, branch_key, budget, bound, tainted=False):
        """Publish a ceilinged solve's CUT: the branch's true ERD at `budget`
        is proven >= bound.  Upserts by (branch_id, budget, tainted), keeping
        MAX(existing bound, new bound): a cut is a durable proof, so a higher
        bound already on record for the same budget/taint class is never
        superseded by a lower one, and never discarded.

        A bound proven for a different budget, or with different taint, is
        kept in its own row rather than overwriting this one — see
        read_cut_result for why neither axis dominates the other.

        tainted records whether the cut's proof involved the remaining-depth
        floor anywhere: a tainted bound holds only among budget-feasible
        strategies, so a consumer must join it into its own floor taint."""
        now = int(time.time())
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute("""
            INSERT INTO cut_results (branch_id, budget, bound, tainted, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (branch_id, budget, tainted)
            DO UPDATE SET bound = MAX(bound, excluded.bound),
                          created_at = excluded.created_at
        """, (branch_id, budget, bound, int(bool(tainted)), now))

    def read_cut_result(self, branch_key):
        """Every cut recorded for this branch, one row per (budget, tainted)
        class it has been proven under, as (bound, budget, tainted) tuples
        ordered most-useful-first (largest budget, then untainted, then
        largest bound). Empty list if none.

        A bound proven at a given budget holds at any consumer budget <= it
        (fewer guesses cannot beat it) but says nothing at a larger one, so a
        bound proven at a lower budget does not dominate one proven at a
        higher budget; a tainted bound holds only among budget-feasible
        strategies (see add_cut_result), so it does not dominate an untainted
        one either. Each (budget, tainted) class therefore keeps its own row,
        and the caller — the only side that knows its own budget and ceiling
        — scans for the first row satisfying `consumer_budget <= budget and
        bound >= ceiling`, or logs the first (most-useful) row as the closest
        miss when none does."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return []
        rows = self._conn.execute("""
            SELECT bound, budget, tainted FROM cut_results
            WHERE branch_id = ?
            ORDER BY budget DESC, tainted ASC, bound DESC
        """, (branch_id,)).fetchall()
        return [(row["bound"], row["budget"], bool(row["tainted"]))
                for row in rows]

    def has_pending_row(self, branch_key) -> bool:
        """True if the branch is user-queued (has a pending_branches row in any
        status).  A user-queued branch always has an exact-result consumer, so
        it is never solved under a ceiling."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        return self._conn.execute(
            "SELECT 1 FROM pending_branches WHERE branch_id = ? LIMIT 1",
            (branch_id,)).fetchone() is not None

    def requeue_pending(self, branch_key) -> bool:
        """Flip an in-flight pending_branches row back to 'pending'.

        Called instead of mark_done when a branch finalizes as a CUT: the cut
        satisfies the promoting parent but a user-queued row wants an exact
        result, which was not produced.  Returns True if a row was reset."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        cur = self._conn.execute("""
            UPDATE pending_branches
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE branch_id = ? AND status != 'done'
        """, (branch_id,))
        return cur.rowcount > 0

    def complete_pending_for_loss(self, branch_key, loss_budget, root_budget) -> bool:
        """Atomically retire only pending work covered by a branch loss.

        A queued branch is reached through its single opener-word response, so
        its remaining budget is root_budget minus one when that spine is known.
        Returns True when no queued row needs further work, False when a queued
        request is left pending because the loss budget does not cover it.
        """
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return True
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("""
                SELECT opener, opener_pattern FROM pending_branches
                WHERE branch_id = ?
            """, (branch_id,)).fetchone()
            if row is None:
                self._conn.execute("COMMIT")
                return True
            pending_budget = root_budget - (
                1 if row["opener"] and row["opener_pattern"] is not None
                else 0)
            if loss_budget is not None and loss_budget >= pending_budget:
                self._conn.execute("""
                    UPDATE pending_branches
                    SET status = 'done', completed_at = ?
                    WHERE branch_id = ?
                """, (int(time.time()), branch_id))
                active = self._conn.execute(
                    "SELECT 1 FROM active_branches WHERE branch_id = ?",
                    (branch_id,)).fetchone()
                if active is None:
                    self._resolve_branch_memberships(branch_id)
                self._conn.execute("COMMIT")
                return True
            self._conn.execute("""
                UPDATE pending_branches
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL
                WHERE branch_id = ? AND status != 'done'
            """, (branch_id,))
            self._conn.execute("COMMIT")
            return False
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def branch_done_candidates(self, branch_key) -> int:
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return 0
        return self._conn.execute(
            "SELECT COUNT(*) FROM candidate_claims WHERE branch_id = ? AND done = 1",
            (branch_id,)).fetchone()[0]

    def branch_bulk_done_candidates(self, branch_key) -> int:
        """Return the legacy combined count completed by ERD pruning."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return 0
        row = self._conn.execute(
            "SELECT bulk_done_candidates FROM active_branches "
            "WHERE branch_id = ?", (branch_id,)).fetchone()
        return 0 if row is None else row[0]

    def branch_erd_pruned_candidate_counts(self, branch_key) -> tuple[int, int]:
        """Return one-level and two-level ERD-pruned candidate counts."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return 0, 0
        row = self._conn.execute("""
            SELECT one_level_erd_pruned_candidates,
                   two_level_erd_pruned_candidates
            FROM active_branches WHERE branch_id = ?
        """, (branch_id,)).fetchone()
        return (0, 0) if row is None else (row[0], row[1])

    def candidate_progress_by_branch_keys(self, branch_keys: list[bytes]) -> dict:
        """Return evaluated and ERD-pruned candidate counts by branch."""
        if not branch_keys:
            return {}
        progress = {
            bytes(branch_key): {
                "completed_candidate_count": 0,
                "bulk_completed_candidate_count": 0,
                "one_level_erd_pruned_candidate_count": 0,
                "two_level_erd_pruned_candidate_count": 0,
            }
            for branch_key in branch_keys
        }
        placeholders = ",".join("?" for _ in branch_keys)
        completed_rows = self._conn.execute(
            f"""SELECT b.branch_key, COUNT(*) AS completed_candidate_count
                FROM candidate_claims c
                JOIN branches b ON b.branch_id = c.branch_id
                WHERE c.done = 1 AND b.branch_key IN ({placeholders})
                GROUP BY c.branch_id""",
            branch_keys,
        ).fetchall()
        for row in completed_rows:
            progress[bytes(row["branch_key"])]["completed_candidate_count"] = (
                row["completed_candidate_count"]
            )
        prune_rows = self._conn.execute(
            f"""SELECT b.branch_key, a.bulk_done_candidates,
                       a.one_level_erd_pruned_candidates,
                       a.two_level_erd_pruned_candidates
                FROM active_branches a
                JOIN branches b ON b.branch_id = a.branch_id
                WHERE b.branch_key IN ({placeholders})""",
            branch_keys,
        ).fetchall()
        for row in prune_rows:
            branch_progress = progress[bytes(row["branch_key"])]
            branch_progress["bulk_completed_candidate_count"] = (
                row["bulk_done_candidates"])
            branch_progress["one_level_erd_pruned_candidate_count"] = (
                row["one_level_erd_pruned_candidates"])
            branch_progress["two_level_erd_pruned_candidate_count"] = (
                row["two_level_erd_pruned_candidates"])
        return progress

    def try_finalize_branch(self, branch_key) -> bool:
        """Atomically transition a branch open -> finalized, exactly once.

        Returns True only for the single caller that wins the transition; that
        caller writes the branch's ERD entry to the persistent cache and then
        calls delete_branch.  Caller must have confirmed all candidates are done
        first; the WHERE status='open' guard makes the finalize idempotent.
        """
        now = int(time.time())
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        cur = self._conn.execute("""
            UPDATE active_branches SET status = 'finalized', finalized_at = ?
            WHERE branch_id = ? AND status = 'open'
        """, (now, branch_id))
        return cur.rowcount == 1

    def reclaim_stale_finalize(self, branch_key, timeout_seconds: int) -> bool:
        """Reopen a branch whose finalizer died mid-finalize.

        The worker that wins try_finalize_branch completes the cache write and
        delete_branch within milliseconds; a row still status='finalized'
        after timeout_seconds has no live finalizer, and without intervention
        every waiting sibling spins on it forever.  Returns True if this call
        reopened the row (the caller should then re-run maybe_finalize)."""
        cutoff = int(time.time()) - timeout_seconds
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        cur = self._conn.execute("""
            UPDATE active_branches SET status = 'open', finalized_at = NULL
            WHERE branch_id = ? AND status = 'finalized' AND finalized_at < ?
        """, (branch_id, cutoff))
        return cur.rowcount == 1

    def delete_branch(self, branch_key):
        """Remove a finished branch and its claim/packer rows to bound the queue DB.

        finalize_bundle_stats already clears bundle_stats before this is
        called; candidate_republish is cleared here alongside candidate_claims
        since neither is meaningful once the branch is gone.
        """
        # The branches registry row is intentionally left in place: it is
        # append-only, so branch_id stays stable if this branch is ever
        # re-promoted, and the blob it holds once is negligible.
        self._conn.execute("BEGIN")
        try:
            branch_id = self._intern_branch(branch_key)
            n_claims = n_republish = 0
            completed_words = []
            if branch_id is not None:
                self._conn.execute(
                    "DELETE FROM candidate_claims WHERE branch_id = ?", (branch_id,))
                n_claims = self._conn.execute("SELECT changes()").fetchone()[0]
                self._conn.execute(
                    "DELETE FROM candidate_republish WHERE branch_id = ?",
                    (branch_id,))
                n_republish = self._conn.execute("SELECT changes()").fetchone()[0]
                self._conn.execute(
                    "DELETE FROM candidate_holes WHERE branch_id = ?",
                    (branch_id,))
            self._conn.execute(
                "DELETE FROM telemetry.bundle_stats WHERE branch_key = ?",
                (branch_key,))
            if branch_id is not None:
                self._conn.execute(
                    "DELETE FROM active_branches WHERE branch_id = ?", (branch_id,))
                unresolved_pending = self._conn.execute("""
                    SELECT 1 FROM pending_branches
                    WHERE branch_id = ? AND status IN ('pending', 'in_progress')
                """, (branch_id,)).fetchone()
                if unresolved_pending is None:
                    completed_words = self._resolve_branch_memberships(branch_id)
            self._conn.execute("COMMIT")
            self._tally_wal_traffic(
                'candidate_claims/delete-branch', n_claims,
                n_claims * _CLAIM_ROW_WAL_BYTES)
            self._tally_wal_traffic(
                'candidate_republish/delete-branch', n_republish,
                n_republish * _CLAIM_ROW_WAL_BYTES)
            return completed_words
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

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
        stale_where = """
            done = 0
              AND claimed_at < ?
              AND claimed_by NOT IN (
                  SELECT worker_id FROM worker_heartbeat
                  WHERE updated_at >= ?
              )
        """
        return self._free_claims_as_holes(
            stale_where, (age_floor, hb_cutoff),
            'candidate_holes/reclaim-stale', 'candidate_claims/reclaim-stale')

    def reclaim_claims_of_worker(self, worker_id: str) -> int:
        """Free all in-flight (done=0) candidate claims held by a specific worker.

        Called by the supervisor when it kills/respawns a worker, so that
        instance's claims are freed deterministically BEFORE a replacement of
        the same name starts heartbeating (which would otherwise make the dead
        instance's claims look live again).  done=1 rows are never touched.
        """
        return self._free_claims_as_holes(
            "done = 0 AND claimed_by = ?", (worker_id,),
            'candidate_holes/reclaim-worker',
            'candidate_claims/reclaim-worker')

    def branches_in_progress(self, opener_work_id=None):
        """Open branches ordered by effective priority then answer count."""
        if opener_work_id is None:
            return self._conn.execute("""
                SELECT row.* FROM active_branch_owner_rows AS row
                WHERE row.opener_work_id IS NULL
                   OR row.opener_work_id = (
                       SELECT selected.opener_work_id
                       FROM active_branch_owner_rows AS selected
                       WHERE selected.branch_id = row.branch_id
                         AND selected.opener_work_id IS NOT NULL
                       ORDER BY selected.owner_priority DESC,
                                selected.opener_work_id
                       LIMIT 1
                   )
                ORDER BY row.owner_priority DESC, row.n_words DESC
            """).fetchall()
        return self._conn.execute("""
            SELECT * FROM active_branch_owner_rows
            WHERE opener_work_id = ?
            ORDER BY owner_priority DESC, n_words DESC
        """, (opener_work_id,)).fetchall()

    def owner_row_for_branch(self, branch_key):
        """Return the canonical owner row selected for one open branch."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        return self._conn.execute("""
            SELECT row.* FROM active_branch_owner_rows AS row
            WHERE row.branch_id = ?
              AND (row.opener_work_id IS NULL
                   OR row.opener_work_id = (
                       SELECT selected.opener_work_id
                       FROM active_branch_owner_rows AS selected
                       WHERE selected.branch_id = row.branch_id
                         AND selected.opener_work_id IS NOT NULL
                       ORDER BY selected.owner_priority DESC,
                                selected.opener_work_id
                       LIMIT 1
                   ))
            LIMIT 1
        """, (branch_id,)).fetchone()

    def direct_branches_in_progress(self):
        """Open branches that have no opener-work ownership."""
        return self._conn.execute("""
            SELECT * FROM active_branch_owner_rows
            WHERE opener_work_id IS NULL
            ORDER BY owner_priority DESC, n_words DESC
        """).fetchall()

    def recover_active_branches(self):
        """Free stale in-flight claims after a restart; completed work survives.

        Every active_branches row — user-queued and cooperative alike — is
        preserved along with its done=1 claims, republish rows, and bundle
        telemetry: register_branch()'s INSERT OR IGNORE means the next worker
        to promote or join the branch adopts the surviving row and continues
        from its pack_cursor, bulk_done_bound, and best_erd, all of which only
        tighten monotonically and so are sound to resume from.  Only done=0
        claims — work that was in flight in a process the restart killed —
        are freed, making those candidates reclaimable as gaps.

        User-queued branches (those with a pending_branches row) additionally
        had their pending row flipped back to 'pending' by
        reset_stale_in_progress(), so a worker re-promotes them; the
        re-promotion joins the preserved row rather than starting over.

        cut_results also survives untouched: each row is a proof, not
        in-flight state.

        Returns (n_branches_resumed, n_claims_freed).
        """
        # A row stuck in status='finalized' had its finalizer killed between
        # winning try_finalize_branch and completing the cache write + delete.
        # No live worker can ever finish it — try_finalize_branch refuses
        # non-'open' rows, so every visitor spins on it forever.  Reopen it;
        # its claims and meta are intact, so the next visitor re-runs the
        # finalize from where the dead worker left off.
        self._conn.execute(
            "UPDATE active_branches SET status = 'open', finalized_at = NULL "
            "WHERE status = 'finalized'")
        n_branches_resumed = self._conn.execute(
            "SELECT COUNT(*) FROM active_branches").fetchone()[0]
        # cut_results rows are left alone: a restart kills every *waiter*, but
        # a cut bound is also a proof ("true ERD >= bound"), and a proof stays
        # true across a restart exactly like a score-cache loss, which is also
        # not cleared here.  A stale row cannot short-circuit a ceilinged solve
        # against a ceiling nobody asked for — a consumer only reuses a row
        # whose recorded budget and bound satisfy its own budget and ceiling
        # (see read_cut_result); it never blocks a solve, only saves one.
        freed = self._free_claims_as_holes(
            "done = 0", (), 'candidate_holes/recover-restart',
            'candidate_claims/recover-restart')
        return n_branches_resumed, freed

    def worker_counts_by_branch(
        self, timeout_seconds: int = WORKER_LIVENESS_SECONDS
    ) -> dict:
        """{branch_key bytes: number of recent workers on it} for status."""
        cutoff = int(time.time()) - timeout_seconds
        rows = self._conn.execute("""
            SELECT b.branch_key AS k, COUNT(*) AS c
            FROM worker_heartbeat h
            JOIN branches b ON b.branch_id = h.current_branch_id
            WHERE h.current_branch_id IS NOT NULL AND h.updated_at > ?
            GROUP BY h.current_branch_id
        """, (cutoff,)).fetchall()
        return {bytes(r["k"]): r["c"] for r in rows}

    def claim_holders_by_branch(self, exclude_worker_id=None) -> dict:
        """{branch_key bytes: workers other than exclude_worker_id holding
        unfinished claims on it}, for work selection.

        Branch occupancy.  An unfinished claim IS a worker on the branch: the
        row is written by the same transaction that hands out the bundle, so a
        branch shows as taken the instant it is taken, and the worker's own
        rows are excluded so it never reads itself as a rival.

        Occupancy needs no liveness test of its own.  A crashed worker's
        done = 0 rows are freed by reclaim_stale_claims, reclaim_claims_of_
        worker on a supervised respawn, and recover_active_branches at
        restart — so a branch nobody is working drains to zero holders and
        becomes claimable again through the existing path.  A branch with no
        unfinished claims is absent from the result rather than present with a
        count of zero.
        """
        rows = self._conn.execute("""
            SELECT branch.branch_key AS k,
                   COUNT(DISTINCT claim.claimed_by) AS c
            FROM candidate_claims AS claim
            JOIN branches AS branch ON branch.branch_id = claim.branch_id
            WHERE claim.done = 0 AND claim.claimed_by IS NOT ?
            GROUP BY claim.branch_id
        """, (exclude_worker_id,)).fetchall()
        return {bytes(r["k"]): r["c"] for r in rows}

    def _other_claim_holders(self, branch_id, worker_id):
        """claim_holders_by_branch for one branch, inside a claim transaction."""
        return self._conn.execute("""
            SELECT COUNT(DISTINCT claimed_by) FROM candidate_claims
            WHERE branch_id = ? AND done = 0 AND claimed_by IS NOT ?
        """, (branch_id, worker_id)).fetchone()[0]

    def _branch_worker_count(self, branch_id, worker_id, now):
        """Count live workers assigned to a branch, including the recorder."""
        if worker_id is None:
            return None
        row = self._conn.execute("""
            SELECT COUNT(*) AS worker_count,
                   MAX(worker_id = ?) AS recorder_is_counted
            FROM worker_heartbeat
            WHERE current_branch_id = ? AND updated_at > ?
        """, (worker_id, branch_id, now - WORKER_LIVENESS_SECONDS)).fetchone()
        return row["worker_count"] + (0 if row["recorder_is_counted"] else 1)

    def _branch_row_dict(self, pending=None, active=None) -> dict:
        """Normalize pending/active branch state for queue visibility commands."""
        p = pending
        a = active
        branch_key = bytes((a or p)["branch_key"])
        claims_done = self.branch_done_candidates(branch_key) if a else 0
        workers = self.worker_counts_by_branch().get(branch_key, 0)
        return {
            "branch_key": branch_key,
            "branch_key_hex": branch_key.hex(),
            "kind": "user" if p is not None else "coop",
            "status": (p["status"] if p is not None
                       else (a["status"] if a is not None else "unknown")),
            "priority": (a["effective_priority"] if a is not None
                         else p["priority"]),
            "n_words": (a["n_words"] if a is not None else p["n_words"]),
            "budget": a["budget"] if a is not None else None,
            "done_candidates": claims_done,
            "n_candidates": a["n_candidates"] if a is not None else None,
            "worker_count": workers,
            "nodes_spent": a["nodes_spent"] if a is not None else 0,
            "best_guess": a["best_guess"] if a is not None else None,
            "best_erd": a["best_erd"] if a is not None else None,
            "best_max_depth": a["best_max_depth"] if a is not None else None,
            "tainted": bool(a["tainted"]) if a is not None else False,
            "opener": ((a["opener"] if a is not None else None)
                            or (p["opener"] if p is not None else None)),
            "opener_pattern": ((a["opener_pattern"] if a is not None else None)
                               if a is not None and a["opener_pattern"] is not None
                               else (p["opener_pattern"] if p is not None else None)),
            "opener_pattern_text": (
                fmt_pattern(a["opener_pattern"])
                if a is not None and a["opener_pattern"] is not None
                else (fmt_pattern(p["opener_pattern"])
                      if p is not None and p["opener_pattern"] is not None
                      else None)),
            "spine": a["spine"] if a is not None else None,
            "created_at": a["created_at"] if a is not None else None,
            "updated_at": ((p["completed_at"] or p["claimed_at"])
                           if p is not None else a["created_at"]),
            "claimed_by": p["claimed_by"] if p is not None else None,
            "claimed_at": p["claimed_at"] if p is not None else None,
            "completed_at": p["completed_at"] if p is not None else None,
        }

    def list_queue_rows(self, filters=None, sort=None, limit=None):
        """Return normalized pending/user and active/cooperative queue rows."""
        filters = filters or {}
        pending = {
            bytes(r["branch_key"]): r
            for r in self._conn.execute(
                "SELECT p.*, b.branch_key FROM pending_branches p "
                "JOIN branches b ON b.branch_id = p.branch_id")
        }
        active = {
            bytes(r["branch_key"]): r
            for r in self._conn.execute(
                f"""SELECT a.*, b.branch_key,
                           CASE WHEN a.status = 'open' THEN COALESCE(
                               (SELECT MAX(owner_priority)
                                FROM live_branch_opener_rows
                                WHERE branch_id = a.branch_id),
                               CASE WHEN a.priority >= {LEGACY_PROMOTED_PRIORITY_MIN}
                                    THEN 0 ELSE a.priority END
                           ) ELSE a.priority END AS effective_priority
                    FROM active_branches a
                    JOIN branches b ON b.branch_id = a.branch_id""")
        }
        keys = set(pending) | set(active)
        rows = [self._branch_row_dict(pending.get(k), active.get(k))
                for k in keys]

        def ok(row):
            if filters.get("status") and row["status"] != filters["status"]:
                return False
            if filters.get("min_words") is not None and row["n_words"] < filters["min_words"]:
                return False
            if filters.get("max_words") is not None and row["n_words"] > filters["max_words"]:
                return False
            if filters.get("budget") is not None and row["budget"] != filters["budget"]:
                return False
            if filters.get("priority") is not None and row["priority"] != filters["priority"]:
                return False
            if filters.get("opener"):
                if (row["opener"] or "").lower() != filters["opener"].lower():
                    return False
            prefix = filters.get("prefix")
            if prefix and not self._row_matches_spine_prefix(row, prefix):
                return False
            return True

        rows = [r for r in rows if ok(r)]
        rows.sort(key=self._queue_sort_key(sort))
        if limit is not None:
            rows = rows[:limit]
        return rows

    @staticmethod
    def _report_filter_value(filters, name, default=None):
        if filters is None:
            return default
        if isinstance(filters, dict):
            return filters.get(name, default)
        return getattr(filters, name, default)

    def report_queue_rows(
        self, filters=None, sort=None, limit=None, generated_at=None,
    ) -> dict:
        """Return normalized, filtered queue rows with pre-limit summaries."""
        branch_statuses = tuple(self._report_filter_value(
            filters, "branch_statuses", ()
        ))
        branch_worker_statuses = tuple(self._report_filter_value(
            filters, "branch_worker_statuses", ()
        ))
        minimum_answer_count = self._report_filter_value(
            filters, "minimum_answer_count"
        )
        maximum_answer_count = self._report_filter_value(
            filters, "maximum_answer_count"
        )
        budget = self._report_filter_value(filters, "budget")
        priority = self._report_filter_value(filters, "priority")
        spine_prefix = self._report_filter_value(filters, "spine_prefix")
        opener = self._report_filter_value(filters, "opener")
        branch_key = self._report_filter_value(filters, "branch_key")
        branch_status_expression = """CASE
            WHEN pending_status = 'done' THEN 'done'
            WHEN active_status = 'finalized' THEN 'finalizing'
            WHEN active_status = 'open' OR pending_status = 'in_progress'
                THEN 'evaluating'
            WHEN pending_status = 'pending' THEN 'queued'
            ELSE 'unqueued' END"""
        branch_worker_status_expression = """CASE
            WHEN pending_status = 'done' THEN NULL
            WHEN active_status = 'finalized'
              OR active_status = 'open' OR pending_status = 'in_progress'
                THEN CASE WHEN worker_count > 0 THEN 'active' ELSE 'waiting' END
            ELSE NULL END"""
        active_only = branch_worker_statuses == ("active",)
        completed_query = """
                SELECT branch_id, COUNT(*) AS completed_candidate_count
                FROM candidate_claims WHERE done = 1 GROUP BY branch_id
        """
        if active_only:
            completed_query = """
                SELECT branch_id, COUNT(*) AS completed_candidate_count
                FROM candidate_claims
                WHERE done = 1 AND branch_id IN (
                    SELECT branch_id FROM workers
                )
                GROUP BY branch_id
            """
        base_query = f"""
            WITH branch_ids AS (
                SELECT branch_id FROM pending_branches
                UNION
                SELECT branch_id FROM active_branches
            ),
            workers AS (
                SELECT current_branch_id AS branch_id, COUNT(*) AS worker_count
                FROM worker_heartbeat
                WHERE current_branch_id IS NOT NULL AND updated_at >= ?
                GROUP BY current_branch_id
            ),
            completed AS (
                {completed_query}
            ),
            joined_rows AS (
                SELECT registry.branch_key,
                       pending.status AS pending_status,
                       active.status AS active_status,
                       pending.priority AS pending_priority,
                       CASE WHEN active.status = 'open' THEN COALESCE(
                           (SELECT MAX(owner_priority)
                            FROM live_branch_opener_rows
                            WHERE branch_id = active.branch_id),
                           CASE WHEN active.priority >= {LEGACY_PROMOTED_PRIORITY_MIN}
                                THEN 0 ELSE active.priority END
                       ) ELSE active.priority END AS active_priority,
                       pending.n_words AS pending_answer_count,
                       active.n_words AS active_answer_count,
                       pending.opener AS pending_opener,
                       active.opener AS active_opener,
                       pending.opener_pattern AS pending_opener_pattern,
                       active.opener_pattern AS active_opener_pattern,
                       pending.claimed_at,
                       pending.completed_at,
                       active.n_candidates AS candidate_count,
                       active.budget,
                       active.best_guess,
                       active.best_erd,
                       active.best_max_depth,
                       active.nodes_spent,
                       active.bulk_done_candidates,
                       active.one_level_erd_pruned_candidates,
                       active.two_level_erd_pruned_candidates,
                       active.ceiling,
                       active.spine,
                       active.created_at,
                       COALESCE(completed.completed_candidate_count, 0)
                           AS completed_candidate_count,
                       COALESCE(workers.worker_count, 0) AS worker_count
                FROM branch_ids AS keys
                JOIN branches AS registry USING (branch_id)
                LEFT JOIN pending_branches AS pending USING (branch_id)
                LEFT JOIN active_branches AS active USING (branch_id)
                LEFT JOIN completed USING (branch_id)
                LEFT JOIN workers USING (branch_id)
            ),
            normalized AS (
                SELECT *,
                       {branch_status_expression} AS branch_status,
                       {branch_worker_status_expression} AS branch_worker_status,
                       COALESCE(active_priority, pending_priority, 0) AS priority,
                       COALESCE(active_answer_count, pending_answer_count) AS answer_count,
                       COALESCE(active_opener, pending_opener) AS opener,
                       COALESCE(active_opener_pattern, pending_opener_pattern) AS opener_pattern
                FROM joined_rows
            )
        """
        conditions = []
        parameters = [
            int(generated_at if generated_at is not None else time.time())
            - WORKER_LIVENESS_SECONDS
        ]
        if branch_statuses:
            placeholders = ",".join("?" for _ in branch_statuses)
            conditions.append(f"branch_status IN ({placeholders})")
            parameters.extend(branch_statuses)
        if branch_worker_statuses:
            placeholders = ",".join("?" for _ in branch_worker_statuses)
            conditions.append(f"branch_worker_status IN ({placeholders})")
            parameters.extend(branch_worker_statuses)
        if minimum_answer_count is not None:
            conditions.append("answer_count >= ?")
            parameters.append(minimum_answer_count)
        if maximum_answer_count is not None:
            conditions.append("answer_count <= ?")
            parameters.append(maximum_answer_count)
        if budget is not None:
            conditions.append("budget = ?")
            parameters.append(budget)
        if priority is not None:
            conditions.append("priority = ?")
            parameters.append(priority)
        scope_conditions = []
        scope_parameters = []
        if spine_prefix:
            scope_conditions.append("(spine = ? OR spine LIKE ?)")
            scope_parameters.extend((spine_prefix, spine_prefix + " %"))
        if opener:
            scope_conditions.append(
                "(spine IS NULL AND LOWER(opener) = ?)"
            )
            scope_parameters.append(opener.lower())
        if branch_key is not None:
            scope_conditions.append("branch_key = ?")
            scope_parameters.append(bytes(branch_key))
        if scope_conditions:
            conditions.append("(" + " OR ".join(scope_conditions) + ")")
            parameters.extend(scope_parameters)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        summary_rows = self._conn.execute(
            base_query
            + " SELECT branch_status, branch_worker_status, COUNT(*) AS branch_count"
            + " FROM normalized"
            + where_clause
            + " GROUP BY branch_status, branch_worker_status",
            parameters,
        ).fetchall()
        by_status = {}
        by_worker_status = {}
        for row in summary_rows:
            by_status[row["branch_status"]] = (
                by_status.get(row["branch_status"], 0) + row["branch_count"]
            )
            by_worker_status[row["branch_worker_status"]] = (
                by_worker_status.get(row["branch_worker_status"], 0) + row["branch_count"]
            )
        matched_rows = sum(by_status.values())
        sort_name = sort or self._report_filter_value(filters, "sort") or "default"
        order_by = {
            "age": "COALESCE(created_at, claimed_at, 0), branch_key",
            "size": "answer_count DESC, branch_key",
            "workers": "worker_count DESC, answer_count DESC, branch_key",
            "priority": "priority DESC, answer_count DESC, branch_key",
            "nodes": "nodes_spent DESC, answer_count DESC, branch_key",
            "slowest": "nodes_spent DESC, created_at, branch_key",
            "default": (
                "CASE branch_worker_status WHEN 'active' THEN 0 ELSE 1 END, "
                "priority DESC, answer_count DESC, branch_key"
            ),
        }[sort_name]
        effective_limit = (
            limit if limit is not None
            else self._report_filter_value(filters, "limit")
        )
        row_query = base_query + " SELECT * FROM normalized" + where_clause
        row_query += " ORDER BY " + order_by
        row_parameters = list(parameters)
        if effective_limit is not None:
            row_query += " LIMIT ?"
            row_parameters.append(effective_limit)
        joined_rows = self._conn.execute(row_query, row_parameters).fetchall()
        returned_rows = []
        for row in joined_rows:
            opener_pattern = row["opener_pattern"]
            returned_rows.append({
                "branch_key": bytes(row["branch_key"]),
                "branch_key_hex": bytes(row["branch_key"]).hex(),
                "branch_status": row["branch_status"],
                "branch_worker_status": row["branch_worker_status"],
                "raw_status": row["pending_status"] or row["active_status"],
                "is_cooperative": row["pending_status"] is None,
                "answer_count": row["answer_count"],
                "priority": row["priority"],
                "budget": row["budget"],
                "candidate_count": row["candidate_count"],
                "completed_candidate_count": row["completed_candidate_count"],
                "bulk_completed_candidate_count": row["bulk_done_candidates"] or 0,
                "one_level_erd_pruned_candidate_count":
                    row["one_level_erd_pruned_candidates"] or 0,
                "two_level_erd_pruned_candidate_count":
                    row["two_level_erd_pruned_candidates"] or 0,
                "worker_count": row["worker_count"],
                "search_node_count": row["nodes_spent"] or 0,
                "ceiling": row["ceiling"],
                "best_guess": row["best_guess"],
                "best_erd": row["best_erd"],
                "best_max_remaining_depth": row["best_max_depth"],
                "opener": row["opener"],
                "opener_pattern": (
                    fmt_pattern(opener_pattern) if opener_pattern is not None else None
                ),
                "spine": row["spine"],
                "created_at": row["created_at"],
                "updated_at": row["completed_at"] or row["claimed_at"] or row["created_at"],
                "is_context": False,
            })
        return {
            "summary": {
                "branch_count": matched_rows,
                "branch_count_by_status": by_status,
                "branch_count_by_worker_status": by_worker_status,
            },
            "matched_rows": matched_rows,
            "rows": returned_rows,
        }

    def report_tree_rows(self, spine_prefix, filters=None, sort=None, limit=None) -> dict:
        """Return report queue rows at or below a recorded spine prefix."""
        scoped_filters = {
            "branch_statuses": self._report_filter_value(
                filters, "branch_statuses", ()
            ) or (),
            "branch_worker_statuses": self._report_filter_value(
                filters, "branch_worker_statuses", ()
            ) or (),
            "minimum_answer_count": self._report_filter_value(
                filters, "minimum_answer_count"
            ),
            "maximum_answer_count": self._report_filter_value(
                filters, "maximum_answer_count"
            ),
            "budget": self._report_filter_value(filters, "budget"),
            "priority": self._report_filter_value(filters, "priority"),
            "sort": self._report_filter_value(filters, "sort"),
            "limit": self._report_filter_value(filters, "limit"),
            "spine_prefix": (spine_prefix or "").strip(),
        }
        return self.report_queue_rows(scoped_filters, sort, limit)

    def _queue_sort_key(self, sort):
        if sort == "nodes":
            return lambda r: (-(r["nodes_spent"] or 0), -r["n_words"], r["branch_key_hex"])
        if sort == "age":
            return lambda r: ((r["created_at"] or r["claimed_at"] or 0), r["branch_key_hex"])
        if sort == "size":
            return lambda r: (-r["n_words"], r["branch_key_hex"])
        if sort == "workers":
            return lambda r: (-r["worker_count"], -r["n_words"], r["branch_key_hex"])
        if sort == "priority":
            return lambda r: (-(r["priority"] or 0), -r["n_words"], r["branch_key_hex"])
        if sort == "slowest":
            return lambda r: (-(r["nodes_spent"] or 0), (r["created_at"] or 0),
                              r["branch_key_hex"])
        return lambda r: (0 if r["status"] in ("in_progress", "open") else 1,
                          -(r["priority"] or 0), -r["n_words"], r["branch_key_hex"])

    @staticmethod
    def row_spine_text(row):
        """Return the most specific recorded spine text for a queue row."""
        if row.get("spine"):
            return row["spine"]
        if row.get("opener") and row.get("opener_pattern_text"):
            return f'{row["opener"].upper()} {row["opener_pattern_text"]}'
        return ""

    def _row_matches_spine_prefix(self, row, prefix):
        spine = row.get("spine") or ""
        if spine == prefix or spine.startswith(prefix + " "):
            return True
        fallback = ""
        if row.get("opener") and row.get("opener_pattern_text"):
            fallback = f'{row["opener"].upper()} {row["opener_pattern_text"]}'
        elif row.get("opener"):
            fallback = row["opener"].upper()
        return fallback == prefix or fallback.startswith(prefix + " ")

    def queue_dashboard(self, limit=8):
        active = [r for r in self.list_queue_rows()
                  if r["status"] in ("in_progress", "open")]
        active.sort(key=self._queue_sort_key(None))
        return {
            "summary": self.queue_summary(),
            "active": active[:limit],
            "pending": self.list_queue_rows(
                {"status": "pending"}, limit=limit),
            "stale": [],
        }

    def queue_tree_rows(self, prefix=None, active_only=False, max_depth=None, limit=None):
        filters = {"prefix": prefix} if prefix else {}
        rows = self.list_queue_rows(filters)
        if active_only:
            rows = [r for r in rows if r["status"] in ("in_progress", "open")]
        if max_depth is not None:
            rows = [r for r in rows
                    if guess_depth_from_spine(r["spine"]) <= max_depth
                    or (not r["spine"] and r["opener"])]
        rows.sort(key=lambda r: (r["spine"] or r["opener"] or "",
                                 r["branch_key_hex"]))
        if limit is not None:
            rows = rows[:limit]
        return rows

    def queue_top_rows(self, sort="nodes", limit=10, prefix=None):
        rows = self.list_queue_rows({"prefix": prefix} if prefix else {}, sort=sort)
        rows = [r for r in rows if r["status"] in ("in_progress", "open")]
        return rows[:limit]

    def queue_summary(self):
        rows = self.list_queue_rows()
        by_status = {}
        by_kind = {}
        by_budget = {}
        by_priority = {"0": 0, "1-999": 0, "coop": 0}
        by_size = {"2-9": 0, "10-99": 0, "100-999": 0, "1000+": 0}
        for r in rows:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
            b = "none" if r["budget"] is None else str(r["budget"])
            by_budget[b] = by_budget.get(b, 0) + 1
            pri = r["priority"] or 0
            if pri >= LEGACY_PROMOTED_PRIORITY_MIN:
                by_priority["coop"] += 1
            elif pri == 0:
                by_priority["0"] += 1
            else:
                by_priority["1-999"] += 1
            n = r["n_words"]
            if n < 10:
                by_size["2-9"] += 1
            elif n < 100:
                by_size["10-99"] += 1
            elif n < 1000:
                by_size["100-999"] += 1
            else:
                by_size["1000+"] += 1
        pending = [r for r in rows if r["status"] == "pending"]
        active = [r for r in rows if r["status"] in ("in_progress", "open")]
        oldest = lambda xs: min(xs, key=lambda r: r["created_at"] or r["claimed_at"] or 0) if xs else None
        largest = lambda xs: max(xs, key=lambda r: r["n_words"]) if xs else None
        return {
            "total": len(rows),
            "by_status": by_status,
            "by_kind": by_kind,
            "by_budget": by_budget,
            "by_priority": by_priority,
            "by_size": by_size,
            "largest_pending": largest(pending),
            "oldest_pending": oldest(pending),
            "largest_active": largest(active),
            "oldest_active": oldest(active),
            "stale_active": [],
        }

    def resolve_branch_ref(self, ref):
        """Resolve a short id, full hex key, or normalized spine prefix."""
        rows = self.list_queue_rows()
        ref = (ref or "").strip()
        if not ref:
            return []
        low = ref.lower()
        matches = [
            r for r in rows
            if r["branch_key_hex"] == low or r["branch_key_hex"].startswith(low)
            or hashlib.sha1(r["branch_key"]).hexdigest()[:4] == low
        ]
        if matches:
            return matches
        return [r for r in rows if self._row_matches_spine_prefix(r, ref)]

    def branch_rows_for_reference_prefix(self, digest_prefix) -> list[dict]:
        """Return queue rows whose stable SHA-1 reference starts with a prefix."""
        normalized_prefix = digest_prefix.lower()
        return [
            row for row in self.list_queue_rows()
            if hashlib.sha1(bytes(row["branch_key"])).hexdigest().startswith(
                normalized_prefix
            )
        ]

    def candidate_republish_for_branch(self, branch_key) -> list[dict]:
        """Return sparse candidate republish counts for one branch."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return []
        return [dict(row) for row in self._conn.execute(
            "SELECT idx, count FROM candidate_republish "
            "WHERE branch_id = ? ORDER BY idx",
            (branch_id,),
        )]

    def branch_detail(self, branch_key, include_claims=False):
        p = self.get_pending_branch(branch_key)
        a = self.get_active_branch(branch_key)
        if p is None and a is None:
            return None
        detail = self._branch_row_dict(p, a)
        if include_claims:
            detail["claims"] = [dict(r) for r in self.claims_for_branch(branch_key)]
        else:
            detail["claims"] = []
        detail["bundle_stats"] = [dict(r) for r in self._conn.execute(
            "SELECT * FROM telemetry.bundle_stats "
            "WHERE branch_key = ? ORDER BY bundle_id",
            (branch_key,))]
        republish_id = self._intern_branch(branch_key)
        detail["republish"] = [] if republish_id is None else [
            dict(r) for r in self._conn.execute(
                "SELECT * FROM candidate_republish WHERE branch_id = ? "
                "ORDER BY idx", (republish_id,))]
        detail["finalize_log"] = [dict(r) for r in self._conn.execute(
            "SELECT * FROM telemetry.branch_finalize_log "
            "WHERE branch_key = ? ORDER BY id DESC LIMIT 5",
            (branch_key,))]
        workers_id = self._intern_branch(branch_key)
        detail["workers"] = [] if workers_id is None else [
            dict(r) for r in self._conn.execute(
                "SELECT * FROM worker_heartbeat WHERE current_branch_id = ? "
                "ORDER BY worker_id", (workers_id,))]
        return detail


    @staticmethod
    def _report_finalization_outcome(row):
        outcome = row["outcome"]
        if outcome == "cut" and row["budget"] is not None and row["ceiling"] is not None \
                and row["ceiling"] > row["budget"]:
            return "loss"
        if outcome in ("exact", "cut", "loss"):
            return outcome
        return "cut" if row["ceiling"] is not None else "unknown"

    @classmethod
    def _report_loss_proof(cls, row):
        if cls._report_finalization_outcome(row) != "loss":
            return None
        if row["budget"] is not None and row["ceiling"] is not None \
                and row["ceiling"] > row["budget"]:
            return "ceiling_above_budget"
        return "candidate_infeasibility"

    def recent_finalized_branches(self, earliest_finalized_at: int):
        """Return finalizations whose completed cards may still be displayed."""
        return [dict(row) for row in self._conn.execute("""
            SELECT * FROM telemetry.branch_finalize_log
            WHERE finalized_at IS NOT NULL AND finalized_at > ?
            ORDER BY finalized_at DESC, id DESC
        """, (earliest_finalized_at,))]

    def report_branch_telemetry(self, branch_key, limit, after=None, before=None) -> dict:
        """Return bounded current and historical telemetry for one branch.

        recent_finalizations is newest-first and keyset-paginated: at most one
        of after/before is a (recorded_at, id) boundary taken from a row
        already seen (after: strictly older, continuing the same newest-first
        order; before: strictly newer, fetched ascending and reversed back to
        newest-first). Unlike an OFFSET, a boundary tied to an actual row
        stays valid as new rows keep landing on a branch the swarm is still
        finalizing -- OFFSET counts from the current head, which shifts under
        it and skips or repeats rows. Neither applies to cut_reuse_misses,
        which always shows the single most recent window.
        """
        bundle_row = self._conn.execute("""
            SELECT COUNT(*) AS bundle_count,
                   SUM(nodes) AS node_count,
                   SUM(wall_millis) AS wall_millis,
                   SUM(censored) AS censored_unit_count,
                   MAX(nodes) AS maximum_bundle_node_count
            FROM telemetry.bundle_stats WHERE branch_key = ?
        """, (branch_key,)).fetchone()
        bundle_summary = None
        if bundle_row["bundle_count"]:
            bundle_summary = dict(bundle_row)
        finalization_total_count = self._conn.execute(
            "SELECT COUNT(*) FROM telemetry.branch_finalize_log "
            "WHERE branch_key = ?",
            (branch_key,)).fetchone()[0]
        if after is not None:
            after_recorded_at, after_id = after
            finalization_rows = self._conn.execute("""
                SELECT * FROM telemetry.branch_finalize_log
                WHERE branch_key = ?
                  AND (recorded_at < ? OR (recorded_at = ? AND id < ?))
                ORDER BY recorded_at DESC, id DESC LIMIT ?
            """, (branch_key, after_recorded_at, after_recorded_at, after_id,
                  limit)).fetchall()
        elif before is not None:
            before_recorded_at, before_id = before
            ascending_rows = self._conn.execute("""
                SELECT * FROM telemetry.branch_finalize_log
                WHERE branch_key = ?
                  AND (recorded_at > ? OR (recorded_at = ? AND id > ?))
                ORDER BY recorded_at ASC, id ASC LIMIT ?
            """, (branch_key, before_recorded_at, before_recorded_at, before_id,
                  limit)).fetchall()
            finalization_rows = list(reversed(ascending_rows))
        else:
            finalization_rows = self._conn.execute("""
                SELECT * FROM telemetry.branch_finalize_log
                WHERE branch_key = ?
                ORDER BY recorded_at DESC, id DESC LIMIT ?
            """, (branch_key, limit)).fetchall()
        finalizations = []
        for row in finalization_rows:
            finalizations.append({
                "finalization_id": row["id"],
                "spine": row["spine"],
                "outcome": self._report_finalization_outcome(row),
                "loss_proof": self._report_loss_proof(row),
                "ceiling": row["ceiling"],
                "best_guess": row["best_guess"],
                "best_erd": row["best_erd"],
                "budget": row["budget"],
                "answer_count": row["n_words"],
                "search_node_count": row["nodes_spent"],
                "created_at": row["created_at"],
                "finalized_at": row["finalized_at"],
                "wall_millis": row["total_bundle_wall_millis"],
                "bundle_count": row["n_bundles"],
                "maximum_bundle_node_count": row["max_bundle_nodes"],
                "censored_unit_count": row["censored_units"],
                "epoch": row["epoch"],
                "evaluated_candidate_count": row["n_claims"],
                "bulk_completed_candidate_count": row["bulk_done_candidates"],
                "one_level_erd_pruned_candidate_count":
                    row["one_level_erd_pruned_candidates"],
                "two_level_erd_pruned_candidate_count":
                    row["two_level_erd_pruned_candidates"],
                # Ranks are 1-based for display; the stored columns are
                # 0-based positions in the branch's best-first order.
                "winner_best_first_rank": best_first_rank(
                    row["winner_best_first_position"]),
                "winner_republish_count": row["winner_republish_count"],
                "candidates_completed_before_winner":
                    row["candidates_completed_before_winner"],
                "weakest_best_first_rank_before_winner": best_first_rank(
                    row["max_best_first_position_before_winner"]),
                "republished_candidate_count": row["republished_candidates"],
                "max_candidate_republish_count":
                    row["max_candidate_republish_count"],
                "recorded_at": row["recorded_at"],
            })
        cut_rows = self._conn.execute("""
            SELECT * FROM telemetry.cut_reuse_misses
            WHERE branch_key = ?
            ORDER BY recorded_at DESC, id DESC LIMIT ?
        """, (branch_key, limit)).fetchall()
        cut_reuse_misses = [{
            "cut_reuse_miss_id": row["id"],
            "answer_count": row["n_words"],
            "budget": row["budget"],
            "wanted_ceiling": row["wanted_ceiling"],
            "available_bound": row["available_bound"],
            "available_budget": row["available_budget"],
            "epoch": row["epoch"],
            "recorded_at": row["recorded_at"],
        } for row in cut_rows]
        return {
            "bundle_summary": bundle_summary,
            "recent_finalizations": finalizations,
            "finalization_total_count": finalization_total_count,
            "cut_reuse_misses": cut_reuse_misses,
        }

    def branch_candidate_eta_sample(self, branch_key, window_seconds, now=None):
        """Return recent candidate-work measurements for one branch.

        A best-ERD improvement changes which candidates survive the two-level
        bound, so that sample begins at the improvement even when it is newer
        than the trailing report window.  Without a best ERD, the sample begins
        at branch creation.  Bound-tagged scans from an older concurrent worker
        are excluded only once a current best exists.  All durations are
        aggregate worker time; the report converts them to wall time using live
        worker count.
        """
        now = int(time.time()) if now is None else now
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        try:
            branch_row = self._conn.execute("""
                SELECT best_erd, best_updated_at, created_at
                FROM active_branches
                WHERE branch_id = ? AND status = 'open'
            """, (branch_id,)).fetchone()
        except sqlite3.OperationalError:
            # Read-only report processes can briefly run before a worker opens
            # the queue and applies this additive telemetry migration.
            return None
        if branch_row is None:
            return None
        best_updated_at = branch_row["best_updated_at"]
        sample_started_at = best_updated_at or branch_row["created_at"]
        since = max(now - window_seconds, sample_started_at or now - window_seconds)
        try:
            prune_row = self._conn.execute("""
                SELECT COALESCE(SUM(inspected_candidate_count), 0)
                           AS inspected_candidate_count,
                       COALESCE(SUM(pruned_candidate_count), 0)
                           AS pruned_candidate_count,
                       COALESCE(SUM(wall_millis), 0) AS inspection_worker_millis,
                       CASE WHEN SUM(wall_millis) > 0
                            THEN SUM(branch_worker_count * wall_millis) * 1.0
                                 / SUM(wall_millis) END
                           AS inspection_worker_count,
                       MIN(branch_worker_count) AS inspection_worker_count_min,
                       MAX(branch_worker_count) AS inspection_worker_count_max,
                       SUM(CASE WHEN wall_millis > 0
                                     AND branch_worker_count IS NULL
                                THEN 1 ELSE 0 END)
                           AS inspection_unknown_worker_count
                FROM telemetry.two_level_prune_telemetry
                WHERE branch_id = ?
                  AND (bound_erd = ? OR ? IS NULL)
                  AND recorded_at >= ? AND recorded_at <= ?
            """, (branch_id, branch_row["best_erd"], branch_row["best_erd"],
                  since, now)).fetchone()
            evaluation_row = self._conn.execute("""
            SELECT COUNT(candidate_evaluation_millis) AS evaluated_candidate_count,
                   COALESCE(SUM(candidate_evaluation_millis), 0)
                       AS evaluation_worker_millis,
                   CASE WHEN SUM(candidate_evaluation_millis) > 0
                        THEN SUM(branch_worker_count * candidate_evaluation_millis) * 1.0
                             / SUM(candidate_evaluation_millis) END
                       AS evaluation_worker_count,
                   MIN(branch_worker_count) AS evaluation_worker_count_min,
                   MAX(branch_worker_count) AS evaluation_worker_count_max,
                   SUM(CASE WHEN candidate_evaluation_millis IS NOT NULL
                                 AND branch_worker_count IS NULL
                            THEN 1 ELSE 0 END)
                       AS evaluation_unknown_worker_count
                FROM telemetry.claim_telemetry
                WHERE branch_id = ?
                  AND (evaluation_bound_erd = ? OR ? IS NULL)
                  AND recorded_at >= ? AND recorded_at <= ?
            """, (branch_id, branch_row["best_erd"], branch_row["best_erd"],
                  since, now)).fetchone()
        except sqlite3.OperationalError:
            return None
        return {
            "best_updated_at": best_updated_at,
            "window_started_at": since,
            **dict(prune_row),
            **dict(evaluation_row),
        }

    def report_root_progress(self, spine_prefix, epoch, recent_window_seconds,
                             active_branch_limit=10, now=None) -> dict:
        """Per-response-group work totals for one spine, plus live rates.

        `spine_prefix` is any spine ending in a word: a bare opener such as
        `SCOPE`, or a deeper one such as `SCOPE -y--- LUBES`.  Rolls up
        telemetry.branch_finalize_log by the response pattern that follows the
        prefix, so every descendant's cost is attributed to the group it sits
        under.  With an explicit epoch, the rollup is limited to that epoch;
        otherwise it includes historical telemetry from every epoch.  The
        rollup is a scan over the selected rows
        (branch_finalize_log carries no spine index), so callers should treat
        it as a seconds-scale query.

        Branches are selected by spine prefix alone rather than by opener word.
        A opener word identifies the root a branch was requested under, which
        for a deeper spine is still the root -- selecting on it would scope the
        open branches to the wrong subtree.  active_branches holds thousands of
        rows, not millions, so scanning it costs single-digit milliseconds.

        A group is `started` once work has opened on it, which is not the same
        as having finalized anything: its first branch can still be open, with
        workers on it right now.  Open branches are therefore rolled up
        alongside finalized ones, by the same spine prefix, so a group being
        worked never reads as untouched.  They contribute a branch count and a
        creation time but no cost -- nodes and worker-time are only known at
        finalize -- so a freshly opened group shows as started with zero
        measured cost, which is the true state.

        `work_started_at` is the earliest branch *creation* across both, which
        is when the swarm first opened work under the prefix -- a different
        question from when the root was requested, which can precede it by days
        while higher-priority roots hold the queue.  A branch created before the
        selected telemetry keeps its original creation time, so this can
        predate the epoch that produced a later finalization.

        Two distinct time bases are reported per group, and they answer
        different questions:
          elapsed_millis  wall-clock span from first work to last, which the
                          swarm's other branches share
          wall_millis     summed worker-time across bundles, which counts a
                          six-worker hour as six hours
        Their ratio is a coarse read on how much parallelism the group drew.

        Progress on still-running branches comes from candidate_claims: a
        branch is `n_candidates` claims wide, and `recent_done_count` over
        `recent_window_seconds` gives the rate that turns the remainder into an
        estimate.  Claim rows carry no epoch, so the window is the only fence.
        Counting claims costs a scan per branch and a root can hold thousands
        of open branches at once, so only the `active_branch_limit` widest are
        counted -- claims are handed out widest-branch-first, so those are the
        ones carrying the work an estimate depends on.
        """
        now = int(time.time()) if now is None else now
        since = now - recent_window_seconds
        # The response pattern follows one space past the prefix.
        pattern_start = len(spine_prefix) + 2
        descendants = spine_prefix + " %"
        epoch_condition = "AND epoch = ?" if epoch is not None else ""
        group_rows = self._conn.execute(f"""
            SELECT SUBSTR(spine, ?, 5) AS pattern,
                   COUNT(*) AS branch_count,
                   SUM(nodes_spent) AS search_node_count,
                   SUM(total_bundle_wall_millis) AS wall_millis,
                   MIN(created_at) AS first_created_at,
                   MAX(finalized_at) AS last_finalized_at,
                   GROUP_CONCAT(DISTINCT epoch) AS telemetry_epochs
            FROM telemetry.branch_finalize_log
            WHERE spine LIKE ? {epoch_condition}
            GROUP BY pattern
        """, (pattern_start, descendants, *(() if epoch is None else (epoch,)))).fetchall()
        groups = {}
        telemetry_epochs = set()
        work_started_at = None
        work_latest_at = None
        for row in group_rows:
            first, last = row["first_created_at"], row["last_finalized_at"]
            group_telemetry_epochs = sorted(
                int(value) for value in row["telemetry_epochs"].split(","))
            telemetry_epochs.update(group_telemetry_epochs)
            groups[row["pattern"]] = {
                "branch_count": row["branch_count"],
                "open_branch_count": 0,
                "search_node_count": row["search_node_count"] or 0,
                "wall_millis": row["wall_millis"] or 0,
                "first_created_at": first,
                "last_finalized_at": last,
                "telemetry_epochs": group_telemetry_epochs,
                "elapsed_millis": (last - first) * 1000
                                  if first is not None and last is not None
                                  else None,
            }
            if first is not None:
                work_started_at = min(work_started_at or first, first)
            if last is not None:
                work_latest_at = max(work_latest_at or last, last)
        open_group_rows = self._conn.execute("""
            SELECT SUBSTR(spine, ?, 5) AS pattern,
                   COUNT(*) AS open_branch_count,
                   MIN(created_at) AS first_created_at
            FROM active_branches
            WHERE status = 'open' AND spine LIKE ?
            GROUP BY pattern
        """, (pattern_start, descendants)).fetchall()
        open_branch_count = 0
        for row in open_group_rows:
            open_branch_count += row["open_branch_count"]
            first = row["first_created_at"]
            group = groups.get(row["pattern"])
            if group is None:
                # Opened but nothing finalized yet: real work with no cost to
                # report.  Zeroes here are measured, not assumed absent.
                group = groups[row["pattern"]] = {
                    "branch_count": 0,
                    "open_branch_count": 0,
                    "search_node_count": 0,
                    "wall_millis": 0,
                    "first_created_at": first,
                    "last_finalized_at": None,
                    "telemetry_epochs": [],
                    "elapsed_millis": None,
                }
            group["open_branch_count"] = row["open_branch_count"]
            if first is not None:
                if group["first_created_at"] is None:
                    group["first_created_at"] = first
                else:
                    group["first_created_at"] = min(group["first_created_at"],
                                                    first)
                work_started_at = min(work_started_at or first, first)
        active_rows = self._conn.execute("""
            SELECT branch_id, n_words, n_candidates, created_at
            FROM active_branches
            WHERE status = 'open' AND spine LIKE ?
            ORDER BY n_words DESC LIMIT ?
        """, (descendants, active_branch_limit)).fetchall()
        active = []
        for row in active_rows:
            branch_id = row["branch_id"]
            claim_row = self._conn.execute("""
                SELECT COUNT(*) AS done_count,
                       COUNT(*) FILTER (WHERE done_at >= ?) AS recent_count
                FROM candidate_claims WHERE branch_id = ? AND done = 1
            """, (since, branch_id)).fetchone()
            active.append({
                "branch_id": branch_id,
                "answer_count": row["n_words"],
                "candidate_count": row["n_candidates"],
                "done_candidate_count": claim_row["done_count"],
                "recent_done_candidate_count": claim_row["recent_count"],
                "created_at": row["created_at"],
            })
        return {
            "groups": groups,
            "active_branches": active,
            "open_branch_count": open_branch_count,
            "counted_branch_count": len(active),
            "work_started_at": work_started_at,
            "work_latest_at": work_latest_at,
            "recent_window_seconds": recent_window_seconds,
            "epoch": epoch,
            "telemetry_epochs": sorted(telemetry_epochs),
        }

    def opener_work_requests_for_word(self, word) -> list:
        """Every opener-work request naming `word`, oldest request first.

        The request time is when the word was *asked for*, which is not when
        the swarm began working it: a request sits behind higher-priority
        roots until workers reach it.
        """
        rows = self._conn.execute("""
            SELECT opener.opener_work_id, opener.requested_priority,
                   opener.requested_at, opener.state,
                   MAX(membership.resolved_at) AS completed_at
            FROM opener_work AS opener
            LEFT JOIN branch_opener_work AS membership
              ON membership.opener_work_id = opener.opener_work_id
            WHERE opener.opener = ?
            GROUP BY opener.opener_work_id
            ORDER BY opener.requested_at, opener.opener_work_id
        """, (word.lower(),)).fetchall()
        return [dict(row) for row in rows]

    def _bounded_sample_metadata(self, table, epoch, since, sample_size,
                                 spine_prefix=None, branch_key=None):
        spine_condition = " AND (spine = ? OR spine LIKE ?)" if spine_prefix else ""
        branch_condition = " AND branch_key = ?" if branch_key is not None else ""
        parameters = [epoch, since]
        if spine_prefix:
            parameters.extend((spine_prefix, spine_prefix + " %"))
        if branch_key is not None:
            parameters.append(branch_key)
        parameters.append(sample_size + 1)
        row = self._conn.execute(
            f"""SELECT COUNT(*) AS sampled_row_count FROM (
                    SELECT id FROM telemetry.{table}
                    WHERE epoch = ? AND recorded_at >= ?
                    {spine_condition}
                    {branch_condition}
                    ORDER BY recorded_at DESC, id DESC LIMIT ?
                )""",
            parameters,
        ).fetchone()
        sampled_with_probe = row["sampled_row_count"]
        return min(sampled_with_probe, sample_size), sampled_with_probe > sample_size

    def report_hotspots(self, field, epoch, since, sample_size, limit,
                        spine_prefix=None, branch_key=None) -> dict:
        """Return an explicitly bounded current or historical hotspot ranking."""
        current_fields = {"nodes", "age", "size", "workers", "priority", "slowest"}
        if field in current_fields:
            filters = {"sort": field, "limit": limit}
            result = (
                self.report_tree_rows(spine_prefix, filters, field, limit)
                if spine_prefix else self.report_queue_rows(filters, field, limit)
            )
            return {
                "population": "current_queue_branches",
                "epoch": epoch,
                "since": since,
                "sample_size": None,
                "sampled_row_count": result["matched_rows"],
                "sample_truncated": result["matched_rows"] > len(result["rows"]),
                "rows": result["rows"],
            }
        if field == "coordination" and spine_prefix:
            raise ValueError("coordination hotspots cannot be attributed to a spine")
        if field == "cut-reuse" and spine_prefix and branch_key is None:
            raise ValueError(
                "cut-reuse hotspots require a singular branch target"
            )
        table = {
            "evaluated-candidates": "branch_finalize_log",
            "bulk-completed-candidates": "branch_finalize_log",
            "one-level-erd-prunes": "branch_finalize_log",
            "two-level-erd-prunes": "branch_finalize_log",
            "cut-reuse": "cut_reuse_misses",
            "coordination": "claim_telemetry",
        }.get(field)
        if table is None:
            raise ValueError(f"unsupported hotspot field: {field}")
        sample_spine_prefix = spine_prefix if table == "branch_finalize_log" else None
        sample_branch_key = branch_key if table == "cut_reuse_misses" else None
        sampled_row_count, sample_truncated = self._bounded_sample_metadata(
            table, epoch, since, sample_size, sample_spine_prefix,
            sample_branch_key,
        )
        if field in (
                "evaluated-candidates", "bulk-completed-candidates",
                "one-level-erd-prunes", "two-level-erd-prunes"):
            metric = {
                "evaluated-candidates": "n_claims",
                "bulk-completed-candidates": "bulk_done_candidates",
                "one-level-erd-prunes": "one_level_erd_pruned_candidates",
                "two-level-erd-prunes": "two_level_erd_pruned_candidates",
            }[field]
            spine_condition = " AND (spine = ? OR spine LIKE ?)" if spine_prefix else ""
            parameters = [epoch, since]
            if spine_prefix:
                parameters.extend((spine_prefix, spine_prefix + " %"))
            parameters.extend((sample_size, limit))
            rows = self._conn.execute(f"""
                WITH sample AS (
                    SELECT * FROM telemetry.branch_finalize_log
                    WHERE epoch = ? AND recorded_at >= ? {spine_condition}
                    ORDER BY recorded_at DESC, id DESC LIMIT ?
                )
                SELECT * FROM sample
                ORDER BY COALESCE({metric}, 0) DESC, id DESC LIMIT ?
            """, parameters).fetchall()
            normalized_rows = [{
                "row_id": f"finalization:{row['id']}",
                "branch_key": bytes(row["branch_key"]) if row["branch_key"] else None,
                "spine": row["spine"],
                "answer_count": row["n_words"],
                "budget": row["budget"],
                "epoch": row["epoch"],
                "outcome": self._report_finalization_outcome(row),
                "loss_proof": self._report_loss_proof(row),
                "evaluated_candidate_count": row["n_claims"] or 0,
                "bulk_completed_candidate_count": row["bulk_done_candidates"] or 0,
                "one_level_erd_pruned_candidate_count":
                    row["one_level_erd_pruned_candidates"] or 0,
                "two_level_erd_pruned_candidate_count":
                    row["two_level_erd_pruned_candidates"] or 0,
                "search_node_count": row["nodes_spent"] or 0,
                "recorded_at": row["recorded_at"],
            } for row in rows]
            population = "recent_branch_finalizations"
        elif field == "cut-reuse":
            branch_condition = " AND branch_key = ?" if branch_key is not None else ""
            parameters = [epoch, since]
            if branch_key is not None:
                parameters.append(branch_key)
            parameters.extend((sample_size, limit))
            rows = self._conn.execute(f"""
                WITH sample AS (
                    SELECT * FROM telemetry.cut_reuse_misses
                    WHERE epoch = ? AND recorded_at >= ?
                    {branch_condition}
                    ORDER BY recorded_at DESC, id DESC LIMIT ?
                )
                SELECT branch_key, COUNT(*) AS miss_count,
                       MAX(recorded_at) AS recorded_at,
                       MAX(n_words) AS answer_count
                FROM sample GROUP BY branch_key
                ORDER BY miss_count DESC, branch_key LIMIT ?
            """, parameters).fetchall()
            normalized_rows = [{
                "row_id": (
                    f"cut-reuse:{bytes(row['branch_key']).hex()}"
                    if row["branch_key"] is not None else "cut-reuse:unknown"
                ),
                "branch_key": (
                    bytes(row["branch_key"])
                    if row["branch_key"] is not None else None
                ),
                "answer_count": row["answer_count"],
                "cut_reuse_miss_count": row["miss_count"],
                "recorded_at": row["recorded_at"],
            } for row in rows]
            population = "recent_cut_reuse_misses"
        else:
            rows = self._conn.execute("""
                WITH sample AS (
                    SELECT * FROM telemetry.claim_telemetry
                    WHERE epoch = ? AND recorded_at >= ?
                    ORDER BY recorded_at DESC, id DESC LIMIT ?
                )
                SELECT n_words, worker_count, COUNT(*) AS claim_count,
                       SUM(coordination_millis) AS coordination_millis,
                       SUM(COALESCE(claim_retries, 0)) AS claim_retry_count,
                       SUM(COALESCE(busy_wait_millis, 0)) AS busy_wait_millis
                FROM sample GROUP BY n_words, worker_count
                ORDER BY coordination_millis DESC, n_words, worker_count LIMIT ?
            """, (epoch, since, sample_size, limit)).fetchall()
            normalized_rows = [{
                "row_id": f"coordination:{row['n_words']}:{row['worker_count']}",
                "answer_count": row["n_words"],
                "worker_count": row["worker_count"],
                "claim_count": row["claim_count"],
                "coordination_millis": row["coordination_millis"],
                "claim_retry_count": row["claim_retry_count"],
                "busy_wait_millis": row["busy_wait_millis"],
            } for row in rows]
            population = "recent_claim_coordination_buckets"
        return {
            "population": population,
            "epoch": epoch,
            "since": since,
            "sample_size": sample_size,
            "sampled_row_count": sampled_row_count,
            "sample_truncated": sample_truncated,
            "rows": normalized_rows,
        }
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

    def epoch_metadata(self):
        """Return the active telemetry epoch and its descriptive metadata."""
        epoch_text = self.get_meta("epoch")
        if epoch_text is None:
            return None
        epoch = int(epoch_text)
        row = self._conn.execute(
            "SELECT label, git_sha, started_at, notes "
            "FROM main.telemetry_epoch WHERE epoch = ?",
            (epoch,),
        ).fetchone()
        return {
            "epoch": epoch,
            "label": row["label"] if row else None,
            "git_sha": row["git_sha"] if row else None,
            "started_at": row["started_at"] if row else None,
            "notes": row["notes"] if row else None,
        }

    def delete_meta(self, key: str):
        self._conn.execute("DELETE FROM run_meta WHERE key = ?", (key,))

    # ------------------------------------------------------------------
    # Checkpoint quiesce flag
    # ------------------------------------------------------------------

    def set_checkpoint_pause(self, paused: bool):
        """Ask workers to stay off the queue database momentarily.

        The flag stores its own timestamp: a worker honours it only while it
        is younger than CHECKPOINT_PAUSE_STALE_SECONDS, so a supervisor that
        dies with the flag set cannot wedge the swarm.
        """
        if paused:
            self.set_meta("checkpoint_pause", str(int(time.time())))
        else:
            self.delete_meta("checkpoint_pause")

    def checkpoint_paused(self) -> bool:
        value = self.get_meta("checkpoint_pause")
        if value is None:
            return False
        try:
            set_at = int(value)
        except ValueError:
            return False
        return time.time() - set_at < CHECKPOINT_PAUSE_STALE_SECONDS

    # ------------------------------------------------------------------
    # Disk watermark: samples, and the stop latch
    # ------------------------------------------------------------------

    def record_disk_sample(self, avail_bytes: int):
        """Append (timestamp, avail_bytes) to a bounded ring in run_meta, for
        the status display's disk growth rate."""
        samples = self.disk_samples()
        samples.append([int(time.time()), int(avail_bytes)])
        self.set_meta("disk_samples", json.dumps(samples[-DISK_SAMPLE_KEEP:]))

    def disk_samples(self) -> list:
        value = self.get_meta("disk_samples")
        if not value:
            return []
        try:
            samples = json.loads(value)
        except ValueError:
            return []
        return samples if isinstance(samples, list) else []

    def set_disk_stop(self, reason: str):
        """Latch the swarm down.  While set, `run` refuses to start; the latch
        survives reboots and systemd restarts, and only clear_disk_stop()
        (the `queue clear-disk-stop` command) releases it."""
        self.set_meta("disk_stop",
                      json.dumps({"at": int(time.time()), "reason": reason}))

    def set_disk_stop_if_unset(self, reason: str) -> bool:
        """Latch the swarm down unless a latch reason is already recorded.

        Returns True when this call created the latch.  A manual hold must not
        replace a disk-fill or WAL-ceiling reason recorded concurrently.
        """
        payload = json.dumps({"at": int(time.time()), "reason": reason})
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO run_meta (key, value) VALUES ('disk_stop', ?)",
            (payload,))
        return cursor.rowcount == 1

    def disk_stop(self):
        """The latch payload as a dict, or None when not latched."""
        value = self.get_meta("disk_stop")
        if not value:
            return None
        try:
            return json.loads(value)
        except ValueError:
            return {"at": None, "reason": value}

    def clear_disk_stop(self):
        self.delete_meta("disk_stop")

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def clear(self):
        """Wipe all queue state: pending/done branches, active branches,
        candidate claims, heartbeats, and run_meta.

        The persistent cache (wordle_cache.sqlite3) is not touched — only
        the transient coordination tables in erd_queue.sqlite3.
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("DELETE FROM candidate_claims")
            self._conn.execute("DELETE FROM candidate_holes")
            self._conn.execute("DELETE FROM active_branches")
            self._conn.execute("DELETE FROM pending_branches")
            self._resolve_branch_memberships(withdraw=True)
            self._conn.execute("DELETE FROM opener_work")
            self._conn.execute("DELETE FROM worker_heartbeat")
            self._conn.execute("DELETE FROM run_meta")
            self._conn.execute("COMMIT")
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def total_branches(self) -> int:
        """Total rows in pending_branches (all statuses)."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM pending_branches").fetchone()[0]

    def get_pending_branch(self, branch_key: bytes):
        """Return the pending_branches row for branch_key, or None."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        return self._conn.execute(
            "SELECT p.*, b.branch_key FROM pending_branches p "
            "JOIN branches b ON b.branch_id = p.branch_id "
            "WHERE p.branch_id = ?", (branch_id,)
        ).fetchone()

    def status_by_branch_keys(self, branch_keys) -> dict:
        """Return {branch_key: pending_branches row} for the given keys.

        A branch_key with no row was never queued; it is simply absent from
        the returned dict.
        """
        ids = self._branch_ids_for_keys(branch_keys)
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        rows = self._conn.execute(
            f"SELECT p.*, b.branch_key FROM pending_branches p "
            f"JOIN branches b ON b.branch_id = p.branch_id "
            f"WHERE p.branch_id IN ({placeholders})", ids
        ).fetchall()
        return {bytes(r["branch_key"]): r for r in rows}

    def active_branches_by_keys(self, branch_keys) -> dict:
        """Return {branch_key: active_branches row} for the given keys.

        Only open branches appear; finalized branches are deleted from this
        table and will be absent from the returned dict.
        """
        ids = self._branch_ids_for_keys(branch_keys)
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        rows = self._conn.execute(
            f"""SELECT a.*, b.branch_key,
                       CASE WHEN a.status = 'open' THEN COALESCE(
                           (SELECT MAX(owner_priority)
                            FROM live_branch_opener_rows
                            WHERE branch_id = a.branch_id),
                           CASE WHEN a.priority >= {LEGACY_PROMOTED_PRIORITY_MIN}
                                THEN 0 ELSE a.priority END
                       ) ELSE a.priority END AS effective_priority
               FROM active_branches a """
            f"JOIN branches b ON b.branch_id = a.branch_id "
            f"WHERE a.branch_id IN ({placeholders})", ids
        ).fetchall()
        return {bytes(r["branch_key"]): r for r in rows}

    def _branch_ids_for_keys(self, branch_keys):
        """The registered branch_ids for the given keys (unregistered keys
        dropped), for an IN-clause lookup on a normalized table."""
        ids = [self._intern_branch(k) for k in branch_keys]
        return [i for i in ids if i is not None]

    def get_active_branch(self, branch_key: bytes):
        """Return the active_branches row for branch_key, or None."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return None
        return self._conn.execute(
            f"""SELECT a.*, b.branch_key,
                       CASE WHEN a.status = 'open' THEN COALESCE(
                           (SELECT MAX(owner_priority)
                            FROM live_branch_opener_rows
                            WHERE branch_id = a.branch_id),
                           CASE WHEN a.priority >= {LEGACY_PROMOTED_PRIORITY_MIN}
                                THEN 0 ELSE a.priority END
                       ) ELSE a.priority END AS effective_priority
               FROM active_branches a """
            "JOIN branches b ON b.branch_id = a.branch_id "
            "WHERE a.branch_id = ?", (branch_id,)
        ).fetchone()

    def claims_for_branch(self, branch_key: bytes):
        """Return all candidate_claims rows for branch_key."""
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return []
        return self._conn.execute(
            "SELECT * FROM candidate_claims WHERE branch_id = ? ORDER BY idx",
            (branch_id,)
        ).fetchall()

    def cancel_active_branch(self, branch_key: bytes,
                             remove_from_queue: bool = False):
        """Atomically remove a branch's candidate claims and active_branches row.

        All queue-file DELETEs run in one transaction so a crash partway
        through cannot leave orphaned candidate_claims rows or a dangling
        active_branches row.  The telemetry-file bundle_stats delete rides in
        the same transaction but commits per-file under WAL: on a crash it can
        survive or vanish independently, which is acceptable for a diagnostic
        table no control path reads.

        With remove_from_queue=True, also deletes the pending_branches row
        (regardless of its status), fully removing the branch from the queue in
        the same transaction.
        """
        self._conn.execute("BEGIN")
        try:
            branch_id = self._intern_branch(branch_key)
            if branch_id is not None:
                self._conn.execute(
                    "DELETE FROM candidate_claims WHERE branch_id = ?",
                    (branch_id,))
                self._conn.execute(
                    "DELETE FROM candidate_republish WHERE branch_id = ?",
                    (branch_id,))
                self._conn.execute(
                    "DELETE FROM candidate_holes WHERE branch_id = ?",
                    (branch_id,))
            self._conn.execute(
                "DELETE FROM telemetry.bundle_stats WHERE branch_key = ?",
                (branch_key,))
            if branch_id is not None:
                self._conn.execute(
                    "DELETE FROM active_branches WHERE branch_id = ?",
                    (branch_id,))
                if remove_from_queue:
                    self._conn.execute(
                        "DELETE FROM pending_branches WHERE branch_id = ?",
                        (branch_id,))
                    self._resolve_branch_memberships(branch_id, withdraw=True)
            self._conn.execute("COMMIT")
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def requeue_completed_branch(self, branch_key: bytes) -> bool:
        """Return a finished branch to 'pending' so it is worked again.

        Clears what a completed branch leaves behind — its candidate claims,
        republish and hole rows, its bundle_stats, and any active_branches row
        — and flips the pending row back to 'pending', all in one transaction.
        Returns True if the branch was reset.

        The pending row is updated in place, never removed.  has_pending_row
        is what tells a worker a branch has an exact-result consumer and so
        must not be solved under a ceiling, and create_branch records a
        ceiling immutably: a row that blinked out could be re-created under
        one and finalize as a cut, which is never cached, leaving the caller
        with neither an exact result nor a queued request to produce one.

        Refuses, returning False, unless the branch is genuinely finished:

        - 'in_progress' in the queue means a worker holds its claims.
        - an *open* active_branches row means a worker is solving it now.
          create_branch takes any branch key regardless of pending status, so
          a branch finished for one request can be re-promoted as another
          request's descendant while its pending row still reads 'done'.  The
          two tables are therefore checked independently; neither implies the
          other.

        Membership is left to the caller: this clears the branch's work state
        but does not revive branch_opener_work, because reviving it blindly
        would attach live membership to requests that have since completed.
        add_pending_many's UPSERT re-establishes membership for the request
        being queued, and is what makes the reset branch claimable again.
        """
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            pending = self._conn.execute(
                "SELECT status FROM pending_branches WHERE branch_id = ?",
                (branch_id,)).fetchone()
            open_active = self._conn.execute(
                "SELECT 1 FROM active_branches "
                "WHERE branch_id = ? AND status = 'open'",
                (branch_id,)).fetchone()
            if (pending is None or pending["status"] != 'done'
                    or open_active is not None):
                self._conn.execute("COMMIT")
                return False
            self._conn.execute(
                "DELETE FROM candidate_claims WHERE branch_id = ?", (branch_id,))
            self._conn.execute(
                "DELETE FROM candidate_republish WHERE branch_id = ?",
                (branch_id,))
            self._conn.execute(
                "DELETE FROM candidate_holes WHERE branch_id = ?", (branch_id,))
            self._conn.execute(
                "DELETE FROM telemetry.bundle_stats WHERE branch_key = ?",
                (branch_key,))
            self._conn.execute(
                "DELETE FROM active_branches WHERE branch_id = ?", (branch_id,))
            self._conn.execute("""
                UPDATE pending_branches
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL,
                    completed_at = NULL
                WHERE branch_id = ?
            """, (branch_id,))
            self._conn.execute("COMMIT")
            return True
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
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        self._conn.execute(
            "UPDATE pending_branches SET priority = ? "
            "WHERE branch_id = ? AND status = 'pending'",
            (priority, branch_id))
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    def set_ownerless_active_priority(self, opener: str, priority: int) -> int:
        """Set priority on open, ownerless branches attributed to opener."""
        check_opener_priority_range(priority)
        cur = self._conn.execute("""
            UPDATE active_branches AS active
            SET priority = ?
            WHERE active.status = 'open'
              AND lower(active.opener) = lower(?)
              AND NOT EXISTS (
                  SELECT 1 FROM live_branch_opener_rows AS owner
                  WHERE owner.branch_id = active.branch_id
              )
        """, (priority, opener))
        return cur.rowcount

    def set_opener_work_priority(self, opener_work_id: int, priority: int) -> bool:
        """Atomically change an opener request and every owned branch priority."""
        check_opener_priority_range(priority)
        self._conn.execute("BEGIN")
        try:
            cur = self._conn.execute("""
                UPDATE opener_work SET requested_priority = ?
                WHERE opener_work_id = ? AND state != 'complete'
            """, (priority, opener_work_id))
            if cur.rowcount:
                self._conn.execute("""
                    UPDATE pending_branches
                    SET priority = (
                        SELECT MAX(owner_priority)
                        FROM live_branch_opener_rows
                        WHERE branch_id = pending_branches.branch_id
                    )
                    WHERE branch_id IN (
                        SELECT branch_id FROM live_branch_opener_rows
                        WHERE opener_work_id = ?
                    )
                """, (opener_work_id,))
                self._conn.execute("""
                    UPDATE active_branches
                    SET priority = (
                        SELECT MAX(owner_priority)
                        FROM live_branch_opener_rows
                        WHERE branch_id = active_branches.branch_id
                    )
                    WHERE branch_id IN (
                        SELECT branch_id FROM live_branch_opener_rows
                        WHERE opener_work_id = ?
                    )
                """, (opener_work_id,))
            self._conn.execute("COMMIT")
            return cur.rowcount == 1
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    def opener_work_rows(self):
        """Return opener requests with their roots and descendants count."""
        return self._conn.execute("""
            SELECT s.*, SUM(m.parent_branch_id IS NULL) AS root_count,
                   COUNT(m.branch_id) AS branch_count
            FROM opener_work s
            LEFT JOIN branch_opener_work m ON m.opener_work_id = s.opener_work_id
            GROUP BY s.opener_work_id
            ORDER BY s.requested_priority DESC, s.opener_work_id
        """).fetchall()

    def opener_rows(self):
        """Return one row per opener word, merging that word's requests.

        Opener work is keyed by (word, priority), so one word can hold several
        requests, and two of them can own the same branch.  Branches are
        therefore counted distinctly rather than summed across requests, which
        would report a shared branch once per owning request.

        direct_branch_count counts the branches the requests asked for
        directly; a branch acquired later by promotion carries a
        parent_branch_id and is counted only in branch_count.
        """
        opener_work_columns = {
            row["name"] for row in self._conn.execute(
                "PRAGMA table_info(opener_work)")
        }
        started_at = (
            "MIN(CASE WHEN s.state != 'complete' THEN s.started_at END)"
            if "started_at" in opener_work_columns else "NULL"
        )
        return self._conn.execute(f"""
            SELECT s.opener,
                   MIN(s.requested_at) AS requested_at,
                   {started_at} AS started_at,
                   MAX(s.requested_priority) AS requested_priority,
                   MAX(m.resolved_at) AS completed_at,
                   COUNT(DISTINCT s.opener_work_id) AS request_count,
                   MAX(s.state = 'active') AS has_active_request,
                   MAX(s.state != 'complete') AS has_incomplete_request,
                   COUNT(DISTINCT m.branch_id) AS branch_count,
                   COUNT(DISTINCT CASE WHEN m.parent_branch_id IS NULL
                                       THEN m.branch_id END)
                       AS direct_branch_count
                   ,COUNT(DISTINCT CASE
                       WHEN m.parent_branch_id IS NULL
                        AND p.status = 'done'
                       THEN m.branch_id END) AS direct_done_branch_count
            FROM opener_work s
            LEFT JOIN branch_opener_work m
              ON m.opener_work_id = s.opener_work_id
            LEFT JOIN pending_branches p ON p.branch_id = m.branch_id
            GROUP BY s.opener
            ORDER BY MAX(s.requested_priority) DESC, s.opener
        """).fetchall()

    def distinct_branch_count_for_words(self, openers):
        """Count the branches owned by these opener words, each branch once.

        Two words can own the same branch, so summing their branch counts
        double-counts exactly the shared ownership the opener report exists to
        show.  Counted over every membership, resolved or not, to match the
        per-word branch_count in opener_rows().
        """
        words = [word for word in openers if word is not None]
        if not words:
            return 0
        placeholders = ",".join("?" for _ in words)
        return self._conn.execute(f"""
            SELECT COUNT(DISTINCT m.branch_id)
            FROM branch_opener_work AS m
            JOIN opener_work AS s ON s.opener_work_id = m.opener_work_id
            WHERE s.opener IN ({placeholders})
        """, words).fetchone()[0]

    def opener_membership_rows(self, opener_work_id=None, opener=None,
                               include_resolved=False):
        """Return one row per (opener_work_id, branch_id) membership.

        This is the canonical detail query behind `view --openers`: every
        request that owns a branch, its recorded requested priority, the
        branch's effective priority (MAX(owner_priority) across every live
        owner, computed the same way set_opener_work_priority materializes it
        onto active_branches/pending_branches — computed directly here rather
        than trusted from that materialization, so a shared branch never
        reports one owner's request as though it were exclusive even when a
        second owner attached without a priority change to re-materialize
        it), and the promotion lineage (opener_pattern traces back to the root
        (word, pattern) this membership descends from; parent_branch_key is
        the immediate parent).

        resolved_at is NULL for a live (currently schedulable) membership and
        set once its request is complete; include_resolved=True also returns
        resolved memberships, so a completed request's ownership history
        remains queryable even though it is no longer schedulable.

        opener_work_id/opener filter to one request or one word's
        requests (a word may have more than one request; see
        set_opener_work_priority's word-to-request ambiguity).
        """
        clauses = []
        params = []
        if not include_resolved:
            clauses.append("membership.resolved_at IS NULL")
        if opener_work_id is not None:
            clauses.append("membership.opener_work_id = ?")
            params.append(opener_work_id)
        if opener is not None:
            clauses.append("opener.opener = ?")
            params.append(opener)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        live_since = int(time.time()) - WORKER_LIVENESS_SECONDS
        return self._conn.execute(f"""
            SELECT membership.opener_work_id,
                   opener.opener,
                   opener.requested_priority,
                   opener.state AS opener_state,
                   membership.branch_id,
                   branch.branch_key,
                   membership.opener_pattern,
                   membership.parent_branch_id,
                   parent_branch.branch_key AS parent_branch_key,
                   membership.resolved_at,
                   pending.status AS pending_status,
                   active.status AS active_status,
                   COALESCE(
                       (SELECT MAX(owner_priority) FROM live_branch_opener_rows
                         WHERE branch_id = membership.branch_id),
                       active.priority, pending.priority
                   ) AS branch_priority,
                   -- Fenced by heartbeat liveness like worker_counts_by_branch()
                   -- and report_queue_rows()'s workers CTE: a heartbeat row
                   -- survives a crashed worker (only unregister_worker/clear
                   -- remove it), so an unfenced count would keep reporting a
                   -- dead worker's branch as active forever.
                   (SELECT COUNT(*) FROM worker_heartbeat h
                     WHERE h.current_branch_id = membership.branch_id
                       AND h.updated_at >= ?) AS worker_count
            FROM branch_opener_work AS membership
            JOIN opener_work AS opener
              ON opener.opener_work_id = membership.opener_work_id
            JOIN branches AS branch ON branch.branch_id = membership.branch_id
            LEFT JOIN branches AS parent_branch
                   ON parent_branch.branch_id = membership.parent_branch_id
            LEFT JOIN pending_branches AS pending
                   ON pending.branch_id = membership.branch_id
            LEFT JOIN active_branches AS active
                   ON active.branch_id = membership.branch_id
            {where}
            ORDER BY opener.requested_priority DESC, membership.opener_work_id,
                     membership.branch_id
        """, [live_since, *params]).fetchall()

    def check_opener_work_invariants(self) -> list[str]:
        """Return human-readable violations of opener scheduling invariants."""
        violations = []

        unresolved_without_work = self._conn.execute("""
            SELECT membership.branch_id, membership.opener_work_id
            FROM branch_opener_work AS membership
            LEFT JOIN pending_branches AS pending
              ON pending.branch_id = membership.branch_id
             AND pending.status IN ('pending', 'in_progress')
            LEFT JOIN active_branches AS active
              ON active.branch_id = membership.branch_id
             AND active.status = 'open'
            WHERE membership.resolved_at IS NULL
              AND pending.branch_id IS NULL
              AND active.branch_id IS NULL
            ORDER BY membership.opener_work_id, membership.branch_id
        """).fetchall()
        for row in unresolved_without_work:
            violations.append(
                "live membership opener_work_id "
                f"{row['opener_work_id']} branch_id {row['branch_id']} "
                "has neither pending/in-progress work nor an open active branch"
            )

        opener_work_rows = self._conn.execute("""
            SELECT opener.opener_work_id, opener.state,
                   EXISTS (
                       SELECT 1 FROM branch_opener_work AS membership
                       WHERE membership.opener_work_id = opener.opener_work_id
                         AND membership.resolved_at IS NULL
                   ) AS has_live_membership
            FROM opener_work AS opener
            ORDER BY opener.opener_work_id
        """).fetchall()
        for row in opener_work_rows:
            is_complete = row["state"] == "complete"
            has_live_membership = bool(row["has_live_membership"])
            if is_complete and has_live_membership:
                violations.append(
                    f"complete opener_work_id {row['opener_work_id']} has a "
                    "live membership"
                )
            elif not is_complete and not has_live_membership:
                violations.append(
                    f"unfinished opener_work_id {row['opener_work_id']} has no "
                    "live membership"
                )

        owners_without_membership = self._conn.execute("""
            SELECT active.branch_id
            FROM active_branches AS active
            WHERE active.status = 'open'
              AND active.requires_opener_membership = 1
              AND NOT EXISTS (
                  SELECT 1 FROM branch_opener_work AS membership
                  WHERE membership.branch_id = active.branch_id
                    AND membership.resolved_at IS NULL
              )
            ORDER BY active.branch_id
        """).fetchall()
        for row in owners_without_membership:
            violations.append(
                f"opener-owned open branch_id {row['branch_id']} has no live "
                "membership"
            )

        # Catches a membership whose parent branch belongs to a different
        # request.  A parent flattened to another branch the *same* request
        # owns — a grandchild recorded against the root instead of its
        # immediate parent — satisfies this query and is not reported here;
        # that lineage is covered by
        # test_nested_cooperative_branch_records_immediate_parent.
        invalid_lineage = self._conn.execute("""
            SELECT child.opener_work_id, child.branch_id, child.parent_branch_id
            FROM branch_opener_work AS child
            WHERE child.parent_branch_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM branch_opener_work AS parent
                  WHERE parent.opener_work_id = child.opener_work_id
                    AND parent.branch_id = child.parent_branch_id
              )
            ORDER BY child.opener_work_id, child.branch_id
        """).fetchall()
        for row in invalid_lineage:
            violations.append(
                f"membership opener_work_id {row['opener_work_id']} branch_id "
                f"{row['branch_id']} has parent_branch_id "
                f"{row['parent_branch_id']} that the same request does not own"
            )

        priority_range = (OPENER_PRIORITY_MIN, OPENER_PRIORITY_MAX)
        invalid_requested_priorities = self._conn.execute("""
            SELECT opener_work_id, requested_priority
            FROM opener_work
            WHERE requested_priority NOT BETWEEN ? AND ?
            ORDER BY opener_work_id
        """, priority_range).fetchall()
        for row in invalid_requested_priorities:
            violations.append(
                f"opener_work_id {row['opener_work_id']} has requested priority "
                f"{row['requested_priority']} outside "
                f"{OPENER_PRIORITY_MIN}..{OPENER_PRIORITY_MAX}"
            )

        legacy_active_priorities = self._conn.execute("""
            SELECT active.opener,
                   EXISTS (
                       SELECT 1 FROM live_branch_opener_rows AS owner
                       WHERE owner.branch_id = active.branch_id
                   ) AS has_live_membership,
                   COUNT(*) AS branch_count
            FROM active_branches AS active
            WHERE active.status = 'open'
              AND active.priority >= ?
            GROUP BY active.opener, has_live_membership
            ORDER BY has_live_membership, active.opener
        """, (LEGACY_PROMOTED_PRIORITY_MIN,)).fetchall()
        for row in legacy_active_priorities:
            opener = row["opener"] or "(none)"
            membership = ("with live membership" if row["has_live_membership"]
                          else "without live membership")
            violations.append(
                f"{row['branch_count']} open branch(es) at or above legacy "
                f"priority {LEGACY_PROMOTED_PRIORITY_MIN:,}: {opener} "
                f"{membership}"
            )

        invalid_effective_priorities = self._conn.execute("""
            SELECT branch_id, opener_work_id, owner_priority
            FROM active_branch_owner_rows
            WHERE owner_priority NOT BETWEEN ? AND ?
            ORDER BY branch_id, opener_work_id
        """, priority_range).fetchall()
        for row in invalid_effective_priorities:
            owner = ("direct" if row["opener_work_id"] is None else
                     f"opener_work_id {row['opener_work_id']}")
            violations.append(
                f"open branch_id {row['branch_id']} owner {owner} has effective "
                f"priority {row['owner_priority']} outside "
                f"{OPENER_PRIORITY_MIN}..{OPENER_PRIORITY_MAX}"
            )

        role_placeholders = ",".join("?" for _ in SCHEDULING_ROLES)
        invalid_scheduling_roles = self._conn.execute(f"""
            SELECT worker_id, opener_work_id, scheduling_role
            FROM worker_heartbeat
            WHERE (scheduling_role IS NOT NULL
                   AND scheduling_role NOT IN ({role_placeholders}))
               OR (opener_work_id IS NOT NULL
                   AND (scheduling_role IS NULL
                        OR scheduling_role NOT IN (?, ?)))
               OR (opener_work_id IS NULL AND scheduling_role IN (?, ?))
            ORDER BY worker_id
        """, (*SCHEDULING_ROLES,
              SCHEDULING_ROLE_PREFERRED, SCHEDULING_ROLE_FALLBACK,
              SCHEDULING_ROLE_PREFERRED, SCHEDULING_ROLE_FALLBACK)).fetchall()
        for row in invalid_scheduling_roles:
            violations.append(
                f"worker {row['worker_id']} has opener_work_id "
                f"{row['opener_work_id']} with scheduling_role "
                f"{row['scheduling_role']!r}"
            )

        return violations

    def remove_pending(self, branch_key: bytes) -> bool:
        """Delete a pending (status='pending') branch from the queue.

        Returns True if a row was deleted.  Does not touch active_branches or
        candidate_claims — call cancel_active_branch() first if the branch is
        currently in progress.
        """
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return False
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "DELETE FROM pending_branches "
                "WHERE branch_id = ? AND status = 'pending'",
                (branch_id,))
            removed = self._conn.execute("SELECT changes()").fetchone()[0] > 0
            if removed:
                self._resolve_branch_memberships(branch_id, withdraw=True)
            self._conn.execute("COMMIT")
            return removed
        except Exception:  # pragma: no cover
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Mid-loop publisher support
    # ------------------------------------------------------------------

    def mark_claims_done(self, branch_key: bytes, indices):
        """Insert already-evaluated candidates as authoritative done=1 claims.

        Called by the mid-loop publisher to record the candidates evaluated
        inline before overrun fired.  Uses INSERT OR REPLACE so a racing
        in-flight (done=0) claim from another worker is superseded by the
        authoritative done=1 record — the evaluation already happened.

        A branch with no active_branches row has already finalized, so
        delete_branch has cleared its claims and cached its result; recording
        more candidates against it would orphan claim rows past the branch they
        belong to.  The INSERT is gated on the active row's existence so such a
        write is a quiet no-op.  A publisher call racing finalization is a
        routine outcome, not an error.
        """
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return
        now = int(time.time())
        indices = list(indices)
        before = self._conn.total_changes
        # Carry any best_first_position the packer stamped through the
        # replace: these are the candidates the publisher evaluated inline, so
        # they are completed work, and dropping their rank would hide them from
        # candidate_schedule_diagnostics' count of work done ahead of the
        # winner — and NULL the winner's own rank whenever the publisher is
        # what finished it, which is common on a branch's best-first prefix.
        self._conn.executemany("""
            INSERT OR REPLACE INTO candidate_claims
                (branch_id, idx, claimed_by, claimed_at, done, done_at,
                 best_first_position)
            SELECT ?, ?, 'publisher', ?, 1, ?,
                   (SELECT best_first_position FROM candidate_claims
                    WHERE branch_id = ? AND idx = ?)
            WHERE EXISTS (SELECT 1 FROM active_branches WHERE branch_id = ?)
        """, [(branch_id, idx, now, now, branch_id, idx, branch_id)
              for idx in indices])
        written = self._conn.total_changes - before
        self._tally_wal_traffic(
            'candidate_claims/publisher-mark-done', written,
            written * _CLAIM_ROW_WAL_BYTES)

    def add_nodes_spent(self, branch_key: bytes, delta: int, *,
                        infeasible: bool = False):
        """Add one candidate's nodes and any depth-infeasibility evidence."""
        if delta <= 0 and not infeasible:
            return
        delta = max(0, delta)
        branch_id = self._intern_branch(branch_key, create=True)
        self._conn.execute("""
            UPDATE active_branches
            SET nodes_spent = nodes_spent + ?,
                infeasible_candidates = infeasible_candidates + ?,
                infeasible_nodes = infeasible_nodes + ?
            WHERE branch_id = ?
        """, (delta, int(infeasible), delta if infeasible else 0, branch_id))
        n = self._conn.execute("SELECT changes()").fetchone()[0]
        self._tally_wal_traffic(
            'active_branches/nodes-spent', n, n * _CLAIM_ROW_WAL_BYTES)

    # ------------------------------------------------------------------
    # Cost model (online time-weighted geometric mean per size bucket)
    # ------------------------------------------------------------------

    def _cost_bucket_row(self, policy: str, n_words: int, budget: int):
        """The cost_model accumulators for one (policy, size_bucket, budget) cell."""
        return self._conn.execute(
            "SELECT weighted_log_sum, weight_sum, weighted_log_sq, last_updated "
            "FROM cost_model WHERE policy = ? AND size_bucket = ? AND budget = ?",
            (policy, cost_size_bucket(n_words), budget)).fetchone()

    def _warm_bucket_row(self, policy: str, n_words: int, budget: int):
        """Warm accumulators for n_words at `budget`, or None when cold."""
        row = self._cost_bucket_row(policy, n_words, budget)
        if row is None or row['weight_sum'] < _COST_MODEL_MIN_WEIGHT:
            return None
        return row

    def get_cost_typical(self, policy: str, n_words: int, budget: int):
        """Geometric-mean node count for n_words at `budget`, or None when cold.

        The estimate is exp(weighted_log_sum / weight_sum).  budget keys the
        model on remaining-guess budget; a cold (size_bucket, budget) cell
        returns None so the caller can use a size-based heuristic.
        """
        row = self._warm_bucket_row(policy, n_words, budget)
        if row is None:
            return None
        return math.exp(row['weighted_log_sum'] / row['weight_sum'])

    def get_cost_spread(self, policy: str, n_words: int, budget: int):
        """Std-dev of ln(nodes) for n_words at `budget` (log-normal sigma), or None.

        Recovered from the stored second log-moment:
            sigma^2 = weighted_log_sq/weight_sum - mu^2
        Round-off can make this marginally negative when every sample is equal;
        clamp to 0.  Used for the over-promotion shade exp(mu - Z*sigma) and for
        offline distribution analysis.
        """
        row = self._warm_bucket_row(policy, n_words, budget)
        if row is None:
            return None
        mu = row['weighted_log_sum'] / row['weight_sum']
        var = row['weighted_log_sq'] / row['weight_sum'] - mu * mu
        return math.sqrt(var) if var > 0 else 0.0

    def update_cost_model(self, policy: str, n_words: int, nodes: int, *,
                          budget: int, weight: float = 1.0, now: int = None):
        """Fold one node-cost sample (value `nodes`, multiplicity `weight`)
        into the (policy, size_bucket, budget) cell.

        weight > 1 records `weight` identical samples of `nodes` in one call.
        For a batch of *distinct* samples whose individual magnitudes matter to
        the spread, use update_cost_model_logsums so each sample reaches the
        second log-moment without a lossy pre-averaging collapse.
        """
        if nodes <= 0 or weight <= 0:
            return
        log_n = math.log(nodes)
        self._fold_cost_sample(policy, n_words, log_n * weight,
                               log_n * log_n * weight, weight, now, budget)

    def update_cost_model_logsums(self, policy: str, n_words: int,
                                  log_sum: float, log_sq_sum: float,
                                  weight: float, *, budget: int, now: int = None):
        """Fold a pre-summed batch of log samples: (Σ ln x, Σ ln²x, count) into
        the (policy, size_bucket, budget) cell.

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
            INSERT INTO telemetry.cost_samples
                (policy, n_words, nodes, wall_millis, budget, censored, source,
                 epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (policy, n_words, nodes, wall_millis, budget, censored, source,
              self.epoch, now))

    def add_claim_telemetry(self, n_words: int, coordination_millis: int,
                            work_nodes: int, worker_count: int,
                            branch_key: bytes = None, spine: str = None,
                            worker_id: str = None, bundle_id: str = None,
                            idx: int = None, bundle_start_idx: int = None,
                            bundle_end_idx: int = None,
                            scheduling_millis: int = 0,
                            candidate_evaluation_millis: int = None,
                            evaluation_bound_erd: float = None):
        """Append a claim coordination record to claim_telemetry for offline analysis.

        claim_retries / busy_wait_millis / claim_transaction_millis /
        claim_commit_millis come from the most recent claim path on this
        connection (set by claim_next_bundle), the direct lock-contention and
        transaction-phase signal; the row is stamped with the active epoch.

        branch_key/spine attribute this row to the branch the claim belonged
        to.  branch_key is interned to its branches-registry branch_id before
        storage (see _intern_branch) rather than stored as the raw BLOB: the
        registry already carries the durable branch_key <-> branch_id mapping
        for every branch that has ever been created, so duplicating the BLOB
        onto every one of a branch's candidate rows would be pure size with
        no independent information — a size difference that matters here,
        since this is the highest-volume table in the schema.  A caller with
        no branch context passes branch_key=None, which stores branch_id
        NULL, not 0 or an interned NULL-key row.  worker_id/bundle_id/idx/
        bundle_start_idx/bundle_end_idx identify which worker evaluated which
        candidate of which bundle (all default to NULL for a caller with no
        branch/bundle context).
        candidate_evaluation_millis is the elapsed solver work for this
        candidate.  It
        is separate from coordination_millis, which spans only the gap between
        candidates.

        scheduling_millis is the caller's work-selection scan for this claim
        (opener-work ordering, pending promotion, joining an in-progress
        branch), already net of any lock wait and claim transaction it
        contained so the phases stay disjoint.  idle_millis is
        coordination_millis minus every other timed phase, so those five
        partition it exactly and idle_millis means genuinely unaccounted
        wait rather than unattributed scan work.

        Every row is one candidate evaluation, so COUNT(*) over the table is a
        claim count.  Branch finalize cost is deliberately not recorded here —
        it belongs to a branch rather than to a claim, and goes to
        branch_finalize_log.cache_write_millis instead.

        Consumed values are reset to 0 immediately after this INSERT reads
        them, so they are attributed to exactly the next telemetry row and
        never repeat on a later, unrelated candidate's row.  This matters
        under bundling: a candidate's own within-candidate sub-branch
        promotion can trigger a nested claim_next_bundle call on this same
        connection before this candidate's own add_claim_telemetry call —
        without the reset, that nested claim's contention numbers would
        otherwise still be sitting here (unconsumed) for whichever LATER,
        unrelated bundle member logs telemetry next.
        """
        now = int(time.time())
        branch_id = (None if branch_key is None
                    else self._intern_branch(branch_key))
        branch_worker_count = (None if branch_id is None else
                               self._branch_worker_count(
                                   branch_id, worker_id, now))
        idle_millis = max(0, coordination_millis
                          - self._last_claim_transaction_millis
                          - self._last_claim_commit_millis
                          - self._last_claim_busy_millis
                          - scheduling_millis)
        self._conn.execute("""
            INSERT INTO telemetry.claim_telemetry
                (n_words, coordination_millis, candidate_evaluation_millis,
                 work_nodes, claim_retries,
                 busy_wait_millis, worker_count, branch_id, spine, worker_id,
                 branch_worker_count, evaluation_bound_erd, bundle_id, idx, bundle_start_idx,
                 bundle_end_idx,
                 claim_transaction_millis, claim_commit_millis,
                 scheduling_millis, idle_millis, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (n_words, coordination_millis, candidate_evaluation_millis, work_nodes,
              self._last_claim_retries,
              self._last_claim_busy_millis, worker_count, branch_id, spine,
              worker_id, branch_worker_count, evaluation_bound_erd, bundle_id, idx, bundle_start_idx,
              bundle_end_idx,
              self._last_claim_transaction_millis, self._last_claim_commit_millis,
              scheduling_millis, idle_millis, self.epoch, now))
        self._last_claim_retries = 0
        self._last_claim_busy_millis = 0
        self._last_claim_transaction_millis = 0
        self._last_claim_commit_millis = 0

    SCHEDULE_DIAGNOSTIC_FIELDS = (
        "winner_best_first_position",
        "winner_republish_count",
        "candidates_completed_before_winner",
        "max_best_first_position_before_winner",
        "republished_candidates",
        "max_candidate_republish_count",
    )

    def candidate_schedule_diagnostics(self, branch_key, winner_idx=None):
        """Best-first scheduling evidence for one branch's own claim rows.

        Read the moment before delete_branch drops those rows, so
        "was the winning candidate reached in rank order, or deferred behind
        weaker ones" stays answerable from branch_finalize_log alone instead of
        being reconstructed from aggregate bundle telemetry.

        winner_best_first_position is the winner's rank in the branch's
        best-first candidate order; candidates_completed_before_winner and
        max_best_first_position_before_winner count and rank the
        worker-evaluated candidates that finished ahead of it, ordered by
        done_at.  That column has one-second resolution, so those two are
        None rather than zero on a branch whose winner shares its second with
        other completions — the ordering inside a second is not recoverable,
        and a zero would read as "no inversion" when the truth is "cannot
        tell".  A weakest rank
        far past the winner's own is a priority inversion: the branch spent
        itself on later candidates while a strong one waited.
        republished_candidates and max_candidate_republish_count say how much
        of that came from bundle overruns.

        A field is None where the branch cannot answer it: no claim rows left
        (solved from reused cache entries), no winner (a cut or a loss), or a
        winner whose rank was never recorded because it did not come through
        the packer.  Ranks come from
        candidate_claims.best_first_position, which only the packer stamps, so
        an ERD-pruned candidate is never counted as completed work.
        """
        diagnostics = {field: None for field in self.SCHEDULE_DIAGNOSTIC_FIELDS}
        branch_id = self._intern_branch(branch_key)
        if branch_id is None:
            return diagnostics
        republish = self._conn.execute(
            "SELECT COUNT(*) AS n, MAX(count) AS worst FROM candidate_republish "
            "WHERE branch_id = ?", (branch_id,)).fetchone()
        diagnostics["republished_candidates"] = republish["n"]
        diagnostics["max_candidate_republish_count"] = republish["worst"]
        if winner_idx is None:
            return diagnostics
        winner_republish = self._conn.execute(
            "SELECT count FROM candidate_republish WHERE branch_id = ? AND idx = ?",
            (branch_id, winner_idx)).fetchone()
        diagnostics["winner_republish_count"] = (
            0 if winner_republish is None else winner_republish["count"])
        winner = self._conn.execute(
            "SELECT best_first_position, done, done_at FROM candidate_claims "
            "WHERE branch_id = ? AND idx = ?", (branch_id, winner_idx)).fetchone()
        if winner is None:
            return diagnostics
        diagnostics["winner_best_first_position"] = winner["best_first_position"]
        if not winner["done"] or winner["done_at"] is None:
            return diagnostics
        ahead = self._conn.execute("""
            SELECT COUNT(*) AS n, MAX(best_first_position) AS worst
            FROM candidate_claims
            WHERE branch_id = ? AND done = 1 AND done_at < ?
              AND best_first_position IS NOT NULL
        """, (branch_id, winner["done_at"])).fetchone()
        if not ahead["n"]:
            # done_at has one-second resolution, so a candidate that finished
            # in the winner's own second is not orderable against it.  A zero
            # here with such candidates present means "cannot tell", not "the
            # winner went first", and a report must not show it as evidence.
            same_second = self._conn.execute("""
                SELECT COUNT(*) FROM candidate_claims
                WHERE branch_id = ? AND done = 1 AND done_at = ? AND idx != ?
                  AND best_first_position IS NOT NULL
            """, (branch_id, winner["done_at"], winner_idx)).fetchone()[0]
            if same_second:
                return diagnostics
        diagnostics["candidates_completed_before_winner"] = ahead["n"]
        diagnostics["max_best_first_position_before_winner"] = ahead["worst"]
        return diagnostics

    def add_branch_finalize_log(self, branch_key, spine, n_words, budget,
                                created_at, finalized_at, nodes_spent, n_claims,
                                n_bundles=None, max_bundle_nodes=None,
                                total_bundle_wall_millis=None, censored_units=None,
                                ceiling=None, outcome=None,
                                bulk_done_candidates=None, best_guess=None,
                                best_erd=None, cache_write_millis=None,
                                one_level_erd_pruned_candidates=None,
                                two_level_erd_pruned_candidates=None,
                                infeasible_candidates=None,
                                infeasible_nodes=None,
                                schedule_diagnostics=None,
                                hint_word=None, hint_was_winner=None,
                                first_best_at=None, nodes_at_first_best=None):
        """Persist a branch's timing/cost the moment before delete_branch drops it.

        The bundle-diagnostic columns (n_bundles, max_bundle_nodes,
        total_bundle_wall_millis, censored_units) come from
        finalize_bundle_stats; they stay NULL when the caller doesn't pass
        them (a branch solved entirely from reused cache entries never
        claims a bundle).  ceiling is the alpha-beta ceiling the branch was
        solved under (NULL = exact solve); outcome is 'exact', 'cut', or 'loss'.
        bulk_done_candidates preserves the legacy combined ERD-prune count;
        the one-level and two-level counts preserve its provenance without
        restoring per-candidate telemetry writes.  best_guess/best_erd capture
        the exact solved line at finalize time so later reports do not need to
        infer it from mutable cache state.  If a one-level ERD-prune sweep
        supersedes an in-flight claim that later finishes, n_claims excludes
        that overlap even though its per-claim telemetry row still exists.
        cache_write_millis is the wall time the finalizing worker spent
        publishing the result (score-cache/loss/cut writes and the cost-model
        fold) — the finalize phase of the coordination breakdown, kept here
        rather than on claim_telemetry because it belongs to the branch.
        schedule_diagnostics is candidate_schedule_diagnostics' mapping, which
        carries the branch's best-first scheduling evidence past the claim
        rows that are about to be deleted; None leaves every one of its
        columns NULL.

        hint_word is the word the run's hint artifact named for this branch and
        hint_was_winner whether the branch's own recomputed winner turned out
        to be that word — the payoff measurement for a hinted rebuild.  Both
        are NULL on a run with no hint artifact.  first_best_at/
        nodes_at_first_best are the branch's first-incumbent stamps, which
        against created_at give the cost of reaching any bound at all.
        """
        schedule_diagnostics = schedule_diagnostics or {}
        if (one_level_erd_pruned_candidates is None
                and two_level_erd_pruned_candidates is None
                and bulk_done_candidates is not None):
            one_level_erd_pruned_candidates = bulk_done_candidates
            two_level_erd_pruned_candidates = 0
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO telemetry.branch_finalize_log
                (branch_key, spine, n_words, budget, epoch, created_at,
                 finalized_at, nodes_spent, n_claims, n_bundles,
                 max_bundle_nodes, total_bundle_wall_millis, censored_units,
                 ceiling, outcome, bulk_done_candidates, best_guess, best_erd,
                 cache_write_millis, one_level_erd_pruned_candidates,
                 two_level_erd_pruned_candidates,
                 infeasible_candidates, infeasible_nodes,
                 winner_best_first_position, winner_republish_count,
                 candidates_completed_before_winner,
                 max_best_first_position_before_winner,
                 republished_candidates, max_candidate_republish_count,
                 hint_word, hint_was_winner, first_best_at,
                 nodes_at_first_best,
                 recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (branch_key, spine, n_words, budget, self.epoch, created_at,
              finalized_at, nodes_spent, n_claims, n_bundles, max_bundle_nodes,
              total_bundle_wall_millis, censored_units, ceiling, outcome,
              bulk_done_candidates, best_guess, best_erd, cache_write_millis,
              one_level_erd_pruned_candidates,
              two_level_erd_pruned_candidates,
              infeasible_candidates, infeasible_nodes,
              schedule_diagnostics.get("winner_best_first_position"),
              schedule_diagnostics.get("winner_republish_count"),
              schedule_diagnostics.get("candidates_completed_before_winner"),
              schedule_diagnostics.get(
                  "max_best_first_position_before_winner"),
              schedule_diagnostics.get("republished_candidates"),
              schedule_diagnostics.get("max_candidate_republish_count"),
              hint_word,
              None if hint_was_winner is None else int(hint_was_winner),
              first_best_at, nodes_at_first_best,
              now))

    def add_cut_reuse_miss(self, branch_key, n_words, budget, wanted_ceiling,
                           available_bound, available_budget):
        """Log a re-solve forced by a cut: a consumer needed this branch but
        the recorded cut could not satisfy it (bound too low for the consumer's
        ceiling, or proven at a smaller budget than the consumer's).
        wanted_ceiling None = the consumer needed an exact value."""
        now = int(time.time())
        self._conn.execute("""
            INSERT INTO telemetry.cut_reuse_misses
                (branch_key, n_words, budget, wanted_ceiling, available_bound,
                 available_budget, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (branch_key, n_words, budget, wanted_ceiling, available_bound,
              available_budget, self.epoch, now))

    def add_candidate_accuracy(self, branch_key, n_words, budget, predicted_work,
                               bound_erd, candidate_cost_lower_bound,
                               erd_lower_bound_pruned, actual_nodes,
                               group_sizes=None, opener=None,
                               candidate_word=None, worker_id=None,
                               bundle_id=None, idx=None, started_at=None,
                               evaluation_millis=None, outcome=None,
                               republish_count=None):
        """Log one predicted-vs-actual work point for the §10 metric-validation gate.

        Under single-candidate claiming a claim is exactly one candidate, so
        actual_nodes is that candidate's true cost.  group_sizes ('-'-joined
        response-group sizes) is the sufficient statistic for recomputing any work
        metric offline; logged only for non-ERD-pruned rows.  opener is the
        root opener of the branch's spine, so a multi-day corpus can be
        segmented per opener (different openers reach differently-shaped
        answer sets).  The claim identity fields match claim_telemetry, while
        candidate_word removes dependence on the candidate order that produced
        idx.  started_at is the evaluation start; evaluation_millis preserves
        its subsecond duration while recorded_at remains the completion time.
        """
        now = int(time.time())
        branch_id = self._intern_branch(branch_key)
        if republish_count is None and branch_id is not None and idx is not None:
            republish = self._conn.execute(
                "SELECT count FROM candidate_republish "
                "WHERE branch_id = ? AND idx = ?", (branch_id, idx)).fetchone()
            republish_count = 0 if republish is None else republish["count"]
        self._conn.execute("""
            INSERT INTO telemetry.candidate_accuracy
                (branch_key, branch_id, candidate_word, worker_id, bundle_id, idx,
                 n_words, budget, predicted_work, bound_erd,
                 candidate_cost_lower_bound, erd_lower_bound_pruned,
                 actual_nodes, group_sizes, opener, started_at,
                 evaluation_millis, outcome, republish_count, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (branch_key, branch_id, candidate_word, worker_id, bundle_id, idx,
              n_words, budget, predicted_work, bound_erd,
              candidate_cost_lower_bound, 1 if erd_lower_bound_pruned else 0,
              actual_nodes, group_sizes, opener, started_at,
              evaluation_millis, outcome, republish_count, self.epoch, now))

    @staticmethod
    def _accuracy_percentiles(values):
        values = sorted(values)
        if not values:
            return {f"p{percentile}": None
                    for percentile in (1, 10, 50, 90, 99)}
        return {
            f"p{percentile}": values[min(
                len(values) - 1, int(percentile / 100 * len(values)))]
            for percentile in (1, 10, 50, 90, 99)
        }

    def report_candidate_accuracy(self, epoch=None, budget=None,
                                  minimum_answer_count=None,
                                  maximum_answer_count=None, opener=None,
                                  branch_key=None, since=None, limit=None,
                                  sample_size=50_000, raw_row_offset=0):
        """Return a bounded candidate-level calibration sample.

        Candidate telemetry is one row per claim.  Sampling random ids across
        an epoch avoids both a full aggregate and treating a few recent minutes
        from one opener as representative of a multi-day epoch.  The seed is
        derived from the epoch and requested size, so a report is reproducible;
        a different requested size is an independent sample.  Requested and
        achieved counts make a selective-filter shortfall explicit.
        """
        epoch = self.epoch if epoch is None else epoch
        where, parameters = ["epoch = ?"], [epoch]
        for condition, value in (
                ("budget = ?", budget),
                ("n_words >= ?", minimum_answer_count),
                ("n_words <= ?", maximum_answer_count),
                ("opener = ?", opener),
                ("branch_key = ?", branch_key),
                ("recorded_at >= ?", since)):
            if value is not None:
                where.append(condition)
                parameters.append(value)
        condition = " WHERE " + " AND ".join(where)
        sampling_condition = " WHERE epoch = ?"
        sampling_parameters = [epoch]
        if since is not None:
            sampling_condition += " AND recorded_at >= ?"
            sampling_parameters.append(since)
        id_bounds = self._conn.execute(
            "SELECT MIN(id), MAX(id) FROM telemetry.candidate_accuracy"
            + sampling_condition, sampling_parameters).fetchone()
        minimum_id, maximum_id = id_bounds
        sampled_rows = []
        if minimum_id is not None:
            # Candidate-accuracy ids are append-only and dense.  Batched point
            # lookups keep the random sample bounded by the epoch/id index.
            random_generator = random.Random(f"{epoch}:{sample_size}")
            sample_ids = random_generator.sample(
                range(minimum_id, maximum_id + 1),
                min(sample_size, maximum_id - minimum_id + 1))
            for start in range(0, len(sample_ids), 500):
                batch = sample_ids[start:start + 500]
                placeholders = ", ".join("?" for _ in batch)
                sampled_rows.extend(dict(row) for row in self._conn.execute(
                    "SELECT * FROM telemetry.candidate_accuracy" + condition
                    + f" AND id IN ({placeholders})", [*parameters, *batch]))
        raw_rows = []
        if limit is not None:
            raw_rows = [dict(row) for row in self._conn.execute(
                "SELECT * FROM telemetry.candidate_accuracy" + condition
                + " ORDER BY id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, raw_row_offset])]
        for row in sampled_rows + raw_rows:
            predicted = row["predicted_work"]
            actual = row["actual_nodes"]
            row["actual_predicted_ratio"] = (
                actual / predicted if predicted is not None and predicted > 0
                else None)
        non_pruned = [row for row in sampled_rows
                      if not row["erd_lower_bound_pruned"]]

        def calibration(group_rows):
            return {
                "row_count": len(group_rows),
                "predicted_work": self._accuracy_percentiles(
                    [row["predicted_work"] for row in group_rows
                     if row["predicted_work"] is not None]),
                "actual_nodes": self._accuracy_percentiles(
                    [row["actual_nodes"] for row in group_rows]),
                "actual_predicted_ratio": self._accuracy_percentiles(
                    [row["actual_predicted_ratio"] for row in group_rows
                     if row["actual_predicted_ratio"] is not None]),
            }

        buckets = {}
        for row in non_pruned:
            buckets.setdefault(
                (cost_size_bucket(row["n_words"]), row["budget"]), []).append(row)
        ratio_rows = [row for row in non_pruned
                      if row["actual_predicted_ratio"] is not None]
        ordered = sorted(ratio_rows, key=lambda row: row["actual_predicted_ratio"])
        population_row_count = None
        if len(where) == 1 or (len(where) == 2 and since is not None):
            population_row_count = self._conn.execute(
                "SELECT COUNT(*) FROM telemetry.candidate_accuracy"
                + sampling_condition, sampling_parameters).fetchone()[0]
        return {
            "epoch": epoch,
            "requested_sample_size": sample_size,
            "sampled_row_count": len(sampled_rows),
            "population_row_count": population_row_count,
            "erd_pruned_row_count": sum(
                row["erd_lower_bound_pruned"] for row in sampled_rows),
            "non_erd_pruned_row_count": len(non_pruned),
            "no_prediction_row_count": sum(
                row["predicted_work"] is None or row["predicted_work"] <= 0
                for row in sampled_rows),
            "raw_row_offset": raw_row_offset,
            "calibration": calibration(non_pruned),
            "answer_count_budget_calibration": [
                {"answer_count_bucket_start": key[0], "budget": key[1],
                 **calibration(group_rows)}
                for key, group_rows in sorted(buckets.items())],
            "largest_over_predicted": ordered[:10],
            "largest_under_predicted": list(reversed(ordered[-10:])),
            "rows": raw_rows,
        }

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
            INSERT INTO telemetry.backstop_telemetry
                (n_words, budget, elapsed_millis, nodes, predicted_nodes,
                 remaining_candidates, epoch, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (n_words, budget, elapsed_millis, nodes, predicted_nodes,
              remaining_candidates, self.epoch, now))
