# ERD cache re-verification plan

## Problem

The reclaim-while-alive bug (fixed in commit `774ac29`, Jun 15 2026) could write
a permanently-cached ERD that is **suboptimal but not wrong in direction** — the
stored value is ≥ the true optimum.  Any branch finalized before that fix is a
candidate for a bad cache entry.  Rather than flushing and recomputing from
scratch, we can exploit the bounded error.

## Key insight

Because the bug can only produce an ERD ≥ the true optimum, every cached value
is a valid alpha-beta ceiling.  A verification pass seeded with those ceilings:

- **Correct entry:** search hits the ceiling immediately, confirms with near-zero
  work, moves on.
- **Wrong entry (too high):** search finds the true optimum (which beats the
  ceiling), updates the cache.

Work done is proportional to how wrong the cache is, not to the size of the full
search space.

## Scope

Re-verify **every entry** in `branch_best_by_policy`.  SALET's root is the most
important, but any branch solved before the fix is suspect.  A full pass costs
little for correct entries and automatically self-limits.

## Implementation sketch

1. **New script `verify_erd_cache.py`** (or a `--verify` flag on an existing
   tool):
   - Iterate all `(policy, branch_key, universe_id)` rows in
     `branch_best_by_policy`.
   - For each row, read the cached `best_guess` ERD as the initial alpha-beta
     ceiling.
   - Submit a fresh solve via the existing `erd_search` path with that ceiling.
   - If the solver finds a strictly better value: update the cache row and log
     the correction.
   - If not: no write, move on.

2. **Parallelism:** the existing swarm machinery (`erd_queue` + workers) can be
   reused.  Each branch is a unit of work; the ceiling is passed as the
   `solve_budget` / alpha-beta parameter already threaded through
   `ERDQueue.create_branch`.

3. **Idempotency:** the pass can be interrupted and resumed.  A branch that was
   already verified (and not updated) is simply re-verified cheaply on the next
   run.  A `verified_ts` column in `branch_best_by_policy` could skip already-
   confirmed rows on resume, but is optional — the re-verify cost for confirmed
   rows is negligible.

4. **Completion signal:** log a summary at the end: N branches checked, M
   corrections written, total wall time.

## What this is NOT

- A flush.  No rows are deleted before the pass starts.
- A full recompute.  The ceiling prevents redundant search on correct entries.
- Part of the rename PR.  This is a separate operational step, run after the
  rename is deployed and the schema is migrated.

## Sequencing

1. Rename PR merges, schema migrated on Linux + phone.
2. Run `verify_erd_cache.py` on Linux against the live cache.
3. Export and sync to phone as normal after the pass completes.
