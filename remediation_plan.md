# Remediation plan for adaptive lookahead/design gaps

## Scope
This plan addresses the specific shortfalls reported by the user and aligns implementation behavior with `design.md`.

## 1) Unify depth-2 and depth-3 semantics

### Problem
Depth 2 currently uses `compute_lookahead(...)` while depth >=3 uses `compute_adaptive_lookahead(...)`, producing different candidate universes, reporting style, and ranking behavior.

### Plan
1. Route *all* lookahead depths through the adaptive engine.
   - For depth=2, run adaptive with `max_depth=1` and disable pruning side effects that can distort fully-evaluated results.
2. Keep the existing two-step routine as an optional verification path (debug flag only), not the user-facing default.
3. Ensure candidate policy is explicit and identical across depths:
   - root candidate set
   - deeper candidate set (`global_candidates` vs hard-mode survivors-only)
4. Add cross-check tests that compare depth=2 adaptive totals against legacy two-step within tolerance for the same policy.

## 2) Replace compressed single-line status with readable multi-line status

### Problem
Current status format is compact and opaque (`C_N`, ranges, `=`, `~`, unlabeled rows).

### Plan
1. Introduce a user-oriented status renderer with multi-line blocks every N ticks.
2. Each block should include:
   - elapsed/budget, queue/frontier sizes, activated words
   - plain-language definitions for symbols
   - top contenders table with columns: `word`, `lower`, `upper`, `exact?`, `gap`, `root_rank`
3. Rename or explain fields inline:
   - `C_N` -> `Top-N cutoff lower bound (C_N)`
   - `=` -> exact bound closed (`lower == upper`)
   - `~` -> not exact yet
4. Add optional `--verbose-status` mode showing one sample active work item path (`root -> child subset size -> depth left`).

## 3) Remove mandatory dependency on prior entropy solve

### Problem
`cmd_lookahead` hard-requires `scores_updated` by entropy solve before running lookahead.

### Plan
1. Remove the early guard that forces command `s` first.
2. On demand, compute any missing root ranking inputs inside lookahead:
   - if no entropy scores exist, compute/cached entropy ranking for required top words automatically.
3. Add shared cache interface so entropy solve and lookahead reuse:
   - root partitions
   - subgroup local 1-step results
4. Keep user-facing behavior flexible: users can run either command first.

## 4) Ensure SQLite persistence is created and used

### Problem
No obvious SQLite artifact appears, suggesting persistence is not wired or not flushed.

### Plan
1. Validate initialization path for `AdaptiveCacheSQLite` in app startup and `Solution` construction.
2. Define explicit cache DB location and print it once per session in debug/info mode.
3. Force writes at safe checkpoints:
   - post-activation batch
   - periodic flush timer
   - end-of-command close/commit
4. Add diagnostics command (`cacheinfo`) to show:
   - db path
   - row count
   - last write time
   - hit/miss counters
5. Add an integration test that runs lookahead and asserts DB file existence + nonzero rows.

## 5) Additional design-alignment gaps to close

1. Implement/verify DAG state interning by `(subset_bits, remaining_depth, policy)` consistency checks.
2. Verify global viability pruning rule exactly follows `U < C_N`.
3. Retain full local rankings in state after `REFINE_CHILD`; do not conflate ranking with pruning.
4. Add replayable benchmark harness (fixed seed, fixed word list) to track:
   - quality progression over time
   - convergence rate of bounds
   - cache hit ratios

## 6) Delivery order

### Phase 1 (behavioral correctness)
- Remove entropy prerequisite.
- Route depth=2 through adaptive path.
- Add clearer status output.

### Phase 2 (persistence and observability)
- Wire/verify SQLite lifecycle.
- Add cache diagnostics and tests.

### Phase 3 (algorithmic consistency hardening)
- Pruning-rule audits, DAG checks, retained-ranking invariants.
- Regression/benchmark harness.

## 7) Acceptance criteria

1. `l` works in a fresh session without `s`.
2. Depth 2 and depth 3 share the same adaptive framework and comparable policy semantics.
3. Status output is multi-line and self-explanatory.
4. SQLite cache file is created automatically and grows after lookahead runs.
5. Re-running lookahead shows measurable cache reuse (reduced work/time).
