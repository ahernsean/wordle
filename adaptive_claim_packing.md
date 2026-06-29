# Adaptive claim packing with republish-on-overrun

Supersedes `adaptive_chunk_claiming.md`. That doc proposed sizing a single
contiguous slice ("chunk") per claim from a node-cost heuristic. This design
keeps its goal — amortize coordination over real work — but fixes the two ways a
naive version regresses to a past failure: (1) it groups candidates by *predicted
work*, not by adjacency or count, so it never recreates the old chunk imbalance;
and (2) when a claim turns out heavier than predicted, it republishes the
unfinished candidates **as work-sized groups, never one-candidate-per-claim**.

### Relationship to issues #67 and #68

- **#67 (adaptive chunk claiming)** — this is the refined plan that addresses it.
  #67's Problem, measurements, invariants, and acceptance criteria carry over
  unchanged; its proposed *Direction* (a contiguous slice sized by a node-cost
  ratio) is replaced here, because a contiguous slice of a best-first list
  reproduces the old chunk imbalance. This document is the design #67 should adopt.
- **#68 (optimistic monotonic `next_idx` cursor)** — **obviated as a standalone
  change, absorbed as a requirement.** #68 optimizes `claim_candidate`, the
  single-index handout this design *deletes* (replaced by `claim_next_group`,
  §6). Its `next_idx`-on-`active_branches` migration therefore does not apply. But
  #68's underlying requirement — O(1)-amortized handout so a branch drains in O(n)
  not O(n²), with reclaimed holes picked up lazily rather than rescanned on every
  call — is a correctness/performance requirement of the packer, folded into §6
  (two-cursor pool walk) and §12 (pool maintenance). #68's equivalence test (an
  injected mid-sweep reclaim yields identical coverage) is retained there.

---

## 1. Background: what the data says

Telemetry from the multi-day run (`claim_telemetry` in `erd_queue.sqlite3`,
70.56M claims, 6 workers, Jun 23–29 2026):

- **99.4% of claims do fewer than 10 search nodes.** All real compute (~19.8B of
  19.8B nodes) lives in the <1% of claims doing ≥1k nodes.
- Real compute ≈ 82 h; summed coordination ≈ 412 h. Aggregate coordination is
  **~83% of in-loop wall**; in the tiny-claim regime it is **~816×** the compute.
- Coordination is **not lock contention**. `worker_count` never varied from 6 in
  the data and `claim_retries` was never populated, so contention is not directly
  testable — but 51% of claims coordinate in sub-millisecond time, which rules out
  pervasive write-lock waits. The cost is per-claim fixed overhead (transaction +
  the `claim_candidate` scan + heartbeat) plus a starvation/finalize tail.

The unit of work (one candidate) is simply too small to amortize the per-claim
coordination. The branch's *node cost* is fine; the *claim count* is the problem.

### Why the obvious fixes each failed before

Per-candidate work at a branch is violently skewed: a small head of strong
splitters fully recurses (the winner plus near-ties that must be proven worse),
while a long tail is cut cheaply by the shared `best_erd` bound. The two past
designs sit at opposite failures of that distribution:

- **Old chunks** cut the best-first-ranked candidate list into equal-**cardinality**
  contiguous slices. The first slice held all the expensive candidates → one
  worker ground it for hours while the rest finished tail slices in milliseconds.
  Radical imbalance. (The old `chunk_size_for` docstring admitted this directly.)
- **Current single-candidate atoms** use granularity 1 → perfect balance but the
  816× coordination above.

The cure is that **claim granularity must track expected work**: the expensive
head self-fragments toward singletons, the cheap tail self-coalesces into bulk.

---

## 2. Goals and non-goals

**Goals**

1. Cut claim count by ~100× so per-claim coordination stops dominating wall time.
2. Preserve load balance under an *inaccurate* cost model — the model's actual
   error is large (see §4), so balance cannot depend on prediction accuracy.
3. Never regress to one-candidate-per-claim on any path (fresh handout, reclaim,
   or republish), except the one correct case: a single candidate whose own
   estimate exceeds the work target.
4. Keep ERD results bit-identical to single-candidate claiming.

**Non-goals**

- Changing the ERD recurrence, scoring, or the per-branch node cost. Coordination
  changes; nodes do not.
- Replacing within-candidate sub-branch promotion. That mechanism
  (overrun-publish a *single candidate's* deep subtree as promoted branches) is
  orthogonal and composes with this one (see §7).

---

## 3. Core idea

Two pieces, one feedback loop:

- **Packing (static seed).** Handing out a claim selects a *group* of currently
  unclaimed candidate indices whose summed predicted work ≈ a target `W`. Heavy
  candidates become singleton groups; cheap candidates coalesce into large groups.
- **Republish-on-overrun (runtime correction).** A worker measures *actual* nodes
  as it evaluates its group. If the group overruns a threshold `T = c·W` before it
  is done, the worker stops, returns its unfinished candidates to the unclaimed
  pool, and they are **re-packed into work-sized groups** on the next handout —
  with their per-candidate estimates scaled up by the observed error.

The division of labor is the whole point:

> The packer uses the **unreliable model** to get claim *count* right.
> Republish-on-overrun uses **measured nodes** to get *balance* right.
> The model only has to be roughly rank-correct at the top; the loop corrects it.

---

## 4. Expected-work metric

Per-candidate predicted work, reusing the existing cost model:

```
predicted_work(c) = Σ over c's response groups g of  typical(|g|, budget − 1)
```

where `typical(k, b)` is the cost model's `exp(μ)` node estimate for a branch of
`k` words at budget `b`. In current code this is
`ERDQueue.get_cost_typical(policy, n_words)` (`erd_queue.py:893`, returns
`exp(weighted_log_sum / weight_sum)`), wrapped per-worker by
`_BranchWorker._typical(n)` (`erd_swarm.py:504`); its spread is
`ERDQueue.get_cost_spread` (`erd_queue.py:905`). Both are keyed on `n_words` only
today — §9.4 adds the `budget` argument `b`. This is the calibrated form of the
`Σk²` ordering already computed in `_solve_subset`'s candidate sort
(`wordle_engine.py:~1182`, inside the function at `wordle_engine.py:1097`, key
`sum(k*k for k in cache.group_counts(c, branch_words).values())`). Computing `predicted_work` for all candidates is one
pass of the same `group_counts` work that sort already does; persist the resulting
per-candidate work vector with the branch (sorted descending) so the packer reads
it without recomputing.

**The model is weak, and we have data proving it:**

- Its own self-reported spread (`weighted_log_sq` → σ) reaches σ(ln nodes)=3.3 →
  a **737× band**; size buckets 8–14 are 10–33×.
- Raw actuals confirm independently: for `n_words=30`, finalized branches span
  16,871 → 26,512,414 nodes — a **1,571× spread** for the same input size.

This is exactly why §3's runtime correction is load-bearing and why packing alone
(however clever) cannot be trusted to balance. Rank-correctness at the top — "which
candidates are the heavy ones" — is a much weaker and more attainable requirement
than "equal predicted work ⇒ equal actual work," and it is all the packer needs.

---

## 5. The packer

Operates over the **currently unclaimed** candidate indices of a branch (the pool
shrinks as claims complete and grows as overruns/reclaims return indices). Keep the
candidate list sorted once by `predicted_work` descending; the pool is a subset.

Two-pointer fill (largest-first with light backfill — the classic bin-packing
heuristic):

```
def next_group(pool_sorted_desc, work, W):
    # pool_sorted_desc: unclaimed indices, heaviest predicted_work first
    if empty(pool): return None
    head = pool_sorted_desc.front()
    if work[head] >= W:
        return [head]                      # heavy candidate → its own claim
    group, total = [], 0
    # take from the heavy front while it fits
    while pool and work[pool.front()] + total <= W:
        i = pool.pop_front(); group.append(i); total += work[i]
    # top up from the cheap back so the bin lands near W
    while pool and work[pool.back()] + total <= W:
        i = pool.pop_back(); group.append(i); total += work[i]
    return group
```

Properties:

- A candidate with `predicted_work ≥ W` is **always its own claim**. So the
  expensive head spreads one-per-worker — good for balance *and* for bound
  propagation (workers evaluate strong splitters first, in parallel, tightening
  `best_erd` fast).
- Each non-singleton bin lands near `W` with several light items, so per-item
  estimate error partially cancels across the bin.
- The pure tail coalesces into large groups (hundreds of <10-node candidates),
  collapsing claim count from `n_candidates` to ≈ `total_predicted_work / W`.

**Cost of the walk (absorbs #68).** Maintain the per-branch pool as the
work-sorted index list with a front pointer (heavy) and a back pointer (light);
`next_group` advances them. Each call does work proportional to the *group it
claims*, so draining a branch is O(n_candidates) total — never the O(n²) per-call
gap scan that #68 set out to remove. This is #68's monotonic cursor generalized:
the cursor advances by a work-sized group instead of by 1, so #68's separate
`next_idx` change is unnecessary. Reclaimed/republished indices re-enter the pool
*behind* the front pointer; they are not rescanned on every call — they are picked
up on a later pass once the forward walk reaches them or is exhausted (§12),
preserving #68's lazy end-of-sweep hole semantics.

`W` is the single tuning dial. Set it so coordination is a small fraction of a
bin's compute: at `W ≈ 100k` nodes (~1.5 s compute at ~15 µs/node), the ~21 ms
average coordination is ~1.5% overhead instead of 816×.

**Handout ordering.** Hand out heavy (high-max-work) groups before light bulk
groups, and **gate the large tail-bulk groups until `best_erd` is set** on the
branch. The tail is only cheap *because* the shared bound cuts it; releasing a
big tail group before any head candidate has established the bound would make the
worker pay full cost on candidates that should have been cut.

---

## 6. Queue representation and the claim API

Today `candidate_claims` is one row per candidate, PK `(branch_key, idx)`, and
`claim_candidate` returns the single lowest unclaimed `idx`. The change:

- **Replace `claim_candidate` with `claim_next_group`**, which runs the §5 packer
  *inside the `BEGIN IMMEDIATE` transaction* over the unclaimed pool, inserts one
  `candidate_claims` row per chosen `idx` (all `claimed_by` = this worker, same
  `claimed_at`), and returns the index list. One transaction per *group*, not per
  candidate — that is where the coordination win comes from.
- A logical claim is the set of rows sharing `(branch_key, claimed_by,
  claimed_at)`. No new table is required; group identity is derivable. (Optionally
  add a `group_id` column for cleaner reclaim queries.)
- **Finalize coverage is unchanged**: a branch finalizes when all `n_candidates`
  rows are `done=1`, whoever observes full coverage.

Crucially, because *handout itself is the packer*, **every** source of work-to-do
flows through it and therefore produces work-sized groups:

- Fresh candidates → packer groups them.
- Reclaimed candidates from a dead worker (`reclaim_stale_claims` deletes the
  `done=0` rows → they reappear in the unclaimed pool) → packer **re-groups** them.
- Republished candidates from an overrun (§7) → packer **re-groups** them.

There is no code path that hands out a lone candidate except the §5 heavy-singleton
case. This is the structural guarantee against the "1 candidate = 1 claim" trap.

---

## 7. Republish-on-overrun (the no-singleton rule)

A worker evaluating a group tracks actual nodes spent. Two distinct overruns:

**(a) Cross-candidate overrun** — the *group* has spent `≥ T = c·W` actual nodes
and still has unfinished candidates `R`. The worker:

1. Marks the candidates it finished `done=1` (their results already folded into
   shared best).
2. Returns `R` to the unclaimed pool (delete the `done=0` rows for `R`).
3. **Scales the estimate** for each `r ∈ R` by the observed error on this group,
   `actual_so_far / predicted_so_far`, and writes the scaled work back to the
   branch's work vector.
4. Does **not** create `|R|` claims. `R` simply re-enters the pool; the next
   `claim_next_group` re-packs it. Because the estimates were just scaled up, the
   re-pack naturally produces *smaller* groups (more of them) where work turned out
   dense — finer granularity exactly where needed, still multi-candidate wherever
   the items remain individually cheap.

Evaluate within a group in `predicted_work` descending order, so that when an
overrun fires, the items already completed are the heaviest ones and `R` is the
lighter remainder — the common case re-packs into a few medium groups rather than
re-isolating a known-heavy item repeatedly.

**(b) Within-candidate overrun** — a *single* candidate's own recursive subtree
exceeds `T`. This is not a packing problem; it is the existing **sub-branch
promotion** path (`_subbranch_solver` / overrun-publish-at-own-level): the
candidate's large response groups are promoted as their own branches for the
swarm, and the candidate's claim completes when its promoted children finalize.
Packing and promotion compose: packing decides *which candidates share a claim*;
promotion decides *how one expensive candidate's subtree is parallelized*.

Guardrails:

- **Floor on group size during re-pack:** keep grouping while items are
  individually `< W`. Only isolate an item when its (possibly scaled) estimate
  reaches `W`. This prevents the re-pack from drifting toward singletons.
- **Bounded republish depth:** cap how many times the same candidate can be
  republished before it is forced through promotion (b) instead, so a
  badly-modeled candidate converges instead of thrashing the pool.
- **Cap a group's wall-time**, not just its nodes, so a worker cannot sit on a
  group long enough to widen the reclaim window after a crash.

---

## 8. Invariants preserved

From `adaptive_chunk_claiming.md` and the ERD pruning invariants:

1. **Shared-best folding stays per candidate.** Grouping defers the queue
   round-trip *between* candidates, never the best update; a worker refreshes and
   may tighten the branch's shared best between candidates in its group.
2. **Sequential-sibling pruning preserved.** Candidates in a group are evaluated in
   order, carrying the partial best, exactly as a single-worker sweep. Across
   groups, the existing shared-best refresh supplies the cross-worker bound. No new
   cross-subtree cancellation is introduced.
3. **Taint propagation unchanged.** Any candidate excluded by the depth budget
   taints the branch, winner or not.
4. **Finalization coverage unchanged.** Finalize fires only when all
   `n_candidates` slots are `done=1`.
5. **Reclaim correctness preserved and improved.** A dead worker's `done=0` rows
   are freed and re-enter the pool; the packer re-groups them. The end state is
   identical to single-candidate claiming, never a swarm of singleton gaps.

An equivalence test must assert: for a fixed branch, packed claiming yields the
same branch best, the same `max_depth`, and the same cache rows as single-candidate
claiming.

---

## 9. Measurement: telemetry epochs (prerequisite, land first)

We must be able to separate **pre-implementation** from **post-implementation**
telemetry so we can recompute the cost model and compare coordination statistics
across the change. Land this *before* the claim-path change so there is a clean
baseline.

### 9.1 Epoch tagging

- New table `telemetry_epoch(epoch INTEGER PRIMARY KEY, label TEXT, git_sha TEXT,
  started_at INTEGER, notes TEXT)`. The running supervisor stamps the active
  `epoch` (read from `run_meta`).
- Add `epoch INTEGER NOT NULL DEFAULT 0` to every append-only telemetry table:
  `claim_telemetry`, `cost_samples`, `backstop_telemetry`, and the new
  `branch_finalize_log` (§9.2). Existing rows default to **epoch 0 = "single-candidate
  atom baseline."**
- On deploying the packing change, insert a new epoch row (label
  `"pack+overrun"`, current `git_sha`) and bump `run_meta.epoch`. Every new
  telemetry row is stamped with it. All comparisons and model fits filter on epoch.

This is an idempotent migration in `ERDQueue._migrate()` (queue is Linux-only).
The `DEFAULT 0` makes the column add safe on the live 70M-row table.

### 9.2 Per-branch / per-subbranch timing

The data needed to ever answer "how long did branch X take" does not survive
today: `cost_samples.wall_millis` is never populated, and
`active_branches.created_at/finalized_at` are deleted at finalize. Add a durable
log, one row written at finalize **before** `delete_branch`:

```
branch_finalize_log(
  branch_key, spine, n_words, budget, epoch,
  created_at, finalized_at,           -- wall-clock span (upper bound; interleaved)
  nodes_spent,                        -- compute volume
  n_claims, total_coord_millis,       -- coordination attributable to this branch
  n_groups, max_group_nodes           -- packing/balance diagnostics
)
```

Subbranches finalize as first-class branches under recursive promotion, so this
captures subbranch timing for free — one row each.

### 9.3 Per-candidate accuracy stream

The packer depends on `predicted_work(c)`; we currently have no record of how wrong
it is per candidate. With grouped claims, log per candidate (or per group):
`predicted_work`, `actual_work_nodes`, `n_words`, `budget`, `epoch`. This is the
stream that lets us (a) measure packing accuracy directly, (b) detect systematic
bias to correct the scale factor in §7, and (c) decide whether `Σk²` or
`Σ typical(|g|)` is the better cheap proxy.

### 9.4 Cost model recomputation

- The per-branch node cost (`cost_model` table, accessed via
  `ERDQueue.get_cost_typical` / `get_cost_spread` / `update_cost_model` /
  `add_cost_sample`, keyed by `cost_size_bucket(n_words)` at `erd_queue.py:30`) is
  **epoch-invariant** — claiming changes coordination, not how many nodes a branch
  costs — so it may carry across epochs as a seed. But **re-key it on
  `(size_bucket, budget)`**: thread `budget` through `cost_size_bucket`, the
  `cost_model` primary key, and all four accessors above. Budget is the dominant
  missing variable behind the 1,571× spread, and `budget` is already known at
  finalize. Recompute the post-epoch model from `cost_samples` filtered to the new
  epoch (and the new key) once enough samples accrue; seed it from the prior model
  so packing is not cold at cutover.
- The genuinely *new* model is the per-candidate work model from §9.3, which did
  not exist under single-candidate claiming at useful fidelity.

### 9.5 Settle the lock-contention question while here

To close the contention question the data currently cannot answer: populate
`claim_retries` and a `busy_wait_millis` on each claim, and decompose
`coordination_millis` into components (claim-scan / complete / finalize /
idle-search). Then a run with `--workers` varied gives the contention signal
directly. Cheap to add alongside the epoch work.

---

## 10. Rollout sequence

1. **Instrument (epoch 0 still running).** Land §9.1–9.3 and §9.5 — epoch table,
   `branch_finalize_log`, per-candidate accuracy, retry/decomposed coordination.
   Stop workers, deploy, restart (per the kill-old-workers rule). Collect a clean
   baseline under single-candidate claiming.
2. **Re-key cost model on `(size, budget)`** (§9.4) and verify predicted-vs-actual
   variance drops materially on the new key.
3. **Implement packing + republish** (§5–§8) behind the same restart discipline.
   Bump to epoch 1.
4. **Compare epoch 0 vs epoch 1** from `branch_finalize_log` and `claim_telemetry`:
   claims/branch, coordination fraction, claims-per-second, straggler (max group
   nodes) distribution, effective throughput, per-branch wall-time.

---

## 11. Acceptance criteria

- Claim transactions to drain a deep branch drop from `n_candidates` to ≈
  `total_predicted_work / W` (target ~100× fewer).
- Coordination share of wall (epoch 1 `claim_telemetry`) falls from ~83% toward
  <10%, with the largest drop at three or more guesses played.
- No straggler regression: the `max_group_nodes` distribution in
  `branch_finalize_log` has no tail of multi-`W` groups that fail to republish —
  i.e., the old chunk imbalance does not reappear.
- ERD results identical to single-candidate claiming (equivalence test, §8).
- Draining a branch is O(n_candidates) total claim work, not O(n²) (the goal of
  #68), and an injected mid-sweep reclaim yields identical coverage to the current
  implementation (#68's equivalence test, retained).

---

## 12. Open questions

- **Choosing `W` and `c`.** Start `W` from `coord_millis / node_time` so a bin's
  compute dwarfs its coordination; choose the overrun factor `c` (e.g. 2–4) from
  the §9.3 accuracy data — large enough not to thrash, small enough to cap
  stragglers.
- **Packer location.** Running the two-pointer inside `BEGIN IMMEDIATE` lengthens
  the write transaction. If that shows up in §9.5's decomposition, precompute group
  boundaries outside the lock and claim a precomputed group id range inside it.
- **Pool maintenance and lazy hole reclaim (the #68 concern).** The unclaimed
  pool changes under reclaim/republish; decide whether to keep it as a live sorted
  structure per active branch (front/back pointers, §5) or derive it per handout
  from `candidate_claims`. The latter is simpler but re-sorts; the former needs
  careful invalidation when estimates are scaled in §7. Either way, adopt #68's
  laziness: do not rewind to chase a reclaimed hole on every call — let the forward
  walk pick it up, or sweep for holes once the forward walk is exhausted, so the
  common (no-hole) path stays O(1)-amortized.
