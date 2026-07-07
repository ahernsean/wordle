# Binary claim packing with republish-on-overrun

This is the plan to replace the swarm's single-candidate claiming with
count-bundled claim packing over an **exact binary split**: at handout time,
every unclaimed candidate is either *provably ERD-lower-bound-eliminated*
against the branch's current real bound (`candidate_cost_lower_bound >= B`,
an exact test, not an estimate) or it is not. Eliminated candidates coalesce
into large count-capped bundles; not-yet-eliminated candidates pack into
small fixed-count bundles in best-first order. A bundle that overruns its
cap republishes its unfinished remainder for re-packing — never as
one-candidate-per-claim.

**There is no work-magnitude model anywhere in this design.** The epoch-0
validation run measured the candidate work estimate
(`estimate_candidate_work`) against actual per-candidate cost and killed it:
81.1% false-expensive rate, Pearson-log correlation 0.191 — predicted
magnitude carries almost no information about actual node cost. The two
things that *did* validate are the only two things the scheme relies on:

1. **ERD-lower-bound elimination is exact.** 100% of eliminated candidates
   completed in ≤ 1 node. `candidate_cost_lower_bound` is the engine's own
   admissible bound; `>= B` *proves* the candidate cannot win and *will* be
   cut for free.
2. **Claim count, not node cost, is the disease.** Coordination is a fixed
   ~27–35 ms per claim while the median claim does 1–2 nodes of work.

The measured claim-count reduction from count-bundling alone (no magnitude
model), per `analyze_swarm_telemetry.py estimate_claim_reduction`:

| `small_count` | Epoch 0 aggregate | Epoch 2 aggregate |
|---:|---:|---:|
| 8 | 44.2× | 104.6× |
| 16 | 82.3× | 174.2× |
| 32 | 144.1× | 260.5× |
| 64 | 230.9× | 346.0× |

### Relationship to issues #67 and #68

- **#67 (adaptive claim packing)** — this document is the plan that
  addresses it. #67's Problem, measurements, invariants, and acceptance
  criteria hold here. Its *Direction* has been superseded twice, each time
  by data: the original contiguous slice (all the expensive candidates land
  in one slice — the old chunk imbalance), then work-sized packing by a
  predicted-work metric (killed by the epoch-0 validation above). The
  binary scheme is the surviving design.
- **#68 (monotonic `next_idx` cursor for `claim_candidate`)** — **obviated
  as a standalone change, absorbed as a requirement.** This design *deletes*
  `claim_candidate` (replaced by `claim_next_bundle`, §6), so #68's
  `next_idx`-on-`active_branches` migration does not apply. Its underlying
  requirement — O(1)-amortized handout so a branch drains in O(n) not
  O(n²), with reclaimed holes picked up lazily rather than rescanned on
  every call — is a requirement of the packer, met in §5 (single forward
  cursor over best-first order) and §12 (shared cursor state). #68's
  equivalence test (an injected mid-sweep reclaim yields identical
  coverage) is retained there.

---

## 1. Background: what the data says

**Epoch 0** (single-candidate baseline, `claim_telemetry` in
`erd_queue.sqlite3`, 115.4M claims):

- **99.3% of claims do fewer than 10 search nodes.** All real compute lives
  in the <1% of claims doing ≥ 1k nodes.
- Mean coordination ≈ 27 ms/claim; aggregate coordination ≈ **5.9×** the
  compute; in the tiny-claim regime ≈ **720×**.
- Coordination is **not lock contention**: 51% of claims coordinate in
  sub-millisecond time, which rules out pervasive write-lock waits. The
  cost is per-claim fixed overhead (transaction + claim scan + heartbeat)
  plus a starvation/finalize tail.

**Epoch 2** (vectorized kernel deployed — full_tree_plan.md §4/§5; 4.61M
claims over a 7.5 h small-branch drain, all branches n ≤ 86):

- ERD-lower-bound elimination rate at evaluation: **93.7%** (epoch 0:
  33.8%). Median `work_nodes`: **1** (epoch 0: 2); 99.8% of claims < 10
  nodes.
- Mean coordination ≈ 35 ms/claim — unchanged in kind. Aggregate
  coordination overhead ≈ **42×** the compute (epoch 0: 5.9×); tiny-claim
  ≈ **2,200×**. (The cross-epoch comparison carries a workload-mix
  confound — the epoch-2 window drained only small branches — but the
  within-epoch instrument agrees:) median branch wall is nearly flat at
  ~430–655 s across cost buckets 2–16 while median nodes spans 13k → 2.2M.
  A 170× range in compute moves wall by 1.5×: **small-branch wall is
  almost pure coordination.**

The kernel work made every candidate cheaper to evaluate and left the
per-claim coordination untouched, so the packing prize *grew*. The unit of
work (one candidate) is simply too small to amortize the per-claim
coordination. The branch's *node cost* is fine; the *claim count* is the
problem.

### Why the earlier designs each failed

Per-candidate work at a branch is violently skewed: a small head of strong
splitters fully recurses (the winner plus near-ties that must be proven
worse), while a long tail is cut for free by the shared `best_erd` bound.

- **Old chunks** cut the best-first-ranked candidate list into
  equal-cardinality contiguous slices. The first slice held all the
  expensive candidates → one worker ground it for hours while the rest
  finished tail slices in milliseconds. Radical imbalance.
- **Single-candidate atoms** (current) use granularity 1 → perfect balance
  but the coordination overhead above.
- **Work-sized packing** (this document's previous revision) bundled by a
  predicted-work metric. The epoch-0 validation showed the magnitude
  prediction is noise (r = 0.191), so bundles sized by it would have been
  sized by noise.

The cure is the part of the skew we can know *exactly*: whether a candidate
is already provably eliminated. That single bit separates the free mass
(93.7% and rising as `B` tightens) from the mass that might do work — and
it can never be wrong.

---

## 2. Goals and non-goals

**Goals**

1. Cut claim count by ~100× so per-claim coordination stops dominating
   wall time. (The epoch-2 model gives 105× at `small_count = 8`.)
2. Depend on **no cost model**. Bundle composition uses only the exact
   elimination test and fixed count dials; balance is protected by
   republish-on-overrun, not by prediction.
3. Never regress to one-candidate-per-claim on any path (fresh handout,
   reclaim, or republish).
4. Keep ERD results equal to single-candidate claiming **within ±1e-5**
   (exact equality is not required; floating-point reordering of the
   weighted sum is fine). The *winning guess* and `max_depth` must match
   exactly.

**Non-goals**

- Changing the ERD recurrence, scoring, or the per-branch node cost.
  Coordination changes; nodes do not.
- Replacing within-candidate sub-branch promotion. That mechanism
  (overrun-publish a *single candidate's* deep subtree as promoted
  branches) is orthogonal and composes with this one (§7).

---

## 3. Core idea

Two pieces, one feedback loop:

- **Packing (static scheduling on an exact signal).** Handing out a claim
  selects a *bundle* of currently unclaimed candidate indices. Each
  candidate is classified by the exact test
  `candidate_cost_lower_bound(c) >= B`, where `B` is the branch's current
  real `best_erd` read inside the claim transaction. Provably-eliminated
  candidates coalesce into bulk bundles of up to `count_cap`;
  not-yet-eliminated candidates pack into small bundles of `small_count`,
  front-to-back in best-first order.
- **Republish-on-overrun (runtime correction).** A worker measures *actual*
  nodes as it evaluates its bundle. If the bundle overruns its node or
  wall cap before it is done, the worker stops, returns its unfinished
  candidates to the unclaimed pool, and they are re-packed on the next
  handout — with a tighter `B` than last time, so remainder mass tends to
  re-pack into *bulk* bundles, not smaller ones.

The division of labor:

> The **exact elimination test** gets claim *count* right for the free
> mass. **Republish-on-overrun** gets *balance* right for the searched
> head, using measured nodes. Nothing anywhere predicts a magnitude.

**Monotonicity (why the bulk path is safe).** `B` only ever tightens
(solved results can only lower `best_erd`), and each candidate's
`candidate_cost_lower_bound` is a constant of the (candidate, branch) pair.
So *provably eliminated at handout* implies *still eliminated at
evaluation*: every member of a bulk bundle completes in O(1) — the engine
re-runs the same test against a `B` at least as tight and returns
immediately. A bulk bundle cannot overrun on nodes; its caps exist for
crash-reclaim windows (§7), not for misprediction.

### Correctness principle (mandatory): only exact values bound

This is an absolute invariant, not a preference:

- **No estimated value may ever be used as a pruning bound or seeded as a
  `best_erd`.** The only values that tighten `best_erd` are the *exact*
  cost of a candidate that has actually been solved and admissible lower
  bounds (`candidate_cost_lower_bound`,
  `remaining_groups_cost_lower_bound`), which are provably ≤ the true
  cost. Extra computation is always acceptable; a bound that could be
  *better* than reality is never acceptable, because it could discard the
  real optimum.
- The packer's classification uses only such exact/admissible quantities,
  so in this scheme scheduling *cannot* diverge from correctness: a stale
  `B` (read a moment before another worker tightened it) only
  under-classifies candidates as "not yet eliminated" — the safe
  direction, costing one small bundle of O(1) evaluations.
- `estimate_candidate_work` / `estimate_candidate_work_cutoff` remain in
  the engine **as telemetry and analysis utilities only** (they feed
  `candidate_accuracy`). They are never a scheduling input, never a bound.

---

## 4. The classification signal (exact, not estimated)

The engine rejects a candidate for free as soon as its admissible lower
bound meets the current bound (`_candidate_cost_lower_bound`,
`wordle_engine.py:1031`):

```
candidate_cost_lower_bound(c) = 3.0 − (number_of_response_groups(c) + has_self) / n
if candidate_cost_lower_bound(c) >= best_erd:   return immediately, ~0 nodes
```

A **weak** splitter (few, large groups) has a *high* bound and is
eliminated for free; a **strong** splitter (many small groups) has a *low*
bound, survives the test, and recurses. This test is the source of the
skew — the 93.7% of eliminated claims *are* the free mass — and it is the
packer's classification signal, evaluated against the branch's **current
real `B`** at handout time.

Properties that make it sufficient on its own:

- **It is exact.** Validated on epoch 0: 100% of eliminated candidates
  completed in ≤ 1 node. Classifying a candidate "free" by this test is a
  proof, not a guess.
- **It is conservative when `B` is loose.** Early in a branch (before any
  candidate is solved) `B` is only the seeded ceiling, so few candidates
  are provably eliminated and most pack into small bundles: more claims,
  never a wrong result. As real solved results tighten `B`, the tail
  coalesces into bulk — where the coordination win is. No fabricated
  bound, no waiting rule, no starvation: workers always have claimable
  head work.
- **It is already computed vectorized.** The §4a kernel pass
  (`candidate_stats`, full_tree_plan.md §3) produces the whole-vocabulary
  `cost_lower_bound` array in one shot when a branch is promoted; the
  packer stores it per branch and only *reads* it inside the claim
  transaction (§12).

The **ordering is the engine's existing best-first sort** (`Σk²`
ascending, strongest splitter first, C2.2): it propagates `B` fastest and
places the likely winner and its near-ties at the front, where small
bundles get them evaluated early and in parallel.

The remaining unknown — how expensive each *surviving* candidate is — is
deliberately not modeled. Survivors pack `small_count` at a time, and the
overrun path (§7) handles the heavy ones with measured nodes. The cost of
not knowing is bounded: at most `small_count − 1` candidates ride behind a
heavy one until the overrun republishes them, which is why `small_count`
is chosen from republish-safety, not from a cost model (§12).

---

## 5. The packer

Candidates are kept in the engine's best-first order (`Σk²` ascending).
The packer walks them front-to-back with a single monotonic cursor.
`eliminated(c) := candidate_cost_lower_bound[c] >= B`, with `B` read once
inside the claim transaction.

```
def next_bundle(cursor, candidates_best_first, B, small_count, count_cap):
    # cursor: lowest not-yet-claimed position in best-first order
    if cursor past end: return holes_pass()        # §6: reclaimed/republished slots
    if eliminated(candidates[cursor]):
        # bulk bundle: absorb consecutive eliminated candidates
        take while eliminated(c) and len(bundle) < count_cap
    else:
        # small bundle: fixed count of not-yet-eliminated candidates,
        # absorbing any eliminated candidates interleaved among them
        take until survivors_taken == small_count or end,
             capped at count_cap total
    return bundle
```

Behaviour:

- **Not-yet-eliminated head → small bundles.** While `B` is loose, the
  front candidates pack `small_count` per claim, so the strong splitters
  are evaluated first and in parallel, tightening `B` fast. Worst case
  (nothing eliminated yet) claim count is `n / small_count` — already ≥ 8×
  below single-candidate claiming.
- **Eliminated tail → bulk bundles.** Once `B` is tight, long eliminated
  runs absorb up to `count_cap` per claim — collapsing claim count where
  the 93.7% live. Every member completes in O(1) (§3 monotonicity).
- **The dials are counts, not work estimates.** `small_count` bounds how
  much searched mass can be stranded behind a heavy candidate before an
  overrun republishes it; `count_cap` bounds a bulk bundle's wall time and
  its crash-reclaim window. Neither requires predicting anything.

**Cost of the walk (absorbs #68).** The cursor advances monotonically
through best-first order, so the common (no-hole) path is O(1) amortized
per candidate and a branch drains in O(n_candidates) total claim work —
never #68's O(n²) per-call gap scan. Reclaimed or republished positions
are picked up by `holes_pass()` once the forward cursor is exhausted (or
on a periodic sweep), preserving #68's lazy end-of-sweep semantics. The
O(n) bound is for the no-overrun path; republish (§7) re-admits positions
and adds work proportional to the re-admitted count, not to n.

---

## 6. Queue representation and the claim API

Today `candidate_claims` is one row per candidate, PK `(branch_key, idx)`,
and `claim_candidate` returns the single lowest unclaimed `idx`. The
change:

- **Replace `claim_candidate` with `claim_next_bundle`**, which runs the §5
  packer *inside the `BEGIN IMMEDIATE` transaction* over the unclaimed
  pool, inserts one `candidate_claims` row per chosen `idx` (all
  `claimed_by` = this worker, same `claimed_at`), and returns the index
  list. One transaction per *bundle*, not per candidate — that is where
  the coordination win comes from.
- A **bundle** is the set of `candidate_claims` rows sharing
  `(branch_key, claimed_by, claimed_at)` — the candidates a worker is
  handed and evaluates together in one transaction. No new table is
  required; bundle identity is derivable. (Optionally add a `bundle_id`
  column for cleaner reclaim queries.)
- **Finalize coverage is unchanged**: a branch finalizes when all
  `n_candidates` rows are `done=1`, whoever observes full coverage.

Crucially, because *handout itself is the packer*, **every** source of
work-to-do flows through it and therefore produces count-bundled claims:

- Fresh candidates → packer bundles them.
- Reclaimed candidates from a dead worker (`reclaim_stale_claims` deletes
  the `done=0` rows → they reappear in the unclaimed pool) → packer
  **re-bundles** them.
- Republished candidates from an overrun (§7) → packer **re-bundles**
  them.

There is no code path that hands out a lone candidate. This is the
structural guarantee against the "1 candidate = 1 claim" trap.

---

## 7. Republish-on-overrun (the no-singleton rule)

A worker evaluating a bundle tracks actual nodes spent, evaluating in
best-first order (the order the candidates already carry): strong
splitters go first, tightening `B` and carrying the partial best
(invariant 2, §8). Two distinct overruns:

**(a) Cross-candidate overrun** — the *bundle* has hit its node or wall
cap and still has unfinished candidates `R`. The worker:

1. Marks the candidates it finished `done=1` (their results already
   folded into shared best).
2. Returns `R` to the unclaimed pool (delete the `done=0` rows for `R`).
3. Does **not** create `|R|` claims. The positions in `R` re-enter as
   holes, picked up by the next `claim_next_bundle` / `holes_pass` (§5,
   §6) and re-packed — against a `B` that is at least as tight as when
   they were first packed, and usually tighter (the bundle head just
   solved). The common case: yesterday's "not yet eliminated" remainder
   re-packs as provably-eliminated bulk. No estimate is scaled, because
   there is no estimate.

**(b) Within-candidate overrun** — a *single* candidate's own recursive
subtree exceeds the cap. This is not a packing problem; it is the existing
**sub-branch promotion** path (`_subbranch_solver` /
overrun-publish-at-own-level): the candidate's large response groups are
promoted as their own branches for the swarm, and the candidate's claim
completes when its promoted children finalize. Packing and promotion
compose: packing decides *which candidates share a claim*; promotion
decides *how one expensive candidate's subtree is parallelized*.

Guardrails:

- **Bounded republish depth:** cap how many times the same candidate can
  be republished before it is forced through promotion (b) instead, so a
  pathological candidate converges instead of thrashing the pool.
- **Cap a bundle's wall-time**, not just its nodes, so a worker cannot
  sit on a bundle long enough to widen the reclaim window after a crash.
  This is the only cap a bulk bundle can hit (§3 monotonicity — its
  members cannot cost nodes).

---

## 8. Invariants preserved

These must continue to hold (the ERD pruning invariants plus the
claim-level ones):

1. **Shared-best folding stays per candidate.** Bundling defers the queue
   round-trip *between* candidates, never the best update; a worker
   refreshes and may tighten the branch's shared best between candidates
   in its bundle.
2. **Sequential-sibling pruning preserved.** Candidates in a bundle are
   evaluated in order, carrying the partial best, exactly as a
   single-worker sweep. Across bundles, the existing shared-best refresh
   supplies the cross-worker bound. No new cross-subtree cancellation is
   introduced.
3. **Taint propagation unchanged.** Any candidate excluded by the depth
   budget taints the branch, winner or not.
4. **Finalization coverage unchanged.** Finalize fires only when all
   `n_candidates` slots are `done=1`.
5. **Reclaim correctness preserved and improved.** A dead worker's
   `done=0` rows are freed and re-enter the pool; the packer re-bundles
   them. The end state is identical to single-candidate claiming, never a
   swarm of singleton gaps.

An equivalence test must assert: for a fixed branch, packed claiming
yields the **same winning guess** and **same `max_depth`** as
single-candidate claiming, and an ERD **within ±1e-5** of it (not
bit-exact — claim bundling reorders the weighted sum). The pruning is the
same minimum either way (the bound is only ever tightened by real solved
results and admissible lower bounds, §3), so the optimum is unchanged;
only float summation order differs.

---

## 9. Measurement layer — **DONE (PR #76, merged; renames in PR #96)**

The measurement layer this section specified is live and has already paid
for itself twice (the epoch-0 go/no-go verdict; the epoch-2 M1 pricing in
full_tree_plan.md §7d). What exists on `main`:

- **Telemetry epochs**: `telemetry_epoch` table + `run_meta.epoch`; an
  `epoch` column on every append-only telemetry table. Epoch 0 =
  single-candidate baseline, epoch 1 = "numpy-kernel" (§4 deploy), epoch 2
  = "s5-group-words-dispatch" (§5 deploy). **The packing deploy bumps to
  epoch 3.**
- **`branch_finalize_log`**: one durable row per finalized branch
  (`branch_key`, spine, `n_words`, budget, epoch, created/finalized
  wall-clock, `nodes_spent`, `n_claims`, coordination and bundle
  diagnostics), written before `delete_branch`. Censored rows are lower
  bounds (survival semantics), never point estimates.
- **`candidate_accuracy`**: the per-claim accuracy stream
  (`candidate_cost_lower_bound`, the `B` it was computed against, the
  `erd_lower_bound_pruned` flag, actual nodes, `n_words`, budget, epoch).
  This is the stream the go/no-go verdict was computed from.
- **Cost model re-keyed** on `(policy, size_bucket, budget)` with a
  `budget = -1` aggregate fallback.
- **Deferred by scope decision**: `claim_telemetry.claim_retries` /
  `busy_wait_millis` stay NULL under single-candidate claiming (counting
  `BEGIN IMMEDIATE` busy retries belongs inside `claim_next_bundle`;
  the current `claim_candidate` relies on SQLite's C-level busy handler,
  so there is no Python-level retry to count without changing live
  locking). **Populate both in `claim_next_bundle`** — they are the
  direct lock-contention signal for the epoch-3 comparison.

---

## 10. Rollout

Already done (the two-PR rollout's PR1, plus the verdicts it enabled):

1. ~~Instrument~~ — §9 landed (PR #76); epoch-0 baseline collected.
2. ~~Re-key cost model on `(size, budget)`~~ — landed (PR #76).
3. ~~Validate the metric — go/no-go~~ — **decided**: elimination exact
   (PASS), magnitude estimate dead (FAIL) → this document's binary scheme.

Remaining (PR2):

4. **Implement packing + republish** (§5–§8): `claim_next_bundle` replaces
   `claim_candidate`; per-branch `cost_lower_bound` vector stored at
   promotion; retry/busy-wait telemetry populated (§9). Same restart
   discipline as every worker-behaviour change (SWARM.md;
   kill-old-workers). **Deploy = epoch 3.**
5. **Compare epoch 2 vs epoch 3** from `branch_finalize_log` and
   `claim_telemetry`: claims/branch, coordination fraction,
   claims-per-second, straggler (max bundle nodes) distribution, effective
   throughput, per-branch wall-time. The full_tree_plan.md §7b designed
   calibration run doubles as this comparison's data source.

---

## 11. Acceptance criteria

- Claim transactions to drain a branch drop ~100× in aggregate (the
  epoch-2 model predicts 105× at `small_count = 8`; measure against
  `branch_finalize_log.n_claims` per branch, epoch 3 vs epoch 2 at
  matched `n_words` buckets).
- Coordination share of wall drops materially; the flat ~430–655 s
  median-wall floor across small-branch buckets (§1) moves visibly
  downward. (The ≥100 ms finalize/starvation tail is only partly
  claim-count driven, so treat a specific percentage as directional, not
  a hard target.)
- No straggler regression: the max-bundle-nodes distribution in
  `branch_finalize_log` has no tail of bundles that fail to republish —
  i.e., the old chunk imbalance does not reappear.
- ERD results match single-candidate claiming **within ±1e-5**, with
  **identical winning guess and `max_depth`** (equivalence test, §8).
- **No fabricated bound:** an audit (or test) confirms `best_erd` is only
  ever tightened by real solved results or admissible lower bounds —
  never by any estimate (§3). The engine's `estimate_candidate_work*`
  utilities appear nowhere in the packer or the claim path.
- Draining a branch is O(n_candidates) total claim work, not O(n²) (the
  goal of #68), and an injected mid-sweep reclaim yields identical
  coverage to the current implementation (#68's equivalence test,
  retained).

---

## 12. Open questions

- **Choosing `small_count` and `count_cap`.** `small_count` trades
  coordination amortization against republish waste: at most
  `small_count − 1` candidates can be stranded behind a heavy head until
  the overrun fires, so pick it from republish-safety (8 is the
  conservative anchor; the model says 105× there, with diminishing
  returns above). `count_cap` bounds a bulk bundle's wall time and
  crash-reclaim window; a provably-eliminated candidate is nearly free
  but not exactly free, so cap by count and wall, and confirm with
  epoch-3 `branch_finalize_log` bundle diagnostics.
- **Cursor and bound state are shared across processes (correctness
  constraint, not optional).** Worker *processes* call `claim_next_bundle`
  concurrently, so the best-first candidate order, the forward cursor,
  and the `B` read must live in the queue DB and be advanced under the
  same `BEGIN IMMEDIATE` that inserts the claim rows — never in
  worker-local memory — or two workers pack overlapping bundles. Open
  choice: store the cursor on `active_branches` (a `next_idx`-style
  column, the one piece of #68 that survives) versus deriving the
  frontier from `candidate_claims` each call. Holes from
  reclaim/republish are handled by `holes_pass` once the forward cursor
  is exhausted (or on a periodic sweep), never by rewinding on every
  call — preserving #68's lazy end-of-sweep semantics and the
  O(1)-amortized common path.
- **Packer cost inside the lock.** The classification is one float
  comparison per candidate against a per-branch `cost_lower_bound` vector
  computed once at promotion (the §4a `candidate_stats` pass) — so the
  in-lock work is a vector scan, far cheaper than the old
  per-candidate-`group_counts` concern. Verify with §9.5-style
  decomposed coordination timings on epoch 3; if the scan still shows up,
  store the vector pre-sorted in best-first order so the scan is
  sequential.
