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
  call — is a correctness/performance requirement of the packer, folded into §5
  (single forward cursor over best-first order) and §12 (shared cursor state).
  #68's equivalence test (an injected mid-sweep reclaim yields identical coverage)
  is retained there.

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
4. Keep ERD results equal to single-candidate claiming **within ±1e-5** (exact
   equality is not required; floating-point reordering of the weighted sum is fine).
   The *winning guess* and `max_depth` must match exactly.

**Non-goals**

- Changing the ERD recurrence, scoring, or the per-branch node cost. Coordination
  changes; nodes do not.
- Replacing within-candidate sub-branch promotion. That mechanism
  (overrun-publish a *single candidate's* deep subtree as promoted branches) is
  orthogonal and composes with this one (see §7).

---

## 3. Core idea

Two pieces, one feedback loop:

- **Packing (static, model-driven scheduling).** Handing out a claim selects a
  *group* of currently unclaimed candidate indices whose summed predicted work ≈ a
  target `W`. Predicted-heavy candidates become singleton groups; predicted-cheap
  ones coalesce into large groups. (This is a scheduling choice only — it never sets
  a bound; see §3.)
- **Republish-on-overrun (runtime correction).** A worker measures *actual* nodes
  as it evaluates its group. If the group overruns a threshold `T = c·W` before it
  is done, the worker stops, returns its unfinished candidates to the unclaimed
  pool, and they are **re-packed into work-sized groups** on the next handout —
  with their per-candidate estimates scaled up by the observed error.

The division of labor is the whole point:

> The packer uses the **unreliable model** to get claim *count* right.
> Republish-on-overrun uses **measured nodes** to get *balance* right.
> The model only has to be roughly rank-correct at the top; the loop corrects it.

### Correctness principle (mandatory): estimates schedule, only real results bound

This is an absolute invariant, not a preference:

- **No estimated ERD may ever be used as a pruning bound or seeded as a
  `best_erd`.** The only value that tightens `best_erd` is the *exact* cost of a
  candidate that has actually been solved (or an admissible lower bound such as
  `cost_lb`, which is provably ≤ the true cost). Any bound the search prunes
  against must be **strictly ≤ the true optimal ERD** — extra computation is always
  acceptable; a bound that could be *better* than reality is never acceptable,
  because it could discard the real optimum.
- The cost model, `predicted_work`, and the overrun scale factor are **scheduling
  signals only**. They decide which candidates share a claim and how big a group
  is. They can make the swarm slower or less balanced if wrong, but they can
  **never** change which guess wins or what ERD is recorded. A wrong estimate costs
  time; it can never cost correctness.

Every mechanism below is built so that the worst a bad estimate can do is waste
work — never produce an ERD that is too low.

---

## 4. Expected-work metric (cutoff-aware)

The cost that matters is **actual** work, and actual work is dominated by the
engine's cutoff, not by un-cut recursion cost. `evaluate_candidate`
(`wordle_engine.py:1027`) rejects a candidate for free as soon as its admissible
lower bound meets the current bound:

```
cost_lb(c) = 3.0 − (number_of_response_groups(c) + has_self) / n
if cost_lb(c) ≥ best_erd:   return immediately, zero sub-branch work
```

A **weak** splitter (few, large groups) has *high* `cost_lb` and is gated for free;
a **strong** splitter (many small groups) has *low* `cost_lb`, passes the gate, and
recurses. This gate is the source of the skew — the 99.4% of <10-node claims are
gated weak splitters. So the naive `Σ typical(|g|)` (the un-cut cost) is exactly the
*wrong* signal: it is largest for the weak splitters that actually cost ~0. Ranking
by it descending would isolate the free candidates and bury the expensive strong
splitters in a bulk group — recreating the imbalance this design exists to prevent.

The correct estimate is **relative to the current real bound `B = best_erd`**:

```
predicted_work(c | B) =
    0                          if cost_lb(c) ≥ B     # provably gated — EXACT, not a guess
    else  ≈ Σ over the groups g that would be solved
          before the partial-sum cutoff,  typical(|g|, budget − 1)   # estimate, ≤ un-cut cost
```

Two properties make this both correct and safe:

- **The gated branch is exact, not estimated.** `cost_lb` is the same admissible
  lower bound the engine uses; `cost_lb(c) ≥ B` means `c` *cannot* beat `B` and
  *will* be cut for free. Predicting ~0 work for it is correct, not a guess.
- **It is conservative when `B` is loose.** Before any candidate is solved, `B` is
  only the budget ceiling, so few candidates are gated and most are predicted
  non-trivial → packed into small groups. That is the safe direction: more, smaller
  claims (a little extra coordination) rather than a bulk group that hides a
  contender. As real solved results tighten `B`, more candidates become provably
  gated and coalesce into bulk groups — where the coordination win is.

`B` here is **always the real `best_erd`** from solved candidates — never a model
estimate (§3 correctness principle). The cost model only sizes the *non-gated* head,
and only ever finer; it can never cause a candidate to be skipped or an ERD to be
wrong.

The estimate reuses existing machinery: `typical(k, b)` is
`ERDQueue.get_cost_typical(policy, n_words)` (`erd_queue.py:893`,
`exp(weighted_log_sum / weight_sum)`), wrapped by `_BranchWorker._typical(n)`
(`erd_swarm.py:504`); spread is `get_cost_spread` (`erd_queue.py:905`). Both are
keyed on `n_words` only today — §9.4 adds the `budget` key. `cost_lb` and the group
sizes come from the same `cache.group_counts(c, branch_words)` pass the best-first
sort already does (`_solve_subset`, `wordle_engine.py:1097`, sort at `~1182`).

**Ordering is the engine's existing best-first sort** (`Σk²` ascending, strongest
splitter first, `wordle_engine.py:1176`): it both propagates `B` fastest and places
the candidates most likely to be non-gated at the front, where the packer keeps
groups small.

**The model is weak — validate before relying on it.** Its self-reported spread
(`weighted_log_sq` → σ) reaches σ(ln nodes)=3.3 (737× band); raw actuals for
`n_words=30` span 16,871 → 26,512,414 nodes (1,571×). So the *magnitude* estimate
for non-gated candidates is noisy — which only affects how finely the head is split,
the safe direction. The *gating* decision, which does the real work of separating
cheap from expensive, is exact. Even so, confirm the metric empirically before
building the packer: under single-candidate claiming (epoch 0) each claim's
`work_nodes` *is* a candidate's actual cost, so log `predicted_work(c | B_at_eval)`
beside it (§9.3) and require a strongly positive correlation as a go/no-go gate
(§10, §11). If it is not positive, the metric is re-derived before any packer code.

---

## 5. The packer

Candidates are kept in the engine's best-first order (`Σk²` ascending — §4). The
packer walks them front-to-back with a single monotonic cursor and grows the group
as the bound proves later candidates free. `predicted_work(c | B)` is evaluated
against the branch's **current real `best_erd`** at handout time.

```
def next_group(cursor, candidates_best_first, B, W, count_cap):
    # cursor: lowest not-yet-claimed position in best-first order
    if cursor past end: return holes_pass()            # §6: reclaimed/republished slots
    group, total = [], 0
    while cursor in range and len(group) < count_cap:
        w = predicted_work(candidates[cursor] | B)     # 0 if provably gated (cost_lb >= B)
        if group and total + w > W:                    # don't exceed the work target
            break
        group.append(cursor); total += w; cursor += 1
        if w >= W:                                     # a single non-gated heavy item
            break                                      # stands alone
    return group
```

Behaviour:

- **Non-gated head → small groups / singletons.** While `B` is loose, the front
  candidates carry real predicted work, so groups stay small and the strong
  splitters (the likely winner and near-ties) are evaluated first and in parallel —
  tightening `B` fast. A single item with `predicted_work ≥ W` stands alone.
- **Gated tail → bulk groups.** Once `B` is tight, every further candidate has
  `predicted_work = 0`, so the loop absorbs candidates up to `count_cap` into one
  group — collapsing claim count where the 99.4% live.
- **No fabricated bound, no gate, no starvation.** There is no "wait until `B` is
  set" rule, and *nothing seeds `B`* (§3). When `B` is loose the tail is simply not
  *provably* free yet, so it packs small (correct, slightly more coordination); when
  real results tighten `B`, it packs bulk. Workers always have claimable head work,
  so the gate-induced starvation failure mode cannot arise.

A bulk group could otherwise absorb an unbounded run of zero-predicted candidates,
so cap each group by **count (`count_cap`) and wall-time**, not only by `W` (§7): a
provably-gated candidate is nearly free but not exactly free, and a crashed worker
holding a huge group widens the reclaim window. `W`, `count_cap`, and the wall cap
are the tuning dials (§12): `W` sets non-gated-head granularity at, e.g., ~100k
nodes (~1.5 s compute), making the ~21 ms coordination ~1.5% overhead; the caps
bound the gated-tail group.

**Cost of the walk (absorbs #68).** The cursor advances monotonically through
best-first order, so the common (no-hole) path is O(1) amortized per candidate and a
branch drains in O(n_candidates) — never #68's O(n²) per-call gap scan. This is
#68's monotonic `next_idx` cursor generalized: it advances by a group, not by 1, so
#68's separate change is unnecessary. Reclaimed or republished positions are picked
up by `holes_pass()` once the forward cursor is exhausted (or on a periodic sweep),
preserving #68's lazy end-of-sweep semantics. The O(n) bound is for the no-overrun
path; republish (§7) re-admits positions and adds work proportional to the
re-admitted count, not to n.

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
3. **Scales the scheduling estimate** for each `r ∈ R` by the observed error on
   this group, `actual_so_far / predicted_so_far`, and writes the scaled work back
   to the branch's work vector. This scaled value is a **scheduling signal only**
   (§3): it makes the re-pack produce smaller groups for dense work; it never
   touches `best_erd` or any ERD.
4. Does **not** create `|R|` claims. The positions in `R` re-enter as holes in the
   best-first order, picked up by the next `claim_next_group` / `holes_pass` (§5,
   §6) and re-packed. Because their estimates were just scaled up, the re-pack
   produces *smaller* groups where work turned out dense — finer granularity exactly
   where needed, still multi-candidate wherever items remain individually cheap.

Evaluate within a group in **best-first order** (the engine's `Σk²` ascending order
the candidates already carry): the strong splitters go first, tightening `B` and
carrying the partial best (invariant 2, §8), and the remainder `R` left by an
overrun is the later, more-likely-gated tail — so the common case re-packs into a
few groups rather than re-isolating a heavy item repeatedly.

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
**same winning guess** and **same `max_depth`** as single-candidate claiming, and
an ERD **within ±1e-5** of it (not bit-exact — claim grouping reorders the weighted
sum). The pruning is the same minimum either way (the bound is only ever tightened
by real solved results, §3), so the optimum is unchanged; only float summation order
differs.

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

### 9.3 Per-candidate accuracy stream (collect in epoch 0 — it gates the packer)

The packer's metric (§4) is the one assumption that can sink the design, so measure
it **before** building the packer. Under single-candidate claiming (epoch 0) each
claim evaluates exactly one candidate, so that claim's `work_nodes` *is* the
candidate's actual cost — the ideal place to validate. Log per claim:
`predicted_work`, **the bound `B` it was computed against**, `cost_lb`, whether the
candidate was gated, `actual_work_nodes`, `n_words`, `budget`, `epoch`.

This stream lets us (a) confirm `predicted_work(c | B)` correlates strongly and
*positively* with actual cost — the §11 go/no-go gate; (b) verify the gating split
(`cost_lb ≥ B` ⇒ ~0 nodes) holds in practice; (c) detect systematic bias to set the
§7 scale factor; (d) decide the cheapest adequate proxy. Logging `B` and `cost_lb`
alongside is essential — without them a gated candidate's near-zero cost looks like
a wildly wrong prediction rather than a correct one.

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
`claim_retries` and a `busy_wait_millis` on each claim (see §9.6), and decompose
`coordination_millis` into components (claim-scan / complete / finalize /
idle-search). Then a run with `--workers` varied gives the contention signal
directly. Cheap to add alongside the epoch work.

### 9.6 Populate the telemetry the schema already declares but never writes

Several columns exist in the schema but are never written, so the diagnostic data
they were meant to hold does not exist. Fill them as part of this work — they are
the cheapest measurements available because the tables already exist:

| Column | State today | Fix |
|---|---|---|
| `cost_samples.wall_millis` | INSERT omits it (`erd_queue.py:983`); 0/1378 rows | Pass wall time into `add_cost_sample`; it is the only per-solve wall figure. |
| `claim_telemetry.claim_retries` | INSERT omits it (`erd_queue.py:992`); 0/72M rows | Count `BEGIN IMMEDIATE` busy retries in `claim_next_group` and pass them — the direct lock-contention signal (§9.5). |
| `backstop_telemetry.predicted_nodes` | Wired but always passed `None`; 0/30 rows | Compute `get_cost_typical(n_words, budget)` at frame entry and pass it — the paired predicted-vs-actual the model accuracy work (§9.4) needs. |
| `active_branches.created_at / finalized_at / nodes_spent` | Populated, then destroyed by `delete_branch` at finalize | Copy into `branch_finalize_log` (§9.2) before delete, so per-branch wall span and node cost survive. |

None of these change behaviour; they only stop discarding measurements the schema
was built to keep. The `branch_finalize_log` (§9.2) plus these four are the complete
set needed to answer "how long / how much did branch X and its subbranches cost,"
which is unanswerable today.

### 9.7 Measurement imbalance: bound every unit, censor honestly

Past runs were derailed by a single unit of work taking hours; measurement must not
repeat that. The risk during the epoch-0 baseline: one candidate (a true contender,
or the winner) recurses for hours, stalling a worker and producing one
gigantic-but-rare sample that distorts the cost model.

Strategy:

- **Bound every measurable unit.** A single candidate never runs monolithically:
  the existing sub-branch promotion (`_subbranch_solver`, §7b) already publishes a
  candidate's large response groups to the swarm when they exceed a threshold. Keep
  that active during measurement so no claim runs for hours; the candidate's cost is
  then distributed across promoted-branch `branch_finalize_log` rows and
  reconstructable offline by spine.
- **Right-censor, don't wait.** When a unit hits the node/wall cap, record the
  sample as **censored** (`actual ≥ cap`, `censored = 1`, `promoted = 1`) rather
  than blocking for the true value. The cost-model fit must treat censored samples
  as lower bounds (survival-style), not as exact points, so a handful of capped
  monsters do not bias `typical(k)` — and do not masquerade as cheap.
- **Cap is the same mechanism as §5/§7.** The per-unit node/wall cap used in
  measurement is exactly the group wall-cap and the within-candidate promotion
  threshold from the packer, so epoch-0 and epoch-1 bound work the same way and the
  baseline is comparable.
- **Flag imbalance in the finalize log.** `branch_finalize_log.max_group_nodes` (and
  a `censored_units` count) surface any unit that ran away, so a regression is
  visible in the data instead of as a mysteriously stalled worker.

The guiding rule mirrors §3: when in doubt, **spend more compute, never fabricate a
value** — a censored lower bound is honest; a guessed point estimate is not.

---

## 10. Rollout sequence

1. **Instrument (epoch 0 still running).** Land §9.1–9.7 — epoch table,
   `branch_finalize_log`, per-candidate accuracy (§9.3), the never-written columns
   (§9.6), measurement caps and censoring (§9.7), retry/decomposed coordination.
   Stop workers, deploy, restart (per the kill-old-workers rule). Collect a clean
   baseline under single-candidate claiming.
2. **Re-key cost model on `(size, budget)`** (§9.4) and verify predicted-vs-actual
   variance drops materially on the new key.
3. **Validate the metric — go/no-go gate.** From the §9.3 epoch-0 stream, confirm
   `predicted_work(c | B)` correlates strongly and positively with actual cost and
   the gating split holds. **Do not build the packer until this passes**; if it
   fails, re-derive the metric (§4) first. This is the one assumption that can sink
   the design.
4. **Implement packing + republish** (§5–§8) behind the same restart discipline.
   Bump to epoch 1.
5. **Compare epoch 0 vs epoch 1** from `branch_finalize_log` and `claim_telemetry`:
   claims/branch, coordination fraction, claims-per-second, straggler (max group
   nodes) distribution, effective throughput, per-branch wall-time.

---

## 11. Acceptance criteria

- **Metric validation passed** (§10 step 3): `predicted_work(c | B)` correlates
  strongly and positively with epoch-0 actual cost, and the `cost_lb ≥ B` gating
  split matches observed near-zero-node claims. This is a prerequisite, not a
  post-hoc check.
- Claim transactions to drain a deep branch drop from `n_candidates` to ≈
  `total_predicted_work / W` (target ~100× fewer).
- Coordination share of wall (epoch 1 `claim_telemetry`) drops materially — most of
  the per-claim fixed-overhead bands disappear; the largest drop is at three or more
  guesses played. (The ≥100 ms finalize/starvation tail is only partly claim-count
  driven, so treat a specific percentage as directional, not a hard target.)
- No straggler regression: the `max_group_nodes` distribution in
  `branch_finalize_log` has no tail of multi-`W` groups that fail to republish —
  i.e., the old chunk imbalance does not reappear.
- ERD results match single-candidate claiming **within ±1e-5**, with **identical
  winning guess and `max_depth`** (equivalence test, §8).
- **No fabricated bound:** an audit (or test) confirms `best_erd` is only ever
  tightened by real solved results or admissible lower bounds — never by a cost-model
  estimate (§3). A deliberately corrupted cost model must change timing only, never
  any recorded ERD.
- Draining a branch is O(n_candidates) total claim work, not O(n²) (the goal of
  #68), and an injected mid-sweep reclaim yields identical coverage to the current
  implementation (#68's equivalence test, retained).

---

## 12. Open questions

- **Choosing `W`, `count_cap`, and `c`.** Start `W` from `coord_millis / node_time`
  so a group's compute dwarfs its coordination; set `count_cap` so a fully-gated
  bulk group still completes within the wall cap; choose the overrun factor `c`
  (e.g. 2–4) from the §9.3 accuracy data — large enough not to thrash, small enough
  to cap stragglers.
- **Cursor and bound state are shared across processes (correctness constraint, not
  optional).** Six worker *processes* call `claim_next_group` concurrently, so the
  best-first candidate order, the forward cursor, and the `B` read must live in the
  queue DB and be advanced under the same `BEGIN IMMEDIATE` that inserts the claim
  rows — never in worker-local memory — or two workers pack overlapping groups. Open
  choice: store the cursor on `active_branches` (a `next_idx`-style column, the one
  piece of #68 that survives) versus deriving the frontier from `candidate_claims`
  each call. Holes from reclaim/republish are handled by `holes_pass` once the
  forward cursor is exhausted (or on a periodic sweep), never by rewinding on every
  call — preserving #68's lazy end-of-sweep semantics and the O(1)-amortized common
  path.
- **Packer cost inside the lock.** Computing `predicted_work(c | B)` per candidate
  while holding `BEGIN IMMEDIATE` lengthens the write transaction. `cost_lb` is
  cheap (one `group_counts`), but if it shows up in §9.5's decomposition, precompute
  the per-candidate `cost_lb`/work vector once when the branch is promoted (the same
  pass as the best-first sort) and only read it inside the lock.

(Resolved during review, no longer open: the earlier "gate the tail until `best_erd`
is set" idea is dropped — it risked seeding/starvation. The bound-relative metric
(§4) packs the not-yet-provably-gated tail *small* instead, so nothing is gated
behind a bound and nothing is seeded.)
