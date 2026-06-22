# Adaptive work decomposition for the ERD swarm

> **Executor note.** This plan is written to be executed by an agent with an
> empty context window. It cites code by **symbol name first**; any line numbers
> are **approximate** (the tree drifts) — navigate by searching for the named
> function/table/column. Run the full test suite (`python -m unittest discover -s
> . -p 'test_*.py'`) green before every commit, per `CLAUDE.md`. The
> `erd_queue.sqlite3` schema is Linux-only; all its migrations are idempotent and
> need no phone coordination (unlike `wordle_cache.sqlite3`).

## Context

The ERD precache swarm decomposes work with two static, count-based guesses:

- **Fixed chunks.** A branch's ranked candidate list is sliced into contiguous
  chunks (`ERDQueue.chunk_size_for`). Because the ranking is best-first
  (`rank_candidates_by_max_group_size_then_entropy_gain`), every expensive head
  candidate lands in chunk 0, so one worker monopolizes the costly work while
  others devour the cheap tail and stall.
- **A hardcoded promotion size.** A sub-branch is promoted to cooperative solving
  only when `len(words) >= PROMOTE_MIN_SIZE` (60), in `ERDSwarmWorker._subbranch_solver`.
  Once a sub-branch commits to inline solving, the engine grinds it to completion
  no matter how wrong that size estimate was — the source of multi-minute
  single-worker stalls.

Both use *count* as a proxy for *cost*, which it isn't: cost is heavy-tailed
(empirically near log-normal) and depends on a branch's substructure, not its
size.

This plan replaces both with **one mechanism applied at every recursion level**:
when solving a branch overruns its predicted work, publish that branch's
*unevaluated candidate remainder* to the swarm for cooperative help, carrying the
partial best as a crowdsourced bound. Promotion is driven by an online cost model
that yields `typical(size)` — a **robust** estimate (not the mean) of the
recursion-node cost to solve a branch of `n` words. Outcome: runaway solves
recruit help at their own level, cheap solves stay single-threaded with zero
coordination, and existing ERD pruning is preserved.

### Decisions grounding the design

1. **Publish the remainder over the shared natural word-list order — not a ranked
   slice.** Ranking is a *local* concern (it builds a tight bound fast on one
   worker) and is *anti-correlated* with parallel scalability (best-first piles the
   expensive candidates into the head). The published remainder needs no ranking
   and no cross-worker order agreement: claims index directly into the shared
   `all_words` list (every worker loads identical `wordle.txt`). The publisher
   marks its already-evaluated candidates *done* by their `all_words` index;
   helpers claim the rest. This drops `rank_candidates_…` / `_ranked_for` out of
   the parallel claim path entirely, and the already-computed prefix (and the tight
   bound it produced) is never recomputed.

2. **The claim atom is one candidate.** Full rename of all "chunk" vocabulary to
   "claim" vocabulary, in code *and* comments (see the rename section). The work
   unit is no longer a chunk; the language must follow.

3. **Lean toward over-promotion, not under-promotion.** A false positive (promote
   something that was actually cheap) costs a few single-candidate claim
   transactions — cheap, and *measured* by `claim_telemetry`. A false negative (fail
   to promote a tarpit) is a single worker stalling for minutes — the exact bug
   this project exists to kill. The asymmetry runs toward publishing eagerly. This
   **reverses** the "lean toward under-promoting" stance of earlier drafts, which
   was inconsistent with the problem statement.

4. **Telemetry is outbound-only.** `claim_telemetry` signals to an external
   monitor and to offline analysis. It is **never** a runtime feedback input. No
   runtime decision reads it. It is freely droppable.

**Prerequisite:** merge `origin/main` into the working branch first.
`_help_other_branch` — the waiting-worker recruiter that drains any open branch
when a worker's own branch is fully claimed — already exists on main and is
adapted below.

---

## Phase 0 — Result-protocol cleanup (foundation, behavior-preserving)

The engine core (`_solve_subset`, `evaluate_candidate` in `wordle_engine.py`)
currently encodes *why* a search failed to return an exact optimum by overloading
`None` and inferring meaning from combinations of `(cost == inf, floor, cutoff)`.
The word **"cutoff" names the result, not the reason** — was the work abandoned
because it blew the depth budget, or because a better ERD made it unable to
improve on the known bound? Those are different facts the cache must treat
differently, and the names must say which.

Replace the overloaded returns with **explicit, reason-named statuses**. Anchor on
the `CLAUDE.md` rule: *a name must include all essential context.* Recommended
spellings (executor may refine, but each must name the **reason**):

| Today (inferred) | Reason | Recommended status |
|---|---|---|
| `return None` (deadline check) | the wall-clock deadline passed | `DEADLINE_EXCEEDED` |
| `return None` (`cancel_check` fired) | an external stop was requested — in the swarm this is `stop_event` (worker shutdown/recycle) | `CANCELED` |
| exact optimum tuple | the true minimum was found | `SOLVED` |
| `(inf, None, True, False)` | depth budget too small — *proven* no winning strategy | `OVER_DEPTH_BUDGET` |
| `(ceiling, None, _, True)` (`cutoff=True`) | every candidate priced ≥ the known ERD bound — *lower bound only, do not cache* | `OVER_ERD_LIMIT` |

The two `return None` cases are detected at **separate sites** (the deadline
comparison vs. the `cancel_check()` call), so they become two distinct statuses,
not one lumped "aborted" — each names its own cause.

`evaluate_candidate`'s status strings (`'ok'`, `'cutoff'`, `'pruned'`, `'useless'`,
`'abort'`) get the same treatment: `'cutoff'` → `OVER_ERD_LIMIT`, `'pruned'` →
`OVER_DEPTH_BUDGET`, `'useless'` → a reason-named constant (e.g. `NO_INFORMATION`
— the split is `k >= n`, zero information gain), `'abort'` → propagate whichever of
`DEADLINE_EXCEEDED` / `CANCELED` the inner frame reported, `'ok'` → `SOLVED`.

The depth-floor **taint** (the property that makes a `SOLVED` result valid only at
*this* budget, currently the `floor` boolean threaded everywhere) becomes a
separate, explicitly-named `budget_tainted` flag carried alongside the status —
not inferred. The cache-reuse logic (`_cache_reuse`, `ScoreCache` reuse rule,
`maybe_finalize`'s tainted/`solve_budget` handling) reads the explicit flag.

**Constraints.**
- **Behavior-preserving.** No ERD value, cache entry, or pruning decision changes.
  The existing suite (`test_*.py`) is the oracle — it must stay green with no
  test logic changes beyond mechanical status renames.
- **Hot path.** `_solve_subset` is called millions of times. Use a lightweight
  status representation (an `IntEnum` / module-level int constants plus the payload),
  **not** a per-node object allocation. Do not regress node throughput.

Phase 0 ships independently and de-risks everything after it: once the cooperative
path's bound semantics are named explicitly, the Phase 2 reasoning below is
unambiguous.

---

## Two bounds — keep them distinct (critical for correctness)

There are **two different** branch-and-bound mechanisms in this engine. Conflating
them causes either lost pruning or a corrupt cache.

1. **Vertical alpha-beta `ceiling`** — passed *down* the recursion within one
   worker (`_solve_subset(..., ceiling=...)`, `sub_ceiling` in
   `evaluate_candidate`). A result that bottoms out against it is `OVER_ERD_LIMIT`:
   a **lower bound only**, solvability unknown, **never cached**.

2. **Horizontal crowdsourced bound** — `bound_provider` in `evaluate_chunk` reads
   the shared `active_branches.best_erd` (every `BEST_REFRESH_SECONDS`) and
   tightens pruning mid-evaluation across all workers on the same branch. This bound
   is a **real achieved value**, so pruning against it is exact and the result **is**
   cacheable. `active_branches.best_erd` is documented as, and must remain, a real
   achieved value.

A cooperatively-solved branch is finalized to the persistent cache as an exact
optimum (`maybe_finalize` → `score_cache.write`). Therefore a published branch
**carries no inherited vertical `ceiling`** (that would make its result a
non-cacheable lower bound) but **does** run under the crowdsourced bound. Workers
on a published branch trade their best-found ERD for early pruning through
mechanism 2 — that is how cooperative pruning already works, and the new
publication paths must preserve it by seeding the shared best (below).

---

## The one mechanism (Phases 1–2)

A unit of work is a `_solve_subset(X)` call. At every such call:

- Record frame-local `nodes_at_entry` (the worker's live `self._nodes`) and
  `predicted = typical(len(X))`.
- Solve depth-first, best-first, **single-threaded**, building a tight local best.
- Two trigger points publish X's remaining candidates as a cooperative branch over
  natural `all_words` order:
  - **Entry (a-priori, Phase 1):** if `typical(len(X)) >= PUBLISH_THRESHOLD`,
    publish immediately. This is the existing `subbranch_solver` hook, retargeted
    from `len < 60` to the cost-model gate.
  - **Mid-loop (overrun, Phase 2):** after each candidate, publish the remainder if

    ```
    self._nodes - nodes_at_entry  >  OVERRUN_K * typical(len(X))   # abnormal for its size
      AND  typical(remaining)      >= PUBLISH_THRESHOLD             # the handoff is worth swarming
    ```

- Once published, this worker keeps participating: it drives X to finalize
  cooperatively (reusing the claim/help/finalize loop); the result is
  `min(partial best, cooperative best)`, returned to the engine so it
  short-circuits (mirrors the existing `subbranch_solver` short-circuit).

**Gate and overrun share one `typical(size)`.** They differ only in the
*comparison*, never the statistic — so they cannot contradict each other (the gate
cannot say "fine inline" while overrun immediately says "too big" for the same
size). The gate asks "is this size *typically* big enough to swarm up front?"; the
overrun asks "is *this instance* running much hotter than typical for its size?"

**One absolute scale only.** `PUBLISH_THRESHOLD` (in node-equivalents) is the sole
absolute number — it answers "is the swarm worth it at all," which is inherent.
Everything else is *relative* to the adaptive `typical(size)`; `OVERRUN_K` is a
dimensionless ratio (~3–5), not a hardcoded node count. `PUBLISH_THRESHOLD` is a
tunable, refined **offline** later from the logged data (below) — not by any
runtime loop.

`PUBLISH_THRESHOLD` will most likely be set from the **DB-coordination break-even**:
publishing pays only when the work handed off exceeds the time spent coordinating
the handoff. To make that question answerable later, the data collection below must
capture both halves of the ratio in compatible units:

- `claim_telemetry.coordination_nanos` — wall time per claim transaction (the
  coordination cost).
- `cost_samples.wall_nanos` alongside `nodes` — so node-count converts to wall time
  (node throughput), letting the coordination cost be expressed in **node-
  equivalents**, the same unit as `PUBLISH_THRESHOLD` and `typical(size)`.

With both logged, the break-even `PUBLISH_THRESHOLD` is computable after a
representative run without any guessing.

**Same-level, not one deeper.** Overrun is a *horizontal* deficit ("X's candidate
list is more work than one worker can carry"), so the response is horizontal help
at X's level. A single tarpit candidate C is handled by the same mechanism
recursively: evaluating C recurses into `_solve_subset(S)` for its expensive
sub-branch S, which overruns at *S's* level and publishes S's remainder.

### Publishing — seeding the crowdsourced bound

- The publisher has evaluated a prefix in its *local* Σk² order (the inline sort,
  unchanged). It maps each evaluated candidate **word → `all_words` index** (one
  `{word: idx}` dict, built once on the worker) and marks those indices done, after
  seeding `active_branches.best_erd`/`best_guess` with the partial achieved
  `(best_guess, best_erd)`.
- **Seed the bound only if a real achieved best exists** (`best_guess is not
  None`). A non-achieved value (e.g. the frame's alpha-beta ceiling) must never be
  written into `active_branches.best_erd` — that would violate its "real achieved
  value" invariant and corrupt every reader's pruning.
- **Overrun may fire before any feasible candidate is found** (`best_guess is None`
  at trip time). In that case: publish the remainder with **no seeded bound**
  (helpers discover the bound themselves) and keep participating. This is just a
  refinement of "always participate in the swarm you request."
- Helpers claim remaining indices one at a time and evaluate `all_words[idx]`.
  Because the carried bound (when seeded) is already tight, they prune hard from
  their first candidate, so the lack of best-first order in the parallel path costs
  little.

### Why remainder-over-natural-order is correct and cheap

- The swarm is always `ERD_ALL`, so a branch's candidate list is always the full
  `all_words` (12,972). Natural file order is shared identically by every worker —
  no ranking, no re-rank-and-agree requirement. (This **replaces** the existing
  `active_branches` invariant comment that workers re-rank locally and agree on
  coverage.)
- Already-computed prefix work, and the tight bound it produced, is never
  recomputed.

### Over-promotion is the safe direction

Lean over-promote (decision 3). A false-positive publish costs a few single-
candidate claims on a branch that finalizes fast; the single-candidate atom makes
that benign (a worker can never be stuck holding a unit that turned out cheap). The
expensive failure is the *false negative* — an undetected tarpit stalling one
worker. Set the entry gate **permissively**. No live un-publish. A size bucket that
keeps finishing cheap pulls `typical(size)` down via time-weighted sampling until
it stops being entry-promoted.

---

## Pruning safety — invariants the implementer MUST preserve

- **Vertical budget pruning unchanged.** `sub_budget = budget - 1` and the
  `cost >= best_erd` cutoff in `evaluate_candidate` stay exactly as they are.
- **Crowdsourced pruning preserved and extended.** Published branches run under the
  `bound_provider` mechanism; the publisher seeds the shared best from its partial
  achieved value (above) so helpers prune from their first evaluation — strictly
  better than today's eager chunking, which starts cold workers before any bound
  exists.
- **No inherited vertical ceiling on published branches.** They solve to exact
  optimum (cacheable), pruned by the crowdsourced (real-achieved) bound only.
- **Sibling response groups stay sequential — in this plan.** A candidate's
  response groups are solved one at a time, largest first, each seeing the
  accumulated cost tightened by its predecessors so the candidate aborts early once
  its partial sum reaches the best. This plan does **not** parallelize them; the
  engine seam only ever publishes the *candidate list of one branch*, never the
  *response groups of one candidate*. This is a scope boundary, **not** a
  correctness prohibition: a candidate's cost is the order-independent sum of its
  groups, so groups *could* be parallelized by the same claim-and-carry-a-bound
  mechanism. Deferring it trades away inter-sibling early-abort pruning — a distinct
  pruning-vs-parallelism decision with its own break-even, belonging to the later
  granularity investigation (see Instrumentation).
- **No cancellation of in-flight cooperative subtrees.** The swarm solves to exact
  optimum for caching. Do not add subtree cancellation.

---

## Cost model → `typical(size)`

- **Unit: recursion nodes.** `self._nodes` already counts nodes (incremented once
  per node inside `_heartbeat`). Load-independent; no wall-clock conversion needed.
  Note: `self._nodes` is **lifetime-monotonic** on the worker (initialized once,
  never reset) — fix its stale "candidate evaluations this chunk" comment as part
  of the rename. Frame-local deltas (`self._nodes - nodes_at_entry`) are therefore
  valid; a parent frame's delta legitimately includes its children's nodes.
- **Storage: `erd_queue.sqlite3` (Linux-only).** New table via the stateless
  check-then-add pattern in `ERDQueue._migrate` (no `schema_migrations` table here).
- **Robust collapse, not the mean.** Node costs per size are heavy-tailed
  (near log-normal); the arithmetic mean lets one tarpit poison a bucket
  ("one bad apple ruins it for everyone"). `typical(size)` must be a robust central
  estimate — a **median / low-percentile**, computed in **log space** (the
  geometric mean / log-domain quantile is the natural collapse for a log-normal).
  Store enough state to recover a robust statistic, not just a running mean:
  - Time-weighted accumulators in **log space** (track `Σ w·ln(x)` and `Σ w` for a
    geometric-mean baseline), **plus** the second moment (`Σ w·ln(x)²`) so a spread
    is recoverable. A streaming low-quantile estimator (e.g. P²) is an acceptable
    alternative if cleaner. The exact estimator is a tunable choice; what is **not**
    optional is that it be robust and log-aware.
- **Exponential time-weighting (the model must adapt as the engine changes).** Each
  sample's influence decays with age so the model tracks the *current* cost
  structure. On update with sample `x` at `now`:
  `decay = exp(-(now - last_updated)/TAU)`; multiply every stored accumulator by
  `decay`, add the new sample's contribution, set `last_updated = now`. `TAU` ≈ one
  day of seconds sets the half-life (tunable). Continuous-time EMA — old samples
  fade smoothly; a changed algorithm re-converges within ~`TAU`.
- **Size bucketing.** Bucket `n_words` geometrically (e.g. `floor(log(n)/log(1.3))`)
  so samples stay dense in the heavy small-size region; interpolate across
  neighbouring buckets for unseen sizes. Return "cold" below a minimum effective
  weight.
- **Cold-start stochastic probing (start converging immediately).** When a bucket
  is cold and a branch of that size is about to be solved, take a stochastic sample
  of ~`2 × worker_count` candidates from the branch's `all_words` list (uniform
  random indices), evaluate just those, and seed the bucket. This is a *starting*
  estimate that the model refines over time; `2 × worker_count` gets a usable read
  cheaply and scales the probe to available parallelism (the probes are independent
  and can be claimed by the swarm). Bias toward over-promotion here is fine and
  intended (decision 3) — it errs on the safe side and washes out as organic
  samples accumulate.
- **In-memory cache on the worker.** Reading the model per frame would be thousands
  of DB reads. Load all buckets into a dict on the worker, refresh on a timer
  (reuse the `BEST_REFRESH_SECONDS` cadence pattern). `typical(size)` reads the
  dict; writes go to the table batched on the heartbeat path.
- **Cold fallback.** When the model is cold for a size: entry gate falls back to
  `PROMOTE_MIN_SIZE = 60` as the bootstrap prior; the overrun trigger stays
  **disarmed** until a real `typical(size)` exists for that bucket.

### Sampling (the model trains itself)

- **Inline frame returns (primary, dense):** on every inline `_solve_subset(X)`
  return, record `(len(X), self._nodes - nodes_at_entry)`. Accurate (one worker did
  all the nodes); covers small/medium sizes abundantly. Batch in memory, flush on
  the heartbeat path.
- **Cooperative finalize (large sizes):** add `nodes_spent` to `active_branches`,
  accumulate per-claim node deltas into it via the heartbeat path, and record
  `(n_words, nodes_spent)` in `maybe_finalize`. Covers sizes solved across multiple
  workers, where a single frame's local delta undercounts.

### Raw sample logging for offline tuning (data collection)

We do **not** yet know the true cost distribution well enough to fix the collapse
operator or the thresholds. So, alongside the live estimator, **persist raw
per-sample rows** — not just the collapsed statistic — so the distribution can be
reconstructed after a multi-day run and the estimator/thresholds chosen
empirically:

- A Linux-only `cost_samples` table: `(n_words, nodes, wall_nanos, source,
  recorded_at)`, where `source` distinguishes inline-return / cooperative-finalize
  / cold-probe. Written throttled via the heartbeat batch path (sampled every Nth
  if volume demands).
- Purpose: after a few days, fit the per-size distribution (confirm/adjust the
  log-normal assumption), choose the robust collapse (median vs p-quantile vs
  log-mean), and set `PUBLISH_THRESHOLD` / `OVERRUN_K` / `TAU` from data instead of
  guesses. This is **offline analysis**, performed by a human/script — never a
  runtime feedback loop.

---

## Instrumentation for the clustering decision (`claim_telemetry`, outbound-only)

Single-candidate claiming is a deliberate simplification: correct and order-free,
but one claim transaction per candidate makes DB coordination frequent and, for
cheap candidates, possibly more costly than the evaluation it guards. We do **not**
guess at clustering now; we instrument so a later, data-driven decision can define
a clustered work unit. **This telemetry is outbound-only — it signals an external
monitor and offline analysis, and is never read by any runtime decision** (that is
the explicit difference from the `cost_model` / `cost_samples` data above). Keep it
in its own Linux-only table, freely droppable.

**Record** (one row per claim, throttled — e.g. every Nth claim — via the heartbeat
batch path, never per-claim synchronously):

- `coordination_nanos` — wall time inside the claim transaction (`BEGIN IMMEDIATE`
  → commit): the parallelization overhead.
- `work_nodes` — recursion nodes the claimed candidate actually cost (`self._nodes`
  delta): the useful work that overhead bought.
- `n_words`, a `claim_retries`/contention indicator, `worker_count` at the time.

**Why** — they make `overhead_fraction = coordination_nanos / (coordination_nanos +
work_node_time)` observable: the quantity that decides whether clustering pays. We
also capture, per branch, the *distribution* of `work_nodes` across candidates
(cheap-tail vs expensive-head spread), because that determines what a good cluster
*is* — uniform cheap candidates cluster trivially; a few heavy candidates among many
cheap argue for size-balanced rather than fixed-count clusters.

**How to evaluate** — after a representative run: (a) median/p90 `overhead_fraction`,
and for which sizes does it exceed a break-even (say > 0.1)? (b) within a branch,
how skewed is `work_nodes` — would clustering K adjacent candidates balance load or
re-create a chunk-0 monopoly? (c) does `overhead_fraction` rise with `worker_count`
(contention) or stay flat (pure transaction cost)? Clustering is justified only
where (a) shows real overhead AND (b) shows the costs are clusterable without
re-introducing imbalance. The same telemetry informs the deferred group-level
(sibling response-group) parallelization question: both are "what is the right
granularity of a work unit" decisions, and this data makes either answerable rather
than speculative.

---

## Queue changes (`erd_queue.py`) — full chunk → claim rename

Purge **all** "chunk" vocabulary; the work unit is a candidate claim.

- **Retire chunk sizing.** Delete `chunk_size_for`, `n_chunks_for`, `chunk_range`
  and the `min_words_per_chunk` / `max_chunk_count` policy. The claim atom is one
  candidate; `n_candidates` is the claim count.
- **Rename `branch_chunks` → `candidate_claims`** (`idx` = candidate index into
  `all_words`) via an idempotent `ALTER TABLE … RENAME TO` migration in `_migrate`,
  following the existing rename pattern. Rewrite the table comment to describe what
  it *is* now (per `CLAUDE.md` comment rules) — drop the "re-rank locally and agree
  on chunk coverage" invariant; coverage is by shared `all_words` index.
- **Rename the claim methods:** `claim_chunk` → `claim_candidate`, `complete_chunk`
  → `complete_candidate`, `branch_done_chunks` → `branch_done_candidates`,
  `reclaim_stale_chunks` → `reclaim_stale_claims`, `reclaim_chunks_of_worker` →
  `reclaim_claims_of_worker`. Logic unchanged: `claim_candidate` still claims the
  lowest-indexed slot with no row — with one candidate per slot it is
  single-candidate claiming, no other change. Keep `BEGIN IMMEDIATE`, stale-claim
  reclaim, and the finalize-on-full-coverage check.
- **`active_branches`:** drop `chunk_size` (or fix it at 1; if `DROP COLUMN` is
  unavailable on the deployed SQLite, fix-at-1 is the fallback). Add `nodes_spent
  INTEGER` (cost sampling). Both via idempotent migration.
- **Heartbeat columns:** rename `chunk_idx` → `claim_idx`, `chunk_started_at` →
  `claim_started_at`, `chunks_done` → `claims_done`, `cand_chunk_size` →
  `claim_total` (or drop — always 1 now) via `RENAME COLUMN` migrations; thread the
  new names through `heartbeat()` and `heartbeats_with_branch()`.
- **New tables (Linux-only, idempotent):**
  - `cost_model` — the live robust estimator state per `size_bucket` (log-space
    time-weighted accumulators + `last_updated`).
  - `cost_samples` — raw per-sample rows for offline tuning (above).
  - `claim_telemetry` — outbound-only coordination/work rows (above).
- **Lazy registration.** Branches are NOT pre-registered. A branch becomes a
  `candidate_claims` set only when published (entry or overrun). Top-level
  user-queued branches are huge, so the gate entry-publishes them immediately;
  smaller sub-branches solve inline until they overrun.
- Honest tradeoff: one claim transaction per candidate is more DB traffic than
  chunking. Accepted deliberately — correctness now, clustering later only if
  `claim_telemetry` justifies it. Do not reintroduce grouping.

---

## Engine seam (`wordle_engine.py`)

- **Keep** the entry `subbranch_solver` hook; `_subbranch_solver` switches from
  `len(words) < 60` to the `typical(size) >= PUBLISH_THRESHOLD` gate with cold
  fallback to `PROMOTE_MIN_SIZE`.
- **Add a `mid_loop_publisher` parameter**, threaded through `_solve_subset` *and*
  `evaluate_candidate` exactly like `subbranch_solver` is (it cannot reuse
  `progress_callback`, which is deliberately top-level-only — passed as `None` into
  recursion). It is a worker-bound object (the engine has no `self`; `self._nodes`
  lives on the worker):
  - At `_solve_subset` entry: `token = mid_loop_publisher.enter(n)` → returns a
    frame token carrying `(nodes_at_entry, predicted = typical(n))`, or `None` (cold
    model / below floor → no overrun check this frame).
  - In the candidate loop, after the best update: if `token` is set, call
    `mid_loop_publisher.check(token, evaluated_words, best_guess, best_erd, budget)`
    where `evaluated_words = candidate_list[:i+1]`. `check` reads `self._nodes`,
    tests `delta > OVERRUN_K * predicted` and `typical(remaining) >=
    PUBLISH_THRESHOLD`; if tripped it registers X as a cooperative branch (prefix
    marked done; bound seeded **only if `best_guess is not None`**), drives it, and
    returns the finished result (Phase-0 status + payload) — the engine returns it
    immediately. Else returns `None`.
- Frame-local `token` is a plain local, so each active DFS frame checks its own
  overrun independently ("get help at my level"). Nodes spent driving an inner
  publication inflate outer frames' deltas — benign (the outer frame is also
  expensive and may legitimately publish next).
- The inline candidate sort (Σk²) is **unchanged** — it builds the local bound.
  Only the *published* remainder uses natural order.

---

## Swarm changes (`erd_swarm.py`)

- `_subbranch_solver`: `typical(size) >= PUBLISH_THRESHOLD` entry gate + cold
  `PROMOTE_MIN_SIZE` fallback.
- New `mid_loop_publisher` (object passed to the engine) implementing
  `enter`/`check`; on trip it builds the published branch over `all_words`, maps
  evaluated words → `all_words` indices via the prebuilt `{word: idx}` map, marks
  those done, seeds the crowdsourced best (only if achieved), then runs the
  claim/help/finalize loop on the remainder.
- `cooperative_solve` and the renamed `evaluate_chunk` (→ `evaluate_claim`) operate
  over `self.all_words` in natural order, one candidate per claim, dropping
  `_ranked_for` / `rank_candidates_…` from the swarm claim path.
- **`_help_other_branch` must move to natural order too — it is NOT reusable
  unchanged.** Today it calls `n_chunks_for` / `claim_chunk` / `_ranked_for` /
  `evaluate_chunk(..., ranked, idx, chunk_size)`. If it kept indexing `ranked[idx]`
  while published branches mark-done and claim by `all_words` index, the two index
  spaces would disagree → candidates skipped or double-evaluated and a **false
  "full coverage" at finalize → a wrong ERD written to the persistent cache.**
  Re-point it at `claim_candidate` over `self.all_words` natural order, exactly like
  `cooperative_solve`. Keep its role (idle/waiting recruiter) and `PROMOTED_PRIORITY`
  so freed workers join published branches first.
- `maybe_finalize`: record the cooperative-branch cost sample `(n_words,
  nodes_spent)` and the raw `cost_samples` row.
- Constructor / `swarm_worker` signature: drop `min_words_per_chunk` /
  `max_chunk_count`.
- Display: `node_rate` is already surfaced in `erd_search.py` (`kN/s`); clean up the
  now-vestigial `cand_rate` heartbeat argument.

---

## CLI / supervisor (`erd_search.py`)

- Remove `--min-words-per-chunk` / `--max-chunk-count` args and their plumbing into
  `swarm_worker`. Update the supervisor start log (it currently prints
  `min_words_per_chunk=… max_chunk_count=…`).
- Display already uses `node_rate`; verify only.

---

## Phasing

- **Phase 0 — result-protocol cleanup.** Reason-named statuses replacing
  `None`/tuple-flag overloading; explicit `budget_tainted`. Behavior-preserving,
  validated by the existing suite. Ships first; makes the bound semantics below
  unambiguous.
- **Phase 1 — foundation.** `cost_model` + `typical(size)` (robust log-space,
  time-decay, cold-start probing, in-memory cache); `cost_samples` raw logging;
  node sampling (inline returns + cooperative finalize); single-candidate claiming
  over natural `all_words` order (retire chunk sizing, full chunk→claim rename,
  including `_help_other_branch`); `claim_telemetry` (outbound-only); entry-gate
  publication via `typical(size)` replacing `PROMOTE_MIN_SIZE`. Delivers: no chunk-0
  monopoly, cost-driven promotion, and the data to later decide clustering.
- **Phase 2 — overrun escape hatch.** The `mid_loop_publisher` engine seam:
  same-level lazy publication on overrun, prefix-marked-done so the expensive head
  is never recomputed, bound seeded only when achieved, publish-and-keep-going when
  no feasible candidate yet. The heart of the inline-tarpit fix; sequenced last
  because it carries the engine-core change.

```mermaid
flowchart TD
    subgraph engine["_solve_subset(X)  (wordle_engine.py)"]
        E0["entry: token = enter(n)\nnodes_at_entry, predicted = typical(n)"]
        E1["inline best-first loop\n(Σk² order, builds local best)"]
        E2{"check(token):\ndelta > K*predicted\nAND typical(remaining) >= PUBLISH_THRESHOLD?"}
        E0 --> E1 --> E2
        E2 -- no --> E1
    end
    subgraph swarm["worker publisher (erd_swarm.py)"]
        P1["map evaluated words -> all_words idx\ncreate_branch (PROMOTED_PRIORITY)\nseed crowdsourced best IF achieved\nmark prefix done"]
        P2["drive remainder + keep participating:\nclaim_candidate (natural order)\n+ _help_other_branch + maybe_finalize"]
        P1 --> P2
    end
    E2 -- "Phase 2: overrun" --> P1
    SH["subbranch_solver hook\n(entry, typical(size) >= PUBLISH_THRESHOLD)"] -- "Phase 1: a-priori" --> P1
    P2 --> R["return min(partial, cooperative)\nengine short-circuits"]
    CM[("cost_model + cost_samples\nerd_queue.sqlite3")]
    E1 -. "inline return sample (n, Δnodes)" .-> CM
    P2 -. "finalize sample (n_words, nodes_spent)" .-> CM
    CM -. "typical(size), cached in-memory" .-> E0
    CM -. "typical(size)" .-> SH
```

---

## Constants (tunable; named, with starting values)

- `PUBLISH_THRESHOLD` — node-equivalent "worth-swarming" floor; the only absolute
  scale. Drives the entry gate and the overrun handoff floor. Start conservative,
  retune offline from `cost_samples`.
- `OVERRUN_K` ≈ 3–5 — dimensionless overrun ratio (`delta > K * typical(size)`).
- `TAU` ≈ 1 day of seconds — cost-model EMA half-life.
- `COST_MODEL_MIN_WEIGHT` — effective-weight below which a bucket reads cold.
- `PROMOTE_MIN_SIZE = 60` — retained only as the cold-start entry bootstrap prior.

## Files

- `erd_queue.py` — `cost_model`, `cost_samples`, `claim_telemetry` tables + read/
  update; `nodes_spent` column; full chunk→claim rename (table, methods, heartbeat
  columns); drop chunk sizing.
- `erd_swarm.py` — `typical(size)` entry gate; `mid_loop_publisher`; per-candidate
  natural-order `cooperative_solve` / `evaluate_claim` / **`_help_other_branch`**;
  cost + raw sampling; constructor signature; `self._nodes` comment fix.
- `wordle_engine.py` — Phase 0 reason-named statuses + `budget_tainted`;
  `mid_loop_publisher` threaded through `_solve_subset` + `evaluate_candidate`;
  frame-local overrun state; inline sort untouched.
- `erd_search.py` — drop chunk-sizing CLI args + plumbing; supervisor log; verify
  node-rate display.

## Reuse (do not reinvent)

- `cooperative_solve` / `_subbranch_solver` / the engine `subbranch_solver` hook.
- `_help_other_branch` — idle recruiter (adapt to natural order; do not rewrite its
  role).
- `claim_one` join-first scheduling + `PROMOTED_PRIORITY`.
- `self._nodes` + `_heartbeat` node counter; `bound_provider` crowdsourced bound.
- `claim_candidate` / `reclaim_stale_claims` / finalize-on-full-coverage (renamed).
- Queue patterns: `create_branch`, `update_branch_best`, `get_branch`, `_migrate`
  idempotent check-then-add / RENAME loop.

## Verification

- **Phase 0:** full suite green with only mechanical status renames in tests; no ERD
  / cache / pruning change. Add a focused test that each reason-named status is
  returned in its situation (abort, exact, unsolvable-within-budget,
  pruned-by-bound) and that `budget_tainted` rides an `ok`-at-budget result.
- **Unit** (`test_erd_queue_unit.py`, `test_erd_swarm_unit.py`,
  `test_erd_parallel.py`, `test_erd_scaling.py`): cost-model update + time-decay
  with synthetic timestamps (an old sample's weight decays toward zero; a changed
  regime re-converges within ~`TAU`); robust collapse is **not** the mean (a single
  injected tarpit sample barely moves `typical(size)` where the mean would jump);
  cold vs warm read; cold-start probe takes ~`2 × worker_count` samples and seeds
  the bucket; entry gate promotes/inlines correctly for a seeded model; overrun
  `check` fires per frame on a forced node delta **and** respects the
  `typical(remaining) >= PUBLISH_THRESHOLD` floor; publish-with-no-bound path when
  `best_guess is None`; single `candidate_claims` claim/reclaim/finalize;
  prefix-marked-done leaves only the remainder claimable; **`_help_other_branch`
  claims and evaluates over natural `all_words` order** (regression guard against the
  index-space mismatch); samples recorded from inline returns and cooperative
  finalize; a throttled `claim_telemetry` row has sane `coordination_nanos` /
  `work_nodes`. Update/remove the `chunk_size_for` and `PROMOTE_MIN_SIZE` tests.
- **Pruning-safety regression:** assert sibling response groups stay sequential;
  solve a small fixed branch both the old (`PROMOTE_MIN_SIZE`) and new path and
  confirm **identical finalized ERD** and total nodes not worse; assert
  `active_branches.best_erd` is only ever seeded with an achieved value.
- **Full suite:** `python -m unittest discover -s . -p 'test_*.py'` — green before
  any commit.
- **End-to-end smoke:** small swarm on a known branch; confirm (a) candidates spread
  one-at-a-time with no chunk-0 monopoly, (b) a deliberately mispredicted-cheap
  tarpit sub-branch publishes mid-solve and is picked up cooperatively rather than
  stalling one worker, (c) finalized ERD matches the pre-change value.

## Rollout

- Stop running systemd workers before deploying (stale workers against changed
  schema/engine corrupt the cache — established rule).
- `erd_queue.sqlite3` migrations are idempotent and Linux-only; no phone
  coordination (unlike `wordle_cache.sqlite3` schema changes).
