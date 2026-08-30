# Issue #118 — Resumable cut branches

Retaining a cut branch's exact-valid partial work so a later exact solve can
resume instead of restarting: the carry-forward piece of #109.

## Status and scope

**Not built, and not to be built for performance.** Issue #118 was closed
`NOT_PLANNED` on 1 Aug 2026, and an offline measurement on 6 Aug 2026 (epochs 9
and 10) settled the question the issue was gated on. The measured result agreed
with the code-level prediction in §1: the reachable prize is small, and the
repeated cut work is the *cheap* part of the ladder.

This document is retained for §1 and §5, not for its plan. §1 is a reading of
the finalize and bound-provider paths that says *why* retention cannot pay,
independent of any epoch's numbers; that argument survives the issue and would
otherwise be re-derived from scratch. §5 names a separate cacheability bug that
is still unfiled. §3 and §4 are costing sketches for work that was never
started, kept only so the price is visible next to the benefit.

Code citations were re-resolved against `main` on 29 Aug 2026. They are a
reading aid, not an anchor — take the symbol name as authoritative.

---

## 1. What the code says is actually retainable

The issue's safety rule asks for a split between exact-valid claims and
ceiling-priced ones. Reading the finalize path, that split is nearly
degenerate on a cut branch, and the exact-valid side is largely persisted
already.

**A cut branch has no achieved best.** `maybe_finalize` classifies a branch as
a cut only when `best_guess is None and cut_occurred` (`erd_swarm.py:1670`),
and `best_erd` on `active_branches` is written only by `update_branch_best`
(`erd_queue.py:3233`), which only a `SOLVED` candidate calls
(`erd_swarm.py:1375`). So on any branch that finalizes as a cut, `local_best`
and `shared_best` were `None` for every claim, and `_bound_provider`'s minimum
over `(local_best, shared_best, branch_ceiling)` (`erd_swarm.py:1258`) was the
ceiling alone. Every `OVER_ERD_LIMIT` price-out on a cut branch is therefore
ceiling-priced — threshold-only — with no "pruned against the branch's own
achieved best" subset to rescue.

**The claims split three ways, and only one way is exact-valid:**

| Claim outcome | Retainable? |
|---|---|
| `OVER_ERD_LIMIT` from a worker evaluation | No — priced against the ceiling only |
| `done=1` from the one-level ERD prune sweep | No — swept against `bound = min(best_erd, ceiling)`, which is the ceiling here (`erd_queue.py:2847`) |
| `OVER_DEPTH_BUDGET` (proven infeasible at this budget) | Yes — a proof no bound can manufacture |

**Ceiling-priced claims are provably worthless to the consumers that record a
miss.** A miss is recorded only when the recorded cut fails
`budget <= cut_budget and cut_bound >= ceiling` (`erd_swarm.py:2056`), i.e. the
consumer wants an exact result, a *higher* ceiling than the bound, or a larger
budget. A retained ceiling-priced claim proves "this candidate ≥ C" for the old
ceiling C. For a consumer wanting ceiling C′ > C that proves nothing. For an
exact consumer, the retained bound could only prune once the resume achieves a
best B\*, and the cut already proved B\* ≥ C, so `C ≥ B*` holds only in the
exact-tie case. A consumer at a larger budget is refused by the join rule
before any of this. There is no configuration in which the expensive retained
work pays.

There is also no tighter bound available to retain: `evaluate_candidate`'s cut
returns are hard-fail, carrying `None` where the bound would go
(`wordle_engine.py:1618`, `:1637`, `:1645`), and `_solve_subset`'s own cut
return hands back the ceiling verbatim (`:1858`). Neither ever computes a
fail-soft value below the ceiling. Retaining per-candidate lower bounds instead
of a taint bit would require making the engine fail-soft first, which is a
separate and much larger change.

**The one retainable class is mostly persisted already.** Proving a candidate
infeasible at `budget` writes every sub-branch disproof it establishes to the
score cache as a loss (`score_cache.write_loss`, `wordle_engine.py:1863`), and
a loss is durable and reusable at any budget the recorded one covers. Likewise
the exact sub-branch optima found under the cut are in the score cache. So the
issue's framing — a later consumer "re-solves the branch from nothing" — is
true of `candidate_claims` but not of the compute: the re-solve re-walks a warm
tree. What retention would save is the shallow re-walk over already-cached
results, not the deep search.

**Net:** the addressable prize is the cost of re-walking the infeasible
candidates of a cut branch against a warm score cache, on the subset of misses
whose budget matches the cut's. That is much smaller than "the cut's original
cost", which is the number the issue's framing invites.

---

## 2. The gate, and what it found

The gate was run offline against `runtime/erd_queue_telemetry.sqlite3` on the
rocky box — the attached-schema telemetry file, not `runtime/erd_queue.sqlite3`
(see AGENTS.md). It ran as a read-only scratchpad script rather than committed
code, because it runs once; `analyze_swarm_telemetry.py` is the committed
precedent for a *repeatable* gate, which this is not.

It was run on epochs 9 and 10 rather than the epoch 5 the issue names, and it
measured the savings ceiling directly instead of going through the miss-row
proxy the metrics below were built on. Both changes made the answer stronger,
not weaker: the direct measure bounds every variant of the idea at once,
including the two §4 does not cover.

### What it found

- **The whole idea is capped at 7.26% of epoch compute** (epoch 10; 7.41% on
  epoch 9), measured as Σ(nodes per `branch_key`) − max(nodes) over branch keys
  that finalized more than once. This reconciles with the epoch-9 gate's
  5.6–6.0%. An earlier framing of "repeats are 51% of compute" is true but is
  not a savings figure: solving once still pays the most expensive solve.
- **A branch-level bound recovers 0.00%** of 39.5 billion expensive re-solve
  nodes. `min_x cost_lower_bound(x)` is an unconditionally valid ERD lower
  bound needing no cut at all, but it only covers cuts that already cost zero
  nodes. Where real search happened it sits *below* the ceiling (median −0.09).
- **Per-candidate retention — the §4 design — caps at 12.7%** node-weighted,
  with a median of 0%. A cut at ceiling C1 leaves bounds only for the
  candidates whose closed-form bound is under C1; a later request at C2 must
  search a strict superset, and the new members have no stored bound.
- **Why every number is small:** over 33,960 ladders that ended in an exact
  solve, the entire cut ladder cost 2.05 billion nodes and the single exact
  solve 13.16 billion. The ladder is **6.4× cheaper than the solve it defers**.
  Escalating ceilings are cheap probes, not redundant work — this is alpha-beta
  working as intended. The repeated work is the cheap part.

That last point is the one worth carrying forward. It inverts the issue's
premise: the ladder is not waste to be recovered, it is the mechanism that
keeps the expensive solve from being entered blind.

A side finding, not about retention: 90.34% of cut ceilings already sit on the
branch's 1/n lattice, and the 9.66% that do not would gain a mean 0.039 ERD — a
full lattice step — from rounding. Worth doing for exactness if it is ever
convenient, not for speed.

### The metrics the plan specified

Kept because they describe the miss-row join, which no committed report
performs and which any future question in this area would need to rebuild. Per
(branch_key, budget) that recorded at least one miss, walk
`branch_finalize_log` ordered by `recorded_at` and pair each miss with the
`outcome='cut'` row before it and the next finalize row after it.

- **M1 — volume.** Distinct (branch_key, budget) pairs with a miss, and raw
  miss rows, against the count of `outcome='cut'` finalizes in the epoch. Both
  counts, because a miss is logged per promotion attempt.
- **M2 — miss classification.** Each miss row assigned to exactly one class by
  the first failing test in `_read_satisfying_cut`'s order: `budget >
  available_budget` (budget-short), `wanted_ceiling IS NULL` (exact wanted),
  `wanted_ceiling > available_bound` (higher ceiling wanted).
- **M3 — re-solve cost.** `nodes_spent` and `finalized_at - created_at` of the
  finalize that followed each miss, totalled, against the epoch's
  `SUM(nodes_spent)`.
- **M4 — addressable share.** M3 restricted to misses whose re-solve budget
  equals the cut's budget. The join rule makes retained work meaningless
  otherwise, so this — not M3 — is the number retention could attack, and even
  within it only the infeasible-candidate fraction is reachable.
- **M5 — warm-cache check.** Re-solve `nodes_spent` against the preceding cut's
  `nodes_spent` for the same branch. A small ratio confirms §1's "the re-solve
  re-walks a warm tree" claim empirically.

The thresholds fixed before looking were: M4 below 1% of epoch nodes closes the
issue, above 5% builds, in between goes to §3. The direct measurement came in
at 7.26% for the *entire* idea — an upper bound over every variant, where M4 is
a strictly smaller subset — so the close was the only reading available.

### Biases to state alongside any re-run

- `add_cut_result` is `INSERT OR REPLACE` per branch (`erd_swarm.py:1719`), so
  only the latest cut is ever visible to compare against. **A stale-bound miss
  may be attributed to the wrong cut.**
- A miss is logged per promotion attempt, not per re-solve, so raw miss counts
  **overcount** re-solves.

The plan also warned that `recover_active_branches` wiped `cut_results` on
supervisor restart, making misses undercount. That is no longer true: the
method now leaves `cut_results` untouched on the grounds that each row is a
proof rather than a delivery channel to a killed waiter
(`erd_queue.py:3686`) — independently adopting what §4 below proposed as a
deliberate divergence. A re-run should not carry the undercount caveat.

---

## 3. The savings-side measurement gap (not closed)

The gate above measured what re-solves cost. It could not measure what
resumption would have saved from telemetry alone, because nothing records how
many of a cut branch's done claims were the exact-valid (infeasible) kind. The
direct 7.26% ceiling made closing that gap unnecessary. Smallest change that
would close it, if some later question needs it:

- Two counters on `active_branches`, incremented in the `UPDATE` that
  `add_nodes_spent` already issues per claim (`erd_queue.py:5739`), so no extra
  write lands on the hot path: `infeasible_candidates` and `infeasible_nodes` —
  the `OVER_DEPTH_BUDGET` count and its node cost.
- Both copied into `branch_finalize_log` at finalize, alongside
  `bulk_done_candidates`, which is the existing precedent for exactly this
  shape.
- Idempotent `_add_columns` migrations in `ERDQueue._migrate` (queue file,
  Linux-only — no phone coordination needed).

The savings ceiling would then be directly readable: on a flagged branch, the
retained fraction is `infeasible_nodes / nodes_spent` of the *original* cut,
and the resume saves at most the warm-cache re-walk of that fraction.

Cost: two integer columns and one arithmetic change in an existing statement.

---

## 4. Retention (not built)

Sketched so the cost stays visible against the measured benefit. §2 measured
this exact design at a 12.7% node-weighted ceiling with a median of 0%.

**Schema (queue file, `ERDQueue._migrate`):**

- `candidate_claims.exact_valid INTEGER NOT NULL DEFAULT 0`. Set in the
  `UPDATE` `complete_candidate` already issues (`erd_queue.py:3135`), so no
  extra row write. Definition: 1 iff the claim's status was
  `OVER_DEPTH_BUDGET` (proven infeasible at this budget) or `SOLVED` (exact
  cost folded into the best). One-level ERD prune inserts leave it 0. This
  requires passing the status into `complete_candidate`, which currently takes
  only `(branch_key, idx)`.
- `retained_cut_claims(branch_id, budget, idx)`, `PRIMARY KEY (branch_id,
  idx)`. Written at cut finalize, before `delete_branch`, from
  `done = 1 AND exact_valid = 1`. On a cut branch that is the infeasible set
  only — a `SOLVED` row implies `best_guess` is set, which implies not a cut.
- `active_branches.retained_done_candidates`, so the finalize log's
  `n_claims = completed_candidates - bulk_done_candidates` arithmetic
  (`erd_swarm.py:1669`) stays honest and seeded claims are not reported as
  worker evaluations.

**Seed path:** in `create_branch` (`erd_queue.py:2412`), when a branch is
created at budget B and `retained_cut_claims` holds rows for (branch_id, B),
insert them as `done = 1, claimed_by = 'retained-cut'` and set the counter.
Budget equality is required, never `<=`: infeasibility at B says nothing at a
larger budget, and the join rule requires equality regardless.

**Restart soundness:** retained infeasibility proofs survive a restart for the
same reason score-cache losses do. `recover_active_branches` already treats
`cut_results` this way, so retention would follow the rule the method now
states rather than diverging from it.

**Never** seed a ceiling-priced claim, under any consumer. That is the
inherited #109 safety rule, and §1 shows it also has no upside to trade
against.

**Tests:** `tests/test_erd_queue_unit.py` (retain/seed round trip; budget
inequality refuses the seed; a ceiling-priced claim is never retained;
restart preserves retention), `tests/test_erd_swarm_unit.py` (cut finalize
retains, exact resume consumes and still finalizes at the same optimum as an
unseeded solve), `tests/test_branch_id_migration.py` (the new tables carry
through the branch_id rebuild).

---

## 5. Adjacent, out of scope, still unfiled

The issue's interaction note names a prefix budget-taint gap in
`_MidLoopPublisher.check`. It is real, independent of everything above, and
still present on `main` as of 29 Aug 2026.

`_solve_subset` accumulates `node_floor = node_floor or budget_tainted` across
its candidate loop (`wordle_engine.py:1830`) and then returns the publisher's
result outright when a handoff fires (`:1838`), discarding that accumulated
taint. The returned tuple is `cooperative_solve`'s, which carries only the
*published* branch's taint — it cannot know what the prefix proved. Meanwhile
`check` marks the evaluated prefix done on the published branch
(`erd_swarm.py:506`) and seeds only `update_branch_best`; it never calls
`mark_branch_tainted`. So a branch whose prefix was budget-tainted can
finalize as untainted-exact and be written to the score cache with
`solve_budget = NULL`, i.e. claimed reusable at any budget ≥ its recorded
maximum remaining depth.

This is a cacheability-correctness bug, not a resumption question. It should be
its own issue, and it should not have been bundled into #118's gate.

---

## 6. Outcome

The gate ran and closed the issue, which is what §1 predicted it would do.

§1's code-level argument was that retention's reachable prize is the warm-cache
re-walk of a cut branch's infeasible candidates, on the budget-matched subset
of misses, and that the expensive ceiling-priced mass is unusable by
construction to every consumer that can record a miss. The measurement agreed
and went further: the ceiling over *every* variant of the idea — branch-level
bound, per-candidate retention, cold storage — is 7.26% of epoch compute, and
the per-candidate design §4 describes reaches 12.7% of a much smaller pool with
a median of 0%.

The finding that outlived the issue is the 6.4× ratio. A cut ladder is not
duplicated work waiting to be recovered; it is a sequence of cheap probes that
each cost a fraction of the exact solve they defer. Any future proposal to
"stop redoing cut work" should be checked against that ratio first.
