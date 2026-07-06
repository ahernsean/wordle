# Full best-ERD tree for all ~13k openers — implementation plan

The goal: populate `branch_best_by_policy` (policy `ERD_ALL`) with the best-ERD
result for every response branch of every opener in `wordle.txt` (~12,972
words), on one small Linux box, in weeks rather than years — and make the
result usable from the iPhone.

This document is written to be executed section-by-section by lower-capability
agents. Part I is shared context every implementer must read. Part II is the
non-negotiable rules. Part III is the work, one section per agent-sized task.

Related artifacts:

| Artifact | What it is |
|---|---|
| Issue #77 / PR #80 | Remove the `queue-add` 300-answer-word cap — **closed/merged**; §1 records what landed |
| PR #76 (`claude/issue-67-review-8v65i1`) | Adaptive-claim-packing **measurement layer** — **merged**; the epoch-0 run it fed is complete and analyzed (C3, U6) |
| `adaptive_claim_packing.md` | The packing plan PR #76 instruments; the U6 verdict killed its magnitude metric, so the document needs revision to the surviving binary scheme — its own track, orthogonal to this plan |
| PR #78 (`claude/wordle-search-algorithm-kv7kij`) | This document — the **integration branch**; section PRs merge into it (§2 = PR #84, §3 = PR #85, both merged) |

The three multiplicative levers this plan is built on, against a measured
baseline of **~900 branch sweeps/day**. That figure is derived, not quoted:
`adaptive_claim_packing.md` §1 reports 70.56M claims over the ~6-day
Jun 23–29 run, one claim = one candidate under single-candidate claiming,
and a branch sweep is ~12,972 candidates — 70.56M / 12,972 ≈ 5,400 sweeps
/ 6 days ≈ 900/day (an upper bound on distinct branches, since promoted
sub-branches also consume claims):

| Lever | Expected gain | Where |
|---|---|---|
| Claim packing (coordination overhead ~83% of in-loop wall) | ~5–6× | `adaptive_claim_packing.md` — go/no-go decided (U6): the magnitude metric is **dead**; the lever survives as exact ERD-lower-bound pruning + count-bundling + republish-on-overrun, with a measured claim-count reduction of 44–231× depending on bundle cap |
| Vectorized partition kernel (NumPy) | ~10–50× on node rate and on the warm-cache sweep floor | §2–§5 |
| Candidate equivalence-class dedupe | up to ~10× on deep branches | §6 |

Plus one prerequisite that is not a speedup: a monster-branch calibration
solve (§7). The telemetry still contains no *finished* branch over 300 answer
words, but a first hard calibration point now exists (U3): a 208-word
budget-4 ALIBI branch ran ~90 hours / 7.6 billion nodes for ~8.7% of its
optimality certificate — ~43 days per such node extrapolated, pure Python.
That number settles the headline question by itself: the pure-Python engine
cannot finish the tree in acceptable time, and the vectorized kernel (§2–§5)
is the unconditional critical path. §7's over-300-word calibration (ALIBI's
841-word all-gray branch, already queued at priority 100,000) waits for the
§4 kernel so it measures the engine that will actually run the job. The
former prerequisite, uncapping `queue-add`, is **done** — issue #77 was
implemented by PR #80 and merged (§1 records the landed state).

**All file:line references in this document are against `main` @ `364566a`.**
PR #76 (now merged) changed `wordle_engine.py`, `erd_queue.py`,
`erd_swarm.py`, and `erd_search.py`, so line numbers have drifted — re-locate
by symbol name, not line number.

---

# Part I — Context

## C1. Vocabulary and identifier glossary

The four anchored terms (CLAUDE.md) apply everywhere: **guess** (a word
actually played), **candidate** (a word under evaluation, not yet played),
**branch** (the remaining answer words after a guess + response), and the four
qualified depth terms (**guess_depth**, **budget**, **ERD**,
**max_remaining_depth**). Never write a bare `depth`.

Additional identifiers used throughout this plan:

| Identifier | Type | Meaning |
|---|---|---|
| `n` | int | Number of answer words in the branch under evaluation (`len(branch_words)`). |
| `branch_words` | list[str] | The branch's answer words. |
| `candidate_list` | list[str] | Candidate vocabulary swept at a node; the full ~12,972 words for `ERD_ALL`. |
| `pattern` / `pattern_int` | int 0–242 | A response encoded base-3 (gray=0, yellow=1, green=2), most-significant digit first. 242 = all-green. |
| response group | — | One cell of the partition of `branch_words` by the response each word would give to a candidate. Not "subgroup". |
| `groups` | dict[int, list[str]] | `{pattern_int: response group}` for one candidate against one branch (`ResponseCache.group_words`). |
| `G` | int | `len(groups)` — number of non-empty response groups, including the self group when present. |
| `has_self` | bool/int | Whether the all-green pattern is present, i.e. the candidate itself is in `branch_words`. |
| `best_erd` | float | The running branch-and-bound bound at a node: the best exact candidate cost found so far (or the alpha-beta `ceiling` it was seeded with). |
| `cost_lb` | float | Admissible lower bound on one candidate's cost — see C2.1. |
| `rest_lb` | list[float] | Admissible lower bound on the weighted cost of the response groups after position i (`wordle_engine.py:1050`). |
| `branch_indices` | np.ndarray[int32] | New in this plan: a branch expressed as indices into the canonical answer-word list (column indices of the pattern matrix). Full words per CLAUDE.md — not `subset_idx` (both halves would be wrong: "subset" is the retired vocabulary for branch, and `idx` is an abbreviation). |
| `counts` | np.ndarray[int32] (G_vocab × 243) | New in this plan: response-group sizes of every candidate against one branch, one row per candidate. |
| `answer_list_id` | str | SHA-256 identity of the answer universe; keys every cache table. |
| `branch_key` | bytes | `ScoreCache.encode_subset(branch_words)`: sorted words concatenated, 5 bytes/word. |
| epoch | int | Telemetry era from PR #76 (`telemetry_epoch` table, `run_meta.epoch`). Epoch 0 = single-candidate-claim baseline. |
| ERD-lower-bound pruned | — | A candidate discarded because its admissible ERD lower bound (`cost_lb`, C2.1) already meets `best_erd`. Sanctioned short form: **"ERD-pruned"**. Never a bare "pruned" (the engine has other prunes: `rest_lb` partial-sum cutoffs, ceiling cutoffs) and never "gate/gated/pre-gated" (legacy vocabulary — §9b). |

## C2. The two quantities every implementer must understand

Both are exact, derived quantities — not heuristics. Getting either subtly
wrong corrupts pruning decisions or evaluation order, which corrupts cached
results.

### C2.1 `cost_lb` — the candidate cost lower bound

Defined at `wordle_engine.py:1026`:

```python
cost_lb = 3.0 - (len(groups) + (1 if has_self else 0)) / n
```

**Meaning:** the lowest ERD this candidate could possibly achieve on this
branch, granted the most optimistic continuation imaginable. Derivation:

- Playing the candidate costs 1 guess, always.
- A non-self response group of size k then costs at least `2 − 1/k` expected
  further guesses: the best conceivable next guess identifies every word in
  the group immediately, so one word costs 1 more guess and the other k−1 cost
  2 more, mean `(1 + 2(k−1))/k = 2 − 1/k`. Nothing can beat this.
- The self group (the candidate itself, if it is in `branch_words`) costs 0
  further guesses.

Summing `1 + Σ (k_i/n)(2 − 1/k_i)` over non-self groups and simplifying gives
exactly `3 − (G + has_self)/n`. Intuition: "3 minus the resolution rate" —
each additional response group moves one more answer word from the
2-more-guesses column toward the 1-more-guess column, lowering the bound by
`1/n`. More groups ⇒ lower bound ⇒ stronger candidate.

**Why it is safe to prune with:** it is *admissible* — never higher than the
candidate's true cost — so `cost_lb >= best_erd` proves the candidate cannot
beat the current best and can be discarded with zero recursion. This one-line
ERD-lower-bound pruning test is why 99.4% of measured claims cost <10 nodes.

**Naming:** `cost_lb` violates the no-abbreviations rule; §9 proposes the
rename (`candidate_cost_lower_bound`) and when it is safe to do.

### C2.2 `Σk²` — the sum of squared response-group sizes

The best-first sort key at `wordle_engine.py:1181-1186`:

```python
key=lambda c: sum(k * k for k in cache.group_counts(c, branch_words).values())
```

**Meaning:** `Σk²/n` is the **expected number of answer words remaining after
playing the candidate**. The answer lands in response group i with probability
`k_i/n`, and then exactly `k_i` words remain, so
`E[remaining] = Σ (k_i/n) · k_i = Σk²/n`. The engine sorts by `Σk²` (same
ordering, skips the division). Ascending order = strongest expected splitter
first, which tightens `best_erd` after the fewest evaluations.

**The square is not a tuning knob.** The k² arises because group size appears
twice for structural reasons: once as the probability weight of landing in the
group (`k/n`) and once as the size of what you are then left with (`k`). It is
a probability-weighted mean, not a variance-flavored penalty. This is the same
quantity as the `WEIGHTED_AVG` scoring method (`Σk²/N`, `score_groups` at
`wordle_engine.py:459`), so the sort and the scoring method must stay
consistent — there is one definition, used twice.

**Ordering is correctness-relevant even though it is "just" ordering:** when
two candidates have exactly equal optimal ERD, the one evaluated *first* wins
`best_guess` (strict `<` at `wordle_engine.py:1224`). Any change that reorders
equal-`Σk²` candidates can change which tied word is cached as `best_guess` —
not wrong, but no longer byte-identical, which breaks the equivalence tests
this plan relies on. Hence the stable-sort requirements in §4/§5.

## C3. Coordination with the measurement run (PR #76) — resolved; surviving rules

**Final state (Jul 4):** PR #76 is merged to `main`, the epoch-0 baseline
run is complete, and the swarm is stopped. The final corpus
(`analyze_swarm_telemetry.py`, epoch 0) is ~10.7M `candidate_accuracy` rows,
33.8% of them ERD-lower-bound pruned (the telemetry's legacy `gated` flag —
§9b) — much larger and differently mixed than the mid-run snapshot (6.8M
rows, 4.7% ERD-pruned), and the verdict held across both, so it is robust to
corpus composition:

- **ERD-lower-bound pruning is exact — PASS.** 100% of pruned rows ≤ 1
  node; the §4 false-expensive trap is 0% once the budget-keyed cost model
  is warm. This is the half §4b of this plan vectorizes.
- **The magnitude metric is dead — FAIL.** 81.1% false-expensive rate;
  log-log slope 0.850 but Pearson-log correlation 0.191 — predicted
  magnitude carries almost no information about actual node cost.
- **Packer design decided:** exact ERD-lower-bound pruning +
  count-bundling + republish-on-overrun, with **no work-magnitude model**.
  Measured claim-count reduction from count-bundling alone: 44× / 82× /
  144× / 231× at small-bundle caps of 8 / 16 / 32 / 64.

Consequence unchanged: the **vectorized kernel (§2–§5) is the critical
path**. The packing track proceeds per `adaptive_claim_packing.md`, which
still needs its revision to the binary scheme.

Rules 1–3 below are satisfied and kept only because later sections cite
them by number; rules 4–5 remain binding on every deploy.

1. *Satisfied.* Nothing was to deploy before the run ended; the run has
   ended and the swarm is stopped, so deploys are unblocked
   (stop-workers-before-deploy still applies, SWARM.md).
2. *Satisfied.* Engine/swarm/queue sections (§4, §5, §6) were to wait for
   the PR #76 merge; it is merged, and this integration branch already
   contains `main`, so those sections are implementable now.
3. *Satisfied.* The new-file sections are done — §2 (PR #84) and §3
   (PR #85). §9a's design.md fix is still open.
4. **Any section that changes node timing (§4, §5, §6) must bump the
   telemetry epoch on deploy** using PR #76's own machinery
   (`ERDQueue.set_epoch`, a new `telemetry_epoch` row with a descriptive
   label and git SHA), so cost-model fits never mix node rates from
   different kernels. One epoch bump per deploy, not per section.
5. **The `metric_observer` contract must be preserved.** PR #76 feeds
   `candidate_accuracy` from a hook in `evaluate_candidate`. §4's
   ERD-lower-bound pruning and §6's dedupe both *skip* `evaluate_candidate`
   calls; each section below
   states what the observer must still see. Read
   `test_claim_packing_measurement.py` before touching the candidate loop.

**Remaining schedule:**

| When | What |
|---|---|
| Done | §2 (PR #84) and §3 (PR #85) merged into this branch; epoch-0 run complete; packer go/no-go decided (U6). |
| Next | §4 implemented and reviewed on this branch; deploy = one worker restart, one epoch bump (the box pulls PR #80's uncapped `queue-add` in the same deploy). |
| After §4 deploys | §7a census, then §7b calibration (the queued 841-word ALIBI branch), then §7c schedule memo. |
| After §7 numbers | Decide §5/§6 scope; §8 lands before the first full-tree phone sync. |

## C4. The phone usage model — three modes the plan must serve

The cache is exported from Linux and imported into Pythonista on the iPhone
(SWARM.md, `erd_search.py export`). Three distinct phone workloads:

**Mode A — on-tree descent (the daily solve).** Play the cached best guess,
enter Wordle's feedback, look up the resulting branch, repeat. Requires: the
branch row for every position on any best-ERD path from any opener the owner
actually starts with. §8's reachable-only export covers this by construction.

**Mode B — off-tree recovery (mistyped feedback).** The owner occasionally
mistypes a response into the tool, plays the (now wrong) suggested word in
real Wordle, and is stranded on a branch no best-ERD path reaches. Wordle has
no undo. The tool must still descend well from there. **This already works
and is not an export problem:** `ERDSolver` (`wordle.py:2749`) live-solves
the *current* branch with `min_expected_guesses`, consulting the cache for
every sub-branch it touches — an off-tree branch's subtree overlaps heavily
with cached on-tree subtrees, so the live solve is far cheaper than cold.
Two verification items for §8's acceptance:
  - **(B-1) Measure it:** time a live `ERDSolver` run in Pythonista on
    representative off-tree branches (n ≈ 10, 40, 100) against a
    reachable-only cache. Unknown U5 in C5. If n ≈ 100 takes minutes, the
    fallback plan is a coarser first step (entropy ranking, already instant)
    while `ERDSolver` refines in the background — which is exactly how the
    interactive flow already behaves.
  - **(B-2) Budget gap — now issue #79:** `ERDSolver` calls
    `min_expected_guesses` with no `budget`, i.e. the unconstrained optimum,
    so late in a recovered game the recommended guess is not necessarily
    feasible within the guesses actually remaining (its
    `max_remaining_depth` may exceed them). Issue #79 carries the fix plan
    (pass `budget = GAME_GUESSES − guess_depth`, surface the proven-loss
    outcome in the UI); it is independent of this plan's sections and can be
    implemented on the same post-PR-#76 schedule as the other
    `wordle.py`-touching work.

**Mode C — family analysis (the ALIBI question).** The owner's sons play
independently and compare afterward; the owner wants to say *precisely* how
good or bad their choices were: root ERD of an arbitrary opener (e.g. ALIBI),
words remaining at each of their steps, and what better choices existed.
Consequences for this plan:

- **Any first guess is covered by §8 as specified**, because the reachable
  export's *seed set* is every (opener, pattern) branch for **all ~13k
  openers** — not just the owner's favorites. A son's opener + its actual
  response is a seed row: the phone has its best guess, its ERD, and the
  whole best-ERD descent under it. (Interim state: the epoch-0 run solved
  most of ALIBI's smaller branches — one 208-word branch remains mid-
  certificate (U3) — so the next export gives *partial* ALIBI coverage; its
  root ERD stays a labeled lower bound until the remaining branches are
  solved, including the queued 841-word all-gray monster, §7b.)
- **The opener's root ERD is computable from those same rows** — it does not
  need to be stored per opener, but it does need every branch of that opener
  present (including the monster branches — another reason the cap removal (PR #80) precedes the
  full run). §8b adds the small reporting command that does the arithmetic.
- **A son's second-and-later off-best guesses** leave the exported tree, the
  same as Mode B, and are served the same way: live `ERDSolver` on the
  branch, heavily cache-assisted. If U5 measurement shows this is too slow on
  the phone for comfortable dinner-table use, the escalation path is a small
  HTTP lookup service on the Linux box exposing the *full* Linux cache
  (which contains far more than the reachable export) to Pythonista. That
  service is explicitly **out of scope for this plan** — noted here so the
  option is not forgotten. Decide after U5 is measured.

## C5. Assumptions and unknowns register

Every schedule claim in this plan rests on these. Confirm or measure each;
record the answer next to it when known.

| # | Unknown / assumption | How to resolve | Blocks |
|---|---|---|---|
| U1 | **Resolved (Jul 4): Pythonista bundles NumPy 1.22.3.** That is far newer than the worst case Part II rule 3 assumed — every API on its original forbidden list exists in 1.22.3. The rule's floor is now 1.22; the conservative choices already baked into §2/§3 (and §4's `kind='mergesort'`) stand as written — no rework. | Done. | Nothing |
| U2 | Pattern-matrix memory: the 41MB matrix itself (~12,972 × ~3,185 uint8) **plus the per-call transients of `counts_for_all_candidates`**, which recur at every solved node once §4 lands and dominate the matrix unless bounded. | §2 persists a `.npy` and workers load with `mmap_mode='r'` (one page-cached copy); the chunked kernel bounds transients at ~16MB/call. Confirm total and per-call-transient RSS with 6 workers running, including a large-n solve. | §4 deploy |
| U3 | Monster-branch cost. **First hard data point (Jul 4):** a 208-word budget-4 ALIBI branch — a legitimate feasible solve, not a pathology — spent ~90h / 7.6B nodes on ~8.7% of its optimality certificate ⇒ ~43 days/node extrapolated, pure Python. That settles the headline schedule question (pure Python = years; kernel unconditional). The over-300-word regime is still unmeasured. | §7b calibration on the queued 841-word ALIBI all-gray branch, after §4 deploys; sum `branch_finalize_log` over the branch and its promoted descendants (censoring-aware). | Schedule *refinement* only — the go/no-go answer is already in |
| U4 | Warm-cache average nodes per branch (early cold sample: ~3.7M). | Trend of `branch_finalize_log.nodes_spent` as coverage grows, per epoch. | Schedule refinement |
| U5 | Phone live-solve latency on off-tree branches (Modes B/C) with a reachable-only cache. | Timed `ERDSolver` runs in Pythonista at n ≈ 10 / 40 / 100. | Whether Mode C needs the out-of-scope lookup service |
| U6 | **Resolved (final epoch-0 analysis, ~10.7M rows).** Pruning half exact — 100% of ERD-lower-bound-pruned rows (the telemetry's legacy `gated` flag, §9b) ≤ 1 node (PASS). Magnitude half dead — 81.1% false-expensive, Pearson-log 0.191 (FAIL). Packer design: exact ERD-lower-bound pruning + count-bundling + republish-on-overrun, no work-magnitude model; measured claim reduction 44–231× by bundle cap. | Done. Remaining: revise `adaptive_claim_packing.md` to this scheme (its own track). | Nothing in this plan's sections |
| U7 | Top-level branch count: estimated 1.5–2M distinct (opener, pattern) subsets with ≥ 2 words. | §7's census script computes it exactly (cheap once §2 exists). | Schedule precision; §8 export size projection |
| U8 | Exact-equality equivalence between pure-Python and vectorized paths is achievable (stable ordering everywhere). | §4/§5/§6 acceptance tests enforce it; any failure is a design bug to fix, not a tolerance to widen. | §4, §5, §6 |

---

# Part II — Ground rules (every section, every implementer)

1. **Result-identical vectorization.** The NumPy paths replace *how* a
   quantity is computed, never *which* quantity or *in what order* candidates
   or response groups are considered. For a fixed branch and budget, the solve
   must produce the **identical `best_guess`, bit-identical ERD, and identical
   `max_remaining_depth`** as the pure-Python path, and the identical set of
   cache writes. Identical ordering ⇒ identical float summation order ⇒
   bit-identical results. If a test needs a tolerance, the implementation is
   wrong (see C2.2's tie-order note and U8).
2. **The pure-Python path stays, permanently — as the reference
   implementation, not a runtime fallback.** NumPy is a hard requirement on
   every deployment target (Linux, CI, and Pythonista all have it — U1).
   The pure-Python implementations are prohibited from calling NumPy and
   exist so the equivalence tests (rule 1) always have an independent
   oracle; selecting them is a caller choice (`pattern_matrix=None`), never
   an availability check. There is no NumPy-absent configuration to support
   or test.
3. **NumPy 1.22 is the API floor** (U1 resolved: Pythonista bundles 1.22.3).
   Anything present in 1.22 is allowed; anything newer is not. In practice
   the code already targets a much older floor (`np.frombuffer`,
   `np.bincount`, `argsort(kind='mergesort')`, `np.add.at`, fancy indexing,
   `np.load(..., mmap_mode='r')`) and there is no reason to churn it. Prefer
   `kind='mergesort'` over `kind='stable'` — they select the same algorithm,
   but the explicit name states the determinism requirement (C2.2) rather
   than delegating it to an alias.
4. **No estimated value ever bounds the search** (same law as
   `adaptive_claim_packing.md` §3): `best_erd` is tightened only by exact
   solved costs or admissible lower bounds (`cost_lb`, `rest_lb`, the
   `_CEIL_EPS`-padded ceiling). Nothing in this plan introduces a new bound;
   reviewers verify each section preserves this.
5. **Deployment discipline.** Sections that change worker behaviour follow
   SWARM.md: stop supervisor → verify workers gone → deploy → start. Sections
   that change node *timing* (§4, §5, §6) additionally bump the telemetry
   epoch (C3 rule 4). Never let old workers outlive an engine change.
6. **Tests before push.** `python -m unittest discover -s . -p 'test_*.py'`
   green before every pushed commit. New behaviour ⇒ new tests in the same
   commit.
7. **Vocabulary and naming.** CLAUDE.md rules apply to every new identifier:
   full words, no abbreviations, acronyms uniform-cased, scoring methods by
   their canonical names, "response group" not "subgroup", no bare `depth`.
8. **Comment style.** Comments describe what the code *is* and its
   invariants — never the change history, never "replaces the old X".

---

# Part III — The work

Each section states: what to read first, current behaviour, the change,
step-by-step detail, edge cases, what to measure, acceptance criteria, and
what is explicitly out of scope. "Definition of done" = all acceptance items
checked, suite green, docs touched if named.

## §1. Remove the `queue-add` 300-answer branch-size cap — **DONE (issue #77, PR #80, merged)**

No implementation work remains; this section records what landed so later
sections can rely on it. PR #80 (merged to `main` at `67f0b74`):
`--max-branch-size` defaults to `None` = unlimited (`erd_search.py`, argparse
block and both cap checks), SWARM.md's `queue-add` section updated, and the
acceptance test added (`test_queue_add.py` — over-cap branch queued by
default, skipped under an explicit `--max-branch-size 300`).

**The one live remainder is operational:** the calibration monster is
already queued with the uncapped `queue-add` (§7b — ALIBI all-gray, 841
words), so PR #80's purpose here is served; the Linux box still picks up the
code in the first §4 deploy. And the out-of-scope warning
stands: do NOT mass-queue monster branches just because the cap is gone; §7
calibrates exactly one first, and mass-queueing is an operational decision
taken after U3 is known.

## §2. Pattern matrix module — new file `pattern_matrix.py` — **DONE (PR #84, merged)**

Implemented as specified: `pattern_matrix.py` + `test_pattern_matrix.py`
landed via PR #84, and CI installs NumPy. The spec below stands as the
module's reference documentation, with one later policy change (Jul 5):
the NumPy import guard and `available()` were removed — NumPy is a hard
requirement (Part II rule 2), so the acceptance items about a NumPy-absent
import no longer apply.

**Read first:** `ResponseCache` (`wordle_engine.py:352-435`) — the matrix is
its data, reshaped; `ScoreCache.read_decomposition` / `write_decomposition`
(`cache_sqlite.py`); C1 glossary; ground rules 2–3.
**Prerequisite:** none (new file).

**What it is.** One class owning a single `uint8` array:

```python
class PatternMatrix:
    """Response patterns for every (guess, answer) pair.

    matrix[g, a] is the encoded response pattern (0-242) of guess word g
    (row, canonical guess-list order) against answer word a (column,
    canonical answer-list order) — exactly the byte ResponseCache stores
    per guess, all guesses stacked.
    """
```

Shape (~12,972 × ~3,185) ≈ 41MB. Rows are byte-for-byte the
`ResponseCache` decomposition blobs.

**Construction and persistence.**
- `PatternMatrix.build(guess_words, answer_words, score_cache=None)`:
  for each guess word, obtain its decomposition blob — from
  `score_cache.read_decomposition(guess)` when available, else compute via
  `calculate_response` + `_encode_response` per answer (and write it back
  through `score_cache.write_decomposition`, so building the matrix also
  warms the SQLite decomposition table). Stack with
  `np.frombuffer(blob, dtype=np.uint8)` into the matrix.
- `PatternMatrix.save(path)` / `PatternMatrix.load(path, guess_words,
  answer_words)`: persist as `.npy`; load with `np.load(path,
  mmap_mode='r')` so N worker processes share one page-cached copy (U2).
  The filename embeds `answer_list_id` (and the guess-list length) so a
  stale file for a different universe can never be loaded; on any shape or
  identity mismatch, rebuild.
- This module is the **only** place the engine imports NumPy — a hard,
  unguarded dependency (NumPy is present on every deployment target; U1).

**Index plumbing.**
- `guess_index(word) -> int` (KeyError on unknown — callers decide fallback).
- `answer_indices(words) -> np.ndarray[int32]` — a branch as column indices
  (`branch_indices`); raises on any word not in the answer universe (see §5
  for the fallback path; the swarm's branches are always answer subsets).

**The core primitive** — response-group sizes of *every* candidate against a
branch, one call. Chunked over guess rows so the per-call transient is a
fixed ~16MB regardless of branch size — an unchunked
`matrix[:, branch_indices].astype(...)` would materialize
`n_guesses × n × itemsize` bytes (≈100–200MB per worker on a 1,900-word
monster branch or §7a's full-answer-list census), and *that*, not the 41MB
mmap-shared matrix, would be the real memory risk on a 6-worker box:

```python
_COUNT_CHUNK_ROWS = 1024   # transient = 2 × _COUNT_CHUNK_ROWS × n × 4 bytes

def counts_for_all_candidates(self, branch_indices):
    """(n_guesses, 243) int32: counts[g, p] = number of words in the branch
    whose response to guess-word g encodes to pattern p."""
    counts = np.empty((self.n_guesses, 243), dtype=np.int32)
    row_offsets = np.arange(_COUNT_CHUNK_ROWS, dtype=np.int32)[:, None] * 243
    for start in range(0, self.n_guesses, _COUNT_CHUNK_ROWS):
        stop = min(start + _COUNT_CHUNK_ROWS, self.n_guesses)
        rows = stop - start
        branch_patterns = self.matrix[start:stop, branch_indices].astype(np.int32)
        offset_patterns = branch_patterns + row_offsets[:rows]
        counts[start:stop] = np.bincount(
            offset_patterns.ravel(), minlength=rows * 243
        ).reshape(rows, 243)
    return counts
```

`int32` is safe: the largest offset key is `_COUNT_CHUNK_ROWS × 243 ≈ 249k`
and the largest count is the answer-list size (~3,185). One C-speed pass over
`n_guesses × n` elements (~1.3M for n = 100 — milliseconds) replacing ~13k
Python-level `group_counts` calls (~seconds).
Also provide `patterns_for_candidates(candidate_indices, branch_indices)`
returning the raw `(len(candidate_indices), n)` uint8 slice (§5, §6 reuse
it).

**What to measure.** Build time cold (all blobs computed) and warm (all blobs
in SQLite); `.npy` load time; `counts_for_all_candidates` wall time at
n ∈ {8, 30, 100, 300, 1000, 1900}; peak RSS with the matrix mmap-loaded in 6
processes **including per-call transient RSS at the large-n end** — the
transients, not the shared matrix, are the dominant allocation (U2).

**Acceptance.**
- New `test_pattern_matrix.py`:
  - ~200 random (guess, answer) pairs:
    `matrix[g, a] == _encode_response(calculate_response(guess, answer))`.
  - ~20 random (candidate, branch) pairs: the nonzero entries of
    `counts_for_all_candidates(branch_indices)[g]` equal
    `ResponseCache.group_counts(candidate, branch_words)` as a dict —
    including at least one branch larger than `_COUNT_CHUNK_ROWS` worth of
    work and one that exercises the final partial chunk.
  - Save/load round-trip equals the built matrix; identity mismatch rebuilds.
- No file outside `pattern_matrix.py` + its test file is modified.

## §3. Vectorized candidate statistics — extends `pattern_matrix.py` — **DONE (PR #85, merged)**

Implemented via PR #85: `candidate_stats(branch_indices)` returns a
`CandidateStats` NamedTuple. Two implementation notes beyond the original
spec: the documented precondition is `len(branch_indices) >= 1` (an empty
branch would silently produce NaN through the 0/0 divisions rather than
raise), and the entropy computation routes zero counts through a
double-`np.where` so `log2` never sees zero.

**Read first:** §2; C2 (both derivations); `score_groups`
(`wordle_engine.py:441-477`); the `cost_lb` line (`wordle_engine.py:1026`).
**Prerequisite:** §2.

**What it is.** `candidate_stats(branch_indices)`: from one
`counts_for_all_candidates` result, the whole-vocabulary versions of the
quantities the engine currently derives one candidate at a time. Returns a
small NamedTuple of parallel arrays, index = guess-word (matrix-row) space —
see §4a's index-alignment hazard before consuming them in candidate-list
order:

| Field | dtype | Formula per row `c` | Engine twin |
|---|---|---|---|
| `group_count` | int32 | `(counts[c] > 0).sum()` | `len(groups)` |
| `has_self` | bool | `counts[c, 242] > 0` (all-green ⇔ candidate ∈ branch) | `_ALL_GREEN_PATTERN in groups` |
| `cost_lower_bound` | float64 | `3.0 - (group_count + has_self) / n` | `cost_lb`, `wordle_engine.py:1026` |
| `sum_squared_group_sizes` | int64 | `(counts[c] ** 2).sum(dtype=int64)` | sort key, `wordle_engine.py:1184` |
| `max_group_size` | int32 | `counts[c].max()` | `score_groups` MAX_GROUP_SIZE |
| `entropy_gain` | float64 | `-Σ p·log2(p)` over nonzero sizes, `p = k/n` | `score_groups` ENTROPY_GAIN |

Notes for the implementer:
- `sum_squared_group_sizes` **must be integer** — it is compared for exact
  ordering equality with Python's arbitrary-precision `sum(k*k ...)`, so the
  int64 dtype is an exactness spec, not overflow protection. Response groups
  partition the branch (Σk = n ≤ ~3,185), so Σk² ≤ n² ≈ 10.1M — comfortably
  inside int32 (an earlier revision claimed a worst case of 3,200² × 243,
  which wrongly assumed 243 groups could each be full-size). Squares are
  computed in int32 and accumulated with `.sum(axis=1, dtype=np.int64)`,
  which avoids materializing an int64 upcast of the whole counts array
  (~25MB at full vocabulary — the U2 transient concern).
- `cost_lower_bound` must be computed as `3.0 - (g + s) / n` with float64
  division — the same expression shape as the engine line — so the float is
  bit-identical to the scalar computation.
- Entropy: mask zeros before `log2`; accumulate in float64. This one is
  allowed `<1e-12` test tolerance (the engine's own accumulation order over a
  dict is not canonical), because nothing orders or prunes on entropy at
  solve time — it is a display/ranking aid (`rank_candidates_by_...` tie
  break). Everything else: exact.

**Acceptance.** In `test_pattern_matrix.py`: for fixed branches of sizes
{8, 30, 100, 500} (at least one containing candidate words and one not),
every field matches the per-candidate Python computation — integers exactly,
`cost_lower_bound` exactly, entropy within 1e-12.

## §4. Engine integration: vectorized ranking and ERD-lower-bound pruning

**Read first:** `_solve_subset` in full (`wordle_engine.py:1097-1259`);
`evaluate_candidate` (`wordle_engine.py:971-1094`); C2.2's tie-order warning;
C3 rules 2, 4, 5 (merge-first, epoch bump, deploy discipline); the
`metric_observer` hook and its tests on merged `main`
(`test_claim_packing_measurement.py`).
**Prerequisites:** §3, PR #76 merged — **both satisfied; this is the next
section to implement.**

This is the highest-value section: it removes the two per-candidate Python
sweeps that run at **every** solved node with `n >= ORDER_MIN_N` (= 8), even
nodes where everything beneath is a cache hit.

**4a. Vectorized best-first sort.** Current code
(`wordle_engine.py:1181-1186`) sorts `candidate_list` by the Python `Σk²`
scan. Replacement when a matrix is available:

```python
if pattern_matrix is not None and cache and n >= ORDER_MIN_N and len(candidate_list) > 1:
    branch_indices = pattern_matrix.answer_indices(branch_words)
    stats = pattern_matrix.candidate_stats(branch_indices)        # §3
    candidate_rows = np.array([pattern_matrix.guess_index(c)
                               for c in candidate_list], dtype=np.int32)
    sort_keys = stats.sum_squared_group_sizes[candidate_rows]     # aligned to candidate_list
    order = np.argsort(sort_keys, kind='mergesort')               # stable
    candidate_list = [candidate_list[i] for i in order]
    candidate_cost_lower_bounds = stats.cost_lower_bound[candidate_rows][order]
```

Correctness argument the implementer must preserve: Python's `sorted` is
stable, so equal-`Σk²` candidates keep their `candidate_list` relative order;
`kind='mergesort'` is the stable argsort that reproduces exactly that. The
keys are integers on both paths, so there is no float-comparison ambiguity.
Result: byte-identical candidate order, hence identical winner on ties
(C2.2). Cache `stats`/`branch_indices` on the frame — 4b and §6 reuse them.

**Index-alignment hazard (silent-wrong-answer severity).** Three index
spaces exist here: matrix guess rows, pre-sort `candidate_list` positions,
and post-sort positions. Every §3 vector is in *matrix row* space; the loop
in 4b runs in *post-sort* space. The bound array 4b reads must therefore be
permuted by **both** `candidate_rows` and `order` — exactly the last line of
the snippet. Reading an unpermuted array with the loop index checks each
candidate against *some other candidate's* bound: a non-admissible prune
that can silently discard the true winner and cache a wrong ERD. The
acceptance tests below include a fixture built to catch this.

If any word in `candidate_list` is missing from the guess vocabulary
(possible only in exotic interactive states), fall back to the pure-Python
sort for that node rather than special-casing.

**4b. ERD-lower-bound pruning in the candidate loop.** Inside the loop at
`wordle_engine.py:1198`, guard the unconditional `evaluate_candidate` call
with the ERD-lower-bound pruning test, **falling through** to the rest of the
loop body — no
early `continue`, because everything after the call (abort dispatch, taint
fold, the mid-loop publisher check, the status dispatch) must still run for
an ERD-pruned candidate exactly as it runs for one `evaluate_candidate`
rejected itself:

```python
if (candidate_cost_lower_bounds is not None
        and candidate_cost_lower_bounds[i] >= best_erd):
    # ERD-lower-bound pruned: the same admissible bound evaluate_candidate
    # would compute (C2.1) already proves this candidate cannot beat best_erd.
    status, cost, max_remaining_depth, budget_tainted = (
        OVER_ERD_LIMIT, None, None, False)
else:
    status, cost, max_remaining_depth, budget_tainted = evaluate_candidate(
        branch_words, candidate, cache, score_cache, ...)
# ... unchanged from here: abort dispatch, node_floor fold, mid-loop
# publisher check, then `if status == OVER_ERD_LIMIT: cutoff_occurred = True;
# continue` — the ERD-pruned candidate's tuple flows through the same dispatch.
```

This is decision-identical to the ERD-lower-bound pruning test `evaluate_candidate` itself
applies at `wordle_engine.py:1027` — same admissible bound (C2.1), same `>=`
comparison, against the same running `best_erd`, and the same result tuple
`(OVER_ERD_LIMIT, None, None, False)` — but costs an array read instead of a
full `group_words` partition (the dominant cost of an ERD-pruned candidate).

Three invariants to preserve, each already load-bearing today:
1. **The mid-loop publisher check runs every iteration** including ERD-pruned
   ones (today's loop runs it before the status `continue`s —
   `wordle_engine.py:1211-1217`). The 4b snippet's fall-through structure
   exists precisely for this: evaluate-or-skip, then publisher check, then
   dispatch on status. Never write the ERD-pruning check as an early `continue`.
2. **`cutoff_occurred` semantics:** an ERD-pruned candidate is an
   `OVER_ERD_LIMIT`-equivalent, so it must set `cutoff_occurred = True`
   (otherwise a node where *every* candidate is ERD-pruned would fall through
   to the "proven unsolvable" branch at `wordle_engine.py:1232-1243` and
   **write a false loss row** — the single worst bug this section could
   introduce).
3. **`metric_observer`:** on merged `main`, decide per its actual contract:
   either the observer receives the same observation it would have received
   from `evaluate_candidate` (~0 nodes, its legacy `gated` flag set — §9b),
   or the ERD-pruning check is disabled when an observer is attached (observers
   ride swarm claims, where the claim loop — not this inline loop —
   dominates; disabling there costs little). State the choice in the PR
   description.

**4c. Plumbing.** Thread one shared `PatternMatrix` exactly the way
`ResponseCache` is shared:
- New optional parameter `pattern_matrix=None` on `min_expected_guesses`,
  `_solve_subset`, `evaluate_candidate` (for its §5 future use), defaulted so
  every existing caller is unchanged.
- `erd_swarm.py`: `_BranchWorker` loads (mmap) or builds the matrix once per
  process, alongside its `ResponseCache`, and passes it down.
- `wordle.py`: build lazily at session start; interactive
  commands that rank the full vocabulary (`s`, `b`) may use §3 stats in a
  follow-up commit — optional, not part of this section's acceptance.

**4d. Epoch bump on deploy** (C3 rule 4): new `telemetry_epoch` row, label
`numpy-kernel`, current git SHA; deploy with the stop→deploy→start sequence.

**What to measure (drives the §5/§6 decision).** New diagnostic
`diag_kernel_bench.py` (same family as the other `diag_*.py`, not part of the
unittest suite): for branches of sizes {8, 30, 81, 146, 500}, solve
matrix-on vs matrix-off; assert identical results; report wall time, node
count, and the per-node share still spent in `group_words` (that share is
§5's entire justification).

**Acceptance.**
- The equivalence test (the core deliverable), in a new
  `test_kernel_equivalence.py`: for fixed branches of sizes ~{8, 30, 81, 146}
  and budgets {None, 5, 4}, `_solve_subset` with and without the matrix
  produces identical `best_guess`, bit-identical ERD, identical
  `max_remaining_depth`, identical taint, and identical rows written to a
  fresh in-memory ScoreCache (compare full table dumps).
- A regression test for invariant 2: a branch/budget where every candidate is
  ERD-pruned (tight seeded ceiling) still returns `OVER_ERD_LIMIT`-style
  cutoff, and writes **no** loss row.
- A regression test for the index-alignment hazard (4a): a fixture branch
  where the best-first order differs substantially from vocabulary order and
  the winning candidate sits late in vocabulary order but early in sorted
  order — chosen so that ERD-pruning any candidate against a misaligned bound
  changes the recorded winner or ERD, making the equivalence assertion catch
  an unpermuted or half-permuted bound array.
- Full suite green.
- PR description states the `metric_observer` choice (invariant 3).

## §5. Vectorized group partitioning (`group_words` fast path) — only if §4's measurement justifies it

**Read first:** §4's `diag_kernel_bench.py` output (the `group_words` share);
`ResponseCache.group_words` (`wordle_engine.py:418-430`); the group-ordering
subtleties below.
**Prerequisites:** §2, §4 deployed, and a measured `group_words` share that
still dominates. If §4 leaves `group_words` under ~30% of node time, skip
this section — the complexity is not free.

**Measurement verdict (2026-07-06): GO.** The §4 deploy bench
(`diag_kernel_bench.py`, 900s deadline, epoch-1 box) puts the matrix-on
`group_words` share at 90.3–99.7% for branch sizes 30/81/146 — decomposition
of surviving candidates is the node cost. Corroborated in production: the
208-word calibration branch advances only ~1.4× faster under §4 because
almost none of its candidates are ERD-prunable at n=208 (pruning needs ≤ 21
response groups), so each takes a full multi-million-node evaluation that is
pure-Python decomposition throughout.

**Current behaviour.** Each `evaluate_candidate` call that survives
ERD-lower-bound pruning builds `{pattern: [words]}` with a Python loop over
the branch (`wordle_engine.py:1013`). After §4 this is the main surviving
per-node Python cost.

**Change.** Inside `ResponseCache.group_words`, accept an optional
`(pattern_matrix, branch_indices)` fast path that produces an **identical dict**
— same keys, same value lists, same *iteration order* — via one matrix row
read plus a stable argsort, instead of the per-word loop.

Why iteration order is load-bearing: `evaluate_candidate` sorts groups by
size **descending with a stable sort** (`wordle_engine.py:1038`), so
equal-size groups are processed in dict insertion order; group processing
order determines float accumulation order of `cost` and, under a ceiling,
*which* sub-branch triggers a cutoff first. Today's insertion order is
**first-appearance order of each pattern while walking the branch words**.
The fast path must reproduce exactly that:

1. `branch_patterns = matrix[g, branch_indices]` (uint8, length n).
2. `order = np.argsort(branch_patterns, kind='mergesort')` — sorts by
   pattern value, preserving branch order within a pattern (so each group's
   word list is in branch order, matching today).
3. Walk `order` once, cutting at pattern-value boundaries → groups keyed by
   pattern, each with its word list.
4. Emit the dict **in first-appearance order**: compute each present
   pattern's first index in `branch_patterns` (e.g. via the boundary walk
   plus a second pass ordering group keys by `first_index[pattern]`), and
   insert in that order. Do not emit in ascending-pattern order; that
   reorders equal-size ties.
5. Words outside the answer universe (the interactive fallback mode —
   `wordle_engine.py:424-429`) cannot be in `branch_indices`; when the
   caller cannot form `branch_indices` (any branch word unknown), it must
   pass none and take the existing loop. The swarm path always can.

**Acceptance.**
- Property test: for ~50 random (guess, branch) pairs, fast and slow
  `group_words` return dicts equal in keys, values, and **iteration order**
  (compare `list(d.items())`).
- `test_kernel_equivalence.py` re-run with §5 enabled: still byte-identical.
- `diag_kernel_bench.py` before/after numbers in the PR description.

## §6. Candidate equivalence-class dedupe

**Read first:** C2, §4 (reuses its cached frame data); the winner-tie
argument below — it is the whole correctness case.
**Prerequisites:** §2; §4 recommended first (shares the per-frame
`patterns_for_candidates` slice). Independent of §5.

**Fact this exploits.** A candidate's evaluation depends only on (a) the
partition it induces on `branch_words` and (b) whether it is itself in the
branch — and both are fully encoded in its pattern row restricted to the
branch (`matrix[g, branch_indices]`; membership shows as pattern 242). Two
candidates with identical rows are indistinguishable to the recurrence: same
groups, same costs, same `max_remaining_depth`, same taint. On deep branches
over few distinct letters, most of the ~13k candidates collapse into few
classes (extreme case: every word sharing no letters with the branch induces
the single all-gray group).

**Change.** In `_solve_subset`, after the §4a ordering, when a matrix is
available and `guesses is not None` (full-vocabulary sweeps only): build
`{row_bytes: representative}` keeping the **first** class member in the
already-sorted order (a plain dict walk — `bytes(row)` as key), and run the
candidate loop over representatives only.

**Why the recorded winner is unchanged (do not skip this reasoning):** class
members have equal `Σk²` (the key is a function of the partition), so under
the stable sort they are adjacent and the representative is the earliest in
evaluation order. Today, the first member either becomes `best_guess` or
fails; later members compute the identical cost and can never pass the strict
`cost < best_erd` (`wordle_engine.py:1224`), so they can never become
`best_guess` anyway. Skipping them changes no recorded value — winner, ERD,
`max_remaining_depth`, taint, and cache writes are all identical, not merely
equivalent.

**Boundaries.**
- **Inline recursion only.** Swarm top-level claims hand out candidates by
  index (`candidate_claims`), and finalize requires all `n_candidates` slots
  done — claim-level dedupe belongs to the packing design (a bundle carrying
  a whole class, marked done together). Add it to
  `adaptive_claim_packing.md` §12 as an open extension; do not build it here.
- **`metric_observer`:** skipped members produce no observation. Confirm the
  observer is not attached on inline recursion frames (expected — it rides
  claims); if it can be, disable dedupe on observed frames, same policy as
  §4b invariant 3.
- The signature pass costs one `(G_vocab × n)` slice already materialized by
  §4a — reuse it; do not re-slice.

**What to measure.** Class-count vs candidate-count distribution by branch
size (log in `diag_kernel_bench.py`): expect collapse to grow as branches
shrink; this number decides how much §6 matters below the promotion threshold.

**Acceptance.**
- Unit test: a small branch over few distinct letters yields far fewer
  classes than candidates (sanity), and the solve with dedupe on equals the
  solve with it off — byte-identical across all recorded values and cache
  writes.
- `test_kernel_equivalence.py` fixtures re-run with dedupe on: identical.
- Full suite green.

## §7. Monster-branch calibration and top-level census — operational

**Read first:** §1 (what PR #80 landed); PR #76's `branch_finalize_log` and
censoring semantics; C5 U3/U4/U7.
**Prerequisites:** §4 deployed (calibrating the slow kernel mismeasures the
plan); PR #76's instrumentation live (satisfied — merged). The calibration
branch itself is already queued (7b).

**7a. Census (30-minute script, do first).** New `diag_toplevel_census.py`:
using the §2 matrix, for every opener compute its partition of the answer
list; count distinct `branch_key`s with ≥ 2 words across all openers, the
size histogram, and specifically the count above 300 words (the never-queued
regime). Resolves U7 exactly and sizes §8's export. Read-only; runs anywhere.

**7b. Calibration solve.** The calibration branch is **already queued**:
ALIBI's all-gray branch (841 answer words, priority 100,000, pending in
`erd_queue.sqlite3`). ALIBI is the right choice: the epoch-0 run solved most
of its smaller branches (so its subtree cache is warm and the marginal cost
measured is the monster's own), and its all-gray branch is simultaneously
the missing piece of the Mode C root-ERD answer (C4). Do **not** start it
before §4 deploys — calibrating the pure-Python kernel mismeasures the plan,
and the U3 data point already shows that regime is ~43 days/node. Once the
swarm restarts on the new kernel, let it drain the branch; sub-branch
promotion fans it out. Then aggregate cost from
`branch_finalize_log` over the branch and its promoted descendants (join by
spine), **treating censored rows as lower bounds** (PR #76's §9.7 semantics),
plus wall-clock span.

**7c. The schedule memo.** Extrapolate: census counts × measured per-regime
costs, adjusted by the U4 warm-cache trend ⇒ the revised whole-job estimate.
Deliverable: a dated section appended to this file with the numbers and the
go/no-go call on "weeks", plus the §5/§6 scope decision. No code.

## §8. Phone export: reachable-only filter, plus the opener report

**Read first:** `export_cache.py` (standalone since PR #88); `verify_erd_cache`'s
best-guess walk (`wordle_engine.py:1329-1360`) — the BFS to imitate; C4 (all
three modes); §7a census output for size projection.
**Prerequisites:** PR #76 merged — satisfied. PR #88 merged — satisfied (see
note below). §2 useful but optional (decomposition blobs suffice for seeds).

**PR #88 landed part of this section's groundwork independently.** Export and
import are no longer `erd_search.py` subcommands: `export_cache.py` produces
the phone snapshot (four tables — `answer_list`, `response_decomposition`,
`branch_best_by_policy`, and the whole of `candidate_scores`, which the phone
needs for ranking at uncached positions), and `import_cache.py` (renamed from
`merge_cache.py`) ingests it. Export is safe against a live swarm (WAL
snapshot) and incremental (INSERT OR IGNORE). What remains of §8 is exactly
8a's reachable-only filter and 8b's opener report.

**Problem.** `branch_best_by_policy` holds 3M+ rows already
(`cache_sqlite.py:220` comment) and a full run adds tens of millions; most
rows are search memoization (sub-branches of *losing* candidates) that Mode
A/C lookups never touch. Syncing multi-GB SQLite over iCloud to Pythonista is
the bottleneck the phone does not need.

**8a. `export_cache.py --reachable-only`.**
- **Seed set:** every (opener, pattern) branch with ≥ 2 answer words, for
  **every opener in `wordle.txt`** — this is what makes Mode C's "any first
  guess" work (C4). Enumerate via decomposition blobs (or the §2 matrix);
  the seed *keys* are exactly the census (§7a) population.
- **BFS:** for each seed present in the cache: read its `best_guess`,
  partition its words by that guess (`ResponseCache.group_words`), enqueue
  every ≥ 2-word response group, and export every visited row. A node absent
  from the cache terminates its path silently — export is best-effort on a
  partial cache and re-runnable as coverage grows.
- Export the same four tables as today, row-filtered on
  `branch_best_by_policy` only; `candidate_scores` stays whole-table (PR
  #88's documented phone need is not position-bound). `--reachable-only` is
  a new flag; default behaviour unchanged. Existing incremental semantics
  (INSERT OR IGNORE) unchanged.
- Also export `branch_loss_by_policy` rows for visited keys if the phone
  code reads losses (verify; if it never does, say so in the PR and skip).

**8b. Opener report (`opener-erd`).** The Mode C arithmetic, small and
read-only. Given an opener w and the cached rows:

```
root_erd(w) = 1
            + Σ over response groups g with |g| >= 2:   (|g|/n) · ERD(g)
            + (count of non-self singleton groups) / n
```

where n = answer-list size, ERD(g) comes from the branch row, and the self
group (w itself, when w is a possible answer) contributes 0 beyond the
initial 1. If any ≥ 2-word branch of w is uncached, report the partial sum
**as a lower bound, clearly labeled** — never as the ERD (the same honesty
rule as everywhere else: a bound is honest, a guess is not).
- Linux CLI: a standalone `opener_erd.py --word alibi` (following PR #88's
  one-script-per-tool pattern; do not grow `erd_search.py`) printing root
  ERD, the per-pattern table (pattern, group size, cached best guess, ERD),
  and coverage (branches cached / total).
- Phone: same computation exposed in the interactive tool (natural home: the
  `t` analyse-word command, which already aggregates per-word views), so
  "how awful was ALIBI, exactly" is answerable at the dinner table:
  `root_erd(ALIBI) − root_erd(best opener)` in expected guesses.

**Mode B verification rides this section** (C4): B-1 timed phone
measurements (U5) against a reachable-only export, and the B-2
`ERDSolver`-budget observation, both reported in the PR description —
findings, not code changes.

**Acceptance.**
- Unit test on a small synthetic cache: reachable rows exported; a planted
  sub-branch row of a *non-best* candidate is excluded; uncached nodes
  terminate cleanly; re-export is incremental.
- `opener-erd` unit test on a synthetic cache with known arithmetic,
  including the partial-coverage lower-bound path and the self-group case.
- Size report in the PR: full vs reachable row counts and file bytes on the
  real cache.
- SWARM.md export section updated.

## §9. Naming and documentation sync

**Read first:** CLAUDE.md naming rules; C2.1; C3 rule 2.

**9a. Doc fixes (can do now).**
- design.md's Budget subsection of "Parallel ERD Precache (Swarm)"
  (`design.md:362-366`): "Workers solve branches under `ROOT_BUDGET = 5`"
  is stale — the code has `ROOT_BUDGET = GAME_GUESSES` with each queued
  branch solved at `ROOT_BUDGET − guess_depth` (`erd_swarm.py:98-101`).
  (SWARM.md contains no budget text; the stale line lives in design.md.)
- After §2/§4 land: add `pattern_matrix.py` to the layer tables in CLAUDE.md
  and design.md, plus a short design.md section: what the matrix is, which
  engine paths consult it, and the fallback rule (Part II rule 2).

**9b. Rename proposal (owner decision; the epoch-0 run and its analysis
are complete, so this is unblocked once the owner considers the epoch-0
corpus archived).** `cost_lb` and `rest_lb` violate the no-abbreviations
rule and under-describe themselves (C2.1), and the `gated` family names the
mechanism's shape (a gate) while omitting the criterion (the ERD lower
bound) — the same flaw as `MINIMAX`. Prose already says "ERD-lower-bound
pruned" throughout this document; the identifiers should follow. Proposed:
- `cost_lb` → `candidate_cost_lower_bound`
- `rest_lb` → `remaining_groups_cost_lower_bound`
- `gated` (the `candidate_accuracy` flag, the `metric_observer` parameter,
  `ACCURACY_GATED_SAMPLE_EVERY`, and the analyzer's output labels) → the
  `erd_lower_bound_pruned` family.
- Scope: Python identifiers in `wordle_engine.py`, `erd_swarm.py`,
  `analyze_swarm_telemetry.py`, and docs. The `candidate_accuracy.cost_lb`
  and `.gated` SQLite columns (queue DB, Linux-only) can be renamed by an
  idempotent `ERDQueue._migrate()` step — but **not while
  `analyze_swarm_telemetry.py` still needs to read the epoch-0 corpus**, so
  schedule after the epoch-0 analysis is finished and archived. If the
  owner prefers, the columns may simply keep their names with the mapping
  documented where they are read.

---

# Sequencing

```
already done      §1 (issue #77 / PR #80) · §2 (PR #84) · §3 (PR #85) ·
                  §4 (PR #86) — all merged to main (PR #78)
                  §4 DEPLOYED 2026-07-05: epoch 1 "numpy-kernel"
                  (git e1ab50f), 6 workers restarted on it
                  §7a census run 2026-07-06 (U7 resolved: 569,132 distinct
                  branches; 20,219 over 300 words; largest 1,955)
                  §9a doc fixes; epoch-0 run complete; packer go/no-go
                  decided (U6); export/import split (PR #88, part of §8's
                  groundwork)
in progress       §7b calibration — the 841-word ALIBI branch is queued but
                  waits behind the 208-word branch's certificate, which is
                  decomposition-bound (~9.7% done as of 2026-07-06); at §4
                  speeds that is weeks away, so §7b/§7c effectively wait
                  on §5
next              §5 — measurement verdict GO (see §5); THE critical path
then              §7b drain → §7c schedule memo → §6 per its numbers
then              §8 (8a reachable-only + 8b opener report) before first
                  full-tree sync
anytime           §9b rename (unblocked once the epoch-0 corpus is archived)
parallel track    claim packing per adaptive_claim_packing.md — binary scheme
                  (exact ERD-lower-bound pruning + count-bundling +
                  republish-on-overrun); that document still needs revision
                  to the scheme
```

Dependency summary: §5 unblocks §7b in practice; §2→§6 (§2 done); §8
independent (base landed via PR #88); §9b last. The packing lever multiplies
with all of it and is managed by its own document.
