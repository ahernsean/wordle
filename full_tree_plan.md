# Full best-ERD tree for all ~13k openers — implementation plan

The goal: populate `branch_best_by_policy` (`ERD_ALL`) with the best-ERD result
for every branch of every ~13k opener, on one small Linux box, in weeks rather
than years.

The measured baseline (see `adaptive_claim_packing.md` §1) delivers ~900 branch
sweeps/day; the full job is ~1.5–2M top-level branches. Three multiplicative
levers close the gap:

| Lever | Expected gain | Status |
|---|---|---|
| Claim packing (coordination overhead) | ~5–6× | Planned in `adaptive_claim_packing.md`; measurement layer on `claude/issue-67-review-8v65i1` |
| Vectorized partition kernel (NumPy) | ~10–50× on node rate and on the warm-cache sweep floor | This plan, §2–§5 |
| Candidate equivalence-class dedupe | up to ~10× on deep branches | This plan, §6 |

Plus two prerequisites that are not speedups but are mandatory for the goal:
removing the `queue-add` branch-size cap (§1 — a full tree needs every branch),
and a monster-branch calibration solve (§7 — the current telemetry contains no
branch over 300 answer words, so no schedule estimate is trustworthy without one).

Each section below is scoped to be independently implementable: it names the
files touched, the exact behaviour change, and the acceptance tests. Sections
marked **[independent]** can be done in any order; others list their
prerequisites.

---

## §0. Ground rules (read before implementing any section)

These are absolute constraints, in the spirit of `adaptive_claim_packing.md` §3:

1. **Vectorization must be result-identical, not just value-close.** The NumPy
   paths in §2–§5 replace *how* a quantity is computed, never *which* quantity
   or *in what order* candidates are evaluated. For a fixed branch and budget,
   the solve must produce the **identical winning guess, identical ERD (bit-for-
   bit), and identical `max_remaining_depth`** as the pure-Python path. This is
   achievable because the plan requires identical candidate ordering (stable
   sorts on the same integer keys) and identical response-group ordering — so
   every floating-point sum happens in the same order.
2. **The pure-Python path stays.** Every vectorized function has the existing
   implementation as its fallback, selected by NumPy availability at import.
   Pythonista bundles NumPy, but the fallback guarantees the phone (and the
   tests) never *require* it.
3. **Old-NumPy-safe APIs only.** Pythonista's bundled NumPy can lag years behind.
   Allowed: `frombuffer`, `bincount`, `argsort(kind='mergesort')` (stable),
   `add.at`, basic fancy indexing, `nonzero`, `unique`. Forbidden:
   `kind='stable'` (needs ≥1.15), `unique(axis=...)` semantics beyond 1D unless
   guarded, anything newer.
4. **No estimated value ever bounds the search.** Unchanged from the packing
   plan: `best_erd` is tightened only by exact solved costs or admissible lower
   bounds. Nothing in this plan touches that; a reviewer should verify each
   section preserves it.
5. **Deployment discipline.** Any section that changes worker behaviour follows
   SWARM.md's stop → deploy → start sequence, and any section that changes node
   *timing* (§4, §5, §6) must bump the telemetry epoch (`telemetry_epoch`,
   `run_meta.epoch` — the §9.1 machinery from the packing plan) so cost-model
   fits never mix node rates from different kernels.
6. **Tests before push.** `python -m unittest discover -s . -p 'test_*.py'`
   passes before every commit that gets pushed.

---

## §1. Remove the `queue-add` 300-answer branch-size cap **[independent]**

Tracked as a GitHub issue; this section is its implementation.

**Problem.** `erd_search.py` `cmd_queue_add` skips any branch with more than
`--max-branch-size` answer words, default 300 (`erd_search.py:1629`, skip logic
at `erd_search.py:291` and `:303`). An opener's root ERD is
`1 + Σ (k/n)·ERD(branch)` over **all** of its response branches, so skipping
the large ones (every weak opener's all-gray branch is 300–1,900 words) makes
every root ERD uncomputable. The cap also means the swarm's telemetry contains
zero data about large branches.

**Change.**
- Default `--max-branch-size` to unlimited (`None`); when `None`, queue every
  branch with ≥ 2 answer words. Keep the flag so an explicit cap can still be
  passed for deliberately bounded runs.
- Update the `--max-branch-size` help text, the `cmd_queue_add` docstring, and
  the SWARM.md `queue-add` examples (drop "skips branches with >300 answer
  words"; document the flag as an optional bound).

**Notes.** No engine change is needed: large branches already work — they were
only filtered at queue time, and sub-branch promotion (`erd_swarm.py`,
`PROMOTE_MIN_SIZE` / the adaptive publish threshold) is the mechanism that keeps
one monster from monopolizing a worker. Do **not** mass-queue monsters as part
of this section; §7 calibrates one first.

**Acceptance.**
- `queue-add --word <weak-opener>` queues its all-gray branch (unit test:
  branch with > 300 words appears in `pending_branches`).
- `queue-add --word X --max-branch-size 300` reproduces the old behaviour.
- Existing `test_cli_*` / queue tests pass.

---

## §2. Pattern matrix module (`pattern_matrix.py`) **[independent; foundation for §3–§6]**

A new module owning one object:

```python
class PatternMatrix:
    """uint8 matrix of encoded response patterns: rows = guess words in
    canonical guess-list order, columns = answer words in canonical
    answer-list order.  matrix[g, a] == encoded pattern (0-242) of guess g
    against answer a — exactly the byte ResponseCache stores per guess."""
```

**Construction.** Rows are exactly the `ResponseCache` decomposition blobs
(one byte per answer, canonical order — `wordle_engine.py:352-435`). Build by
stacking `np.frombuffer(blob, dtype=np.uint8)` for each guess word, loading
blobs through the existing `ScoreCache.read_decomposition` path and computing
(and persisting) any that are missing, same as `ResponseCache._ensure`. Size:
12,972 × ~3,185 ≈ 41MB — fine on Linux and on a modern iPhone.

API (all pure NumPy, no engine imports beyond the encoders):

- `PatternMatrix.build(guess_words, answer_words, score_cache=None)`
- `guess_index(word) -> int`, `answer_indices(words) -> np.ndarray[int32]`
- `row(guess_word) -> np.ndarray[uint8]` (length = n_answers)
- `counts_for_all_candidates(subset_idx) -> np.ndarray[int32]` of shape
  `(n_guesses, 243)`: response-group sizes of **every** candidate against the
  branch identified by `subset_idx`, in one shot. Implementation is the
  offset-bincount trick:

  ```python
  sub = self.matrix[:, subset_idx]                     # (G, n) uint8
  flat = sub.astype(np.int64) + np.arange(G)[:, None] * 243
  counts = np.bincount(flat.ravel(), minlength=G * 243).reshape(G, 243)
  ```

  For n = 100 that is ~1.3M elements through one C loop — milliseconds, versus
  ~13k Python-level `group_counts` calls.

- `patterns_for_candidates(candidate_idx, subset_idx) -> np.ndarray[uint8]` of
  shape `(len(candidate_idx), n)` — raw pattern rows restricted to a branch
  (used by §5 and §6).

Optionally cache the built matrix as an `.npy` file next to the SQLite cache,
keyed by `answer_list_id` in the filename, to skip the blob-stacking on every
process start (worker recycling every 3h makes startup cost matter). Loading
41MB from disk is near-instant; rebuilding from SQLite blobs is seconds.

**Acceptance.**
- New `test_pattern_matrix.py`: for ~200 random (guess, answer) pairs,
  `matrix[g, a] == _encode_response(calculate_response(guess, answer))`.
- For ~20 random (candidate, subset) pairs, `counts_for_all_candidates`
  row equals `ResponseCache.group_counts` as a dict (zeros dropped).
- Module imports and degrades cleanly when NumPy is absent
  (`PatternMatrix.available()` → False; nothing else imports NumPy at top level
  of the engine).

---

## §3. Vectorized candidate statistics (in `pattern_matrix.py`) — requires §2

Derived, whole-vocabulary vectors from one `counts_for_all_candidates` result.
These are the quantities the engine currently derives per candidate in Python:

| Vector | Formula (per row `c` of `counts`) | Engine twin |
|---|---|---|
| `n_groups` | `(counts[c] > 0).sum()` | `len(groups)` |
| `has_self` | `counts[c, 242] > 0` (all-green occurs iff candidate ∈ branch) | `_ALL_GREEN_PATTERN in groups` |
| `cost_lb` | `3.0 - (n_groups + has_self) / n` | `wordle_engine.py:1026` |
| `sum_k_squared` | `(counts[c] ** 2).sum()` (int64) | sort key at `wordle_engine.py:1184` |
| `max_group_size` | `counts[c].max()` | `score_groups` MAX_GROUP_SIZE |
| `entropy_gain` | `-Σ p log2 p` over nonzero sizes | `score_groups` ENTROPY_GAIN |

Package as `candidate_stats(subset_idx) -> dict of named vectors` (or a small
NamedTuple of arrays). `sum_k_squared` must be integer (exact), so the §4 sort
key is identical to the Python `sum(k*k for k in ...)`.

**Acceptance.** For several fixed branches (sizes ~8, 30, 100, 500), every
vector matches the per-candidate Python computation exactly (integers equal;
floats to full precision for `cost_lb`, `<1e-12` for entropy). Include a branch
containing candidate words (so `has_self` varies) and one not.

---

## §4. Engine integration: vectorized ranking and gating — requires §3

The highest-value change: eliminate the two per-candidate Python sweeps that
run at **every** solved node with `n >= ORDER_MIN_N`, including nodes where
everything below is a cache hit.

**4a. Best-first sort** (`wordle_engine.py:1181-1186`). Today:

```python
candidate_list = sorted(candidate_list,
    key=lambda c: sum(k*k for k in cache.group_counts(c, branch_words).values()))
```

Replace, when a `PatternMatrix` is available, with `sum_k_squared` from §3 and
a **stable** argsort (`kind='mergesort'`) over the keys *in `candidate_list`
order*. Python's `sorted` is stable, so equal-key candidates keep their
`candidate_list` order; the mergesort argsort reproduces exactly that. This is
what makes §0 rule 1 (identical winner) hold — do not skip the stability
requirement.

**4b. Pre-gating.** In `_solve_subset`'s candidate loop, before calling
`evaluate_candidate`, consult the precomputed `cost_lb` vector: if
`cost_lb[c] >= best_erd` (current running bound), record the same outcome the
engine would have produced (`OVER_ERD_LIMIT` → `cutoff_occurred = True`) and
skip the call. This is decision-identical to `evaluate_candidate`'s own gate at
`wordle_engine.py:1027` — same admissible bound, same comparison, evaluated
against the same running `best_erd` — but costs an array lookup instead of a
full `group_words` partition. The mid-loop publisher overrun check must still
run per iteration exactly as today (`wordle_engine.py:1211-1217`).

**4c. Plumbing.** Thread one shared `PatternMatrix` instance the same way
`ResponseCache` is shared today:

- `erd_swarm.py`: `_BranchWorker` builds/loads it once per process, next to its
  `ResponseCache`.
- `wordle.py`: build lazily at session start when NumPy is present (interactive
  use benefits too, e.g. the `s`/`b` full-vocabulary scoring commands can use
  §3 vectors — optional, separate commit).
- Signature suggestion: give `_solve_subset` / `min_expected_guesses` an
  optional `pattern_matrix=None` parameter, defaulting to the current behaviour.

**4d. Epoch bump.** Node timing changes → new `telemetry_epoch` row (label
`"numpy-kernel"`) on deploy, per §0 rule 5.

**Acceptance.**
- Equivalence test (the core deliverable): for a fixed set of branches (sizes
  ~8, 30, 81, 146 — reuse the `diag_order_tune.py` branch fixtures if
  convenient), solve with `pattern_matrix=None` and with the matrix; assert
  **identical** best_guess, ERD (exact float equality), `max_remaining_depth`,
  and identical cache writes.
- Benchmark script `diag_kernel_bench.py` (new): times both paths on the same
  branches, prints speedup; asserts result equality. Not part of the unittest
  suite (it is a diagnostic like the other `diag_*.py`).
- Full test suite passes with and without NumPy importable (run once with
  NumPy hidden via a test that monkeypatches availability, or a CI env split).

---

## §5. Vectorized group partitioning (`group_words` fast path) — requires §2; do after §4 and only if profiling justifies it

After §4, the remaining per-node Python cost is `cache.group_words` inside each
non-gated `evaluate_candidate` (`wordle_engine.py:1013`) — one Python loop over
the branch per candidate that survives gating. Replace its interior with the
matrix while preserving the exact structure the recursion depends on:

- Compute `pats = matrix[g, subset_idx]`, then build the `{pattern: [words]}`
  dict. **Group iteration order must match today's** (first-appearance order of
  each pattern while walking the subset, because `evaluate_candidate` sorts
  groups by size with a *stable* sort — `wordle_engine.py:1038` — so tie order
  inherits dict insertion order, and a different tie order can change which of
  several equal-ERD winners is found first). Recipe: stable-argsort `pats`,
  find group boundaries, then emit groups ordered by each pattern's first index
  in the subset. Words within a group keep subset order (stable argsort gives
  this for free).
- Membership fallback for words outside the answer universe stays (the
  `_answer_index.get(word) is None` path at `wordle_engine.py:412-415` /
  `:424-429`) — branch words are always answers in the swarm path, but the
  interactive fallback mode can pass non-answers; route those through the
  existing per-word path.

This is a drop-in inside `ResponseCache.group_words` (accepting an optional
precomputed index array), so no engine call sites change.

**Acceptance.** Property test: for random (guess, subset) pairs, new and old
`group_words` return dicts equal in keys, values, **and iteration order**. The
§4 equivalence test re-run with §5 enabled still shows identical solve results.

---

## §6. Candidate equivalence-class dedupe — requires §2; independent of §4/§5

Two candidates that induce the **same partition** of `branch_words` have the
same ERD against it: the recurrence (`evaluate_candidate`) reads only the
groups, the self-singleton, and the budget. On deep/small branches most of the
13k candidates collapse into few classes (e.g. all words whose letters are
disjoint from the branch's letters induce the single all-gray group).

**Where.** In `_solve_subset`, immediately after the §4a best-first ordering
and before the candidate loop, when a `PatternMatrix` is available and
`guesses is not None` (full-vocabulary sweeps only — answers-only sweeps are
small):

- Signature per candidate = the raw bytes of `matrix[g, subset_idx]`
  (from `patterns_for_candidates`). Build `{signature: first_candidate_in_order}`;
  evaluate only the representatives, in their existing order.

**Why the winner is unchanged.** The representative is the *earliest* class
member in the evaluation order. Today, the first evaluated member of a class
either sets `best_erd` or fails to beat it; later members of the same class
compute the identical cost and never *strictly* beat it (`cost < best_erd` at
`wordle_engine.py:1224`), so they can never become `best_guess`. Skipping them
changes nothing about the result — winner, ERD, and `max_remaining_depth` are
identical, not merely equivalent. Taint aggregation is also unaffected: a
skipped member would have produced the same `floor` flag as its representative.

**Scope limit.** Engine-inline solves only. Swarm top-level claiming hands out
candidates by index (`candidate_claims`), and finalize requires all
`n_candidates` slots done — claim-level dedupe therefore interacts with the
claim protocol and belongs with the packing work (a bundle could carry a whole
class and mark all members done together). Note it in `adaptive_claim_packing.md`
§12 as an open extension; do not attempt it here.

**Cost note.** The signature pass costs one `(G × n)` slice — the same data §4
already materializes; reuse it rather than re-slicing.

**Acceptance.**
- Unit test: on a branch over few distinct letters, class count is much smaller
  than candidate count (sanity), and the solve result equals the non-deduped
  solve exactly (winner, ERD, max_remaining_depth, cache writes).
- The §4 equivalence fixtures re-run with dedupe on: identical results.

---

## §7. Monster-branch calibration (operational, no code) — requires §1; §4 strongly recommended first

The schedule estimate for the full tree is dominated by an unmeasured regime.
Before mass-queueing anything:

1. Pick one weak opener with a large all-gray branch (over 1,000 answer words).
2. `queue-add --word <opener> --pattern ..... --priority 1000` (no cap after §1).
3. Let the swarm drain it; sub-branch promotion will fan it out.
4. Read the cost from the issue-67 measurement layer (`branch_finalize_log`
   rows for the branch and its promoted descendants, summed by spine), plus
   wall-clock span.
5. Extrapolate: (number of >300-word branches across all openers) × (measured
   cost, adjusted for cache warming) — this is the go/no-go number for the
   "weeks" schedule and decides how much §5/§6 matter.

Deliverable: a short section appended to this file (or a note in the issue)
with the measured numbers and the revised whole-job estimate.

---

## §8. Phone export filter (`export --reachable-only`) **[independent; needed before full-tree sync]**

`branch_best_by_policy` already holds 3M+ rows; a full run adds tens of
millions, most of which are search memoization (sub-branches of *losing*
candidates) the phone never consults. The playable tree is tiny by comparison:
from any top-level (opener, pattern) branch, play only follows `best_guess`
partitions downward.

**Change.** Add `--reachable-only` to `erd_search.py export`:

- Seed set: every branch with ≥ 2 words for every opener in `wordle.txt`
  (equivalently: every branch_key reachable as (opener, pattern) — these are
  the rows the phone looks up after the first guess). Computing seeds needs the
  pattern matrix or the decomposition blobs; reuse §2 if landed, else
  `ResponseCache.group_words`.
- BFS: for each cached seed, partition its words by its `best_guess` (the same
  walk `verify_erd_cache` does at `wordle_engine.py:1329-1360`), enqueue each
  ≥2-word sub-branch, export every visited row. Uncached nodes terminate their
  path (export is best-effort on a partial cache).
- Export the same three tables as today (`answer_list`,
  `response_decomposition`, `branch_best_by_policy`), just row-filtered on the
  third. Keep default behaviour (full export) unchanged.

**Acceptance.** Unit test on a small synthetic cache: reachable rows exported,
an unreachable row (sub-branch of a non-best guess) excluded; re-running is
incremental (INSERT OR IGNORE) as today. Document in SWARM.md's export section.

---

## §9. Documentation sync **[independent, small]**

- SWARM.md still says "Workers solve branches under `ROOT_BUDGET = 5`"; the
  code now has `ROOT_BUDGET = GAME_GUESSES` with per-branch budget =
  `ROOT_BUDGET − guess_depth` (`erd_swarm.py:98-101`). Fix the Budget section.
- After §1: SWARM.md `queue-add` examples and flag description (covered there).
- After §2/§4: add `pattern_matrix.py` to the layer tables in CLAUDE.md and
  design.md, and a short design.md section describing the matrix and which
  engine paths consult it.

---

## Sequencing

```
§1 (uncap queue-add)  ──────────────┐
§2 (pattern matrix)  → §3 (stats) → §4 (rank+gate) → §5 (group_words, if profiling says so)
                       §2 ─────────→ §6 (dedupe)
§1 + §4 ────────────────────────────→ §7 (monster calibration → schedule decision)
§8 (export filter)  — any time before the first full-tree phone sync
§9 (docs)           — with each section it trails
Claim packing        — proceeds on its own plan/gates, orthogonal to all of the above
```

Suggested order of landing: §1, §2, §3, §4 (the big win), §7 (measure), then
decide how much of §5/§6 the numbers demand while the packing work proceeds in
parallel.
