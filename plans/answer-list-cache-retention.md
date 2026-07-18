# Vocabulary correction: cache retention, recertification, and reverification plan

## Context

Two staleness discoveries, in escalating order:

1. **Answer list.** The 2026-07-15 Wordle answer was PSHAW — absent from
   `NYT_wordlist.txt`. The corrected list is captured as
   `NYT_wordlist_2026-07-17.txt`: **9 additions, 0 deletions**
   (3,200 → 3,209 words):

       aught  genus  krone  prate  pshaw  ravel  shlep  stagy  techy

2. **Candidate universe.** Wordle Tools reports the NYT full dictionary at
   **14,855 words**; our `wordle.txt` has 12,972 — a stale snapshot from the
   pre-NYT era. Direct local evidence of the staleness: 14 answer words
   (`amaro crema flowy glamp glowy hacky janky koran popup queso quran
   runup untag venti`) appear in the answer list but not in `wordle.txt`,
   which is impossible in real Wordle (every answer is an accepted guess).

   **Sourced and confirmed.** The current 14,855-word dictionary is shipped
   client-side in the NYT Wordle web app's own webpack bundle — needed for
   instant offline guess validation. (The answer list is not shipped there;
   revealing it would spoil the game, which is why it stays sourced from
   Wordle Tools' third-party reconstruction instead.) `Get_NYT_Words.py`
   extracts it by shape — a run of ≥5,000 quoted 5-letter words inside a
   downloaded chunk — rather than a hardcoded chunk filename, since NYT's
   chunk hashes and numbering change on every deploy. The delta against
   `wordle.txt` is **1,883 additions, 0 removals**; the corrected answer
   list is confirmed a subset of the corrected candidate list, including
   all 14 previously-missing words and all 9 PSHAW-era additions. Zero
   removals resolves Part 2's table to the growth-only row throughout:
   `ERD_ALL` rows are valid upper bounds/incumbents, not mere heuristics.

The candidate-universe change dominates the planning: `ERD_ALL`'s guess
vocabulary is `wordle.txt`, so the namespace that the answer-list-only
analysis retained as exact is precisely the one a candidate-universe change
demotes. Both corrections must land as **one combined uplift**, candidate
universe first in the analysis: recertifying `ERD_ALL` against a vocabulary
known to be wrong, then invalidating it again, would do the work twice.

Operating decisions already made:

- The phone stays on the old lists — no pull from origin — until the uplift
  is entirely done. The Linux side flips wholesale when execution starts.
- The epoch swarm is already stopped for the duration; epoch testing
  resumes after retirement.

## Part 1 — Foundations (unchanged by the second discovery)

### Content-addressed keys freeze each row's answer set

`ScoreCache.encode_subset` keys every row by the sorted concatenation of the
branch's actual words. Growing the answer list changes *which keys arise*
in the game tree — never the meaning of an existing key. `_solve_subset`
partitions `branch_words` itself to form sub-branches and never consults
the global list to enumerate answers. The only input that can vary for a
fixed key is the guess vocabulary its policy draws from.

### Theorem: guess-vocabulary growth can only lower ERD (branch fixed)

Fix a branch word set B and guess vocabulary V. A strategy for (B, V) is a
decision tree with every node labeled by a guess g ∈ V; an answer a ∈ B
descends along the `calculate_response(g, a)` edge at each node and
terminates at the first node labeled a; cost(a) is the path length. ERD and
max_remaining_depth are the min over strategies of the mean and max of
cost(a) over a ∈ B. (This models the code: `guesses` is threaded unchanged
into every recursive `_solve_subset` call, so unfiltered policies draw from
the same fixed V at every node.)

**Claim.** V ⊆ V′ implies ERD(B, V′) ≤ ERD(B, V),
max_remaining_depth(B, V′) ≤ max_remaining_depth(B, V), and any budget
feasible for (B, V) is feasible for (B, V′).

**Proof.** Any strategy T for (B, V) has every node label in V ⊆ V′, so T
is verbatim a strategy for (B, V′). Its cost profile is identical in both
problems: `calculate_response(g, a)` depends only on the pair (g, a), and
the tree's labels and the answer set B are unchanged, so every a ∈ B traces
the same path. Hence the strategy set for (B, V′) contains that for (B, V)
with costs preserved, and a minimum over a superset is at most the minimum
over the subset. Apply with the mean (ERD), the max (max_remaining_depth),
and existence within a budget (feasibility). ∎

**Corollaries.**
- *Growth:* old exact values become upper bounds; old loss proofs become
  invalid (a new guess word may rescue the branch). A loss over the larger
  vocabulary implies a loss over any sub-vocabulary.
- *Removal:* the inequalities reverse — old exact values become lower
  bounds, and a stored best_guess may no longer be a legal word.
- *Mixed additions and removals:* no per-row direction is provable; old
  values are heuristics only.

### ERD is not monotone in the answer set (why the root must recompute)

ERD is a mean, not a sum. Adding a word that resolves in 2 guesses to a
branch of n words with ERD e yields (n·e + 2)/(n + 1) < e whenever e > 2;
adding a hard word raises the mean. Monotone nondecreasing under set
growth: max_remaining_depth, the cost sum, and the mean restricted to the
original words. Per-candidate root ERDs can move both ways and the best
first guess can change — root results recompute regardless (the full-set
`branch_key` changes anyway).

### The answer-side recompute region is a union of 9 thin cones

For any first guess, the 9 new answer words land in at most 9 of its
nonempty response groups; every other group is a byte-identical key.
Within an affected group, the next guess isolates each new word into one
sub-group again, so the changed region under each new word is a single
narrowing cone. (This bounds which *keys* are new; whether the retained
keys' *values* still serve depends on the vocabulary analysis below.)

## Part 2 — Disposition by namespace under the combined change

| Policy | Guess vocabulary | Effect of combined change | Disposition |
|---|---|---|---|
| `erd_answers_compliant` (`ERD_ANSWERS`) | the branch words themselves (`guesses=None` ⇒ `candidate_list = branch_words`, recursively) | pure function of `branch_key`; neither list enters: **exact** | retain — the only fully exact ERD namespace |
| `erd_words_unfiltered` (`ERD_ALL`) | `all_candidates.txt` (formerly `wordle.txt`) — grew by 1,883, 0 removals (confirmed) | growth-only case applies: rows demote to **upper bounds** (legal best_guess, valid incumbent); losses **invalid** regardless | retain rows as **recertification seeds**, not servable truths; drop losses |
| `erd_answers_unfiltered` (`ERD_ANSWERS_UNFILTERED`) | answer list — grew by 9 | rows demote to upper bounds; losses invalid | rebuild via sandwich seeding (Part 4) after the `ERD_ALL` sweep |
| `erd_words_compliant` (`ERD_CONSTRAINED`) | path-dependent | never persisted | n/a |

Other tables:

- **`candidate_scores`** — a score is a function of (branch word set,
  candidate, method) only; retain all rows. Rows for removed candidates
  become dead weight (harmless); scores for added candidates and grown
  branches are new keys computed on demand.
- **`branch_loss_by_policy`** — retain `erd_answers_compliant` losses
  (exact); drop `ERD_ALL` and `erd_answers_unfiltered` losses (invalid
  under vocabulary growth). Losses rebuild organically as the swarm
  re-proves deep branches.
- **`response_decomposition`** — keyed (guess, answer_list_id); rebuilds
  for the new id. Blobs for retained guesses can be spliced (old blob + 9
  bytes); added guesses need fresh blobs; removed guesses' blobs are dead
  weight.

## Part 3 — The `ERD_ALL` recertification sweep (the centerpiece)

`verify_erd_cache.py` already implements the needed shape: leaves-first
re-verification of every `ERD_ALL` row against the true optimum, reading
sub-branch costs from the cache and correcting rows in place, waves
parallelized within a branch size. Recertification after a vocabulary
change is the same computation with a different candidate delta:

- Process branches leaves-first so corrected child values are in place
  before any parent is re-evaluated.
- Per row: if the stored best_guess was removed from the vocabulary,
  re-solve the branch (warm — Mechanism 1 below). Otherwise re-validate
  the stored value against updated children and evaluate the **added**
  candidates; in the growth-only case the stored value is a valid
  incumbent (upper bound), so most rows close after the added-candidate
  scan.
- Per-branch incremental cost ≈ |added candidates| evaluations plus drift
  re-checks — order 15% of a full sweep per branch (1.9k of 14.9k
  candidates), against the measured ~900 full sweeps/day pre-kernel
  baseline (`full_tree_plan.md`) with the kernel's 10–50× on top. Real
  numbers come from timing a sample wave on rocky before committing.

Whether this lands as an extension of `verify_erd_cache.py` (committed —
vocabulary changes will happen again) or a scratchpad variant is decided on
rocky; the leaves-first wave structure and cache-read model carry over
either way.

## Part 4 — Rebuilding `erd_answers_unfiltered`

### The vocabulary-inclusion sandwich — gated, and only post-sweep

With the corrected lists, verify (do not assume): every answer word ∈ new
`all_candidates.txt`. Then for any branch B, B ⊆ answers′ ⊆ all-words′, and
the theorem gives:

    ERD_ALL′(B)  ≤  ERD_ANSWERS_UNFILTERED′(B)  ≤  ERD_ANSWERS(B)

**The left value must be the post-sweep (recertified) `ERD_ALL` value.**
The sandwich was unsound against the old cache for two reasons: the old
`ERD_ALL` values were computed over a vocabulary that did not contain the
answer list (14 missing words — the inclusion simply failed), and the
vocabulary has now changed besides. Both are repaired only after Phase 0
fixes the lists and Phase 3 recertifies.

Where the outer values are equal, seed the row by **copying the entire
`erd_answers_compliant` row** (best_guess, best_score, max_depth): its
strategy's guesses all lie within the branch ⊆ answers′, so it is a legal
`erd_answers_unfiltered` strategy attaining the pinned score — giving the
seed a usable best_guess and max_depth (a NULL-max_depth row is never
reused at budgeted queries). Guard the pinch on `solve_budget IS NULL` on
both sides: tainted rows hold budget-specific values and pin nothing.

Do **not** seed losses from old `ERD_ALL` losses (invalid under growth).
Once the sweep re-proves an `ERD_ALL` loss under the new vocabulary, it
transfers validly (loss over the superset vocabulary implies loss over
answers′ ⊆ all-words′).

### Cost model

An unfiltered sweep is 3,209 candidates vs. 14,855 — ~4.6× cheaper per
branch than a new-universe `ERD_ALL` sweep. Only un-pinched rows need
sweeps:

    rebuild time ≈ (rows where ERD_ALL′(B) ≠ ERD_ANSWERS(B), untainted)
                   / (unfiltered sweeps per day on current engine)

Numerator: one SQL join after the Phase 3 sweep. Denominator: time a
sample. If the un-pinched population is large, seed the pinched rows
(zero search) and leave the rest lazy.

## Part 5 — How retained work accelerates everything

G′ = G ∪ {added words} denotes a grown branch, G its retained base.

### Mechanism 1 — Recurrence-level reuse (automatic)

To evaluate a candidate c on G′, `_solve_subset` needs each response
group's ERD; the added words land in few groups, and every unchanged group
is the identical word set with a cached row read instead of recursed into.
Per-candidate cost = cached lookups + recursive solves of the few changed
groups, memoized across candidates. A cache-missed key near the root is a
solve against a warm cache. This also powers the sweep's re-solves.

### Mechanism 2 — O(1) transfer bounds (scoped; not yet implemented)

Valid **only for fixed-vocabulary policies** (`erd_words_unfiltered`,
`erd_answers_unfiltered`): in `erd_answers_compliant` the vocabulary *is*
the branch, so growing the branch also grows the vocabulary — the
restriction argument fails, and the lower-bound claims are genuinely false
there (G can be a compliant loss at budget b while G′ is compliant-solvable
at b). A policy-blind implementation would write false losses.

For fixed vocabulary, from G's cached row (E = best_score, md = max_depth,
losses), with k added words:

- **Loss transfer.** A proven loss for G within b is a proven loss for G′
  within b (restriction). Valid only when the loss itself is valid under
  the current vocabulary — post-sweep losses, not pre-uplift ones.
- **Feasibility (one-sided).** The cached md is the worst case *of the
  ERD-optimal strategy*, an upper bound on the true worst-case optimum. So
  budget ≥ md(G) + k ⇒ G′ feasible (extend the cached strategy: run each
  added word through the tree; where its response leaves the tree, extend
  with one guessing node; collisions chain, so the j-th word colliding on
  one path costs up to md + j, but md + k bounds the worst case overall).
  The infeasible side (budget < md ⇒ loss) is **unsound** — a strategy
  with a worse mean can have a smaller max; only a true loss row proves
  infeasibility.
- **ERD envelope.** (n·E + k)/(n + k) ≤ ERD(G′) ≤
  (n·E + k·md(G) + k(k+1)/2)/(n + k). The lower bound is an admissible
  parent-pruning floor; the upper bound's collision term is the honest
  worst case of the extension construction. (ERD(G′) ≥ E is NOT valid —
  see non-monotonicity.)

### Mechanism 3 — Warm-starting the candidate scan

G's cached best_guess evaluated first gives a strong incumbent; G's
candidate_scores ranking is a near-perfect evaluation order (exact scores
shift only through the added words' group membership — an O(1) update per
candidate given their pattern bytes).

### Mechanism 4 — Pattern-matrix splice (minor)

Retained guesses' `response_decomposition` blobs splice (old blob + 9 new
bytes); only the ~1.9k added guesses need fresh blobs. An economy, not a
necessity.

### The honest limit

Nothing above certifies optimality by itself: ERD is a min over all
candidates and an added word can change the winner. Scans always run to
completion; the mechanisms make each scan cheap, not skippable.

## Part 6 — Procedure

One-time operations run from the scratchpad and are never committed;
the list flips and any `verify_erd_cache.py` extension go through PRs.

### Phase 0 — Source, rename, and land the corrected vocabularies

1. Source the current 14,855-word NYT full dictionary — **done**.
   `Get_NYT_Words.py` (new script, mirrors `Get_NYT_Wordlist.py`'s style)
   scrapes the NYT Wordle web client's bundled dictionary directly; verified
   independently against a manual extraction, byte-for-byte identical.
2. Delta against `wordle.txt` computed in both directions — **done**:
   1,883 additions, 0 removals. No mixed-case handling needed anywhere in
   this plan; the growth-only rows throughout apply as written.
3. Gates confirmed: new answer list ⊆ new candidate list (the 14 words
   included, plus all 9 PSHAW-era words); both source lists sorted, unique,
   5-letter lowercase.
4. **Rename** `wordle.txt` → `all_candidates.txt` and `NYT_wordlist.txt` →
   `all_answers.txt`, landing the corrected content under the new names in
   the same PR (no separate rename pass). `all_candidates.txt` names the
   file for what it is — the universe the anchored **candidate** vocabulary
   term (AGENTS.md) is drawn from — and pairs cleanly with
   `all_answers.txt`, whereas `wordle.txt` was opaque and `NYT_wordlist.txt`
   was easily confused with the file it's now renamed away from. Files to
   update: `runtime_paths.py` (`DEFAULT_ANSWER_LIST_PATH`,
   `DEFAULT_CANDIDATE_LIST_PATH`), `wordle.py` (`ANSWER_FILE`,
   `WORDS_FILE`), `erd_swarm.py`, `erd_search.py`, `verify_erd_cache.py`,
   `Get_NYT_Wordlist.py` and `Get_NYT_Words.py` (their write targets), any
   test fixtures/paths referencing the old names, and prose references in
   `SWARM.md`/`AGENTS.md`. No `schema_migrations` entry — this is a file
   rename, not a cache schema change.
5. Land the rename plus the corrected content as one PR. No temporary
   dated-capture file is kept once landed — git history holds the old
   content, and the old-`answer_list_id` cache rows (Phase 2 onward) hold
   the old world's computed results; neither needs a parallel on-disk file
   to survive through the soak. The phone does not pull until Phase 6.

### Phase 1 — Freeze and back up

1. Swarm already stopped; confirm no live claims (`erd_search.py view`).
2. Checkpoint and back up `wordle_cache.sqlite3` and `erd_queue.sqlite3`.
3. Check disk headroom: retained tables roughly double until retirement.

### Phase 2 — Re-tag migration

1. Open the cache once with the new answer list so
   `ScoreCache._ensure_answer_list` registers the new `answer_list_id`.
2. `INSERT OR IGNORE` old-id → new-id:
   - `branch_best_by_policy`: `erd_answers_compliant` rows (exact) and
     `erd_words_unfiltered` rows (recertification seeds).
   - `branch_loss_by_policy`: `erd_answers_compliant` only.
   - `candidate_scores`: all rows.
3. Copy nothing for `erd_answers_unfiltered`; copy no `ERD_ALL` losses.
4. Old-id rows stay intact — rollback path until retirement.

### Phase 3 — `ERD_ALL` recertification sweep

1. Time a sample wave; extrapolate; decide committed-extension vs.
   scratchpad variant of `verify_erd_cache.py`.
2. Run the sweep leaves-first to completion. No `ERD_ALL` row is served
   to a user before its wave completes (phone is frozen; Linux use during
   the sweep is at-your-own-risk and confined to swept sizes).

### Phase 4 — Recompute and seed

1. Clear and re-seed the queue from the new 3,209-word root; the swarm
   computes the root census and the 9 answer-cones (all against the new
   candidate universe, warm via Mechanism 1).
2. Sandwich-seed `erd_answers_unfiltered` per Part 4 (post-sweep values,
   untainted guard, full compliant-row copy, no loss seeding).
3. Run the un-pinched count; decide eager rebuild vs. lazy.

### Phase 5 — Reverification

1. **Structural reconciliation.** Row counts per (policy, answer_list_id)
   match the copy/seed/rebuild ledger.
2. **Sampled exactness check** (`erd_answers_compliant` and seeded
   unfiltered rows): fresh from-scratch solve per sampled branch; require
   `best_score` equality against the fresh solve; then verify the stored
   pair directly — evaluate the stored best_guess on the branch and
   require it to attain the stored best_score and stored max_depth.
   (Fresh-solve max_depth equality is tie-dependent — an equally-optimal
   different guess may carry a different worst case — so the stored pair,
   which is what the row asserts, is what gets checked.)
3. **Sweep audit.** Re-run the Phase 3 sweep's verifier mode over a random
   sample of swept `ERD_ALL` rows with fresh solves (same stored-pair
   discipline).
4. **Loss sweep.** `verify_erd_losses.py` over retained compliant losses.
5. **Test suite.** `python -m unittest discover -s tests -t . -p 'test_*.py'`.

### Phase 6 — Phone catch-up

Only after Phase 5 passes in full. Code/list sync and cache sync are two
separate, decoupled mechanisms on this project, not one combined step:

1. **Code and lists.** The phone (Working Copy) pulls `main` — a plain git
   pull, picking up the Phase 0 rename/content PR and everything since.
2. **Cache.** The already-established export/import dance brings the
   phone's database in line with rocky's post-migration, post-sweep
   `wordle_cache.sqlite3` (`wordle_cache.sqlite3` itself is not tracked in
   git, so step 1 alone never touches it).

Order between the two doesn't threaten correctness either way:
`_ensure_answer_list` always computes `answer_list_id` fresh from whichever
list files are currently loaded, and every read filters on that id, so an
old-code/new-cache or new-code/old-cache mismatch just produces cache
misses (the engine falls through to computing on the fly) rather than a
wrong answer. Do step 1 before step 2 anyway, for a mundane reason: no
sense importing a multi-gigabyte database before the code that can make use
of its new-id rows is even in place.

No schema change and no `schema_migrations` entry either way — this is a
data and file-content migration, not a structural one.

### Phase 7 — Retirement

After enough post-cutover use to trust the new world — the soak should
include simulated games against each of the 9 new answer words and a few
of the 14 rescued guess words:

1. Delete old-`answer_list_id` rows from all four tables;
   vacuum/checkpoint; re-sync the phone.
2. Resume epoch testing (fresh `telemetry_epoch`).

## Open items

- Committed extension vs. scratchpad variant for the `ERD_ALL`
  recertification sweep.
- Eager vs. lazy for the un-pinched `erd_answers_unfiltered` remainder.
- Soak length before retirement.

## Resolved

- **14,855-word candidate list**: sourced via `Get_NYT_Words.py`, scraping
  the NYT Wordle web client's bundled dictionary directly.
- **`wordle.txt` removal set**: 0 removals (1,883 additions only) — the
  growth-only case applies throughout; no mixed-case handling is needed.
- **File naming**: `wordle.txt`/`NYT_wordlist.txt` → `all_candidates.txt`/
  `all_answers.txt`, folded into Phase 0.
- **Phone sync mechanism**: git pull (Working Copy) for code and lists;
  the existing manual export/import dance for the cache, independently —
  see Phase 6.
