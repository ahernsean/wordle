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
   Wordle Tools' third-party reconstruction instead.) `Get_NYT_Candidates.py`
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

**Claim.** V ⊆ V′ implies ERD(B, V′) ≤ ERD(B, V), and any budget feasible
for (B, V) is feasible for (B, V′).

**Proof.** Any strategy T for (B, V) has every node label in V ⊆ V′, so T
is verbatim a strategy for (B, V′). Its cost profile is identical in both
problems: `calculate_response(g, a)` depends only on the pair (g, a), and
the tree's labels and the answer set B are unchanged, so every a ∈ B traces
the same path. Hence the strategy set for (B, V′) contains that for (B, V)
with costs preserved, and a minimum over a superset is at most the minimum
over the subset. Apply with the mean (ERD) and existence within a budget
(feasibility). ∎

**What this does not give: `max_depth` monotonicity.** The *abstract*
quantity "min over all strategies of the worst-case cost" is monotone
nonincreasing under the same restriction argument (T's worst case is
preserved verbatim in V′, so the minimum over a superset is
again at most the minimum over the subset). But that abstract quantity is
not what the cache stores. `_solve_subset` only updates
`best_max_remaining_depth` when a candidate strictly beats the running
`best_erd` (`wordle_engine.py`: `if cost < best_erd: ... best_max_remaining_depth
= max_remaining_depth`) — the stored `max_depth` is the worst case *of
whichever tree won the ERD race*, not a separately minimized worst case.
Growing V can hand the race to a new, lower-ERD tree with a *larger* worst
case than the old winner's, so the stored `max_depth` is not provably
monotone under vocabulary growth in either direction. (Mechanism 2 already
treats cached `max_depth` this way — an upper bound on the true worst-case
optimum, not an exact or monotone value; this note aligns the Theorem with
that rather than contradicting it.)

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
growth: the cost sum, and the mean restricted to the original words (both
provable by the same restriction argument as the Theorem above: any tree
solving the grown branch restricts to a valid, if suboptimal, tree for the
original words). `max_remaining_depth` is deliberately absent from that
list: it has the identical flaw as the Theorem's retracted claim, just
along the answer-set axis instead of the vocabulary axis — the stored
value tracks whichever tree wins the *ERD* race for the grown branch, an
independently-computed tree with no proven relationship to the original
branch's stored worst case. Per-candidate root ERDs can move both ways and
the best first guess can change — root results recompute regardless (the
full-set `branch_key` changes anyway).

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
  for the new id. Blobs are positional (byte *i* = answer *i* in sorted
  order), and the 9 new answer words are interleaved rather than appended,
  so a splice is not the cheap shortcut it sounds like — see Mechanism 4.
  Rebuild.

## Part 3 — The `ERD_ALL` recertification sweep (the centerpiece)

`verify_erd_cache.py` already implements exactly what's needed here — no
new engineering. It re-verifies every `erd_words_unfiltered` row leaves-first
against the true optimum, checking whether *any* candidate in the complete
`all_words` vocabulary beats the stored `best_score` (reading sub-branch
costs from the cache rather than recursing), correcting rows in place. It
opens the cache by loading `ANSWER_FILE`/`WORDS_FILE`
(`cache_sqlite.py`'s `answer_list_id` scoping then confines every query to
that namespace), so once Phase 0's rename lands and those two constants
point at the corrected files, running the tool unmodified *is* the
recertification sweep.

**The scan must cover every candidate, not just the added ones.** An
earlier draft of this section proposed scanning only the ~1,883 added
candidates per row, at an estimated ~15% of a full sweep's cost. That
undercounts: an *old* (non-added) candidate's cost depends recursively on
its response-groups' cached ERDs, so when the leaves-first sweep lowers a
child's value, a previously-losing old candidate can newly win even though
no added word is involved — checking only new candidates would miss that
case and could leave a suboptimal row stored as final. `verify_erd_cache.py`
already scans the complete candidate list per row for exactly this reason;
the "added-candidates-only" shortcut doesn't hold up, and this section no
longer claims it (also resolves the contradiction Part 5 already avoided
by saying scans always run to completion).

- Process branches leaves-first so corrected child values are in place
  before any parent is re-evaluated — this is what makes a full-scan sweep
  correct, not merely thorough: a parent's re-scan reads already-corrected,
  not stale, child costs.
- Per row: scan every word in the new candidate vocabulary (14,855, not
  12,972); correct the row if any candidate beats the stored value.

### Cost: grounded in a prior real run of this exact tool

This is not a projection from first principles — `verify_erd_cache.py` has
already run a full sweep of the whole `ERD_ALL` cache once, for an
unrelated reason (the reclaim-while-alive bug fix, completed 2026-07-12):
**3,485,333 rows, full candidate scan, 13h48m wall time, post-kernel** (the
kernel deployed 2026-07-05, so this figure already reflects it — no further
speedup to project on top). Same tool, same machine, same leaves-first
full-scan shape as this sweep, so it's the best available estimate.
Scaling for today's larger inputs — roughly 3.6M rows now vs. 3,485,333
then, 14,855 candidates vs. 12,972 — gives a rough **15–17 hour**
extrapolation: call it an overnight run, not a multi-day one. That's an
extrapolation from one data point, not a measurement; time an actual
sample wave on rocky first, using the tool's own `--start-size` resumption
support to bound the risk of committing to a bad estimate.

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

### Mechanism 4 — Pattern-matrix rebuild (minor; not a splice)

`response_decomposition` blobs are strictly positional — byte *i* is
answer *i* in canonical (sorted) answer-list order (`wordle_engine.py`:
`_encode_response(calculate_response(guess, answer)) for answer in
self.answer_words`). The corrected answer list is sorted with its 9 new
words interleaved, not appended — `aught` lands at index 179 of 3,209, for
instance — so "old blob + 9 bytes appended" would misalign every byte from
the first insertion point onward for every guess. There's no cheap splice
here: either insert each new byte at its correct index for every retained
blob, or just rebuild. Rebuilding is already fast (the vectorized kernel
path this project deployed 2026-07-05), so that's the recommendation; if a
splice is ever implemented instead, verify it byte-for-byte against a
fresh rebuild before trusting it.

### The honest limit

Nothing above certifies optimality by itself: ERD is a min over all
candidates and an added word can change the winner. Scans always run to
completion; the mechanisms make each scan cheap, not skippable.

## Part 6 — Procedure

One-time operations run from the scratchpad and are never committed; the
list-and-rename flip (Phase 0) is the one exception and goes through a PR.

### Phase 0 — Source, rename, and land the corrected vocabularies

1. Source the current 14,855-word NYT full dictionary — **done**.
   `Get_NYT_Candidates.py` (new script, mirrors `Get_NYT_Answers.py`'s
   style — both already carry their permanent names, renamed ahead of the
   file rename below since neither script's identity depends on which
   filename it currently writes to) scrapes the NYT Wordle web client's
   bundled dictionary directly; verified independently against a manual
   extraction, byte-for-byte identical.
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
   was easily confused with the file it's now renamed away from.

   Exhaustive inventory (verified by grepping the repository for both old
   filenames, not reconstructed from memory — an earlier draft of this
   list was incomplete):

   - **Routes through `runtime_paths.py` already — edit only the one file.**
     `erd_search.py` imports `DEFAULT_ANSWER_LIST_PATH`/
     `DEFAULT_CANDIDATE_LIST_PATH` rather than hardcoding either name, so
     updating `runtime_paths.py` alone is sufficient for it.
   - **Hardcode their own `ANSWER_FILE`/`WORDS_FILE` (or equivalent)
     literals — each needs a direct edit:** `wordle.py`, `erd_swarm.py`,
     `verify_erd_cache.py`, `verify_erd_losses.py` (hardcodes both names;
     Phase 5 runs it, so a missed edit surfaces there, not earlier),
     `import_cache.py` (hardcodes `NYT_wordlist.txt`; feeds the Phase 6
     cache-import workflow), `diag_ab_equiv.py`, `diag_ab_wall.py`,
     `diag_kernel_bench.py`, `diag_ordering.py`, `diag_order_tune.py`,
     `diag_toplevel_census.py`, and `Get_NYT_Candidates.py`'s new
     answer-list plausibility check (reads `NYT_wordlist.txt` directly,
     alongside its existing hardcoded write target — see Part 6's scraper
     fix). None of these currently import from `runtime_paths.py`;
     centralizing them through it instead of editing each literal is a
     legitimate alternative Phase 0 can choose at execution time, but
     that's a separate, larger decision this plan doesn't make now.
   - **Test files with hardcoded literals:**
     `tests/test_diag_toplevel_census.py`, `tests/test_erd_scaling.py`,
     `tests/test_kernel_equivalence.py`, `tests/test_pattern_matrix.py`,
     `tests/test_swarm_vs_engine_overhead.py`.
   - **Prose only — won't break, but will go stale:** `design.md`,
     `full_tree_plan.md`, `SWARM.md`, and a comment in
     `tests/test_queue_add.py`.

   No `schema_migrations` entry — this is a file rename, not a cache
   schema change. Run the full test suite after the rename lands, before
   any later phase begins; a missed hardcoded reference fails loudly there
   (`FileNotFoundError`) rather than silently.
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

1. Time a sample wave (`--start-size` bounds it to a small branch range)
   to confirm the ~15–17 hour extrapolation before committing to the full
   run (Part 3).
2. Run `python3.13 verify_erd_cache.py` unmodified, leaves-first, to
   completion. No `ERD_ALL` row is served to a user before its wave
   completes (phone is frozen; Linux use during the sweep is at-your-own-
   risk and confined to swept sizes).

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

- Eager vs. lazy for the un-pinched `erd_answers_unfiltered` remainder.
- Soak length before retirement.
- Whether to centralize the hardcoded `ANSWER_FILE`/`WORDS_FILE`-style
  consumers listed in Phase 0 through `runtime_paths.py` now, or just edit
  each literal — noted there as a real but separate decision.

## Resolved

- **14,855-word candidate list**: sourced via `Get_NYT_Candidates.py`,
  scraping the NYT Wordle web client's bundled dictionary directly.
- **`wordle.txt` removal set**: 0 removals (1,883 additions only) — the
  growth-only case applies throughout; no mixed-case handling is needed.
- **File naming**: `wordle.txt`/`NYT_wordlist.txt` → `all_candidates.txt`/
  `all_answers.txt`, folded into Phase 0, with an exhaustive (grep-verified)
  inventory of every consumer. Both scraper scripts already carry their
  permanent names (`Get_NYT_Wordlist.py` → `Get_NYT_Answers.py`,
  `Get_NYT_Words.py` → `Get_NYT_Candidates.py`) — only their write-target
  string literals (and `Get_NYT_Candidates.py`'s new answer-list check)
  still need updating at Phase 0.
- **Phone sync mechanism**: git pull (Working Copy) for code and lists;
  the existing manual export/import dance for the cache, independently —
  see Phase 6.
- **Recertification sweep implementation**: no new engineering —
  `verify_erd_cache.py` already scans the full candidate list per row;
  running it unmodified after Phase 0's rename *is* Phase 3. Cost grounded
  in a real prior full sweep of this cache (13h48m, 2026-07-12): expect
  roughly 15–17 hours, confirm with a timed sample wave before committing
  to the full run.
- **Max_remaining_depth is not provably monotone** under either vocabulary
  growth (Part 1) or answer-set growth (Part 1, "ERD is not monotone") —
  the cache only stores the worst case of whichever tree wins the *ERD*
  race, not a separately minimized worst case. Two theorem statements
  corrected; ERD monotonicity and budget-feasibility monotonicity are
  unaffected and still hold.
