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
terminates at the first node labeled a; cost(a) is the path length. ERD is
the min over strategies of the mean of cost(a) over a ∈ B; call the min
over strategies of the max the *strategy-minimal worst case* — deliberately
not named `max_remaining_depth`, because the stored `max_remaining_depth`
is a different quantity (see the note below). (This models the code:
`guesses` is threaded unchanged
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

**What this does not give: `max_depth` monotonicity.** The strategy-minimal
worst case defined above is monotone
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

`verify_erd_cache.py` supplies the sweep's skeleton: it walks every
`erd_words_unfiltered` row leaves-first (ascending branch size, so
corrected child values are in place before any parent is re-evaluated),
re-scores each row by scanning the complete candidate vocabulary with
sub-branch costs read from the cache, and corrects rows in place. It opens
the cache by loading `ANSWER_FILE`/`WORDS_FILE` (`cache_sqlite.py`'s
`answer_list_id` scoping then confines every query to that namespace), so
after Phase 0's rename those two constants point at the corrected files.

**Unmodified, though, the tool audits — it does not recertify.** Two of
its behaviours, both harmless for its original purpose (detecting
possibly-corrupted values under an *unchanged* vocabulary), are unsound
here:

1. **It silently skips any candidate with an uncached response group**
   (`_erd_from_cache`: `if cached is None: return None`). The 1,883 added
   candidates have never been evaluated anywhere, so their partitions of
   existing branches are mostly keys that never arose in prior search —
   the loop iterates all 14,855 words but silently declines to evaluate
   precisely the candidates whose addition is the reason for the sweep.
   A row logged CONFIRMED means only "no *evaluable* candidate beats it."
2. **Its cache reads are taint-blind.** A tainted child (`solve_budget`
   set) holds a budget-limited value — an upper bound on the unconstrained
   optimum — and any evaluation through it inherits that one-sidedness.

Both failures point the same direction: post-sweep values would remain
*upper bounds*, not exact — and an upper bound recorded as exact is
precisely the failure mode Part 4's pinch cannot tolerate (a skipped
candidate can hide a lower true `ERD_ALL′`, so a pinched seed would
overstate the optimum). Exactness also propagates: a row's re-scored value
is exact only if every child value consulted past the admissible pre-gate
was itself exact and present, so one unresolved child taints every
ancestor whose scan reads it.

**Decided design: solve missing sub-branches inline.** Extend the sweep —
a committed extension to `verify_erd_cache.py`, written at Phase 3 time —
so that a candidate's missing sub-branch triggers an engine solve at
unconstrained budget (a warm-cache solve per Mechanism 1, writing a
legitimate new row) instead of skipping the candidate, and a tainted child
read likewise triggers an unconstrained re-solve instead of silent
acceptance. Leaves-first order then maintains the invariant that every
consulted child is exact, so every completed row is exact and Part 4's
pinch may use any post-sweep value.

**Missing sub-branches are the common case, not an edge case — and that's
fine.** Direct measurement (below) confirms why: a new candidate's response
partition of an existing branch is a *combination* of old answer words that
essentially never coincides with a partition any old candidate produced, so
most candidates against most branches hit an uncached sub-branch. In
practice this means most rows cost close to a full fresh solve rather than
a cheap audit-and-skip — measured and priced below, and still a tractable
overnight-class run, not a blocker. The read-only-certified fallback
described in an earlier draft (defer uncertified rows, keep the sweep pure
read-only) is no longer the live design: at the measured cost there is no
need to trade correctness coverage for speed. Retained only as a
contingency if a production wave runs materially slower than sampled.

The scan covers every candidate, not just the added ones: an *old*
candidate's cost depends recursively on its response-groups' cached ERDs,
so when the leaves-first sweep lowers a child, a previously-losing old
candidate can newly win even though no added word is involved. An
"added-candidates-only" shortcut misses that case (an earlier draft
claimed it at ~15% of full cost; withdrawn).

**Why this must stay a standalone forced-recompute pass, never the normal
swarm.** `branch_best_by_policy` keys rows on `(branch_key, policy,
answer_list_id)` only — there is no guess-vocabulary component in the
schema (confirmed by inspection of `cache_sqlite.py`'s table definitions).
`_cache_reuse` (`wordle_engine.py`) has no notion of "verified under the
current vocabulary" either — it accepts any present, untainted-or-
appropriately-tainted row at face value. So a Phase-2-copied-forward row is
indistinguishable, to any ordinary read, from one the sweep has already
recertified: a normal swarm worker claiming that branch would serve the
stale value as if exact. Only an explicit delete-then-resolve pass (what
`verify_erd_cache.py` and this section's extension both do) forces
recomputation. This is also why Phase 4's normal-swarm root-census-and-
cones work is safe *after* Part 3 finishes but the swarm must stay off
this table before then — not just an operational rule (Phase 3 step 3
already says so) but a correctness requirement.

**Monitoring differs between the two halves of the uplift.** The sweep
(this Part) runs as a standalone script — its own worker pool, its own
`--log` file, its own stdout progress lines — and does **not** register
with `erd_search.py view --workers`, since it never touches
`erd_queue.sqlite3`. Watch its log/stdout directly. Phase 4's root-census-
and-cones work, by contrast, *is* routed through the normal swarm and is
fully visible through the usual `erd_search.py view` reports.

### Cost: measured directly (2026-07-19/20, idle machine, two sampling passes)

`verify_erd_cache.py` ran a full sweep of the whole `ERD_ALL` cache once
before, for an unrelated reason (the reclaim-while-alive bug fix, completed
2026-07-12): 3,485,333 rows, full candidate scan, 13h48m wall time. That
number bounds only the tool's original *read-only, unchanged-vocabulary*
behavior and does not price this sweep's inline solves — superseded by
direct measurement below.

**Method.** Against a read-only production cache and a scratch database
(no production writes), for branches sampled from every `ERD_ALL` size
bucket: solve fresh under the new 14,855-word vocabulary (populating
children), then delete just the top row and re-solve with children warm —
isolating the steady-state per-row cost a leaves-first sweep would see.
Two passes: an initial 208-branch stratified sample across sizes 4–200,
then a 251-branch targeted resample of the 16–30-word bucket (the
production histogram's largest concentration — 1.15M of 3.6M rows) with
exact per-size row-count weighting, since that bucket's cost turned out to
be the dominant source of uncertainty.

**Result: 3.0–8.7 days on 6 workers; 5.4 days is the central estimate**
(780 serial hours, trimmed-mean method). The spread comes from a genuine
heavy tail — most branches solve in under a second, but near-exact-tie
branches (alpha-beta gets no early cutoff) can take tens to low hundreds of
seconds, and 4.2% of the 16–30-word bucket didn't finish within a 90s
per-branch cap in the second pass. Plan for the low end of a week; budget
for the full week as a ceiling.

**Correctness, corroborated empirically.** Across both passes, 432 branches
completed a fresh solve; 28 (6.5%) produced a strictly better ERD than the
old stored value (a real, solver-verified improvement — new candidates
winning branches they couldn't reach before), and **zero** produced a worse
one. Zero anomalies is exactly what the Part 1 growth-only theorem
predicts and would have been the signature of real cache corruption had
any existed. It didn't.

**The old cache's cost-side value is the row-key list and leaves-first
scheduling, not the stored scores.** A direct test — price the old
`best_guess` first and feed its cost to the solver as an alpha-beta
ceiling, instead of leaving the ceiling at ∞ — showed no net benefit (71
faster / 110 flat-or-slower across 181 branches, sums within 0.2% of each
other). The engine's own candidate ordering (`wordle_engine.py`: sorts by
Σk², response-group-size proxy, for any branch ≥8 words) already finds a
near-optimal incumbent on its own, and evaluating the old guess separately
just to seed a ceiling mostly duplicates work the ordinary scan would do
anyway. See the corresponding note on Mechanism 2 in Part 5 — this doesn't
touch the mechanism's *correctness*, only its practical payoff here.

## Part 4 — Rebuilding `erd_answers_unfiltered`

### The vocabulary-inclusion sandwich — gated, and only post-sweep

With the corrected lists, verify (do not assume): every answer word ∈ new
`all_candidates.txt`. Then for any branch B, B ⊆ answers′ ⊆ all-words′, and
the theorem gives:

    ERD_ALL′(B)  ≤  ERD_ANSWERS_UNFILTERED′(B)  ≤  ERD_ANSWERS(B)

**The left value must be a post-sweep `ERD_ALL` value that is exact** —
under the inline-solve sweep, any completed row; under the read-only
fallback, only rows the sweep certified (Part 3). A merely-demoted upper
bound on the left can equal `ERD_ANSWERS(B)` while the true `ERD_ALL′(B)`
is lower, in which case the pinch conclusion fails and the seed would
overstate the optimum while recorded as exact. The sandwich was also
unsound against the old cache for two independent reasons: the old
`ERD_ALL` values were computed over a vocabulary that did not contain the
answer list (14 missing words — the inclusion simply failed), and the
vocabulary has now changed besides. All of this is repaired only after
Phase 0 fixes the lists and Phase 3's sweep produces exact values.

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
solve against a warm cache. This is also what keeps the sweep's inline
solves of missing sub-branches cheap (Part 3) — the sweep's read-only scan
itself never recurses.

### Mechanism 2 — O(1) transfer bounds (scoped; measured not worth building)

**Empirical note (2026-07-19/20).** This mechanism transfers bounds along
the *answer-set-growth* axis (G → G′, new answer words landing in an
existing branch); the direct measurement in Part 3 tested the adjacent
*vocabulary-growth* axis instead (same branch, old best_guess used as a
solve ceiling under the grown candidate list) and found no net benefit —
the engine's built-in Σk² candidate ordering already reaches a near-optimal
incumbent without external help. That root cause (the ordering heuristic,
not anything specific to which axis grew) has no reason to stop applying
here, so the same lack of payoff is the expected outcome for Mechanism 2's
ceiling-style reuse too — but this is an inference from an adjacent result,
not a direct measurement of G→G′ specifically. The math below remains
correct regardless (an admissible bound can only prune, never mislead); the
open question is only whether implementing it is worth the code, and
current evidence says no. Retain the proofs — they're cheap insurance if a
future engine change (e.g. a weaker default ordering) makes them pay off —
but do not build the ceiling-injection machinery without re-measuring.

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
  (n·E + k·md(G) + k(k+1)/2)/(n + k). The lower bound requires E ≤ true
  ERD(G) — i.e. an exact (post-sweep, certified) or lower-bound E; a
  demoted upper-bound E inflates the floor, making it inadmissible and
  able to prune the true optimum. Same discipline as the loss-transfer
  bullet. The upper bound needs no such gate (it extends an actual cached
  strategy), and its collision term is the honest worst case of the
  extension construction. (ERD(G′) ≥ E is NOT valid — see
  non-monotonicity.)

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
     `diag_toplevel_census.py`, `Get_NYT_Answers.py` (its hardcoded
     `NYT_wordlist.txt` write target), and `Get_NYT_Candidates.py` (its
     hardcoded `wordle.txt` write target, plus its answer-list
     plausibility check, which reads `NYT_wordlist.txt` directly and
     raises — rather than silently skipping the gate — when the file is
     absent, so a missed edit here fails as loudly as the rest of this
     list). None of these currently import from `runtime_paths.py`;
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

1. Write the sweep extension (inline solves of missing sub-branches,
   tainted-child re-solves; Part 3) as a committed change to
   `verify_erd_cache.py`, through a PR.
2. Confirm the swarm is idle (`erd_search.py view --workers --format json`
   — check, don't assume) before starting; keep it stopped for the whole
   phase. Required, not just operational hygiene: Part 3 explains why a
   live worker could silently serve a not-yet-recertified row as exact.
3. Run the extended sweep leaves-first to completion (~3–8.7 days on 6
   workers, ~5.4 days central estimate — measured directly, Part 3; no
   further sample-wave timing needed). Monitor via the sweep's own
   `--log` file and stdout, not `erd_search.py view` (Part 3 — it doesn't
   run through the queue). No `ERD_ALL` row is served to a user before its
   wave completes (phone is frozen; Linux use during the sweep is
   at-your-own-risk and confined to swept sizes).

### Phase 4 — Recompute and seed

1. Clear and re-seed the queue from the new 3,209-word root; the swarm
   computes the root census and the 9 answer-cones (all against the new
   candidate universe, warm via Mechanism 1).
2. Sandwich-seed `erd_answers_unfiltered` per Part 4 (exact post-sweep
   values only — certified rows if the fallback sweep ran — untainted
   guard, full compliant-row copy, no loss seeding).
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

Resolved since the last revision (previously open): sweep cost (measured,
Part 3), inline-solve vs. read-only-certified fallback (inline-solve
chosen — affordable at the measured cost), whether Mechanism 2's
ceiling-injection machinery is worth building (measured negligible payoff;
proofs retained, implementation deferred), and whether Part 3 can run
through the normal swarm/queue (no — schema has no guess-vocabulary
scoping, so a live worker can't distinguish a certified row from a
copied-forward one; see Part 3).

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
- **Recertification sweep implementation**: `verify_erd_cache.py` is the
  skeleton but must be extended — unmodified it silently skips candidates
  with uncached sub-branches and reads tainted children as if exact, so it
  audits rather than recertifies. Decided design: inline-solve missing
  sub-branches at unconstrained budget (committed extension, written at
  Phase 3 time). Measured cost (2026-07-19/20, idle machine, 432 sampled
  branches across two passes): **3.0–8.7 days on 6 workers, ~5.4 days
  central estimate** — affordable outright, so the read-only-certified
  fallback is no longer the live design (contingency only). Must run
  standalone, never through the normal swarm/queue: `branch_best_by_policy`
  has no guess-vocabulary scoping, so `_cache_reuse` cannot tell a
  recertified row from a copied-forward one — only explicit
  delete-then-resolve forces recomputation (Part 3).
- **Old cache reused as ceiling ("hint") for the sweep — measured, not
  worth it**: pricing the old `best_guess` and passing its cost as an
  alpha-beta ceiling gave no net speedup (181 branches: 71 faster, 110
  flat-or-slower, totals within 0.2%) — the engine's built-in Σk²
  candidate ordering already reaches a near-optimal incumbent unaided. The
  old cache's real value is its row-key list and leaves-first schedule
  (guaranteeing warm children), not its stored scores.
- **Empirical corroboration of the growth-only theorem**: 432 branches
  fresh-solved under the new vocabulary, 28 (6.5%) strictly improved, zero
  regressed — no branch's new optimum was ever worse than its old stored
  value, exactly as Part 1 predicts and the signature real corruption would
  have broken.
- **Max_remaining_depth is not provably monotone** under either vocabulary
  growth (Part 1) or answer-set growth (Part 1, "ERD is not monotone") —
  the cache only stores the worst case of whichever tree wins the *ERD*
  race, not a separately minimized worst case. Two theorem statements
  corrected; ERD monotonicity and budget-feasibility monotonicity are
  unaffected and still hold.
