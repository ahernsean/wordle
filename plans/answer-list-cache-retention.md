# Answer-list correction: cache retention, recompute, and reverification plan

## Context

The 2026-07-15 Wordle answer was PSHAW — present in the all-words candidate
dictionary (`wordle.txt`) but absent from the answer list
(`NYT_wordlist.txt`). The corrected list is captured in this repository as
`NYT_wordlist_2026-07-17.txt`: **9 additions, 0 deletions** (3,199 → 3,208):

    aught  genus  krone  prate  pshaw  ravel  shlep  stagy  techy

All nine were already in `wordle.txt`, so the all-words candidate dictionary
is unchanged; only the answer list grows.

Operating decisions already made:

- **No side-by-side operation.** The phone stays on the old list — no pull
  from origin — until the uplift is entirely done. The Linux side flips
  wholesale to the corrected list when execution starts.
- **The epoch swarm is already stopped** for the duration of the uplift;
  no telemetry-epoch fencing is needed. Epoch testing resumes after
  retirement.

## Part 1 — What survives, and proofs

### Content-addressed keys freeze each row's answer set

`ScoreCache.encode_subset` keys every row by the sorted concatenation of the
branch's actual words. Growing the global answer list changes *which keys
arise* in the game tree — never the meaning of an existing key. A branch
that now contains a new word is a different `branch_key`: a cache miss,
computed fresh. `_solve_subset` partitions `branch_words` itself to form
sub-branches and never consults the global list to enumerate answers, so
the subproblem a row describes is exactly its key. The only input that can
vary for a fixed key is the guess vocabulary its policy draws from.

### Theorem: guess-universe growth can only lower ERD (branch fixed)

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
over the subset. Apply with the mean (ERD), the max
(max_remaining_depth), and existence within a budget (feasibility). ∎

**Loss corollary.** A loss proof over V ("no strategy over V solves B
within b") says nothing about V′ ⊋ V: the superset may contain a winning
tree. Losses in a grown-vocabulary namespace are therefore invalidated —
in the favorable direction (previously-lost branches may now be winnable).
The reverse direction is valid: a loss over V′ implies a loss over every
V ⊆ V′, since any winning tree over V would be a winning tree over V′.

### Disposition by policy namespace

| Policy | Guess vocabulary | Effect of the 9 additions | Disposition |
|---|---|---|---|
| `erd_words_unfiltered` (`ERD_ALL`) | all-words list — unchanged (all 9 already in it) | V′ = V: rows and loss proofs remain **exact** | retain |
| `erd_answers_compliant` (`ERD_ANSWERS`) | the branch words themselves (`guesses=None` ⇒ `candidate_list = branch_words`, recursively) | pure function of `branch_key`; the global list never enters: **exact** | retain |
| `erd_answers_unfiltered` (`ERD_ANSWERS_UNFILTERED`) | full answer list — grew by 9 | ERDs become **upper bounds** (theorem); `best_guess` possibly suboptimal; loss proofs **invalid** (corollary) | rebuild (see Part 3 — old rows are reusable as bounds, not as exact entries) |
| `erd_words_compliant` (`ERD_CONSTRAINED`) | path-dependent | never persisted | n/a |

Note the retained set needs no content filtering: validity above is
content-independent (it depends only on the policy's vocabulary), and old
rows cannot contain the new words anyway.

Other tables:

- **`candidate_scores`** — a score is a function of (branch word set,
  candidate, method) only; retain all rows. Scores of the 9 words as
  candidates, and of any candidate against grown branches, are new keys
  computed on demand.
- **`branch_loss_by_policy`** — same disposition as its policy: retain
  `ERD_ALL` and `ERD_ANSWERS` losses; `erd_answers_unfiltered` losses are
  invalid but can be re-seeded (Part 3).
- **`response_decomposition`** — one byte per answer in canonical
  answer-list order, keyed by (guess, answer_list_id); old rows are
  unusable under the new list and rebuild automatically on demand.

### ERD is not monotone in the answer set (why the root must recompute)

ERD is a mean, not a sum. Adding a word that resolves in 2 guesses to a
branch of n words with ERD e yields (n·e + 2)/(n + 1) < e whenever e > 2;
adding a hard word raises the mean. Monotone nondecreasing under set
growth: max_remaining_depth, the cost sum, and the mean restricted to the
original words. Per-candidate root ERDs can therefore move both ways and
the best first guess can change — root results recompute regardless (the
full-set `branch_key` changes anyway, so this happens naturally).

### The recompute region is a union of 9 thin cones, not the tree

For any first guess, the 9 new words land in at most 9 of its nonempty
response groups; every other group is a byte-identical key with a retained
exact row. Within an affected group, the next guess isolates each new word
into one sub-group again, so the changed region under each new word is a
single narrowing cone. Total genuinely-new work: the root census (every
top-level candidate against the 3,208-word set), the 9 cones, and the
`erd_answers_unfiltered` rebuild. A full restart is not required — the
overwhelming bulk of cached work is provably-exact reuse.

## Part 2 — How retained work accelerates the recompute

Four mechanisms, in order of leverage. Throughout, G′ = G ∪ {w} denotes a
grown branch and G its retained base (strip the new words from G′);
everything generalizes to k added words.

### Mechanism 1 — Recurrence-level reuse (automatic, already implemented)

To evaluate a candidate c on G′, `_solve_subset` needs the ERD of each
response group of G′ under c. The new word w lands in exactly one group;
every other group is the identical word set as the corresponding group of
G, and its retained exact row is read instead of recursed into. So the
per-candidate cost of the "full" solve of G′ is cached lookups plus one
recursive solve of the w-containing group — the same problem one level
down, on a shrinking set, memoized across candidates once computed. A
cache-missed key near the root is in practice a solve against a warm
cache. No new code; this is how retention pays off.

### Mechanism 2 — O(1) transfer bounds from G's row (not yet implemented)

From G's cached (E = best_score, md = max_depth, loss rows), without
search:

- **Loss transfer.** A proven loss for G within budget b is a proven loss
  for G′ within b: any tree solving G′ restricts to one solving G ⊆ G′.
- **max_remaining_depth sandwich.** md(G) ≤ md(G′) ≤ md(G) + k. Lower
  bound by restriction. Upper bound by construction: run each new word
  through G's optimal-worst-case tree; where its response leaves the tree,
  extend with one node guessing it; if it rides to a leaf word a, guess a
  then the new word — cost ≤ md(G) + 1 per new word, G words unchanged.
  The construction is legal in both retained policies. Feasibility is
  decided with zero search whenever the query budget falls outside the
  sandwich's knife edge.
- **ERD envelope.** (n·E + k)/(n + k) ≤ ERD(G′) ≤ (n·E + Σ costs of new
  words in the extension tree)/(n + k) ≤ (n·E + k·(md(G) + 1))/(n + k).
  The lower bound is an admissible floor a parent can use to prune a
  candidate whose sub-branch is G′ without descending into the cone. (Note
  ERD(G′) ≥ E is NOT valid — see the non-monotonicity section.)

### Mechanism 3 — Warm-starting the candidate scan on G′

G's cached best_guess, evaluated first on G′, yields a strong incumbent
that prunes most of the remaining candidate scan. G's candidate_scores
ranking is a near-perfect evaluation order for G′: exact scores shift only
through w's group membership (e.g. max_group_size(c, G′) =
max(max_group_size(c, G), |w's group under c| + 1)), an O(1) update per
candidate given w's pattern byte.

### Mechanism 4 — Pattern-matrix splice (minor)

`response_decomposition` blobs for the new list can be spliced from the old
blobs plus 9 new bytes per guess instead of recomputed at 3,208 × ~13k.
The vectorized rebuild is fast anyway; an economy, not a necessity.

### The honest limit

None of this certifies *optimality* for G′ by itself: ERD(G′) is a min over
all candidates and a new word can change which candidate wins. The scan
always runs to completion; the mechanisms make the scan and each
evaluation cheap, not skippable.

### Recommendation

Rely on Mechanism 1 alone for the first swarm pass — it is free and it is
the bulk of the win. Implement Mechanisms 2–3 as a small "grown-branch
bootstrap" (on a cache miss for key B, strip the new words to get B₀ and
read its retained row) only if the root census or the fat upper-cone nodes
measure slow. Measure first, implement second.

## Part 3 — Rebuilding `erd_answers_unfiltered`, and how long it takes

### The vocabulary-inclusion sandwich

For any branch B, the three policies' vocabularies nest:
B ⊆ answers list ⊆ all-words list. By the Part 1 theorem applied twice:

    ERD_ALL(B)  ≤  ERD_ANSWERS_UNFILTERED(B)  ≤  ERD_ANSWERS(B)

The two outer quantities are retained **exact** rows. Wherever they are
equal — plausibly the common case, since most cached branches are small and
their optima coincide across vocabularies — the middle value is pinned with
**zero search**: write it directly. Where they differ, the solve starts
with the retained `ERD_ANSWERS` value (or the old unfiltered row, which the
theorem makes a valid and tighter upper bound) as incumbent and the
retained `ERD_ALL` value as admissible floor: a branch-and-bound run that
begins nearly closed.

Losses re-seed the same way: a retained `ERD_ALL` loss at budget b is a
valid `erd_answers_unfiltered` loss at b (loss over the largest vocabulary
implies loss over every sub-vocabulary).

### Cost model

Per-branch sweep cost scales with candidate count: an
`erd_answers_unfiltered` sweep is 3,208 candidates vs. 12,972 for
`ERD_ALL` — roughly 4× cheaper. The measured `ERD_ALL` baseline
(full_tree_plan.md, derived from the Jun 23–29 epoch-0 run) is ~900 branch
sweeps/day pre-kernel, so the pre-kernel unfiltered rate is ~3,600
sweeps/day; the NumPy kernel's 10–50× lifts that to ~36k–180k/day. Only
branches the sandwich does not pinch need sweeps at all.

    rebuild time ≈ (unfiltered rows where ERD_ALL(B) ≠ ERD_ANSWERS(B))
                   / (unfiltered sweeps per day on current engine)

Both numerator and denominator are measurable before committing:

- Numerator: join the old unfiltered namespace against the two retained
  namespaces on branch_key and count the un-pinched rows (single SQL query
  against the backup).
- Denominator: time a sample of un-pinched branches with the current
  kernel engine.

### Eager vs. lazy decision rule

Run the numerator query first. If the un-pinched population is small
(likely), eager rebuild is cheap — hours, not days — and worth doing during
the uplift while the swarm is already dedicated. If it is unexpectedly
large, seed the pinched rows and losses eagerly (zero search) and leave the
rest lazy; interactive use then pays a bounded warm-start solve per miss
instead of a cold one.

## Part 4 — Procedure

The re-tag, seeding, and recompute steps are one-time operations: scripts
run from the scratchpad, never committed.

### Phase 0 — Flip the Linux side to the corrected list

1. New list captured as `NYT_wordlist_2026-07-17.txt` (done — verified 9
   additions, 0 deletions vs. `NYT_wordlist.txt`, all 9 present in
   `wordle.txt`). The `sda/new-NYT-answers-list` branch is disposable.
2. Replace the content of `NYT_wordlist.txt` with the corrected list and
   delete the date-stamped capture (git history preserves both lists).
   Source change via PR. The phone does not pull origin until Phase 5, so
   this flip is Linux-only. Rollback until retirement: revert the commit —
   the old cache namespace is never modified.

### Phase 1 — Freeze and back up

1. Swarm workers are already stopped (epoch testing suspended for the
   uplift); confirm no live claims (`erd_search.py view` worker/queue
   reports).
2. Checkpoint and back up `wordle_cache.sqlite3` and `erd_queue.sqlite3`.
3. Check disk headroom on the Linux box: the re-tag copies rows, so
   retained tables roughly double until retirement.

### Phase 2 — Re-tag migration (retention)

1. Open the cache once with the new list so `ScoreCache._ensure_answer_list`
   registers the new `answer_list_id` (canonical form is the newline-joined
   list in file order — do not derive the hash by hand).
2. With both ids in hand, `INSERT OR IGNORE` (new-id rows, if any, win):
   - `branch_best_by_policy`, `branch_loss_by_policy`: copy old-id rows to
     the new id for `erd_words_unfiltered` and `erd_answers_compliant` only.
   - `candidate_scores`: copy all old-id rows to the new id.
3. Seed `erd_answers_unfiltered` under the new id per Part 3: sandwich-
   pinched rows written exact; `ERD_ALL` losses copied in as valid losses;
   nothing else.
4. Old-id rows are left fully intact — rollback path until retirement.

### Phase 3 — Recompute

1. Run the Part 3 numerator query; decide eager vs. lazy for the
   un-pinched `erd_answers_unfiltered` remainder.
2. Clear the queue (`erd_search.py queue clear`) — its seeds and priorities
   were derived from the old root tree.
3. Re-seed from the new 3,208-word root and start the swarm. Workers
   recompute the root census and descend the 9 new-word cones; unchanged
   sub-branches hit retained rows (Part 2, Mechanism 1).
4. If eager: enqueue the un-pinched unfiltered branches after the cones
   finish (they are lower priority than the serving namespace).
5. If cone or census throughput measures slow, implement the grown-branch
   bootstrap (Part 2, Mechanisms 2–3) before brute-forcing.

### Phase 4 — Reverification

Run in this order against the new namespace; each step gates the next.

1. **Structural reconciliation.** Row counts per (policy, answer_list_id):
   new-id counts equal old-id counts for the two retained policies plus
   whatever Phase 3 added; unfiltered counts match the seeding + rebuild
   decision.
2. **Sampled exactness check.** Random sample of re-tagged rows per
   retained policy, stratified by branch size; recompute each from scratch
   under the new list with a fresh in-memory cache; require exact equality
   of best_score and max_remaining_depth (DB column `max_depth`), and that
   the stored best_guess achieves the stored score. The retained set is
   closed under sub-branching (sub-branches of a retained branch are
   subsets of it), so samples resolve without touching new-word cones.
   Include a sample of sandwich-pinched unfiltered rows.
3. **Full ERD_ALL sweep.** `verify_erd_cache.py` re-verifies every
   `erd_words_unfiltered` entry leaves-first against the true optimum
   (swarm must be stopped, per its header). This is the strong "reverify
   ERDs" guarantee: any row the migration should not have kept would be
   caught and corrected here.
4. **Loss sweep.** `verify_erd_losses.py` over retained loss entries.
5. **Test suite.** `python -m unittest discover -s tests -t . -p 'test_*.py'`.

### Phase 5 — Phone catch-up

Only after Phase 4 passes in full: the phone pulls origin (picking up the
list flip) and receives the migrated database in the same sync, so its code
and `answer_list_id` move together. The re-tag itself is data-only — no
schema change, no `schema_migrations` entry, no phone code deploy beyond
the list file.

### Phase 6 — Retirement

After enough post-cutover use to trust the new world:

1. Delete old-`answer_list_id` rows from `branch_best_by_policy`,
   `branch_loss_by_policy`, `candidate_scores`, and
   `response_decomposition`; vacuum/checkpoint; re-sync the phone.
2. Resume epoch testing (fresh `telemetry_epoch`, steady-state claiming
   against the new world).

## Open items

- Green-light to execute Phase 0.2 (the list flip PR) and the migration.
- Eager vs. lazy for the un-pinched `erd_answers_unfiltered` remainder —
  decided by the Phase 3.1 measurement, not in advance.
- How much post-cutover use constitutes enough soak for retirement.
