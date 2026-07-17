# Answer-list correction: cache retention, recompute, and reverification plan

## Context

The 2026-07-15 Wordle answer was PSHAW — present in the all-words candidate
dictionary (`wordle.txt`) but absent from the answer list
(`NYT_wordlist.txt`). The corrected list is on branch
`sda/new-NYT-answers-list`: **9 additions, 0 deletions** (3,199 → 3,208):

    aught  genus  krone  prate  pshaw  ravel  shlep  stagy  techy

All nine were already in `wordle.txt`, so the all-words candidate dictionary
is unchanged; only the answer list grows.

This plan covers: (1) which cache entries provably survive, (2) the
migration that retains them, (3) the recompute of the invalidated region,
and (4) reverification before the migrated cache is trusted or synced.

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

### Disposition by policy namespace

| Policy | Guess vocabulary | Effect of the 9 additions | Disposition |
|---|---|---|---|
| `erd_words_unfiltered` (`ERD_ALL`) | all-words list — unchanged (all 9 already in it) | V′ = V: rows and loss proofs remain **exact** | retain |
| `erd_answers_compliant` (`ERD_ANSWERS`) | the branch words themselves (`guesses=None` ⇒ `candidate_list = branch_words`, recursively) | pure function of `branch_key`; the global list never enters: **exact** | retain |
| `erd_answers_unfiltered` (`ERD_ANSWERS_UNFILTERED`) | full answer list — grew by 9 | ERDs become **upper bounds** (theorem); `best_guess` possibly suboptimal; loss proofs **invalid** (corollary) | discard; recompute on demand |
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
  `ERD_ALL` and `ERD_ANSWERS` losses, discard `ERD_ANSWERS_UNFILTERED`.
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
discarded `erd_answers_unfiltered` namespace. A full restart is not
required — the overwhelming bulk of cached work is provably-exact reuse.

## Part 2 — Procedure

The re-tag and recompute steps are one-time operations: scripts run from
the scratchpad, never committed (only the plan is).

### Phase 0 — Land the corrected list

1. Merge `sda/new-NYT-answers-list` (via PR, per repo rules).
2. Sanity-check in CI/local: 3,208 lines, 9 additions vs. old list, no
   deletions, all additions present in `wordle.txt`.

### Phase 1 — Freeze and back up

1. Stop all swarm workers; confirm no live claims
   (`erd_search.py view` worker/queue reports).
2. Back up `wordle_cache.sqlite3` (and its WAL/shm via a checkpoint first)
   and `erd_queue.sqlite3`.

### Phase 2 — Re-tag migration (retention)

1. Open the cache once with the new list so `ScoreCache._ensure_answer_list`
   registers the new `answer_list_id` (canonical form is the newline-joined
   list in file order — do not derive the hash by hand).
2. With both ids in hand, `INSERT OR IGNORE` (new-id rows, if any, win):
   - `branch_best_by_policy`, `branch_loss_by_policy`: copy old-id rows to
     the new id for `erd_words_unfiltered` and `erd_answers_compliant` only.
   - `candidate_scores`: copy all old-id rows to the new id.
3. Copy nothing for `erd_answers_unfiltered`.
4. Leave old-id rows in place until Phase 4 passes; delete only for space.

### Phase 3 — Recompute (update the cache)

1. Clear the queue (`erd_search.py queue clear`) — its seeds and priorities
   were derived from the old root tree.
2. Re-seed from the new 3,208-word root and restart the swarm. Workers
   recompute the root census and descend the 9 new-word cones; unchanged
   sub-branches hit retained rows.
3. `erd_answers_unfiltered` repopulates lazily on use (or enqueue
   explicitly if that namespace is wanted precomputed).
4. `response_decomposition` blobs rebuild on demand for the new id.

### Phase 4 — Reverification

Run in this order; each step gates the next.

1. **Structural reconciliation.** Row counts per (policy, answer_list_id):
   new-id counts equal old-id counts for the two retained policies plus
   whatever Phase 3 has added; zero `erd_answers_unfiltered` rows under the
   new id unless Phase 3.3 ran.
2. **Sampled exactness check.** Random sample of re-tagged rows per
   retained policy, stratified by branch size; recompute each from scratch
   under the new list with a fresh in-memory cache; require exact equality
   of best_score and max_remaining_depth (DB column `max_depth`), and that
   the stored best_guess achieves the stored score. The retained set is
   closed under sub-branching (sub-branches of a retained branch are
   subsets of it), so samples resolve without touching new-word cones.
3. **Full ERD_ALL sweep.** `verify_erd_cache.py` re-verifies every
   `erd_words_unfiltered` entry leaves-first against the true optimum
   (swarm must be stopped, per its header). This is the strong "reverify
   ERDs" guarantee: any row the migration should not have kept would be
   caught and corrected here.
4. **Loss sweep.** `verify_erd_losses.py` over retained loss entries.
5. **Test suite.** `python -m unittest discover -s tests -t . -p 'test_*.py'`.

### Phase 5 — Phone sync

The re-tag is data-only — no schema change, no `schema_migrations` entry,
no phone code deploy required. Sync the migrated database to the phone only
after Phase 4 passes. (The phone must already run code that reads
`NYT_wordlist.txt` from the synced state, or receive the new list with the
same sync.)

## Open items

- Timing of the `sda/new-NYT-answers-list` merge (Phase 0).
- Whether `erd_answers_unfiltered` is worth precomputing (Phase 3.3) or can
  stay lazy.
- Whether to run Phase 4.3 as a full sweep immediately or start with the
  sampled check and schedule the sweep alongside normal swarm operation.
