# Answer-list correction: cache retention, recompute, and reverification plan

## Context

The 2026-07-15 Wordle answer was PSHAW — present in the all-words candidate
dictionary (`wordle.txt`) but absent from the answer list
(`NYT_wordlist.txt`). The corrected list is captured in this repository as
`NYT_wordlist_2026-07-17.txt`: **9 additions, 0 deletions** (3,199 → 3,208):

    aught  genus  krone  prate  pshaw  ravel  shlep  stagy  techy

All nine were already in `wordle.txt`, so the all-words candidate dictionary
is unchanged; only the answer list grows. (The list originated on branch
`sda/new-NYT-answers-list`, which is disposable now that the file is
captured here.)

This plan covers: (1) which cache entries provably survive, (2) the
transition operating mode — both lists live side by side until the
migration and recomputation are verified, (3) the migration and recompute
themselves, and (4) cutover and retirement of the old list.

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
Solving a grown branch reads retained exact rows for every response group
the new words do not occupy, so a cache-missed key near the root is in
practice a solve against a warm cache, not a cold recomputation.

## Part 2 — Transition operating mode: both lists side by side

The `answer_list_id` in every primary key — the reason retention needs a
re-tag at all — is also what makes coexistence safe: the old and new worlds
are disjoint namespaces in the same database. Nothing in the old namespace
is modified by the migration or the recompute, so the old world keeps
working, byte-identical, until cutover.

During the transition:

- **Old list stays the default.** `NYT_wordlist.txt` remains untouched;
  interactive play, the phone, and any old-world tooling continue against
  the old `answer_list_id` exactly as before.
- **New list lives at `NYT_wordlist_2026-07-17.txt`.** The migration script
  and the recompute swarm run against it explicitly. The answer-list path
  is currently a hardcoded constant in `wordle.py`, `erd_swarm.py`,
  `verify_erd_cache.py`, and `runtime_paths.py`/`erd_search.py`, so Phase 0
  adds a selection mechanism (recommendation below).
- **Epoch testing is suspended.** Swarm telemetry comparisons filter on
  `telemetry_epoch` ("a contiguous run under one claiming regime"); the
  transition workload (root census + new-word cones against a warm cache)
  is not representative of steady-state claiming and would contaminate any
  cross-epoch comparison. Suspend claim-regime testing before the freeze,
  bump a fresh epoch when the transition swarm starts (so transition
  samples are cleanly separable, never mixed into a testing epoch), and
  resume testing only after cutover, in another fresh epoch.
- **Rollback is trivial until retirement.** Reverting means pointing back
  at the old list; the old namespace was never modified.
- **Storage cost.** The re-tag copies rows, so retained tables roughly
  double until retirement deletes the old namespace. Check disk headroom
  before Phase 2 (on both Linux and the phone if the DB syncs before
  retirement).

## Part 3 — Procedure

The re-tag and recompute steps are one-time operations: scripts run from
the scratchpad, never committed. The answer-list selection mechanism is the
exception — it is a permanent source change (list corrections will happen
again) and goes through a PR.

### Phase 0 — Capture the list and add a selection mechanism

1. New list captured as `NYT_wordlist_2026-07-17.txt` (done — verified 9
   additions, 0 deletions vs. `NYT_wordlist.txt`, all 9 present in
   `wordle.txt`). The `sda/new-NYT-answers-list` branch is no longer
   needed; deleting it is the user's call.
2. Add an answer-list override so tools can be pointed at either list.
   Recommended: an environment variable (e.g. `WORDLE_ANSWER_LIST`) read in
   `runtime_paths.py`, with the four hardcoding modules taking their
   default from there; default remains `NYT_wordlist.txt` so behavior is
   unchanged when unset. Alternative (no code change): run the transition
   swarm from a second checkout whose `NYT_wordlist.txt` is the new list,
   sharing the cache via explicit `--cache`; rejected as the default
   because it is easy to get wrong silently and leaves nothing reusable
   for the next list correction.

### Phase 1 — Freeze and back up

1. Suspend epoch testing; note the current epoch number.
2. Stop all swarm workers; confirm no live claims
   (`erd_search.py view` worker/queue reports).
3. Checkpoint and back up `wordle_cache.sqlite3` and `erd_queue.sqlite3`.

### Phase 2 — Re-tag migration (retention)

1. Open the cache once with the new list so `ScoreCache._ensure_answer_list`
   registers the new `answer_list_id` (canonical form is the newline-joined
   list in file order — do not derive the hash by hand).
2. With both ids in hand, `INSERT OR IGNORE` (new-id rows, if any, win):
   - `branch_best_by_policy`, `branch_loss_by_policy`: copy old-id rows to
     the new id for `erd_words_unfiltered` and `erd_answers_compliant` only.
   - `candidate_scores`: copy all old-id rows to the new id.
3. Copy nothing for `erd_answers_unfiltered`.
4. Old-id rows are left fully intact — they are the live old world until
   cutover and the rollback path until retirement.

### Phase 3 — Recompute (update the cache)

1. Clear the queue (`erd_search.py queue clear`) — its seeds and priorities
   were derived from the old root tree. (The queue is Linux-only swarm
   state, not part of the old world's serving path.)
2. Bump a fresh `telemetry_epoch` for the transition run.
3. Re-seed from the new 3,208-word root and start the swarm against the
   new list. Workers recompute the root census and descend the 9 new-word
   cones; unchanged sub-branches hit retained rows.
4. `erd_answers_unfiltered` repopulates lazily on use (or enqueue
   explicitly if that namespace is wanted precomputed).
5. `response_decomposition` blobs rebuild on demand for the new id.

### Phase 4 — Reverification

Run in this order against the new list/namespace; each step gates the next.

1. **Structural reconciliation.** Row counts per (policy, answer_list_id):
   new-id counts equal old-id counts for the two retained policies plus
   whatever Phase 3 has added; zero `erd_answers_unfiltered` rows under the
   new id unless Phase 3.4 ran.
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

### Phase 5 — Cutover

Only after Phase 4 passes in full.

1. Promote the new list to the default: `NYT_wordlist_2026-07-17.txt`
   becomes `NYT_wordlist.txt` (the old file is renamed aside, e.g.
   `NYT_wordlist_pre-2026-07.txt`, kept until retirement as the rollback
   target). Source change via PR.
2. Sync the migrated database to the phone. The re-tag itself is data-only
   (no schema change, no `schema_migrations` entry), but the phone must
   receive the new list with the same sync so its `answer_list_id` matches.
3. Soak: normal interactive and swarm use against the new world, with the
   old namespace still present as rollback.

### Phase 6 — Retirement

Only after the soak period gives confidence that the new world is safe.

1. Delete old-`answer_list_id` rows from `branch_best_by_policy`,
   `branch_loss_by_policy`, `candidate_scores`, and
   `response_decomposition`; vacuum/checkpoint; re-sync the phone.
2. Remove the retired list file from the repository (PR).
3. Resume epoch testing in a fresh `telemetry_epoch` — steady-state
   claiming against the new world, uncontaminated by transition samples.

## Open items

- Approve the answer-list override mechanism (Phase 0.2 recommendation:
  environment variable in `runtime_paths.py`) before any code is written.
- Whether `erd_answers_unfiltered` is worth precomputing (Phase 3.4) or can
  stay lazy.
- Whether to run Phase 4.3 as a full sweep immediately or start with the
  sampled check and schedule the sweep alongside normal swarm operation.
- Length of the Phase 5 soak before retirement.
