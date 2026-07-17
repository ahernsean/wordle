# Answer-list correction: cache retention plan

## Context

The 2026-07-15 Wordle answer was PSHAW — present in the all-words candidate
dictionary but absent from the answer list. The answer list is therefore
inaccurate and must be corrected (at least one addition; removals possible).
The corrected list is still being determined.

Question this plan answers: what, if anything, in `wordle_cache.sqlite3` must
be discarded when the answer list changes, and how to retain the rest.

## Why most of the cache survives: content-addressed branch keys

`ScoreCache.encode_subset` keys every row by the sorted concatenation of the
branch's actual words. A cached row is a statement about that exact word set —
the answer set of the subproblem is frozen into the key. Consequences:

- No existing row's answer set grows when the global list grows. A branch
  that now contains a new word is a different `branch_key`: a cache miss,
  computed fresh. Stale values are never consulted for grown branches.
- No old row can contain a newly added word at all (it was never an answer),
  so every old row remains a true statement about its word set.
- Those same word sets still arise in the new game tree — every path whose
  responses are inconsistent with the new words filters to the identical set.
  Recomputation is confined to the root and the branches that actually
  contain new words (one response group per guess at each level).

The only quantity that can differ between old and new worlds for an existing
row is the **guess vocabulary** its policy draws from.

## Validity by policy namespace

| Policy | Guess vocabulary | Effect of list change | Disposition |
|---|---|---|---|
| `erd_words_unfiltered` (`ERD_ALL`) | all-words list (~13k, unchanged) | none — rows and loss proofs exact | retain |
| `erd_answers_compliant` (`ERD_ANSWERS`) | the branch words themselves (`_solve_subset`: `guesses=None` ⇒ `candidate_list = branch_words`, recursively) | none — the recursion is a pure function of `branch_key`; the global list never enters | retain |
| `erd_answers_unfiltered` (`ERD_ANSWERS_UNFILTERED`) | full answer-shaped list (grew) | ERDs become upper bounds; `best_guess` possibly suboptimal; loss proofs invalid (a new guess word could rescue a proven loss) | discard (engine has no bound-aware reuse) |
| `erd_words_compliant` (`ERD_CONSTRAINED`) | path-dependent | never persisted | n/a |

Other tables:

- **`candidate_scores`** — a score is a function of (branch word set,
  candidate, method) only; the guess vocabulary determines which candidates
  get scored, not the values. Retain all rows. New words' scores are simply
  absent and computed on demand.
- **`branch_loss_by_policy`** — same disposition as its policy above:
  retain for `ERD_ALL` and `ERD_ANSWERS`, discard for
  `ERD_ANSWERS_UNFILTERED`.
- **`response_decomposition`** — one byte per answer in canonical answer-list
  order, keyed by (guess, answer_list_id). Old rows are unusable under the
  new list and rebuild automatically on demand. Delete old-id rows only to
  reclaim space.
- **Removals**, if the corrected list also drops words: rows whose branch
  contains a removed word remain true statements about sets that no longer
  arise — harmless dead weight. Optionally prune (decode `branch_key` in
  5-byte slices and test membership); correctness does not require it.

## Why the root must be recomputed regardless (ERD is not monotone)

ERD is a mean, not a sum, so growing a branch can move it either way. Adding
a word that resolves in 2 guesses to a branch of n words with ERD e yields
(n·e + 2)/(n + 1) < e whenever e > 2. Monotone nondecreasing under set
growth: `max_remaining_depth`, the cost sum, and the mean restricted to the
original words. Since per-candidate ERDs at the root can move in both
directions, the previously best first guess can be dethroned; root-level
results and full-set `candidate_scores` rows recompute fresh (the full-set
`branch_key` changes anyway). That recomputation reuses every retained
sub-branch row.

## The blocker: `answer_list_id` in every primary key

`answer_list_id` is a SHA-256 of the canonical answer list and participates
in the primary key of `branch_best_by_policy`, `branch_loss_by_policy`, and
`candidate_scores`. The corrected list yields a new id, so every read misses
— including the rows shown above to be exact. Retention requires a one-time
re-tag of valid rows to the new id.

## Migration procedure (one-time, run from scratchpad — never committed)

Prerequisite: the corrected answer list is confirmed and checked in.

1. Stop swarm workers; back up `wordle_cache.sqlite3`.
2. Open the cache once with the new list so `ScoreCache` registers the new
   `answer_list_id` in `answer_list` via its own canonicalization (do not
   re-derive the hash by hand).
3. Re-tag, using `INSERT OR IGNORE` so rows already computed under the new
   id take precedence:
   - `branch_best_by_policy` and `branch_loss_by_policy`: copy old-id rows
     to the new id for policies `erd_words_unfiltered` and
     `erd_answers_compliant` only.
   - `candidate_scores`: copy all old-id rows to the new id.
4. Do not copy `erd_answers_unfiltered` rows or losses.
5. Leave old-id rows in place until verification passes; delete afterwards
   only for space.

Queue (`erd_queue.sqlite3`, Linux-only): pending and active branch keys were
seeded from the old root tree. Retained branches are still solvable and
useful, but priorities and seeds derived from the old root may be stale.
Simplest sound course: clear and re-seed the queue after the new list lands
(`erd_search.py queue clear` / re-seed). Decide at migration time based on
how much unfinished work the queue holds.

## Verification

1. Row counts per (policy, answer_list_id) before and after re-tag match the
   expected copy sets.
2. Sample re-tagged branches across sizes for each retained policy;
   recompute each from scratch under the new list with a fresh in-memory
   cache; compare ERD, `best_guess` score, and `max_remaining_depth`
   (DB column `max_depth`) exactly.
3. Run `verify_erd_cache.py` against the migrated database.
4. Run the test suite.

## Phone coordination

The re-tag is a data-only operation — no schema change, no
`schema_migrations` entry, no phone code deploy required. Sync the migrated
database to the phone only after verification passes.

## Open items

- Confirm the corrected answer list (in progress).
- Whether the correction includes removals (affects optional pruning only).
- Queue: clear-and-reseed vs. retain, judged by pending work at migration time.
