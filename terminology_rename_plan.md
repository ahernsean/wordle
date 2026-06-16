# Terminology rename plan (deferred)

Goal: make the *engine* and *cache* internals speak the same vocabulary we
settled on at the swarm layer, so a reader never has to translate.

Anchored vocabulary:

- **guess** — a word played as a turn (any of the ~12,972).
- **branch** — a position to solve: the answer words remaining after a
  guess+response. Identified at each level by a (guess, pattern) pair.
- **candidate** — a guess being evaluated against a branch.
- **chunk** — a contiguous slice of a branch's ranked candidate list.

## Why this is deferred (not done now)

1. The persisted cache schema is load-bearing on a multi-GB file **shared with
   the phone**. Renaming `subgroup_best_by_policy` / `subset_key` means a
   migration that must run on the Linux cache *and* every phone copy — the two
   must be on the new code together or a merge will split into stale + renamed
   tables.
2. `wordle_engine.min_expected_guesses` uses `remaining` / "subgroup" /
   `best_word` pervasively, and is the most-tested code in the repo. Renaming
   it mid-feature would churn the engine and the 499-test suite at the same
   time as the swarm work.
3. The swarm layer already reads cleanly in the new vocabulary; the internal
   names are below the waterline and don't block anything.

## Mapping (engine + cache)

| current | rename to |
|---|---|
| "subgroup" (concept, comments) | branch |
| `subset_key` (column / var) | `branch_key` |
| `subgroup_best_by_policy` (table) | `branch_best_by_policy` |
| `best_word` (column) | `best_guess` |
| `remaining` (engine param) | `branch_words` |
| `word_scores` (table) | `candidate_scores` |
| `guesses=` / `guess_list` (engine) | keep (`guess` already right) |

Leave the **ERD policy strings** (`erd_words_unfiltered`, …) alone — those are
data values, already migrated once, and not identifiers.

## Approach when we do it

1. **Cache (cache_sqlite.py):** SQLite `ALTER TABLE … RENAME` and
   `RENAME COLUMN` are metadata-only (cheap even on a big file). Add the rename
   steps to the existing `_ensure_schema` migration chain (it already does
   exactly this for two prior renames), reading old names forward so an
   un-migrated DB still opens.
2. **Engine (wordle_engine.py):** pure code rename — no migration. Update
   `min_expected_guesses`, `evaluate_guess`, `verify_erd_cache`, and callers.
3. **Swarm layer (erd_queue/erd_swarm/erd_search):** flip the two retained
   shims (`subset_key` → `branch_key`, `best_word` → `best_guess`) so the whole
   stack is uniform.
4. **Coordinate the rollout:** Linux and phone on the new code before the next
   export/merge, since the migration touches a shared schema. Migrations are
   idempotent and run on open, so each side upgrades its own copy on first
   start with the new code.

Do it as one dedicated PR with the full test suite green (499 tests across 14
files as of Jun 2026), not folded into a feature change.

---

## Deferred: opening ERD progress N/M at startup

The startup message shows "Opening ERD: in progress" when the root position
isn't solved yet.  The user wants N/M (candidates evaluated / total) instead.

The problem: checking coverage candidate-by-candidate requires O(covered ×
subgroups_per_word) SQLite reads, which is fast when coverage is 0% (breaks on
first miss) but gets **slower** as coverage grows (more subgroups to verify per
covered word).  In testing, 0% coverage takes ~3s and 50% coverage would take
~9s — worse as the solver makes progress.

**Fix requires a metadata counter** (incrementing a `cache_stats` row on each
ERD write in `ScoreCache.write()`) so the count can be read in O(1) at startup.
The counter would track `(policy, universe_id) → subgroup_count`.  Linking this
count back to "candidates fully covered" still requires knowing the subgroup
structure, but tracking the raw subgroup count is a good starting point.

---

## Collision: "universe" means two different things

There are two unrelated concepts both called "universe":

**`universe_id` / `universe` table (cache_sqlite.py)** — a SHA-256 fingerprint
of the *answer word list* (the 3,200 NYT words).  It namespaces cache rows so
that a different answer list produces a clean, non-conflicting namespace in the
same database file.  This can change — if NYT updates their word list, the
fingerprint changes and all prior cache entries become unreachable.

**`GuessUniverse` enum (wordle_engine.py)** — which words are eligible as
*guesses*: the full 12,972-word dictionary (`ALL_WORDS`) or only the 3,200
answer words (`ALL_ANSWERS`).  This is a per-session strategy toggle (the `c`
command), completely independent of the answer list.

### Rename target

| current | rename to | where |
|---|---|---|
| `universe_id` (column/var) | `answer_list_id` | cache_sqlite.py, schema |
| `universe` (table) | `answer_list` | cache_sqlite.py, schema |
| `_ensure_universe` (method) | `_ensure_answer_list` | cache_sqlite.py |

`GuessUniverse` stays — it correctly describes what it is (the universe of
valid guesses).

### Coordination note

`universe_id` is in a shared SQLite schema (Linux + phone).  The rename needs
the same migration-chain treatment as the other table/column renames: add
`ALTER TABLE universe RENAME TO answer_list` and update the column reference in
`_ensure_schema`, guarded by the `schema_migrations` table so it runs once.
