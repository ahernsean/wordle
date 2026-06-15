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
   it mid-feature would churn the engine and the 229-test suite at the same
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

Do it as one dedicated PR with the full test suite green, not folded into a
feature change.
