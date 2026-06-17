# Wordle solver — codebase instructions

## Project structure

A Wordle solver with four layers:
- **Engine** (`wordle_engine.py`): core ERD search, scoring, response simulation
- **Cache** (`cache_sqlite.py`): SQLite-backed persistence of branch results and candidate scores
- **Swarm** (`erd_swarm.py`, `erd_queue.py`, `erd_search.py`): parallel ERD precache workers
- **CLI** (`wordle.py`): interactive game interface and all user-facing commands

---

## Anchored vocabulary

These four terms have precise meanings throughout the codebase. Use them consistently and do not substitute synonyms.

| Term | Meaning |
|---|---|
| **guess** | A word actually played as a turn in the game. |
| **candidate** | A word under evaluation during search — not yet played. A candidate becomes the guess when it wins. |
| **branch** | The remaining answer words after a guess + response. Identified by a (guess, pattern) pair at each level. |
| **chunk** | A contiguous slice of a branch's ranked candidate list; the unit of work claimed by one swarm worker. |

The phase boundary between candidate and guess is explicit in the code:
```python
for i, candidate in enumerate(candidate_list):
    status, cost, md, floor = evaluate_candidate(branch_words, candidate, ...)
    if cost < best_erd:
        best_guess = candidate   # ← candidate becomes the guess here
```

---

## Naming rules

**Use full names. Do not abbreviate identifiers.**
- `max_group_size`, not `max_grp` or `grp`
- `candidate_list`, not `cand_list`
- `branch_words`, not `branch_wds`
- `entropy_gain`, not `ent`

**Acronyms and initialisms keep uniform casing in identifiers.**
An initialism is not a word, so do not title-case it when embedding it in a name.
- Class names: `ERDQueue`, not `ErdQueue`; `ScoreCache` is correct (Score and Cache are words)
- Local variables and parameters: `erd_score`, `erd_policy` (snake_case lowercases everything — that is correct and does not violate this rule)
- The rule applies at identifier boundaries: `ERDQueue` is right; `Erdqueue` and `ErdQueue` are both wrong for the same reason

**Names must include all essential context.**
The clearest violation of this rule in this codebase's history was `ScoringMethod.MINIMAX`. `MINIMAX` names an optimization strategy (minimize the maximum) but omits what is being minimized — the group size. Without that context the name is uninterpretable. The canonical name is `MAX_GROUP_SIZE`. The same principle applies everywhere: a name must be self-describing without external context.

`MINIMAX` is now only a legacy SQLite key in migration code. Never introduce it elsewhere.

**Scoring method names** — use these and no others:

| Enum | SQLite key | Short display | Long display |
|---|---|---|---|
| `MAX_GROUP_SIZE` | `max_group_size` | `max-grp` | `Worst-case group size` |
| `ENTROPY_GAIN` | `entropy_gain` | — | `Entropy gain (bits)` |
| `WEIGHTED_AVG` | `weighted_avg` | — | `Weighted avg remaining` |
| `PROB_FINISH` | `prob_finish` | — | `P(finish next turn)` |

Do not use `minimax`, `max`, `group`, `g-max`, `wt`, or other shorthands for `MAX_GROUP_SIZE`. The word "group" alone has no meaning; it could refer to count, size, or identity.

**Response partitions** are "response groups" (the sets of answer words that produce the same response pattern to a given guess). Do not call them "subgroups" — that was old vocabulary for what is now "branch."

---

## Comment style

**Describe what the code is, not how it got there.**

Do not write comments that narrate history, explain renames, or reference prior states:
```python
# Bad: was renamed from foo to bar because of X
# Bad: previously used subset_key, now branch_key
# Bad: this replaces the old approach of ...
```

Write comments that describe current behaviour and non-obvious invariants:
```python
# Good: a worker that has heartbeat within the timeout is never reclaimed
# Good: each sub-branch of size k costs >= 2 - 1/k (admissible lower bound)
```

**Migration code is the only exception.** In `_ensure_schema`, comments may name old table/column structures because a reader needs to understand what legacy databases look like. Even there, describe what the old structure *is* (so a reader can recognise it), not the sequence of decisions that led to the rename.

**Do not comment things that are obvious from the names and types.** A comment that restates what the code clearly says adds noise without value.

---

## Schema coordination (Linux + phone)

The cache (`wordle_cache.sqlite3`) is shared between Linux and the iOS app. Any schema change must:
1. Be implemented as an idempotent migration in `ScoreCache._ensure_schema`, guarded by the `schema_migrations` table
2. Deploy new code to the phone **before** syncing a migrated Linux database to it
3. Never require manual SQL — migrations run automatically on first open

The queue (`erd_queue.sqlite3`) is Linux-only; its migrations live in `ErdQueue._migrate()`.
