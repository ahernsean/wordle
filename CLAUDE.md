# Wordle solver — codebase instructions

## Project structure

A Wordle solver with five layers:
- **Engine** (`wordle_engine.py`): core ERD search, scoring, response simulation
- **Kernel** (`pattern_matrix.py`): NumPy response-pattern matrix and vectorized candidate statistics; the engine's sole NumPy import point. NumPy is a hard requirement on every target; the pure-Python engine paths remain permanently as the reference implementation (they never call NumPy), selected by passing `pattern_matrix=None`
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

The phase boundary between candidate and guess is explicit in the code:
```python
for i, candidate in enumerate(candidate_list):
    status, cost, md, floor = evaluate_candidate(branch_words, candidate, ...)
    if cost < best_erd:
        best_guess = candidate   # ← candidate becomes the guess here
```

### Depth and budget

The guess axis is **remaining depth** (guesses still needed under optimal play),
viewed four ways. A bare `depth` is forbidden — always qualify it.

| Term | Meaning |
|---|---|
| **guess_depth** | Guesses already *played* to reach a branch — the number of guesses on its spine. Root (no guess) = 0; a one-guess seed = 1. The source of truth. |
| **budget** | *Allowed* remaining depth (the cap). The only quantity the ERD recurrence reads; each level spends one. |
| **ERD** | *Expected* remaining depth — the mean line length; the objective (`cost` / `best_erd`). |
| **max_remaining_depth** | *Maximum* remaining depth — worst-case line length. Feasibility gate (solvable iff `≤ budget`) and cache-reuse key (reusable at any `budget ≥ max_remaining_depth`). |

`GAME_GUESSES = 6` is the root's budget (zero guesses). Invariant at every node:
`budget + guess_depth = GAME_GUESSES`, so `guess_depth = GAME_GUESSES − budget`.
ERD and `max_remaining_depth` are the mean and max of the same per-answer distribution.

There is no "promotion depth" or "recursion depth" — both were uninformative
`depth` aliases of `budget`/`guess_depth` and were removed. Human-queued vs
swarm-spawned is decided by `pending_branches` membership, not a depth.

---

## Naming rules

**Use full names. Do not abbreviate identifiers.**
- `max_group_size`, not `max_grp` or `grp`
- `candidate_list`, not `cand_list`
- `branch_words`, not `branch_wds`
- `entropy_gain`, not `ent`

**Acronyms and initialisms keep uniform casing in identifiers.**
Neither an acronym (NASA, POSIX) nor an initialism (ERD, SQL, HTML) is an ordinary word. When embedding one in a PascalCase or UPPER_CASE identifier, preserve its casing as a unit — do not title-case it.
- `ERDQueue`, `SQLCache`, `HTMLParser`, `MPIWorker`, `NASAFeed`, `POSIXPath` — correct
- `ErdQueue`, `SqlCache`, `HtmlParser`, `MpiWorker`, `NasaFeed`, `PosixPath` — wrong: these misrepresent the acronym/initialism as an ordinary word
- `ScoreCache`, `BranchWorker` — correct: Score, Cache, Branch, Worker are ordinary words, not acronyms
- snake_case variables and parameters are exempt: `erd_policy`, `sql_query`, `posix_path` lowercase everything uniformly, which does not misrepresent anything

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

## Dangerous operations — always ask first

Never perform any of the following without explicit instruction from the user:

- **Merging a PR** — "CI is green" or "looks good" is not a merge instruction.
- **Deleting a branch**
- **Rewriting git history** — including rebase, amend, or any operation that changes existing commits.
- **Force push**

## Never commit directly to main — no exceptions

Committing to `main` is forbidden, always. This rule has no override: not
"just do it", not "no need for a PR", not any other phrasing. Such phrasing
waives ceremony, not the branch — the work still goes on a branch with a PR
for Sean to merge. If an instruction genuinely seems to require a direct
commit to `main`, stop and ask; do not infer permission.

Persist source changes; do not persist one-time operations. A migration or
data movement that will only ever run once on this machine is executed
here (from the scratchpad) and never committed. Only code that must keep
working in the future belongs in the repository.

These actions are hard or impossible to reverse and affect shared state. Always confirm before proceeding, no matter how obvious it seems.

---

## Pull request style

PR descriptions need a **Summary** section only. Do not include a "Test plan" section.

PR descriptions describe *changes* — what is added, removed, or fixed, and why. This is distinct from the comment style rule ("describe what the code is, not how it got there"), which applies to inline code comments. A PR description is inherently a description of a diff; talking about what changed is the point.

Always update the PR title and description when pushing new commits.

---

## Before committing and pushing

Always run the test suite before committing and pushing:

```
python -m unittest discover -s . -p 'test_*.py'
```

Commits with failing tests must not be pushed.

---

## Respond to what's actually being asked

Before acting, classify the request:
- **"What should I do / what's your recommendation / what are your thoughts?"** → reason through the tradeoffs and give a recommendation in words. Do not write or push code. Stop there and wait for a decision.
- **"Implement X" / explicit instruction to make a change** → code is appropriate.

A bug report or "this is wrong" is not, by itself, authorization to start editing files. Diagnose and propose a fix in conversation first, unless the user's phrasing already asks for the fix to be made.

When in doubt about which mode applies, ask, or default to discussion rather than to code — reverting unwanted code is more disruptive than a follow-up question.

**The `stop-hook-git-check.sh` Stop hook is advisory, not a directive.** It fires on every turn end and pushes toward committing/pushing whenever the tree is dirty, signing looks wrong, or commits are unpushed — it has no idea whether a change was meant for discussion or was actually approved. Neither the user nor the assistant edits this hook, so the only way to keep it from overriding the discussion-first norm above is judgment when it fires: read its complaint, then act on what the user actually wants in the conversation, not on the hook's say-so by itself. Committing/pushing unapproved work just to satisfy the hook is the wrong move — it's fine to leave changes uncommitted and explain the tension to the user instead.

---

## Schema coordination (Linux + phone)

The cache (`wordle_cache.sqlite3`) is shared between Linux and the iOS app. Any schema change must:
1. Be implemented as an idempotent migration in `ScoreCache._ensure_schema`, guarded by the `schema_migrations` table
2. Deploy new code to the phone **before** syncing a migrated Linux database to it
3. Never require manual SQL — migrations run automatically on first open

The queue (`erd_queue.sqlite3`) is Linux-only; its migrations live in `ERDQueue._migrate()`.

---

## Session token usage

To check remaining session token budget and reset time, run:

```
claude -p /usage
```

There is no in-session tool that exposes this — it is the only way to see the limits.
