# Wordle solver — codebase instructions

## Project structure

A Wordle solver with six layers:
- **Engine** (`wordle_engine.py`): core ERD search, scoring, response simulation
- **Kernel** (`pattern_matrix.py`): NumPy response-pattern matrix and vectorized candidate statistics; the engine's sole NumPy import point. NumPy is a hard requirement on every target; the pure-Python engine paths remain permanently as the reference implementation (they never call NumPy), selected by passing `pattern_matrix=None`
- **Cache** (`cache_sqlite.py`): SQLite-backed persistence of branch results and candidate scores
- **Hints** (`hint_cache.py`): a second, read-only cache consulted for candidate order only
- **Swarm** (`erd_swarm.py`, `erd_queue.py`, `erd_search.py`): parallel ERD precache workers
- **CLI** (`wordle.py`): interactive game interface and all user-facing commands

### Swarm reporting and queue operations

All read-only swarm inspection uses `erd_search.py view`. It provides text,
JSON, JSON Lines, optional watch, semantic word/branch selection, queue,
worker, cache, tree, hotspot, and root-progress reports. Check `view --help`
and use `SWARM.md` for examples.

`view --root-progress WORD` answers "how far along is this root, and what is
holding it up": per-response-group branch/node/time totals, which groups are
untouched, request time versus work-start time, and a completion estimate
drawn from observed claim throughput. It deliberately does not estimate
unstarted groups — see SWARM.md for why the cost model cannot rank them.

Use `erd_search.py epoch show` to inspect the active telemetry epoch. See
SWARM.md for the stopped-swarm procedure required to change it.

Queue mutations remain grouped under `erd_search.py queue`: `add`, `remove`,
`clear`, `priority`, `opener-priority`, `reset-stale`, and
`reconcile-orphaned-ownership`. The `queue` group has no read-only dashboard
commands.

### Priority ladders, and the fan-out they prevent

**Openers tied at one priority all become eligible at once, and the swarm
spreads across every one of them.** When a worker blocks on a dependency,
`_help_other_branch` prefers *starting an opener with no branch open yet* over
*joining a branch of the opener it is already on*: its first loop skips any
opener that already holds a **joinable** open branch and promotes a fresh one.
The guard that would stop this is a strict `priority < fallback_priority`
comparison, and ties are let through deliberately (issue #214 — a worker
blocked on an opener's only branch must be able to widen that opener). So a
flat batch of N openers becomes N simultaneously-active openers, one recruited
per blocking event, each left with a single open branch and the rest of its
response groups still pending. The signature in `view --queue` is many active
openers each showing `open=1` with dozens pending behind them.

Joinable means **open and unoccupied**. An opener whose open branches all have
workers is not covered by the join loop, so it promotes another of its own
pending branches instead of recruiting a fresh opener — the widening that keeps
one opener's response groups moving rather than starting the next opener. See
one worker per branch below.

`queue add` therefore lays openers on a **descending priority ladder**,
`--priority-step` apart (default 5), which breaks every tie so the guard fires
and the swarm finishes one opener before starting the next.

`queue add` **appends**: with no `--priority`, a batch descends from just below
the lowest priority the queue still owes work, so new work never preempts
queued work. Ladders run downward from the top of the range — that is what
leaves room beneath each batch for the next to append into. Only work still
*owed* holds the ceiling down, so a drained queue returns the full range.
`--priority` names the *last* opener's rung and is the deliberate way to jump
the queue; a request the range cannot hold is refused rather than quietly
seated lower.

`SOURCE_PRIORITY_MIN`/`SOURCE_PRIORITY_MAX` bound requested priorities at
0–999,999 — one rung per opener with room to spare, since the full candidate
list at step 5 occupies 75,000 values. Priorities at or above
`LEGACY_PROMOTED_PRIORITY_MIN` (1,000,000) are the legacy promoted band and
never preempt requested work.

**The scan cost is linear in queued openers.** Both work-selection paths walk
every unfinished request on each claim, and `_help_other_branch` additionally
issues one query per request. Measured on rocky: 0.6 ms per claim at 64
openers, 157 ms at 15,000. Invisible at today's batch sizes and fatal at the
scale of a full sweep — see the open issue before queueing thousands.

### One worker per branch

**Two workers on one branch is a loss at production vocabulary, not a gain.**
They evaluate candidates against a stale `best_erd` ceiling and explore subtrees
a sequential best-first search prunes. Measured on a 296-word production branch
at the full candidate list: six workers took 955.4s against 584.9s for one, on
1.79x the nodes. Spreading the same six one-per-branch drained six real branches
2.29x faster than ganging them.

The effect is **regime-dependent, and candidate count is the lever** — room to
run ahead of the ceiling. Below ~3,000 candidates concentration genuinely wins
(1.82x at 230); above it, it loses (0.79x at 3,062). Production runs at 14,855,
so a fixture built at test scale will show the opposite sign. That is why
`TestSingleBranchDoesNotConcentrateWorkers` fixes `_BASE_CANDIDATES` above the
crossover, and why it asserts structurally rather than on the clock.

Work selection therefore takes an unoccupied branch first, promotes one of the
current opener's pending branches next, and only pairs a second worker onto an
occupied branch when nothing anywhere is free. Never a third.

**Occupancy is unfinished claims, never heartbeats.** A `done = 0`
`candidate_claims` row *is* a worker on that branch: it is written by the same
transaction that hands out the bundle, so a branch reads as taken the instant it
is taken. Heartbeats are reporting state — they lag by design (a worker writes
one only once it is already evaluating), so at startup every worker would read
every branch as free. `worker_counts_by_branch` is for `status` and the reports;
`claim_holders_by_branch` is for scheduling.

**The cap is enforced in the claim transaction; the filter is an optimization.**
`claim_next_bundle` re-counts holders inside its `BEGIN IMMEDIATE`, which is what
makes the cap hold when several workers pick the same branch before any has
claimed it. Filtering the candidate list first changes no outcome — it saves
opening a write transaction against each occupied branch. Do not remove the
transaction check on the grounds that selection already filtered.

**Occupancy needs no liveness test of its own.** A crashed worker's `done = 0`
rows are freed by `reclaim_stale_claims`, by `reclaim_claims_of_worker` on a
supervised respawn, and by `recover_active_branches` at restart — so a branch
nobody is working drains to zero holders through paths that already exist. A
worker that simply finished its bundle leaves no unfinished claims at all, which
is how a partly-solved branch resumes: sequential hand-off over a branch's life
is normal, and only *concurrent* occupancy is capped.

### A branch's exact results are keyed by scope

A branch has two kinds of exact result and they are separate facts: the
**unrestricted optimum** in `branch_best_by_policy`, and one **budget-specific**
result per budget in `branch_best_by_policy_and_budget` (keyed by
`(branch_key, policy, answer_list_id, solve_budget)`).  Both can be right and
differ, so neither table's write may displace the other's.

Read through `ScoreCache.read_for_budget`, never `read_with_depth`, when a
budget is in play: the unrestricted result wins when its `max_depth` fits the
budget, and only otherwise does the result solved at *exactly* that budget
apply.  `wordle_engine._cache_reuse` stays the single statement of the rule;
`read_for_budget` only selects which result to put to it.  The memory mirror is
keyed by scope for the same reason the tables are.

**A second exact result at a scope already stored does not replace it** — it
is returned, and the caller adopts it before folding anything.  `max_depth` is
ancestor-visible, so a solver that kept its own worst case would hand its
parent a value the stored child does not support: the same inconsistent
ancestry, reached without an overwrite.  `_solve_subset` therefore takes
`write`'s return value rather than its own locals.  A second result that
disagrees on *cost* cannot be reconciled by adoption and raises
`CacheWriteConflict`.

**Sameness is `exact_results_agree`: equal cost AND equal `max_depth`.**
`import_cache` states the same rule in SQL to compare whole tables; a merge
cannot adopt, because the incoming ancestors are already folded, so it refuses
instead.

**Creating the row is the check.**  `write` inserts with `ON CONFLICT DO
NOTHING` and reconciles only when the insert finds the scope taken; a read
followed by an insert leaves a window two workers both pass through, and the
loser's write would displace a result an ancestor had folded with neither
noticing.  The reconciliation reads through `_read_stored_row`, never the
session mirror, which can predate the other writer.  Anything else that
invalidates a branch clears the mirror by matching the branch, not by the
scopes some earlier query happened to list.

Only `branch_best_by_policy` is the "one row per branch" table.  Any count,
report, or query that means branches must not union the two: a branch with
results at three budgets is one branch.

### A candidate's own ERD is derived, never stored

A branch result is a certificate; a **candidate's** ERD at a branch is a *fold*
over the results of that candidate's response groups, and the two are not
alike.  `report_model._candidate_erd_summary` is the only thing that produces
one, and it produces it on every read from the group facts the caller has
already materialized.  Nothing persists it.

The reason is that a fold has no way to defend itself.  It asserts "every one
of my response groups is an exact result", and every path that deletes a branch
result — a repair, a reverification, a `queue add --delete-erd-cache` — would
falsify that assertion silently.  A stored fold is keyed by the *parent*
branch, so given a deleted child there is no way to ask which folds read it; a
reverse index would cost up to 243 rows per fold, and a generation counter
would invalidate everything on each deletion anyway.  Deriving costs less than
either: the callers already hold the rows, so a fold is arithmetic over memory
(measured at ~40 µs per candidate across the whole vocabulary), and the report
then describes the cache as it actually stands.

**A fold must select each child at the budget the parent would use.**  Callers
read group facts through `report_branch_states` or
`report_branch_states_from_maps` at the branch's own `group_budget`, which
applies `_exact_row_for_budget` and the same reusability gate `read_for_budget`
does.  A child whose only exact result was solved at some other budget arrives
as `missing`, and the candidate reads `pending` — never folded in.

Do not reintroduce a durable memo for this value, and do not add one to
`EXPORT_TABLES`/`TABLES`.

### A hint cache names a word and nothing else

`--hint-cache` opens a quarantined historical cache alongside the live one.
Its rows were produced by earlier solver versions, so they are descriptive
history, not certificates: a historical row may put its word first in a
branch's candidate order, and may do nothing else.  It is never an exact hit,
never a fold input, never a ceiling, never a loss, never a queue-admission
answer, and never an export source.

`HintCache` is the whole interface, and the guarantee is structural rather
than procedural: its queries select `best_guess` alone, so no stored ERD,
`max_remaining_depth`, or `solve_budget` has a path out of the module.  Do not
widen them, and do not hand the historical file to anything that takes a
`ScoreCache` — that type has `read_for_budget`, and any caller holding one can
take an exact hit from it.

The artifact is opened `mode=ro&immutable=1`, which is stricter than
`read_only=True` on a `ScoreCache`: plain `mode=ro` still touches the `-shm`
sidecar.  `immutable=1` in exchange ignores an uncheckpointed WAL, so the open
refuses a nonempty `-wal` instead of silently reading a stale snapshot.

The hint is applied at two levels, and both ask at the branch's own budget so
the artifact's budget-specific rows are not passed over.
`wordle_engine._hint_first` runs inside the recursion, on the budget the frame
is solving at.  `_BranchWorker._hint_first_in_order` runs on the swarm's branch
packing order, on `active_branches.budget` — written once by `create_branch`,
never updated, and refused to a joiner at a different budget, which is what
keeps `claim_next_bundle`'s shared `pack_cursor` indexing one stable order.
`_packing_stats_cache` is keyed by `(branch_key, budget)` for the same reason:
a finalized branch can be re-created at another budget, and the order left
behind describes the old one.  Both sites are ordering; neither can change an
optimum.

**The hint must reach every frame, not just the entry one.**
`_solve_subset` passes `hint_cache` to `evaluate_candidate` in its candidate
loop as well as forwarding it into the sub-branch recursion.  Dropping either
leaves deeper frames unhinted while every result assertion still passes, so
`test_descendant_frames_get_their_own_ordering_hints` asserts on the branch
sizes actually looked up.

**Hint counters come in two populations and must not be mixed.**
`hint_lookups`/`hint_hits`/`hint_accepted`/`hint_rejected` count *lookups* from
both sites — a cooperative branch is looked up once per worker that computes
its packing order — so they give coverage and legality rates, never branch
counts.  `hint_inline_placements`/`hint_inline_wins` are the separate
same-population pair: one worker owns an inline `_solve_subset` frame from
placement to winner, so their ratio is meaningful.  A cooperative branch's
winner is decided once, at finalize, and is recorded per branch in
`branch_finalize_log.hint_was_winner` — never added to a process counter that
several workers each contributed placements to.

### Completed work has two records, and they can disagree

"Already solved" is answered by the **permanent cache**
(`ScoreCache.report_branch_states`, budget-aware), never by the queue and never
by telemetry. The queue is consulted only to report new-versus-already-queued.

But the queue keeps its own record: a finished branch holds a `done`
`pending_branches` row, and `add_pending_many`'s UPSERT carries priority
forward without touching `status`. A branch can therefore be absent from the
cache and still unclaimable, which is the seam any "recompute this" operation
has to close on **both** sides.

**Liveness lives in `active_branches` + `candidate_claims`, not in
`pending_branches.status`.** `create_branch` accepts any branch key regardless
of pending status, so a branch finished for one request can be re-promoted as
another request's descendant and hold live claims while its pending row still
reads `done`. Any guard deciding "is a worker using this?" must read the active
row; the queue status alone will delete claims out from under their owner.

### ERD-prune provenance

Use **one-level ERD prune** for a candidate completed by the vectorized
closed-form candidate bound, and **two-level ERD prune** for a candidate
completed by the response-group `BranchFloorTable` bound. Reports must show
these as separate metrics wherever candidate-prune completion is displayed.

The SQLite names `bulk_done_candidates`, `bulk_done_bound`, and the historical
`claimed_by = 'bulk-elimination'` value predate that vocabulary. They remain as
legacy storage and compatibility names, not user-facing terminology. Existing
`bulk-elimination` claims are one-level ERD prunes. New claims use
`one-level-erd-prune` or `two-level-erd-prune`, and the legacy aggregate remains
the sum of both provenance-specific counters. Do not rename the legacy columns
without a separate compatibility migration.

---

## Anchored vocabulary

These terms have precise meanings throughout the codebase. Use them
consistently and do not substitute synonyms.

| Term | Meaning |
|---|---|
| **guess** | A word actually played as a turn in the game. |
| **candidate** | A word under evaluation during search — not yet played. A
candidate becomes the guess when it wins. |
| **branch** | The remaining answer words after a guess + response. Identified
by a (guess, pattern) pair at each level. |
| **spine** | The sequence of (guess, pattern) pairs played to reach a branch:
one *path* to it, not the branch itself. The same branch — its remaining answer
set — can be reached by multiple spines, and those spines may be of different
lengths. |
| **opener** | A candidate word selected as the first guess of the game. Every
word the swarm sweeps is queued as an opener. |
| **opener work** | A queued request to solve one opener's whole tree at a
priority, and the ownership every branch in that tree inherits. Keyed by
(opener, priority), so one opener may own several. It answers "who asked for
this branch?", never "where is it?" — position is the spine's job. |

**An opener is an opener by construction, not by convention.** `queue add` is
the only path that creates a top-level request, and it hardcodes `branch_budget
= GAME_GUESSES - 1`; with the invariant `budget + guess_depth = GAME_GUESSES`
that fixes `guess_depth = 1`. There is no `--spine` flag. A word queued at any
other depth is not an unused-but-valid state — it is unrepresentable, because
the recorded budget would assert a `guess_depth` the spine contradicts.

A descendant inherits its opener and the opener's response pattern from its
parent unchanged, so a branch four levels deep still names the opener that
requested it; the guess at the root of *that* branch appears only in the spine.

**Legacy naming.** Tier B (#280) renamed the Linux-only queue identifiers and
schema — `erd_queue.py`, `erd_swarm.py`, `erd_search.py`, `report_model.py`,
`report_terminal.py`, and their tests — from `source_*` to `opener_*`. Tier C
(#281) did the same for the phone-shared cache: `cache_sqlite.py`'s
`completed_source_summaries` and `root_response_group_summaries` tables,
their `source_word` columns, and every caller now read `opener`/
`completed_opener_summaries`/`opener_response_group_summaries`. The rename is
a migration (`ScoreCache._rename_source_summaries_to_opener`), not a
find/replace: it runs once per database, guarded by `schema_migrations`, and
fails loudly if it ever finds both an old- and new-named table present at
once (a half-applied prior run) rather than guessing which copy holds the
good rows. One `source_*` spelling remains deliberately: `report_model.py`'s
branch/membership JSON output keeps the wire key `"root_pattern"` even though
the field is read internally from the renamed `opener_pattern` column —
`report_client.html` reads that exact key and is outside Tier B's rename
scope, so the key name is a deliberate compatibility exception, not a
leftover.

Do not introduce new `source_*` names anywhere else; read every other
`source_*` spelling you encounter as a bug.

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
| **guess_depth** | Guesses already *played* to reach a branch — the number of
guesses on its spine. Root (no guess) = 0; a one-guess seed = 1. The source of
truth. |
| **budget** | *Allowed* remaining depth (the cap). The only quantity the ERD
recurrence reads; each level spends one. |
| **ERD** | Expected remaining depth — the mean line length; the objective
(`cost` / `best_erd`). |
| **max_remaining_depth** | Maximum remaining depth — worst-case line length.
Feasibility gate (solvable iff `≤ budget`) and cache-reuse key (reusable at any
`budget ≥ max_remaining_depth`). |

The Wordle game must be solved in three guesses to win. In the code,
`GAME_GUESSES = 6` is the root's budget.  Invariant at every node: `budget +
guess_depth = GAME_GUESSES`, so `guess_depth = GAME_GUESSES − budget`.  ERD and
`max_remaining_depth` are the mean and max of the same per-answer distribution.

---

## Naming rules

**Use full names. Do not abbreviate identifiers.**
- `max_group_size`, not `max_grp` or `grp`
- `candidate_list`, not `cand_list`
- `branch_words`, not `branch_wds`
- `entropy_gain`, not `ent`

**Acronyms and initialisms keep uniform casing in identifiers.**
Neither an acronym (NASA, POSIX) nor an initialism (ERD, SQL, HTML) is an
ordinary word. When embedding one in a PascalCase or UPPER_CASE identifier,
preserve its casing as a unit — do not title-case it.
- Correct: `ERDQueue`, `SQLCache`, `HTMLParser`, `MPIWorker`, `NASAFeed`, `POSIXPath`
- Wrong: `ErdQueue`, `SqlCache`, `HtmlParser`, `MpiWorker`, `NasaFeed`, `PosixPath`. These misrepresent the acronym/initialism as an ordinary word
- Correct: `ScoreCache`, `BranchWorker`. Score, Cache, Branch, Worker are ordinary words, not acronyms
- Correct: `erd_policy`, `sql_query`, `posix_path`. Snake case variables and parameters that use lowercase uniformly are proper and do not misrepresent anything.

**Names must include all essential context.**

Be careful to consider context when choosing a name. Picking a bound without
naming what it bounds hides context. Choosing an optimization strategy that
omits what is being optimized hides context.

The clearest violation of this rule in this codebase's history was
`ScoringMethod.MINIMAX`. `MINIMAX` names an optimization strategy (minimize the
maximum) but omits what is being minimized — the **group size**. Without that
context, the "minimax" name is uninterpretable. The proper term in this case is
"max group size". The name of the optimization strategy cannot omit "group
size". `MINIMAX` is now only a legacy SQLite key in migration code. Never
introduce it elsewhere.

The same principle applies everywhere: a name must be self-describing without
external context.

**Scoring method names** — use these and no others:

| Enum | SQLite key | Short display | Long display |
|---|---|---|---|
| `MAX_GROUP_SIZE` | `max_group_size` | `max-grp` | `Worst-case group size` |
| `ENTROPY_GAIN` | `entropy_gain` | — | `Entropy gain`
| `WEIGHTED_AVG` | `weighted_avg` | — | `Weighted avg group size` |
| `PROB_FINISH` | `prob_finish` | — | `P(finish next turn)` |

Do not use `minimax`, `max`, `group`, `g-max`, `wt`, or other shorthands for
`MAX_GROUP_SIZE`. The word "group" alone refers to a group of words. It has no
meaning as a number; its use as one would ambiguously refer to count, size, or
identity.

**Response partitions** are "response groups" (the sets of answer words that
produce the same response pattern to a given guess). Do not call them
"subgroups" — that was old vocabulary for what is now "branch."

**A longer spine is not a "nested" one.** A spine is a sequence of (guess,
pattern) pairs; how many it holds is its `guess_depth`. `SCOPE -y--- LUBES` is
a spine at guess_depth 2, not a word nested inside a spine — nothing is
contained in anything. Say "a spine of more than one guess", "a deeper spine",
or name the `guess_depth` outright.

---

## Display conventions

### Integers carry comma separators

Unless separators would mislead, every integer shown to a user is rendered with
thousands separators: `571,359`, not `571359`. This holds in the web client, the
terminal renderer, and CLI output alike.

Separators mislead when the number is an identifier rather than a quantity —
epoch numbers, branch ids, worker ids, ports, years. Those stay bare.

The web client already formats correctly wherever a number reaches `metric()`
or `labeledFacts()`, because `valueOrDash` routes integers through
`formatInteger`. The gap is string concatenation, where a raw number is glued
into a sentence:

```javascript
// Wrong: renders "Shown 1170 of 14855 matched"
"Shown "+shownRows.length+" of "+matchedRows+" matched"
// Right
"Shown "+numText(shownRows.length)+" of "+numText(matchedRows)+" matched"
```

In Python, use the `:,` format spec (`f"{count:,}"`).

### Dates run day, month, year

`1 Aug 2026`, never `Aug 1, 2026`. The American form orders the fields
little-endian then big-endian within one date, which reads as neither. Times go
with it on a 24-hour clock: `1 Aug 2026, 05:06:53`.

In the web client this is the `en-GB` locale — never the viewer's default,
which is American on the machines this runs on:

```javascript
new Date(seconds*1000).toLocaleString("en-GB",{dateStyle:"medium",timeStyle:"medium"})
```

A fully big-endian format is also fine where one already reads well, which is
why the terminal renderer's `%Y-%m-%d %H:%M` stays as it is. Only the mixed
ordering is ruled out.

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

**Migration code is the only exception.** In `_ensure_schema`, comments may
name old table/column structures because a reader needs to understand what
legacy databases look like. Even there, describe what the old structure *is*
(so a reader can recognise it), not the sequence of decisions that led to the
rename.

**Do not comment things that are obvious from the names and types.** A comment
that restates what the code clearly says adds noise without value.

Do not use meta language. Do not refer to things that a cold reader would not
presume. A cold reader does not have access to your context, so do not write
comments with references to information in your context.

---

## Errors are for the exceptional, not the routine

Do not model an expected, recurring state as an error. A raised exception, an
error return, or an HTTP error status (4xx/5xx) is a claim that something went
wrong — reserve it for cases that genuinely did. When a normal lifecycle
transition starts surfacing as an error, the fix is to make the normal path
return a normal result, not to teach the caller to treat the error as success.
Swallowing an exception or an error status to keep going on a routine path is a
red flag: it means the wrong thing is raising.

The clearest example in this codebase's history: a finalized branch could no
longer be resolved by its queue *reference* (a one-way hash the queue must
invert), so the report server answered `404` — and a first attempt "fixed" the
resulting client breakage by catching the `404` and rendering it as though it
were a normal report. That conflates a transport failure with an application
state. The real fix pinned the branch view to its *spine*, which resolves from
the answer list with no queue dependency, so finalization returns an ordinary
`200` and `404` again means only what it should: a reference that never
resolved.

Prefer designs where the common case cannot raise.

---

## Dangerous operations — always ask first

Never perform any of the following without explicit instruction from the user:

- **Merging a PR** — "CI is green" or "looks good" is not a merge instruction.
- **Deleting a branch**
- **Rewriting git history** — including rebase, amend, or any operation that changes existing commits.
- **Force push**

## Never commit directly to main — no exceptions

Committing to `main` is forbidden. This rule has no override: not "just do it",
not "no need for a PR", not any other phrasing. Work always goes on a branch
with a PR for the user to merge. If an instruction genuinely seems to require a
direct commit to `main`, stop and ask; do not infer permission. Honor GitHub's
warning about prohibited operations. Surface these to the user for
confirmation.

Persist source changes; do not persist one-time operations. A migration or data
movement that will only ever run once is executed here (from the scratchpad)
and never committed. Only code that must keep working in the future belongs in
the repository.

These actions are hard or impossible to reverse and affect shared state. Always
confirm before proceeding, no matter how obvious it seems.

---

## Pull request style

PR descriptions need a **Summary** section only. Do not include a "Test plan"
section.

PR descriptions describe *changes* — what is added, removed, or fixed, and why.
This is distinct from the comment style rule ("describe what the code is, not
how it got there"), which applies to inline code comments. A PR description is
inherently a description of a diff; talking about what changed is the point.

Every PR that fully implements a tracked issue must include a GitHub closing
keyword such as `Closes #192` in its Summary so the PR is linked to the issue
and merging it closes the issue automatically. If the PR is only partial work,
link the issue with `Refs #192` instead.

Always update the PR title and description when pushing new commits. Never
leave a PR state that describes only a proper subset of the commits. A PR
should always be left in a reviewable state.

**Always subscribe to a PR's activity immediately after publishing it** — no
need to ask first. This is standing authorization for that one action; it
does not extend to autofixing CI failures or replying to review comments
without checking in, which stay governed by whatever the agent's normal
watch/babysit behavior specifies.

---

## Development environment setup

Dependencies are pinned in two files:

```
pip install -r requirements-dev.txt   # runtime + test deps
pip install -r requirements.txt       # runtime only (numpy)
```

- **NumPy** is a hard runtime requirement (`pattern_matrix.py`). The suite fails
  to import `erd_swarm`/`erd_search` without it.
- **Playwright** drives the browser contract tests in
  `tests/test_report_client.py`. Install the Python package (in
  `requirements-dev.txt`) but do **not** run `playwright install` in the managed
  environment: a Chromium build is already present under
  `PLAYWRIGHT_BROWSERS_PATH`. The browser tests launch the default bundled
  revision when present and otherwise fall back to that pre-installed build by
  path (`_launch_chromium`), so any installed playwright version works without
  matching browser revisions. A browser that will not start is a **failure**,
  never a skip.

  **The suite refuses to run below Python 3.13.** `tests/__init__.py` raises on
  import, because on rocky playwright is installed only under `python3.13` (in
  `~/.local/lib/python3.13/site-packages`) while the default `python`/`python3`
  is 3.9 — which has numpy but not playwright, so the browser suites would skip
  themselves and a run that never touched the report client would still report
  OK. Run everything with `python3.13`.

  **Codex sandbox note:** Codex's default command sandbox may deny the browser
  fixture server's bind to `127.0.0.1:0` with
  `PermissionError: [Errno 1] Operation not permitted`. This is a sandbox
  networking restriction, not evidence of a product or browser-test failure.
  Rerun the same focused browser test command with approved elevated
  execution so it has loopback access, then judge the real test result. This
  note is Codex-specific; other coding harnesses may have different sandbox
  policies.

**Claude Code on the web** starts from a bare image with neither dependency
installed. `.claude/hooks/session-start.sh` installs them into `python3.13`.
Browser tests need nothing set to run; the hook only sets
`SKIP_WEBKIT_CONTAINER_TESTS=1`, and prints that it did, when the image carries
neither podman nor docker. It no-ops on a local checkout (`CLAUDE_CODE_REMOTE`),
which keeps whatever environment the developer set up.

The hook runs asynchronously: the session starts immediately and the install
lands a few seconds later, so a `ModuleNotFoundError` for numpy or playwright
in the first moments of a session means the install is still in flight — wait
and retry rather than installing by hand. It skips the reinstall on a compact
or a clear, which keep the same container.

- **WebKit** launches two ways, tried in order by `_launch_webkit` in
  `tests/test_report_client.py`. First, Playwright's bundled native build
  (`playwright install webkit`) — this is what CI uses, and what any
  environment with a current-enough glibc can use. When that raises,
  `tests/webkit_container.py` falls back to running WebKit inside the
  official `mcr.microsoft.com/playwright` container instead, in `run-server`
  mode: only the browser process lives in the container, started with
  `--network host` so it can reach the fixture server's `127.0.0.1` binding.
  The test process itself stays local and connects over `playwright.webkit.connect()`
  via a `ws://` endpoint — nothing about `tests/test_report_client.py`'s
  existing test bodies changes; `ReportClientWebKitBrowserTest` replays them by
  subclassing `ReportClientBrowserTest` and swapping only
  `setUpClass`/`tearDownClass`. The `run-server` wire protocol requires the
  container image tag and the installed `playwright` package to be the exact
  same version, so the tag is derived from the installed version at run time,
  never pinned.

  **Rocky's bundled WebKit build needs a glibc newer than the box has**, so
  native launch always fails there and every run falls through to the
  container — there is no pre-installed native fallback the way there is for
  Chromium. Rocky has `podman` (not `docker`) and the image pulled;
  `tests/webkit_container.py` tries `podman` first and falls back to `docker`.

  **WebKit runs by default, and a browser that will not start (natively or
  via the container) fails the suite.** This client is used overwhelmingly
  from WebKit — Safari and iOS Chrome — so a green run that covered only
  Chromium would leave the primary engine untested.

  `SKIP_WEBKIT_CONTAINER_TESTS=1` opts out for a machine that can start
  neither the native build nor the container fallback, and
  `SKIP_BROWSER_TESTS=1` opts out of both engines. Setting either is a
  deliberate statement that the run does not cover that engine — reach for
  them only when the environment genuinely cannot host the browser, never to
  get a red suite green.

## Before committing and pushing

Before committing and pushing a code change, run the targeted tests that cover
the changed behavior and its related paths. Use `python3.13`; for example:

```
python3.13 -m unittest tests.test_report_model tests.test_report_terminal
```

Write tests for every new or changed executable path.

**Prove a new test can fail.** Disable the fix — stub the function to a no-op,
or patch the constant back — and confirm the tests that are supposed to catch
the bug do. A test that passes both ways proves nothing, and the failure mode
is quiet: a fixture that never reaches the state it asserts about will report
`0 == 0` forever. Where a guard has a second, *over-eager* way to be wrong,
check that shape too — a predicate that is necessary but not sufficient passes
every test written against the necessary half.

**Build fixtures in the shape production actually leaves behind.** Stopping a
setup helper one call short of what the code does yields a state the system
never reaches, and every assertion built on it is then testing fiction. Follow
the real call sequence when constructing "already finished" or "already
failed" states.

Run the full suite when the change has broad cross-layer risk, when targeted
tests do not provide enough confidence, or when the user asks for it:

```
python3.13 -m unittest discover -s tests -t . -p 'test_*.py'
```

GitHub CI enforces total coverage of at least 98%; run its coverage gate
locally when running the full suite:

```
rm -f .coverage; find . -maxdepth 1 -name '.coverage.*' -delete
python3.13 -m coverage run --parallel-mode \
    -m unittest discover -s tests -t . -p 'test_*.py'
python3.13 -m coverage combine
python3.13 -m coverage report --fail-under=98
```

**`--parallel-mode` and `combine` are required, not optional.** Without them
every process writes the single `.coverage` path, and the scaling tests fork
workers that inherit the tracer and overwrite the parent's data when they exit.
Last writer wins, so the total swings run to run — 98.0% and 93.9% from two
identical runs, the whole difference being `erd_swarm.py`, the module those
workers execute. CI already does this because it combines across its jobs; only
the local single-file invocation was exposed. (Clear the data files with `find
-delete`: under zsh a `.coverage.*` glob that matches nothing aborts the whole
command, silently skipping the run chained after it.)

### Coverage source inventory

`.coveragerc`'s `source` list is the production Python inventory admitted to
the 98% gate, not a hand-picked subset selected to preserve the percentage.
Every module that implements the game, solver/kernel, cache, swarm, operator
CLI, reporting interfaces, or a shared production utility must either be in
that list or be named in an active, staged coverage-backfill issue. A module
is not exempt because another covered module imports it: coverage only records
files named in `source`.

Test helpers, one-off dataset and diagnostic scripts, and `verify_*.py` audit
passes are outside the gate. Any other exclusion needs a comment in
`.coveragerc` explaining why it is not production code or naming its active
backfill issue. When adding, removing, or materially repurposing a top-level
Python module, review this inventory in the same change and run the combined
coverage gate. If the expanded scope falls below 98%, add tests for the missing
production behavior rather than narrowing the inventory.

Commits with failing tests must not be pushed.

A commit whose diff is entirely markdown, documentation, or other non-code
files cannot change test outcomes, so the suite is not required for it. Push it
as is; do not fix unrelated pre-existing failures to clear the gate. If the
diff touches code at all — including test files, build configuration, or a code
snippet embedded in a doc that the suite executes — the rule above applies in
full.

**`SWARM.md`'s command examples are executable.**
`test_every_swarm_guide_command_example_parses` extracts each one and feeds it
to the CLI parser as argv, so a shell-only construct — a pipe, a redirect, a
`$(…)` — fails the suite even though the change is "just prose". Editing that
file is a code change; run
`python3.13 -m unittest tests.test_report_terminal` after touching it.

**Stage files explicitly, by path.  `git add -A`, `git add .`, `git add -u`,
and `git commit -a` are prohibited — no exceptions.**  The working tree
routinely holds untracked files that must never be committed: SQLite
databases and their WAL/shm files, logs, scratch output from ad-hoc runs.
Bulk staging sweeps them into the commit silently.  Run `git status` first,
then name every file: `git add <path> <path> ...`.

---

## A screenshot is not evidence about layout width

A full-page screenshot (`full_page=True`) sizes the image to the page's
`scrollWidth`, not to the viewport it was captured at. A page that overflows
horizontally therefore renders as a *wider image whose contents look correctly
laid out* — every row fits, nothing is clipped, and the defect is encoded in
the image's dimensions rather than anywhere in the picture. Reading that
screenshot as "the phone layout is fine" is reading the overflow as the design.

So a rendering never settles a question about width. Measure:

```python
page.evaluate("() => [document.documentElement.scrollWidth,"
              " document.documentElement.clientWidth]")
```

`scrollWidth > clientWidth` at any required width (375, 390, 480, 800, 1200) is
a horizontal-scroll regression.
`test_no_horizontal_scroll_at_required_widths` measures exactly this across
every view; run it when a change touches layout, and extend it when a view it
does not visit is added — a guard that never navigates to a view has never
covered it.

When a screenshot is still the clearest way to show a rendering, confirm its
width equals the viewport width before concluding anything from it, and say
which width it was captured at.

---

## Respond to what's actually being asked

Before acting, classify the request:
- **"What should I do / what's your recommendation / what are your thoughts?"**:
  Teason through the tradeoffs and give a recommendation in words. Do not
  write or push code. Stop there and wait for a decision.
- **"Implement X" / explicit instruction to make a change**:  code is appropriate.

A bug report or "this is wrong" is not, by itself, authorization to start
editing files. Diagnose and propose a fix in conversation first, unless the
user's phrasing already asks for the fix to be made.

When in doubt about which mode applies, ask, or default to discussion rather
than to code — reverting unwanted code is more disruptive than a follow-up
question. Writing code speculatively burns tokens unnecessarily.

**The `stop-hook-git-check.sh` Stop hook is advisory, not a directive.** It
fires on every turn end and pushes toward committing/pushing whenever the tree
is dirty, signing looks wrong, or commits are unpushed — it has no idea whether
a change was meant for discussion or was actually approved. Neither the user
nor the assistant edits this hook, so the only way to keep it from overriding
the discussion-first norm above is judgment when it fires: read its complaint,
then act on what the user actually wants in the conversation, not on the hook's
say-so by itself. Committing/pushing unapproved work just to satisfy the hook
is the wrong move — it's fine to leave changes uncommitted and explain the
tension to the user instead.

---

## Schema coordination (Linux + phone)

The cache (`runtime/wordle_cache.sqlite3`) is shared between Linux and the iOS
app. Any schema change must:
1. Be implemented as an idempotent migration in `ScoreCache._ensure_schema`,
   guarded by the `schema_migrations` table
2. Deploy new code to the phone **before** syncing a migrated Linux database to it
3. Never require manual SQL — migrations run automatically on first open

The queue (`runtime/erd_queue.sqlite3`) is Linux-only; its migrations live in
`ERDQueue._migrate()`. A queue migration needs no coordination — but workers
fork from the supervisor, so **merging does not deploy**. Stop the swarm, pull,
migrate, then start. A worker running pre-migration code against a migrated
database fails on a column that no longer exists, and the symptom is `view`
erroring on a column the migration "already ran". Stopping costs nothing here:
precache work has no deadline, `stop` drains cleanly, and in-progress claims
survive a restart.

**Only five tables cross between Linux and the phone.** `export_cache.py`'s
`EXPORT_TABLES` and `import_cache.py`'s `TABLES` both move `answer_list`,
`response_decomposition`, `branch_best_by_policy`,
`branch_best_by_policy_and_budget` and `candidate_scores`. A cache table
outside that list — the per-opener summaries, for instance — is written locally
on each machine and never travels, so changing its shape cannot produce a file
the other side fails to read.

An older export may still carry `candidate_erd_by_policy`. It is not in either
list, so it is skipped by the same path any unrecognized table takes.

`import_cache.py` migrates the target itself: `main()` calls
`_bootstrap_target_schema` before merging a single row, which opens the target
through `ScoreCache` so `_ensure_schema` runs first. A phone cache is therefore
brought current by the import, with no separate migration step to sequence. The
import is run from Pythonista with **no arguments**, so its defaults are part of
the contract: source `wordle_erd_export.sqlite3` in the working directory,
target `runtime/wordle_cache.sqlite3`, and the source file is deleted after a
successful merge unless `--keep-source` is given.

Swarm telemetry lives in a **separate** Linux-only file,
`runtime/erd_queue_telemetry.sqlite3` (`<stem>_telemetry<ext>`, computed by
`derive_telemetry_path`), which `ERDQueue` opens as an attached schema named
`telemetry`. The `claim_telemetry` and `branch_finalize_log` tables are there, not
in the main queue file — `add_claim_telemetry` and `add_branch_finalize_log` write
`telemetry.<table>`, and reads join through the `telemetry.` prefix. Because
attached-schema tables do not appear in the main file's `sqlite_master`, running
`.tables` on `runtime/erd_queue.sqlite3` shows no telemetry tables; open the
telemetry file directly (or query through a live `ERDQueue`) to inspect them. Rows
are fenced by the active telemetry epoch (`erd_search.py epoch show`).

### Disk: measure free pages before reclaiming any

Only the queue accumulates free pages, because only it deletes rows in bulk;
the telemetry and cache files are append-mostly and sit at 0%. Check before
assuming a large file has anything to give back:

```
PRAGMA page_size; PRAGMA page_count; PRAGMA freelist_count;
```

`VACUUM` rewrites the file, so it needs the swarm stopped and free space for a
full copy. Back up first. Row-count-per-table and `PRAGMA integrity_check`
before and after are what confirm it only moved bytes.

A completed branch costs almost nothing: `pending_branches` is a few thousand
rows. The queue's bulk is the append-only `branches` registry — millions of
`branch_key` blobs that everything else references by `branch_id`, so it cannot
be trimmed without breaking those references.

---

## How to interact with GitHub

### Creating a PR

When you create a PR, make sure that the title and body are always up to date
and reflect all committed changes. Whenever you push additional commits, check
again that the title and body are reflective of all the work.

If you have the ability to subscribe to GitHub events, do so and monitor the PR
for comments and activities. If you see CI failures, diagnose and push changes.

### Reviewing a PR

If you are asked to review a PR, post your findings (negative and/or positive)
as comments on the PR, either to the PR general discussion or (better) inline
comments anchored to specific files and line numbers. Post your testing methods
along with your findings. If you have recommendations about how to fix the
problems you find, provide those as well.

If you have the ability to subscribe to GitHub events, do so and monitor the PR
for new changes. If you discover new commits, review the PR again, following
the above instructions about posting your findings.


## Claude token usage

If you are a Claude agent, check remaining token budget and reset time, run:

```
claude -p /usage
```

There is no in-session tool that exposes this — it is the only way to see the limits.

There are three important lines in the usage report: current session, current
week (all models), and current week (Fable only). Take into account all of
them. If you are finding you are running short on tokens, reduce your work.

## All-model token usage

Err on the side of writing scripts to process data instead of consuming tokens
reading output files in their entirety yourself.

---

## Cross-agent memory (read-only)

If you are running on this physical machine (the "rocky" box), regardless of
which agent or tool you are (Claude Code, Codex, or otherwise), consult:

```
/home/ahern/.claude/projects/-home-ahern-work-wordle/memory/
```

`MEMORY.md` in that directory indexes topic files covering the user's setup,
standing feedback on how to work in this repo, and ongoing-project state that
is not derivable from the code or git history alone. Reading it before
starting non-trivial work will surface context you would otherwise have to
re-derive.

**That directory is read-only to every agent except Claude Code**, which owns
it as its persistent memory store. Unless you are Claude Code, do not create,
edit, or delete files there under any circumstances — treat it the same as
any other tool's private state you happen to have filesystem access to.
