# Wordle solver — codebase instructions

## Project structure

A Wordle solver with five layers:
- **Engine** (`wordle_engine.py`): core ERD search, scoring, response simulation
- **Kernel** (`pattern_matrix.py`): NumPy response-pattern matrix and vectorized candidate statistics; the engine's sole NumPy import point. NumPy is a hard requirement on every target; the pure-Python engine paths remain permanently as the reference implementation (they never call NumPy), selected by passing `pattern_matrix=None`
- **Cache** (`cache_sqlite.py`): SQLite-backed persistence of branch results and candidate scores
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
`clear`, `priority`, `source-priority`, `reset-stale`, and
`reconcile-orphaned-ownership`. The `queue` group has no read-only dashboard
commands.

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
  matching browser revisions. Without the package the browser tests skip; set
  `REQUIRE_PLAYWRIGHT_BROWSER=1` to make them hard-fail instead of skipping.

  **On rocky, playwright is installed only under `python3.13`** (in
  `~/.local/lib/python3.13/site-packages`). The default `python`/`python3` is
  3.9, which has numpy but **not** playwright — run the suite under it and the
  browser tests silently skip. Run with `python3.13` to actually exercise
  `tests/test_report_client.py`.

**Claude Code on the web** starts from a bare image with neither dependency
installed. `.claude/hooks/session-start.sh` installs them into `python3.13`
and sets `REQUIRE_PLAYWRIGHT_BROWSER=1`, so the browser tests fail loudly
there rather than skipping. It no-ops on a local checkout
(`CLAUDE_CODE_REMOTE`), which keeps whatever environment the developer set
up.

The hook runs asynchronously: the session starts immediately and the install
lands a few seconds later, so a `ModuleNotFoundError` for numpy or playwright
in the first moments of a session means the install is still in flight — wait
and retry rather than installing by hand. It skips the reinstall on a compact
or a clear, which keep the same container.

## Before committing and pushing

Run the test suite before committing and pushing any change that touches code:

```
python3.13 -m unittest discover -s tests -t . -p 'test_*.py'
```

Use `python3.13`, not the default `python` (3.9): only 3.13 has playwright, so
under 3.9 the browser contract tests skip rather than run.

Commits with failing tests must not be pushed.

A commit whose diff is entirely markdown, documentation, or other non-code
files cannot change test outcomes, so the suite is not required for it. Push it
as is; do not fix unrelated pre-existing failures to clear the gate. If the
diff touches code at all — including test files, build configuration, or a code
snippet embedded in a doc that the suite executes — the rule above applies in
full.

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
`ERDQueue._migrate()`.

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
