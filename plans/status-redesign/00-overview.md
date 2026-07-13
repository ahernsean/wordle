# Unified reporting system — overview and index

This directory specifies one read-only reporting system for the ERD swarm.
The system has three clients over one presentation-neutral model:

- `erd_search.py view` for terminal exploration, JSON output, and watched
  refresh;
- a small HTTP service exposing the same reports;
- a responsive browser client for navigation and visual presentation.

The browser is not a replacement for a separate terminal status
implementation. Both clients consume the same report collectors, selectors,
filters, identities, and lifecycle semantics.

## How to use these plans

1. Read the repository root `AGENTS.md` first. Its vocabulary, naming,
   comment, test, git, and pull-request rules are binding.
2. Implement exactly one numbered phase per branch and pull request, in
   numeric order. Do not start a phase until its prerequisites are merged.
3. Touch only the files listed by that phase.
4. Treat every specified public name, field, selector rule, and acceptance
   requirement as normative. If the current code contradicts the plan, stop
   and report the discrepancy rather than improvising.
5. Run the full suite before every commit and push:

       python -m unittest discover -s tests -t . -p 'test_*.py'

6. A phase is complete only when every acceptance item is true.

## User-facing grammar

The primary command is:

    python erd_search.py view [SPINE ...] [REPORT OPTIONS]

The positional input is always a spine selector. Its final token determines
what is being selected; the user never names `word` or `branch` as a command
kind:

| Selector form | Inferred selection | Default report |
|---|---|---|
| omitted | root | operational overview |
| final token is an unpatterned five-letter word | word at the preceding branch | response-group coverage across queue and cache |
| final token is a response pattern | branch reached by the complete spine | branch detail |
| single `@<digest-prefix>` token | queued branch identity | branch detail |

Examples:

    python erd_search.py view
    python erd_search.py view CRANE
    python erd_search.py view "CRANE -y--g ALIBI"
    python erd_search.py view "CRANE -y--g ALIBI g-g--"
    python erd_search.py view @8b31f30d421a

In `CRANE -y--g ALIBI`, `CRANE -y--g` is the branch context and
`ALIBI` is the word whose response groups are being inspected. Adding a
pattern after `ALIBI` identifies one resulting branch.

Response-pattern parsing retains the existing conventions: `g` is green,
`y` is yellow, and `-` or `.` is gray. A selector containing a dash-leading
pattern should be quoted as one shell argument, or use dots for gray.

### Tree is a layout, not an object

`--tree` changes the selected report from a single-level/detail presentation
to its recorded descent topology:

    python erd_search.py view --tree
    python erd_search.py view CRANE --tree
    python erd_search.py view "CRANE -y--g" --tree

- With no selector, the tree starts at the root.
- With a trailing word, the first level is that word's response branches and
  recorded queued descendants continue below them.
- With a branch selector, the selected branch is the root and recorded queued
  descendants continue below it.

The selected root is retained as context even when it does not satisfy a row
filter.

### Global object reports use flags

`CACHE` and `QUEUE` are valid Wordle words, so positional reserved nouns would
make the CLI ambiguous. Global report kinds therefore use flags:

    python erd_search.py view --queue
    python erd_search.py view --workers
    python erd_search.py view --worker 2
    python erd_search.py view --cache
    python erd_search.py view --hotspots --by coordination

A spine selector may scope a compatible global report:

    python erd_search.py view "CRANE -y--g" --queue
    python erd_search.py view "CRANE -y--g" --hotspots --by nodes

At most one of `--queue`, `--workers`, `--worker`, `--cache`, and
`--hotspots` may be supplied. The database overrides are named
`--queue-path` and `--cache-path`, so they cannot collide with report-kind
flags.

### Common filters and output controls

Every branch-collection report supports the same normalized filters where
they are meaningful:

    --active-only
    --status pending|active|finalizing|done|unqueued
    --minimum-answer-count N
    --maximum-answer-count N
    --budget N
    --priority N
    --sort FIELD
    --limit N

`--active-only` is the ergonomic shorthand for `--status active` and is
mutually exclusive with explicit `--status` arguments. Active means a
user-queued `in_progress` branch or a cooperative `open` branch. It excludes
pending, finalizing, and done branches.

All report kinds support:

    --watch [SECONDS]     repeat; omitted value defaults to 30 seconds
    --format text|json|jsonl
    --no-color
    --queue-path PATH
    --cache-path PATH

`--format json` is one-shot. A watched structured report uses JSON Lines so
each refresh remains independently parseable. In a non-TTY, watched text
appends timestamped snapshots and never emits terminal cursor-control
sequences.

## Normalized lifecycle

SQLite uses different status values for user-queued and cooperative work.
The report model exposes one lifecycle vocabulary:

| Report lifecycle | Source state |
|---|---|
| `pending` | `pending_branches.status = 'pending'` |
| `active` | user `in_progress` or cooperative `active_branches.status = 'open'` |
| `finalizing` | finalized branch still referenced by a live heartbeat |
| `done` | `pending_branches.status = 'done'` |
| `unqueued` | a word response branch absent from queue state |

Cache state is independent of lifecycle and uses `exact`, `loss`, `missing`,
or `not_applicable`. A cut is transient telemetry/coordination state, not an
exact cache state.

## Architecture

    erd_queue.sqlite3 ─────┐
    queue telemetry DB ────┼─► report_model collectors ─► report envelope
    wordle_cache.sqlite3 ──┘              │
                                          ├─► report_terminal
                                          │      └─ erd_search.py view
                                          │
                                          └─► report_server
                                                   └─ report_client.html

The terminal calls the collectors directly. It never requires the HTTP
service. The server is a transport adapter over the same collectors, not a
second data-assembly implementation.

## Report envelope

Every collector returns a JSON-serializable dictionary with the same
top-level shape:

| Key | Meaning |
|---|---|
| `schema_version` | version of the shared report contract |
| `report_kind` | `overview`, `word`, `branch`, `queue`, `workers`, `cache`, or `hotspots` |
| `generated_at` | Unix timestamp at collection |
| `selector` | normalized inferred selector, or null |
| `filters` | normalized filters actually applied |
| `sources` | queue, telemetry, and cache paths, health, telemetry epoch, and Git revision where available |
| `data` | report-kind-specific presentation-neutral payload |

The model contains domain semantics, not database accidents: normalized
lifecycle, `guess_depth`, `max_remaining_depth`, cooperative status, answer
flags, and stable identities. Renderers own percentages, ETA, abbreviations,
color, truncation, column choice, and layout.

## Data-volume boundary

There is deliberately no “complete everything” polling snapshot.

- Overview and collection reports contain bounded summaries and rows.
- Candidate claims, completion indices, bundles, republished candidates,
  finalization history, and cut-reuse misses are branch detail loaded on
  demand.
- Telemetry reports require a branch, epoch, time window, or limit backed by
  bounded/indexed queries.
- No two-second browser poll may scan the full claim or telemetry tables.

This boundary keeps terminal watch and browser polling cheap while allowing
deep inspection when requested.

## Identity

`branch_key_hex` is the durable machine identity used for diffing and joins.
Human-facing reports show `branch_reference`, an `@` followed by the first
12 hexadecimal characters of the existing SHA-1 branch-key digest. Resolvers
must collect all prefix matches and reject ambiguity; they never silently
choose a colliding branch. Semantic spines remain the preferred durable
human reference.

Workers are keyed by `worker_id`. Candidates within a branch are keyed by
candidate index.

## Refresh semantics

One-shot and watched output use the same collector and renderer. Watch is a
transport/session wrapper with these responsibilities:

- maintain sticky branch and worker display order by stable identity;
- compare semantic fields between snapshots;
- preserve selected branch/worker context through finalization;
- refresh immediately on space and exit on `q` or Ctrl-C in a TTY;
- restore terminal state and cursor visibility on every exit path.

Browser polling follows the same identity rules. A value moving to a
different screen position is not itself a change.

## Read-only boundary

`view`, the report server, and the browser are strictly read-only.

Lifecycle commands remain top-level:

    start  stop  restart  run

Queue mutations remain under `queue`:

    queue add
    queue remove
    queue clear
    queue priority
    queue reset-stale

After phase 3, `queue` contains mutations only. The legacy `status`,
`cache-status`, and read-only queue subcommands are removed; backward
compatibility is not required.

## Phases

| Plan | Deliverable | Depends on |
|---|---|---|
| `01-report-model.md` | shared report envelope, selector/filter model, normalized overview entities and collector | issue #92 path decision |
| `02-terminal-view.md` | terminal renderer and `erd_search.py view` overview with text/JSON/watch | 01 |
| `03-object-reports.md` | inferred spine exploration, queue/workers/cache/hotspot reports, tree layout, legacy read-command removal | 02 |
| `04-report-server.md` | stdlib HTTP adapter over all reports and per-report fixtures | 03 |
| `05-browser-client.md` | navigable polling browser client over the same selectors, filters, and reports | 04 |
| `06-visual-modalities.md` | independent browser visual upgrades | 05 |
| `07-landscape-view.md` | non-implementable semantic-zoom vision and prerequisites | 05 plus relevant phase 06 items |

## Runtime-path sequencing

Issue #92 proposes one owner for runtime paths. Phase 1 must start only after
that issue's path/module decision is merged, or this plan must first be
updated to name the actual shared path owner. No report module or server may
copy default database paths by value.

## Design principles

- One selector grammar and filter vocabulary serve CLI, HTTP, and browser.
- One report model owns domain semantics.
- Full identities are explicit; display references are collision-checked.
- The terminal remains first-class and works without a server.
- The browser is responsive, self-contained, and usable without internet.
- Shipped server/client code adds no third-party runtime dependency.
- Expensive detail is fetched only when requested.
- Queue/cache unavailability produces a partial report with source errors
  rather than destroying the rest of the view.
- Reports never mutate queue, telemetry, cache, or service state.

## Glossary

| Term | Meaning |
|---|---|
| report | One presentation-neutral, JSON-serializable result for a report kind, selector, and filter set. |
| report envelope | The shared top-level structure surrounding report-specific data. |
| selector | Parsed root, word, branch-spine, or `@branch_reference` input. |
| spine | Guesses played from the root to a branch, represented as word/pattern pairs. |
| trailing word | The final unpatterned word in a selector; the word whose response groups are inspected within the preceding branch. |
| descent | A worker's live recursion path below its claimed branch. |
| branch reference | Collision-checked 12-hex SHA-1 digest prefix displayed with `@`. |
| active | Normalized lifecycle for user `in_progress` or cooperative `open` work. |
| tree layout | Hierarchical presentation of a selection and recorded queued descendants; not a separate domain object. |
| snapshot | One report collected at one instant; successive snapshots feed watch/poll comparison. |
