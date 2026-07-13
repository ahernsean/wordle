# Unified reporting system — overview and index

This directory specifies one read-only reporting system for the ERD swarm.
It replaces separate terminal status, queue-inspection, and browser concepts
with one report model served through three clients:

- `erd_search.py view` for terminal use, structured output, and optional
  refresh;
- a small HTTP adapter over the same reports;
- a responsive browser client for navigation and visual presentation.

The terminal remains a direct client of the model and never requires the HTTP
service. The browser is another presentation of the same report semantics,
not a second monitoring implementation.

## Product shape

The finished system supports three kinds of use without making them separate
products:

1. an operational overview of swarm progress;
2. exploration of work by semantic spine;
3. focused reports for queue state, workers, cache coverage, and hotspots.

Semantic input is inferred from spine form, so users do not have to choose a
`word` or `branch` command before entering the work they want to inspect.
Tree presentation is a layout option over extant queue topology, not another
domain object. Exact command syntax, inference rules, and collection filters
belong to phase 3.

Cache results remain flat reporting data. They may annotate a branch that
still exists in queue state, but cached ERD results never create or
reconstruct tree nodes after their queue topology is gone.

## Architecture

    erd_queue.sqlite3 ─────┐
    queue telemetry DB ────┼─► report_model collectors ─► report envelope
    wordle_cache.sqlite3 ──┘              │
                                          ├─► report_terminal
                                          │      └─ erd_search.py view
                                          │
                                          └─► report_server
                                                   └─ report_client.html

The report model owns normalized domain meaning, stable identities, source
health, and bounded data assembly. Clients own layout, interaction, display
formatting, and refresh sessions. The HTTP layer adapts transport parameters
to the same request objects used by the terminal; it contains no parallel SQL
or normalization logic.

## System boundaries

All reporting paths are read-only. Service lifecycle commands remain
top-level, and queue mutations remain under `queue`. Once object reports ship,
legacy read-only status and queue-inspection commands are removed; backward
compatibility is not required.

Reports are intentionally bounded:

- overviews and inventories return summaries plus bounded rows;
- expensive claims, telemetry, and candidate detail load only on demand;
- watched terminal output and browser polling never scan complete history;
- an unavailable queue, telemetry, or cache source yields a partial report
  with source health rather than erasing available data.

Tree and landscape views have an additional boundary: their hierarchy comes
only from currently recorded queue spines. There is no cache topology index,
cache-parent migration, or inferred historical tree in this plan.

## Delivery phases

| Plan | Deliverable | Depends on |
|---|---|---|
| `01-report-model.md` | shared envelope, normalized overview entities, lifecycle semantics, and overview collector | issue #92 path decision |
| `02-terminal-view.md` | terminal overview renderer with text, JSON, and optional watch | 01 |
| `03-object-reports.md` | semantic selection, collection filters, object reports, live queue tree layout, and legacy read-command removal | 02 |
| `04-report-server.md` | stdlib HTTP adapter over all reports and per-report fixtures | 03 |
| `05-browser-client.md` | navigable polling browser client over the shared reports | 04 |
| `06-visual-modalities.md` | independent browser visual and interaction upgrades | 05 |
| `07-landscape-view.md` | non-implementable live-work landscape vision and prerequisites | 05 plus relevant phase 06 items |

Implement one numbered phase per branch and pull request, in numeric order.
A phase starts only after its prerequisites are merged and is complete only
when every acceptance item in its document is true.

## Sequencing constraint

Issue #92 proposes one owner for runtime paths. Phase 1 must start only after
that issue's path/module decision is merged, or these plans must first be
updated to name the actual shared path owner. Report modules and the server
must import defaults from that owner rather than copying path strings.

## Design principles

- One report model owns domain semantics for every client.
- Semantic work selection is shared across CLI, HTTP, and browser.
- The terminal is first-class and works without a server.
- Optional refresh wraps the same one-shot report collection.
- Full identities drive joins, navigation, and change detection.
- Expensive detail is fetched only when requested.
- Queue topology is live operational state; cache state is not historical
  topology.
- Shipped server and client code add no third-party runtime dependency.
- Reports never mutate queue, telemetry, cache, or service state.

## How to use these plans

Read the repository root `AGENTS.md` before implementation. Its vocabulary,
naming, comment, testing, git, and pull-request rules are binding. Each phase
is the normative implementation specification for its own level of detail;
later phases may extend earlier public structures only where they say so.

Before every commit and push, run:

    python -m unittest discover -s tests -t . -p 'test_*.py'

If current code contradicts a phase specification, stop and report the
discrepancy rather than improvising across phase boundaries.

## Glossary

| Term | Meaning |
|---|---|
| report | One presentation-neutral, JSON-serializable result for a report request. |
| report envelope | The shared top-level structure surrounding report-specific data. |
| selector | Semantic input identifying the root, a word within branch context, a branch spine, or a queue branch reference. |
| spine | Guesses played from the root to a branch, represented as word/pattern pairs. |
| descent | A worker's live recursion path below its claimed branch. |
| tree layout | Hierarchical presentation of extant queue rows and their recorded spines. |
| snapshot | One report collected at one instant; successive snapshots feed watch or polling. |
