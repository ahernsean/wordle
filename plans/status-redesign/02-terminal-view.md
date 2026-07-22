# Phase 2 — Terminal overview, JSON, and watch

Read `00-overview.md`, `01-report-model.md`, and `AGENTS.md` first. Requires
phase 1 merged.

## Goal

Add the first client of the report model:

    python erd_search.py view
    python erd_search.py view --watch
    python erd_search.py view --watch 10
    python erd_search.py view --format json
    python erd_search.py view --format jsonl --watch 2

This phase proves that one collected report supports one-shot text, structured
output, and refresh without the terminal owning any database assembly.

## Non-goals

- No positional branch_target or object-report flags yet; phases 3a–3c add them.
- Do not remove `status` or read-only `queue` commands yet.
- No branch/worker drill-down hotkeys yet.
- No HTTP or browser code.

## Files touched

- `report_terminal.py` — new
- `erd_search.py` — register `view` and delegate to the terminal client
- `tests/test_report_terminal.py` — new
- `tests/test_status_sections.py` — existing legacy tests remain unmodified

## Public terminal API

`report_terminal.py` exports:

    render_overview(report: dict,
                    previous_report: dict | None = None,
                    *,
                    color: bool = False,
                    width: int | None = None,
                    display_order: DisplayOrder | None = None) -> str

    run_view(args) -> None

    WatchSession
    DisplayOrder

No renderer function opens SQLite or imports `ERDQueue`/`ScoreCache`. The
only data entry point is `collect_report` from `report_model`.

`DisplayOrder` is a normal dataclass or small class, not state stored on a
function object. It keeps branch keys and worker IDs stable across refreshes:
existing identities retain relative order, new identities append in incoming
order, and identities absent from both the new report and any pinned
selection are removed.

`WatchSession` owns the prior report, display order, terminal mode, selected
identity state reserved for phase 3d, and cursor restoration. No refresh state
is stored on `render_overview` or `collect_report`.

## CLI parser

Register `view` in `erd_search.py` with:

    --watch [SECONDS]     nargs="?", const=30, type=float
    --format              choices=("text", "json", "jsonl"), default="text"
    --no-color
    --queue-path PATH
    --cache-path PATH

Default paths come from the shared runtime-path owner required by phase 1.
Reject a watch interval less than 0.2 seconds with an argparse error.

Rules:

- `--format json --watch` is invalid; direct the user to `jsonl`.
- `--format jsonl` without `--watch` emits one JSON line and exits.
- `--no-color` wins even in a TTY.
- Color defaults on only for watched text in a TTY.
- JSON/JSONL is the report envelope from `collect_report` with
  `json.dumps(..., sort_keys=True)`; the terminal renderer does not reshape it.

## Text overview

Derive presentation values from report fields:

- cache hit rate from live-worker totals;
- ERD-cutoff and remaining-depth-pruned rates;
- branch completion percentage;
- branch ETA from candidate progress and elapsed time;
- heartbeat ages from `generated_at - updated_at`;
- node-rate and duration abbreviations.

The report model remains the authority for branch status/phase, answer flags,
`guess_depth`, `best_max_remaining_depth`, and identities.

The text view has named sections:

1. Header: generation time, source paths/health, telemetry epoch/revision.
2. Cache: exact/loss totals, recent exact rows, live-worker cache hit rate.
3. Queue: pending/active/finalizing/done counts and pruning effectiveness.
4. Active branches: stable branch reference, answer count, `guess_depth`,
   candidate completion, bulk-done count when nonzero, running best, workers,
   ETA, and compact spine.
5. Workers nested under branches, followed by idle/finalizing/dead workers.

### Adaptive width contract

Terminal text is designed for 50–60-column sessions first, not obtained by
rendering a wide row and clipping its tail. Every repeated row set is a table
with one header. Field names such as `Phase`, `Age`, and `Current` appear once
in that header; rows contain only values and never repeat `phase=`, `age=`,
`candidate=`, or similar labels.

`report_terminal.py` owns one reusable column-layout helper. Each column
declares:

- a short, unambiguous heading;
- left or right alignment;
- whether it is essential;
- its removal priority when space tightens;
- a minimum and optional maximum display width;
- whether its values may be head- or tail-truncated.

The helper measures headings and already-formatted values, including
production-sized counts, then applies this order:

1. include every column that fits at its measured width;
2. remove nonessential columns in their declared order until the table fits;
3. shrink only columns explicitly marked elastic;
4. below the minimum tabular width, use stacked heading/value rows.

Branch references, worker IDs, warnings, response patterns, and numeric
progress remain complete. Spines, source paths, answer-word lists, and other
free text are elastic; spines retain their most recent end when truncated.
Summary prose wraps at field boundaries rather than disappearing behind a
line-wide ellipsis.

The overview's 50–59-column branch table keeps `Ref`, `GuessD`, `Phase`,
`Done`, and `W`; wider layouts add answer count, bulk completion, best result,
`MaxRD`, ETA, and spine in priority order. Worker tables keep identity, state,
age, current candidate, and absolute maximum guess depth before optional rate
and counter columns. At width 80 or greater, source paths and expanded metrics
may appear, but the same measured layout still decides what actually fits.

These rules apply to every terminal renderer added in phases 3a–3c and to the
navigation footer in phase 3d. The report model never receives terminal width
or presentation-priority state.

## Semantic change highlighting

Watched text compares reports by `branch_key_hex` and `worker_id`.

- New identity: green.
- `completed_candidate_count` increase or `best_erd` decrease: green.
- Other changed semantic values: red.
- Heartbeat age and ETA ticking alone: no change highlight.
- Crossing stale/dead thresholds: amber/red persistent warning.
- Row movement caused by grouping or source query order: no highlight.

Implement comparison on values before formatting. Do not diff rendered
characters. ANSI escape sequences are introduced only by the renderer after
the semantic class is known.

## Watch behavior

### TTY text

Use a `WatchSession`:

1. Hide cursor and clear once.
2. Collect and render immediately.
3. Wait until the deadline while polling stdin at most every 0.2 seconds.
4. Space refreshes immediately.
5. `q`, `Q`, Ctrl-D, or Ctrl-C exits.
6. Refresh only changed named sections, clearing leftover lines when a section
   shrinks.
7. Restore terminal attributes and cursor in `finally`.

Do not use `tty.setraw()`; preserve output processing as the existing status
watch does.

### Non-TTY text

Never hide the cursor, clear the screen, or emit cursor-control sequences.
Append a complete snapshot per interval, prefixed by:

    --- generated_at=<unix-seconds> ---

Flush after every snapshot.

### JSON Lines

Emit one compact JSON object and newline per refresh, then flush. Do not emit
terminal control codes or human separators, regardless of TTY state.

## Error behavior

A partial report with `sources.*.ok == false` renders the source error and
continues displaying available sections. A collector exception outside the
report's partial-failure contract:

- prints one concise error to stderr in one-shot mode and exits nonzero;
- renders an error section in watch mode and retries on the next interval;
- never leaves the terminal in noncanonical mode or with the cursor hidden.

## Tests

`tests/test_report_terminal.py` uses hand-built report dictionaries and mocked
collectors/clocks; it must not depend on live databases.

Required cases:

1. One-shot text contains source health, queue counts, branch reference,
   worker, `guess_depth`, and no ANSI when color is false.
2. Overview rendering at widths 50, 55, 59, 60, 79, 80, and 120 has no line
   longer than requested and exposes exactly the expected priority columns.
3. Production-shaped values such as `12,616/12,972`, multi-digit workers, and
   long spines cannot shift a value under the wrong heading.
4. Narrow branch and worker rows use headers and contain no repeated field
   labels; semantic identities and warnings remain complete.
5. JSON output round-trips to the exact report dictionary.
6. JSONL watch emits independently parseable lines.
7. `json + watch` and sub-0.2-second intervals are rejected.
8. Non-TTY watched text contains separators and no escape character.
9. A branch progress improvement is green; an unrelated worker field change
   is red; ticking timestamps alone produce no highlight.
10. Reordered input identities keep prior display order.
11. A shrinking section clears its old terminal lines.
12. Simulated collector failure followed by success retries and restores
    cursor/terminal state.
13. One source unavailable still renders the other source.
14. Legacy `status` tests remain green unmodified.

## Acceptance checklist

- [ ] `erd_search.py view` renders the overview through `report_model`.
- [ ] One-shot text, JSON, and watched JSONL describe the same envelope.
- [ ] Watch state lives in an object, not function attributes.
- [ ] Change detection is semantic and identity-based.
- [ ] Every overview table follows the adaptive width contract at 50–120
      columns without losing identities or misaligning values.
- [ ] Non-TTY output contains no cursor-control sequences.
- [ ] Existing `status` remains available during this transition.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
