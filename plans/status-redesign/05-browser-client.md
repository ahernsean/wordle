# Phase 5 — Navigable browser report client

Read phases 00–04, including 3a–3d, and `AGENTS.md` first. Requires phase 4
merged.

## Goal

Replace the placeholder with one self-contained responsive browser client
that can navigate the same inferred selectors and report kinds as the
terminal:

- operational overview;
- word response-group coverage;
- branch detail;
- tree layout;
- queue inventory;
- workers;
- cache;
- hotspots.

The browser polls the currently selected report, preserves navigation/filter
state in the URL, and highlights semantic changes by stable identity.

## Non-goals

- No canvas or semantic-zoom landscape.
- No queue/service controls.
- No historical persistence or charting beyond data already returned by
  bounded reports.
- No framework, build step, CDN, or runtime dependency.

## Files touched

- `report_client.html` — replace placeholder
- `tests/test_report_client.py` — new Playwright tests

## Page shell

Single HTML file with inline CSS and JavaScript. Dark theme, system monospace
font, and the Wordle palette:

| Meaning | Color |
|---|---|
| background | `#121213` |
| text | `#d7dadc` |
| dim | `#818384` |
| green tile/improvement | `#538d4e` |
| yellow tile | `#b59f3b` |
| gray tile | `#3a3a3c` |
| alert red | `#cc4444` |
| stale amber | `#d0a215` |

Top-level controls:

1. Report navigation buttons: Overview, Queue, Workers, Cache, Hotspots.
2. One selector input labeled “Spine or @branch reference.”
3. Apply and Back buttons.
4. Tree toggle.
5. Expandable filters for branch status, branch phase, answer count, budget, priority, sort,
   limit, hotspot field/epoch/window.
6. Connection chip and generated timestamp.

The selector input always uses inference:

- `CRANE` opens a word report;
- `CRANE -y--g` opens branch detail;
- Tree changes layout without changing selector;
- the UI never asks the user to choose word versus branch.

Overview and entering a selector switch to inferred mode. Clicking
Queue/Workers/Cache/Hotspots selects the corresponding explicit kind; the
selector may remain as scope when compatible. Navigation retains only filters
valid for the destination report.

## URL state

Encode navigation in `location.search`:

    ?selector=CRANE+-y--g
    &kind=auto
    &tree=1
    &branch_status=active,pending
    &branch_phase=evaluating
    &sort=nodes
    &limit=25
    &poll=2000

Omit default values. Use one comma-separated parameter per branch-filter axis;
`all` explicitly disables the root overview's default active filter. On Apply/filter
changes, call `history.pushState`; Back uses browser history. `popstate`
reconstructs controls and fetches immediately.

Build the API URL from this normalized state:

- `kind=auto` → `/api/view`;
- explicit kind → `/api/view/<kind>`;
- copy only API-supported report parameters;
- `poll` remains client-only.

Do not encode word/branch type in the URL. The server returns the inferred
`report_kind`.

## Client architecture

Use three testable layers:

    parsePageState(location) -> state
    buildAPIURL(state) -> string
    applyReport(report, previousReport, state) -> DOM

Assign all three to `window` for Playwright.

`applyReport`:

- performs all report-kind dispatch, rendering, and semantic diff;
- does not fetch;
- does not read the browser clock for report data age;
- derives report ages from `report.generated_at` and entity timestamps;
- may read viewport-independent expansion sets keyed by full identity;
- never mutates the report objects.

The poll loop:

- fetches the URL from `buildAPIURL`;
- remembers the previous report only when report kind, selector, and filters
  are the same comparison context;
- calls `applyReport`;
- polls every `poll` milliseconds, default 2000;
- skips while `document.hidden` and fetches immediately when visible;
- leaves prior data visible through disconnects.

After two consecutive failures, mark disconnected and show time since last
successful response. On recovery, clear the warning without treating every
row as newly added.

## Shared header

Every report renders:

- report kind and selector;
- generated local time;
- queue, telemetry, and cache source paths;
- telemetry epoch/label/revision;
- partial source errors;
- active branch-status and branch-phase filters;
- connection state.

Paths collapse behind a details disclosure on phone widths.

## Inferred word view

Render:

- trailing word with answer flag;
- context spine as Wordle pattern tiles;
- response-group summary counts;
- sortable response-group table/cards containing pattern, answer count,
  branch status, branch phase, cache state, best guess/ERD,
  `max_remaining_depth`, workers, and
  branch reference.

Click/tap a response group appends its pattern to the selector and navigates
to branch detail. No separate “open branch” mode chooser appears.

Branch filters hide unmatched response rows but leave the unfiltered summary
visible and label the shown/matched counts.

## Branch detail

Sections:

1. Semantic spine and identity.
2. Queue state: branch status, branch phase, priority, budget, candidate progress, bulk-done
   count, running best, ceiling, nodes, workers.
3. Cache state: exact/loss/missing, best guess/ERD,
   `max_remaining_depth`, reusable-budget information.
4. Live workers and descents.
5. Recent finalizations: exact/cut/loss, ceiling, evaluated claims,
   bulk-done candidates, bundles, nodes/wall, epoch.
6. Cut-reuse misses.
7. Republished candidates and bundle summary.
8. Optional answer-word and candidate-detail disclosures.

Opening answer words refetches with `answers=1`. Opening candidate detail
refetches with `claims=1` and keeps that flag only while the disclosure is
open. Do not request either payload for collapsed detail.

Bulk-eliminated candidates display as proofs, not worker activity.
A selected branch remains visible if a refresh moves it outside its parent
filter. The detail calls that out; Back returns to the refreshed filtered
parent, where the branch is absent.

## Tree layout

`tree=1` renders the report's flat topology as a collapsible nested DOM tree.
It is a layout of the current root/word/branch selection, not its own
navigation tab. Its nodes derive only from extant queue rows and their
recorded spines; cache results may annotate queued nodes but never create
nodes.

- Each node shows guess/pattern tiles, `guess_depth`, branch status and phase, answer count,
  progress, branch reference, and worker chips where available.
- Context nodes remain visible when branch filters exclude descendants.
- Unknown legacy spine segments render as `?` at their known
  `guess_depth`.
- Clicking a branch node removes tree mode and navigates to its semantic
  selector when available, otherwise its `@branch_reference`.
- Tree collapse state is keyed by stable `node_id` and survives polling.
- Below 480 px, descendants begin collapsed but the context path remains
  visible.
- If a selected semantic branch exists only in cache, show “No extant queue
  topology” and keep branch detail available outside tree mode.

## Overview and object reports

### Overview

Responsive header metrics, filtered branch cards, nested worker rows, and
idle/finalizing/dead worker lane. Preserve the useful information from the
terminal overview without matching its exact line layout.

### Queue

Summary groups above filtered rows. Table on desktop, stacked rows on phone.
Tree toggle reuses the common tree renderer.

### Workers

One stable card/chip per worker with state, heartbeat age, branch,
candidate, absolute `guess_depth`, node rate, cache/prune counters, and
descent. Clicking its branch navigates to branch detail.

### Cache

Global exact/loss/recent totals and distributions, or selector-scoped cache
coverage/detail. Never label a cut as cached.

### Hotspots

Clearly label population and sampling:

- active branch ranking;
- recent finalized branch ranking;
- cut-reuse ranking;
- coordination workload buckets.

Show sample/window/epoch and a truncation badge when applicable.

## Semantic change highlighting

Compare only within the same report context. Identities:

- branches: `branch_key_hex`;
- workers: `worker_id`;
- response groups: `branch_key_hex`;
- tree nodes: `node_id`;
- candidates: `candidate_index` within branch;
- hotspot buckets: collector-provided stable bucket identity.

Rules:

| Change | Class |
|---|---|
| identity appears | `flash-added`, green border pulse |
| `completed_candidate_count` increases | `flash-improved` |
| `best_erd` decreases | `flash-improved` |
| exact cache row appears | `flash-improved` |
| other semantic field changes | `flash-changed`, red fade |
| stale threshold crossed | persistent amber |
| worker liveness threshold crossed | persistent red |

Generated time, heartbeat age, ETA, and connection timer ticking alone do not
flash. Moving between cards/branches is detected from identity fields, not DOM
position.

Remove transient classes after 1.5 seconds. Sticky row/card ordering keeps
existing identities in relative order and appends new identities.

## Responsive behavior

- No horizontal body scrolling at 375, 390, 480, 800, or 1200 px.
- Tap targets at least 32 px high.
- Tables become stacked label/value rows below 600 px where column removal
  would hide essential semantics.
- Pattern tiles wrap only between word/pattern steps.
- Dense telemetry and candidate sections are collapsed by default on phones.
- Desktop may use multi-column cards; phone uses one column.

Use CSS media queries, not JavaScript width branching, for ordinary layout.

## Security and robustness

All database-derived strings are untrusted for HTML purposes. Prefer DOM APIs
and `textContent`. If template strings are used, pass every interpolated text
value through one tested HTML-escape helper. Never assign raw spine, word,
error, source path, or telemetry text to `innerHTML`.

Reject malformed report shapes with a visible error while retaining the last
valid report. Require matching `schema_version`.

The HTML has no external `src`, `href`, CSS import, font, or network request
other than same-origin API fetches.

## Automated browser tests

`tests/test_report_client.py` uses Playwright/headless Chromium when available
and skips cleanly otherwise. Start the real fixture server on an ephemeral
loopback port. Use a fresh page per test.

Required cases:

1. Selector parser/UI infers CRANE as word and CRANE -y--g as branch without a
   type chooser.
2. CACHE and QUEUE positional selectors request inferred word reports;
   navigation buttons request explicit cache/queue endpoints.
3. Tree and comma-separated branch filters alter URL/API state and keep the context node.
4. Clicking a word response group navigates to its full branch spine.
5. Clicking a tree/branch reference navigates to detail.
6. Overview defaults to active branches and renders every worker state.
7. Candidate disclosure performs a claims request only while expanded and
   labels bulk elimination without a worker chip.
8. Exact/cut/loss and cut-reuse sections render distinctly.
9. Semantic changes flash only their matching identities/fields.
10. Ticking generated time does not flash.
11. Sticky order survives input reordering and branch finalization.
12. Expansion/collapse state survives polls and browser Back.
13. Disconnect preserves prior data and recovers.
14. URL state round-trips comma-separated status/phase values, including
    explicit `all`, and compatible filters.
15. Selected detail survives leaving its parent filter; Back returns to the
    refreshed filtered collection.
16. Malicious fixture text renders literally and cannot create an element or
    execute script.
17. Tile colors match the Wordle palette.
18. No horizontal body scroll at all required widths.
19. Screenshot artifacts at 390 and 1200 px for overview, word, branch, and
    tree views are written to a temporary directory for reviewer inspection.

## Manual checklist

- [ ] On the actual iPhone, selector entry, Back, filters, tree collapse, and
      branch disclosures are comfortable.
- [ ] Against live databases, browser and terminal JSON reports agree for the
      same selector/filter request.
- [ ] A two-second overview poll does not create visible queue contention or
      sustained telemetry scans.
- [ ] Navigating to a claim-heavy branch loads detail only on expansion.

## Acceptance checklist

- [ ] One self-contained browser client supports every report kind.
- [ ] Word versus branch is inferred only from selector form.
- [ ] Tree and branch status/phase are ordinary shared controls.
- [ ] URL state is deep-linkable and browser-history aware.
- [ ] Semantic identity/change behavior matches the terminal contract.
- [ ] Bulk elimination is never presented as a worker.
- [ ] All content is responsive and safely escaped.
- [ ] Full test suite passes or Playwright tests skip cleanly when unavailable.
- [ ] No file outside the phase list is modified.
