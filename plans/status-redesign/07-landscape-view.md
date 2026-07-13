# Phase 7 — Explored search landscape

**Status: vision capture only; not implementable.**

Do not create code from this document. It becomes an implementation plan only
after the prerequisites at the end are met and the open data/layout questions
are decided with the user.

## Vision

A pinch-zoomable map of the explored search landscape:

- zoomed out: density and concentration of cached/explored work;
- closer: regions resolve into branch topology and lifecycle;
- closer again: individual branches, workers, finalization outcomes, and
  cache state;
- finest useful level: a branch's candidate sweep—evaluated, bulk eliminated,
  in flight, republished, cut, and remaining.

This is semantic zoom: what an element represents changes with scale. It is a
map sharing selectors and data semantics with the report system, not a giant
version of the overview dashboard.

## Selector and filter continuity

The landscape uses the same inferred selector grammar:

    CRANE
    CRANE -y--g
    CRANE -y--g ALIBI
    CRANE -y--g ALIBI g-g--
    @8b31f30d421a

Opening the map from a word report centers on that word's response branches.
Opening it from branch detail centers on that branch. A copied URL preserves
the selector and viewport.

`--tree` has no separate meaning here: the landscape is another
representation of the same topology learned by the phase 5 tree layout and
phase 6 item D.

`active-only` filters the live queue/worker overlay, not the cached terrain.
An inactive selected context remains visible. Other filters may alter overlay
or terrain styling only when their population is explicit.

## Context anchoring

Zooming must not destroy situational awareness.

### Minimap

The main view shows detail; a small inset shows the whole available landscape:

- viewport rectangle;
- live worker markers;
- active branch markers;
- recent completion density within the reported window;
- selected context.

Tapping a marker recenters the main view.

### Loupe

The main view remains broad while one magnified window follows a selected
worker/branch. A leader line anchors it to its actual position. “Follow worker
N” from the DOM tree becomes a loupe that moves by full branch identity.

Minimap and loupe are composable forms of the same viewport-plus-scene
function, not special one-off renderers.

## Why canvas

The ordinary browser reports show tens or hundreds of entities and should
remain DOM-based for accessibility, layout, hit testing, and testing.

The explored landscape may contain millions of cached branch results.
DOM/SVG cannot hold that population, and semantic zoom requires rebuilding
visible representation by scale. Use Canvas 2D first; consider WebGL only
after measured Canvas failure.

## Architecture commitments

### Scene graph before painting

    buildScene(data, viewport, filters) -> display list

This pure function produces typed positioned drawables with semantic
level-of-detail choices already made. A thin painter draws the list.
Hit-testing uses the same display list.

Playwright tests assert scene objects and hit-test results, not pixels.
Screenshots remain review artifacts.

### Tile-style data access

The map cannot poll a complete report envelope containing the terrain. It
needs cacheable on-demand tiles such as:

    children/aggregates below this selector
    within this guess_depth range
    at this aggregation level
    for this viewport/region

This becomes a separate read-only route family beside `/api/view`. It reuses
`ReportSources`, selector parsing, identity, lifecycle, and source health, but
not the ordinary polling payload.

### Static terrain plus live overlay

- Terrain: explored/cache-derived branch structure and aggregates.
- Overlay: ordinary lightweight queue/worker report data.

The overlay continues polling by stable identity. Terrain tiles are
client-cached and invalidated by explicit version/epoch metadata, not
refetched every two seconds.

### Stable coordinates

Any semantic node must map to the same tree-space position from its durable
identity/spine and layout parameters, independent of:

- load order;
- current filters;
- current viewport;
- which tiles happen to be cached;
- active-only state.

This allows minimaps and loupes to place activity without loading full detail.

### Viewport composition

`buildScene` takes center, scale, pixel dimensions, and filters. The main map,
minimap, and each loupe call the same function with different viewports and
share one client tile cache.

### Palette

Keep Wordle colors for pattern/completion semantics and alert colors for
change/staleness. Exploration heat, exact/cut/loss, bulk elimination, and
unexplored state need distinct shape/texture/opacity semantics decided before
implementation and usable without color vision.

## Fundamental data gap

`branch_best_by_policy` is keyed by encoded answer subset and does not store a
durable parent spine for every cached row. Therefore “draw the entire cached
tree” is not a query the current cache can answer directly.

Before implementation, choose and validate one source of terrain topology:

1. reconstruct descendants on demand from a selected branch's best guess and
   response decomposition;
2. maintain an external Linux-only derived topology/index;
3. extend persisted cache data with parent/edge information through the
   required idempotent phone-coordinated migration;
4. deliberately map answer-subset space rather than pretending it is a unique
   guess tree.

These choices have different correctness, storage, phone-deployment, and
layout consequences. Do not pick one inside an implementation session.

## Open design questions

### Topology

A branch can be reached by multiple guess spines even when its answer subset
is identical. Decide whether the landscape maps:

- the optimal-policy tree;
- every recorded queue spine;
- unique answer-subset DAG nodes;
- or a selected-root reconstruction.

The choice determines identity and whether “parent” is even unique.

### Layout

Candidate layouts:

- radial tree;
- icicle/flame layout;
- treemap weighted by answer count or measured work;
- layered DAG;
- hybrid overview density plus local tree.

Phase 5 tree usage and phase 6 item D should provide evidence. Avoid
force-directed layout unless deterministic stable coordinates can be proven.

### Aggregation

At each zoom level, define exactly what a region summarizes:

- branch count;
- answer-count mass;
- nodes/wall;
- exact/cut/loss outcomes;
- bulk/evaluated work;
- recency within a named window;
- worker activity.

Every aggregate needs an explicit population, epoch/window, and complete
versus sampled label.

### Pruning and candidate outcomes

Worker heartbeats aggregate `n_ok`, `n_cutoff`, and `n_pruned`. They do not
persist every candidate/branch outcome historically. The finest landscape
must use only provenance that actually exists:

- current sparse claim state;
- bulk-elimination markers;
- bounded finalization telemetry;
- exact cache results;
- transient cuts and recorded reuse misses.

Do not infer nonexistent per-candidate history from aggregate counters.

### Phone interaction

Decide:

- minimap always visible versus collapsible;
- maximum one loupe versus several;
- tap/long-press meanings;
- pinch/pan with Pointer Events and `touch-action: none`;
- desktop wheel/drag equivalents;
- context breadcrumb and browser Back behavior.

Target 60 fps interaction on the actual iPhone before adding visual density.

### Performance

Set measured budgets for:

- display-list object count;
- tile response bytes;
- tile build/query time;
- client tile-cache memory;
- overlay polling cost;
- paint time per frame.

The system must degrade level of detail before missing interaction budgets.

## Testing direction

When promoted to a real plan:

- unit-test selector-to-tile requests;
- unit-test deterministic layout coordinates;
- call `buildScene` through Playwright and assert semantic scene contents at
  several scales;
- test hit targets and navigation;
- test active-only overlay without terrain movement;
- test minimap/loupe coordinate agreement;
- use screenshots only for review;
- measure live iPhone frame time and memory.

## Prerequisites

- Phase 5 is merged and used regularly.
- Phase 6 item D is merged and has produced real tree-navigation feedback.
- Candidate detail correctly distinguishes evaluated and bulk-eliminated work.
- The terrain-topology data gap above is resolved in conversation.
- Aggregation populations and persistence are decided.
- Any cache schema change has an idempotent migration and phone-first
  deployment plan under `AGENTS.md`.
- A small prototype proves stable coordinates and phone performance before
  this becomes a numbered implementation phase.
