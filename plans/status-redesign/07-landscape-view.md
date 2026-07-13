# Phase 7 — Live work landscape

**Status: vision capture only; not implementable.**

Do not create code from this document. Promote it to an implementation plan
only after the prerequisites are met and the remaining layout and interaction
questions are decided with the user.

## Vision

A pinch-zoomable map of the swarm's extant work:

- zoomed out: the shape and concentration of queued work;
- closer: recorded spines resolve into branches and lifecycle;
- closer again: workers, progress, finalization, and live descent become
  visible;
- finest useful level: a selected branch's current candidate sweep.

This is semantic zoom over live operational topology. It is a map sharing the
report system's identities and navigation, not a historical reconstruction of
all solved ERD work.

## Authoritative data boundary

The landscape contains only topology that exists in queue state:

- user and cooperative queue rows provide branch nodes;
- their recorded spines provide parent/child placement and any structural
  ancestor nodes needed to connect them;
- current heartbeats, claims, candidate progress, and bounded telemetry
  provide live annotations;
- incomplete legacy spines produce explicit unknown segments.

Cache data may annotate an extant queue node with its current exact, loss, or
missing state. It never creates a node or edge. When a branch no longer has
queue topology, it disappears from the landscape even if its result remains
cached.

Consequences:

- no cache-parent schema or phone-coordinated cache migration is needed;
- no cache-derived topology index or answer-subset DAG is needed;
- no cached branch is reconstructed from best guesses;
- the ordinary bounded queue-tree report is the starting data contract.

## Navigation continuity

Opening the landscape from a word, branch, queue, or worker report preserves
the current semantic selector. The selected context remains anchored while
the user pans and zooms, and a copied URL preserves selector and viewport.

The shared collection filters continue to apply according to phase 3. They
change which live rows are emphasized or included without changing branch
identity or inventing missing topology.

### Minimap

A small inset may show the complete extant queue landscape with:

- the main viewport rectangle;
- live worker and active-branch markers;
- the selected context;
- bounded recent completion activity where available.

Selecting a marker recenters the main view.

### Loupe

A focused view may follow one selected worker or branch while the main view
retains broader context. A leader line anchors the loupe to its actual
position. Worker following uses full branch identity so movement cannot be
confused with row reordering.

Minimap and loupe should share the same viewport and layout primitives as the
main view.

## Implementation direction

Start with accessible DOM/SVG using the bounded live queue population already
returned by the tree report. Do not assume Canvas is necessary. Move painting
to Canvas 2D only after measurement shows that DOM/SVG misses explicit
interaction budgets on the actual target phone; consider WebGL only after a
measured Canvas failure.

If a retained scene abstraction proves useful, keep layout separate from
painting:

    buildScene(data, viewport, filters) -> display list

The same semantic display list should drive rendering, hit testing, minimap
placement, and tests. Stable coordinates must derive from recorded spine and
full identity, independent of input row order, current viewport, and display
filtering.

The ordinary `/api/view?tree=1` report is the initial transport. A specialized
route is justified only if measured live queue size or payload cost exceeds a
named budget. This phase does not assume cache tiles or a separate terrain
service.

## Open design questions

### Layout and aggregation

Evaluate deterministic layouts using feedback from the phase 5 tree and
phase 6 context navigator:

- radial tree;
- icicle/flame layout;
- treemap weighted by answer count or measured work;
- hybrid overview density plus local tree.

For each zoom level, define what a cluster summarizes and name its population
and time window. Possible live measures include queued branch count,
answer-count mass, nodes/wall, candidate progress, lifecycle, and worker
activity. Avoid force-directed layout unless stable coordinates can be
proven.

### Candidate detail

Worker heartbeats aggregate evaluation outcomes and do not persist complete
candidate history. The finest view may use only provenance that exists:

- current sparse claim state;
- evaluated and bulk-eliminated completion markers;
- republished candidates;
- bounded finalization and cut-reuse telemetry.

Do not infer per-candidate history from aggregate counters.

### Phone interaction and accessibility

Decide:

- minimap always visible versus collapsible;
- maximum one loupe versus several;
- tap, long-press, keyboard, and screen-reader behavior;
- pinch/pan and desktop wheel/drag equivalents;
- breadcrumb and browser Back behavior.

The map must retain a nonvisual semantic representation and usable focus
order. Visual density cannot replace labels or report navigation.

### Performance

Set measured budgets for node count, response bytes, layout time, memory, and
frame time. Test on the actual iPhone. The display should reduce level of
detail before missing its interaction budget.

## Testing direction

When promoted to an implementation plan:

- unit-test deterministic coordinates from recorded queue spines;
- verify filters do not change coordinates of retained identities;
- assert semantic scene contents and hit tests at several scales;
- test branch and worker navigation, Back, minimap, and loupe agreement;
- verify cache-only branches never appear as landscape nodes;
- test keyboard and screen-reader alternatives;
- measure live phone frame time and memory;
- use screenshots only as review artifacts.

## Prerequisites

- Phase 5 is merged and used regularly.
- Phase 6 item D has produced real tree-navigation feedback.
- Candidate detail correctly distinguishes evaluated and bulk-eliminated
  work.
- The largest realistic live queue has been measured through the ordinary
  tree report.
- Layout, aggregation populations, interaction budgets, and accessibility
  behavior are decided.
- A small prototype proves stable coordinates and phone performance before
  this becomes an implementation phase.
