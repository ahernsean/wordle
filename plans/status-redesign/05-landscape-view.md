# Phase 5 — Search landscape view (PLACEHOLDER — not yet implementable)

**Status: vision capture only.** This document preserves design context so it
is not lost between sessions. Do NOT implement from it. It becomes a real
plan only after phase 3 is in daily use and phase 4 item D (the spine tree)
has taught us the guess tree's layout semantics at DOM scale. An implementer
agent that lands here by mistake should stop.

## The vision

A pinch-zoomable map of the entire search landscape. Zoomed out: the broad
shape of exploration — where in the guess tree work has concentrated, as
density/heat rather than individual nodes. Pinching in resolves regions into
branches, then into individual branches with their state, then into a
branch's candidates: what has been fully evaluated, what is being computed
right now (and by which worker), what was cut off or pruned, and what
remains.

This is **semantic zoom**: what an element *is* changes with scale, not just
its size. It is a map, not a dashboard — a different artifact from the phase
3/4 status page, sharing the same data service.

### Context anchoring: the loupe / minimap

Zooming must never cost situational awareness: at any zoom level the user
should still see where in the larger forest the current view sits, and where
work is happening elsewhere. Two dual forms of the same **overview + detail**
pattern (both standard in mapping and visualization UIs):

- **Minimap**: the main view is the zoomed detail; a small inset shows the
  whole forest with (a) a rectangle marking the main view's viewport and
  (b) persistent markers for live activity — every worker's current branch,
  plus recent-completion heat. Tapping a marker pans/zooms the main view to
  it.
- **Loupe**: the main view is the broad forest; one or more small magnified
  windows float over it, each anchored by a leader line to its true position
  in the forest and showing detail-level content (e.g. one worker's live
  descent into a sub-branch) regardless of the main view's zoom. "Follow
  worker N" is a loupe pinned to that worker's position as it moves.

Both let the user watch one worker chew through a sub-branch while
simultaneously seeing where that sub-branch sits in the whole landscape.

## Why this is canvas (and phases 3–4 are not)

The status page shows dozens of live entities; DOM/SVG handles that with
free hit-testing, styling, and testability. The landscape shows the
*explored tree* — millions of cached branch results — orders of magnitude
past what a DOM can hold, and semantic zoom requires redrawing content per
zoom level anyway. Immediate-mode rendering (canvas; WebGL only if canvas
proves too slow) is the right tool at that scale.

## Architecture commitments (decided now so earlier phases don't foreclose them)

1. **Scene graph separated from painting.** A pure function
   `buildScene(data, viewport) → display list` produces typed, positioned
   drawables with all level-of-detail decisions already made; a thin paint
   loop walks the list onto the canvas. The display list is plain data, so
   the phase 3 Playwright approach still works: tests call `buildScene` via
   `page.evaluate` and assert on the scene ("at this zoom, this region is a
   density blob; two ticks in, it is 14 branch nodes"), never on pixels.
   Hit-testing is a lookup into the same scene, so taps are testable too.
   Screenshots remain review artifacts, never assertions.
2. **Tile-style data access.** The landscape cannot ride the
   poll-everything snapshot. It needs an on-demand endpoint over the cache
   database (`branch_best_by_policy`) shaped like map tiles: "children of
   this node, within this depth window, at this aggregation level",
   fetched as the user zooms and pans, cacheable client-side. This is
   additive to `status_server.py` — a new route beside `/api/status`,
   same raw-data-keyed-by-identity principle.
3. **Live state overlays static landscape.** The explored tree (cache DB)
   is the terrain; the phase 1 snapshot (queue DB: open branches, workers,
   claims) is a small live layer drawn on top of it. The two data sources
   stay separate; the client composites them.
4. **Views are viewport-parameterized and composable.** `buildScene` takes
   the viewport (center, scale, pixel size) as an argument, so the minimap
   and each loupe are just additional `buildScene` calls with different
   viewports painted onto their own small canvases, sharing one client-side
   tile cache with the main view. No view is special; all are equally
   testable through the scene graph.
5. **Tree-space coordinates are stable and addressable.** The layout function
   must map any node to its position deterministically from the node's
   identity/spine alone — never from which parts of the tree happen to be
   loaded or how the view arrived there. This is what lets a minimap place a
   worker marker (whose location is known only as a spine from the
   heartbeat) without loading that region's tiles, lets a loupe anchor its
   leader line correctly at any zoom, and keeps positions consistent across
   all simultaneous views.
6. **Same palette, same meanings.** Wordle tile colors for pattern/word
   semantics; the phase 3 alert-red/amber only for change/staleness. New
   encodings (e.g. heat for exploration density, a distinct treatment for
   pruned vs. cut-off vs. unexplored) must not collide with those.

## Open questions (to answer when this becomes a real plan)

- **Layout**: how to place a 243-way branching tree spatially — radial,
  icicle/flame, treemap, or force-directed? Phase 4 item D's experience
  decides.
- **Aggregation semantics**: what a "region" summarizes at each zoom level
  (node counts? ERD distribution? recency of work?), and where those
  aggregates are computed (SQL at request time vs. a maintained summary
  table — a schema change, which triggers the CLAUDE.md Linux+phone
  migration rules).
- **Prune/cutoff visibility**: per-candidate outcome (`n_ok` / `n_cutoff` /
  `n_pruned`) is currently aggregated per worker heartbeat, not recorded
  per branch node. Showing "what got pruned" at the map's finest zoom may
  need the engine to persist more than it does today — scope carefully;
  possibly a coarser proxy (cached vs. absent) is enough for v1.
- **Overview + detail ergonomics on a phone**: minimap and loupe are duals;
  which is the default given iPhone screen real estate? (Likely: minimap
  always present as a small corner inset; loupes opt-in, one at a time,
  spawned from "follow worker N".) How many simultaneous views fit in the
  performance budget?
- **Gesture handling**: pinch/pan via Pointer Events with
  `touch-action: none` on the canvas; inertia and zoom-about-point math;
  what the desktop equivalent is (wheel + drag).
- **Performance budget**: target 60 fps pan/zoom on an iPhone; decide the
  display-list size cap and tile prefetch strategy against that.

## Prerequisites before promoting this to a real plan

- Phase 3 shipped and in daily use (there is something concrete to react to).
- Phase 4 item D (spine tree) merged — the DOM-scale precursor.
- A decision on the aggregation/persistence questions above, taken with the
  user in conversation, not by an implementer agent.
