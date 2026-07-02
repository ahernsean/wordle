# Phase 4 — Visual modalities (a menu, not a single phase)

Read `00-overview.md` first. Requires phase 3 merged.

This plan is different from the others: it is a **menu of independent
increments**, each small enough for one session. Implement exactly ONE item
per session/PR, chosen by the user, in any order unless an item names a
prerequisite. Everything here is client-side only (`status_client.html`),
except item E which also extends the model and server.

Shared constraints for every item:

- The file stays self-contained (inline CSS/JS/SVG, no external resources).
- Keep the Wordle tile palette from phase 3 for anything meaning
  green/yellow/gray; use the phase 3 alert-red/amber only for change/staleness
  meanings. Do not introduce colors that collide with those meanings.
- Change-highlighting and identity rules from phase 3 apply to new elements:
  anything showing a value diffs by identity, ticking-with-time values don't
  flash.
- Every drawing must degrade at phone width (390 px) without horizontal body
  scroll.
- Every item extends `test_status_client.py` (phase 3's Playwright harness)
  with cases for its new DOM: drive rendering via `applySnapshot` snapshot
  pairs, assert on elements/classes/computed styles, and add a screenshot
  artifact of the new visual at 390 px and 1200 px.

## Item A — Candidate sweep bar with worker markers

Replace the expanded card's `sweep done/total` text line with a horizontal
bar (plain `div`s, no canvas):

- Track = full candidate range `0..n_candidates`; filled portion =
  `done_candidates / n_candidates`, in green.
- One marker per worker on the branch, positioned at
  `claim_idx / n_candidates` of the track width, labeled with its
  `worker_number`. Markers that would overlap nudge right until free (mirror
  the terminal's nudge in `_print_status`'s branch-detail bar).
- The un-expanded card header keeps its numeric `%` — the bar is
  detail-level only.
- Acceptance: with the fixture, the bar shows the fill and both worker
  markers; done-count increases flash the fill green per phase 3 rules.

## Item B — Per-branch completion ring in the card header

A small (≈1.2 em) SVG donut in each card header showing
`done_candidates / n_candidates`, replacing nothing (the `%` number stays
next to it). One `<circle>` with `stroke-dasharray`; green on tile-gray.
- Acceptance: ring fraction matches the number at several fixture values,
  including 0 % and 100 %.

## Item C — Worker chips and a swarm lane

Give workers a persistent visual identity so movement between branches is
followable:

- Each worker renders as a **chip**: rounded box `W3`, tinted by a stable
  per-worker hue (`hue = (numeric worker_number * 47) % 360`, fixed
  saturation/lightness consistent with the dark theme).
- Branch cards show their workers' chips in the card header (replacing the
  bare worker-count number); the idle/finalizing/dead section becomes a
  single horizontal "lane" of the remaining chips.
- When a chip's `branch_key_hex` differs from the previous snapshot, the chip
  gets `flash-added` in its new location (phase 3 rule — this makes the
  "two workers moved onto a branch" event, the original problem 1 example,
  legible at a glance).
- Acceptance: editing the fixture to move a worker between branches makes the
  chip appear pulsing under the new branch on the next poll.

## Item D — Branch spine tree

A structural view of where in the guess tree the open branches sit. Prereq:
none, but read `_fmt_spine_path`/spine handling in phase 1 first.

- Build a tree client-side by merging all branches' `spine` arrays: nodes are
  `(guess, pattern)` steps; the root is the bare root (guess_depth 0). Each
  open branch is a leaf badge (`#branch_id`, % done, worker chips if item C
  is merged).
- Render as nested `<ul>` indentation (no canvas needed) with the guess word
  + pattern tiles per node; branches with unrecorded spines group under a
  `?` node at their `guess_depth`.
- Collapsible at each node; placed in a section below the branch cards,
  collapsed by default on < 480 px viewports.
- Acceptance: with the fixture, both branches appear under their spine paths;
  toggling nodes works; phone width shows the section collapsed.

## Item E — Candidate grid (sibling view) — needs model + server work

Shows a branch's whole sweep as a grid of candidate slots, one cell per index
`0..n_candidates-1`: done, in-flight (claimed, with worker number), or
unclaimed. This is the "candidates in relation to their siblings" view.

Model/server extension (mirror the phase 1/2 style):

1. `erd_queue.py` already has the needed read-only method — use it, do not
   add another: `claims_for_branch(branch_key)` returns full
   `candidate_claims` rows ordered by `idx` (a superset of the
   `idx`/`claimed_by`/`done` fields consumed below).
2. `status_model.py`: add
   `collect_branch_detail(queue_path, branch_key_hex) -> dict` returning
   `{"schema_version", "generated_at", "branch_key_hex", "claims": [{"idx": int, "worker_number": str or null, "done": bool}]}`
   (derive `worker_number` from `claimed_by` the same way workers do; null
   `claimed_by` → null). Unknown branch → `{"claims": []}`.
3. `status_server.py`: route `GET /api/branch/<branch_key_hex>` to it (404 on
   malformed hex). Fixture mode: serve `status_fixture_branch.json` (new
   fixture file) for any branch key.
4. Client: fetch on card expansion only (not per poll; refetch per poll only
   while expanded). Render as a CSS-grid of small squares: green = done,
   worker-tinted (item C hue if merged, else amber) = in-flight, tile-gray =
   unclaimed. Tooltip per cell: `idx`, worker, state.
5. Tests: unit tests for the queue method and `collect_branch_detail`
   (fixtures as in `test_status_model.py`), server test for the new route.
- Acceptance: expanding a fixture branch shows the grid; a
  `complete_candidate` call against a live queue flips its cell green on the
  next poll; full test suite passes.

## Explicitly out of scope (all items)

- Historical time-series (rates over time, sparklines) — would need the model
  to persist history; propose separately if wanted.
- Any control operations (pausing workers, re-prioritizing branches) — the
  status service is strictly read-only.
