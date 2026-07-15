# Phase 6 — Independent visual modalities

Read phases 00 and 05 first. Requires phase 5 merged.

This is a menu, not one implementation phase. Choose exactly one item per
branch and pull request. Items may land in any order unless a prerequisite is
stated.

Every item:

- uses the existing report contract; if required data is absent, stop and
  propose a model phase rather than adding ad hoc SQL to the server/client;
- keeps `report_client.html` self-contained;
- adds no external runtime dependency;
- preserves semantic identity/change rules;
- works at 390 and 1200 px without horizontal body scroll;
- extends `tests/test_report_client.py` with DOM/behavior assertions and
  temporary screenshot artifacts;
- is presentation only and never mutates queue/cache state.

The Wordle green/yellow/gray palette remains reserved for pattern and
completion semantics. Alert red/amber remain reserved for change, dead, and
stale meanings. New distinctions use shape, texture, labels, opacity, or
worker-specific hues rather than conflicting semantic colors.

## Item A — Candidate sweep density

In expanded branch detail, replace the plain candidate progress line with a
fixed-bucket density track.

- Cover candidate indices `0..candidate_count-1`.
- Compute completed density per bucket from sparse claim rows.
- Do not render completion as one contiguous prefix; claims finish out of
  order.
- Evaluated completion uses solid green.
- Bulk lower-bound elimination uses a green hatched/striped treatment so it
  is visibly a proof rather than worker evaluation.
- In-flight candidates receive worker markers positioned by candidate index.
- Overlapping markers nudge without changing their semantic position.
- Republished candidates receive a small outline/tick, not an alert color.
- Newly completed density flashes according to completion kind.

Acceptance:

- noncontiguous evaluated and bulk-eliminated fixture indices are distinct;
- every live worker marker appears;
- the number of represented candidate positions equals `candidate_count`;
- no `bulk-elimination` pseudo-worker appears.

## Item B — Completion ring

Add a small SVG ring to branch cards and tree branch nodes:

- fraction is `completed_candidate_count / candidate_count`;
- 0 and 100 percent render correctly;
- bulk completion does not change the fraction semantics;
- keep the numeric percentage adjacent for accessibility;
- SVG has an accessible label.

Use one circle with `stroke-dasharray` and the normal completion green.

## Item C — Worker chips and movement

Give each worker persistent visual identity:

- chip label `W<number>`;
- stable hue from the complete `worker_id` hash, not only numeric suffix, so
  nonnumeric IDs do not collide;
- branch cards/tree nodes contain chips for attached workers;
- idle/finalizing/dead workers occupy one “swarm lane”;
- movement to a different `branch_key_hex` pulses at the destination;
- stale/dead state adds border/icon semantics without replacing identity hue.

Clicking a chip opens the same worker detail/navigation as the Workers
report. No duplicated worker-detail implementation.

## Item D — Tree focus and context navigator

The basic collapsible tree already ships in phase 5 because tree is a core
layout. This item improves navigation rather than creating a second tree:

- sticky breadcrumb for the selected context spine;
- compact overview rail showing collapsed sibling/ancestor activity;
- “follow worker” action that expands and scrolls to the worker's current
  branch while preserving the selected root;
- branch-status/phase badges on collapsed nodes;
- a reset-focus action returning to the selected context.

On phones, the overview rail collapses to a single context button. This DOM
scale experience is the required precursor to the landscape plan.

Acceptance:

- following a moving worker changes focus by full branch identity;
- branch filtering never removes the selected context breadcrumb;
- Back restores prior tree focus/collapse state.

## Item E — Candidate grid

Render one grid cell per candidate index in expanded branch detail. Build the
complete visual array client-side from `candidate_count` plus sparse claims.

States:

| State | Rendering |
|---|---|
| unclaimed | tile gray |
| in flight | worker hue with worker label/tooltip |
| evaluated done | solid green |
| bulk eliminated | striped/hatched green |
| republished and currently unclaimed | gray with outline/tick |
| provenance unknown | neutral patterned cell with explicit tooltip |

Tooltips/accessible labels include candidate index, state, worker, bundle,
completion kind, and republish count when known.

The client requests `claims=1` only while the grid disclosure is expanded.
At 12,972 candidates, use CSS grid and a bounded number of DOM buckets if
individual cells do not meet the phone performance budget. If aggregation is
needed, each bucket must support drill-down to individual candidate cells;
document and test the threshold.

Acceptance:

- completing a candidate flips only its cell;
- bulk elimination never renders as a worker;
- expanding/collapsing starts/stops claim-detail polling;
- interaction remains responsive on an iPhone-sized viewport at maximum
  candidate count.

## Item F — Finalization and reuse ledger

Add a compact visual explanation of why recent work ended and whether a cut
caused later rework:

- exact, cut, and loss shown as labeled shapes/icons, not color alone;
- evaluated versus bulk-done candidate composition;
- nodes, wall time, bundles, ceiling, budget, and epoch;
- cut-reuse misses connected/listed beneath the cut that supplied the
  insufficient bound when identity/time information permits;
- explicit “no reuse misses observed in this bounded window” versus “data not
  collected/available.”

This is not a time-series chart. It visualizes the bounded branch-detail and
hotspot data already returned.

Acceptance:

- fixtures distinguish exact/cut/loss without relying on color;
- a cut and its reuse miss are visibly associated where the data supports it;
- the UI never implies absence outside the reported epoch/window.

## Out of scope

- Historical sparklines or durable time-series storage.
- Full live-work semantic zoom (phase 7).
- Control actions.
- New queue/cache schema added merely for decoration.
