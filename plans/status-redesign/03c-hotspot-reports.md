# Phase 3c — Bounded telemetry and hotspot reports

Read phases 00–03b and `AGENTS.md` first. Requires phase 3b merged.

## Goal

Add bounded historical branch detail and hotspot reports without changing
swarm behavior or the phone-shared cache schema.

## Files touched

- `report_model.py`
- `report_terminal.py`
- `erd_queue.py`
- `erd_search.py`
- `tests/test_report_model.py`
- `tests/test_report_terminal.py`
- `tests/test_report_objects.py`
- `tests/test_queue_visibility.py`

## Request and CLI extensions

Extend `ReportRequest` with:

    hotspot_field: str | None = None
    epoch: int | None = None
    since_seconds: int | None = None
    sample_size: int | None = None

Extend `view` with:

    --hotspots
    --by FIELD
    --epoch N
    --since-seconds N
    --sample-size N

`--hotspots` joins the mutually exclusive report-kind flags from phase 3b.
Reject `--tree --hotspots`: a historical ranking has no live work topology.

## Branch telemetry extension

Extend phase 3a branch detail with:

    "bundle_summary": {...} | null
    "recent_finalizations": [...]
    "cut_reuse_misses": [...]

Finalizations and cut-reuse misses default to the newest five rows and obey
`--limit` when supplied. Normalize outcome as `exact`, `cut`, or `loss` and
preserve ceiling, budget, nodes, wall time, bundle count, censored units,
epoch, and evaluated versus bulk-completed candidate counts.

## Hotspot populations

`--hotspots --by` choices have explicit populations:

| `--by` | Population/result |
|---|---|
| `nodes`, `age`, `size`, `workers`, `priority`, `slowest` | current queue branches |
| `evaluated-candidates`, `bulk-completed-candidates` | bounded recent `branch_finalize_log` branch rows |
| `cut-reuse` | bounded recent `cut_reuse_misses` branch rows |
| `coordination` | workload buckets from bounded recent `claim_telemetry` |

`claim_telemetry` has no branch key. Coordination hotspots rank
answer-count/worker-count workload buckets, not branches, and reject a spine
selector rather than pretending attribution exists.

Phase 3b filters apply to the current-queue hotspot kinds. Historical
finalization, cut-reuse, and coordination populations do not carry a current
branch status or phase, so reject branch filters for those kinds rather than
applying present-day queue state retroactively.

Historical options:

- `--epoch N` defaults to the current epoch.
- `--since-seconds N` defaults to 3600.
- `--sample-size N` defaults to 50,000 and is capped at 1,000,000.
- `--limit N` defaults to 10 for hotspots.

Return population, epoch/window, sample size, and truncation metadata so an
approximation cannot be mistaken for complete history.

## Telemetry index migrations

Add idempotent Linux queue migrations in `ERDQueue._migrate()` creating these
indexes for bounded report queries:

    telemetry.branch_finalize_log(branch_key, recorded_at)
    telemetry.cut_reuse_misses(branch_key, recorded_at)
    telemetry.claim_telemetry(epoch, id)

These are indexes over existing attached telemetry tables, not new tables or
columns and not phone-shared cache migrations. Give each index an explicit,
self-describing name.

Add bounded helpers:

    ERDQueue.report_branch_telemetry(branch_key, limit) -> dict
    ERDQueue.report_hotspots(field, epoch, since, sample_size, limit,
                             spine_prefix=None) -> dict

`report_hotspots` selects the explicit population above and returns stable
row or bucket identities. It rejects unsupported spine attribution for
coordination data. No helper scans an unbounded telemetry population.

## Terminal rendering

Add the hotspot renderer and bounded telemetry sections to branch detail.
Always label population, epoch/window, sample size, and truncation. A cut is
transient coordination history, never an exact cache state.

Hotspot identities and metrics render as an adaptive table. Identity and the
primary ranked metric are essential; secondary metrics disappear by declared
priority as width tightens. Population/window/sample metadata wraps at field
boundaries. Branch telemetry uses the same phase 2 helper and remains legible
at 50 columns without clipping outcome, epoch, or evaluated-versus-bulk
completion counts.

## Tests

Required coverage:

1. Migrations create the three named indexes idempotently on existing tables.
2. Branch telemetry is bounded and distinguishes exact, cut, and loss.
3. Evaluated and bulk-completed counts remain distinct.
4. Coordination hotspots identify workload buckets and reject spine scope.
5. Every historical ranking labels its population, epoch/window, sample size,
   and truncation.
6. Defaults and upper bounds for time window, sample, and limit are enforced.
7. Hotspot queries use the new indexes and do not scan unbounded history.
8. `--tree --hotspots` is rejected.
9. Hotspot and branch-telemetry text has exact 50–120-column coverage,
   including rows whose secondary metrics must be hidden.
10. Existing queue migration and visibility tests remain green.

## Acceptance checklist

- [ ] Historical branch detail and hotspot populations are explicitly bounded.
- [ ] Telemetry changes are index-only Linux queue migrations.
- [ ] Exact, cut, and loss remain distinct.
- [ ] Coordination data is never falsely attributed to a branch.
- [ ] Hotspot and telemetry terminal tables remain operational at 50 columns.
- [ ] Legacy inspection commands remain available.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
