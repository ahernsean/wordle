# Phase 1 — Shared report model and overview

Read `00-overview.md` and the repository root `AGENTS.md` before starting.
This phase requires the shared `runtime_paths.py` owner described in phase 00.
If it does not exist yet, create it here with current paths; do not move
runtime artifacts as part of this phase.

## Goal

Create the presentation-neutral foundation used by every later client:

- one importable owner for database and word-list paths;
- one versioned report envelope;
- one minimal overview request;
- normalized branch and worker entities;
- normalized lifecycle and identity semantics;
- a lightweight operational overview collector.

The model prints nothing and contains no terminal, HTTP, or HTML logic.

## Non-goals

- No `erd_search.py view` command yet.
- No queue inventory, word coverage, branch detail, tree, cache-detail, or
  hotspot collectors yet.
- No percentages, ETA, abbreviations, colors, truncation, or sticky display
  order.
- No candidate-claim arrays or telemetry-table scans in the overview.

## Files touched

- `report_model.py` — new
- `runtime_paths.py` — new if the shared path owner does not already exist
- `erd_queue.py` — bounded batch read helper described below
- `cache_sqlite.py` — cache-summary read helper described below
- `erd_search.py` — move and alias the two shared definitions below
- `tests/test_report_model.py` — new
- `tests/test_status_sections.py` — must pass unmodified

## Public API

`report_model.py` exports:

    SCHEMA_VERSION = 1
    WORKER_LIVENESS_SECONDS = 30

    branch_reference(branch_key: bytes) -> str
    parse_rich_spine(path: str | None) -> list[RichSpineStep]
    normalize_worker_descent(parsed_path: list[RichSpineStep],
                             answer_set: set[str]) -> list[dict]
    collect_overview_report(sources: ReportSources,
                            request: ReportRequest | None = None) -> dict
    collect_report(sources: ReportSources, request: ReportRequest) -> dict

And these frozen dataclasses:

    ReportRequest(
        report_kind: str = "overview",
    )

    ReportSources(
        queue_path: str,
        cache_path: str,
        answer_list_path: str,
        guess_list_path: str,
        telemetry_path: str | None = None,
    )

    ReportSources.defaults() -> ReportSources

`collect_report` dispatches `overview` in this phase and raises
`ValueError("unsupported report kind: ...")` for anything else. Later phases
extend the dispatcher.

`RichSpineStep` is the legacy-compatible tuple:

    tuple[int | None, str | None, str | None, str]

Its fields are `(guess_depth, word, pattern, answer_count_text)`. Keeping the
four-tuple contract lets phase 1 alias the parser into the legacy renderer
without adapting its call sites.

`ReportSources.defaults()` reads all defaults from `runtime_paths.py`.
`runtime_paths.py` owns current database, telemetry, answer-list, and
guess-list paths. Export full names such as `DEFAULT_ANSWER_LIST_PATH` and
`DEFAULT_GUESS_LIST_PATH`; `erd_search.py` may import those as its legacy
module-level aliases so existing callers and tests continue to work. Phase 1
does not move any of the files.

## Branch reference and legacy transition helpers

Implement `report_model.branch_reference` with the same SHA-1 input semantics
as the existing `erd_search._branch_id`, but return the first 12 hexadecimal
characters rather than four. Do not include `@` in the function result;
renderers add it. Leave the legacy four-character `_branch_id` definition in
`erd_search.py` until phase 3d removes the legacy status renderer; changing it
now would alter an intentionally untouched client and its tests.

In `erd_search.py` import:

    from report_model import (
        WORKER_LIVENESS_SECONDS,
        parse_rich_spine as _parse_spine,
    )

Move `WORKER_LIVENESS_SECONDS` and the body of `_parse_spine` into
`report_model.py`. The aliases keep the legacy renderer working without
duplicating these semantics. The moved `parse_rich_spine` retains the exact
four-tuple return contract. `normalize_worker_descent` converts those tuples
to presentation-neutral dictionaries for report worker objects; it is not
aliased into the legacy renderer. `report_model.py` must not import
`erd_search`.

## Presentation-neutral semantics

The model normalizes business meaning that every client must agree on:

- branch lifecycle;
- `guess_depth`;
- `best_max_remaining_depth` from the legacy
  `active_branches.best_max_depth` column;
- `current_max_guess_depth` from the legacy heartbeat
  `cur_max_depth` column;
- cooperative status;
- answer flags;
- stable identities.

The model does not calculate:

- heartbeat age strings;
- percentages or rates expressed as percentages;
- ETA;
- duration formatting;
- human abbreviations;
- layout or ordering that exists only for display stability.

`is_live` is a normalized lifecycle fact and may be included, calculated from
`generated_at - updated_at <= WORKER_LIVENESS_SECONDS`. Keep `updated_at` so
renderers can calculate the displayed age.

### Lifecycle vocabulary

SQLite uses different status values for user-queued and cooperative work.
Normalize them before producing branch objects:

| Report lifecycle | Source state |
|---|---|
| `pending` | `pending_branches.status = 'pending'` |
| `active` | user `in_progress` or cooperative `active_branches.status = 'open'` |
| `finalizing` | finalized branch still referenced by a live heartbeat |
| `done` | `pending_branches.status = 'done'` |

Phase 3 adds `unqueued` for a derived word response branch absent from queue
state. Cache state is independent of lifecycle and uses `exact`, `loss`,
`missing`, or `not_applicable`. A cut is transient telemetry/coordination
state, not an exact cache state.

## Bounded queue helper

Add:

    ERDQueue.candidate_progress_by_branch_keys(branch_keys: list[bytes]) -> dict

It returns:

    {
        branch_key_bytes: {
            "completed_candidate_count": int,
            "bulk_completed_candidate_count": int,
        },
        ...
    }

Use one grouped query over `candidate_claims` for done counts and one bounded
lookup of `active_branches.bulk_done_candidates` for the supplied keys. Empty
input returns `{}` without SQL containing an empty `IN ()`. This helper must
not import the report module.

## Cache summary helper

Add:

    ScoreCache.erd_report_summary(policy: str, recent_since: int) -> dict

For the current `answer_list_id`, return:

    {
        "exact_branch_count": int,
        "recent_exact_branch_count": int,
        "loss_branch_count": int,
    }

Count `branch_best_by_policy` rows for the first two values and
`branch_loss_by_policy` rows for the third. The helper owns the SQL so the
report model does not reach into `ScoreCache._conn`.

## Overview assembly

`collect_overview_report` takes one `generated_at = int(time.time())` and
attempts queue and cache collection independently. A failure in one source
must not erase data from the other.

### Queue collection

Using one `ERDQueue` connection:

1. Read queue status counts.
2. Read open active branches with `branches_in_progress()`.
3. Read heartbeats and worker counts.
4. Find heartbeat branch keys absent from the open rows; load those active
   rows with `active_branches_by_keys`. Retain only finalized rows referenced
   by a live heartbeat. These normalize to `finalizing`.
5. Fetch candidate progress for all retained branch keys with the new batch
   helper.
6. Read `run_meta.epoch` and its matching unqualified `telemetry_epoch` row
   from the main queue database for epoch label and Git revision. A missing
   row yields null fields, not source failure.
7. Close in `finally`.

The overview includes active and finalizing branches only. Pending/done work
appears in counts and later collection reports, not as thousands of overview
rows.

### Cache collection

Load the answer list from `sources.answer_list_path` once in file order and
also retain a cached answer set. Open
`ScoreCache(..., checkpoint_on_close=False)` and call
`erd_report_summary(ERD_ALL, generated_at - 300)`. Close in `finally`.

### Report envelope

Return every top-level key even under partial failure:

    {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "overview",
        "generated_at": generated_at,
        "selector": null,
        "filters": {},
        "sources": {
            "queue": {
                "path": ...,
                "ok": bool,
                "error": str | null,
                "epoch": int | null,
                "label": str | null,
                "git_sha": str | null,
            },
            "telemetry": {
                "path": ...,
                "ok": bool,
                "error": str | null,
            },
            "cache": {"path": ..., "ok": bool, "error": str | null},
        },
        "data": {
            "queue_counts": {
                "pending_branch_count": int,
                "active_user_branch_count": int,
                "active_cooperative_branch_count": int,
                "finalizing_branch_count": int,
                "done_branch_count": int,
            },
            "cache_summary": {
                "exact_branch_count": int,
                "recent_exact_branch_count": int,
                "loss_branch_count": int,
            },
            "worker_totals": {
                "cache_hit_count": int,
                "cache_miss_count": int,
                "solved_evaluation_count": int,
                "erd_cutoff_evaluation_count": int,
                "remaining_depth_pruned_evaluation_count": int,
            },
            "branches": [...],
            "workers": [...],
        },
    }

Only live workers contribute to `worker_totals`. All heartbeat rows remain in
`workers` so dead/stale state can be shown.

Queue epoch metadata lives under `sources.queue` because `run_meta` and
`telemetry_epoch` are both main-queue tables. `sources.telemetry` describes
only the health and path of the attached telemetry file. If attaching that
file prevents the queue connection from opening, both source entries report
the failure; epoch metadata is never read from the attached database.

## Normalized branch object

Every overview branch has:

| Key | Type |
|---|---|
| `branch_reference` | str, 12 hex without `@` |
| `branch_key_hex` | str |
| `lifecycle` | `active` or `finalizing` |
| `raw_status` | source status string |
| `answer_count` | int |
| `candidate_count` | int |
| `completed_candidate_count` | int |
| `bulk_completed_candidate_count` | int |
| `priority` | int |
| `is_cooperative` | bool |
| `source_word` | lowercase str or null |
| `source_pattern` | normalized pattern or null |
| `best_guess` | lowercase str or null |
| `best_guess_is_answer` | bool |
| `best_erd` | float or null |
| `best_max_remaining_depth` | int or null |
| `budget` | int or null |
| `guess_depth` | int |
| `spine` | array of `{"word", "pattern", "word_is_answer"}` |
| `worker_count` | int |
| `created_at` | int or null |
| `search_node_count` | int |
| `ceiling` | float or null |

`guess_depth` is the number of stored spine pairs. For a legacy row without a
spine, use 1 only when both `source_word` is present and `source_pattern is
not None`; pattern code 0 is valid.

## Normalized worker object

Every worker has:

| Key | Type |
|---|---|
| `worker_id` / `worker_number` | str |
| `pid` | int |
| `updated_at` | int |
| `is_live` | bool |
| `branch_reference` / `branch_key_hex` | str or null |
| `candidate_index` | int or null |
| `claim_started_at` | int or null |
| `completed_claim_count` | int |
| `current_candidate` | lowercase str or null |
| `current_candidate_is_answer` | bool |
| `current_max_guess_depth` | int or null |
| `current_node_count` | int or null |
| `nodes_per_second` | float or null |
| `descent` | normalized array from `normalize_worker_descent` |
| `cache_hit_count` / `cache_miss_count` | int |
| `solved_evaluation_count` / `erd_cutoff_evaluation_count` / `remaining_depth_pruned_evaluation_count` | int |
| `best_guess` | lowercase str or null |
| `best_erd` / `bound_erd` | float or null |

Sort the model's worker array deterministically by numeric worker number when
numeric, followed by nonnumeric `worker_id`. Sticky presentation order is
still a renderer concern.

## Tests

`tests/test_report_model.py` must cover:

1. `parse_rich_spine` preserves the legacy four-tuple contract and the alias
   keeps all legacy spine tests unchanged.
2. `normalize_worker_descent` produces dictionaries without changing the
   tuple parser.
3. Branch-reference stability, 12-character length, and distinct keys.
4. Empty queue/cache produces a JSON-round-trippable overview with every
   envelope key.
5. Unavailable queue and unavailable cache fail independently.
6. Queue epoch metadata comes from the main queue database and is reported
   under `sources.queue`; telemetry source health has no epoch fields.
7. Custom answer-list and guess-list paths flow through `ReportSources`
   without importing `erd_search`.
8. One active branch and worker normalize spine, `guess_depth`, answer flags,
   candidate progress, and renamed guess-axis fields correctly.
9. The normalized worker uses `candidate_index`; no report worker key is
   named `claim_idx`.
10. All-gray pattern code 0 produces `guess_depth == 1` in the legacy-spine
   fallback.
11. A cooperative branch normalizes to `active` and contributes to
   `active_cooperative_branch_count`.
12. A finalized branch referenced by a live heartbeat normalizes to
    `finalizing`; one referenced only by a dead heartbeat is not retained.
13. Candidate progress batch lookup handles empty, missing, evaluated, and
    bulk-eliminated candidates.
14. No overview branch object contains `best_max_depth` and no worker object
    contains `cur_max_depth`.

## Acceptance checklist

- [ ] The public API and envelope above exist.
- [ ] All default paths have one importable owner; word-list sources are
      explicit and no runtime artifact is moved by this phase.
- [ ] Legacy rich-spine tuple callers remain compatible while report descent
      is normalized separately.
- [ ] Overview collection performs no per-branch claim-table query.
- [ ] Overview contains no candidate index arrays or unbounded telemetry.
- [ ] Queue/cache failure is partial and JSON serializable.
- [ ] `report_model.py` does not import `erd_search`.
- [ ] Existing terminal status behavior remains operational through aliases.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
