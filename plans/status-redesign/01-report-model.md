# Phase 1 — Shared report model and overview

Read `00-overview.md` and the repository root `AGENTS.md` before starting.
Do not begin until issue #92's runtime-path owner is merged and this plan's
references to that module have been updated to its actual public names.

## Goal

Create the presentation-neutral foundation used by every later client:

- one inferred spine-selector parser;
- one normalized request/filter representation;
- one versioned report envelope;
- normalized branch and worker entities;
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
    parse_rich_spine(path: str | None) -> list[dict]
    parse_report_selector(parts: list[str] | str | None) -> ReportSelector
    collect_overview_report(sources: ReportSources,
                            request: ReportRequest | None = None) -> dict
    collect_report(sources: ReportSources, request: ReportRequest) -> dict

And these frozen dataclasses:

    SpineStep(word: str, pattern: str)

    ReportSelector(
        kind: str,                    # root | word | branch | branch_reference
        steps: tuple[SpineStep, ...],
        trailing_word: str | None,
        branch_reference: str | None,
        input_text: str,
    )

    ReportSelector.root() -> ReportSelector

    ReportFilters(
        active_only: bool = False,
        statuses: tuple[str, ...] = (),
        minimum_answer_count: int | None = None,
        maximum_answer_count: int | None = None,
        budget: int | None = None,
        priority: int | None = None,
        sort: str | None = None,
        limit: int | None = None,
    )

    ReportRequest(
        report_kind: str = "overview",
        selector: ReportSelector = ReportSelector.root(),
        tree: bool = False,
        filters: ReportFilters = ReportFilters(),
        include_claims: bool = False,
    )

    ReportSources(
        queue_path: str,
        cache_path: str,
    )

Implement `ReportSelector.root` as a class method returning the normalized
empty selector. Use `dataclasses.field(default_factory=...)` rather than
constructing dataclass instances as function or field defaults. The types
above describe the contract; spell the final definitions in valid Python.

`collect_report` dispatches `overview` in this phase and raises
`ValueError("unsupported report kind: ...")` for anything else. Later phases
extend the dispatcher.

## Selector grammar

`parse_report_selector` accepts either a string or already-tokenized parts.
When given a list, join it with spaces before parsing so CLI and HTTP inputs
share identical behavior.

Normalize words to lowercase and patterns to five characters using
`g`/`y`/`-`. A dot normalizes to `-`. Reject malformed word and pattern
tokens with a message that identifies the token and expected form.

Rules:

1. Empty input produces `kind="root"`.
2. A sole token beginning with `@` produces `kind="branch_reference"`.
   Require 4–40 hexadecimal characters after `@`. Store them lowercase.
3. Otherwise, tokens alternate five-letter word and five-character response
   pattern.
4. If the final token is a word with no following pattern, produce
   `kind="word"`, put all complete pairs in `steps`, and put that word in
   `trailing_word`.
5. If every word has a pattern, produce `kind="branch"` and put every pair in
   `steps`.
6. A missing pattern anywhere except after the final word is invalid.

Required examples:

| Input | Kind | Meaning |
|---|---|---|
| empty | root | operational root |
| `CRANE` | word | inspect CRANE at root |
| `CRANE -y--g ALIBI` | word | inspect ALIBI in branch CRANE -y--g |
| `CRANE -y--g ALIBI g-g--` | branch | branch after both guesses |
| `@8B31` | branch_reference | queued branch digest prefix |

The parser does not inspect databases. Branch-reference resolution belongs to
phase 3.

## Branch reference

Implement `report_model.branch_reference` with the same SHA-1 input semantics
as the existing `erd_search._branch_id`, but return the first 12 hexadecimal
characters rather than four. Do not include `@` in the function result;
renderers add it. Leave the legacy four-character `_branch_id` definition in
`erd_search.py` until phase 3 removes the legacy status renderer; changing it
now would alter an intentionally untouched client and its tests.

In `erd_search.py` import:

    from report_model import (
        WORKER_LIVENESS_SECONDS,
        parse_rich_spine as _parse_spine,
    )

Move `WORKER_LIVENESS_SECONDS` and the body of `_parse_spine` into
`report_model.py`. The aliases keep the legacy renderer working without
duplicating these semantics. `report_model.py` must not import `erd_search`.

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
6. Read `run_meta.epoch` and its matching `telemetry.telemetry_epoch` row for
   epoch label and Git revision. A missing row yields null fields, not source
   failure.
7. Close in `finally`.

The overview includes active and finalizing branches only. Pending/done work
appears in counts and later collection reports, not as thousands of overview
rows.

### Cache collection

Load the answer list once in file order and also retain a cached answer set.
Open `ScoreCache(..., checkpoint_on_close=False)` and call
`erd_report_summary(ERD_ALL, generated_at - 300)`. Close in `finally`.

### Report envelope

Return every top-level key even under partial failure:

    {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "overview",
        "generated_at": generated_at,
        "selector": normalized root selector,
        "filters": normalized filter dictionary,
        "sources": {
            "queue": {"path": ..., "ok": bool, "error": str | null},
            "telemetry": {
                "path": ...,
                "ok": bool,
                "error": str | null,
                "epoch": int | null,
                "label": str | null,
                "git_sha": str | null,
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
| `claim_idx` | int or null |
| `claim_started_at` | int or null |
| `completed_claim_count` | int |
| `current_candidate` | lowercase str or null |
| `current_candidate_is_answer` | bool |
| `current_max_guess_depth` | int or null |
| `current_node_count` | int or null |
| `nodes_per_second` | float or null |
| `descent` | normalized array from `parse_rich_spine` |
| `cache_hit_count` / `cache_miss_count` | int |
| `solved_evaluation_count` / `erd_cutoff_evaluation_count` / `remaining_depth_pruned_evaluation_count` | int |
| `best_guess` | lowercase str or null |
| `best_erd` / `bound_erd` | float or null |

Sort the model's worker array deterministically by numeric worker number when
numeric, followed by nonnumeric `worker_id`. Sticky presentation order is
still a renderer concern.

## Tests

`tests/test_report_model.py` must cover:

1. Selector inference for root, trailing word, complete branch, dot-pattern
   normalization, and branch reference.
2. `CACHE` and `QUEUE` as positional words both infer `kind="word"`.
3. Invalid alternating forms and malformed patterns produce useful errors.
4. Branch-reference stability, 12-character length, and distinct keys.
5. Empty queue/cache produces a JSON-round-trippable overview with every
   envelope key.
6. Unavailable queue and unavailable cache fail independently.
7. One active branch and worker normalize spine, `guess_depth`, answer flags,
   candidate progress, and renamed guess-axis fields correctly.
8. All-gray pattern code 0 produces `guess_depth == 1` in the legacy-spine
   fallback.
9. A cooperative branch normalizes to `active` and contributes to
   `active_cooperative_branch_count`.
10. A finalized branch referenced by a live heartbeat normalizes to
    `finalizing`; one referenced only by a dead heartbeat is not retained.
11. Candidate progress batch lookup handles empty, missing, evaluated, and
    bulk-eliminated candidates.
12. No overview branch object contains `best_max_depth` and no worker object
    contains `cur_max_depth`.

## Acceptance checklist

- [ ] The public API and envelope above exist.
- [ ] Selector inference is independent of CLI, HTTP, and database access.
- [ ] Overview collection performs no per-branch claim-table query.
- [ ] Overview contains no candidate index arrays or unbounded telemetry.
- [ ] Queue/cache failure is partial and JSON serializable.
- [ ] `report_model.py` does not import `erd_search`.
- [ ] Existing terminal status behavior remains operational through aliases.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
