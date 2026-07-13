# Phase 3b — Collection reports and live queue tree

Read phases 00–03a and `AGENTS.md` first. Requires phase 3a merged.

## Goal

Add bounded collection exploration over the semantic report model:

- shared lifecycle, answer-count, budget, priority, sort, and limit filters;
- queue, worker, and cache reports;
- live queue topology as an optional layout over compatible selections;
- SQL-backed filtering and summaries before pagination.

Legacy inspection commands remain available until phase 3d.

## Files touched

- `report_model.py`
- `report_terminal.py`
- `erd_queue.py`
- `cache_sqlite.py`
- `erd_search.py`
- `tests/test_report_model.py`
- `tests/test_report_terminal.py`
- `tests/test_report_objects.py`

## Filter and request model

Add the frozen dataclass:

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

Extend `ReportRequest` with:

    tree: bool = False
    filters: ReportFilters = ReportFilters()
    worker_id: str | None = None

Use `dataclasses.field(default_factory=...)` for the filter default.

## CLI additions

Extend `view` with:

    --tree
    --queue
    --workers
    --worker N
    --cache

    --active-only
    --status STATUS          repeatable
    --minimum-answer-count N
    --maximum-answer-count N
    --budget N
    --priority N
    --sort FIELD
    --limit N

Keep all phase 2 and 3a options. `--queue`, `--workers`, `--worker`, and
`--cache` are mutually exclusive. `--worker N` selects the workers report and
filters it to that worker. A positional selector may scope any compatible
kind. Reject meaningless combinations with a specific argparse error.

`--active-only` and explicit `--status` are mutually exclusive.
`--active-only` normalizes to lifecycle `active` and remains true in the
echoed filter object.

## Dispatch and scoping

Extend `collect_report`:

| Explicit kind | Selector | Collector |
|---|---|---|
| none | root, no `--tree` | overview |
| none | word | word |
| none | branch/reference | branch |
| none | any + `--tree` | live queue tree scoped by selection |
| `--queue` | optional, with optional `--tree` | queue inventory scoped by selector |
| `--workers` / `--worker` | optional, with optional `--tree` | workers scoped by selector or placed on live queue topology |
| `--cache` | optional | cache summary/coverage scoped by selector |

The envelope's `report_kind` is the inferred or explicit domain kind. Echoed
request state records tree layout; there is no `tree` domain object. Reject
`--tree` with `--cache` because a cache report has no live work topology.

Collection scoping is consistent:

- root selector: complete report population;
- trailing-word selector: derived response branches for word coverage, or
  only matching extant queue rows where descendants are included;
- branch selector/reference: selected branch plus recorded queued descendants.

A singular branch report still describes only its selected branch. Scoping
never uses cache rows to create descendants.

## Filter behavior

Normalize raw queue status through phase 1's lifecycle vocabulary before
filtering. Push filters into SQL wherever practical; do not load the complete
queue merely because legacy helpers do.

Collection reports return:

    "summary": counts over the unpaginated selected population
    "rows": filtered, sorted, limited rows
    "matched_rows": count before limit

When selected context is represented by queue-derived topology, retain it
even when it does not match the row filter and mark `is_context=true`.
Selector context remains in root metadata when no queue node is available.

`--active-only` applies to queue rows, word response branches, tree
descendants, and worker-associated branches. It does not alter singular
branch detail, global cache totals, or the selected context root.

## Queue query helpers

Add bounded helpers:

    ERDQueue.report_queue_rows(filters, sort, limit) -> dict
    ERDQueue.report_tree_rows(spine_prefix, filters, sort, limit) -> dict

`report_queue_rows` returns summary counts, matched count, and normalized
source rows. It may use a SQL `UNION ALL`/CTE over user pending rows and
cooperative active rows, but must not duplicate a user branch that also has
active state.

`report_tree_rows` returns extant rows with recorded spines below the prefix.
Any maximum-level parameter in this reporting path is named
`max_guess_depth`. Do not delete lower-level queue methods used by swarm
behavior.

## Cache query helpers

Add:

    ScoreCache.report_recent_rows(policy, since, limit) -> list[dict]
    ScoreCache.report_cache_distributions(policy) -> dict

`report_cache_distributions` returns counts grouped by
`max_remaining_depth`, solve budget, taint, and exact/loss state for the
current answer list. Name every returned count by what it counts. No cache
schema change belongs in this phase.

## Tree layout

Tree collection is selected by `--tree`, not a separate report noun. Its
authoritative topology is extant queue rows and their recorded spines. Nodes
may represent queue rows or structural ancestors required by those spines.
Cache rows never create nodes or edges; cache state may annotate a queued
branch.

Return a flat, identity-addressable node array:

    {
        "root": normalized selector context,
        "topology_source": "queue",
        "tree_available": bool,
        "unavailable_reason": str | null,
        "nodes": [
            {
                "node_id": stable semantic path string,
                "parent_node_id": str | null,
                "step": {"word": str, "pattern": str} | null,
                "branch_key_hex": str | null,
                "branch_reference": str | null,
                "lifecycle": str | null,
                "answer_count": int | null,
                "guess_depth": int,
                "worker_count": int,
                "completed_candidate_count": int | null,
                "candidate_count": int | null,
                "is_context": bool,
            }
        ],
    }

Semantics:

- Root: extant queue rows and recorded spines form the tree.
- Word: the word is selector context; nodes contain only its extant queued
  response branches and descendants.
- Branch: include context when a matching row or descendant spine establishes
  it, then include recorded descendants.
- A branch with no matching row or descendant returns
  `tree_available=false`, `unavailable_reason="no extant queue topology"`,
  and no nodes. Do not synthesize a root from cache.
- Missing legacy spine segments are explicit unknown nodes at known
  `guess_depth`; never fabricate guesses.

Sort siblings deterministically by word, pattern, then branch key. Sticky
ordering across refreshes remains a client concern.

## Queue, worker, and cache reports

### `--queue`

Default output is summary plus filtered inventory. Sort choices are:

    default, age, size, workers, priority, nodes, slowest

`--tree` switches the selected rows to tree layout.

### `--workers` / `--worker N`

Show live, idle, finalizing, stale, and dead workers. A branch selector limits
workers to that branch and descendants; a word selector limits workers to its
extant response branches. Detail includes absolute `guess_depth`, current
candidate, held bundle/claim, node rate, cache/prune counters, and live
descent. In tree layout, workers annotate queue-derived nodes.

### `--cache`

- Root: exact/loss totals, recent throughput, reusable
  `max_remaining_depth`/budget distributions, and bounded recent rows.
- Word: response-group cache coverage without queue columns.
- Branch: normalized cache detail for that branch.

Never label a transient cut as cached.

## Terminal rendering

Add renderers for each collection kind and tree layout. Text and JSON consume
the same envelopes. Phase 2 watch sessions retain sticky order by full
identity and preserve selected context through refresh.

## Tests

Required coverage:

1. `view --queue`, `--workers`, `--worker`, and `--cache` dispatch correctly;
   positional `QUEUE` and `CACHE` remain word selectors.
2. `--active-only` exactly matches normalized active lifecycle and excludes
   finalizing.
3. Filtering occurs before limit and summaries count the unpaginated
   population.
4. Word, branch, and root trees derive only from extant queue rows and their
   spines, have stable parents, and expose absolute `guess_depth`.
5. Tree context survives filtering while unmatched descendants do not.
6. Cache-only branch detail remains available but its tree is unavailable;
   word trees do not synthesize unqueued response groups.
7. Missing legacy spine segments never fabricate guesses.
8. Queue rows are not duplicated across user and cooperative state.
9. Cache distributions obey existing reuse semantics.
10. Incompatible report/option combinations fail with specific errors.
11. All legacy inspection tests remain green.

## Acceptance checklist

- [ ] Collection filters have one normalized meaning across report kinds.
- [ ] Queue, worker, and cache reports are bounded and presentation-neutral.
- [ ] Tree is an option over compatible reports, not a report noun.
- [ ] Tree nodes derive only from extant queue rows and recorded spines.
- [ ] Cache-only solutions are never reconstructed as topology.
- [ ] Legacy inspection commands remain available.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
