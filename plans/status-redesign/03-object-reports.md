# Phase 3 — Semantic exploration and object reports

Read phases 00–02 and `AGENTS.md` first. Requires phase 2 merged.

## Goal

Complete the terminal reporting system:

- add semantic work exploration and queue, worker, cache, and hotspot reports;
- make live queue topology an optional layout over compatible reports;
- provide consistent bounded collection controls;
- add on-demand claims, answer words, bundle, finalization, and cut-reuse
  detail;
- remove the legacy read-only command surfaces.

After this phase, all terminal inspection flows through `erd_search.py view`.

## Files touched

- `report_model.py`
- `report_terminal.py`
- `erd_queue.py`
- `cache_sqlite.py`
- `erd_search.py`
- `SWARM.md`
- `AGENTS.md`
- `tests/test_report_model.py`
- `tests/test_report_terminal.py`
- `tests/test_report_objects.py` — new
- `tests/test_queue_visibility.py`
- `tests/test_status_sections.py` — delete after equivalent report-terminal
  coverage is present

No engine or swarm-worker behavior changes in this phase.

## Request, selector, and filter model

Extend `report_model.py` with:

    parse_report_selector(parts: list[str] | str | None) -> ReportSelector

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

Extend the phase 1 request. A default request still produces the operational
overview through automatic root inference:

    ReportRequest(
        report_kind: str = "auto",
        selector: ReportSelector = ReportSelector.root(),
        tree: bool = False,
        filters: ReportFilters = ReportFilters(),
        worker_id: str | None = None,
        include_claims: bool = False,
        include_answers: bool = False,
        hotspot_field: str | None = None,
        epoch: int | None = None,
        since_seconds: int | None = None,
        sample_size: int | None = None,
    )

Use `dataclasses.field(default_factory=...)` for dataclass instance defaults.
`ReportSelector.root()` returns the normalized empty selector.

### Selector grammar

`parse_report_selector` accepts a string or already-tokenized parts. When
given a list, join it with spaces before parsing so CLI and HTTP use identical
behavior. Normalize words to lowercase and patterns to five characters using
`g`/`y`/`-`; a dot normalizes to `-`.

Rules:

1. Empty input selects the root.
2. A sole `@` token followed by 4–40 hexadecimal characters selects a queued
   branch reference. Normalize the digest prefix to lowercase.
3. Otherwise, tokens alternate a five-letter word and a five-character
   response pattern.
4. A final word without a pattern selects that word within the preceding
   branch context.
5. A final response pattern selects the branch reached by the complete spine.
6. A missing pattern anywhere except after the final word is invalid.

| Input | Inferred selection |
|---|---|
| empty | operational root |
| `CRANE` | word CRANE at root |
| `CRANE -y--g ALIBI` | word ALIBI within branch CRANE -y--g |
| `CRANE -y--g ALIBI g-g--` | branch after both guesses |
| `@8B31` | queued branch digest prefix |

Reject malformed tokens with a message that identifies the token and expected
form. Parsing never inspects a database; reference and semantic-spine
resolution happen during collection.

## Final CLI

Extend `view` with positional `SPINE` using `nargs="*"` and:

    --tree
    --queue
    --workers
    --worker N
    --cache
    --hotspots
    --by FIELD

    --active-only
    --status STATUS          repeatable
    --minimum-answer-count N
    --maximum-answer-count N
    --budget N
    --priority N
    --sort FIELD
    --limit N

    --claims
    --answers
    --epoch N
    --since-seconds N
    --sample-size N

Keep phase 2's output, watch, color, and path flags.

The report-kind flags `--queue`, `--workers`, `--worker`, `--cache`, and
`--hotspots` are mutually exclusive. `--worker N` selects the workers report
and filters to that worker. A positional selector may scope any compatible
kind. Reject meaningless combinations with a specific argparse error rather
than ignoring an option.

`--active-only` and explicit `--status` are mutually exclusive.
`--active-only` normalizes to lifecycle `active` in the report request and
remains true in the echoed filter object.

When a dash-leading response pattern would confuse argparse, the documented
forms are one quoted selector string or dot-gray syntax. Preserve support for
`--` before positional selector parts.

## Dispatch and inference

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
| `--hotspots` | optional | bounded hotspot report scoped where supported |

The envelope's `report_kind` is the inferred domain kind. The echoed request
state records tree layout; there is no `tree` domain object.
Reject `--tree` with `--cache` or `--hotspots`; neither report has live work
topology to render.

## Selector resolution

Add presentation-neutral resolver functions to `report_model.py`:

    resolve_selector_branch(selector, all_answers, score_cache) -> ResolvedBranch
    resolve_branch_reference(queue, digest_prefix) -> bytes

`ResolvedBranch` contains the selected answer words, encoded branch key,
normalized complete spine, and optional trailing word.

For a semantic spine, start with all answers and apply each guess/pattern
response group in order through `ResponseCache`. An empty response group is a
valid resolved branch with zero answers; report it rather than treating it as
parse failure.

For `@digest-prefix`:

1. Search user and cooperative queue rows.
2. Hash every candidate branch key using the same SHA-1 function.
3. Require exactly one prefix match.
4. Zero matches is not found; multiple matches returns a disambiguation error
   listing 12-character references and spines.

Digest references are queue-local conveniences. A cache-only branch must be
selected semantically because scanning the entire cache to resolve a digest
is prohibited.

### Selector scoping

Collection reports interpret selectors consistently:

- root selector: the complete report population;
- trailing-word selector: derived response branches for word coverage, or
  only matching extant queue rows where a report includes descendants;
- branch selector/reference: the selected branch plus recorded queued
  descendants.

A singular inferred branch detail still describes only the selected branch.
Scoping never claims that unrecorded cache rows have a known parent spine or
uses them to create descendants.

## Filter behavior

Normalize raw queue status into the lifecycle table in phase 1 before
filtering. Push filters into SQL wherever practical; do not load the whole
queue and filter it in Python merely because the old helpers do.

Collection reports return:

    "summary": counts over the unpaginated selected population
    "rows": filtered, sorted, limited rows
    "matched_rows": count before limit

When the selected context is represented by queue-derived topology, retain it
even when it does not match the row filter. Mark it `is_context=true` so
renderers do not mistake it for a match. Selector context remains in the
report's `root` metadata even when no queue-derived node is available.

`--active-only` applies to:

- queue rows;
- word response branches;
- descendants in tree layout;
- worker-associated branches;
- active/historical hotspot kinds where lifecycle exists.

It does not alter a singular branch detail lookup, global cache totals, or
the selected context root.

## Queue query helpers

Replace Python-wide queue filtering with bounded helpers:

    ERDQueue.report_queue_rows(filters, sort, limit) -> dict
    ERDQueue.report_tree_rows(spine_prefix, filters, sort, limit) -> dict
    ERDQueue.resolve_branch_reference(digest_prefix) -> list[dict]

`report_queue_rows` returns summary counts, matched count, and normalized
source rows needed by the model. It may use a SQL `UNION ALL`/CTE over user
pending rows and cooperative active rows, but it must preserve the rule that
the same user branch is not duplicated when it also has active state.

`report_tree_rows` returns rows with recorded spines below the prefix. Any
maximum-level parameter in the existing helper must be named
`max_guess_depth` throughout this reporting path.

Do not delete lower-level queue methods still used by swarm behavior.

## Cache query helpers

Add read-only helpers to `ScoreCache`:

    report_branch_state(branch_key, policy, budget=None) -> dict
    report_branch_states(branch_keys, policy, budget=None) -> dict
    report_recent_rows(policy, since, limit) -> list[dict]
    report_cache_distributions(policy) -> dict

`report_branch_state` normalizes:

    exact
    loss
    missing
    not_applicable

An exact state includes best guess, ERD, `max_remaining_depth`, solve budget,
taint, and updated timestamp. A loss state includes the budget validity
available from `branch_loss_by_policy`. Use existing cache-reuse semantics;
do not invent a second interpretation in the reporting layer.

`report_branch_states` must batch response-group coverage rather than making
one connection/query setup per branch. Up to 242 keyed lookups is the normal
word-report workload.

`report_cache_distributions` returns counts grouped by
`max_remaining_depth`, solve budget, taint, and exact/loss state for the
current answer list. Name every returned count by what it counts.

No cache schema change belongs in this reporting phase.

## Word report

For a trailing-word selector:

1. Resolve the preceding patterned spine to `branch_words`.
2. Partition those answers by the trailing word.
3. Include every nonempty response group. Groups with fewer than two answers
   have cache state `not_applicable`.
4. Batch lookup queue lifecycle and cache state by encoded branch key.
5. Attach active worker counts and branch references.

Payload:

    {
        "word": "alibi",
        "word_is_answer": bool,
        "context": resolved branch identity/spine/answer count,
        "response_group_counts": {
            "response_group_count": int,
            "trivial_response_group_count": int,
            "queued_response_group_count": int,
            "active_response_group_count": int,
            "exact_response_group_count": int,
            "loss_response_group_count": int,
            "missing_response_group_count": int,
        },
        "response_groups": [
            {
                "pattern": str,
                "answer_count": int,
                "branch_reference": str,
                "branch_key_hex": str,
                "lifecycle": str,
                "priority": int | null,
                "worker_count": int,
                "cache_state": str,
                "best_guess": str | null,
                "best_erd": float | null,
                "max_remaining_depth": int | null,
                "updated_at": int | null,
            }
        ],
    }

`--answers` adds `answer_words` to each returned response group after filters
and limit. Without it, do not send thousands of repeated words.

Text renders one combined queue/cache table. JSON uses the same payload.

## Branch report

A branch report joins its semantic identity across queue, telemetry, workers,
and cache:

    {
        "branch": {
            identity, spine, guess_depth, budget, answer_count,
            optional answer_words
        },
        "queue": {
            lifecycle, raw statuses, priority, candidate progress,
            running best, ceiling, nodes, created/finalized timestamps
        } | null,
        "cache": normalized cache state,
        "workers": [...],
        "bundle_summary": {...} | null,
        "republished_candidates": [...],
        "recent_finalizations": [...],
        "cut_reuse_misses": [...],
        "claims": [...] | null,
    }

`--answers` controls `answer_words`. `--claims` controls claim detail.
Finalizations and cut-reuse misses default to the newest five rows and obey
`--limit` when explicitly supplied.

Normalize legacy telemetry fields:

- outcome is `exact`, `cut`, or `loss`;
- `best_max_depth` becomes `best_max_remaining_depth`;
- expose `bulk_completed_candidate_count` separately from evaluated candidate
  count;
- preserve ceiling, budget, nodes, wall, bundle count, censored units, and
  epoch.

### Candidate claim classification

Never derive a worker number from arbitrary `claimed_by` text. Each returned
claim has:

    {
        "candidate_index": int,
        "state": "done" | "in_flight",
        "completion_kind": "evaluated" | "bulk_eliminated" | null,
        "worker_id": str | null,
        "bundle_id": str | null,
        "claimed_at": int | null,
        "done_at": int | null,
        "republish_count": int,
    }

`claimed_by == "bulk-elimination"` maps to
`completion_kind="bulk_eliminated"` and `worker_id=null`. A done row from a
worker maps to `evaluated`. Existing rows without enough provenance may use
`evaluated` only when that is sound; otherwise use null and surface
`provenance_unknown=true` in branch detail.

Unclaimed indices are not materialized as 12,972 JSON objects. Consumers infer
them from `candidate_count` and the sparse claim list. A later visual renderer
may build cells client-side.

## Tree layout

Tree collection is selected by `--tree`, not a separate report noun.
Its authoritative topology is extant queue rows and their recorded spines.
Nodes may represent queue rows or structural ancestors required by those
spines. Cache rows never create nodes or edges. Cache state may annotate a
queued branch, but a cache-only solution has no tree representation.

The model returns a flat, identity-addressable node array:

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

Return flat nodes so terminal indentation and browser DOM/tree rendering use
one topology without nesting large repeated structures.

- Root selection: extant queue rows and their recorded spines form the tree.
- Word selection: the word is selector context; nodes contain only its extant
  queued response branches and their recorded queued descendants.
- Branch selection: include the selected branch as context when a matching
  queue row exists or a descendant's recorded spine establishes that context,
  then include recorded queued descendants.
- A semantic branch with no matching queue row or recorded queued descendant
  returns `tree_available=false`, `unavailable_reason="no extant queue
  topology"`, and no nodes. Do not synthesize a root from its cached result.
- Missing legacy spine segments appear as explicit unknown nodes at the
  known `guess_depth`; never fabricate guesses.

Sort siblings deterministically by word, pattern, then branch key. Sticky
screen ordering across refreshes remains a client concern.

## Queue, workers, and cache reports

### `--queue`

Default output is a summary followed by filtered inventory. Sort choices:

    default, age, size, workers, priority, nodes, slowest

`--tree` switches the same selected rows to tree layout. There is no separate
`queue tree` command.

### `--workers` / `--worker N`

Show live, idle, finalizing, stale, and dead workers. A branch selector limits
workers to that branch/descendants; a word selector limits workers to its
response branches. Worker detail includes the absolute `guess_depth` path,
current candidate, held bundle/claim, node rate, cache/prune counters, and
live descent.

### `--cache`

- Root: exact/loss totals, recent throughput, reusable
  `max_remaining_depth`/budget distributions, and recent rows.
- Word selector: the same response-group cache coverage as the word report,
  without queue columns.
- Branch selector: normalized cache detail and recent exact/cut/loss
  telemetry for that branch.

## Hotspot report

`--hotspots --by` choices have explicit populations:

| `--by` | Population/result |
|---|---|
| `nodes`, `age`, `size`, `workers`, `priority`, `slowest` | current queue branches |
| `evaluated-candidates`, `bulk-completed-candidates` | bounded recent `branch_finalize_log` branch rows |
| `cut-reuse` | bounded recent `cut_reuse_misses` branch rows |
| `coordination` | workload buckets from bounded recent `claim_telemetry` |

`claim_telemetry` has no branch key. Coordination hotspots therefore rank
answer-count/worker-count workload buckets, not branches, and reject a spine
selector rather than pretending attribution exists.

Historical telemetry options:

- `--epoch N` defaults to the current epoch.
- `--since-seconds N` defaults to 3600.
- `--sample-size N` defaults to 50,000 and is capped at 1,000,000.
- `--limit N` defaults to 10.

Return sample size and truncation metadata so approximated rankings cannot be
mistaken for complete history.

Add idempotent Linux queue migrations for bounded branch detail:

    telemetry.branch_finalize_log(branch_key, recorded_at)
    telemetry.cut_reuse_misses(branch_key, recorded_at)
    telemetry.claim_telemetry(epoch, id)

These are queue/telemetry migrations in `ERDQueue._migrate()`, not
phone-shared cache migrations.

Add bounded query helpers:

    ERDQueue.report_branch_telemetry(branch_key, limit) -> dict
    ERDQueue.report_hotspots(field, epoch, since, sample_size, limit,
                             spine_prefix=None) -> dict

`report_hotspots` chooses the explicit population in the table above and
returns population, sample size, truncation, and stable row/bucket identities.
It must reject unsupported spine attribution for coordination data.

## Terminal rendering and navigation

Add a renderer for every report kind and tree layout. Text and JSON consume
the same envelope.

TTY watch adds:

- branch hotkeys mapped to full `branch_key_hex`;
- numeric worker selection;
- backspace/Escape returns to the previous report request;
- space refreshes;
- `q` quits.

Selections are report requests pushed onto a small navigation stack. Do not
encode navigation by calling legacy command handlers. A branch that
transitions to finalizing remains pinned by identity until dismissed.

## Legacy removal

Remove:

- `status` parser and handler;
- `cache-status`;
- read-only `queue` dashboard, `ls`, `tree`, `show`, `summary`, `top`, and
  `coverage` handlers/parsers;
- terminal-only data assembly and character-diff helpers superseded by
  `report_model`/`report_terminal`;
- obsolete legacy tests after their semantics are represented in report
  tests.

Retain:

- `start`, `stop`, `restart`, `run`;
- `queue add/remove/clear/priority/reset-stale`;
- shared queue-reference behavior only through the new selector parser.

Update `AGENTS.md` and `SWARM.md` in the same commit so no documented command
is left broken.

## Tests

Required coverage includes:

1. Positional inference for root, word, branch, and digest reference.
2. `view QUEUE` explores the word; `view --queue` reports queue inventory.
   The same assertion applies to `CACHE`/`--cache`.
3. Word coverage combines queue and cache states and handles trivial groups.
4. Semantic branch selection works when the branch is cache-only.
5. Digest resolution rejects zero and multiple matches.
6. `--active-only` exactly matches normalized active lifecycle and excludes
   finalizing.
7. Tree context survives active-only filtering while inactive descendants do
   not.
8. Word, branch, and root tree topologies derive only from extant queue rows
   and their recorded spines, have stable parents, and expose absolute
   `guess_depth`.
9. A cache-only branch has normal branch detail but reports no available tree;
   a word tree does not synthesize unqueued response groups.
10. Branch detail classifies bulk elimination without inventing a worker.
11. Claims and answer words are absent unless explicitly requested.
12. Cache state obeys existing budget and `max_remaining_depth` reuse rules.
13. Coordination hotspots identify workload buckets and reject spine scope.
14. Historical reports are bounded and label truncated samples.
15. Queue filtering occurs before limit and summaries count the unpaginated
    population.
16. TTY navigation preserves branch identity through finalization.
17. Removed commands fail argparse while all mutation commands still work.
18. Every `SWARM.md` example parses.

## Acceptance checklist

- [ ] The CLI grammar specified in this phase is implemented.
- [ ] Users never specify `word` versus `branch`; selector form infers it.
- [ ] `--tree` is a layout option at root, word, and branch selections.
- [ ] Tree nodes derive only from extant queue rows and recorded spines;
      cache-only branches are never reconstructed.
- [ ] `--active-only` is available consistently.
- [ ] Word reports join queue and cache coverage.
- [ ] Branch detail includes current packing and cut/reuse semantics.
- [ ] No report performs an unbounded telemetry scan.
- [ ] Queue commands are mutation-only.
- [ ] `status` and `cache-status` are removed.
- [ ] Documentation and tests name only the unified reporting system.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
