# Phase 3a — Semantic word and branch reports

Read phases 00–02 and `AGENTS.md` first. Requires phase 2 merged.

## Goal

Add semantic work exploration without removing any legacy inspection command:

- infer root, word, branch, and queue-reference selections from positional
  spine form;
- add word response-group coverage and singular branch detail;
- join current queue, worker, cache, claim, and republish state;
- render those reports in one-shot and watched terminal sessions.

## Non-goals

- No collection filters, tree layout, queue inventory, global worker report,
  or global cache report; phase 3b adds them.
- No historical branch telemetry or hotspots; phase 3c adds them.
- No TTY drill-down navigation or legacy-command removal; phase 3d owns the
  cutover.

## Files touched

- `report_model.py`
- `report_terminal.py`
- `erd_queue.py`
- `cache_sqlite.py`
- `erd_search.py`
- `tests/test_report_model.py`
- `tests/test_report_terminal.py`
- `tests/test_report_objects.py` — new

## Request and branch_target model

Extend `report_model.py` with:

    parse_report_branch_target(parts: list[str] | str | None) -> ReportBranchTarget

And these frozen dataclasses:

    SpineStep(word: str, pattern: str)

    ReportBranchTarget(
        kind: str,                    # root | word | branch | branch_reference
        steps: tuple[SpineStep, ...],
        trailing_word: str | None,
        branch_reference: str | None,
        input_text: str,
    )

    ReportBranchTarget.root() -> ReportBranchTarget

Supersede phase 1's overview-only request with:

    ReportRequest(
        report_kind: str = "auto",
        branch_target: ReportBranchTarget = ReportBranchTarget.root(),
        include_claims: bool = False,
        include_answers: bool = False,
    )

Use `dataclasses.field(default_factory=...)` for dataclass instance defaults.
`ReportBranchTarget.root()` returns the normalized empty branch_target. The default
`report_kind` intentionally changes from `overview` to `auto`; root inference
still produces the operational overview. Update phase 1 and 2 tests that
assert the request's echoed default while preserving their observable
overview behavior.

### BranchTarget grammar

`parse_report_branch_target` accepts a string or already-tokenized parts. When
given a list, join it with spaces before parsing so CLI and later HTTP clients
use identical behavior. Normalize words to lowercase and patterns to five
characters using `g`/`y`/`-`; a dot normalizes to `-`.

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

Reject malformed tokens with a message identifying the token and expected
form. Parsing never inspects a database; reference and semantic-spine
resolution happen during collection.

## CLI additions

Extend phase 2's `view` parser with positional `SPINE` using `nargs="*"` and:

    --claims
    --answers

When a dash-leading response pattern would confuse argparse, the documented
forms are one quoted branch_target string or dot-gray syntax. Preserve support for
`--` before positional branch_target parts.

Dispatch without an explicit report kind:

| BranchTarget | Collector |
|---|---|
| root | overview |
| word | word |
| branch/reference | branch |

The envelope's `report_kind` is the inferred domain kind.

## BranchTarget resolution

Add presentation-neutral resolver functions to `report_model.py`:

    resolve_branch_target(branch_target, all_answers) -> ResolvedBranch
    resolve_branch_reference(queue, digest_prefix) -> bytes

`ResolvedBranch` contains selected answer words, encoded branch key,
normalized complete spine, and an optional trailing word.

For a semantic spine, start with all answers loaded from
`ReportSources.answer_list_path` and apply each guess/pattern response group
in order through `ResponseCache(all_answers, score_cache=None)`. Use the pure
reference path so branch_target resolution never reads or writes persistent cache
state. Produce the encoded branch key with the static
`ScoreCache.encode_subset` helper. An empty response group is a valid resolved
branch with zero answers; report it rather than treating it as parse failure.

For `@digest-prefix`:

1. Ask the queue for all user and cooperative rows matching the prefix.
2. Require exactly one match.
3. Zero matches is not found; multiple matches returns a disambiguation error
   listing 12-character references and spines.

Digest references are queue-local conveniences. A cache-only branch must be
selected semantically because scanning the entire cache to resolve a digest
is prohibited.

Add the bounded queue helper:

    ERDQueue.branch_rows_for_reference_prefix(digest_prefix) -> list[dict]

This helper returns candidate rows; only `report_model.resolve_branch_reference`
performs single-result resolution and raises not-found or ambiguity errors.

## Cache query helpers

Add read-only helpers to `ScoreCache`:

    report_branch_state(branch_key, policy, budget=None) -> dict
    report_branch_states(branch_keys, policy, budget=None) -> dict

Normalize cache state as `exact`, `loss`, `missing`, or `not_applicable`. An
exact state includes best guess, ERD, `max_remaining_depth`, solve budget,
taint, and updated timestamp. A loss state includes its available budget
validity. Use existing cache-reuse semantics rather than defining a second
interpretation in reporting code.

`report_branch_states` batches response-group coverage instead of opening one
connection or query setup per branch. Up to 242 keyed lookups is the normal
word-report workload. No cache schema change belongs in this phase.

## Word report

For a trailing-word branch_target:

1. Resolve the preceding patterned spine to `branch_words`.
2. Partition those answers by the trailing word.
3. Include every nonempty response group. Groups with fewer than two answers
   have cache state `not_applicable`.
4. Batch lookup current branch status, phase, and cache state by encoded branch key.
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
                "branch_status": str,
                "branch_phase": str | null,
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

`--answers` adds `answer_words` to every response group. Without it, do not
send repeated word arrays. Text renders one combined queue/cache table; JSON
uses the same payload.

## Branch report

A branch report joins semantic identity across current queue, workers, claims,
republish state, and cache:

    {
        "branch": {
            identity, spine, guess_depth, budget, answer_count,
            optional answer_words
        },
        "queue": {
            branch status, branch phase, raw statuses, priority, candidate progress,
            running best, ceiling, nodes, created/finalized timestamps
        } | null,
        "cache": normalized cache state,
        "workers": [...],
        "republished_candidates": [...],
        "claims": [...] | null,
    }

`--answers` controls `answer_words`; `--claims` controls sparse claim detail.
Phase 3c extends this payload with bounded bundle, finalization, and cut-reuse
telemetry.

Normalize legacy telemetry fields already present in current rows:

- `best_max_depth` becomes `best_max_remaining_depth`;
- expose `bulk_completed_candidate_count` separately from evaluated candidate
  count;
- preserve ceiling, budget, nodes, and current packing state.

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
worker maps to `evaluated`. Existing rows without sound provenance use null
and surface `provenance_unknown=true` in branch detail.

Unclaimed indices are not materialized as 12,972 JSON objects. Consumers infer
them from `candidate_count` and the sparse claim list.

## Terminal rendering

Add word and branch renderers. Text and JSON consume the same envelope, and
phase 2 watch behavior continues to collect the current request on every
refresh. Semantic change comparison uses branch keys, worker IDs, and
candidate indices; it does not diff rendered characters.

Every response-group, queue-state, worker, claim, finalization, and cut-reuse
row uses phase 2's adaptive column helper. At 50–59 columns, response groups
retain pattern, answer count, phase, cache state, and full branch reference;
branch detail retains semantic identity plus the smallest table that still
explains queue/cache state. Optional answer lists wrap or truncate as elastic
text instead of forcing the semantic columns wider.

This phase adds no drill-down hotkeys. Users select a different report by
starting another command until phase 3d adds an in-session navigation stack.

## Tests

Required coverage:

1. Positional inference for root, word, branch, dot-pattern normalization,
   and digest reference.
2. `view QUEUE` and `view CACHE` explore words; neither is a reserved noun.
3. The phase 1 `report_kind="overview"` default is intentionally superseded;
   default phase 2 behavior still produces the overview.
4. Word coverage combines current queue and cache states and handles trivial
   groups.
5. Semantic branch selection works when the branch is cache-only.
6. Semantic branch_target resolution performs no persistent cache read or write.
7. Digest resolution rejects zero and multiple matches.
8. The queue prefix helper and model resolver have distinct names and return
   contracts.
9. Branch detail classifies bulk elimination without inventing a worker.
10. Claims and answer words are absent unless explicitly requested.
11. Cache state obeys existing budget and `max_remaining_depth` reuse rules.
12. Watched word and branch reports preserve stable identities.
13. Word and branch text at widths 50, 55, 59, 60, 79, 80, and 120 obeys the
    shared column-removal order and preserves patterns, references, worker
    IDs, and candidate indices.
14. All legacy status and read-only queue tests remain green.

## Acceptance checklist

- [ ] Users never specify `word` versus `branch`; branch_target form infers it.
- [ ] Word reports join current queue and cache coverage.
- [ ] Branch detail works for queued, active, done, and cache-only branches.
- [ ] Claim provenance distinguishes evaluated and bulk-eliminated work.
- [ ] Word and branch terminal tables remain operational at 50 columns.
- [ ] Queue prefix lookup and single-reference resolution are unambiguous APIs.
- [ ] Legacy inspection commands remain available.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
