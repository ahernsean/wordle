# Phase 1 — Status model: `status_model.py`

Read `00-overview.md` and the repo root `CLAUDE.md` before starting.

## Goal

Create `status_model.py`, a module that assembles the complete swarm status
into one JSON-serializable dict (the **snapshot**), with no printing and no
presentation logic. This is the data source every later phase builds on.

## Non-goals

- Do NOT change what `erd_search.py status` prints. The terminal display keeps
  its own assembly code; the only edits to `erd_search.py` are the two
  function moves and one constant move described below (aliased so behavior
  and existing tests are unchanged).
- No HTTP server, no HTML, no percentages/ETAs/formatting in the snapshot.

## Files touched

- `status_model.py` — new
- `test_status_model.py` — new
- `erd_search.py` — remove three definitions, import them back (details below)
- `test_status_sections.py` — must keep passing UNMODIFIED (it references
  `erd_search._branch_id`, which the alias import preserves)

## Step 1 — Move shared helpers out of `erd_search.py`

Move these, bodies verbatim, from `erd_search.py` into `status_model.py`,
renaming the functions to public names:

| Now in `erd_search.py` | Becomes in `status_model.py` |
|---|---|
| `_branch_id(branch_key)` | `branch_id(branch_key)` |
| `_parse_spine(path)` | `parse_rich_spine(path)` |
| `WORKER_LIVENESS_SECONDS = 30` (with its comment block) | `WORKER_LIVENESS_SECONDS = 30` |

Then in `erd_search.py`, where the definitions were, add:

```python
from status_model import (
    WORKER_LIVENESS_SECONDS,
    branch_id as _branch_id,
    parse_rich_spine as _parse_spine,
)
```

Move any imports the function bodies need (e.g. `hashlib`) along with them.
`status_model.py` must NOT import `erd_search` (that would be a circular
import); `_fmt_spine_path`, `_compact_spine_path`, `_spine_sizes` are
display-only and stay in `erd_search.py`.

## Step 2 — `collect_status`

Module header:

```python
"""status_model.py — Assemble the swarm status into one JSON-serializable snapshot.

The snapshot carries raw data only: identities, counts, timestamps, words,
patterns.  Percentages, ETAs, staleness, and any other derived or formatted
values are renderer concerns and are computed by the consumer.
"""
```

Public constant and function:

```python
SCHEMA_VERSION = 1

def collect_status(queue_path: str, cache_path: str) -> dict:
```

Required imports (mirror `erd_search.py`, which uses the same sources):

```python
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue, guess_depth_from_spine
from wordle_engine import ERD_ALL, load_word_list
from wordle_ui import fmt_pattern
```

Use `ANSWER_FILE = 'NYT_wordlist.txt'` (module-level constant, same value as
`erd_search.ANSWER_FILE` — do not import it from `erd_search`).

### Assembly algorithm

Follow the data-gathering part of `erd_search._print_status` (the code between
the `_section_break('header', ...)` call and the first `print(...)`), which is
the reference implementation. Concretely:

1. `generated_at = int(time.time())`.
2. Load the answer set once: `answer_set = set(load_word_list(ANSWER_FILE))`.
   Cache it in a module-level variable so repeated calls don't re-read the
   file (same trick `_print_status` uses with `_answer_set`).
3. **Queue data**, inside `try/except Exception as e`:
   - `queue = ERDQueue(queue_path)`
   - `counts = queue.counts_by_status()`
   - `branches = queue.branches_in_progress()`
   - `heartbeats = queue.heartbeats_with_branch()`
   - `worker_counts = queue.worker_counts_by_branch()`
   - `done_candidates = {bytes(b['branch_key']): queue.branch_done_candidates(b['branch_key']) for b in branches}`
   - `queue.close()`
   On exception: `queue_ok = False`, `queue_error = str(e)`, and all five
   collections default to empty.
4. **Cache data**, inside a separate `try/except Exception as e`:
   - `score_cache = ScoreCache(cache_path, list(answer_set_in_file_order), checkpoint_on_close=False)`
     — pass the *list* from `load_word_list(ANSWER_FILE)`, not the set.
   - `total_erd_branches`: `SELECT COUNT(*) FROM branch_best_by_policy WHERE policy=? AND answer_list_id=?` with `(ERD_ALL, score_cache.answer_list_id)`
   - `recent_5m_branches`: same query plus `AND updated_at > ?` with `generated_at - 300`
   - `score_cache.close()`
   On exception: `cache_ok = False`, `cache_error = str(e)`, both counts 0.
5. Build and return the snapshot dict per the schema below.

### Snapshot schema (normative)

Every key below must be present in every snapshot, even when its value is
null/empty. All values must survive `json.dumps` unchanged (no bytes, no
sqlite3.Row, no datetime objects).

Top level:

| Key | Type | Value |
|---|---|---|
| `schema_version` | int | `SCHEMA_VERSION` |
| `generated_at` | int | Unix seconds at assembly time |
| `worker_liveness_seconds` | int | `WORKER_LIVENESS_SECONDS` |
| `queue` | object | `{"ok": bool, "error": str or null}` |
| `cache` | object | `{"ok": bool, "error": str or null, "total_erd_branches": int, "recent_5m_branches": int}` |
| `counts` | object | `{"pending": int, "in_progress": int, "done": int, "cooperative": int}` — first three from `counts_by_status()` (`.get(key, 0)`); `cooperative` = number of rows in `branches` with `(priority or 0) >= 1_000_000` |
| `worker_totals` | object | `{"cache_hits", "cache_misses", "n_ok", "n_cutoff", "n_pruned"}` — each int, summed over **live** workers only (heartbeat age ≤ `WORKER_LIVENESS_SECONDS`), treating NULL columns as 0 |
| `branches` | array | One object per row of `branches_in_progress()`, in query order (priority DESC, n_words DESC) |
| `workers` | array | One object per row of `heartbeats_with_branch()`, sorted with the tuple key `(0, int(worker_number), '')` when `worker_number` is all digits, else `(1, 0, worker_id)` — digit-numbered workers first in numeric order, the rest after in `worker_id` order. (A bare conditional key of `int` vs `str` raises `TypeError` when the two kinds coexist.) |

Each element of `branches`:

| Key | Type | Value |
|---|---|---|
| `branch_id` | str | `branch_id(bytes(row['branch_key']))` |
| `branch_key_hex` | str | `bytes(row['branch_key']).hex()` |
| `n_words` | int | `row['n_words'] or 0` |
| `n_candidates` | int | `row['n_candidates'] or 0` |
| `done_candidates` | int | from the `done_candidates` map, default 0 |
| `priority` | int | `row['priority'] or 0` |
| `is_cooperative` | bool | `priority >= 1_000_000` |
| `source_word` | str/null | `row['source_word']` (as stored, lowercase) |
| `source_pattern` | str/null | `fmt_pattern(row['source_pattern'])` if the column is not NULL, else null |
| `best_guess` | str/null | `row['best_guess']` |
| `best_guess_is_answer` | bool | `(row['best_guess'] or '').lower() in answer_set` |
| `best_erd` | float/null | `row['best_erd']` |
| `best_max_depth` | int/null | `row['best_max_depth']` |
| `budget` | int/null | `row['budget']` (guard with `'budget' in row.keys()` — legacy rows may predate the column) |
| `created_at` | int/null | `row['created_at']` |
| `nodes_spent` | int | `row['nodes_spent'] or 0` |
| `guess_depth` | int | if the row has a non-empty `spine`: `guess_depth_from_spine(spine)`; else `1 if (row['source_word'] and row['source_pattern'] is not None) else 0`. CAUTION: the pattern check must be `is not None`, never truthiness — pattern code `0` (all gray) is a valid, falsy value and counts as set. (Mirrors `_branch_guess_depth` in `_print_status`.) |
| `spine` | array | Parsed from the `spine` text: split on whitespace, take tokens pairwise as `(guess, pattern)`; each entry `{"guess": <UPPERCASE str>, "pattern": <pattern str>, "guess_is_answer": bool}`. NULL/empty spine → `[]`. (A branch with `guess_depth > len(spine)` has an unrecorded spine; renderers handle that.) |
| `worker_count` | int | `worker_counts.get(bytes(row['branch_key']), 0)` |

Each element of `workers` (heartbeat row `h`; guard optional columns with
`'col' in h.keys()` exactly as `_print_status` does):

| Key | Type | Value |
|---|---|---|
| `worker_id` | str | `h['worker_id']` |
| `worker_number` | str | portion after the last `-` in `worker_id`, or the whole id if no `-` (mirrors `_worker_num`) |
| `pid` | int | `h['pid']` |
| `updated_at` | int | `h['updated_at']` |
| `age_seconds` | int | `generated_at - h['updated_at']` |
| `is_live` | bool | `age_seconds <= WORKER_LIVENESS_SECONDS` |
| `branch_id` | str/null | `branch_id(bytes(h['current_branch_key']))` if not NULL |
| `branch_key_hex` | str/null | hex of `current_branch_key` if not NULL |
| `claim_idx` | int/null | `h['claim_idx']` |
| `claims_done` | int | `h['claims_done'] or 0` |
| `claim_started_at` | int/null | `h['claim_started_at']` |
| `cur_candidate` | str/null | `h['cur_candidate']` |
| `cur_candidate_is_answer` | bool | `(cur_candidate or '').lower() in answer_set` |
| `cur_max_depth` | int/null | `h['cur_max_depth']` |
| `cur_nodes` | int/null | `h['cur_nodes']` |
| `node_rate` | float/null | `h['node_rate']` |
| `descent` | array | `parse_rich_spine((h['cur_path'] or '').replace('>', '→'))` — the replace normalizes legacy heartbeat rows whose levels are `>`-separated, exactly as `_print_status` does before displaying a path — mapped to `{"guess_depth": int or null, "guess": <UPPERCASE str> or null, "pattern": str or null, "size": str or null}` per tuple. `size` stays the string the parser yields (old-format tokens may be non-numeric sentinels — do not coerce to int). Missing/NULL `cur_path` → `[]`. Do not filter entries — renderers decide which depths to show. |
| `cache_hits` / `cache_misses` / `n_ok` / `n_cutoff` / `n_pruned` | int | each `h[col] or 0` |
| `best_guess` | str/null | `h['best_guess']` |
| `best_erd` | float/null | `h['best_erd']` |
| `bound_erd` | float/null | `h['bound_erd']` |

## Step 3 — Tests: `test_status_model.py`

Model the fixtures on `test_erd_queue_unit.py` (temp directory via
`tempfile.TemporaryDirectory`, `ERDQueue(os.path.join(tmp, 'q.sqlite3'))`).
For the cache, `ScoreCache(os.path.join(tmp, 'c.sqlite3'), answers, checkpoint_on_close=False)`
with a small answer list creates a valid empty cache. NOTE: `collect_status`
loads the real `NYT_wordlist.txt` for the answer set — that file is in the
repo, so tests may rely on real words like `salet` not being answers and
`crane` being one.

Required test cases:

1. **Empty databases**: create fresh queue and cache files, call
   `collect_status`; assert `queue['ok']` and `cache['ok']` are True,
   `branches == []`, `workers == []`, all `counts` values 0, and every
   top-level schema key is present.
2. **Unavailable queue**: pass a queue path inside a *nonexistent directory*;
   assert `queue['ok'] is False` and `queue['error']` is a non-empty string,
   and the function still returns a complete snapshot.
3. **One branch, one worker**: create a branch with
   `queue.create_branch(branch_key, n_words=5, n_candidates=3, priority=0,
   source_word='crane', source_pattern=0, budget=5,
   spine='CRANE -----')`, write a heartbeat with
   `queue.heartbeat('worker-3', pid=1, current_branch_key=branch_key,
   n_words=5, started_at=<now>, claims_done=2, claim_idx=1,
   cur_candidate='salet', cur_max_depth=2, cur_nodes=100, node_rate=50.0,
   cur_path='2:SALET:-y---/12→3:7')`, then mark one candidate done via
   `queue.claim_candidate(...)` + `queue.complete_candidate(...)`.
   Assert on the snapshot:
   - one branch: correct `branch_id` (equal to `branch_id(branch_key)`),
     `branch_key_hex`, `n_candidates == 3`, `done_candidates == 1`,
     `guess_depth == 1`, `spine == [{"guess": "CRANE", "pattern": "-----", "guess_is_answer": True}]`,
     `is_cooperative is False`, `worker_count == 1`
   - one worker: `worker_number == '3'`, `is_live is True`,
     `branch_key_hex` matches the branch, `claim_idx == 1`,
     `cur_candidate == 'salet'`, `cur_candidate_is_answer is False`,
     `descent == [{"guess_depth": 2, "guess": "SALET", "pattern": "-y---", "size": "12"}, {"guess_depth": 3, "guess": None, "pattern": None, "size": "7"}]`
4. **Cooperative flag**: a branch created with `priority=1_000_000` yields
   `is_cooperative is True` and `counts['cooperative'] == 1`.
5. **Fallback guess_depth with all-gray pattern**: a branch created with
   `spine=None`, `source_word='crane'`, `source_pattern=0` yields
   `guess_depth == 1` and `spine == []` (catches the truthiness bug where
   pattern code 0 is misread as "not set").
6. **JSON round-trip**: `json.loads(json.dumps(snapshot))` equals the
   snapshot (dict equality).
7. **Helper stability**: `status_model.branch_id(b'key')` returns the same
   4-char value on repeated calls and differs for a different key (mirrors the
   existing `test_status_sections.py` assertions, now against the public name).

## Acceptance checklist

- [ ] `status_model.py` exists; `collect_status`, `branch_id`,
      `parse_rich_spine`, `WORKER_LIVENESS_SECONDS`, `SCHEMA_VERSION` are its
      public API; it does not import `erd_search`.
- [ ] `erd_search.py` no longer defines `_branch_id`, `_parse_spine`, or
      `WORKER_LIVENESS_SECONDS`; it imports them from `status_model` (with the
      underscore aliases shown above).
- [ ] `python -m unittest discover -s tests -t . -p 'test_*.py'` passes, including the
      untouched `test_status_sections.py` and the new `test_status_model.py`.
- [ ] `python -c "import json, status_model; print(json.dumps(status_model.collect_status('erd_queue.sqlite3','wordle_cache.sqlite3'), indent=2)[:500])"`
      runs without error (with or without live databases present).
- [ ] No file outside the "Files touched" list is modified.
