# Plan: Parallel ERD_ALL Root-Word Precache on Linux

> Working notes for implementing this on the Rocky Linux 9.7 box (8 cores,
> 16GB RAM, systemd `--user` + linger enabled, SELinux enforcing/unconfined,
> Python 3.13.13 at `/usr/bin/python3.13`). Repo lives at `~/work/wordle`.
> This doc is the handoff from the design session — read it fully before
> starting implementation.

## Context

The Wordle solver's `ERD_ALL` policy (`erd_words_unfiltered` in
`subgroup_best_by_policy`) gives the exact expected-remaining-guesses score
for any subset of answer words, computed by
`wordle_engine.min_expected_guesses`. Today this cache is filled in
on-demand/single-threaded (e.g. via `cmd_precache` /
`BranchPrecacheSolver`, one root word at a time). The goal is to
pre-populate this cache for **every root word** (all 12,972 words in
`wordle.txt`) so that `_erd_solve_scores` at the start of a game becomes a
pure cache read — and to do this on the Rocky box, in parallel, over
multiple days, while the user keeps playing (and growing the cache) on
iPhone/iPad in the meantime.

Requirements driving the design:
- Headless: launch, monitor, and restart workers without a persistent
  terminal/SSH session.
- Multiple workers must coordinate so they don't redo or clobber each
  other's work.
- Output is the existing `wordle_cache.sqlite3` format, portable back to
  iPhone/iPad.
- Exploit both axes of parallelism (root words, and the branch subgroups
  each root word recurses into) without redundant cross-root-word work.
- Workers share one cache so sub-work from one benefits all.

No changes to `wordle_engine.py` or `cache_sqlite.py` are needed — this is
a pure additive orchestration layer on top of existing, already-correct
primitives.

---

## Recommended Approach

### 1. Unified subgroup work queue

`min_expected_guesses` recursion, as a side effect, fills in
`subgroup_best_by_policy[ERD_ALL]` for *every* non-pruned branch subgroup of
every candidate guess it considers — not just for the top-level word. So
instead of treating "root words" and "branches" as two separate parallel
axes, flatten everything into **one deduplicated queue of distinct answer
subsets** that need an `ERD_ALL` entry:

- A **bootstrap** pass walks all 12,972 root words, uses
  `ResponseCache.group_words(word, all_answers)` (which also persists
  `response_decomposition` rows for free) to get each word's response-group
  partition, and enqueues every branch of size ≥ 2
  (`enumerate_branches`-equivalent) keyed by
  `ScoreCache.encode_subset(words)`.
- Dedup is automatic via `PRIMARY KEY(subset_key)` + `INSERT OR IGNORE` —
  the same ~50-200 word subgroup recurring across dozens of root words is
  only enqueued once.
- Workers then repeatedly **claim the largest pending subgroup**, check
  `score_cache.read(subset_key, ERD_ALL)` first (cheap skip if another
  worker's recursion already filled it in), and otherwise call
  `min_expected_guesses(words, rcache, score_cache, guesses=all_words,
  policy=ERD_ALL, cancel_check=..., heartbeat=...)` to full completion (no
  `deadline=` — since a cancelled call writes nothing, a deadline would just
  cause endless retries; let it run for days and rely on `cancel_check` for
  clean shutdown only).

Processing largest-first maximizes the "free" fill-in of smaller subgroups
via recursion, so the queue shrinks faster than its raw count suggests.

### 2. New files (all additive, in `~/work/wordle/`)

```
erd_queue.py      ErdQueue class — sidecar SQLite coordination DB
erd_search.py     CLI: bootstrap | run | status | reset-stale
erd_worker.py     worker process entry point (multiprocessing.Process target)
merge_cache.py    CLI: merge an iPhone/iPad wordle_cache.sqlite3 into Linux's
wordle-erd.service   systemd --user unit
erd_queue.sqlite3    sidecar coordination DB (gitignored, created at runtime)
```

**`erd_queue.sqlite3` schema** (WAL, `synchronous=NORMAL`, autocommit):

```sql
CREATE TABLE pending_subgroups (
    subset_key   BLOB PRIMARY KEY,   -- same encoding as ScoreCache.encode_subset
    n_words      INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|in_progress|done
    claimed_by   TEXT, claimed_at INTEGER, completed_at INTEGER
);
CREATE INDEX idx_pending_status_n ON pending_subgroups(status, n_words DESC);

CREATE TABLE worker_heartbeat (
    worker_id TEXT PRIMARY KEY, pid INTEGER NOT NULL,
    current_subset_key BLOB, n_words INTEGER,
    started_at INTEGER, updated_at INTEGER NOT NULL,
    subgroups_done INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE run_meta (key TEXT PRIMARY KEY, value TEXT);
-- bootstrap_status, bootstrap_done_roots (comma list), bootstrap_completed_at
```

`ErdQueue` provides: `add_pending_many`, `claim_next` (atomic
`BEGIN IMMEDIATE` → pick largest `pending` by `n_words DESC` → mark
`in_progress`), `mark_done`, `reset_stale_in_progress` (run on every
startup — a crashed worker's claim just gets re-claimed), `heartbeat`,
`counts_by_status`, `words_remaining_done`, `get/set_meta`. Re-export
`encode_subset`/`decode_subset` from `cache_sqlite.ScoreCache` rather than
duplicating the encoding.

**Atomicity of `claim_next`**: wraps the select-then-update in
`BEGIN IMMEDIATE` so the write lock is taken *before* the read. Without
this, two workers could both `SELECT` the same `pending` row and then race
to `UPDATE` it (TOCTOU). With `BEGIN IMMEDIATE`, the whole sequence is
serialized — by the time worker B's transaction starts, worker A has
already committed the `status='in_progress'` update, so B's `SELECT WHERE
status='pending'` won't see that row. Under contention, `timeout=30.0` on
`sqlite3.connect()` makes SQLite retry automatically rather than raising
`database is locked`. If a worker crashes mid-claim, the row just sits as
`in_progress` until `reset_stale_in_progress()` resets it on the next `run`
— never a double-compute, only a possible redo.

### 3. `erd_search.py bootstrap`

Single-threaded pre-pass over all 12,972 root words (the existing
~21-minute decomposition-build cost): for each word, call
`rcache.group_words(word, all_answers)`, enqueue every group of size ≥ 2.
Resumable via `run_meta.bootstrap_done_roots` + periodic
`score_cache.checkpoint()`. Running this **before** workers start avoids 6
processes redundantly racing to build the same 12,972 decomposition blobs
(racing would be *correct* — `write_decomposition` is `INSERT OR
REPLACE` — just wasteful).

### 4. `erd_worker.py` (worker loop)

Each worker opens its **own** `ScoreCache`, `ResponseCache`, and `ErdQueue`
connections (sqlite3 connections aren't process-shareable — same pattern
`BranchPrecacheSolver` already uses). Loop:

1. `claim_next(worker_id)` → `(subset_key, n_words)` or `None` (sleep 2s and
   retry if empty — bootstrap may still be enqueueing, or the queue is
   briefly drained).
2. Heartbeat (throttled to ~once every 2s — `min_expected_guesses` heartbeat
   fires on every recursive call, far too often to write every time).
3. Skip (`mark_done` immediately) if `score_cache.read(subset_key,
   ERD_ALL)` is already populated — a side-effect fill-in from another
   worker's larger subgroup.
4. `rank_guesses_by_group_then_entropy` to order candidates, then
   `min_expected_guesses(..., guesses=all_words, policy=ERD_ALL,
   cancel_check=stop_event.is_set, heartbeat=...)`. On cancellation, leave
   the row `in_progress` (picked up by `reset_stale_in_progress` next run);
   otherwise `mark_done` and periodically `score_cache.checkpoint()`.
5. Exit the loop (clean process exit) after `--recycle-after` (default
   1000) subgroups — this is the recycling mechanism for `_mem_cache`
   growth (see Memory below); no IPC needed beyond `stop_event`.

### 5. `erd_search.py run` (supervisor)

- Calls `reset_stale_in_progress()`, warns (doesn't block) if
  `bootstrap_status != 'done'`.
- Spawns `--workers N` (default 6) via `multiprocessing.Process` (not
  `Pool` — workers run indefinitely pulling from a queue, which doesn't fit
  `Pool`'s per-task model).
- Polls every 5s: respawns any worker that exited (recycle) or exceeded
  `--recycle-hours` (default 3, force-`terminate()`); declares the run
  complete when `pending == 0 and in_progress == 0`.
- `SIGTERM`/`SIGINT` → `multiprocessing.Event` (`stop_event`) → workers
  notice via `cancel_check` (checked at every recursion depth) and exit
  promptly, even mid-recursion on a large subgroup.

### 6. `erd_search.py status`

Read-only report, safe to run concurrently with `run`. Two halves:

- **Queue health** (from `erd_queue.sqlite3`): counts of
  `pending`/`in_progress`/`done` subgroups, and the same broken down by
  total *words* (a few huge subgroups dominate wall-clock time far more
  than the subgroup count suggests).
- **Cache growth** (from `wordle_cache.sqlite3`): total `ERD_ALL` rows, rows
  written in the last 5 minutes → rows/min rate → a naive ETA. Keep this
  even though it's a rough lower bound early on (recursive side-effect
  fill-in makes the real rate climb over time) — label it clearly as
  "rough, ignores side-effect fill-in" but it's still useful signal.
- **Per-worker table**: from `worker_heartbeat` — pid, current subgroup
  (decoded word + size), subgroups completed, uptime, seconds since last
  heartbeat. A stale heartbeat on a live PID = genuine hang (heartbeat fires
  on every recursive call, so a live worker's timestamp should update at
  least every couple seconds even deep in a huge subtree).

`--watch [SECONDS]` loops + clears the screen (`watch`-style, default 30s) —
no curses dependency, just for an SSH session you're actively watching.

### 7. Headless launch/monitor/restart

**systemd `--user` unit** (`~/.config/systemd/user/wordle-erd.service`):
`Type=simple`, `ExecStart=/usr/bin/python3.13 ~/work/wordle/erd_search.py run
--workers 6`, `Restart=on-failure`, `KillSignal=SIGTERM`/`TimeoutStopSec=120`
for graceful drain, `Nice=10`, `MemoryHigh=12G`/`MemoryMax=14G` as a cgroup
backstop. Paired with `loginctl enable-linger` (already enabled). Monitor via
`systemctl --user status wordle-erd` and `journalctl --user -u wordle-erd -f`
(persistent journal already set up at `/var/log/journal`); restart after a
bug fix via `systemctl --user restart wordle-erd`.

Fallback if needed: `nohup setsid python3.13 erd_search.py run --workers 6 >
erd_search.log 2>&1 < /dev/null & disown`; stop via `pgrep -f 'erd_search.py
run'` + `kill <pid>` (same `SIGTERM` → graceful drain path). Both paths use
the same `run` entry point and signal handling.

### 8. iPhone ↔ Linux cache merge

**Merge, not overwrite**: a straight overwrite (replace the iPhone's
`wordle_cache.sqlite3` with the Linux one) would technically be safe since
every cache table is deterministic given the same `all_answers`/`all_words`
(matching keys imply identical values) — but a merge is strictly better at
near-zero cost: the iPhone may have computed rows for subgroups the Linux
bootstrap/queue never enqueued (e.g. from live-game branches outside any
root word's partition), and `INSERT OR IGNORE` makes those additive with
zero conflict risk. `merge_cache.py` ATTACHes the iPhone DB and runs `INSERT
OR IGNORE INTO main.<table> SELECT ... FROM iphone.<table>` across all 4
cache tables (`universe`, `response_decomposition`,
`subgroup_best_by_policy`, `word_scores`), plus `--dry-run` to report counts
first. No in-progress "current game" state is persisted to SQLite, so this
only ever adds cache entries, never disrupts a live game on either device.

### Critical precondition: word-list parity

`universe_id = sha256("\n".join(answer_words))` (over `NYT_wordlist.txt`,
**order-sensitive**) scopes every row in every cache table. If the Linux
box's `NYT_wordlist.txt` differs from the iPhone's — even just in line
order — the entire multi-day Linux computation lands under a different
`universe_id` and is **invisible** to the iPhone, merge or no merge.
`wordle.txt` also affects `min_expected_guesses`'s candidate list and thus
which `best_word`/`best_score` gets computed for a given subset.

**Action required before `bootstrap`**: copy `NYT_wordlist.txt` and
`wordle.txt` from the iPhone/iPad to the Rocky box byte-for-byte (don't
regenerate/re-sort them independently — they should already match since
both come from the same git repo, but confirm via the commands below before
committing to a multi-day run). `erd_search.py bootstrap` should print the
resulting `universe_id` up front so this can be sanity-checked against the
iPhone's `c` (cache info) command output.

```bash
sha256sum NYT_wordlist.txt wordle.txt   # compare against iPhone's copy
```

### Scale and memory

- Distinct-subgroup count after full bootstrap is the dominant unknown —
  estimated **300K-700K** (~125-220MB `erd_queue.sqlite3`), comfortably
  within 16GB. **Recommend a small dry-run bootstrap first** (e.g.
  `--root-words` pointing at a ~200-word file) to measure actual growth
  before running the full 12,972-word bootstrap.
- `ResponseCache._cache` is bounded (~41.5MB/worker, plateaus once a worker
  has seen all 12,972 guess words).
- `score_cache._mem_cache` is **unbounded** per connection (one entry per
  distinct subgroup visited during recursion, not just completed) —
  estimated 75-600MB/worker over a long run. This is the reason for
  `--recycle-after` (default 1000 completed subgroups) and
  `--recycle-hours` (default 3) — each worker exits cleanly and the
  supervisor respawns it, resetting the cache. Calibrate via `ps -o
  pid,rss,etime -p $(pgrep -f erd_worker)` during the first few hours.

### Defaults (all CLI flags, adjustable)

`--workers 6` (of 8 cores — leaves 2 free), `--recycle-after 1000`,
`--recycle-hours 3`, `--checkpoint-every 100`.

---

## Critical Files

| File | Status |
|---|---|
| `erd_queue.py` | new — `ErdQueue` class + schema |
| `erd_search.py` | new — `bootstrap` / `run` / `status` / `reset-stale` CLI |
| `erd_worker.py` | new — worker process `main()` |
| `merge_cache.py` | new — iPhone/Linux merge CLI (`--dry-run` supported) |
| `wordle-erd.service` | new — systemd `--user` unit (+ documented nohup fallback) |
| `.gitignore` | add `erd_queue.sqlite3`(+`-shm`/`-wal`), `erd_search.log`, `erd_worker_*.log` |

Reused as-is, no edits:
- `wordle_engine.enumerate_branches`/`ResponseCache.group_words` (branch partitioning)
- `wordle_engine.min_expected_guesses` with `policy=ERD_ALL`, `cancel_check`, `heartbeat`
- `wordle_engine.rank_guesses_by_group_then_entropy`
- `cache_sqlite.ScoreCache.encode_subset`/`read`/`write`/`checkpoint`/`read_decomposition`/`write_decomposition`

---

## Implementation Order

1. `erd_queue.py` — independently unit-testable with `sqlite3.connect(':memory:')`.
2. `erd_search.py bootstrap` — test against a small `--root-words` file (~20
   words) + throwaway `--cache`/`--queue`, validate dedup counts.
3. `erd_worker.py` — test by calling `main()` directly (no
   `multiprocessing`) against a tiny pre-seeded queue; validate
   claim → compute → mark_done and the already-cached skip path.
4. `erd_search.py run` (supervisor) — test with `--workers 2` against the
   small queue from step 2; verify `SIGTERM`/`Ctrl-C` drains cleanly.
5. `erd_search.py status` / `reset-stale` — read-only, buildable any time
   after step 1.
6. `merge_cache.py` — independent; test with two small throwaway
   `ScoreCache` files seeded with disjoint + overlapping rows
   (`--dry-run` then real merge).
7. systemd unit + nohup docs — last, after `run`'s `SIGTERM` handling is
   validated manually with `kill -TERM`.
8. **Word-list sync check**, then the full 12,972-word `bootstrap` (the
   ~21-minute single-threaded pass) as a deliberate, monitored step —
   confirms `universe_id` matches the iPhone before starting workers.
   Only then start `run --workers 6` under systemd.

---

## Verification

- **Unit-level**: `erd_queue.py`'s claim/mark_done/reset_stale logic against
  an in-memory DB, mirroring `test_wordle.py`'s existing style.
- **Bootstrap dry run**: 20-word `--root-words` file → inspect
  `pending_subgroups` row count and a few decoded `subset_key`s against
  hand-computed `enumerate_branches` output.
- **Worker loop**: pre-seed a small queue, run `erd_worker.main()` directly,
  confirm `subgroup_best_by_policy[ERD_ALL]` rows appear and
  `pending_subgroups.status` transitions to `done`.
- **Supervisor**: `erd_search.py run --workers 2` against the small queue;
  send `SIGTERM`, confirm workers exit within a few seconds and
  `in_progress` rows are correctly reclaimed via `reset_stale_in_progress`
  on the next start.
- **Status**: `erd_search.py status` reflects the small run's
  progress/heartbeats correctly while `run` is active.
- **Merge**: two throwaway `wordle_cache.sqlite3` files with disjoint and
  overlapping rows → `merge_cache.py --dry-run` reports expected new-row
  counts, real run produces the union with no errors.
- **Full run**: after the word-list sync check, run the full bootstrap,
  confirm `universe_id` matches the iPhone's `c` (cache info) output, then
  start `run --workers 6` under systemd and confirm `status --watch` shows
  steady throughput and per-worker heartbeats over the first 30+ minutes.
