# Swarm Operations Guide

The ERD_ALL precache is built by `erd_search.py`, a CLI that manages a
supervisor process and its worker pool.  All commands run from the `wordle/`
directory with `python3.13 erd_search.py <subcommand>`.

Two SQLite files are involved, both under `runtime/`:

| File | Purpose |
|---|---|
| `runtime/wordle_cache.sqlite3` | Durable ERD results; shared with the iPhone |
| `runtime/erd_queue.sqlite3` | Transient coordination only (queue, candidate claims, heartbeats) |

`queue clear` wipes only the queue file — the cache is never touched by any
queue command.

---

## Service: start, stop, status

The supervisor runs as a systemd user service named `wordle-erd`.  The report
web server (`report_server.py`) runs as a separate systemd user service named
`wordle-report-server`.

### Install the services (one-time)

```bash
ln -s ~/work/wordle/wordle-erd.service ~/.config/systemd/user/
ln -s ~/work/wordle/wordle-report-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable wordle-erd             # start automatically on login
systemctl --user enable wordle-report-server   # start automatically on login
```

### Start and stop

```bash
python3.13 erd_search.py start    # systemctl --user start wordle-erd wordle-report-server
python3.13 erd_search.py stop     # systemctl --user stop wordle-erd wordle-report-server
python3.13 erd_search.py restart  # systemctl --user restart wordle-erd wordle-report-server

systemctl --user status wordle-erd            # raw systemd status
systemctl --user status wordle-report-server  # raw systemd status
```

`start`, `stop`, and `restart` all act on both services, whichever of them
isn't already in the target state.  Pass `--swarm-only` to any of the three to
act on the supervisor alone and leave the report web server untouched, or
`--web-only` to act on the report web server alone (handy for iterating on the
web interface without disturbing the swarm):

```bash
python3.13 erd_search.py restart --swarm-only  # supervisor only
python3.13 erd_search.py restart --web-only    # report web server only
```

`restart` is a stop followed by a start in one step, per service.  Like
`start`, it prints the post-action service status (with `--no-pager`, so it
does not drop into a pager).

Stopping sends SIGTERM to the supervisor, which sets the stop event.  Workers
finish their current candidate evaluation (a few seconds at most) and exit
cleanly.  The supervisor waits up to 120 s before killing stragglers.

### Run directly (development / one-shot)

```bash
python3.13 erd_search.py run --workers 6
```

Output goes to `runtime/erd_search.log`.  Ctrl-C signals a clean shutdown identical to
SIGTERM.  Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--workers N` | 6 | Number of worker processes |
| `--recycle-hours H` | 3.0 | Restart each worker after H hours to bound per-worker memory growth |
| `--worker-timeout-seconds S` | 30 | Declare a worker dead after S seconds without a heartbeat |

---

## Monitor the swarm

### Unified report view

All read-only inspection uses `view`:

```bash
python3.13 erd_search.py view
python3.13 erd_search.py view --watch
python3.13 erd_search.py view --watch 10
python3.13 erd_search.py view --format json --queue
python3.13 erd_search.py view --format jsonl --watch 10 --workers
```

The default report is the operational overview. Watched text is interactive in
a TTY: branch letters and worker numbers open detail, Backspace or Escape goes
back, Space refreshes, and `q` quits. Non-TTY text and structured output remain
noninteractive. The overview includes filesystem fullness, queue WAL size, and
a fresh fill-rate estimate with time remaining to the disk-stop threshold.

Semantic branch targets infer words and branches from spine form:

```bash
python3.13 erd_search.py view CRANE
python3.13 erd_search.py view "CRANE .y..g"
python3.13 erd_search.py view "CRANE .y..g ALIBI"
python3.13 erd_search.py view "CRANE .y..g ALIBI g.g.." --claims
```

A trailing word reports its response groups. A trailing pattern reports the
resulting branch. Pattern syntax is `g` for green, `y` for yellow, and `.` or
`-` for gray. A displayed branch reference can also select a queued branch.

These are two forms of one thing: a **branch target** names a single branch by
either its **spine** — the explicit word/pattern steps — or a **branch
reference**, the `@`-prefixed hash handle shown in reports. A spine resolves
from the answer list alone and is durable; a reference resolves only while the
branch is still in the queue.

Focused collections and live topology use the same report model:

```bash
python3.13 erd_search.py view --queue --branch-status active,pending --sort size --limit 25
python3.13 erd_search.py view --queue --tree "CRANE .y..g"
python3.13 erd_search.py view --workers
python3.13 erd_search.py view --worker 2
python3.13 erd_search.py view --cache
python3.13 erd_search.py view --cache CRANE
python3.13 erd_search.py view --hotspots --by nodes
python3.13 erd_search.py view --hotspots --by coordination --since-seconds 900
python3.13 erd_search.py view --sources
python3.13 erd_search.py view --sources CRANE
```

`--sources` reports every source-work request: its recorded requested
priority, the request state, and every branch it owns — including branches
shared with another request, shown as one row per owning request with both
that request's own requested priority and the branch's effective (highest
live owner's) priority side by side, plus its promotion lineage (root
pattern and immediate parent). A trailing word narrows to that word's
request(s). Each worker's own report row shows whether it is serving its
preferred (highest-priority eligible) source or fallback work claimed
because the preferred source had no claimable bundle.

Use `--answers` for answer-word arrays on word or branch reports and `--claims`
for sparse candidate detail on one branch. Collection filters include branch
status, branch phase, answer-count bounds, budget, priority, sort, and limit.
Status tracks the branch's relationship to current work: active, pending, done,
or unqueued. Phase tracks durable search progress from queued through evaluating
and finalizing to complete. The two axes have a constrained set of combinations; run
`erd_search.py view --help` for their transition diagram. Historical hotspots
are explicitly bounded by epoch, time window, and sample size. `--tree` uses
only extant queue topology; cache rows never reconstruct historical trees.

### Log files

| File | Content |
|---|---|
| `runtime/erd_search.log` | Supervisor spawn, recycle, and queue-empty events |
| `runtime/erd_worker_N.log` | Per-worker candidate timing, finalize events, and RAM warnings |

```bash
tail -f runtime/erd_search.log
tail -f runtime/erd_worker_0.log
```

---

## Disk safety and WAL maintenance

Workers periodically checkpoint the queue WAL in PASSIVE mode. The supervisor
also checks the WAL every five seconds; above 2 GB it briefly asks workers to
stay off the queue database while it retries a TRUNCATE checkpoint. The pause
flag expires automatically after 60 seconds if the supervisor exits.

If the WAL nevertheless reaches the 32 GB hard ceiling (override with
`QUEUE_WAL_HARD_CEILING_GIB`), the supervisor captures each worker's current
candidate, signals live workers to dump all-thread stacks into their logs,
sets the disk-stop latch, and exits before the WAL can fill the filesystem.
Each worker also enforces the same ceiling on its own, so a worker that
outlives the supervisor stops itself instead of writing unsupervised, and
supervisor shutdown escalates to SIGKILL for any worker still alive 30 seconds
after SIGTERM. Workers log per-table WAL traffic attribution every 30 seconds
and once more on shutdown so the runaway writer is visible in the post-mortem.
At 90% filesystem use, the supervisor records a persistent disk-stop latch and
stops the swarm. `run` refuses to restart while the latch is set or the live
filesystem remains at the threshold. Free disk space first, then release it:

```bash
python3.13 erd_search.py queue clear-disk-stop
systemctl --user start wordle-erd
```

To hold the swarm down for maintenance across a reboot or systemd restart, set
the latch with a reason:

```bash
python3.13 erd_search.py queue set-disk-stop --reason "maintenance hold"
```

An existing latch remains unchanged so an automatic disk-fill or WAL-ceiling
reason is never replaced.

Completed candidate claims and the running best survive the restart; only
claims held by processes that stopped are made available again.

---

## Telemetry epochs

An epoch is the validity key for comparing swarm telemetry. Change it when a
claiming-regime change would make new measurements incomparable with the old
ones; this is the regime boundary described by the `telemetry_epoch` schema.

Inspect the active epoch and its label, Git SHA, start time, and notes with:

```bash
python3.13 erd_search.py epoch show
```

To create a new regime, stop the swarm, set the epoch, then start it again:

```bash
python3.13 erd_search.py stop --swarm-only
python3.13 erd_search.py epoch set 8 --label "healthy-post-145-redesign" \
    --notes "claiming-regime change"
python3.13 erd_search.py start --swarm-only
```

When run from a checkout, `epoch set` records the current abbreviated Git SHA
unless `--git-sha SHA` supplies one. Use `--label TEXT` and `--notes TEXT` to
describe the regime.

Every `ERDQueue` connection caches its epoch when it opens. `epoch set` refuses
while a worker heartbeat is live, because changing the database pointer before
workers restart would stamp two regimes during the recycle window. Stop the
swarm first. `--force` overrides that protection only for an intentional live
cutover; restart every worker immediately afterward.

### Branch-attributed claim telemetry

`telemetry.claim_telemetry` (in the attached telemetry file, see "Schema
coordination" in AGENTS.md) carries a `branch_id`/`spine`, `worker_id`,
`bundle_id`, and `idx` (with `bundle_start_idx`/`bundle_end_idx`) on every
row, so a slow branch's coordination cost can be attributed to it directly
instead of only to its `n_words`/epoch bucket. `branch_id` is the
`branches` registry surrogate (`_intern_branch`), not the raw `branch_key`
BLOB — at this table's row volume the BLOB would roughly double the bytes
per row, most of it a repeat of what the registry already carries. The
registry is append-only, so a `branch_id` here resolves back to its
`branch_key`/word-list indefinitely, including long after the branch
itself is finalized and its `active_branches` row is gone: `SELECT
branch_key FROM branches WHERE branch_id = ?`. `branches` lives in the
*main* queue file, though, not this attached telemetry one — a live
`ERDQueue` already has both open on one connection, so `WHERE branch_id =
?` (an index exists for this) or a join against `branches` works directly;
querying the telemetry file standalone (e.g. the `sqlite3` CLI) needs an
explicit `ATTACH 'erd_queue.sqlite3' AS q` first, then join against
`q.branches`. Query `WHERE spine LIKE ...` needs no such join, since
`spine` is small enough to carry directly on each row.
`coordination_millis` is also partitioned into
`claim_transaction_millis` (claim-scan and write, inside
`claim_next_bundle`'s transaction) + `claim_commit_millis` (its `COMMIT`) +
`busy_wait_millis` (write-lock wait, across every claim path taken while
coordinating — both `claim_next_bundle` and `claim_next`) +
`scheduling_millis` (the work-selection scan that chose this branch:
source-work ordering, pending promotion, joining an in-progress branch) +
`idle_millis` (the remainder); those five sum to `coordination_millis`
exactly.

All five measure time *between* candidate evaluations, which is what
`coordination_millis` spans. Queue work a candidate does during its own
evaluation — sub-branch promotion taking the write lock, for instance — is
inside the evaluation span, which `coordination_millis` excludes, so it is
deliberately not counted here; folding it in would make the parts exceed
the whole.

Scheduling is broken out rather than left in the remainder because it is
real work, and it grows with the number of source-work groups: folded into
`idle_millis` a large value reads as starved workers, when the true cause
may be that work selection is eating the window.  `idle_millis` therefore
means genuinely unaccounted wait.  One exception worth knowing: a scan that
finds nothing claimable is not billed to any branch — the worker had not
chosen one yet — so that time stays in the next row's `idle_millis`, which
is the correct reading for a worker that searched and found no work.

Every row is one candidate evaluation, so `COUNT(*)` is a claim count. The
finalize phase is deliberately *not* here: it belongs to a branch rather
than to any single claim, and is recorded once per branch as
`branch_finalize_log.cache_write_millis` (the score-cache/loss/cut writes
and the cost-model fold). `branch_finalize_log` carries the raw `branch_key`
directly (one row per branch, not per claim, so the BLOB there costs far
less) while `claim_telemetry` carries `branch_id`; join the two for a
branch's full coordination picture through `branches`: `claim_telemetry.
branch_id = branches.branch_id AND branches.branch_key =
branch_finalize_log.branch_key`.

The bucketed rollup of this table is exposed as `erd_search.py view --by
coordination` (aggregated by `n_words`/`worker_count`); the per-row branch
attribution above has no CLI reader yet, so query the telemetry file
directly (or a live `ERDQueue`'s `telemetry` attached schema) for it.

---

## Queue mutations

Read-only queue reporting uses `view --queue`. The `queue` group contains only
mutations.

### Add branches to the queue

```bash
# All branches for one opener word (every branch with >= 2 answer words):
python3.13 erd_search.py queue add --word salet

# One specific branch (word + response pattern):
python3.13 erd_search.py queue add --word salet --pattern .....

# All words in a file (queues every branch for every word, unbounded --
# including each opener's monster all-gray branch; pass --max-branch-size
# to bound a bulk run):
python3.13 erd_search.py queue add --word-list all_candidates.txt

# All words in a file, with a subset prioritized (others queued at 0):
python3.13 erd_search.py queue add --word-list all_candidates.txt \
    --priority-words salet crane --priority 1

# Bound a deliberately limited run to branches of at most 300 answer words:
python3.13 erd_search.py queue add --word salet --max-branch-size 300

# Force a recompute of an already-cached branch:
python3.13 erd_search.py queue add --word salet --pattern ..... \
    --delete-erd-cache --priority 1000
```

`queue add` is idempotent: already-queued branches are never duplicated, and
priority is upgraded (never downgraded) if the new request is higher.  Setting
a high priority on a large branch makes every idle worker in the running
swarm converge on it: `claim_one` prefers joining any in-progress branch
before promoting a new one, and both the pending and in-progress branch lists
are ordered by priority — there is no separate "dedicated worker" mechanism.

Pattern syntax: `g`=green, `y`=yellow, `-` or `.`=gray.  Use dots (not
dashes) for patterns that start with a gray position to avoid the shell/argparse
leading-dash trap (e.g. `--pattern .....` for all-gray, `--pattern =-y-g-` or
`--pattern=.y.g.`).

Priority values: 0 = default; 1 = high; use 0–999 for normal work.

### Change priority

```bash
python3.13 erd_search.py queue priority --word salet --pattern ..... --priority 1
```

Only affects branches with status `pending` (priority is read at claim time, so
changing it on an in-progress branch has no effect).

### Change a source-work request's priority

```bash
python3.13 erd_search.py queue source-priority --word salet --priority 1

# Disambiguate when the word owns more than one open request:
python3.13 erd_search.py queue source-priority --word salet --source-work-id 7 --priority 1
```

Unlike `queue priority`, which changes one branch, this changes a
source-work *request* — the whole `queue add --word salet` request that
covers all of `salet`'s branches. The new requested priority takes effect
immediately for both the request's still-pending roots and its
already-active/promoted descendants. A branch owned by more than one live
request keeps the higher of their requested priorities, so lowering one
request's priority does not necessarily lower a branch it shares with a
higher-priority request. A word that owns more than one open source-work
request is ambiguous; the command lists the candidate ids with enough detail
(priority, state, root/branch counts, request time) to choose, and
`--source-work-id` picks one — it can also name a completed request
directly, which is reported as such. A completed request cannot be
reprioritized.

### Remove a branch

```bash
# Remove from the pending queue (no-op if in-progress or done):
python3.13 erd_search.py queue remove --word salet --pattern .....

# Also cancel any in-progress work (workers move on after their current candidate):
python3.13 erd_search.py queue remove --word salet --pattern ..... --force
```

### Clear the entire queue

```bash
python3.13 erd_search.py queue clear       # prompts for confirmation
python3.13 erd_search.py queue clear --yes # skip prompt
```

Wipes pending, in-progress, done branches, candidate claims, heartbeats, and run
metadata.  The ERD cache (`runtime/wordle_cache.sqlite3`) is not touched.

### Reset stuck in-progress rows

```bash
python3.13 erd_search.py queue reset-stale
```

If a supervisor crash left `pending_branches` rows stuck in `in_progress`, this
resets them to `pending` so they are re-queued on the next run.  The supervisor
does this automatically on startup; use it manually only when the supervisor is
stopped and you want to inspect or requeue before restarting.

### Reconcile orphaned branch ownership

```bash
python3.13 erd_search.py queue reconcile-orphaned-ownership
```

A branch promoted under a source-work request can lose every membership that
justified `requires_source_membership`, while itself staying `open`: bulk
elimination retracts the in-flight candidate that promoted it, or the request
completes moments before a racing `create_branch` call attaches it.  Either
way the branch becomes invisible to every claim path and is flagged by
`check_source_work_invariants()` as a "source-owned open branch ... has no
live membership" violation.  The supervisor self-heals this automatically on
every membership resolution (see `_resolve_branch_memberships` in
`erd_queue.py`); use this command only to reconcile branches stranded before
that existed, or accumulated while the swarm was down.

---

## Cache operations

Cache coverage inspection uses `erd_search.py view --cache` with an optional semantic branch target.

### Export for the iPhone

```bash
python3.13 export_cache.py
python3.13 export_cache.py --output wordle_erd_export.sqlite3
```

Creates a trimmed snapshot (`wordle_erd_export.sqlite3`) containing
`answer_list`, `response_decomposition`, `branch_best_by_policy`, and
`candidate_scores` — a phone without a cached ERD result for its current
position still needs `candidate_scores`' entropy/max-group-size numbers to
rank candidates, and that isn't limited to the opening guess, so the whole
table is carried.  Safe to run while workers are active.  Re-running is
incremental (INSERT OR IGNORE skips rows already present).

### Import a cache from another machine

```bash
python3.13 import_cache.py <source_db> [--target runtime/wordle_cache.sqlite3] [--dry-run]
```

Creates the target cache if it doesn't exist yet, or merges into it if it
already exists: adds rows from `<source_db>` not already present in the
local cache.  Before merging, the target is opened once through
`ScoreCache` itself, so a table under a pre-rename legacy name is migrated
in place first — e.g. importing an `export_cache.py` snapshot straight onto
a fresh device produces a cache schema-identical to one that's always been
managed normally.  An incoming unconstrained (untainted) entry replaces an
existing tainted (depth-limited) one for the same branch, since the
unconstrained result is strictly more reusable.  Run `--dry-run` first to
preview row counts.  Prefer to run while workers are stopped.

---

## Deployment sequence for breaking changes

When deploying schema changes or engine changes that would corrupt cached state
if old workers keep running:

1. Stop the supervisor: `python3.13 erd_search.py stop`
2. Verify workers are gone: `systemctl --user status wordle-erd`
3. Deploy the new code.
4. Start the supervisor: `python3.13 erd_search.py start`

Never let old workers outlive a schema change — they will write rows the new
code cannot interpret.
