# Swarm Operations Guide

The ERD_ALL precache is built by `erd_search.py`, a CLI that manages a
supervisor process and its worker pool.  All commands run from the `wordle/`
directory with `python3.13 erd_search.py <subcommand>`.

Two SQLite files are involved:

| File | Purpose |
|---|---|
| `wordle_cache.sqlite3` | Durable ERD results; shared with the iPhone |
| `erd_queue.sqlite3` | Transient coordination only (queue, candidate claims, heartbeats) |

`queue clear` wipes only the queue file — the cache is never touched by any
queue command.

---

## Service: start, stop, status

The supervisor runs as a systemd user service named `wordle-erd`.

### Install the service (one-time)

```bash
ln -s ~/work/wordle/wordle-erd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable wordle-erd   # start automatically on login
```

### Start and stop

```bash
python3.13 erd_search.py start    # systemctl --user start wordle-erd
python3.13 erd_search.py stop     # systemctl --user stop wordle-erd
python3.13 erd_search.py restart  # systemctl --user restart wordle-erd

systemctl --user status wordle-erd   # raw systemd status
```

`restart` is a stop followed by a start in one step.  Like `start`, it prints
the post-action service status (with `--no-pager`, so it does not drop into a
pager).

Stopping sends SIGTERM to the supervisor, which sets the stop event.  Workers
finish their current candidate evaluation (a few seconds at most) and exit
cleanly.  The supervisor waits up to 120 s before killing stragglers.

### Run directly (development / one-shot)

```bash
python3.13 erd_search.py run --workers 6
```

Output goes to `erd_search.log`.  Ctrl-C signals a clean shutdown identical to
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

Semantic selectors infer words and branches from spine form:

```bash
python3.13 erd_search.py view CRANE
python3.13 erd_search.py view "CRANE .y..g"
python3.13 erd_search.py view "CRANE .y..g ALIBI"
python3.13 erd_search.py view "CRANE .y..g ALIBI g.g.." --claims
```

A trailing word reports its response groups. A trailing pattern reports the
resulting branch. Pattern syntax is `g` for green, `y` for yellow, and `.` or
`-` for gray. A displayed branch reference can also select a queued branch.

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
```

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
| `erd_search.log` | Supervisor spawn, recycle, and queue-empty events |
| `erd_worker_N.log` | Per-worker candidate timing, finalize events, and RAM warnings |

```bash
tail -f erd_search.log
tail -f erd_worker_0.log
```

---

## Disk safety and WAL maintenance

Workers periodically checkpoint the queue WAL in PASSIVE mode. The supervisor
also checks the WAL every five seconds; above 2 GB it briefly asks workers to
stay off the queue database while it retries a TRUNCATE checkpoint. The pause
flag expires automatically after 60 seconds if the supervisor exits.

At 90% filesystem use, the supervisor records a persistent disk-stop latch and
stops the swarm. `run` refuses to restart while the latch is set or the live
filesystem remains at the threshold. Free disk space first, then release it:

```bash
python3.13 erd_search.py queue clear-disk-stop
systemctl --user start wordle-erd
```

Completed candidate claims and the running best survive the restart; only
claims held by processes that stopped are made available again.

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
python3.13 erd_search.py queue add --word-list wordle.txt

# All words in a file, with a subset prioritized (others queued at 0):
python3.13 erd_search.py queue add --word-list wordle.txt \
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

Priority values: 0 = default; 1 = high; use 0–999 for normal work.  The swarm
internally uses 1,000,000 for promoted sub-branches so they always drain before
fresh top-level branches.

### Change priority

```bash
python3.13 erd_search.py queue priority --word salet --pattern ..... --priority 1
```

Only affects branches with status `pending` (priority is read at claim time, so
changing it on an in-progress branch has no effect).

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
metadata.  The ERD cache (`wordle_cache.sqlite3`) is not touched.

### Reset stuck in-progress rows

```bash
python3.13 erd_search.py queue reset-stale
```

If a supervisor crash left `pending_branches` rows stuck in `in_progress`, this
resets them to `pending` so they are re-queued on the next run.  The supervisor
does this automatically on startup; use it manually only when the supervisor is
stopped and you want to inspect or requeue before restarting.

---

## Cache operations

Cache coverage inspection uses `erd_search.py view --cache` with an optional semantic selector.

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
python3.13 import_cache.py <source_db> [--target wordle_cache.sqlite3] [--dry-run]
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
