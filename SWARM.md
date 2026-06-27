# Swarm Operations Guide

The ERD_ALL precache is built by `erd_search.py`, a CLI that manages a
supervisor process and its worker pool.  All commands run from the `wordle/`
directory with `python3.13 erd_search.py <subcommand>`.

Two SQLite files are involved:

| File | Purpose |
|---|---|
| `wordle_cache.sqlite3` | Durable ERD results; shared with the iPhone |
| `erd_queue.sqlite3` | Transient coordination only (queue, candidate claims, heartbeats) |

`queue-clear` wipes only the queue file — the cache is never touched by any
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

### Live status display

```bash
python3.13 erd_search.py status             # one-shot snapshot
python3.13 erd_search.py status --watch     # refresh every 30 s
python3.13 erd_search.py status --watch 10  # refresh every 10 s
```

The display has three sections:

**Header** — queue counts (pending / done / in-progress) and cache ERD entry count.

**Branches in progress** — one row per branch currently being swarmed.  Columns:
`Source` (opener word + response pattern), `Ans` (answer-word count),
`Cands` (done/total and percent), `Best guess` (running-best candidate),
`ERD` (running-best score), `Wkrs` (active worker count), `ETA`.

**Workers** — one row per worker with its liveness heartbeat age (`hb=Ns`),
the branch it is on, claim index held and hold time, total claims done, and the
candidate currently under evaluation (`[WORD N/total depth D M evals K/s
path:X>Y>Z]`).  A `!!STALE` flag appears when a heartbeat is more than 120 s
old.  A `!!HANG` flag appears when the heartbeat is fresh but the node rate is
zero (evaluation stuck).

### Log files

| File | Content |
|---|---|
| `erd_search.log` | Supervisor: spawn/recycle events, queue-empty signal |
| `erd_worker_N.log` | Per-worker: candidate timing, finalize events, RAM warnings |

```bash
tail -f erd_search.log
tail -f erd_worker_0.log
```

Workers log a summary line after each candidate claim completes:
```
claim N done: K nodes in T.1s (R.1/s)  ok=X pruned=Y useless=Z  best=WORD E.EEEE
```

---

## Queue operations

### Add branches to the queue

```bash
# All branches for one opener word (skips branches with >300 answer words):
python3.13 erd_search.py queue-add --word salet

# One specific branch (word + response pattern):
python3.13 erd_search.py queue-add --word salet --pattern .....

# All words in a file:
python3.13 erd_search.py queue-add --word-list wordle.txt

# All words in a file, with a subset prioritized (others queued at 0):
python3.13 erd_search.py queue-add --word-list wordle.txt \
    --priority-words salet crane --priority 1

# A branch too large for the default cap, pushed to the front of the queue:
python3.13 erd_search.py queue-add --word salet --pattern ..... \
    --max-branch-size 999 --priority 1000

# Force a recompute of an already-cached branch:
python3.13 erd_search.py queue-add --word salet --pattern ..... \
    --delete-erd-cache --priority 1000
```

`queue-add` is idempotent: already-queued branches are never duplicated, and
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

### Inspect a branch

```bash
python3.13 erd_search.py queue-inspect --word salet --pattern .....
```

Shows the pending-queue status (pending / in_progress / done), priority, and
— if the branch is currently active — the candidate claim table with which
worker holds each claim.

### Change priority

```bash
python3.13 erd_search.py queue-priority --word salet --pattern ..... --priority 1
```

Only affects branches with status `pending` (priority is read at claim time, so
changing it on an in-progress branch has no effect).

### Remove a branch

```bash
# Remove from the pending queue (no-op if in-progress or done):
python3.13 erd_search.py queue-remove --word salet --pattern .....

# Also cancel any in-progress work (workers move on after their current candidate):
python3.13 erd_search.py queue-remove --word salet --pattern ..... --force
```

### Clear the entire queue

```bash
python3.13 erd_search.py queue-clear       # prompts for confirmation
python3.13 erd_search.py queue-clear --yes # skip prompt
```

Wipes pending, in-progress, done branches, candidate claims, heartbeats, and run
metadata.  The ERD cache (`wordle_cache.sqlite3`) is not touched.

### Reset stuck in-progress rows

```bash
python3.13 erd_search.py reset-stale
```

If a supervisor crash left `pending_branches` rows stuck in `in_progress`, this
resets them to `pending` so they are re-queued on the next run.  The supervisor
does this automatically on startup; use it manually only when the supervisor is
stopped and you want to inspect or requeue before restarting.

---

## Cache operations

### Check ERD coverage for a word

```bash
python3.13 erd_search.py cache-status --word salet
python3.13 erd_search.py cache-status --word salet --missing-only
```

For each of the (up to 242) response patterns for WORD, reports whether the
branch has a cached ERD entry, along with the best guess, score, and timestamp
for hits.  Trivial patterns (0 or 1 answer word) are skipped — they need no
ERD.

### Export for the iPhone

```bash
python3.13 erd_search.py export
python3.13 erd_search.py export --output wordle_erd_export.sqlite3
```

Creates a trimmed snapshot (`wordle_erd_export.sqlite3`) containing only the
three iPhone-relevant tables: `answer_list`, `response_decomposition`,
`branch_best_by_policy`.  Safe to run while workers are active.  Re-running is
incremental (INSERT OR IGNORE skips rows already present).

### Merge a cache from another machine

```bash
python3.13 merge_cache.py <source_db> [--target wordle_cache.sqlite3] [--dry-run]
```

Adds rows from `<source_db>` not already present in the local cache.  An
incoming unconstrained (untainted) entry replaces an existing tainted
(depth-limited) one for the same branch, since the unconstrained result is
strictly more reusable.  Run `--dry-run` first to preview row counts.  Prefer
to run while workers are stopped.

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
