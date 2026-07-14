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

### Live status display

```bash
python3.13 erd_search.py status             # one-shot snapshot
python3.13 erd_search.py status --watch     # refresh every 30 s
python3.13 erd_search.py status --watch 10  # refresh every 10 s
```

The display has three sections:

**Header** — queue counts (pending / done / in-progress), cache ERD entry
count, and a disk line:

```
Disk: 53.2G/387G (14%)  queue WAL 1.31G  filling 34.1 MB/s: 90% in ~2.1 h
```

Fullness is live (`df` semantics on the filesystem holding the queue).  Above
80% the figure is drawn in red.  The fill rate and the time remaining until
the 90% stop threshold come from samples the supervisor records every 30 s;
with no fresh samples (swarm stopped) only fullness and WAL size appear.

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

## Disk safety and WAL maintenance

The queue database's WAL grows as long as readers overlap (SQLite can only
recycle it in a moment with no read snapshot inside it), so a busy swarm needs
deliberate reclamation:

- **Workers** checkpoint PASSIVE only (backfill without blocking anyone), on
  jittered ~5-minute intervals.
- **The supervisor** runs a PASSIVE checkpoint every 5 minutes, and when the
  WAL exceeds 2 GB it quiesces: it sets a `checkpoint_pause` flag that workers
  honour between claims and heartbeats (evaluation compute continues), retries
  `wal_checkpoint(TRUNCATE)` for up to 15 s, and clears the flag.  The flag
  self-expires after 60 s, so a dead supervisor cannot wedge the swarm.
  Truncation results (or failures) are logged in `erd_search.log`.

### Disk-stop latch (90%)

If the filesystem holding the queue reaches **90% full**, the swarm stops and
latches down: the supervisor (or any worker, as a backstop) writes a
`disk_stop` row into `run_meta`, all processes exit cleanly, and `run` refuses
to start while the latch is set — including via systemd restarts and reboots.
The reserved 10% keeps the rest of the OS healthy and leaves room to diagnose.

To bring the swarm back, free disk space first, then release the latch:

```bash
python3.13 erd_search.py queue clear-disk-stop
systemctl --user start wordle-erd
```

Restarts do not discard branch progress: completed candidate claims and the
branch's running best survive; only claims that were in flight in a killed
process are freed for re-claim.

---

## Queue operations

Start with the queue dashboard when you do not already know the branch:

```bash
python3.13 erd_search.py queue
```

Use `queue ls` to find work, `queue tree <partial-spine>` to understand promoted
children, `queue show <branch-ref>` to inspect one branch, and
`queue coverage <partial-spine>` when asking which response branches under a
path are queued.

Branch references are queue-first spine fragments:

```bash
CRANE
CRANE -y--g
CRANE -y--g ALIBI
CRANE -y--g ALIBI g-g--
```

The final word may omit a pattern, meaning "show branches below this guess."
Pattern syntax is `g`=green, `y`=yellow, and any other character as gray; quote
refs containing leading dashes so the shell does not treat them as options.

### Find and inspect work

```bash
python3.13 erd_search.py queue ls
python3.13 erd_search.py queue tree "CRANE -y--g"
python3.13 erd_search.py queue show "CRANE -y--g ALIBI"
python3.13 erd_search.py queue top --by nodes "CRANE -y--g"
python3.13 erd_search.py queue summary
python3.13 erd_search.py queue coverage CRANE
```

`queue show` accepts the 4-hex branch id printed by `queue ls`, a full branch
key prefix, a word/pattern pair, or a partial/full spine. If a reference matches
multiple branches, it prints a disambiguation table.

#### `queue`: dashboard

```bash
python3.13 erd_search.py queue
python3.13 erd_search.py queue --limit 20
python3.13 erd_search.py queue --json
```

The dashboard is the default read-only entry point. It shows aggregate queue
counts, active branches, top pending branches, and stale/held work when present.
Use it before you know which word or branch you care about.

#### `queue ls`: inventory

```bash
python3.13 erd_search.py queue ls
python3.13 erd_search.py queue ls --status pending --min-words 100
python3.13 erd_search.py queue ls --source-word crane --limit 50
python3.13 erd_search.py queue ls --prefix "CRANE -y--g" --json
```

Lists queue rows without requiring a word first. Rows include the stable 4-hex
branch id, kind (`user` or `coop`), status, priority, answer count, candidate
progress, live worker count, nodes spent, and spine/source path.

Useful filters:

| Filter | Meaning |
|---|---|
| `--status pending|in_progress|done|open` | Limit by pending-row or active-row status |
| `--min-words N`, `--max-words N` | Limit by answer count |
| `--budget N` | Limit by active solve budget |
| `--priority N` | Limit by exact priority |
| `--source-word WORD` | Limit to branches first queued from that word |
| `--prefix SPINE` | Limit to descendants of a partial spine |
| `--limit N` | Cap displayed rows |
| `--json` | Emit machine-readable rows |

Default sort is active work first, then priority descending, then branch size
descending.

#### `queue tree`: spine view

```bash
python3.13 erd_search.py queue tree
python3.13 erd_search.py queue tree CRANE
python3.13 erd_search.py queue tree "CRANE -y--g ALIBI"
python3.13 erd_search.py queue tree --active-only --max-depth 3
```

Groups work by recorded spine so promoted cooperative children are easier to
understand. Use this when a branch has spawned sub-work and `queue ls` is too
flat. `--active-only`, `--max-depth`, `--limit`, and `--json` are supported.

#### `queue show`: branch drill-down

```bash
python3.13 erd_search.py queue show 04d6
python3.13 erd_search.py queue show "CRANE -----"
python3.13 erd_search.py queue show "CRANE -y--g ALIBI"
python3.13 erd_search.py queue show --claims 04d6
```

Shows one branch’s pending row, active row, candidate progress, workers, bundle
stats, republish count, current best guess/ERD, budget, taint flag, nodes spent,
and spine. `--claims` includes detailed candidate claim rows. If the reference
is ambiguous, it prints matching rows and asks for a more specific spine/pattern
or branch id.

#### `queue summary`: aggregate view

```bash
python3.13 erd_search.py queue summary
python3.13 erd_search.py queue summary --json
```

Reports counts by status, kind, budget, priority bucket, and answer-count
bucket, plus largest/oldest pending and active branches. This is the quickest
way to see queue shape without row-level detail.

#### `queue top`: hotspots

```bash
python3.13 erd_search.py queue top --by nodes
python3.13 erd_search.py queue top --by workers "CRANE -y--g"
python3.13 erd_search.py queue top --by size --limit 25
```

Ranks active/open work. `--by` accepts `nodes`, `age`, `size`, `workers`,
`priority`, or `slowest`. A trailing partial spine filters to descendants.

#### `queue coverage`: response-pattern coverage

```bash
python3.13 erd_search.py queue coverage CRANE
python3.13 erd_search.py queue coverage "CRANE -y--g ALIBI"
python3.13 erd_search.py queue coverage CRANE --queued-only
python3.13 erd_search.py queue coverage CRANE --missing-only
```

This is the old word-centric coverage question under the new queue group. It
answers “for the next guess at this path, which response branches are pending,
in progress, done, cooperative-active, or not queued?” Use it when checking
whether a word/path has complete queue coverage rather than when looking for
unknown work.

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
