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
python3.13 erd_search.py view --openers
python3.13 erd_search.py view --openers CRANE
python3.13 erd_search.py view --root-progress CRANE
python3.13 erd_search.py view --root-progress CRANE --epoch 10
```

`--root-progress` reports one opener's work: every response group with the
branches, search nodes, and node share spent under it, which groups have not
been opened at all, when work actually began on it, and a completion estimate.
It is the report that answers "why is
this root taking so long" — cost concentrates hard, and the node-share column
names the group holding it. It takes any spine ending in a word: a bare root,
or a deeper spine such as `SCOPE -y--- LUBES`, which asks the same question at
a greater guess_depth. The rollup scopes telemetry by spine prefix, so a longer
spine is simply a longer prefix.

The `State` column says where each response group sits: `waiting` (no work
opened), `working`, `solved` (a proven line), or `loss` (proven unsolvable
within budget). `solved` and `loss` both mean there is no more work to do here,
for opposite reasons, so they stay apart. State comes from the cache, not the
queue, because a group can be solved with no branch open and nothing finalized
in this epoch — a group of one answer needs no search at all.

This matters for reading the estimate. The estimate excludes groups still
`waiting`, not groups the queue never opened: SCOPE has 34 groups it never
opened, and every one of them is already solved (29 hold a single answer). They
are not a backlog, and counting them as one would invent work that does not
exist.

A group counts as started once any branch has opened on it, finalized or not,
so a group being worked right now never reads as untouched. The two branch
columns are named for the lifecycle phases the rest of the report uses:
`Evaluating` counts branches in flight, `Done` counts branches that have
finalized. Both span every depth under the group, so promoted sub-branches are
included — unlike the word report's own response-group count, which sees only
the root's direct groups. A started group that has finalized nothing shows
measured zeros for those columns but `—` for elapsed and worker-time, which
exist only at finalize; an unopened group shows `—` throughout.

Node, share, elapsed and worker-time figures are fenced to one telemetry epoch,
which the report names. Work that finalized under an earlier epoch is not
counted, so a root whose work began before the current epoch shows a start time
older than any of its costs.

A request time appears only when the queue holds one that precedes the work it
asked for. Rebuilding the queue's `source_work` rows restamps every one of them
with the rebuild's clock while the branches keep their true creation times,
which leaves the recorded request later than the work; that stamp is dropped
rather than displayed.

Two time bases appear per group and they answer different questions. `Elapsed`
is wall-clock from the group's first branch to its last, which the swarm's
other work shares; `WorkerTime` is summed across bundles, so six workers for
an hour reads as six hours. Their ratio is a coarse read on parallelism drawn.

The estimate covers only branches with observed throughput, and says how many
waiting groups and stalled branches it excludes. Waiting groups are not
estimated: the cost model is keyed on `(size, budget)` and branches of
near-identical size differ in cost by orders of magnitude, so it cannot rank
them. Groups the swarm has not opened show `—`, never `0`.

The rollup scans the epoch's `branch_finalize_log` rows, which carry no spine
index, so it takes seconds. The web client fetches it after the word report
renders and caches it per target; the terminal report pays it on each run.

`--openers` reports one row per opener — ten queued openers are ten
rows, whatever the branch count underneath them. Each row carries the word's
requested priority, its state, its own ERD, how many branches it has ever
owned, how many of those are still open versus done, the live workers on them,
and how long ago it was requested.

The `ERD` column is the word's own expected remaining depth, folded from the
cached result of each of its response groups the way the word report folds it:
an exact value once every group is solved, `∞` once one is proven unsolvable,
and `solved/total` groups while it is still being searched. The fold is done
only for the rows on the page — about 6 ms a word against the full answer list
— so a long queue costs no more than a short one. In the browser the same
number is on each card, and clicking the card opens that word's full report,
where its response groups and root progress are; the card's own `Branches`
control is what lists the branches it owns. `Direct` counts the branches the word asked for outright
— its own response groups; `Branches` spans every depth beneath it, so
sub-branches promoted during the search are included too, and the two are
equal until promotion starts.

Opener work is keyed by (word, priority), so queueing one word twice at
different priorities makes two requests. They merge into that word's single
row: a `Reqs` column appears, the priority shown is the highest (the one that
schedules), and a branch both requests own is counted once.

`--opener-state` narrows to `queued`, `active`, `complete`, or `all` — a
word's own lifecycle, not a branch's, so it is a separate filter from
`--branch-status`. When it hides anything the count reads `Openers: 3 of
10`, so a filtered report never looks like the whole queue. `--sort` takes
`word`, `priority` (the default, and the order the swarm serves them in),
`branches`, `open`, `done`, `workers`, or `age`; those four opener-only sorts
are rejected for other reports rather than silently ignored. `--limit` caps
the rows the terminal prints, and the count says so (`Openers: 3 of 10`).

Grouping and paging are browser-only. The Openers tab groups by state out of
the box — what is running, what is waiting, what is finished — and can group by
worker presence or priority instead, or not at all; each group collapses under
a rollup of the words it holds. There, `limit` is a page size rather than a cap: with `opener_offset`
it pages the word list (`Showing 6–10 of 12 words`), so the words past the
first page stay reachable. Changing a filter, the sort or the grouping returns
to the first page, since page 3 of one ordering is not page 3 of another.

A trailing word narrows to that word's request(s) **and** lists the branches
it owns — including branches shared with another request, shown as one row per
owning request with both that request's own requested priority and the
branch's effective (highest live owner's) priority side by side, plus its
promotion lineage (root pattern and immediate parent). That per-branch list is
deliberately opt-in: a root can own hundreds of branches, and printing them
for every word buries the words themselves.

Each worker's own report row shows whether it is serving its preferred
(highest-priority eligible) opener or fallback work claimed because the
preferred opener had no claimable bundle. The browser report serves the same
view under its Openers tab — one card per opener, and clicking one opens
that word's branches — and names the same scheduling role on every worker
card.

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
opener-work ordering, pending promotion, joining an in-progress branch) +
`idle_millis` (the remainder); those five sum to `coordination_millis`
exactly.

All five measure time *between* candidate evaluations, which is what
`coordination_millis` spans. Queue work a candidate does during its own
evaluation — sub-branch promotion taking the write lock, for instance — is
inside the evaluation span, which `coordination_millis` excludes, so it is
deliberately not counted here; folding it in would make the parts exceed
the whole.

Scheduling is broken out rather than left in the remainder because it is
real work, and it grows with the number of opener-work groups: folded into
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

# Multiple words in one call (space-separated after a single --word):
python3.13 erd_search.py queue add --word salet crane raise

# One specific branch (word + response pattern):
python3.13 erd_search.py queue add --word salet --pattern .....

# All words in a file (queues every branch for every word, unbounded --
# including each opener's monster all-gray branch; pass --max-branch-size
# to bound a bulk run):
python3.13 erd_search.py queue add --words-file all_candidates.txt

# All words in a file, with a subset laddered above the rest (others at 0):
python3.13 erd_search.py queue add --words-file all_candidates.txt \
    --priority-words salet crane --priority 1

# Bound a deliberately limited run to branches of at most 300 answer words:
python3.13 erd_search.py queue add --word salet --max-branch-size 300

# Force a recompute of an already-cached branch:
python3.13 erd_search.py queue add --word salet --pattern ..... \
    --delete-erd-cache --priority 1000
```

`queue add` is idempotent: already-queued branches are never duplicated, and
priority is upgraded (never downgraded) if the new request is higher.  Setting
a high priority on an opener makes every idle worker in the running swarm
converge on that opener, one worker to a branch: `claim_one` takes an
unoccupied branch of the highest-priority source that has one, and otherwise
promotes another of that source's pending branches, so the workers spread
across the opener's response groups rather than stacking on one of them.  Both
the pending and in-progress branch lists are ordered by priority — there is no
separate "dedicated worker" mechanism.

A worker joins a branch another worker already holds only when no unoccupied
branch is available anywhere, and never as the third worker on one branch.
Workers sharing a branch race against a stale `best_erd` ceiling and explore
subtrees a sequential search prunes, so at the full candidate list six workers
on one branch finish slower than one worker alone.

### The priority ladder

Words are queued on a descending ladder in the order given: the first word
gets the highest priority and each subsequent word drops by `--priority-step`
(default 5).  The gap leaves room to reorder one word later with `queue
opener-priority` without disturbing its neighbours.

The ladder exists because words tied at one priority all become eligible at
once.  A worker that blocks on a dependency looks for useful work elsewhere,
and it prefers starting an opener with no branch open yet over opening another
branch of the opener already running — so a flat batch of N words fans out
into N simultaneously-active words, one per blocking event, each served by a
single branch.  Distinct priorities break every tie, holding the swarm on one
word until that word has no claimable work left.

`queue add` **appends**.  With no `--priority`, a batch descends from just
below the lowest priority the queue still owes work, so adding words never
preempts words already queued:

```bash
# Into an empty queue -- takes the top of the range:
python3.13 erd_search.py queue add --word salet crane raise
#   salet=999,999  crane=999,994  raise=999,989

# A later batch lands underneath, untouched by the first:
python3.13 erd_search.py queue add --word tulip video
#   tulip=999,988  video=999,983

# Ladder 20 apart instead of 5:
python3.13 erd_search.py queue add --word salet crane raise --priority-step 20

# Flat batch (every word starts at once) -- the pre-ladder behaviour:
python3.13 erd_search.py queue add --word salet crane raise --priority-step 0
```

Ladders run downward from the top of the range rather than upward from its
floor; that is what leaves room beneath each batch for the next one to append
into.  `queue add` reports the rungs it took and the priority it queued behind.

The append ceiling is `lowest_unfinished_source_priority() - 1`, so it
**ratchets downward** as batches accumulate and only returns to the top of the
range once the queue fully drains.  A long-running sweep therefore works from
whatever the last batch left, not from 999,999 — `queue add` reports the
priority it queued behind, so check that line before queueing a large batch.

Naming `--priority` opts out of appending: it fixes the *last* word's rung, so
the batch is placed wherever you ask — including ahead of queued work.  To jump
a word to the front, name a priority above whatever the current top rung is
(there is no headroom above a batch that took `SOURCE_PRIORITY_MAX` itself):

```bash
# With queued work topping out at 999,983, this runs ahead of all of it:
python3.13 erd_search.py queue add --word rocky --priority 999990
```

`--priority` is honoured exactly: `queue add` refuses rather than seating the
batch lower, so a request whose ladder would run past 999,999 is an error
naming the rung it would have needed, not a silent demotion.

A list too long to seat on distinct rungs above 0 gives them to the leading
words and ties the remainder on the minimum; `queue add` says so when that
happens.  The tail is undifferentiated but still ranks below every seated
word, and can be re-laddered later with `queue opener-priority`.  Appending
onto queued work that already sits at priority 0 has nowhere to go at all: the
batch ties with it, and `queue add` reports the tie rather than claiming to
rank below it.

Pattern syntax: `g`=green, `y`=yellow, `-` or `.`=gray.  Use dots (not
dashes) for patterns that start with a gray position to avoid the shell/argparse
leading-dash trap (e.g. `--pattern .....` for all-gray, `--pattern =-y-g-` or
`--pattern=.y.g.`).

Priority values: 0–999,999 for requested work.  The range seats one rung per
opener with room to spare — the full candidate list is ~15,000 words, so
a ladder of every candidate at the default step of 5 occupies 75,000 of the
million values and leaves the rest for appending beneath and inserting above.
Priorities at or above 1,000,000 are the legacy promoted band and never
preempt requested work.

### Change priority

```bash
python3.13 erd_search.py queue priority --word salet --pattern ..... --priority 1

# Repair open branches with no live opener-work membership for one opener:
python3.13 erd_search.py queue priority --opener-word salet --priority 1
```

The `--word`/`--pattern` form affects only branches with status `pending`.
The `--opener-word` form affects only open branches without a live opener-work
membership, including active branches left by older queue data. It never changes
branches owned by a live request; use `queue opener-priority` for those.

### Change an opener-work request's priority

```bash
python3.13 erd_search.py queue opener-priority --word salet --priority 1

# Disambiguate when the word owns more than one open request:
python3.13 erd_search.py queue opener-priority --word salet --opener-work-id 7 --priority 1
```

Unlike `queue priority`, which changes one branch, this changes an
opener-work *request* — the whole `queue add --word salet` request that
covers all of `salet`'s branches. The new requested priority takes effect
immediately for both the request's still-pending roots and its
already-active/promoted descendants. A branch owned by more than one live
request keeps the higher of their requested priorities, so lowering one
request's priority does not necessarily lower a branch it shares with a
higher-priority request. A word that owns more than one open opener-work
request is ambiguous; the command lists the candidate ids with enough detail
(priority, state, root/branch counts, request time) to choose, and
`--opener-work-id` picks one — it can also name a completed request
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

A branch promoted under an opener-work request can lose every membership that
justified its promotion, while itself staying `open`: bulk elimination
retracts the in-flight candidate that promoted it, or the request completes
moments before a racing `create_branch` call attaches it.  Either way the
branch becomes invisible to every claim path and is flagged as an
opener-owned open branch with no live membership.  The supervisor self-heals
this automatically on every membership resolution (see
`_resolve_branch_memberships` in `erd_queue.py`); use this command only to
reconcile branches stranded before that existed, or accumulated while the
swarm was down.

---

## Cache operations

Cache coverage inspection uses `erd_search.py view --cache` with an optional semantic branch target.

### A branch's two kinds of exact result

`branch_best_by_policy` holds one row per branch: the **unrestricted optimum**,
the best strategy over all strategies, reusable at any remaining budget its own
`max_depth` can meet.  `branch_best_by_policy_and_budget` holds the
**budget-specific** results, keyed by `(branch_key, policy, answer_list_id,
solve_budget)` — the optimum among strategies feasible at exactly that budget.

Both can be right and differ, which is why they are not one row.  Sharing a key
made either write destroy the other, after ancestors may already have folded
the value it displaced, and nothing records which one they folded (issue #302).

A search at budget `b` reads the unrestricted result first and takes it when
`max_depth <= b`: globally optimal is also optimal within any budget it can
meet.  Only when it does not fit is the budget-specific result consulted, and
only the one solved at exactly `b` — a row from another budget is optimal
against a different set of feasible strategies.  An unlimited search reads the
unrestricted table alone.  `ScoreCache.read_for_budget` makes that selection;
`wordle_engine._cache_reuse` remains the one place the rule is stated.

**A second exact result for a branch at a scope it already holds does not
replace it.**  Two exact searches of one scope agree on the cost, and
equal-cost strategies can still differ in `max_depth`, so overwriting would
leave every ancestor that folded the stored depth describing a subtree the
cache no longer holds.  A second result that *disagrees* on the cost cannot be
reconciled that way and raises `CacheWriteConflict` — the two searches cannot
both be right, and recording either invalidates whichever ancestors folded the
other.  Expect that never to fire; if it does, the log line names the branch,
policy, budget and both values.

Counts say which they mean.  `exact_branch_count` counts branches with an
unrestricted result; `budgeted_result_count` counts budget-specific results and
`budgeted_branch_count` the branches holding them.  Never union the two tables
for a branch count — a branch with results at three budgets is one branch.

**Migration.**  Opening a pre-split cache moves its `solve_budget IS NOT NULL`
rows into the budget table and clears `candidate_erd_by_policy`, whose every
row memoised a fold under the old branch-row identity.  The canonical table is
not rebuilt, so the migration is a row move on the minority rather than a
rewrite of a multi-GB file.

**Deploy before syncing.**  The canonical table's shape is unchanged, so an
older reader handed a newer export still consumes the unrestricted rows it
understands and ignores the budget table it does not.  A newer importer routes
an older writer's budget-specific rows — which arrive in the canonical table —
into the budget table, never into the canonical one.  Deploy the new code
before merging any export into a migrated cache.

**The quarantined cache is hints-only.**  Moving its rows under the new schema
does not certify them: nothing in the migration re-derives a value or repairs
an ancestor that folded a displaced one.  Treat that file as candidate-ordering
hints and write clean exact results under the new schema.

### Audit the max_depth column

```bash
python3.13 verify_branch_depths.py
python3.13 verify_branch_depths.py --list 20
python3.13 verify_branch_depths.py --repair
```

A branch row's `max_depth` is determined by its own `best_guess` and the
`max_depth` of each response group that guess produces, so folding it back up
turns any disagreement into a finding rather than an opinion.  It matters
because `branch_best_by_policy` keys a branch without `solve_budget`: a
branch's tainted and untainted values compete for one row, and an ancestor
that folded the value the last write replaced is left describing a subtree the
cache no longer holds.  Nothing records which value a parent folded, so those
ancestors are reachable only by redoing the fold.

Stored below the fold is the direction that matters — `_cache_reuse` gates an
untainted entry on `max_depth <= budget`, so an understated depth hands out a
strategy at a budget it cannot meet.  Stored above the fold only refuses reuse
that was available.  The pass runs bottom-up, so a branch corrected in this run
is what its parents are folded against; a fold that re-read stored children
would agree with every parent that folded the same understated child, and its
count is a floor rather than a measurement.

`--repair` writes each folded depth back, and only that column.  A `best_score`
that disagrees with its own fold is counted but never rewritten — a wrong ERD
may mean `best_guess` is no longer the argmin, which only a re-search
(`verify_erd_cache.py`) settles.

The two directions are not repaired alike.  Raising a depth only withdraws
reuse, so it is always applied.  Lowering one widens the budget range the row
is offered at, which is a claim about a strategy — so it is applied only when
the row's `best_score` agrees with its own fold, and withheld otherwise rather
than extending the reach of a score the same pass just contradicted.  The
report counts what it withheld.

A repair also drops the policy's `candidate_erd_by_policy` folds.  Each memoises
a fold over the rows being repaired and is trusted on a matching
response-group count alone — which a depth repair does not change — so a report
would otherwise keep serving the pre-repair depth.  There is no reverse index
from a branch to the folds that read it, so the whole policy goes; each is
re-earned by one fold.

**A repair does not travel between caches.**  It moves `updated_at`, so an
incremental `export_cache.py --since` carries the row, but `import_cache.py`
keeps the target's row for any collision that is not tainted→untainted — so
the repaired value does not land.  Repair each cache on its own machine; the
fold is deterministic, so both arrive at the same answer.

An audit-only run opens the cache **read-only** (SQLite `mode=ro`): it writes
no schema migration, no answer-list row, and no response decomposition, so it
is safe against a live cache while workers are active.  A cache path that does
not exist is an error, not an empty clean audit.  An audit-only run exits 1
when it finds a row stored below its fold.  Stop the swarm before `--repair`.

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
