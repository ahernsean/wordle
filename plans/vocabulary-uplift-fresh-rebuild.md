# Vocabulary uplift: fresh cache rebuild, SALET-first

## Status and scope

This plan supersedes the retention/recertification plan (PR #151, revised on
PR #160). The retired analysis — the growth-only theorems, the namespace
disposition table, the budget-versioned result model — remains readable in
git history under this file's former name, `plans/answer-list-cache-retention.md`.

Phase 0 is complete and merged (PR #158): `all_candidates.txt` (14,855 words)
and `all_answers.txt` (3,209 words) are live on `main`, with 0 removals on
either side, answers ⊆ candidates verified, and every consumer rewired. The
remaining work is entirely operational: rebuild the cache under the corrected
lists and get the result to the phone. This document is written to be
executed by hand, one numbered step at a time.

## Decision record: why a fresh rebuild

The retention plan was measured and reviewed into irrelevance:

- **Retention saved no solve work.** The recertification design already
  started the new-vocabulary `ERD_ALL` namespace empty and fresh-solved every
  branch; old scores were never consulted. Feeding old best-guess costs to
  the solver as ceilings was measured at zero net benefit (181 branches,
  totals within 0.2%) — the engine's own Σk² candidate ordering finds a
  near-optimal incumbent unaided. The old cache's only assets were its
  branch-key list and a leaves-first schedule.
- **The bookkeeping was the risk.** Four review rounds on PR #160 each found
  blocking correctness gaps — not in the interval algebra, which was
  eventually verified sound, but in the seams between that algebra and the
  engine, cache, queue, migration, import, and phone-sync surfaces. Every
  fix enlarged the seam surface that produced the next round's error.
- **A fresh rebuild deletes the error class instead of defending against
  it.** Every row is computed by the current engine under the current lists.
  No new schema, no migration, no old-ID/new-ID merge, no budget-versioned
  result model, no interval reuse. The correctness argument is one sentence.

The cost is bounded, visible compute: the same solves the retention plan
would have run anyway (measured 3.0–8.7 days on 6 workers for the full old
population; SALET-first scope is a fraction of that and reaches the
production milestone — the SALET tree on the phone — earliest).

## Dropped from the retired plan, and why

| Component | Why it is not built |
|---|---|
| `branch_best_by_policy_and_budget` table + exactness-interval algebra | Existed to preserve tainted finite-budget results as durable evidence; any attempt consuming such evidence is itself tainted, so the machinery could never help produce the canonical rows the run exists to produce |
| Old-ID → new-ID migration and merge, family validator, combined-state gates | No rows are carried across the vocabulary change, so there is nothing to merge or validate |
| Manifest extraction, certification-budget ladder, deferred-key ledger, wave barriers | The work list is now the queue seeded from live spines; every branch has a spine, so budgets derive exactly as the ordinary swarm always has |
| Compliant-row and `candidate_scores` retention | Both tables are `answer_list_id`-scoped, so carrying them forward needs precisely the re-tag machinery being deleted. They rebuild organically and cheaply. The archived database keeps them if a future measurement ever justifies a re-tag import |
| Old `ERD_ALL` / `erd_answers_unfiltered` losses | Invalid under vocabulary growth in every design |

## Carried forward from the retired plan

1. **The taint-join engine fix** — the one code change this rebuild needs
   (step 1 below).
2. **The standing machine-state check** (below, verbatim in substance).
3. **The disk-stop latch discipline**, with a newly confirmed gotcha
   (step 4).
4. **The two-mechanism phone sync and its hard boundary** (step 10).
5. **Provenance**: `Get_NYT_Candidates.py` / `Get_NYT_Answers.py` re-scrape
   the sources by shape, not filename, and remain the way to re-verify the
   lists.
6. **The vocabulary-inclusion sandwich** as an optional follow-up (see
   "Later", below), now genuinely sound because answers ⊆ candidates holds.
7. **The archived database as rollback and comparison data.**

## Standing machine-state check (run before any step that touches cache or queue)

Never trust a written "the swarm is stopped" claim, and never trust a single
signal. All three checks must agree:

1. `python3.13 erd_search.py view --workers` — reads queue heartbeats
   directly, but a missing, stale, or wrong-path queue can still show no
   live workers.
2. `systemctl --user is-active wordle-erd wordle-report-server` — neither
   may be active; the report server reads the production cache even when no
   worker heartbeat exists.
3. `ps -ef | grep -E '[e]rd_search.py run|[s]warm_worker|[r]eport_server.py'`
   — needs neither queue nor systemd state and does not match its own grep.

If anything is running: full `python3.13 erd_search.py stop` (also stops the
report server), then re-run all three checks.

## Procedure

### 1. Deploy the taint-join fix — landed in this PR

`evaluate_candidate` was discarding a sub-branch's `sub_budget_tainted` when
the sub-branch returned `OVER_ERD_LIMIT` (`wordle_engine.py`: the early
return in the sub-branch loop ran before the `floor = floor or
sub_budget_tainted` join). A floor-tainted child's ceiling refutation is
budget-contaminated — an unconstrained strategy below the floor could beat
the ceiling — so the cutoff must carry the child's taint. Without the fix, a
branch could be written `solve_budget IS NULL` (claimed unconstrained-exact,
reusable at any budget ≥ `max_depth`) when the floor in fact pruned its
search. That false certificate would poison a fresh rebuild exactly as it
would have poisoned recertification.

The fix joins the child's taint before every dispatch on the child's status,
with a deterministic test forcing a child to encounter the remaining-depth
floor and then return `OVER_ERD_LIMIT` under its parent's ceiling, verifying
the taint reaches the candidate result, the top-level branch, and the cache
write. The same gap existed on the swarm's ceilinged-cut sharing path
(`cut_results`); fixed with a `tainted` column threaded through publish and
both consumption sites, with an idempotent migration.

Deploy to rocky (a normal `git pull` of `main` after merge) before the
rebuild starts — stale workers are stopped in step 2 regardless.

### 2. Freeze

Run the standing machine-state check; full `erd_search.py stop` if anything
is live. Re-run the checks.

### 3. Archive the old cache

```
mv wordle_cache.sqlite3 wordle_cache_pre_uplift_2026-07-23.sqlite3
```

If `wordle_cache.sqlite3-wal` / `-shm` exist after the stop, the freeze was
not clean — investigate, do not archive a database with a live WAL. The
archive is the rollback path and the old-world comparison data; it is kept
until retirement (step 11). A fresh `wordle_cache.sqlite3` is created
automatically on first open, with the corrected lists registering their
`answer_list_id`s.

Optionally copy `erd_queue.sqlite3` aside too; it is transient coordination
state, but the copy is cheap.

### 4. Clear the queue — and know what that wipes

```
python3.13 erd_search.py queue clear --yes
```

**Gotcha (confirmed in code):** `queue clear` deletes the whole `run_meta`
table, which holds both the **disk-stop latch** and the **telemetry epoch
pointer**. After this command the latch engaged on 2026-07-22 is gone, so a
reboot would auto-start `wordle-erd`. Post-archive this is safe — the
service would find corrected lists, a fresh cache, and an empty queue — but
do this step only after step 3, never before.

### 5. Set a fresh telemetry epoch (optional, recommended)

Keeps rebuild telemetry windowable apart from epoch 8's baseline. There is
no CLI (issues #147/#159); run a one-off from the scratchpad:

```python
from erd_queue import ERDQueue
from runtime_paths import DEFAULT_QUEUE_PATH
q = ERDQueue(DEFAULT_QUEUE_PATH)
q.set_epoch(9, label="fresh-rebuild-salet", notes="vocabulary uplift rebuild")
```

### 6. Seed SALET only

```
python3.13 erd_search.py queue add --word salet
```

This enqueues every SALET root response branch with ≥2 answer words, each
carrying its (guess, pattern) spine, so budgets derive exactly as they
always have (`budget = GAME_GUESSES − guess_depth`). Nothing else about
worker behavior changes: missing children solve inline, tainted results
write `solve_budget`-marked rows under the existing exact-budget reuse rule,
and taint propagates through cache reuse as the engine already does.

Adding more openers later is the same command with a different `--word`
(or `--word-list` with `--priority-words`); shared sub-branches are already
warm cache hits.

### 7. Start and monitor

```
python3.13 erd_search.py start
python3.13 erd_search.py view --watch
```

`view --workers` for worker health, `view salet` for the cone, `view
--queue` for queue totals. A reboot mid-run simply resumes the queue.

### 8. Drain, then decide

When the SALET cone drains (no pending or active branches), the SALET tree
is complete in the new cache. Decide: add more openers (step 6 again), or
proceed to the phone. Partial coverage is fine for the phone — anything
uncached solves on device, slower but correct.

### 9. Light verification

- Full suite: `python3.13 -m unittest discover -s tests -t . -p 'test_*.py'`.
- Simulated games on Linux against each of the 9 new answer words and a few
  of the 14 rescued candidate words (`amaro crema flowy glamp glowy hacky
  janky koran popup queso quran runup untag venti`).
- Optional spot-check: fresh-solve a handful of completed branches into a
  scratch cache and compare scores. There is no old/new reconciliation to
  run — provenance is uniform by construction.

### 10. Phone catch-up — the hard boundary

**The phone must not `git pull` until this step.** Pulling `main` *is* the
flip: PR #158's corrected lists land with the code in one pull. Do the
whole step in one sitting:

1. Phone (Working Copy) pulls `main` — code and lists flip together.
2. Run the established export/import dance (`export_cache.py` on rocky →
   transfer → import on the phone) so the phone's cache matches rocky's
   rebuilt `wordle_cache.sqlite3`.
3. Smoke-test a game on the phone, including one of the 9 new answers.

Ordering safety net, not a substitute for discipline: `answer_list_id` is
recomputed from whichever lists are loaded, so a mismatched pairing
produces cache misses, never wrong answers. There is no schema change in
this uplift, so there is no code-before-cache migration ordering to manage.

Before this boundary the phone stays on the old world entirely. After it,
the phone is on the new world; repeat the export/import dance whenever
rocky's tree has grown enough to be worth shipping.

### 11. Soak and retirement

After enough post-boundary use to trust the new world:

1. Delete `wordle_cache_pre_uplift_2026-07-23.sqlite3`.
2. Resume ordinary epoch testing when desired: the queue is already the
   default one; seed whatever the next experiment needs.

## Rollback

Before the phone boundary: stop the swarm, restore the archived cache file,
and check out the pre-#158 list files from git history — the old world is
those two artifacts and nothing else. After the boundary, roll forward
instead: the rebuild is a fresh file, so a bad state is fixed by deleting it
and reseeding, not by surgery.

## Later (explicitly out of scope now)

- **More openers / the full 3,209-root census** — same seeding command,
  whenever wanted.
- **The vocabulary-inclusion sandwich** for `erd_answers_unfiltered`:
  with answers ⊆ candidates now true, `ERD_ALL′(B) ≤
  ERD_ANSWERS_UNFILTERED′(B) ≤ ERD_ANSWERS(B)`, and where the outer values
  are equal the compliant row can seed the unfiltered row. Both outer rows
  must be canonical untainted (`solve_budget IS NULL`) rows from the fresh
  cache. Optional throughput icing once both namespaces are populated.
- **Re-tag import of compliant rows / `candidate_scores` from the archive**
  — only if measurement ever shows organic rebuild of those tables is a
  real cost. Requires an old-ID → new-ID re-tag pass; do not build it
  speculatively.
