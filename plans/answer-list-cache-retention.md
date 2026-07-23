# Vocabulary correction: cache retention, budget-versioned recertification, and reverification plan

## Context

Two staleness discoveries, in escalating order:

1. **Answer list.** The 2026-07-15 Wordle answer was PSHAW — absent from
   `NYT_wordlist.txt`. The corrected list is captured as
   `NYT_wordlist_2026-07-17.txt`: **9 additions, 0 deletions**
   (3,200 → 3,209 words):

       aught  genus  krone  prate  pshaw  ravel  shlep  stagy  techy

2. **Candidate universe.** Wordle Tools reports the NYT full dictionary at
   **14,855 words**; our `wordle.txt` has 12,972 — a stale snapshot from the
   pre-NYT era. Direct local evidence of the staleness: 14 answer words
   (`amaro crema flowy glamp glowy hacky janky koran popup queso quran
   runup untag venti`) appear in the answer list but not in `wordle.txt`,
   which is impossible in real Wordle (every answer is an accepted guess).

   **Sourced and confirmed.** The current 14,855-word dictionary is shipped
   client-side in the NYT Wordle web app's own webpack bundle — needed for
   instant offline guess validation. (The answer list is not shipped there;
   revealing it would spoil the game, which is why it stays sourced from
   Wordle Tools' third-party reconstruction instead.) `Get_NYT_Candidates.py`
   extracts it by shape — a run of ≥5,000 quoted 5-letter words inside a
   downloaded chunk — rather than a hardcoded chunk filename, since NYT's
   chunk hashes and numbering change on every deploy. The delta against
   `wordle.txt` is **1,883 additions, 0 removals**; the corrected answer
   list is confirmed a subset of the corrected candidate list, including
   all 14 previously-missing words and all 9 PSHAW-era additions. Zero
   removals resolves Part 2's table to the growth-only row throughout:
   `ERD_ALL` rows are valid upper bounds/incumbents, not mere heuristics.

The candidate-universe change dominates the planning: `ERD_ALL`'s guess
vocabulary is `wordle.txt`, so the namespace that the answer-list-only
analysis retained as exact is precisely the one a candidate-universe change
demotes. Both corrections must land as **one combined uplift**, candidate
universe first in the analysis: recertifying `ERD_ALL` against a vocabulary
known to be wrong, then invalidating it again, would do the work twice.

Operating decisions already made:

- The phone stays on the old lists — no pull from origin — until the uplift
  is entirely done. The Linux side flips wholesale when execution starts.
- The ordinary epoch swarm must be stopped before any uplift phase reads or
  writes the cache, and epoch testing resumes only after retirement. Phase 3's
  dedicated recertification supervisor and Phase 4's controlled root/cone run
  are the only planned worker pools; each has exclusive cache ownership while
  it runs. Machine state is never assumed: the ordinary swarm may have been
  run opportunistically between work sessions, and a reboot restarts it via
  its `wordle-erd` systemd unit, so every phase opens with the check below.

### Standing machine-state check (run at the start of every phase)

Never trust a written "the swarm is stopped" claim, and never trust a single
signal. All three checks must agree:

1. `erd_search.py view --workers` against the queue path the supervisor is
   expected to use. This reads queue heartbeats directly, but a missing,
   stale, or wrong-path queue can still show no live workers.
2. Query both systemd user services independently:
   `systemctl --user is-active wordle-erd wordle-report-server`. Neither may
   be active; the report server reads the production cache even when no worker
   heartbeat exists.
3. Cross-check for manually started or orphaned processes with a
   self-excluding match:
   `ps -ef | grep -E '[e]rd_search.py run|[s]warm_worker|[r]eport_server.py'`.
   This check needs neither queue nor systemd state and does not match its own
   `grep` command.

If any check shows a service or process running, stop it — the operator CLI
wraps `systemctl --user`:

- **Before any active uplift phase (anything touching the cache or queue): full
  `erd_search.py stop`.** This also stops the report web server, which reads
  the production database; keep it down so nothing reads the DB mid-migration
  or mid-recertification. The CLI `view` remains available as a direct,
  read-only queue/cache inspector when explicit paths are supplied.
- `erd_search.py stop --swarm-only` is only for a transient look where you
  want `view` to stay up and accept the report server reading the DB. Do not
  use it as the posture for a phase that mutates the cache.

Re-run all three checks after stopping, then proceed.

**Reboot-proofing: the disk-stop latch.** A full stop does not survive a
reboot — the `wordle-erd` unit is `enabled` with `Restart=on-failure`, so a
reboot restarts the swarm. For any hold that must survive a reboot (most of
this uplift, and especially the multi-day Part 3 run), engage the disk-stop
latch in the **default** queue: a row durable across reboots and systemd
restarts. On startup the ordinary service reads it and refuses with a clean
exit (no `Restart` loop), until it is released with `erd_search.py queue
clear-disk-stop`. The dedicated recertification supervisor uses its explicit
alternate queue path and its own disk guard; the default latch remains set
throughout Phase 3. There is currently **no CLI to set it** (only to clear —
it is normally written by the disk-fill guard), so set it with a one-off:
`ERDQueue(DEFAULT_QUEUE_PATH).set_disk_stop('<reason>')`. Adding a
`queue set-disk-stop` command is part of Phase 3's implementation because the
procedure engages the latch more than once. Releasing the latch is required
before each controlled ordinary-swarm run (Phase 4 and, finally, Phase 7).

This is a correctness requirement, not hygiene: concurrent ordinary and
recertification supervisors could evaluate current-ID branches under
different budgets and queue ownership, while a worker on stale code can write
under the wrong `answer_list_id`. One explicitly selected supervisor owns the
cache at a time.

## Part 1 — Foundations (unchanged by the second discovery)

### Content-addressed keys freeze each row's answer set

`ScoreCache.encode_subset` keys every row by the sorted concatenation of the
branch's actual words. Growing the answer list changes *which keys arise*
in the game tree — never the meaning of an existing key. `_solve_subset`
partitions `branch_words` itself to form sub-branches and never consults
the global list to enumerate answers. The only input that can vary for a
fixed key is the guess vocabulary its policy draws from.

### Theorem: guess-vocabulary growth can only lower ERD (branch fixed)

Fix a branch word set B and guess vocabulary V. A strategy for (B, V) is a
decision tree with every node labeled by a guess g ∈ V; an answer a ∈ B
descends along the `calculate_response(g, a)` edge at each node and
terminates at the first node labeled a; cost(a) is the path length. ERD is
the min over strategies of the mean of cost(a) over a ∈ B; call the min
over strategies of the max the *strategy-minimal worst case* — deliberately
not named `max_remaining_depth`, because the stored `max_remaining_depth`
is a different quantity (see the note below). (This models the code:
`guesses` is threaded unchanged
into every recursive `_solve_subset` call, so unfiltered policies draw from
the same fixed V at every node.)

**Claim.** V ⊆ V′ implies ERD(B, V′) ≤ ERD(B, V), and any budget feasible
for (B, V) is feasible for (B, V′).

**Proof.** Any strategy T for (B, V) has every node label in V ⊆ V′, so T
is verbatim a strategy for (B, V′). Its cost profile is identical in both
problems: `calculate_response(g, a)` depends only on the pair (g, a), and
the tree's labels and the answer set B are unchanged, so every a ∈ B traces
the same path. Hence the strategy set for (B, V′) contains that for (B, V)
with costs preserved, and a minimum over a superset is at most the minimum
over the subset. Apply with the mean (ERD) and existence within a budget
(feasibility). ∎

**What this does not give: `max_depth` monotonicity.** The strategy-minimal
worst case defined above is monotone
nonincreasing under the same restriction argument (T's worst case is
preserved verbatim in V′, so the minimum over a superset is
again at most the minimum over the subset). But that abstract quantity is
not what the cache stores. `_solve_subset` only updates
`best_max_remaining_depth` when a candidate strictly beats the running
`best_erd` (`wordle_engine.py`: `if cost < best_erd: ... best_max_remaining_depth
= max_remaining_depth`) — the stored `max_depth` is the worst case *of
whichever tree won the ERD race*, not a separately minimized worst case.
Growing V can hand the race to a new, lower-ERD tree with a *larger* worst
case than the old winner's, so the stored `max_depth` is not provably
monotone under vocabulary growth in either direction. (Mechanism 2 already
treats cached `max_depth` this way — an upper bound on the true worst-case
optimum, not an exact or monotone value; this note aligns the Theorem with
that rather than contradicting it.)

**Corollaries.**
- *Growth:* old exact values become upper bounds; old loss proofs become
  invalid (a new guess word may rescue the branch). A loss over the larger
  vocabulary implies a loss over any sub-vocabulary.
- *Removal:* the inequalities reverse — old exact values become lower
  bounds, and a stored best_guess may no longer be a legal word.
- *Mixed additions and removals:* no per-row direction is provable; old
  values are heuristics only.

### ERD is not monotone in the answer set (why the root must recompute)

ERD is a mean, not a sum. Adding a word that resolves in 2 guesses to a
branch of n words with ERD e yields (n·e + 2)/(n + 1) < e whenever e > 2;
adding a hard word raises the mean. Monotone nondecreasing under set
growth: the cost sum, and the mean restricted to the original words (both
provable by the same restriction argument as the Theorem above: any tree
solving the grown branch restricts to a valid, if suboptimal, tree for the
original words). `max_remaining_depth` is deliberately absent from that
list: it has the identical flaw as the Theorem's retracted claim, just
along the answer-set axis instead of the vocabulary axis — the stored
value tracks whichever tree wins the *ERD* race for the grown branch, an
independently-computed tree with no proven relationship to the original
branch's stored worst case. Per-candidate root ERDs can move both ways and
the best first guess can change — root results recompute regardless (the
full-set `branch_key` changes anyway).

### The answer-side recompute region is a union of 9 thin cones

For any first guess, the 9 new answer words land in at most 9 of its
nonempty response groups; every other group is a byte-identical key.
Within an affected group, the next guess isolates each new word into one
sub-group again, so the changed region under each new word is a single
narrowing cone. (This bounds which *keys* are new; whether the retained
keys' *values* still serve depends on the vocabulary analysis below.)

## Part 2 — Disposition by namespace under the combined change

| Policy | Guess vocabulary | Effect of combined change | Disposition |
|---|---|---|---|
| `erd_answers_compliant` (`ERD_ANSWERS`) | the branch words themselves (`guesses=None` ⇒ `candidate_list = branch_words`, recursively) | pure function of `branch_key`; neither list enters: **exact** | retain — the only fully exact ERD namespace |
| `erd_words_unfiltered` (`ERD_ALL`) | `all_candidates.txt` (formerly `wordle.txt`) — grew by 1,883, 0 removals (confirmed) | growth-only case applies: rows demote to **upper bounds** (legal best_guess, valid incumbent); losses **invalid** regardless | leave old rows under the old `answer_list_id` as source inventory and rollback data; validate answer-only keys into the recertification manifest; populate the new namespace through the recertification swarm; copy no losses |
| `erd_answers_unfiltered` (`ERD_ANSWERS_UNFILTERED`) | answer list — grew by 9 | rows demote to upper bounds; losses invalid | rebuild via sandwich seeding (Part 4) after `ERD_ALL` recertification |
| `erd_words_compliant` (`ERD_CONSTRAINED`) | path-dependent | never persisted | n/a |

Other tables:

- **`candidate_scores`** — a score is a function of (branch word set,
  candidate, method) only; retain all rows. Rows for removed candidates
  become dead weight (harmless); scores for added candidates and grown
  branches are new keys computed on demand.
- **`branch_loss_by_policy`** — retain `erd_answers_compliant` losses
  (exact); drop `ERD_ALL` and `erd_answers_unfiltered` losses (invalid
  under vocabulary growth). Losses rebuild organically as the swarm
  re-proves deep branches.
- **`response_decomposition`** — keyed (guess, answer_list_id); rebuilds
  for the new id. Blobs are positional (byte *i* = answer *i* in sorted
  order), and the 9 new answer words are interleaved rather than appended,
  so a splice is not the cheap shortcut it sounds like — see Mechanism 4.
  Rebuild.

## Part 3 — The `ERD_ALL` recertification swarm (the centerpiece)

### The old namespace is manifest source inventory, not a seed cache

The corrected answer list has a different content-addressed
`answer_list_id`. `ScoreCache` reads and writes only rows carrying the ID of
the answer list with which it was opened. Do **not** copy old `ERD_ALL` rows
to the new ID in Phase 2. Leave them under the old ID and derive the immutable
recertification manifest from their `branch_key` values.

The old-ID row set is source inventory, not automatically the manifest.
Interactive answer-list fallback can persist `ERD_ALL` branches containing
candidate words that are not answers under the same `answer_list_id`. Such a
key is valid cache data for that interaction, but it is not an answer branch
and the swarm's matrix packer cannot map it through
`PatternMatrix.answer_indices`.

Manifest extraction therefore validates every old-ID `ERD_ALL` key before
admission:

1. Decode fixed-width five-byte words, then require valid UTF-8, a nonempty
   key whose length is a multiple of five, lowercase five-letter words,
   strict sorted uniqueness, and byte-for-byte equality after canonical
   `ScoreCache.encode_subset` re-encoding. A malformed key is cache
   corruption and aborts extraction before a queue is created.
2. Require every decoded word to be in the corrected current answer list.
   A well-formed key containing any other candidate word is an expected
   interactive fallback branch: exclude it from the manifest without deleting
   or rewriting its old-ID row.
3. Form the source population as the distinct union of canonical and
   budget-versioned old-ID best-row keys. Record both tables' raw row counts,
   the distinct source-key count, eligible-manifest and excluded-fallback
   counts, deterministic digests and per-answer-count histograms for the
   eligible and excluded sets, and a bounded sample of excluded keys in the
   migration ledger. Require `source keys = eligible + excluded`.

There were no answer removals, so every ordinary old answer branch passes the
membership test. The manifest intentionally does not contain branch keys
introduced by the nine added answers; Phase 4's root census and answer cones
create those keys after the fixed eligible old-key population has been
recertified. Excluded fallback branches recompute on demand if the same
interactive fallback recurs under the corrected lists.

This namespace shortcut is specific to the combined uplift: the answer-list
change gives the corrected candidate vocabulary a fresh cache namespace for
free. A future candidate-only change with an unchanged answer list would not
get a new `answer_list_id`; it would require an explicit candidate-vocabulary
identity in the cache schema or another fresh namespace before ordinary reuse
could be sound.

This removes the ambiguity that forced the standalone design in an earlier
draft. Under the new ID:

- an absent row is work still to do;
- a canonical row was solved against the corrected answer list and candidate
  vocabulary without a remaining-depth floor;
- a budget-versioned row was solved exactly at one finite remaining-depth
  budget;
- ordinary `_cache_reuse` is sound; and
- a worker restarting after writing the cache row but before completing the
  queue row can safely recover the exact attempt result from its budget-keyed
  slot.

The old row's score is not copied or consulted by the solver. Direct
measurement found no net benefit from evaluating the old best guess merely
to seed a ceiling; the old cache's useful assets are the branch-key
population and its ascending-size schedule. Reporting may compare a completed
new score with the old score after finalization, but the old value never
affects search. Keeping the result namespaces disjoint gives the scheduling
benefits without ever making an upper bound look exact.

### Each recertification target is an ordinary swarm branch

A manifest `branch_key` decodes to the same ordinary branch word set the
engine and swarm already solve. Recertification describes *why that branch
was admitted*, not a different recurrence or result type. Once admitted
under the new ID, the branch uses the normal swarm contract:

1. workers claim disjoint candidates from the complete corrected candidate
   vocabulary;
2. `evaluate_candidate` recursively prices every response group;
3. missing sub-branches solve inline and write legitimate current-ID rows;
4. the running best is shared through `active_branches`; and
5. full candidate coverage finalizes either an unconstrained exact row or a
   budget-specific row that the controller schedules for a higher-budget
   attempt.

### Finite-budget certification ladder

The ordinary game-tree swarm derives a remaining-depth budget from a branch's
spine. Manifest branches have no unique historical spine, so the
recertification controller assigns a **certification budget** instead. Start
at 5, then retry floor-tainted branches at successively larger finite budgets.
The scratch pilot chooses and freezes the remaining ladder before production;
`5, 6, 8, 12` is the initial schedule to measure, not a correctness constant.

The implementation starts from an explicit result model. These states are not
interchangeable:

| State | Meaning | Durable representation | Reuse |
|---|---|---|---|
| **unconstrained exact** | Complete candidate coverage proved the global ERD optimum without any remaining-depth floor | canonical `branch_best_by_policy` row with `solve_budget IS NULL` | unlimited queries, and any finite budget at least its `max_remaining_depth` |
| **budget-specific exact at b** | Complete candidate coverage proved the optimum among strategies feasible at budget b, but some explored path encountered the remaining-depth floor | `branch_best_by_policy_and_budget` row keyed by `solve_budget=b` | exactly budget b in the baseline contract |
| **loss at b** | Complete candidate coverage proved that no strategy is feasible at budget b | `branch_loss_by_policy`, retaining the greatest `loss_budget` | every query budget at most b |
| **ceiling cutoff at (b, c)** | An admissible lower bound proved only that the current candidate or branch cannot beat ERD ceiling c during the attempt at b | transient `OVER_ERD_LIMIT`; never a cache row or terminal class | none |
| **abort** | Deadline, cancellation, worker death, or another interrupted attempt | retrying queue state only | none |

For a fixed branch and policy, let `E_b` be the budget-specific exact ERD at
budget b and `E_*` the unconstrained exact ERD. Increasing b expands the set
of feasible strategies, so `E_b` is nonincreasing until it reaches `E_*`.
That monotonicity does **not** make two finite-budget rows substitutes for one
another: a strategy winning at b may be infeasible below b, while a result at
b need not have considered strategies admitted only above b. The baseline
therefore reuses `E_b` only at b. A loss has the opposite safe range: loss at
b implies loss at every smaller budget.

A finite-budget attempt certifies `E_*` only when all of these proof
obligations hold:

1. every candidate is either evaluated exactly or rejected by an admissible
   candidate/ceiling lower bound;
2. the attempt completes rather than aborting;
3. remaining-depth-floor taint is the logical OR of every completed
   descendant's taint, including nonwinning candidates and descendants
   returning `OVER_ERD_LIMIT`; and
4. that accumulated taint is false.

The third condition is deliberately status-independent. A child can encounter
the remaining-depth floor and later prove that its candidate exceeds the
current ERD ceiling. The cutoff is enough to reject that candidate at budget
b, but discarding the child's taint would falsely claim that no hidden
unconstrained strategy was pruned. `evaluate_candidate`, the inline engine,
and the swarm evaluator must join `sub_budget_tainted` before dispatching on
every normal child status. Early-return paths preserve the joined taint;
abort paths discard the entire attempt rather than manufacturing a result.
Each candidate-claim result carries that taint even when the candidate was
cut off, and branch finalization ORs taint across every completed claim rather
than inspecting only the winning candidate.

When those obligations hold with false taint, the attempt writes an
unconstrained exact row, including the winning strategy's
`max_remaining_depth`. With true taint it writes `E_b` to the budget-keyed
table. If every candidate is infeasible, it writes a loss at b. A candidate
cutoff may complete that candidate's coverage while carrying taint into the
branch; a branch-level cutoff or abort writes no result and cannot advance the
manifest ledger.

### Version finite-budget results instead of ordering incomparable writes

`branch_best_by_policy` has one row for `(branch_key, policy,
answer_list_id)`. It is the canonical slot for unconstrained exact rows; it
must not also act as a lossy register for many `E_b` values. Add:

    branch_best_by_policy_and_budget (
        branch_key       BLOB    NOT NULL,
        policy           TEXT    NOT NULL,
        answer_list_id   TEXT    NOT NULL,
        solve_budget     INTEGER NOT NULL,
        best_guess       TEXT    NOT NULL,
        best_score       REAL    NOT NULL,
        updated_at       INTEGER NOT NULL,
        max_remaining_depth INTEGER NOT NULL,
        PRIMARY KEY (
            branch_key, policy, answer_list_id, solve_budget
        )
    )

The schema and cache contracts are:

- `solve_budget IS NULL` writes go only to
  `branch_best_by_policy`. Integer-budget writes go only to
  `branch_best_by_policy_and_budget`; finite budgets never compete with one
  another or overwrite the canonical certificate.
- Unlimited reads consult only the canonical table. A finite-budget read
  first uses a canonical row if its `max_remaining_depth` fits, otherwise it
  looks up the exact requested budget in the versioned table. Legacy canonical
  rows with NULL `max_depth` remain unavailable to finite-budget reuse.
- SQLite and the process-local cache use the same composite identity,
  including `solve_budget` for finite results. A read or write at one finite
  budget cannot hide a row at another budget.
- Repeating a write to the same finite-budget key is deterministic. The exact
  score must agree within the solver's comparison tolerance; the winning
  guess and its `max_remaining_depth` may differ only under an exactly tied
  optimum.
  A disagreement is a correctness failure, not a last-writer-wins event.
- `branch_loss_by_policy` remains the monotone loss representation:
  `write_loss` keeps the greatest proven `loss_budget`.

The stored rows must also satisfy cross-state invariants, checked on writes,
startup, and Phase 5:

- every budget-versioned winning strategy has
  `max_remaining_depth <= solve_budget`;
- for two stored exact budgets `b1 < b2`, `E_b2 <= E_b1` within comparison
  tolerance;
- a loss through b cannot coexist with an exact row at any budget at most b;
  it also cannot coexist with a canonical row whose `max_remaining_depth`
  fits b; and
- a finite loss below a canonical strategy's `max_remaining_depth`, or below
  an exact row at a larger budget, is not inherently contradictory.

Conflicting exact and loss evidence is a correctness failure. It is never
resolved by last-write-wins deletion.

Implement this as an idempotent, `schema_migrations`-guarded
`ScoreCache._ensure_schema` migration. It creates the versioned table and
indexes, validates that every existing `branch_best_by_policy` row with
`solve_budget IS NOT NULL` has a positive budget and non-NULL
`max_remaining_depth`, copies it to the matching budget key, verifies source
and destination counts and values, then deletes only those migrated finite
rows from the canonical table. Invalid legacy finite rows stop migration with
their identities; they are not silently coerced. A rerun is a no-op with the
same verified state.
The migration applies to every answer-list ID and ERD policy, not merely this
recertification run. `MemoryScoreCache`, import/export, reporting, audit
queries, and phone consumers receive the same two-store contract.

Each retry is a new attempt generation. Candidate claims, the shared
incumbent, and completion counts are scoped by `(branch_key,
certification_budget, attempt_generation)` so work or bounds from a smaller
budget cannot finalize a larger-budget attempt. Existing unconstrained exact
children remain reusable from the canonical table; budget-specific children
follow the exact-budget reuse rule.

This retains the depth floor as a practical guard while still producing
canonical unconstrained rows. It also has a finite correctness endpoint:
every informative guess strictly shrinks its branch, and answer words are
legal candidates, so a branch of `n` answers always has a strategy of at most
`n` remaining guesses. Production does not jump to that theoretical cap.
Keys still tainted at the pilot-selected maximum become durably **deferred**,
with their last budget and reason recorded. Their final-budget `E_b` cannot be
overwritten by a later solve at another budget because the budget is part of
the row identity. They are safe to revisit through a higher finite ladder. No
worker automatically falls through to `budget=None`. Only a completed
final-budget attempt backed by the exact matching budget row or a loss proof
covering that budget may become deferred. A ceiling cutoff, deadline,
cancellation, or worker death leaves the key retrying; absence of a result is
not a terminal classification.

ERD pruning still applies recursively at every level. Each descended branch
orders its own candidates, establishes its own incumbent, applies its own
candidate lower bounds and accumulated-cost cutoffs, and may inherit a tighter
ceiling from its parent. A 10-, 20-, or 50-guess tail can survive only if the
candidate remains ERD-competitive at every descended branch. That makes such
tails unlikely, but not impossible: the winning strategy's surviving path
must still be evaluated exactly, and the first incumbent-building candidate
may begin without a finite ERD ceiling. The certification ladder treats that
rare case as observable deferred work rather than assuming it away.

Recursive cooperative promotion stays disabled for the first implementation.
The top branch is already split across workers by candidate claims, while
missing children follow the ordinary inline-solve behavior under the current
certification budget. The pilot measures long inline claims; if they dominate,
recursive promotion is a measured follow-up rather than a prerequisite for
correctness.

### Dedicated queue and leaves-first admission

Set aside `erd_queue.sqlite3` unchanged and run the recertification swarm
against `erd_recertification_queue.sqlite3`. The dedicated queue uses the
ordinary `ERDQueue` branch, candidate-claim, heartbeat, recovery, WAL, and
disk-stop mechanisms. Its `run_meta` records at least:

- run kind (`erd_all_recertification`);
- old and new `answer_list_id`;
- corrected candidate-list digest and count;
- old-ID canonical and budget-versioned raw row counts; distinct source-key,
  eligible-manifest, and excluded-fallback counts, digests, and
  per-answer-count histograms;
- frozen certification-budget ladder and deferred-key policy;
- current branch-size wave, certification budget, and attempt generation; and
- the implementation Git revision.

Startup refuses to resume if any recorded identity differs from the current
files or cache. This prevents a partially completed queue from silently
mixing vocabularies or implementations.

An admission controller extracts and freezes the eligible old-ID manifest,
reads its keys in ascending branch size, and adds only the current size wave
to the dedicated queue in bounded batches. It never offers excluded fallback
keys to a worker. Within a wave it runs the frozen certification ladder,
retrying only keys whose preceding attempt was floor-tainted or a
budget-specific loss. It admits size `n + 1` only after:

1. every size-`n` pending/active attempt is terminal;
2. every eligible manifest key in that wave is either certified by a
   current-ID unconstrained exact row or durably deferred after a completed
   final-budget attempt, with the deferred entry pointing to the exact
   budget-versioned row or a loss proof covering that budget; and
3. certified, deferred, retrying, and total counts reconcile with the
   manifest histogram.

Every informative response group is strictly smaller than its parent, so
the barrier maximizes exact-child reuse before a parent starts. Certified
children are ordinary cache hits. A deferred or newly created child is solved
recursively under the parent's current certification budget; retaining a
budget-specific child never makes it unconstrained exact. Bounded admission
avoids duplicating all 3.6 million branch blobs in the queue at once; the
current-ID cache plus the queue's deferred ledger are the durable completion
record. A manifest key has exactly one current terminal class: certified or
deferred. If later recursive work creates an unconstrained exact row for a
deferred key, reconciliation promotes it to certified and marks the deferred
ledger entry superseded. Keeping the historical budget row is harmless;
counting both current classes is not.

### Monitoring

`erd_search.py view` accepts the dedicated queue and production cache paths,
so recertification workers, current candidates, branch coverage, queue WAL,
disk state, and recent finalizations use the normal report pipeline. Add a
recertification overview, selected from `run_meta.run_kind`, with:

- current wave, certification budget, attempt generation, and wave completion;
- manifest processed, certified, retrying, and deferred counts;
- floor-tainted and budget-specific-loss counts by certification budget;
- eligible-manifest and excluded-fallback counts and identities;
- `max_remaining_depth` distribution for certified rows;
- budget-versioned row counts and cache-hit counts by budget;
- current and rolling rows/hour;
- ETA by both row count and measured work;
- unchanged/improved result counts;
- active branch references and answer counts; and
- source-health and identity fields for both vocabularies.

Spine/tree layout is unavailable for manifest roots because a
content-addressed branch can be reached through many historical spines and
the cache does not store a canonical one. That is a presentation limitation,
not a different branch-solving contract: each active item and every
candidate claim remains an ordinary swarm branch.

### Implementation acceptance

Before the production pilot:

1. A schema fixture starts with the legacy mixed table containing finite rows
   at several budgets, runs `_ensure_schema` twice, and proves each finite row
   moved exactly once from source column `max_depth` to
   `branch_best_by_policy_and_budget.max_remaining_depth` while unconstrained
   and non-ERD rows retained their values. Source/destination reconciliation,
   indexes, and the `schema_migrations` guard are verified.
2. Cache tests write one unconstrained row and several finite-budget rows for
   the same branch and policy in every order. SQLite and `MemoryScoreCache`
   retain all versions. Unlimited lookup sees only the unconstrained row;
   finite lookup prefers a fitting unconstrained row and otherwise sees only
   the exact requested budget.
3. A finite-budget complete candidate scan with no remaining-depth floor
   writes the canonical `solve_budget=NULL` row, records the winning
   strategy's `max_remaining_depth`, matches an unconstrained reference solve,
   and reuses at every budget that fits it.
4. A deterministic nested solve forces a child to encounter the
   remaining-depth floor and then return `OVER_ERD_LIMIT` under its parent's
   ceiling. The child's taint reaches the candidate, top branch, cache write,
   and queue result: no level may write or count an unconstrained certificate.
   Companion tests cover each normal early-return status and prove taint is a
   monotone OR across completed descendant work.
5. A floor-tainted exact result writes only its matching budget version; a
   finite-budget loss widens only `loss_budget`. Each schedules a clean
   higher-budget attempt generation and cannot increment the certified count.
   A cutoff or abort writes no result and remains retrying.
6. A final-budget tainted row backs a deferred entry, then a later parent
   solves the child at both a smaller and a larger budget. Every finite row
   remains addressable and the deferred backing still resolves to the exact
   final budget. If later work produces an unconstrained row, reconciliation
   promotes the key to certified and marks the deferred entry superseded.
7. A manifest fixture contains an ordinary answer branch, a well-formed
   interactive fallback branch containing a guess-only word, and a malformed
   key. The ordinary key is admitted, the fallback key is excluded and
   accounted without reaching `PatternMatrix.answer_indices`, and the
   malformed key aborts extraction. With only well-formed rows, source count
   equals eligible plus excluded and both digests reproduce across restart.
8. The Phase 2 migration fixture copies compliant canonical, budget-versioned,
   and loss rows plus candidate scores, but no `ERD_ALL` canonical,
   budget-versioned, or loss row. Same-key/same-budget collisions verify exact
   score agreement; loss collisions retain the greatest `loss_budget`.
9. Manifest admission never reads old scores into a search bound, never writes
   the old namespace, and cannot advance a wave with an unaccounted eligible
   key. A completed final-budget attempt becomes deferred only when its exact
   budget row or covering loss proof exists.
10. Restart tests cover a worker dying before its cache write, after its cache
    write but before queue completion, and during candidate evaluation; each
    resumes the correct attempt generation without accepting an old-ID value
    or losing a candidate claim.
11. Startup rejects mismatched old/new answer IDs, eligible or excluded
    manifest identity, candidate digest/count, queue run kind, or
    implementation revision with all conflicting values in the error.
12. Recertification reports expose source inventory, eligible/excluded
    identities, wave/overall progress, certification budgets, retries,
    deferred and superseded keys, certified `max_remaining_depth`, workers,
    rate, ETA, and partial-source failures in text, JSON, and watched JSON
    Lines. Text remains useful at 50–60 columns through the shared adaptive
    terminal layout.
13. Terminal reconciliation maps every eligible manifest key to exactly one
    current class: an unconstrained current-ID best row, or a nonsuperseded
    durable deferred entry backed by the exact final-budget row or covering
    loss proof. Only the first class enters the sandwich.
14. The full suite and the scratch correctness/throughput pilot pass before
    production cache or queue paths are accepted.

**Missing sub-branches are the common case, not an edge case — and that's
fine.** Direct measurement confirms why: a new candidate's response
partition of an existing branch is a *combination* of old answer words that
essentially never coincides with a partition any old candidate produced, so
most candidates against most branches hit an uncached sub-branch. In
practice this means most rows cost close to a full fresh solve. The swarm
does that work explicitly rather than treating an uncached group as a reason
to skip a candidate.

The scan covers every candidate, not just the added ones: an *old*
candidate's cost depends recursively on its response-groups' cached ERDs,
so when leaves-first recertification lowers a child, a previously-losing old
candidate can newly win even though no added word is involved. An
"added-candidates-only" shortcut misses that case (an earlier draft
claimed it at ~15% of full cost; withdrawn).

### Cost and throughput gate

`verify_erd_cache.py` ran a full sweep of the whole `ERD_ALL` cache once
before, for an unrelated reason (the reclaim-while-alive bug fix, completed
2026-07-12): 3,485,333 rows, full candidate scan, 13h48m wall time. That
number bounds only the tool's original *read-only, unchanged-vocabulary*
behavior and does not price this sweep's inline solves — superseded by
direct measurement below.

**Method.** Against a read-only production cache and a scratch database
(no production writes), for branches sampled from every `ERD_ALL` size
bucket: solve fresh under the new 14,855-word vocabulary (populating
children), then delete just the top row and re-solve with children warm —
isolating the steady-state per-row cost a leaves-first sweep would see.
Two passes: an initial 208-branch stratified sample across sizes 4–200,
then a 251-branch targeted resample of the 16–30-word bucket (the
production histogram's largest concentration — 1.15M of 3.6M rows) with
exact per-size row-count weighting, since that bucket's cost turned out to
be the dominant source of uncertainty.

Those unlimited-depth direct solves estimated **3.0–8.7 days on 6 independent
workers; 5.4 days was the central estimate** (780 serial hours, trimmed-mean
method). They remain useful as a stress reference, not as the production
ladder's ETA. The sample did not record `max_remaining_depth`, and its heavy
tail was censored: most branches solved in under a second, but near-exact-tie
branches took tens to low hundreds of seconds, and 4.2% of the 16–30-word
bucket did not finish within a 90s per-branch cap in the second pass. It
therefore does not justify starting every production branch with
`budget=None`.

Before the production run, compare the recertification profile with the
six-process direct engine on the same scratch-cache samples and vocabulary at
each proposed certification budget. For rows reported untainted, require
exact agreement with an unconstrained reference score, then evaluate each
stored best guess directly and require it to attain its stored score and
`max_remaining_depth`; equally optimal guesses need not match. For tainted
rows and losses, require the next attempt generation to use the next frozen
budget and re-evaluate complete candidate coverage.

Measure the certified, retrying, and deferred fractions by branch size;
`max_remaining_depth` among certified rows; time and nodes per attempt;
completed rows/hour; worker CPU utilization; queue coordination time; and
queue WAL growth. Measure budget-versioned table growth, per-budget hit rates,
and repeated work where no exact requested-budget row exists during later
waves and a scratch Phase 4 root/cone sample. Storage or cache-hit results may
change the operational ladder, but they cannot collapse incomparable budgets
back into one row. Include every production manifest key above 200 answers in
the pilot rather than extrapolating the sparse largest tail. Freeze the ladder
and its maximum only after this measurement.

At each budget the swarm profile must sustain at least 80% of the direct
pool's aggregate attempts/hour; otherwise tune its top-level branch
concentration so multiple ordinary branches can be active concurrently, and
repeat the gate. Do not weaken candidate coverage, exact-row qualification,
or durable accounting of deferred keys to meet the throughput gate.

**Correctness, corroborated empirically.** Across both passes, 432 branches
completed a fresh solve; 28 (6.5%) produced a strictly better ERD than the
old stored value (a real, solver-verified improvement — new candidates
winning branches they couldn't reach before), and **zero** produced a worse
one. Zero anomalies is exactly what the Part 1 growth-only theorem
predicts and would have been the signature of real cache corruption had
any existed. It didn't.

**The old cache's cost-side value is the row-key list and leaves-first
scheduling, not the stored scores.** A direct test — price the old
`best_guess` first and feed its cost to the solver as an alpha-beta
ceiling, instead of leaving the ceiling at ∞ — showed no net benefit (71
faster / 110 flat-or-slower across 181 branches, sums within 0.2% of each
other). The engine's own candidate ordering (`wordle_engine.py`: sorts by
Σk², response-group-size proxy, for any branch ≥8 words) already finds a
near-optimal incumbent on its own, and evaluating the old guess separately
just to seed a ceiling mostly duplicates work the ordinary scan would do
anyway. See the corresponding note on Mechanism 2 in Part 5 — this doesn't
touch the mechanism's *correctness*, only its practical payoff here.

## Part 4 — Rebuilding `erd_answers_unfiltered`

### The vocabulary-inclusion sandwich — gated, and only post-recertification

With the corrected lists, verify (do not assume): every answer word ∈ new
`all_candidates.txt`. Then for any branch B, B ⊆ answers′ ⊆ all-words′, and
the theorem gives:

    ERD_ALL′(B)  ≤  ERD_ANSWERS_UNFILTERED′(B)  ≤  ERD_ANSWERS(B)

**The left value must be a current-ID `ERD_ALL` row produced by the
recertification swarm.** Because Phase 2 copies no old `ERD_ALL` rows into
that namespace, every `solve_budget IS NULL` current-ID row is an
unconstrained exact result rather than a stale seed. Budget-specific and
deferred rows are excluded from the sandwich. A merely-demoted old-ID upper
bound on the left can equal `ERD_ANSWERS(B)` while the true `ERD_ALL′(B)` is
lower, in which case the pinch conclusion fails and the seed would overstate
the optimum while recorded as exact. The sandwich was also unsound against
the old cache for two independent reasons: the old `ERD_ALL` values were
computed over a vocabulary that did not contain the answer list (14 missing
words — the inclusion simply failed), and the vocabulary has now changed
besides. All of this is repaired only after Phase 0 fixes the lists and Phase
3 produces the certified exact subset.

Where the outer values are equal, seed the row by **copying the entire
`erd_answers_compliant` row** (best_guess, best_score, max_depth): its
strategy's guesses all lie within the branch ⊆ answers′, so it is a legal
`erd_answers_unfiltered` strategy attaining the pinned score — giving the
seed a usable best_guess and max_depth (a NULL-max_depth row is never
reused at budgeted queries). Guard the pinch on canonical
`solve_budget IS NULL` rows on both sides: budget-versioned values pin
nothing.

Do **not** seed losses from old `ERD_ALL` losses (invalid under growth).
Phase 3 may write finite-budget losses while climbing the certification
ladder, but they do not complete recertification and do not participate in
the sandwich. Once budgeted root/cone work proves an `ERD_ALL` loss under the
new vocabulary, it transfers validly at that budget (loss over the superset
vocabulary implies loss over answers′ ⊆ all-words′).

### Cost model

An unfiltered sweep is 3,209 candidates vs. 14,855 — ~4.6× cheaper per
branch than a new-universe `ERD_ALL` sweep. Only un-pinched rows need
sweeps:

    rebuild time ≈ (rows where ERD_ALL′(B) ≠ ERD_ANSWERS(B), untainted)
                   / (unfiltered sweeps per day on current engine)

Numerator: one SQL join after Phase 3. Denominator: time a
sample. If the un-pinched population is large, seed the pinched rows
(zero search) and leave the rest lazy.

## Part 5 — How retained work accelerates everything

G′ = G ∪ {added words} denotes a grown branch, G its retained base.

### Mechanism 1 — Recurrence-level reuse (automatic)

To evaluate a candidate c on G′, `_solve_subset` needs each response group's
ERD; the added words land in few groups, and every unchanged group is the
identical word set. During Phase 3 it is a current-ID cache hit once its
smaller-size manifest key has certified at a compatible budget. Deferred
children and new partition keys solve recursively under the parent's
certification budget and remain governed by the same taint rules.
Per-candidate cost = compatible cached lookups + recursive solves of the
remaining keys, memoized across candidates. A cache-missed key near the root
is a solve against a mostly warm current-ID cache. This is what keeps the
recertification swarm's inline solves of missing sub-branches tractable.

### Mechanism 2 — O(1) transfer bounds (scoped; measured not worth building)

**Empirical note (2026-07-19/20).** This mechanism transfers bounds along
the *answer-set-growth* axis (G → G′, new answer words landing in an
existing branch); the direct measurement in Part 3 tested the adjacent
*vocabulary-growth* axis instead (same branch, old best_guess used as a
solve ceiling under the grown candidate list) and found no net benefit —
the engine's built-in Σk² candidate ordering already reaches a near-optimal
incumbent without external help. That root cause (the ordering heuristic,
not anything specific to which axis grew) has no reason to stop applying
here, so the same lack of payoff is the expected outcome for Mechanism 2's
ceiling-style reuse too — but this is an inference from an adjacent result,
not a direct measurement of G→G′ specifically. The math below remains
correct regardless (an admissible bound can only prune, never mislead); the
open question is only whether implementing it is worth the code, and
current evidence says no. Retain the proofs — they're cheap insurance if a
future engine change (e.g. a weaker default ordering) makes them pay off —
but do not build the ceiling-injection machinery without re-measuring.

Valid **only for fixed-vocabulary policies** (`erd_words_unfiltered`,
`erd_answers_unfiltered`): in `erd_answers_compliant` the vocabulary *is*
the branch, so growing the branch also grows the vocabulary — the
restriction argument fails, and the lower-bound claims are genuinely false
there (G can be a compliant loss at budget b while G′ is compliant-solvable
at b). A policy-blind implementation would write false losses.

For fixed vocabulary, from G's cached row (E = best_score, md = max_depth,
losses), with k added words:

- **Loss transfer.** A proven loss for G within b is a proven loss for G′
  within b (restriction). Valid only when the loss itself is valid under
  the current vocabulary — post-uplift losses, not pre-uplift ones.
- **Feasibility (one-sided).** The cached md is the worst case *of the
  ERD-optimal strategy*, an upper bound on the true worst-case optimum. So
  budget ≥ md(G) + k ⇒ G′ feasible (extend the cached strategy: run each
  added word through the tree; where its response leaves the tree, extend
  with one guessing node; collisions chain, so the j-th word colliding on
  one path costs up to md + j, but md + k bounds the worst case overall).
  The infeasible side (budget < md ⇒ loss) is **unsound** — a strategy
  with a worse mean can have a smaller max; only a true loss row proves
  infeasibility.
- **ERD envelope.** (n·E + k)/(n + k) ≤ ERD(G′) ≤
  (n·E + k·md(G) + k(k+1)/2)/(n + k). The lower bound requires E ≤ true
  ERD(G) — i.e. an exact current-ID or lower-bound E; a
  demoted upper-bound E inflates the floor, making it inadmissible and
  able to prune the true optimum. Same discipline as the loss-transfer
  bullet. The upper bound needs no such gate (it extends an actual cached
  strategy), and its collision term is the honest worst case of the
  extension construction. (ERD(G′) ≥ E is NOT valid — see
  non-monotonicity.)

### Mechanism 3 — Warm-starting the candidate scan

G's cached best_guess evaluated first gives a strong incumbent; G's
candidate_scores ranking is a near-perfect evaluation order (exact scores
shift only through the added words' group membership — an O(1) update per
candidate given their pattern bytes).

### Mechanism 4 — Pattern-matrix rebuild (minor; not a splice)

`response_decomposition` blobs are strictly positional — byte *i* is
answer *i* in canonical (sorted) answer-list order (`wordle_engine.py`:
`_encode_response(calculate_response(guess, answer)) for answer in
self.answer_words`). The corrected answer list is sorted with its 9 new
words interleaved, not appended — `aught` lands at index 179 of 3,209, for
instance — so "old blob + 9 bytes appended" would misalign every byte from
the first insertion point onward for every guess. There's no cheap splice
here: either insert each new byte at its correct index for every retained
blob, or just rebuild. Rebuilding is already fast (the vectorized kernel
path this project deployed 2026-07-05), so that's the recommendation; if a
splice is ever implemented instead, verify it byte-for-byte against a
fresh rebuild before trusting it.

### The honest limit

Nothing above certifies optimality by itself: ERD is a min over all
candidates and an added word can change the winner. Scans always run to
completion; the mechanisms make each scan cheap, not skippable.

## Part 6 — Procedure

One-time operations run from the scratchpad and are never committed; the
list-and-rename flip (Phase 0) is the one exception and goes through a PR.

### Phase 0 — Source, rename, and land the corrected vocabularies — complete

Completed by PR #158. The checklist remains as the source/provenance and
consumer inventory for the landed file transition; do not rerun it during the
remaining uplift.

1. Source the current 14,855-word NYT full dictionary — **done**.
   `Get_NYT_Candidates.py` (new script, mirrors `Get_NYT_Answers.py`'s
   style — both already carry their permanent names, renamed ahead of the
   file rename below since neither script's identity depends on which
   filename it currently writes to) scrapes the NYT Wordle web client's
   bundled dictionary directly; verified independently against a manual
   extraction, byte-for-byte identical.
2. Delta against `wordle.txt` computed in both directions — **done**:
   1,883 additions, 0 removals. No mixed-case handling needed anywhere in
   this plan; the growth-only rows throughout apply as written.
3. Gates confirmed: new answer list ⊆ new candidate list (the 14 words
   included, plus all 9 PSHAW-era words); both source lists sorted, unique,
   5-letter lowercase.
4. **Renamed** `wordle.txt` → `all_candidates.txt` and `NYT_wordlist.txt` →
   `all_answers.txt`, landing the corrected content under the new names in
   the same PR (no separate rename pass). `all_candidates.txt` names the
   file for what it is — the universe the anchored **candidate** vocabulary
   term (AGENTS.md) is drawn from — and pairs cleanly with
   `all_answers.txt`, whereas `wordle.txt` was opaque and `NYT_wordlist.txt`
   was easily confused with the file it's now renamed away from.

   Exhaustive inventory (verified by grepping the repository for both old
   filenames, not reconstructed from memory — an earlier draft of this
   list was incomplete):

   - **Routed through `runtime_paths.py` already — only that file changed.**
     `erd_search.py` imports `DEFAULT_ANSWER_LIST_PATH`/
     `DEFAULT_CANDIDATE_LIST_PATH` rather than hardcoding either name, so
     updating `runtime_paths.py` alone covered it.
   - **Had their own `ANSWER_FILE`/`WORDS_FILE` (or equivalent) literals and
     were edited directly:** `wordle.py`, `erd_swarm.py`,
     `verify_erd_cache.py`, `verify_erd_losses.py`, `import_cache.py`,
     `diag_ab_equiv.py`, `diag_ab_wall.py`,
     `diag_kernel_bench.py`, `diag_ordering.py`, `diag_order_tune.py`,
     `diag_toplevel_census.py`, `Get_NYT_Answers.py`, and
     `Get_NYT_Candidates.py` (including its answer-list plausibility check).
     These consumers still do not import from `runtime_paths.py`;
     centralizing them remains a separate optional cleanup.
   - **Test files whose literals were updated:**
     `tests/test_diag_toplevel_census.py`, `tests/test_erd_scaling.py`,
     `tests/test_kernel_equivalence.py`, `tests/test_pattern_matrix.py`,
     `tests/test_swarm_vs_engine_overhead.py`.
   - **Prose updated with the code:** `design.md`,
     `full_tree_plan.md`, `SWARM.md`, and a comment in
     `tests/test_queue_add.py`.

   No `schema_migrations` entry was added — this was a file rename, not a
   cache schema change. The full test suite passed before merge.
5. The rename and corrected content landed together. No temporary
   dated-capture file remains — git history holds the old
   content, and the old-`answer_list_id` cache rows (Phase 2 onward) hold
   the old world's computed results; neither needs a parallel on-disk file
   to survive through the soak. The phone does not pull until Phase 6.

### Phase 1 — Freeze and back up

1. Run the standing machine-state check (all three checks; stop with full
   `erd_search.py stop` if any service, process, or heartbeat is active). Then
   confirm no live claims and engage the disk-stop latch in the default queue
   so the ordinary systemd swarm cannot restart during recertification.
2. Checkpoint and back up `wordle_cache.sqlite3` and `erd_queue.sqlite3`.
3. Record the old `answer_list_id`, raw `ERD_ALL` row count, distinct
   branch-key count, and per-answer-count histogram in the migration ledger so
   the post-schema union can be reconciled exactly.
4. Check disk headroom: retained tables roughly double until retirement, and
   the dedicated recertification queue adds temporary coordination state.

### Phase 2 — Schema upgrade and re-tag migration

1. Land the persistent recertification implementation through a PR and install
   it on Linux before opening the production cache. It includes the
   budget-versioned schema and idempotent migration, taint propagation,
   manifest validation, queue/controller lifecycle, reporting, and
   import/export support described in Part 3.
2. Open the backed-up production cache with the new code. Let
   `ScoreCache._ensure_schema` create
   `branch_best_by_policy_and_budget`, move and verify all existing finite
   rows, and record its `schema_migrations` entry. Reopen it and prove the
   migration is a no-op before continuing.
3. Open the cache with the new answer list so
   `ScoreCache._ensure_answer_list` registers the new `answer_list_id`.
4. Copy old-id → new-id with table-specific conflict handling:
   - `branch_best_by_policy`: canonical `erd_answers_compliant` rows only. On
     a target collision, require exact `best_score` agreement within solver
     tolerance; a tied `best_guess` or its strategy's
     `max_remaining_depth` may differ. Retain the target after verification.
   - `branch_best_by_policy_and_budget`: budget-specific
     `erd_answers_compliant` rows only. Compare only the identical
     `solve_budget`; require the same score agreement and retain the target
     after verification.
   - `branch_loss_by_policy`: `erd_answers_compliant` only. On collision store
     `MAX(target.loss_budget, source.loss_budget)`, the widest reusable proof.
   - `candidate_scores`: all rows via `INSERT OR IGNORE`.
5. Copy **no** `ERD_ALL` canonical, budget-versioned, or loss row. Copy
   nothing for `erd_answers_unfiltered`.
6. Reconcile that the new-ID `ERD_ALL` population is empty in both best-row
   tables and the loss table before Phase 3. Any row there means the
   namespaces are no longer a trustworthy completion boundary; stop and
   investigate rather than deleting blindly.
7. Extract and freeze the eligible manifest and excluded-fallback inventory
   from the union of old-ID canonical and budget-versioned `ERD_ALL` best-row
   keys. Old losses are invalid under candidate-vocabulary growth and do not
   add targets. Reconcile source, eligible, excluded, and malformed counts
   before creating the dedicated queue.
8. Reconcile compliant inserts, verified canonical and same-budget
   collisions, retained target rows, and widened loss proofs against the
   source and target counts. Phase 0 is already live on Linux, so pre-existing
   new-ID compliant rows are expected input rather than assumed absent.
9. Old-ID rows stay intact as source inventory and rollback data until
   retirement.

### Phase 3 — `ERD_ALL` recertification swarm

1. Verify the installed persistent recertification support:
   - finite certification budgets assigned independently of a historical
     spine;
   - attempt generations and clean candidate-claim reset on every
     higher-budget retry;
   - canonical and budget-versioned best-row identities in both SQLite and
     `MemoryScoreCache`;
   - floor-taint propagation through every normal child status, especially
     `OVER_ERD_LIMIT`;
   - unconstrained exact, budget-specific exact, retrying, loss, deferred,
     and superseded lifecycle states with `max_remaining_depth` retained;
   - validated old-ID manifest extraction and strict ascending-size admission;
   - run-identity validation and restart recovery on a dedicated queue path;
   - adaptive recursive promotion disabled for this profile; and
   - the recertification overview in the shared report pipeline; and
   - `queue set-disk-stop`, since the procedure now deliberately engages the
     default latch before Phase 3 and again before Phase 5.
2. Against scratch cache and queue copies, run the Part 3 correctness and
   throughput gate at each proposed certification budget, including every
   manifest key above 200 answers. Do not begin production until certified
   results agree with unconstrained references, tainted attempts retry
   cleanly, the ladder maximum has a measured deferred rate, and the swarm
   reaches the required fraction of direct-pool throughput. If worker
   concentration fails the gate, allow multiple ordinary top-level branches
   to remain active and repeat the measurement.
3. Run the standing machine-state check before production — all three checks —
   and confirm the default queue's disk-stop latch remains set. The ordinary
   epoch swarm and report server stay stopped; only the direct recertification
   supervisor may write the production cache.
4. Initialize `erd_recertification_queue.sqlite3` from the recorded old/new
   IDs, candidate-list digest, eligible and excluded manifest identities,
   frozen certification ladder and deferred policy, and implementation
   revision. Refuse a nonempty queue whose identity does not match exactly.
5. Start six recertification workers against the dedicated queue and the
   production cache. Admit one answer-count wave at a time in bounded batches.
   Monitor through:

       python3.13 erd_search.py view \
           --queue-path erd_recertification_queue.sqlite3 \
           --cache-path wordle_cache.sqlite3 --watch

6. At every wave and certification-budget boundary, reconcile unconstrained
   exact, budget-specific exact, retrying, loss, deferred, and superseded
   states against current-ID cache rows and the eligible manifest count before
   advancing. Resolve every deferred backing reference at its exact budget.
   Resume from the dedicated queue after a clean or unclean stop; never use a
   manual `--start-size` assertion as a substitute for the durable ledger.
7. Run until every eligible manifest key is certified or durably deferred.
   Use the pilot's measured attempts/hour, retry fractions, and deferred
   fraction for the production ETA; the earlier 3.0–8.7-day unlimited-depth
   estimate is only a stress reference. The phone remains frozen, and no
   user-facing solve reads the partially populated new namespace during this
   phase.

### Phase 4 — Recompute and seed

1. Stop the recertification supervisor cleanly and reconcile the entire
   eligible old-ID manifest against current-ID certified or deferred terminal
   states. Retain the dedicated queue through reverification as the execution
   ledger.
2. Clear and re-seed the default queue from the new 3,209-word root. Release
   its disk-stop latch and run the controlled ordinary swarm to compute the
   root census and the 9 answer cones against the new candidate universe,
   warm via Mechanism 1. If this work creates an unconstrained exact row for a
   deferred manifest key, promote it to certified and mark its deferred entry
   superseded.
3. Sandwich-seed `erd_answers_unfiltered` per Part 4 using current-ID exact,
   untainted outer rows; copy the full compliant row and seed no losses.
4. Run the un-pinched count; decide eager rebuild vs. lazy.
5. After the controlled queue drains, stop both services and re-engage the
   default queue's disk-stop latch so Phase 5 has exclusive, stable inputs.

### Phase 5 — Reverification

1. **Structural reconciliation.** Row counts per table, policy, and
   `answer_list_id` match the schema-migration and copy/seed/rebuild ledgers.
   Reproduce the eligible and excluded manifest digests from untouched old-ID
   rows. For `ERD_ALL`, map every eligible key to exactly one current class:
   - **certified:** one current-ID canonical best row with
     `solve_budget IS NULL` and non-NULL `max_remaining_depth`; any historical
     deferred ledger entry is marked superseded; or
   - **deferred:** one nonsuperseded durable deferred entry naming its final
     certification budget, attempt generation, and reason, backed by either
     the current-ID budget-versioned best row at exactly that budget or a
     current-ID loss proof whose `loss_budget` covers it, and no canonical
     best row.

   Reject missing, doubly current, or unbacked keys. Require every excluded
   fallback key to remain outside the eligible queue ledger. Do not require
   total current-ID best-row count to equal the manifest count, because
   deferred losses have no best row, finite budgets have separate rows, and
   recursive solves legitimately create additional new partition keys. Only
   the certified class is eligible for the sandwich.
2. **Sampled exactness check** (`erd_answers_compliant` and seeded
   unfiltered rows): fresh from-scratch solve per sampled branch; require
   `best_score` equality against the fresh solve; then verify the stored
   pair directly — evaluate the stored best_guess on the branch and
   require it to attain the stored best_score and stored max_depth.
   (Fresh-solve max_depth equality is tie-dependent — an equally-optimal
   different guess may carry a different worst case — so the stored pair,
   which is what the row asserts, is what gets checked.)
3. **Recertification audit.** Use `verify_erd_cache.py` in read-only audit mode
   over a stratified random sample of certified `ERD_ALL` rows, with fresh
   solves and the same stored-pair discipline. Separately sample deferred
   entries and resolve their exact final-budget row/loss backing. Sample
   budget-versioned rows at several budgets against fresh finite-budget solves,
   including a forced floor-then-ceiling-cutoff case. The verifier is an
   independent checker, not the production computation path.
4. **Loss sweep.** `verify_erd_losses.py` over retained compliant losses.
5. **Test suite.**
   `python3.13 -m unittest discover -s tests -t . -p 'test_*.py'`.

### Phase 6 — Phone catch-up

Only after Phase 5 passes in full. Code/list sync and cache sync are two
separate, decoupled mechanisms on this project, not one combined step:

1. **Code and lists.** The phone (Working Copy) pulls `main` — a plain git
   pull, picking up the Phase 0 rename/content PR and everything since.
2. **Cache.** The already-established export/import dance brings the
   phone's database in line with rocky's post-migration, post-recertification
   `wordle_cache.sqlite3` (`wordle_cache.sqlite3` itself is not tracked in
   git, so step 1 alone never touches it).

The order is mandatory: complete step 1 before step 2. The new cache-writing
code understands both canonical and budget-versioned rows; old phone code
does not know the new table or lookup contract. `answer_list_id` scoping
prevents cross-vocabulary reads but cannot supply missing schema support.
Deploy the new code to the phone and let its idempotent `_ensure_schema`
migration complete before placing the migrated Linux database on it. Never
repair the phone schema with manual SQL.

### Phase 7 — Retirement

After enough post-cutover use to trust the new world — the soak should
include simulated games against each of the 9 new answer words and a few
of the 14 rescued guess words:

1. Delete old-`answer_list_id` rows from
   `branch_best_by_policy`, `branch_best_by_policy_and_budget`,
   `branch_loss_by_policy`, `candidate_scores`, and
   `response_decomposition`; vacuum/checkpoint; re-sync the phone.
2. Archive or remove `erd_recertification_queue.sqlite3` and its telemetry
   only after the soak no longer needs its execution evidence.
3. Release the disk-stop latch (`erd_search.py queue clear-disk-stop`) —
   `run` refuses to start while it is set — then resume epoch testing (fresh
   `telemetry_epoch`).

## Open items

- Eager vs. lazy for the un-pinched `erd_answers_unfiltered` remainder.
- Soak length before retirement.
- Certification budgets after the initial budget 5, the production ladder
  maximum, and the acceptable deferred fraction; freeze all three from the
  Part 3 scratch pilot.
- Whether to centralize the remaining `ANSWER_FILE`/`WORDS_FILE`-style
  consumers listed in Phase 0 through `runtime_paths.py` as a later cleanup.

Resolved since the earlier standalone design: inline-solving missing children
(retained inside ordinary swarm candidate evaluation), whether Mechanism 2's
ceiling-injection machinery is worth building (measured negligible payoff;
proofs retained, implementation deferred), and whether Phase 3 branches fit
the swarm (yes — leave old `ERD_ALL` rows under the old `answer_list_id` and
admit their keys into a dedicated queue). The earlier unlimited-depth cost
sample remains evidence about the stress path; it is not the finite ladder's
production estimate.

## Resolved

- **14,855-word candidate list**: sourced via `Get_NYT_Candidates.py`,
  scraping the NYT Wordle web client's bundled dictionary directly.
- **`wordle.txt` removal set**: 0 removals (1,883 additions only) — the
  growth-only case applies throughout; no mixed-case handling is needed.
- **File naming**: `wordle.txt`/`NYT_wordlist.txt` → `all_candidates.txt`/
  `all_answers.txt`, completed in Phase 0 with an exhaustive (grep-verified)
  consumer update. The scraper scripts carry their permanent names
  (`Get_NYT_Wordlist.py` → `Get_NYT_Answers.py`, `Get_NYT_Words.py` →
  `Get_NYT_Candidates.py`) and their write targets and plausibility checks
  now use the corrected filenames.
- **Phone sync mechanism**: git pull (Working Copy) for code and lists;
  the existing manual export/import dance for the cache, independently —
  see Phase 6.
- **Recertification execution**: old `ERD_ALL` rows remain exclusively under
  the old content-addressed `answer_list_id`; strict decoding and current
  answer membership admit only ordinary answer branches to the dedicated
  recertification queue, while valid interactive fallback branches are
  excluded and accounted. The current-ID namespace begins empty, so each
  target is an ordinary swarm branch run first at certification budget 5. A
  floor-free attempt writes a canonical unconstrained exact row. A
  floor-tainted result writes a separately keyed finite-budget row, with taint
  preserved even through ceiling cutoffs, then retries under the frozen ladder
  and becomes durably deferred at its measured maximum rather than falling
  through to `budget=None`. Budget-versioned storage prevents solves at other
  budgets from destroying either a certificate or deferred backing. The
  terminal ledger classifies every eligible manifest key as currently
  certified or deferred and permits a later certificate to supersede a
  deferred entry. Missing children solve inline, leaves-first admission makes
  certified retained children warm, and `verify_erd_cache.py` remains an
  independent audit tool. Prior
  unlimited-depth samples estimate
  **3.0–8.7 days of stress-path work on 6 workers, ~5.4 days central**; the
  production pilot measures the ladder's throughput, retry and deferred
  fractions, certified `max_remaining_depth`, and concurrent top-level branch
  concentration.
- **Old cache reused as ceiling ("hint") for recertification — measured, not
  worth it**: pricing the old `best_guess` and passing its cost as an
  alpha-beta ceiling gave no net speedup (181 branches: 71 faster, 110
  flat-or-slower, totals within 0.2%) — the engine's built-in Σk²
  candidate ordering already reaches a near-optimal incumbent unaided. The
  old cache's real value is its row-key list and leaves-first schedule
  (guaranteeing warm children), not its stored scores.
- **Empirical corroboration of the growth-only theorem**: 432 branches
  fresh-solved under the new vocabulary, 28 (6.5%) strictly improved, zero
  regressed — no branch's new optimum was ever worse than its old stored
  value, exactly as Part 1 predicts and the signature real corruption would
  have broken.
- **Max_remaining_depth is not provably monotone** under either vocabulary
  growth (Part 1) or answer-set growth (Part 1, "ERD is not monotone") —
  the cache only stores the worst case of whichever tree wins the *ERD*
  race, not a separately minimized worst case. Two theorem statements
  corrected; ERD monotonicity and budget-feasibility monotonicity are
  unaffected and still hold.
