# ERD cache re-verification plan (v2)

## Problem

The mid-loop overrun publisher could hand off a frame that was still riding its
entry alpha-beta ceiling (no achieved best yet), marking the frame's
ceiling-priced prefix candidates as done in an **exact** cooperative branch.
The published branch's finalize then judged only the remainder.  Fixed in
PR #115 (ceilinged frames now publish ceilinged branches); any branch finalized
by the swarm while the publisher was live (epochs 3–4, roughly 2026-07-08 to
2026-07-10) is a candidate for two kinds of bad cache entry:

1. **Suboptimal best** — the remainder's winner was cached although a discarded
   prefix candidate may have been better.  Same error class as the June
   reclaim-while-alive bug: the stored value is **≥ the true optimum**, and it
   is always the ERD of a real strategy (achievable, never fabricated).
2. **False loss** — the remainder was all infeasible, so a loss was cached,
   although a discarded prefix candidate (priced out ≥ ceiling, never proven
   infeasible) may have been feasible.  This class did not exist in June: the
   reclaim bug could not manufacture losses.  A false loss also poisons every
   budget below the stored one, and can propagate upward — a parent that
   trusted it may have discarded a feasible candidate and cached a value that
   is (again) achievable but too high.

Handoff events themselves were not durably recorded, so the exact victims
cannot be enumerated.  What bounds the damage is the finalize log: every branch
the swarm finalized in the affected epochs has a `branch_finalize_log` row.
Intersecting its 32,926 distinct branch keys with the cache gives the suspect
populations (measured 2026-07-11):

- **8,850 loss entries** in `branch_loss_by_policy`
- **30,570 best entries** in `branch_best_by_policy`

## Two passes, in this order

### Pass 1 — refute suspect losses (new tool)

The June machinery cannot touch this class: there is no cached value to check
for consistency, and no ceiling to seed.  A loss is verified by re-running the
budget-capped solve with the fixed engine:

- **Refuted** (a winning strategy exists within the stored budget): delete the
  loss row and write the found best through the normal cache path.
- **Confirmed** (exhaustive disproof succeeds again): keep the row.

Scope: exactly the 8,850 suspect keys (loss ∩ finalize-log), processed
leaves-first (ascending `n_words`) so a refuted child loss is corrected before
any parent that may have inherited its poison is examined.  Refutation is the
cheap direction (finding one feasible candidate ends the check); confirmation
repeats the full disproof, which the budget cap keeps bounded — suspect losses
are concentrated at small budgets.

Tool: `verify_erd_losses.py`, mirroring `verify_erd_cache.py`'s wave/resume
structure.  Derive the suspect key list from `branch_finalize_log` in the
telemetry file at startup; no schema changes.

### Pass 2 — value consistency sweep (existing tool)

`verify_erd_cache.py` already implements the right check for the suboptimal
best class, because the error direction is identical to June's: leaves-first
waves, each entry re-verified against its sub-branches' *cached* values (cache
arithmetic, no re-solving), corrections written in place, resumable.  Run it
over the full `branch_best_by_policy` as before — the full sweep is
self-limiting (correct entries confirm in near-zero work), automatically
covers the 30,570 suspect keys and any upward propagation beyond them, and
re-covers anything the June pass ran before later contamination.

Before running: review the script against the current schema and engine
interfaces — it last ran in June and the codebase has moved (tests/ layout,
budget-keyed cost model, telemetry split).  Fix forward if bitrotted; the
algorithm stands.

Pass 1 must complete before pass 2 starts: pass 2 trusts cached sub-branch
values, and a false child loss reads as "no strategy" to every parent above it.

## What this is NOT

- **Not a flush.**  No rows are deleted up front; corrections are surgical.
- **Not the June pass.**  That pass (reclaim-while-alive bug, commit `774ac29`)
  was executed to completion; this plan supersedes its document.  The
  contamination window, the false-loss class, and pass 1 are new; pass 2
  reuses the June tool unchanged in role.
- **Not part of PR #115.**  The fix prevents new contamination; this is the
  cleanup of what epochs 3–4 may have left behind.

## Sequencing

1. PR #115 (ceiling propagation) merges and deploys.
2. With all swarm workers stopped (they are, since the 2026-07-10 shutdown):
   run pass 1, then pass 2, on the Linux cache.
3. Epoch-5 re-queue and swarm restart only after both passes complete — the
   epoch-5 corpus (the shelved trap branches) sits exactly in the suspect
   neighbourhoods, and solving on top of uncorrected entries would launder
   them into new results.
4. Export and sync to the phone only after the passes complete; the phone's
   cache copy carries the same suspect entries until then.
