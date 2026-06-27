# Adaptive chunk claiming for cheap candidates

## Problem

A swarm worker claims and evaluates **one candidate at a time** against a branch.
`ERDQueue.claim_candidate` hands out a single candidate index; the worker loop
(`_BranchWorker.cooperative_solve` and `_BranchWorker.run` in `erd_swarm.py`)
calls `evaluate_claim` for that one index, folds the result into the branch's
shared best, marks the slot done, and claims the next index. A branch finalizes
only once all `n_candidates` slots are done.

Each claim carries fixed coordination cost: an `INSERT`/lookup transaction to
take the slot, a shared-best read, a finalize-coverage check, and a telemetry
write. That cost is justified when the candidate's evaluation does real work. It
is not justified when the candidate is cheap.

At a branch reached after several guesses, most candidates are cheap: the branch
holds few answer words, so a candidate either splits them into singletons or is
dominated immediately, doing only a handful of search nodes before returning.

Measurements from a multi-day run quantify this. In `claim_telemetry`
(`erd_queue.sqlite3`), over a 24-hour window:

- 99.7% of claims (12.09M of 12.13M) evaluated their candidate in **fewer than
  10 search nodes**.
- Claim cadence was ~141 claims/sec across 6 workers — about 42 ms per claim
  cycle, of which the candidate evaluation itself was microseconds.
- Mean node rate fell from 6.73k/s to 0.14k/s as the mean number of guesses
  already played rose from ~1.5 to ~3.

`claim_retries` was ~0 throughout, so this is not lock contention. The unit of
work is simply too small to amortize the per-claim coordination.

## Proposed direction

Hand out a **chunk** — a contiguous slice of the candidate list — per claim,
with the chunk size chosen so evaluation cost amortizes coordination cost.

A chunk is the anchored unit of work in this codebase: a contiguous slice of a
branch's ranked candidate list, claimed by one worker. Today every chunk is
size 1. The proposal is to size it adaptively:

- **Expensive candidates → chunk size 1** (current behavior). A candidate that
  does substantial search work should still be claimed alone, so it can be
  redistributed if its worker dies and so its result publishes promptly.
- **Cheap candidates → larger chunk.** When predicted per-candidate work is far
  below coordination cost, a worker claims a run of indices in one transaction,
  evaluates them in-process without a queue round-trip between them, folds each
  into the shared best, then marks the whole run done at once.

The cost inputs already exist and currently gate the inline-vs-cooperative
decision in `_subbranch_solver`:

- `_typical(n)` — predicted node cost for a branch of `n` words.
- `_node_time_ema` — measured seconds per node.
- `_coord_ema` — measured coordination seconds per claim.

A first cut: `chunk_size ≈ clamp(1, MAX, coord_seconds / per_candidate_seconds)`,
where `per_candidate_seconds = node_time × predicted_nodes_per_candidate`.

## Invariants to preserve

These hold for single-candidate claiming and must continue to hold. They are
documented in the engine and queue code; an implementer should re-read them
before changing the claim loop.

1. **Shared-best folding stays per candidate.** Chunking defers the queue
   round-trip between candidates, not the best update. A worker still updates
   (and may refresh) the branch's shared best between candidates in its chunk.
2. **Taint propagation is unchanged.** Any candidate excluded by the depth
   budget taints the branch, whether or not it is the winner.
3. **Finalization coverage is unchanged.** A chunk marks a contiguous set of
   slots done; finalize still triggers only when all `n_candidates` are done.
4. **Sequential-sibling pruning is preserved.** Candidates within a chunk are
   evaluated in order, so the carried partial best tightens exactly as in a
   single-worker sweep. Across chunks held by different workers, the existing
   shared-best refresh supplies the cross-worker bound. No new cross-subtree
   cancellation is introduced.
5. **Reclaim still works.** A worker that dies holding a chunk leaves its slots
   `done=0`; `reclaim_stale_claims` frees them and they reappear as gaps,
   re-handed-out (possibly in a smaller chunk). The end state is identical.

## Open questions for the implementer

- **Single-candidate publish semantics.** The current single-candidate unit lets
  a candidate's result publish at its own level as soon as it finishes. A chunk
  defers publication of the whole run. Decide how large a chunk may grow before
  this delay hurts pruning or load balance, and whether a worker should publish
  partial-chunk progress.
- **Load imbalance.** One slow candidate inside a large chunk blocks that chunk's
  completion and reduces parallelism. Cap chunk size and/or split on the first
  candidate that exceeds a node threshold.
- **Reclaim window.** Holding more `done=0` slots widens the window in which a
  dead worker's claims wait for reclaim. Bound chunk size accordingly.

## Acceptance criteria

- For a branch whose candidates each cost below the threshold, the number of
  claim transactions to drain it drops from `n_candidates` to about
  `n_candidates / chunk_size`.
- The coordination share of wall-clock (from `claim_telemetry`) drops materially
  at three or more guesses played.
- ERD results are identical to single-candidate claiming: the same branch best,
  the same cache rows. An equivalence test should assert this on a fixed branch.
