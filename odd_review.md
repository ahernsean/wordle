❯ Review the work in worktree e3ad888                                           

● I'll start by locating the worktree and understanding what commit e3ad888
  contains.

  Ran 2 shell commands (ctrl+o to expand)

● The worktree at .claude/worktrees/adaptive-waddling-abelson is checked out at
  e3ad888. Let me enter it and examine the work done there.

● Entering worktree(/home/ahern/work/wordle/.claude/worktrees/adaptive-waddling-
                   abelson)
  ⎿  Switched to worktree on branch worktree-adaptive-waddling-abelson
     /home/ahern/work/wordle/.claude/worktrees/adaptive-waddling-abelson

  Listed 1 directory, ran 1 shell command (ctrl+o to expand)

● Let me get the PR details and the full diff to understand what the agent was
  asked to implement.

  Ran 2 shell commands (ctrl+o to expand)

● Now let me run the test suite first to verify the green claim, and read the
  key changed files in parallel.

  Ran 2 shell commands (ctrl+o to expand)

● Tests are green (762 tests). Let me now review the core implementation files.

  Ran 2 shell commands (ctrl+o to expand)

● Now let me read the new methods in erd_queue.py and the test file.

  Read 1 file, ran 1 shell command (ctrl+o to expand)

● Let me check a few specific correctness points and the analyze_epoch0.py
  script more carefully.

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Let me read the exact docstring and the accumulation code in
  estimate_candidate_work_cutoff.

  Read 2 files (ctrl+o to expand)

● I have enough to give a full review. Tests are all passing (762 green). Here's
  the assessment:

  ---
  ★ Insight ─────────────────────────────────────
  The runtime code path — erd_queue.py, erd_swarm.py, wordle_engine.py — is the
  load-bearing layer here. Bugs there corrupt the cache or skew the cost model.
  Bugs in analyze_epoch0.py only affect the offline validation gate. The review
  below separates these clearly.
  ─────────────────────────────────────────────────

  ---
  Review: PR #76 — Measurement Layer (issue #67, §9)

  Scope: Budget-keyed cost model, new telemetry tables (branch_finalize_log,
  candidate_accuracy, telemetry_epoch), backfilled columns on existing tables,
  estimate_candidate_work[_cutoff], metric_observer hook, and the offline
  analyze_epoch0.py gate script.

  Tests: 762 green (PR claimed 724 — the extra 38 are from this branch itself,
  matching the new test_claim_packing_measurement.py). The test suite validates
  cost-model re-key, migration idempotency, epoch tagging, telemetry inserts,
  and both estimate functions.

  ---
  Issues Found

  1. Docstring misrepresents the cutoff accumulator — wordle_engine.py:1019-1022

  The docstring of estimate_candidate_work_cutoff says:

  ▎ "the accumulation uses each sub-branch's lower bound (2 - 1/k)"

  The code at line 1057 uses (3.0 - 2.0 / k). The inline comment at lines
  1051-1056 correctly identifies this as a "monotone imperfect-split proxy," not
  the admissible lower bound. These two statements directly contradict each
  other — the docstring is wrong. A future maintainer calibrating the proxy
  (which the inline comment says is "the dial to calibrate offline") would look
  at the docstring and think they're working with a lower bound when they're
  not.

  Fix: the docstring should read "accumulating a per-group ERD proxy (3 − 2/k)"
  and drop the lower-bound claim.

  2. estimate_claim_reduction doesn't filter branch_finalize_log by epoch — 
  analyze_epoch0.py:172-173

  The candidate_accuracy query correctly filters WHERE epoch = ? (line 176-177).
  But branch_finalize_log is read without an epoch filter:

  total_by = {bytes(r[0]): r[1] for r in conn.execute(
      "SELECT branch_key, n_claims FROM branch_finalize_log WHERE n_claims > 
  0")}

  When epoch 1 data arrives (after the packer deploys), total_by will contain
  branches from all epochs. Those epoch-1 branches will have N = 0 from ng_by
  (which only sees epoch-0 candidate_accuracy rows), so they'll be treated as
  all-gated — inflating the reduction estimate. The gate calculation silently
  becomes wrong without any error signal.

  Fix: add WHERE epoch = ? to the branch_finalize_log query and bind it to
  epoch.

  3. has_self hardcoded to False in offline metric recomputation — 
  analyze_epoch0.py:138

  evaluate_metric passes has_self=False unconditionally. When a candidate
  appears in its own answer set (has_self=True), the cost_lb computation
  differs: 3 - (G+1)/n vs 3 - G/n. Since group_sizes stores sizes only (not
  which singleton is the self-group), has_self can't be recovered. This is a
  genuine limitation — but it's undocumented. A candidate near the gate boundary
  could be mis-classified between gated and non-gated in the offline replay.

  Fix: add a comment at the has_self=False call noting that the column isn't
  stored and that the approximation is acceptable for the validation purpose
  (since mis-classification is rare and the direction is conservative).

  4. Schema comment contradicts code — erd_queue.py, candidate_accuracy table

  The table comment says:

  ▎ "Written only for non-gated rows (a gated row's predicted work is exactly 
  ▎ 0)."

  But erd_swarm.py does write gated rows (down-sampled
  1-in-ACCURACY_GATED_SAMPLE_EVERY). The comment should reflect the actual
  down-sampling policy.

  5. Naming: _pctile — analyze_epoch0.py:113

  _pctile abbreviates percentile. Per the codebase naming rule ("use full names,
  do not abbreviate identifiers"), this should be _percentile.

  6. Naming: local variable fe — analyze_epoch0.py:138–145

  fe for "false expensive count" is opaque at a glance. false_expensive_count
  (or at minimum false_expensive) keeps the naming consistent with the codebase
  convention.

  7. Test name mismatch — test_erd_swarm_unit.py / 
  test_claim_packing_measurement.py

  test_cutoff_never_underestimates_reached_groups names a property ("cutoff >=
  actual reached groups") but only asserts cut <= uncut — trivially true since
  the cutoff stops early. The actual non-underestimation property (cutoff >=
  real engine reached groups) is not tested. The test name should say what it
  actually tests: that the cutoff is bounded above by the uncut sum, i.e.,
  test_cutoff_no_greater_than_uncut.

  ---
  Summary

  The runtime path — cost-model re-key, budget propagation, schema migration,
  metric_observer hook, telemetry writes — is correct and clean. The migration
  is idempotent, the fallback chain (specific budget → aggregate) is consistent
  everywhere it's threaded, and the ERD pruning invariants are untouched. All
  762 tests pass.

  Findings 1 and 2 are the ones to act on. Finding 1 (docstring) is a
  maintenance hazard for anyone calibrating the proxy. Finding 2 (epoch filter)
  is a latent bug that will silently corrupt the reframed gate calculation the
  moment epoch 1 data co-exists in the database — exactly the transition this PR
  is preparing for. Findings 3-7 are lower priority but worth fixing before
  this analysis script is relied on for a go/no-go decision.

  result: PR #76 measurement layer reviewed — runtime is clean, 762 tests green;
  two findings warrant action: docstring misrepresents the cutoff proxy
  accumulator (wordle_engine.py:1019), and estimate_claim_reduction omits epoch
  filter on branch_finalize_log (analyze_epoch0.py:172), which silently mixes
  epochs once epoch 1 data exists.

✻ Sautéed for 5m 25s

※ recap: Reviewed PR 76, the measurement layer before the adaptive claim packer. Two fixes needed: the cutoff-metric docstring misstates its accumulator, and estimate_claim_reduction skips an epoch filter that will corrupt the reframed gate once epoch 1 data exists. (disable recaps in /config)
