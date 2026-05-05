# Adaptive Frontier Remediation Plan (Design Gap Closure)

Date: 2026-05-05

This plan compares the current adaptive implementation against `design.md` and defines a single-PR implementation plan to close all identified gaps.

## 1) Findings from current implementation

### A. "Scheduler mode" is overloaded and misleading in status output
- The code has two different concepts:
  1) A **phase** (`normal`, `stagnation-escape`, `endgame-stabilize`) computed from elapsed time and stagnation.
  2) A **pick category** (`exploit_cutoff`, `exploit_upside`, `explore_new_roots`, `anti_starvation`, `cache_harvest`, `stagnation_escape`, `endgame`) sampled each dispatch.
- Status currently reports `scheduler_mode` using phase, while `mode_counts` tracks pick-category samples.
- This creates confusing output because a scheduler does not operate in one global mode over an interval; it makes per-dispatch choices.

### B. `mode_counts` are near-binary because they reset each status interval
- `mode_counts` are reset immediately after `_emit_status()`, so values only reflect a tiny post-status window.
- This explains reports like `...=0/1` and makes counts non-diagnostic.
- Design intent is instrumentation meaningful for tuning and behavior interpretation.

### C. "stagnation intervals" is undefined and currently too coarse
- Stagnation is currently incremented when top-5 `(word, rounded lower bound)` signature is unchanged at status emit time.
- It is not documented in status payload and not tied to units beyond “status callback intervals”.
- This makes interpretation unclear and sensitive to 10-second cadence/rounding artifacts.

### D. Root exploration can stall when few roots are initially activated
- Initial enqueue excludes `dormant_root_keys`; only `top_words` are activated immediately.
- Additional root activation depends on `_maybe_expand_roots()` which is only called in some dispatch categories and probabilistically gated.
- This permits long periods where the run refines one root subtree instead of broadening top-k contenders, conflicting with design goals of incremental global refinement.

### E. Top-k report can show fewer than k contenders due to activation policy
- Reported contenders come from `root_candidate_records`, which only includes activated roots.
- If activation is sparse, "best guesses if halted now" can have 1 row despite `top_k=20` request.
- Design expectation is useful partial top-N ranking under budget, with breadth-first viability.

### F. Root upper bounds appear untrustworthy to users (e.g., >100 bits)
- Current upper uses `first_entropy + sum(p_i * log2(|S_i|))` recursively bounded by depth. Large values are mathematically possible for multi-step entropy sums, but currently unexplained.
- The status format lacks decomposition (immediate vs deeper bound), making values look suspect.

### G. Missing scheduler/data model alignment with design.md
- `design.md` calls for globally top-N-oriented work scoring and viability; implementation uses type preferences + weighted random category sampling with limited expected-value scoring.
- There is no explicit per-item expected global top-k improvement metric exposed for scheduling.

## 2) PR scope (single PR) to close gaps

## 2.1 Reporting and terminology cleanup
1. Replace `scheduler_mode` in status with two explicit fields:
   - `scheduler_phase`: one of `normal`, `stagnation_escape`, `endgame_stabilize`.
   - `dispatch_mix_window`: per-category dispatch counts over the **last status window**.
2. Add cumulative counts:
   - `dispatch_mix_total`: lifetime counts since run start.
3. Rename printed labels in `wordle.py`:
   - `scheduler phase`
   - `dispatch mix (window)`
   - `dispatch mix (total)`
4. Add short explanatory legend in periodic status lines.

## 2.2 Stagnation metric definition and observability
1. Define stagnation explicitly as:
   - number of consecutive status windows without improvement to top-k quality metrics.
2. Track and report concrete metrics:
   - `topk_min_lower_bound` delta,
   - `topk_signature_changed` bool,
   - optionally `activated_root_words` delta.
3. Rename status field from `stagnant_intervals` to `stagnation_windows` and include window duration seconds.

## 2.3 Guarantee minimum root breadth before deep refinement dominates
1. Introduce `min_activated_roots` target:
   - default `max(top_k, configurable floor)` capped by eligible roots.
2. Scheduler rule:
   - while activated roots < target, enforce activation-heavy dispatch (hard preference for `ACTIVATE_TOP_WORD`).
3. Deterministic dormants expansion pass:
   - remove pure-probability gate during bootstrap; use batch activation until breadth target is met.
4. Keep stochastic exploration for post-bootstrap only.

## 2.4 Make top-k output always represent a true top-k frontier snapshot
1. Ensure enough roots are activated to produce up to `top_k` rows early.
2. If fewer than k eligible roots exist, print explicit reason (`eligible roots = X`).
3. Add report field `activated root words: a/b (target t)`.

## 2.5 Scheduler instrumentation and selection quality
1. Keep weighted categories, but compute per-item score components and log sampled decision basis:
   - viability margin (`U - C_N`),
   - expected lower-bound lift proxy,
   - starvation age,
   - root-activation urgency.
2. Add a lightweight "decision trace" counter summary per window (no verbose per-step log by default).
3. Align categories/labels with design nomenclature and use consistent snake_case/hyphenation.

## 2.6 Bound explainability and sanity diagnostics
1. Extend contender report columns:
   - `immediate`, `deeper_lower`, `deeper_upper` (or equivalent decomposition).
2. Add one-line note clarifying cumulative multi-step entropy can exceed single-step `log2(|S|)`.
3. Add optional debug assert/check mode for bound invariants:
   - `lower <= upper`,
   - monotonic tightening after refinements,
   - prune cutoff consistency.

## 2.7 Tests and validation additions
1. Add deterministic seeded scheduler tests validating:
   - counts accumulation behavior,
   - phase vs dispatch fields,
   - bootstrap activation reaches breadth target.
2. Add integration test/smoke harness for status payload schema.
3. Reproduce the “single contender at top-20” scenario and verify it no longer occurs under same budget/seed.

## 3) Implementation sequence inside one PR

1. **Status model refactor** (field rename/additions + CLI output updates).
2. **Stagnation metric formalization** (new counters + labels).
3. **Root breadth bootstrap policy** (min activated roots, deterministic expansion).
4. **Scheduler instrumentation improvements** (window + total counts, trace summaries).
5. **Bound decomposition reporting** (contender table columns and explanatory text).
6. **Tests** (unit + integration/smoke), then calibration/tuning constants.

## 4) Acceptance criteria

- Top-20 request reports at least 20 contenders once 20 roots are eligible and budget allows bootstrap activation.
- Status output terminology is unambiguous:
  - phase vs dispatch mix separated.
- `dispatch_mix_window` counts are meaningfully >1 in active runs; totals monotonic.
- Stagnation metric has explicit unit and documented trigger.
- Scheduler demonstrates early breadth then deeper exploitation in traces.
- Bound decomposition removes confusion around large cumulative upper bounds.

## 5) Risks and mitigations

- Risk: over-prioritizing breadth hurts depth quality late in budget.
  - Mitigation: only enforce breadth until `min_activated_roots` target, then revert to weighted scheduler.
- Risk: extra instrumentation increases callback overhead.
  - Mitigation: aggregate counters, avoid per-dispatch textual logs by default.
- Risk: changed status schema may break consumers.
  - Mitigation: keep backward-compatible aliases for one release window if needed.

## 6) Notes for collaborative iteration

This plan is intentionally implementation-ready for a single PR with multiple commits. The highest-value first cut is:
1) reporting clarity + counts fix,
2) breadth bootstrap guarantee,
3) stagnation metric rename/definition.
