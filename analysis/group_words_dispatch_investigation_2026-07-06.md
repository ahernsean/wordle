# group_words fast-path dispatch: n=500 regression investigation

Written while addressing PR #94 review comments. Records the investigation
into the reviewer's CONFIRMED finding that the unconditional `branch_indices`
computation (and, more precisely, the unconditional `group_words` fast-path
dispatch) regresses large, deep-recursion branches like n=500, and why the
fix committed alongside this document (warm/cold dispatch + ski-rental
promotion, `_GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD = 250`) does not fully
close the gap. Intended for the next implementer/reviewer to pick up from,
not as a design document to follow blindly.

## The acceptance bar

From the review thread: "rerun this A/B — matrix-on nodes at n=500 must be
>= main's, and the n=8/30 walls must stay at their current post-PR values."

## Root cause, precisely

`ScoreCache` deduplicates identical sub-branches across different parent
paths in the search tree (`_solve_subset`'s `score_cache.read_with_depth`
reuse check). This means the *effective* number of distinct (guess, branch)
pairs that ever reach `group_words` depends entirely on how much the
recursion tree's leaves collide onto the same branch content — which is a
property of the branch size relative to the vocabulary, not something
observable from any single node.

Measured directly (instrumented `ResponseCache.group_words` call sizes for a
full n=30, budget=5 solve):

```
sizes seen (size: count): [(2, 25944), (30, 1)]
total group_words calls: 25945
```

n=30's *entire* solve makes only ~26,000 total `group_words` calls, almost
all against a small number of distinct size-2 sub-branches reached
repeatedly from different parents and served from `ScoreCache` after the
first visit. There is essentially no "reuse" of any individual guess beyond
a handful of times — the fast path's ~250x edge over a cold `_ensure` decode
dominates for the whole solve, and this holds regardless of dispatch
strategy (warm/cold check, any promotion threshold, or none at all — a
controlled sweep at K ∈ {40, 250, 1000, ∞} all completed n=30's exact solve
in 0.43-0.50s).

n=500 has no such luck: the sub-branch space at that size is too large for
meaningful cache collision, so each of the ~13,000 guesses in `guesses=vocab`
genuinely gets re-evaluated against many *different* branches — hundreds of
times per guess over a full run. In this regime, the pure-Python "off" mode's
strategy — decode every guess once, up front, via the root's O(vocab)
candidate-ordering sweep (an expensive ~60-65s one-time cost, since decoding
one guess against the full ~3200-word answer universe measures ~5ms) — turns
out to beat every `group_words` dispatch variant tried, including the one
committed:

| Config | n=500, 120s deadline | nodes |
|---|---|---|
| off (unchanged reference) | full 120s | 14.1M – 15.8M (varies by run) |
| on, original bug (always fast path, no dispatch) | reviewer's own A/B | 5.6M (vs main's 12.4M) |
| on, warm/cold + K=250 promotion (this PR's fix) | full 120s | 7.27M |
| on, warm/cold, promotion **disabled** (K=∞, i.e. always take fast path when cold) | full 120s, clean isolated A/B | 10.4M (vs off's 15.8M in the same run) |

Even with promotion completely disabled — i.e. the dispatch is just "take
the fast path unless already warm," which the guess *never* becomes because
nothing ever calls `_ensure` — n=500 still trails off mode by a real, ~1.5x
margin in a clean, controlled, single-process A/B (same `PatternMatrix`
instance, same branch, sequential runs, no cross-run interference). This
rules out promotion *timing* as the cause: the fast path's per-call NumPy
overhead (~15-25us, measured directly in this environment) essentially never
amortizes away in a regime where a guess really is called hundreds of times,
because a warm Python loop over a tiny branch costs on the order of 1-2us —
a >10x gap that dominates once you're doing millions of such calls.

## What didn't work, and why

Tried, in order:

1. **Ski-rental promotion, K=250** (this PR's final threshold). Fixes the
   *n=30/8/81/146* regime cleanly (no measurable difference from K=40,
   K=1000, or no cap at those sizes, since promotion never triggers there).
   Only partially recovers n=500 (7.27M vs off's 14-16M) because by the time
   most guesses cross the 250-use threshold (given real per-guess reuse
   rates around 40-50 per 20s, extrapolating to roughly the threshold itself
   over a 120s run), the run has already spent most of its budget paying
   fast-path overhead *and* is only just reaching the "everyone's warm"
   payoff phase as the deadline arrives.
2. **No promotion at all (K=∞)**. Still ~1.5x behind off mode at n=500 in a
   clean, controlled run (10.4M vs 15.8M) — see above. Rules out "promotion
   is happening at the wrong time" as the whole story; the fast path itself
   is simply slower than a warm loop at the volumes n=500 reaches, with or
   without ever converting to warm.
3. **Size-based gate** (`len(subset) >= ORDER_MIN_N`, no warm/cold tracking
   at all): fixes n=500 cleanly (8.03M vs off's 6.85M in one controlled
   run) but **completely breaks n=30** (60.14s wall, matching off mode
   exactly) — because n=30's few size-2 sub-branches, when forced onto the
   loop, each trigger the *first* `_ensure` for whichever guesses reach them,
   and since nothing else in "on" mode's vectorized path calls `_ensure` at
   all, this reintroduces the full ~60s "decode all 13k guesses from
   scratch" cost, just spread across sub-branches instead of concentrated at
   the root — the same total tax, paid a different way.

The throughline: **any total cost the search pays for decoding a guess must
be paid exactly once per guess it actually uses, however that payment is
distributed in time** (root-level batch, ski-rental promotion, or forced
warm-loop-on-small-branches). Off mode pays it all at once, up front, in a
single large but predictable block — and because *all* 13k guesses in
`guesses=vocab` end up needed eventually in a genuinely deep n=500 search,
front-loading that cost turns out to be a good bet. Nothing tried here beats
that specific bet for n=500 without regressing n=30, because n=30's win
depends on it being wrong (very few guesses are truly needed, so batch-
decoding all 13k up front is wasteful there).

## What would likely actually fix it

Not implemented — a larger design change than this section's scope:

A **global, cross-branch signal** (not per-guess, not per-node-size) that
detects "this solve is in the many-guesses-reused-many-times regime" and
switches to batch-decoding the remaining un-warmed guesses in one shot,
combined with keeping the current vectorized ordering/pre-filter (§4) for
the win on ordering/pruning cost, which is unrelated to this dispatch
question and already works well at every size tested. A plausible signal:
a running total of `group_words` fast-path calls *across all guesses*
within one `ResponseCache` instance's lifetime; once it crosses a threshold
that n=8/30/81/146's *entire* solves never reach (their total volumes are in
the tens of thousands, not millions), switch permanently to "always ensure,
always loop" for that instance. This was not implemented or validated here
for lack of remaining scope/time in this section — flagging it as the
follow-up path rather than guessing further.

## What's committed

`_GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD = 250` (the ski-rental design the
review originally proposed) — it does not regress n=8/30/81/146 at all, and
it measurably improves n=500 over the original bug (5.6M → 7.27M in the
reviewer's own methodology), even though it does not fully close the gap to
off mode's throughput at that size. Full test suite green; correctness
(exactness of the fast/slow path equivalence) is unaffected by any of this —
every configuration above produces bit-identical `ERD MATCH` results, this
is purely a wall-clock/throughput question.
