# Wordle adaptive refinement frontier search design

2026-04-27 @ 22:41:44 EDT

## Problem statement

Given a current set of possible Wordle answers, rank candidate guesses by expected information gain over multiple future guess levels, subject to a bounded computation budget.

The search must support the following behavior:

- Return useful partially refined rankings when the time budget expires
- Continue refining candidates incrementally rather than fully solving one candidate before touching others
- At any time, provide current “best words” list as reflected by progressive refinement
- Reuse durable knowledge across different Wordle problems
- Avoid redundant recomputation of equivalent subgroup states
- Preserve correctness of bound-based pruning while allowing controlled exploration

This design describes a frontier-based adaptive refinement search over Wordle answer subgroups.

This is to be integrated into the existing Wordle framework in the GitHub repo. It runs on iOS/iPadOS Pythonista and Linux console. The primary goal is to make the most effective use the limited computation available on mobile devices in a fixed amount of time to give a list of good "next guess" words for solving a Wordle puzzle. This design goal motivates the remainder of the design, including caching previous results, persisting word and pattern relationships for use on subsequent runs for future puzzles, minimal iterative refinement to advance the search frontier.

## Goals

### Search goals

The search should:

- begin with 1-step entropy information
- refine promising candidates incrementally
- compare candidates using lower and upper bounds on achievable multi-step entropy
- choose the next unit of work according to expected value to the global top-`N` result
- retain enough local information to avoid repeating work unnecessarily
- stop cleanly at any time and return the best current ranking with known bounds

### Persistent cache goals

The persistent cache should store reusable facts about subgroup states that remain valid across different Wordle problems.

The persistent cache should:

- be keyed by subgroup state, remaining depth, and candidate policy
- store summary knowledge only
- support incremental updates
- be safe under interruption
- avoid storing run-specific frontier state

### Runtime cache goals

The per-problem runtime cache should optimize one active solve.

The runtime cache should:

- intern subgroup states to eliminate redundant computation
- retain local rankings and frontier information needed for fast context switching
- support shared subproblems through a DAG structure
- remain correctness-independent if later bounded or evicted
- optimize compute time first; memory-management policy is secondary

## Assumptions

The design assumes:

- there is a stable answer universe with stable ordering
- the answer list is sorted alphabetically or otherwise loaded into a stable order
- subgroup state is defined by the remaining possible answers only
- candidate guesses may come from either the full guess list or the current survivor set, depending on policy
- the answer set tends to shrink sharply after each informative guess, making local 1-step entropy solves much cheaper than the root-level solve
- the main early bottleneck is computation time, not memory pressure

## Definitions

### Answer universe

Let `A` be the full set of allowed answer words.

Each answer word is assigned a stable integer index in `[0, |A|)`.

### Candidate guess universe

Let `G` be the set of allowed guess words under the current semantic policy.

Typical policies include:

- `ALL_GUESSES`: any allowed guess word may be used
- `SURVIVORS_ONLY`: only words still present in the current survivor set may be used

### Survivor subgroup

A survivor subgroup is a subset `S ⊆ A` of remaining possible answers.

A subgroup is represented in memory as a Python `int` bitset over the answer universe:

- bit `i` is set iff answer `A[i]` is still possible

On disk, the subgroup bitset is stored as a SQLite `BLOB`.

### Remaining depth

For a subgroup state, `d` is the remaining number of future guess levels whose deeper entropy is still being evaluated from that state.

### Wordle response partition

For a candidate guess `g` and subgroup `S`, the Wordle response function partitions `S` into child subgroups `{S_i}` according to the response pattern produced by comparing `g` against each answer in `S`.

Each child subgroup has probability weight:

`p_i = |S_i| / |S|`

### Immediate entropy

For a candidate guess `g` evaluated on subgroup `S`, immediate entropy is:

`H_immediate(g, S) = - Σ_i p_i log2(p_i)`

where the sum is over the response partition children `{S_i}`.

### State value

The value of a subgroup state is the **deeper entropy still achievable from that state**, not including any immediate entropy already accounted for by a parent candidate.

For state `(S, d, policy)`, let:

- `L(S, d)` be a lower bound on deeper achievable entropy
- `U(S, d)` be an upper bound on deeper achievable entropy

### Candidate value at a state

For candidate guess `g` evaluated at subgroup state `(S, d)`, with partition children `{S_i}` and weights `{p_i}`:

`L_g(S, d) = H_immediate(g, S) + Σ_i p_i * L(S_i, d - 1)`

`U_g(S, d) = H_immediate(g, S) + Σ_i p_i * U(S_i, d - 1)`

The subgroup state's own bounds are the best candidate bounds currently known at that state:

`L(S, d) = max_g L_g(S, d)` over candidates evaluated so far

`U(S, d) = max_g U_g(S, d)` over candidates evaluated so far

### Fresh-state initialization

For a newly created subgroup state with subgroup size `n = |S|` and remaining depth `d`:

- initial lower bound:

  `L(S, d) = 0`

- initial upper bound:

  `U(S, d) = d * log2(n)`

Special exact cases:

- if `n <= 1`, then `L = U = 0`
- if `d <= 0`, then `L = U = 0`
- if `n == 2` and `d >= 1`, then `L = U = 1.0`

### Top-`N` viability rule

A candidate or work item is globally non-viable when its upper bound falls below the current top-`N` cutoff.

Let `C_N` be the current `N`th-best lower bound among the top-level candidates.

Then a candidate or subtree is globally non-viable when:

`U < C_N`

Child refinement should stop when no dependent parent candidate can remain top-`N` even if that child were further refined.

This is the governing pruning rule.

## High-level algorithm

The algorithm is a **frontier-based adaptive refinement search** over subgroup states.

### Root behavior

At the current Wordle problem root:

1. compute 1-step entropy information for top-level candidates
2. create work items that can activate dormant top-level candidates
3. refine candidates incrementally under a bounded compute budget
4. return the best current ranking using known lower and upper bounds

### Search structure

The runtime search is a DAG of interned subgroup states, not a duplicated tree.

Equivalent subgroup states reached from different paths are represented once and shared.

This eliminates redundant computation and allows refinement of one subgroup to benefit multiple parents.

## Scheduler

A single unified scheduler manages all work.

There is no separate top-level scheduler and child scheduler.

### Work-item kinds

The scheduler supports at least:

- `ACTIVATE_TOP_WORD`
- `REFINE_CHILD`

### Unified scheduling principle

The scheduler should choose the next unit of work most likely to improve the current top-`N` answer quality.

It is mostly greedy, with a small amount of stochastic exploration.

### Exploration policy

A small exploration rate is used, on the order of a few percent.

When exploring stochastically, sampling is done from the full eligible set rather than only a narrow top band.

The exact percentage is tunable.

## `ACTIVATE_TOP_WORD`

`ACTIVATE_TOP_WORD` performs the minimum useful work required to turn a dormant first word into a real contender.

Given top-level guess `g` on current survivors `S_root`, activation does the following:

1. compute or retrieve the step-1 partition of `S_root` under `g`
2. compute `H_immediate(g, S_root)`
3. create the top-level candidate record for `g`
4. create or intern all immediate child states `(S_i, d - 1, policy)`
5. compute candidate bounds:

   `L_g = H_immediate(g, S_root)`

   `U_g = H_immediate(g, S_root) + Σ_i p_i * U(S_i, d - 1)`

6. enqueue downstream refinement work for unresolved nontrivial children
7. stop

Activation does not perform deeper recursive refinement beyond that first partition layer.

### Trivial child handling during activation

During activation:

- if `|S_i| == 0` or `|S_i| == 1`, exact deeper value is `0`
- if `|S_i| == 2` and remaining depth is at least `1`, exact deeper value is `1.0`
- only unresolved nontrivial children generate downstream refinement work

### Persistent-cache interaction during activation

Persistent cache can initialize child states with any previously known:

- lower bound
- upper bound
- best guess
- exact/partial flag

Persistent cache does not remove the need to compute the immediate step-1 partition of the current activated guess.

## `REFINE_CHILD`

`REFINE_CHILD` also performs the minimum useful step.

It refines one child state only if that state is still worth work under the current bound-based top-`N` logic.

### Bounds-gated refinement

Before refinement:

1. if the child state is already exact, stop
2. if the child state is globally non-viable for all dependent parents, stop

Only then does refinement proceed.

### Refinement action

If refinement is justified, `REFINE_CHILD` performs a **full local 1-step entropy solve** on the child subgroup.

This local breadth-1 solve is a bound-tightening operation.

It does the following:

1. evaluate the full local 1-step entropy ranking for the child subgroup
2. compute candidate-specific immediate entropies and child partitions
3. create or intern descendant subgroup states
4. tighten the child state's lower and upper bounds using the evaluated local candidates
5. mark dependent parents dirty
6. expose resulting descendant work to the scheduler according to global viability and priority
7. stop

`REFINE_CHILD` does not continue recursive descent within the same work item.

### Why a full local 1-step solve is used

Local child subgroups are much smaller than the root subgroup in typical use.

Therefore, a full local breadth-1 entropy solve is usually an appropriate minimum useful step:

- it is much cheaper than the root-level 1-step solve
- it provides principled local information rather than a weak proxy
- it improves bound quality substantially

## Local ranking retention

When a `REFINE_CHILD` operation computes a full local 1-step ranking at a state, the full local ranking is retained as state-local knowledge.

This ranking is not discarded simply because some candidates are not locally top-ranked.

### Local ranking is not local pruning

Shallow local ranking alone does not justify pruning.

A candidate that looks weak after one local breadth-1 pass may still become strong under deeper refinement.

Therefore:

- local rankings are retained
- local rankings do not by themselves prune candidates
- pruning is governed only by global bound viability

### Retained ranking vs active frontier

Retaining a full local ranking does not require enqueuing every descendant immediately.

Instead:

- the full local ranking remains attached to the state
- the scheduler decides which resulting work items become active frontier items

This preserves information while avoiding queue explosion.

### Scheduler access to local ranking

The scheduler should not be artificially deprived of local-ranking information.

However, it also should not perform repeated full scans of local rankings merely to choose the next work item.

Therefore, states should expose scheduler-ready work through an indexed or heap-friendly mechanism derived from the retained local ranking.

## Runtime data structures

The following data structures are expected.

### `StateNode`

A unique runtime state keyed by:

- `subset_bits`
- `remaining_depth`
- `policy`

Expected fields:

- `subset_bits: int`
- `remaining_depth: int`
- `policy: enum`
- `lower_bound: float`
- `upper_bound: float`
- `best_guess_id: int | None`
- `is_exact: bool`
- `candidate_records: mapping[guess_id -> CandidateRecord]`
- `dependents: set[(parent_state_key, parent_guess_id)]`
- `retained_local_ranking`: ranking or indexed structure derived from the latest local breadth-1 solve
- dirty / status flags as needed

### `CandidateRecord`

A candidate guess evaluated at one state.

Expected fields:

- `guess_id: int`
- `immediate_entropy: float`
- `lower_bound: float`
- `upper_bound: float`
- `is_exact: bool`
- `children: sequence[ChildEdge]`
- dirty / status flags as needed

### `ChildEdge`

A weighted relation from one candidate record to one child state.

Expected fields:

- `parent_guess_id: int`
- `child_state_key` or direct child state reference
- `path_weight: float`

Bound contribution is read from the child state itself rather than duplicated on the edge unless profiling later proves snapshots worthwhile.

### Scheduler work items

At minimum:

#### `ACTIVATE_TOP_WORD`
Expected fields:

- `guess_id`
- any precomputed root-level priority information needed by the scheduler

#### `REFINE_CHILD`
Expected fields:

- `parent_state_key`
- `parent_guess_id`
- `child_state_key`
- `path_weight`
- any cached priority metadata needed by the scheduler

## Runtime DAG and dependent propagation

Because the runtime search is a DAG, one child state may have multiple parents.

When a child state's bounds improve, dependent parent candidates should be marked dirty and recomputed lazily.

This allows shared subgroup refinement to benefit all relevant parents.

## Persistent cache

### Purpose

The persistent cache stores reusable facts about subgroup states that remain valid across different Wordle problems.

### Storage technology

Persistent cache uses SQLite.

Reasons:

- incremental updates
- safer under interruption
- easier inspection and maintenance
- natural fit for keyed subgroup-state storage

### Persistent key

The persistent cache key is:

- subgroup bitset
- remaining depth
- policy

### Persistent value

The persistent cache stores summary fields such as:

- `lower_bound`
- `upper_bound`
- `best_guess_id`
- `is_exact`

Persisted `best_guess_id` is treated as a **strong hint**, not an absolute command, unless the cached state is exact.

### SQLite representation

A single table is sufficient.

Representative fields:

- `subset_blob` BLOB
- `remaining_depth` INTEGER
- `policy` INTEGER
- `lower_bound` REAL
- `upper_bound` REAL
- `best_guess_id` INTEGER
- `is_exact` INTEGER

Primary key:

- `(subset_blob, remaining_depth, policy)`

## Runtime cache versus persistent cache

### Runtime cache

The runtime cache is per-problem and optimized for the current solve.

It may hold:

- interned states
- retained local rankings
- live frontier information
- dependency links
- scheduler acceleration structures

### Persistent cache

The persistent cache is cross-problem and stores only durable subgroup facts.

It does not store the live frontier.

## Memory management

The design prioritizes compute-time reduction over aggressive memory optimization.

Runtime memory eviction is not a primary design concern for the first implementation.

If needed later, runtime memory management should be:

- simple
- tunable
- correctness-independent

A later bounded in-memory policy such as LRU is acceptable, but the core design should not depend on eviction. Remember, the key is to make best use of time-limited and Python-constrained computation on a mobile device. Memory is likely available. Heavyweight computation may not be.

## Summary of core design choices

- subgroup state is represented over the answer universe only
- candidate guesses are governed separately by policy
- the search uses a DAG of interned subgroup states
- one unified scheduler manages mixed work-item types
- activation and refinement both perform the minimum useful step
- refinement uses full local 1-step entropy solves, gated by global bounds
- local rankings are retained, but shallow local ranking alone never prunes
- global pruning is governed by upper bounds versus the current top-`N` cutoff
- persistent cache stores reusable subgroup summaries in SQLite
- runtime cache stores richer per-problem frontier state


## Integration with the existing framework

This design is intended to be implemented **within the existing Python framework**, not as a greenfield rewrite.

### Existing architectural split

The current codebase already has a clear separation of responsibilities:

#### `wordle_engine.py`

This module owns:

- scoring primitives
- response computation
- restriction application and survivor filtering
- group partitioning
- reusable `ResponseCache`
- the `Solution` model
- current lookahead algorithms:
  - `compute_lookahead(...)`
  - `compute_deep_lookahead(...)`

This module should remain the home of the adaptive refinement frontier search implementation.

#### `wordle.py`

This module currently owns:

- platform adaptation and terminal / Pythonista behavior
- progress display and human-facing reporting
- command parsing and dispatch
- on-disk caching policy for current command outputs
- high-level orchestration of solving workflows
- interaction with `Solution`

This module should continue to own command-line / Pythonista integration, user prompts, status display, and human-facing orchestration.

It should **not** own engine persistence semantics or low-level cache management for the adaptive search.

### Design requirement: preserve the framework split

The adaptive refinement frontier search must fit into the current framework with minimal architectural disruption.

The design should preserve the following principles:

- no UI logic in `wordle_engine.py`
- no core search logic in `wordle.py`
- `Solution` remains the natural owner of problem-local search state
- human-facing progress and status reporting remain callback-driven from engine to UI
- persistent storage policy is orchestrated from the UI layer, while reusable cache data structures and search semantics remain engine-defined

The design could evolve to use additional Python files in some situations:
- It might make sense to have the persistent cache as a separate object, with SQLite interactions implemented behind an interface.
- The scheduler might be complex enough to warrant a separate file.
- The classes involved with frontier management might themselves be worth separating out into a file that the engine uses.

These additional Python files are not mandated. They are offered as options for possibly cleaner design.

## Existing engine structures that the design should reuse

The current engine already provides several pieces that should be reused rather than replaced.

### `Solution`

`Solution` already owns:

- `current_words`
- guess history
- cached score lists
- answer-set helpers
- application of guesses
- score computation against current survivors

The adaptive frontier search should extend `Solution` rather than bypass it.

In particular, the new search should treat:

- `Solution.current_words`

as the root survivor subgroup for the current problem.

### `ResponseCache`

`ResponseCache` already lazily caches:

- for each guess word
- the encoded Wordle response against every answer word

This cache is valuable and should remain central.

It already makes repeated group partitioning much cheaper by replacing repeated response computation with table lookup.

The adaptive search should continue using `ResponseCache` for:

- immediate partition construction
- local 1-step entropy solves
- child subgroup generation

But conversely, first-level entropy solves done by the adaptive algorithm ought to be cached in such a way that a later local 1-step solve can reuse immediately, saving computation.

### Current lookahead APIs

The existing engine already has two lookahead paths:

- `compute_lookahead(...)` for depth 2
- `compute_deep_lookahead(...)` for deeper depth with pruning and callbacks

The adaptive frontier search should be introduced as an additional engine-level lookahead path, not as a replacement of the engine/UI split. If the adaptive frontier turns out to be a good algorithm, both the naive compute_lookahead() and the compute_deep_lookahead() paths might be deleted in the future. Both conceptually ought to be able to be replaced by the adaptive algorithm with custom schedulers.

Recommended structure:

- introduce an engine-side class such as `AdaptiveFrontierSearch`
- keep `Solution` as the current puzzle-state holder
- optionally expose a thin convenience method on `Solution` that constructs and runs `AdaptiveFrontierSearch`

The existing `cmd_lookahead(...)` flow in `wordle.py` should become the natural entry point for invoking the new algorithm.

## Existing UI structures that the design should reuse

### Callback-based progress reporting

The current deep-lookahead path already supports:

- `progress_callback`
- `status_callback`
- periodic reporting via `report_interval`

This is the correct integration seam.

The adaptive frontier search should continue to report status through callbacks rather than printing directly.

The engine should expose machine-readable snapshots.
The UI layer should decide how to present them.

### Existing command flow

`cmd_lookahead(...)` in `wordle.py` already:

- checks that entropy scores exist
- chooses the candidate word source based on current mode
- prompts for maximum depth
- prints progress and results

This command should be adapted rather than replaced.

It should continue to:

- gather user options
- create progress/status callbacks
- invoke the engine
- format the final ranked results

### Existing solve/scoring context

The current code already distinguishes among:

- all guesses
- current word list
- solved words

The adaptive design's semantic policy should map cleanly onto existing concepts already present in the code:

- `InputSet.ALL_GUESSES`
- `InputSet.CURRENT_WORDLIST`

and engine-side equivalents such as:

- `ALL_GUESSES`
- `SURVIVORS_ONLY`

## Concrete adaptation requirements

### 1. Add, do not replace

The new algorithm should be added alongside the current scoring and lookahead code.

The design should not assume a full rewrite of:

- `Solution`
- `ResponseCache`
- command dispatch
- progress reporting infrastructure

### 2. Maintain callback-driven engine/UI separation

The engine should expose:

- search progress snapshots
- candidate ranking updates
- status summaries

through callbacks or structured return values.

The engine should not own terminal formatting, color, or display cadence decisions beyond report-interval triggering.

### 3. Fit persistent caching into the engine subsystem

The adaptive search introduces cache layers that are part of engine behavior, not user-facing command behavior.

The new persistent subgroup-state cache should therefore be treated as an **engine-side persistence concern**.

A clean model is:

- engine defines subgroup-state cache semantics, keys, and stored values
- engine owns the persistence adapter for SQLite subgroup-state storage
- UI code remains unaware of low-level persistence details except for top-level configuration, if any

This keeps cache management with the search subsystem that understands the meaning of the cached data.

The existing pickle-based command-result caches in `wordle.py` are separate conveniences and should not be used as the model for adaptive-search persistence.

### 4. Preserve current root-level workflow

The existing workflow is:

1. compute entropy scores
2. select top words
3. run lookahead

The adaptive frontier search should preserve this workflow shape.

It should still begin from the existing entropy solve and top-word selection process, because:

- those scores already exist in the current framework
- they determine the initial top-level activation candidates
- they fit naturally with the current `cmd_lookahead(...)` experience

### 5. Distinguish current command-result caches from subgroup-state caches

The current pickle cache in `wordle.py` stores complete command-result tables keyed by file timestamps and score method.

The adaptive design introduces a different kind of cache:

- reusable subgroup-state summaries across problems

These are distinct and should not be conflated.

Therefore:

- existing pickle caches may continue to exist for current score-table conveniences
- the new SQLite cache serves a different purpose and should be documented separately
- the design should make that distinction explicit

## Expected implementation placement

### Engine-side additions

Expected new or expanded engine-side components include:

- adaptive frontier search driver
- runtime DAG state structures
- state-node / candidate-record / child-edge logic
- scheduler work-item logic
- subgroup-state persistent-cache interface
- integration hooks on `Solution`

These belong in `wordle_engine.py` but could evolve into separate Python files if needed, as discussed above.

### UI-side additions

Expected UI-side changes include:

- adapting `cmd_lookahead(...)` to invoke the adaptive frontier search path
- presenting adaptive search progress snapshots periodically to keep the user informed
- presenting final bounded results
- configuring time budget, depth, and possibly exploration settings
- managing the SQLite cache file location and lifecycle, if this remains a UI responsibility

These belong in `wordle.py`.

## Expected compatibility behavior

The design should aim for compatibility with the current user workflow.

A user who already understands the current flow should still be able to:

- (optionally) run 1-step entropy solve
- ask for multi-level lookahead
- see progress
- receive a ranked answer table

The difference is that the deeper search implementation becomes adaptive and frontier-driven rather than sequentially deepening one candidate at a time.

## Existing-code-informed constraints

The following constraints arise directly from the structure of the current files.

### Root candidate ranking may be reused, but must not be required

The current `cmd_lookahead(...)` path depends on an existing entropy solve and uses `soln.scores[:count]`.

The adaptive design should be able to **reuse** that existing root entropy information when it is present and compatible, because reusing prior work is a primary requirement.

However, the adaptive search must not require a prior explicit `(s)olve` command.

Root-level entropy scoring is an internal phase of the adaptive search and should be computed on demand when needed.

### Response partition caching already exists

Because `ResponseCache.group_words(...)` and `ResponseCache.group_counts(...)` already exist, the design should treat them as the first-line partitioning mechanism.

This reduces implementation risk and aligns with the existing optimization strategy.

### Progress reporting already has a machine/UI seam

Because the deep lookahead path already uses `status_callback(info)` with structured dictionaries, the adaptive design should follow the same pattern for status snapshots.

This keeps `wordle.py` in control of presentation while allowing richer adaptive-search instrumentation.

### Full-result persistence already exists separately

Because `wordle.py` already persists root score tables with pickle, the adaptive subgroup-state SQLite cache must be described and implemented as a separate cache layer with a different purpose.

## Recommended adaptation to the design vocabulary

Within this existing framework, the adaptive refinement frontier search should be framed as:

- an engine-side search subsystem operating on a `Solution`
- an engine-level replacement for the current deep recursive lookahead algorithm
- invokable from the existing lookahead command path
- built on top of existing `Solution` and `ResponseCache` infrastructure
- integrated with the current callback/reporting structure
- capable of reusing root entropy work whether it was computed by a prior command or by the adaptive search itself

That framing is more accurate than describing it as a standalone solver or as a UI-driven workflow.


## Reuse requirements

Efficient reuse is a primary design goal.

### Reuse within a single active Wordle problem

While solving one current puzzle, the implementation should reuse as much work as possible, including:

- root-level entropy computations
- local 1-step entropy solves on subgroup states
- subgroup partitions obtained through `ResponseCache`
- retained local rankings
- interned DAG subgroup states
- partial bounds and best-known guesses
- scheduler-ready frontier information

A prior root entropy solve and a later adaptive search should share compatible data rather than recomputing it.

### Reuse across different Wordle problems

Across separate puzzles, the implementation should persist reusable subgroup-state knowledge, including summary information such as:

- subgroup-state bounds
- best known guess
- exact/partial status

The design should also allow future persistence of reusable subgroup partition knowledge if that proves worthwhile.

The implementation should avoid conflating:

- per-problem live frontier state
- reusable subgroup-state summaries
- command-result convenience caches

## Replacement of the current deep lookahead

The current deep recursive lookahead path is superseded by the adaptive refinement frontier search.

The existing `compute_deep_lookahead(...)` algorithm may be deleted or retired once the adaptive search is implemented and validated.

The depth-2 path may remain temporarily during transition if useful, but the intended long-term direction is one adaptive multi-level engine rather than parallel deep-lookahead systems.

## Engine-side subsystem structure

The preferred engine structure is:

- `Solution` as the owner of the current puzzle state
- `AdaptiveFrontierSearch` as the engine-side search subsystem
- a separate engine-side persistence adapter for subgroup-state SQLite storage

This means:

- `Solution` should not absorb all scheduler, DAG, and persistence machinery directly
- `AdaptiveFrontierSearch` should own runtime DAG state, scheduler logic, and refinement workflow
- persistence should be separated from both UI code and the scheduler core

## Persistence placement

Low-level cache file management for adaptive-search persistence belongs with the persistence adapter in the engine subsystem, not in `wordle.py`.

Reasoning:

- the user-facing layer should not need to understand subgroup-state cache semantics
- cache keys and values are defined by engine concepts
- the persistence adapter is part of the search subsystem's infrastructure
- keeping this logic out of `wordle.py` preserves the UI/engine separation

`wordle.py` may still provide high-level configuration knobs if needed, but should not own the operational details of subgroup-state cache storage.

## Scheduler implementation note

Scheduler priority entries may become stale as bounds tighten and states are refined.

Therefore the design should explicitly allow:

- lazy invalidation
- dirty flags
- recomputation or validation when entries are popped from the heap

The scheduler should not assume that all stored priorities remain exact over time.

## Incremental implementation plan

Implementation should proceed incrementally rather than in one large rewrite.

A recommended sequence is:

1. introduce the engine-side `AdaptiveFrontierSearch` class with no UI disruption
2. make root entropy scoring an internal phase of adaptive search, while reusing existing compatible entropy results when present
3. replace the current deep-lookahead command path with the new engine-side search
4. preserve existing callback-based reporting and adapt it to richer adaptive-search snapshots
5. add the persistent subgroup-state SQLite adapter
6. add retained local rankings and scheduler acceleration structures
7. tune exploration, pruning, and optional runtime memory controls after behavior is validated

This sequence reduces risk and makes it easier to test correctness and performance step by step.
