# Wordle Solver — Design Notes

## Overview

An interactive Wordle assistant implemented in Python, designed to run in
Pythonista on iOS (iPhone/iPad) and on standard Linux terminals. It helps
with a live Wordle game by tracking remaining possible answers and
recommending guesses, and also supports multi-board variants
(Quordle, Dordle, etc.).

Five layers:

| File | Role |
|------|------|
| `wordle_engine.py` | All algorithms: scoring, ERD search, response simulation. No I/O. |
| `pattern_matrix.py` | NumPy pattern matrix + vectorized candidate statistics. Sole (guarded) NumPy import point; optional at runtime. |
| `wordle.py` | REPL, display, all command handlers. |
| `cache_sqlite.py` | SQLite-backed `ScoreCache`: ERD results, candidate scores, response decompositions. |
| `erd_swarm.py` / `erd_queue.py` / `erd_search.py` | Parallel precache workers: branch assignment, candidate claiming, cooperative ERD solving. |

Support files: `merge_cache.py` (merge two `.sqlite3` files), `backfill_max_depth.py`
(populate `max_depth` for legacy rows), diagnostic scripts (`diag_*.py`).

---

## Vocabulary

These terms have precise meanings throughout the codebase:

| Term | Meaning |
|---|---|
| **guess** | A word actually played as a turn in the game. |
| **candidate** | A word under evaluation during search — not yet played. A candidate becomes the guess when it wins. |
| **branch** | The remaining answer words after a guess + response. Identified by a (guess, pattern) pair at each level. |

The phase boundary between candidate and guess is explicit:
```python
for i, candidate in enumerate(candidate_list):
    status, cost, md, floor = evaluate_candidate(branch_words, candidate, ...)
    if cost < best_erd:
        best_guess = candidate   # ← candidate becomes the guess here
```

### Depth and budget

Everything on the guess axis is **remaining depth** — guesses still needed to
solve, measured per answer under optimal play — viewed four ways. Naming a bare
`depth` is forbidden; always say which one.

| Term | Meaning |
|---|---|
| **`guess_depth`** | Guesses already *played* to reach a branch — the number of guesses on its spine. The root (no guess) is `guess_depth 0`; a one-guess seed is `guess_depth 1`. The single source of truth. |
| **`budget`** | The *allowed* remaining depth (the cap). The only quantity the ERD recurrence reads. Each recursive level spends one. |
| **ERD** | *Expected* remaining depth — the probability-weighted mean line length. The objective the solver minimizes (`cost` / `best_erd`). |
| **`max_remaining_depth`** | *Maximum* remaining depth — the worst-case line length under optimal play. The feasibility gate (a branch is solvable iff `max_remaining_depth ≤ budget`) and the cache-reuse key (a result is reusable at any `budget ≥ max_remaining_depth`). |

`GAME_GUESSES = 6` is the root's budget — zero guesses played. The invariant
`budget + guess_depth = GAME_GUESSES` holds at every node, because each recursive
level spends one budget and adds one guess; so `guess_depth = GAME_GUESSES − budget`.

ERD and `max_remaining_depth` are the **mean** and the **max** of the same
per-answer remaining-depth distribution — the same recurrence, two reductions:

```python
cost    += (k / n) * sub_cost          # weighted MEAN over sub-branches → ERD
cand_md  = max(cand_md, 1 + sub_md)    # MAX over sub-branches           → max_remaining_depth
```

ERD is what we optimize; `max_remaining_depth` is the hard 6-guess constraint
the mean can't see (a low average with one 7-long line still loses).

There is **no** "promotion depth" or "recursion depth." The number of
cooperative spawns and the engine's recursion level were both `depth` names
that carried no information `budget`/`guess_depth` don't — they were removed.
Whether a branch is human-queued or swarm-spawned is answered by `pending_branches`
membership, not by any depth.

---

## Word Lists

Two files at startup:

- **`NYT_wordlist.txt`** — ~3,200 answer words (the set the NYT Wordle draws from).
- **`wordle.txt`** — ~12,972 valid guess words (superset including answers).

Answers are the universe for "what is the answer?"; the full word list is the
universe for "what should I guess next?" The two are kept separate throughout.
All scoring is done by partitioning the *answer set* into response groups; the
candidate being evaluated can come from either list.

---

## Response Encoding

A response is a list of five colors: `'green'`, `'yellow'`, or `'gray'`.

```
calculate_response(test_word, answer_word) -> ['green', 'gray', 'yellow', ...]
```

Two-pass algorithm handles duplicate letters correctly:
- Pass 1: mark greens and consume those answer positions.
- Pass 2: mark yellows (present but wrong position) and grays.

For computation, responses are encoded as integers 0–242 (base 3,
gray=0 / yellow=1 / green=2, most-significant digit first). All
internal group-counting uses these ints as dict keys. `decode_response`
reverses the encoding for display.

---

## Scoring Methods

The core operation: partition the remaining answer words (`branch_words`) by
what response they would give to a candidate guess. Each partition bucket is a
*response group*. Four scoring methods evaluate how informative the partition is:

| Enum | SQLite key | Formula | Direction |
|---|---|---|---|
| `WEIGHTED_AVG` | `weighted_avg` | Σ(nᵢ²) / N | lower is better |
| `ENTROPY_GAIN` | `entropy_gain` | −Σ(pᵢ log₂ pᵢ) | higher is better |
| `MAX_GROUP_SIZE` | `max_group_size` | max(nᵢ) | lower is better |
| `PROB_FINISH` | `prob_finish` | (# groups of size 1) / N | higher is better |

`ENTROPY_GAIN` (Shannon entropy in bits) is the primary method for interactive
scoring and lookahead. `MAX_GROUP_SIZE` (worst-case group size) is useful for
risk analysis and is paired with entropy in the board (Pareto) view.

`max_entropy(n)` returns log₂(n) — the theoretical ceiling for n remaining
words. In practice, with only 243 possible responses (3⁵), no word can
achieve this ceiling unless very few words remain.

---

## ERD Search

**Expected Remaining Depth (ERD)** is the minimum expected number of guesses
to solve a branch, playing optimally. It replaces the older two-step entropy
lookahead as the primary search algorithm.

```
min_expected_guesses(branch_words, cache, score_cache, ...) -> float | None
```

The search is depth-limited: the budget is the number of guesses remaining in
the game (5, after the opener). A branch that cannot be solved within the
budget is a loss; its cost is `inf`.

### evaluate_candidate

```
evaluate_candidate(branch_words, candidate, cache, score_cache, *,
                   best_erd, budget, ...) -> (status, cost, max_depth, floor_hit)
```

Evaluates one candidate's exact ERD for solving `branch_words`. Returns one of:
- `('ok', cost, md, floor)` — fully evaluated; `cost < best_erd`
- `('pruned', None, md, floor)` — can't beat `best_erd` or infeasible within budget
- `('cutoff', None, None, floor)` — provably can't beat the bound (admissible lower bound >= best_erd)
- `('useless', None, None, floor)` — a response group is all of `branch_words`
- `('abort', None, None, floor)` — deadline/cancel fired

**Branch-and-bound:** a running `best_erd` is shared across all candidate evaluations
for the same branch. A candidate that provably can't beat `best_erd` is pruned immediately.

**Alpha-beta per sub-branch:** within one candidate's evaluation, each sub-branch is
solved under a derived ceiling so deep nodes prune from tight values rather than `inf`.
`rest_lb[i]` is an admissible lower bound on the weighted cost of all sub-branches after
position `i` (each sub-branch of k answers costs ≥ 2 − 1/k).

### Candidate ordering

Before the main search loop, `rank_candidates_by_max_group_size_then_entropy_gain`
sorts candidates by ascending `MAX_GROUP_SIZE`, breaking ties by descending
`ENTROPY_GAIN`. This front-loads strong candidates so `best_erd` tightens early,
making subsequent pruning more aggressive.

### ERD policies

The cache namespace identifies which universe+compliance combination was used:

| Policy constant | Meaning |
|---|---|
| `ERD_ALL` | ~12,972 words, no clue filter — the main precache target |
| `ERD_ANSWERS` | ~3,200 answer words only, compliant with revealed clues |
| `ERD_CONSTRAINED` | ~12,972 words, must satisfy all clues (Wordle hard mode; transient, never persisted) |
| `ERD_ANSWERS_UNFILTERED` | ~3,200 words, no clue filter |

---

## ResponseCache

```python
class ResponseCache:
    def __init__(self, answer_words, score_cache=None): ...
    def group_counts(self, guess, subset) -> {pattern_int: count}
    def group_words(self, guess, subset) -> {pattern_int: [words]}
```

For each guess word encountered, caches the full `{answer → pattern_int}` mapping
across all answer words. Built lazily on first access. Subsequent calls against any
subset of answers are then pure dict lookups — no `calculate_response` calls needed.

If a `ScoreCache` is provided at construction, it is consulted first for a
precomputed decomposition blob, avoiding even the initial scan.

One `ResponseCache` instance is shared across all `Solution` objects for the session.

---

## ScoreCache (SQLite)

Defined in `cache_sqlite.py`. Shared between Linux and the iOS app via iCloud sync.

```python
class ScoreCache:
    def read(self, branch_key, policy) -> (best_guess, best_score) | None
    def read_with_depth(self, branch_key, policy) -> (best_guess, best_score, max_depth, solve_budget) | None
    def write(self, branch_key, policy, best_guess, best_score, max_depth, solve_budget)
    def has_scores(self, branch_key, method) -> bool
    def read_scores(self, branch_key, method) -> [(word, score)] | None
    def write_scores(self, branch_key, scores, method)
    @staticmethod
    def encode_subset(words) -> bytes  # sorted words concatenated, 5 bytes each
```

### Schema

```sql
CREATE TABLE answer_list (
    answer_list_id TEXT PRIMARY KEY,   -- SHA-256 of sorted newline-joined answer words
    answer_hash    TEXT NOT NULL,
    answer_count   INTEGER NOT NULL,
    created_at     INTEGER NOT NULL
)

CREATE TABLE branch_best_by_policy (
    branch_key     BLOB NOT NULL,      -- encode_subset(branch_words)
    policy         TEXT NOT NULL,      -- ERD policy string
    answer_list_id TEXT NOT NULL,
    best_guess     TEXT NOT NULL,
    best_score     REAL NOT NULL,      -- policy-dependent: ERD cost or entropy
    max_depth      INTEGER,            -- worst-case line length; NULL = unknown
    solve_budget   INTEGER,            -- budget under which result was computed; NULL = unconstrained
    updated_at     INTEGER NOT NULL,
    PRIMARY KEY (branch_key, policy, answer_list_id)
)

CREATE TABLE candidate_scores (
    branch_key     BLOB NOT NULL,
    answer_list_id TEXT NOT NULL,
    method         TEXT NOT NULL,      -- scoring method key (e.g. 'entropy_gain')
    scores_blob    BLOB NOT NULL,      -- packed array of (word, score) pairs
    updated_at     INTEGER NOT NULL,
    PRIMARY KEY (branch_key, answer_list_id, method)
)

CREATE TABLE response_decomposition (
    guess          TEXT NOT NULL,
    answer_list_id TEXT NOT NULL,
    patterns       BLOB NOT NULL,      -- packed {answer -> pattern_int} map
    updated_at     INTEGER NOT NULL,
    PRIMARY KEY (guess, answer_list_id)
)
```

### Cache reuse contract

A `branch_best_by_policy` row is reusable if:
- `solve_budget IS NULL` — result is the unconstrained ERD optimum, valid at any budget
- `solve_budget IS NOT NULL AND max_depth <= remaining_budget` — a depth-limited result is valid
  only when the remaining budget is at least as large as the worst-case depth it was computed under

Rows with `max_depth IS NULL` (legacy, pre-depth-tracking) are treated as unreusable for
depth-limited queries and recomputed.

### Schema coordination (Linux + phone)

The cache file is synced via iCloud between Linux and the iOS Pythonista app. Any schema change must:
1. Be implemented as an idempotent migration in `ScoreCache._ensure_schema`, guarded by `schema_migrations`
2. Deploy new code to the phone **before** syncing a migrated Linux database to it
3. Never require manual SQL — migrations run automatically on first open

---

## Solution Class

`Solution` tracks the state of a single board:

```python
class Solution:
    current_words    # remaining candidate answers
    guesses          # [(word, response), ...]
    answer_word      # set in simulation mode; None otherwise
    fallback_active  # True if current_words came from full guess vocab
```

### Undo

`undo_guess()` pops the last entry from `self.guesses`, resets `current_words`
to the full answer list, and replays all remaining guesses. Works in all modes
(live game, simulation, multi-game).

### Hard Mode Words

`hard_mode_words(all_words)` filters the full ~12K word list to words consistent
with all prior responses — green letters fixed in position, yellow letters present
but not in that position. This is real Wordle hard mode: guesses must respect all
revealed constraints, but are not limited to remaining answers.

### Multi-Game Join

`Solution.join(solutions)` merges the `current_words` from multiple active boards
into a single Solution for computing a shared best guess (useful for Quordle-style play).

### Fallback

If `apply_guess` would leave `current_words` empty (the guessed word is not in the
answer list and its response pattern matches nothing), the engine replays all guesses
against the full ~12K word vocabulary. `fallback_active` is set to True; the UI
displays a warning.

---

## Parallel ERD Precache (Swarm)

The precache fills `branch_best_by_policy` for `ERD_ALL` across all branches
reachable from the opener (typically SALET). Because evaluating ~12,972 candidates
against a branch is slow, multiple workers cooperate:

### Architecture

- `erd_queue.sqlite3` — coordination-only database (separate from `wordle_cache.sqlite3`
  to avoid contention). Contains the `pending_subgroups` table of branches to solve,
  candidate claims, heartbeats, and done flags.
- `_BranchWorker` (`erd_swarm.py`) — one per OS process. Claims one candidate at a time,
  evaluates it, writes sub-branch results to `wordle_cache.sqlite3`, and updates claim
  state in `erd_queue.sqlite3`.
- `ERDQueue` (`erd_queue.py`) — single writer to `erd_queue.sqlite3`. Used by workers to
  claim candidates, record heartbeats, mark claims done, and promote large sub-branches to the queue.

### Branch lifecycle

1. A branch is added to the queue with one candidate slot per candidate word.
2. Each worker calls `claim_one()` to atomically claim an unclaimed (or timed-out) candidate.
3. The worker evaluates the candidate, writing sub-branch ERD results as it goes.
4. On `done=1`, the worker calls `maybe_finalize`: if every candidate for this branch is done, it
   writes the final `branch_best_by_policy` row and removes the branch from the queue.

### Trust model

A claim is advisory; only a `done=1` claim is authoritative. A branch is finalized only
once every candidate is done. A crashed worker's claim times out (`HB_TIMEOUT_SECONDS = 120`) and is
reclaimed — never skipped, never double-counted.

### Sub-branch promotion

When `evaluate_candidate` recurses into a sub-branch with ≥ 60 words, the worker promotes that
sub-branch to the queue at elevated priority (`PROMOTED_PRIORITY = 1,000,000`) so freed workers
prefer joining in-flight depth over starting fresh top-level branches.

### Budget

`ROOT_BUDGET = GAME_GUESSES` (= 6, the whole game). Each queued branch is solved at
`ROOT_BUDGET − guess_depth`, where `guess_depth` counts the guesses already played on the
branch's spine — so a branch queued after the opener (`guess_depth` 1) is solved at budget 5.
A branch unsolvable within its budget gets `cost = inf` — not a finite expected depth.

### Pattern matrix (vectorized kernel)

`pattern_matrix.py` owns one uint8 matrix — `matrix[g, a]` is the encoded response of guess
word g against answer word a, exactly `ResponseCache`'s decomposition blobs stacked — persisted
as `.npy` beside the cache and mmap-shared across worker processes. From one vectorized pass,
`candidate_stats()` derives every candidate's response-group count, admissible cost lower
bound, Σk² sort key, max group size, and entropy gain against a branch.

Engine paths that consult it (via the optional `pattern_matrix` parameter threaded down from
`min_expected_guesses`): the best-first candidate sort in `_solve_subset`, and the
ERD-lower-bound pruning check in its candidate loop — skipping `evaluate_candidate` for any
candidate whose admissible bound already meets `best_erd`, decision-identical to the check
`evaluate_candidate` itself applies. Results are bit-identical to the pure-Python path by
construction and by test (`test_kernel_equivalence.py`).

NumPy is a hard requirement on every deployment target. The pure-Python implementations stay
permanently — not as a runtime fallback but as the reference implementation the vectorized
path is tested against; they are prohibited from calling NumPy, and selecting them is a
caller choice (`pattern_matrix=None`). The phone (Pythonista bundles NumPy 1.22.3) sets the
API floor — nothing newer than 1.22.

---

## Two-Step Entropy Lookahead

The interactive `l` (lookahead) command uses a lighter-weight two-step search: for each
candidate first guess, compute the weighted average best-entropy second guess across all
response groups. This is faster than ERD for interactive use and works well for identifying
strong openers without needing the full precache.

```python
soln.compute_lookahead(
    top_words,           # [(word, first_entropy), ...]
    second_step_words,   # None = hard mode; list = full mode
    total_callback,
    progress_callback,
) -> [(word, step1, step2, combined), ...]
```

Groups of size 1 are already solved; groups of size 2 contribute exactly 1.0 bit.

---

## GuessUniverse and ComplianceFilter

```python
class GuessUniverse(Enum):
    ALL_WORDS    = 'words'    # ~12,972
    ALL_ANSWERS  = 'answers'  # ~3,200

class ComplianceFilter(Enum):
    UNFILTERED = 'unfiltered'  # any word from the universe
    COMPLIANT  = 'compliant'   # must satisfy all clues revealed so far
```

The interactive `h` command cycles through effective modes:
`ALL_WORDS/UNFILTERED → ALL_WORDS/COMPLIANT → ALL_ANSWERS/COMPLIANT → ALL_WORDS/UNFILTERED`

---

## Interactive Commands

| Key | Handler | Description |
|-----|---------|-------------|
| `g` | `cmd_guess` | Enter a guess word and response pattern |
| `s` | `cmd_solve` | Score all candidates by chosen method |
| `b` | `cmd_grid` | Pareto entropy vs max group size display |
| `l` | `cmd_lookahead` | Two-step entropy lookahead on top N words |
| `d` | `cmd_display` | Show remaining words (scored if available) |
| `t` | `cmd_test` | Analyse a specific word (all methods + lookahead) |
| `i` | `cmd_include` | Keep only words containing specified letters |
| `x` | `cmd_exclude` | Remove words containing specified letters |
| `u` | `cmd_undo` | Remove the last guess and restore prior word count |
| `v` | `cmd_verify_erd` | Spot-check cached ERD entries against their subtrees |
| `p` | `cmd_precache` | Show precache progress or trigger a focused branch solve |
| `r` | `cmd_reset` | Reset one or all boards |
| `a` | `cmd_answer` | Set a known answer for simulation mode |
| `w` | `cmd_wordcount` | Set number of games (Quordle, Dordle, etc.) |
| `c` | `cmd_candidates` | Show candidate pool size and active filters |
| `h` | Hard mode | Cycle through guess universe / compliance filter |
| `?` | `cmd_help` | Print command summary |

### Response entry format (`g`)

The response to a guess is entered as a 5-character string: `g` = green, `y` = yellow,
any other character (including `-`, `0`, `_`, em-dash) = gray. The parser accepts
punctuation liberally because mobile keyboards sometimes substitute punctuation for hyphens.

### Board (`b`) — Pareto View

Scores all candidates simultaneously on entropy and max group size, then displays the
**Pareto frontier**: the max-group-size levels where no other level has both higher entropy
and lower max group size.

**Algorithm:**

1. For each unique max group size, find the best entropy among all candidates at that size.
2. Sort sizes ascending. Walk the list keeping a running maximum entropy seen so far.
   A size is on the frontier if and only if its best entropy exceeds the running maximum.
3. For each frontier level, show the top 4 words by entropy.

Words in the answer set are marked `*`. Each frontier level shows how many total words share
that max group size.

---

## Platform Notes

### Color Output

On Pythonista, `console.set_color(r, g, b)` is used. On Linux, ANSI escape codes are used
when stdout is a TTY and `NO_COLOR` is not set.

### Display Width

`get_display_width()` tries the following in order:

1. `shutil.get_terminal_size()` — queries the OS via `TIOCGWINSZ`.
2. Pythonista pixel-based estimation: reads the console view width in points via
   `console.get_size()`, then tries several monospace fonts via `ui.measure_string`
   and picks the one whose column count is closest to an integer.
3. Falls back to 80 columns.

### Progress Bar

`ProgressTracker` prints a width-aware progress indicator during long computations:
- One `.` per percentage point completed.
- `25%` / `50%` / `75%` / `100%` milestone labels.
- An ETA label (e.g. `1m30s`) every 10 seconds.
- Lines wrap at `DISPLAY_WIDTH - 6` columns.
