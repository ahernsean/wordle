# Wordle Solver — Design Notes

## Overview

An interactive Wordle assistant implemented in Python, designed to run in
Pythonista on iOS (iPhone/iPad) and on standard Linux terminals. It helps
with a live Wordle game by tracking remaining possible answers and
recommending guesses, and also supports multi-board variants
(Quordle, Dordle, etc.).

Three files hold essentially everything:

| File | Role |
|------|------|
| `wordle_engine.py` | All algorithms. No I/O. |
| `wordle.py` | REPL, display, command handlers. |
| `adaptive_cache_sqlite.py` | SQLite-backed lookahead result cache. |

---

## Word Lists

Two files at startup:

- **`NYT_wordlist.txt`** — ~3,200 answer words (the set the NYT Wordle draws from).
- **`wordle.txt`** — ~12,972 valid guess words (superset including answers).

Answers are the universe for "what is the answer?"; guesses are the
universe for "what should I guess next?" The two are kept separate
throughout. All scoring is done by partitioning the *answer set* into
response groups; the guess being evaluated can come from either list.

---

## Response Encoding

A response is a list of five colors: `'green'`, `'yellow'`, or `'gray'`.

```
calculate_response(guess, answer) -> ['green', 'gray', 'yellow', ...]
```

Two-pass algorithm handles duplicate letters correctly:
- Pass 1: mark greens and consume those answer positions.
- Pass 2: mark yellows (present but wrong position) and grays.

For computation, responses are encoded as integers 0–242 (base 3,
gray=0 / yellow=1 / green=2, most-significant digit first). All
internal group-counting uses these ints as dict keys. `decode_response`
reverses the encoding for display.

---

## Group Analysis and Scoring

The core operation: partition the remaining answer words by what response
they would give to a candidate guess. Each partition bucket is a
*response group*. The guess that creates the most informative partition is
the best guess.

Four scoring methods:

| Method | Formula | Direction |
|--------|---------|-----------|
| `WEIGHTED_AVG` | Σ(nᵢ²) / N | lower is better |
| `ENTROPY_GAIN` | −Σ(pᵢ log₂ pᵢ) | higher is better |
| `MINIMAX` | max(nᵢ) | lower is better |
| `PROB_FINISH` | (# groups of size 1) / N | higher is better |

`ENTROPY_GAIN` (Shannon entropy in bits) is the primary method used by
lookahead and most automated ranking. `MINIMAX` is useful for worst-case
risk analysis. `PROB_FINISH` is the probability that a single guess solves
the puzzle outright.

`max_entropy(n)` returns log₂(n) — the theoretical ceiling for n remaining
words. In practice, with only 243 possible responses (3⁵), no word can
achieve this ceiling unless very few words remain.

---

## ResponseCache (in-memory)

```python
class ResponseCache:
    def __init__(self, answer_words): ...
    def group_counts(self, guess, subset) -> {pattern_int: count}
    def group_words(self, guess, subset) -> {pattern_int: [words]}
```

For each guess word encountered, caches the full `{answer → pattern_int}`
mapping across all answer words. Built lazily on first access. Subsequent
calls against any subset of answers are then pure dict lookups —
no `calculate_response` calls needed.

One `ResponseCache` instance is shared across all `Solution` objects for
the session.

---

## Solution Class

`Solution` tracks the state of a single board:

```python
class Solution:
    current_words   # remaining candidate answers
    guesses         # [(word, response), ...]
    word_scores     # {word: {ScoringMethod: score}} — per-word score cache
    answer_word     # set in simulation mode; None otherwise
    fallback_active # True if current_words came from full guess vocab
```

### Score Cache

`word_scores` is a per-word, per-method cache. `compute_scores` and
`compute_scores_multi` populate it incrementally — a word already scored
under a method is not recomputed. This means the board command (which
scores entropy + minimax together) and the solve command (entropy only)
share computed entropy values with no redundant work.

The cache is cleared whenever `current_words` changes: after `apply_guess`,
`undo_guess`, `include_letters`, or `exclude_letters`.

### Undo

`undo_guess()` pops the last entry from `self.guesses`, resets
`current_words` to the full answer list, and replays all remaining guesses.
Works in all modes (live game, simulation, multi-game).

### Hard Mode Words

`hard_mode_words(all_guesses)` filters the full 12K guess list to words
consistent with all prior responses — green letters fixed in position,
yellow letters present but not in that position. This is real Wordle hard
mode: guesses must respect all revealed constraints, but are not limited
to remaining answers.

### Multi-Game Join

`Solution.join(solutions)` merges the `current_words` from multiple
active boards into a single Solution for computing a shared best guess
(useful for Quordle-style play).

### Fallback

If `apply_guess` would leave `current_words` empty (the guessed word is not
in the answer list and its response pattern matches nothing), the engine
replays all guesses against the full 12K guess vocabulary. `fallback_active`
is set to True; the UI displays a warning.

---

## Two-Step Entropy Lookahead

```python
soln.compute_lookahead(
    top_words,           # [(word, first_entropy), ...]
    second_step_words,   # None = hard mode; list = full mode
    total_callback,      # called once with total work units
    progress_callback,   # called per work unit
) -> [(word, step1, step2, combined), ...]
```

**Algorithm:**

For each candidate first guess:
1. Partition `current_words` into response groups.
2. For each group of size > 2, find the best second guess: the word in
   `second_step_words` (or the subgroup itself in hard mode) that maximises
   entropy against that subgroup.
3. Compute the weighted average second-step entropy:
   `step2 = Σ (|group| / N) × best_entropy(group)`
4. `combined = step1 + step2`

Groups of size 1 are already solved; groups of size 2 contribute exactly
1.0 bit (any distinguishing word resolves them).

**Hard mode vs full mode:**

- **Hard mode** (`second_step_words=None`): the second guess must come from
  the subgroup itself. Fast; realistic for hard-mode play.
- **Full mode** (`second_step_words=[...]`): search a provided word list
  (typically the top N² ranked guesses) for the best second step. Slower;
  finds better second steps.

The test command uses hard mode. The lookahead command uses full mode,
searching the top N² candidates as second-step words.

**SQLite subgroup cache:**

Completed subgroup results are cached in `LookaheadCache` (see below).
Before scanning candidates for a subgroup, the cache is checked. On a hit,
the scan is skipped entirely. Cache hits are excluded from the work-unit
count so the progress bar reflects only real computation.

---

## LookaheadCache (SQLite)

Defined in `adaptive_cache_sqlite.py`.

```python
class LookaheadCache:
    def read(self, subset_blob, policy) -> (best_word, best_entropy) | None
    def write(self, subset_blob, policy, best_word, best_entropy)
    def stats() -> (row_count, last_updated_ts)
    @staticmethod
    def encode_subset(words) -> bytes  # b"\0".join(sorted(words).encode())
```

**Schema:**

```sql
CREATE TABLE universe (
    universe_id  TEXT PRIMARY KEY,  -- SHA-256 of sorted answer list
    answer_hash  TEXT NOT NULL,
    answer_count INTEGER NOT NULL,
    created_at   INTEGER NOT NULL
)

CREATE TABLE lookahead_result (
    subset_blob  BLOB    NOT NULL,  -- NUL-joined sorted subgroup words
    policy       TEXT    NOT NULL,  -- 'hard' or 'full'
    universe_id  TEXT    NOT NULL,
    best_word    TEXT    NOT NULL,
    best_entropy REAL    NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (subset_blob, policy, universe_id)
)
```

`universe_id` is the SHA-256 of the newline-joined sorted answer list.
Results from a different answer set are silently ignored via the universe
join.

The cache is **exact**: only completed subgroup scans are stored. No
partial or approximate values are written. Because small subgroups (≤ ~20
words) recur frequently across sessions regardless of which first guess
produced them, the cache grows incrementally and makes repeated lookahead
runs progressively faster.

---

## InputSet and Hard Mode Toggle

The `InputSet` enum controls which words are eligible as guesses:

| Value | Candidate pool |
|-------|---------------|
| `ALL_GUESSES` | All ~12,972 words |
| `HARD_MODE` | Words from all_guesses satisfying all revealed constraints |
| `CURRENT_WORDLIST` | Remaining possible answers only (strictest) |
| `SOLVED_WORDS` | Multi-game only: the solved answer from each board |

The `h` command cycles `ALL_GUESSES → HARD_MODE → CURRENT_WORDLIST → ALL_GUESSES`.
The current setting applies to the `s`, `b`, and `l` commands.

---

## Interactive Commands

The REPL loop reads a single character per command.

| Key | Command | Description |
|-----|---------|-------------|
| `g` | Guess | Enter a guess word and response pattern |
| `s` | Solve | Score all candidates by chosen method |
| `b` | Board | Pareto entropy vs max-group-size display |
| `l` | Lookahead | Two-step entropy lookahead on top N words |
| `d` | Display | Show remaining words (scored if available) |
| `t` | Test | Analyse a specific word (all methods + lookahead) |
| `i` | Include | Keep only words containing specified letters |
| `x` | Exclude | Remove words containing specified letters |
| `u` | Undo | Remove the last guess and restore prior word count |
| `r` | Reset | Reset one or all boards |
| `a` | Answer | Set a known answer for simulation mode |
| `w` | Words | Set number of games (Quordle, Dordle, etc.) |
| `h` | Hard mode | Cycle through input sets |
| `c` | Cache info | Show SQLite cache statistics |
| `?` | Help | Print command summary |

### Response entry format (`g`)

The response to a guess is entered as a 5-character string:
`g` = green, `y` = yellow, any other character (including `-`, `0`, `_`,
em-dash) = gray. The parser accepts punctuation liberally because mobile
keyboards sometimes substitute punctuation for hyphens.

### Solve (`s`)

Scores every word in the current input set against the remaining answers
under one of the four methods. When run at the start of a full game
(all answers remaining, all-guesses mode), results are saved to a `.p`
pickle file and reloaded on subsequent runs if the engine has not changed.

### Board (`b`) — Pareto View

Scores all candidates simultaneously on entropy and max-group-size, then
displays the **Pareto frontier**: the max-group-size levels where no other
level has both higher entropy and lower max-group-size.

**Algorithm:**

1. For each unique max-group-size value, find the best entropy among all
   words at that size.
2. Sort sizes ascending. Walk the list keeping a running maximum entropy
   seen so far. A size is on the frontier if and only if its best entropy
   exceeds the running maximum.
3. For each frontier size, show the top 4 words by entropy.

Words in the answer set are marked `*`. Each frontier level shows how many
total words share that max-group-size.

### Lookahead (`l`)

Ranks the top N first guesses by combined two-step entropy. Prompts for N
(default 20). Uses full mode: searches the top N² ranked words as
second-step candidates.

A progress bar reflects only uncached work. After computing, displays a
table of `word / step1 / step2 / combined`.

### Test (`t`)

Analyses a single word in detail:
- If simulation mode is active, shows the response pattern against the
  known answer.
- Checks consistency with all prior guesses; explains each conflict.
- Scores the word under all four methods and shows group count.
- Runs a single-word two-step lookahead (hard mode): step1, step2, combined.
- Shows the 5 largest response groups with up to 6 example words each.

---

## Pickle Cache

When `s` is run at game start on the full word list, results are saved to
`weights-<n_guesses>-<method>.p`. On subsequent runs, the file is loaded if
its mtime is newer than `wordle_engine.py`. This avoids the ~1-minute full
scoring run each session.

---

## Multi-Game Mode

`w` sets the number of simultaneous games (e.g. 4 for Quordle). Each game
gets its own `Solution`. The `g` command can target one board or all boards
at once. `s`, `b`, and `l` can operate on a single board or a joined view
of all unsolved boards.

---

## Platform Notes

### Color Output

On Pythonista, `console.set_color(r, g, b)` is used. On Linux, ANSI escape
codes are used when stdout is a TTY and `NO_COLOR` is not set.

### Display Width

`get_display_width()` tries the following in order:

1. `shutil.get_terminal_size()` — queries the OS via `TIOCGWINSZ`. Works
   in standard terminals and may work in Pythonista if the console exposes
   terminal size.
2. Pythonista pixel-based estimation: reads the console view width in
   points via `console.get_size()`, then tries several monospace fonts via
   `ui.measure_string('M', font=...)` and picks the one whose column count
   is closest to an integer (i.e. most likely to match the actual console
   font). Font candidates: Menlo 12/13/14, DejaVuSansMono 16, Courier 12/14.
3. Falls back to 80 columns.

The detected width is printed at startup: `Display width: N columns`.

### Progress Bar

`ProgressTracker` prints a width-aware progress indicator during long
computations:
- One `.` per percentage point completed.
- `25%` / `50%` / `75%` / `100%` milestone labels.
- An ETA label (e.g. `1m30s`) every 10 seconds.
- Lines wrap at `DISPLAY_WIDTH - 6` columns. Before each wrap, the current
  line is padded to the margin with dots so all rows are the same width.
