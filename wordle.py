"""
wordle.py - Interactive Wordle solver (Pythonista for iOS).

Supports single-game and multi-game (quordle, etc.) modes.
When running a single game, redundant prompts are skipped
for a streamlined experience.
"""

import os
import pickle
from datetime import datetime
import contextlib

import console
import sound

import wordle_engine
from wordle_engine import (
    Solution, ScoringMethod, InputSet,
    load_word_list, calculate_response,
    calculate_group_counts, score_groups,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANSWER_FILE = "NYT_wordlist.txt"
GUESS_FILE = "wordle.txt"
ENGINE_PATH = wordle_engine.__file__


# ---------------------------------------------------------------------------
# Console helpers (Pythonista-specific)
# ---------------------------------------------------------------------------

console.set_color()
sound.set_volume(1)


def get_display_width():
    """
    Auto-detect console width in characters.

    Uses console.get_size() for the view width in
    points, then tries common monospace fonts to find
    which one divides the width into a clean integer
    column count.
    """
    try:
        import ui
        w_points, _ = console.get_size()
        # Try common Pythonista monospace fonts/sizes
        candidates = [
            ('Menlo', 12), ('Menlo', 14),
            ('DejaVuSansMono', 16),
            ('Courier', 12), ('Courier', 14),
        ]
        best_cols = None
        best_err = 999
        for name, size in candidates:
            try:
                cw, _ = ui.measure_string(
                    'M', font=(name, size)
                )
                cols = w_points / cw
                err = abs(cols - round(cols))
                if err < best_err:
                    best_err = err
                    best_cols = int(cols)
            except Exception:
                continue
        if best_cols and best_cols >= 20:
            return best_cols
    except Exception:
        pass
    return 42  # safe iPhone fallback


DISPLAY_WIDTH = get_display_width()
print(f'Display width: {DISPLAY_WIDTH} columns')


@contextlib.contextmanager
def colored_text(color):
    colors = {
        "red":    (1, 0, 0),
        "green":  (0, 0.6, 0),
        "yellow": (0.6, 0.6, 0),
    }
    if isinstance(color, list):
        console.set_color(*color)
    elif color in colors:
        console.set_color(*colors[color])
    else:
        console.set_color()
    try:
        yield
    finally:
        console.set_color()


def print_error(msg):
    sound.play_effect("Error")
    with colored_text("red"):
        print(msg)


def print_success(msg):
    with colored_text("green"):
        print(msg)


# ---------------------------------------------------------------------------
# Response formatting and parsing
# ---------------------------------------------------------------------------

RESPONSE_ABBREV = {'green': 'g', 'yellow': 'y', 'gray': '-'}


def format_response(response):
    """Format a response list into a compact -yg string."""
    return ''.join(RESPONSE_ABBREV[r] for r in response)


def print_colored_pattern(response):
    """Print a -yg pattern string with colors."""
    for sq in response:
        ch = RESPONSE_ABBREV.get(sq, '?')
        if sq in ('green', 'yellow'):
            with colored_text(sq):
                print(ch, end='')
        else:
            print(ch, end='')


def print_colored_word(word, response):
    """Print a word with each letter colored by its response."""
    for letter, color in zip(word, response):
        with colored_text(color):
            print(letter.upper(), end='')


def _is_gray_char(ch):
    """
    Accept 0, _, and any non-alphanumeric character as gray.
    Accommodates mobile keyboards where -- becomes an em dash
    and .. becomes a period-space.
    """
    return ch == '0' or ch == '_' or not ch.isalnum()


def parse_response(response_str):
    """
    Parse a 5-character response string.

    g = green, y = yellow,
    0 / _ / any punctuation = gray.
    """
    if len(response_str) == 5:
        result = []
        for ch in response_str:
            if ch == 'g':
                result.append('green')
            elif ch == 'y':
                result.append('yellow')
            elif _is_gray_char(ch):
                result.append('gray')
            else:
                print_error(
                    f"Invalid '{ch}'. "
                    "g=green, y=yellow, 0=gray."
                )
                return None
        return result
    else:
        parts = response_str.split()
        if len(parts) == 5:
            return parts
        print_error(
            "Need 5 characters (e.g., 00yg0)."
        )
        return None


# ---------------------------------------------------------------------------
# Progress tracker (mobile-friendly)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Width-aware progress for narrow screens.

    Prints dots and percentage labels at 25%
    milestones, wrapping to fit DISPLAY_WIDTH.
    """

    def __init__(self, total):
        self.count = 0
        self.total = total
        self.start_time = datetime.now()
        self.chars_printed = 0
        self.next_milestone = 25
        print('  ', end='', flush=True)
        self.chars_printed = 2

    def update(self):
        self.count += 1
        pct = (self.count * 100) // self.total
        prev = ((self.count - 1) * 100) // self.total
        if pct <= prev:
            return
        # New percentage point reached
        if pct >= self.next_milestone:
            label = f'{self.next_milestone}%'
            print(label, end='', flush=True)
            self.chars_printed += len(label)
            self.next_milestone += 25
        else:
            print('.', end='', flush=True)
            self.chars_printed += 1
        # Wrap before screen edge, but not after 100%
        if (self.chars_printed >= DISPLAY_WIDTH - 6
                and pct < 100):
            print('\n  ', end='', flush=True)
            self.chars_printed = 2

    def finish(self):
        if self.next_milestone <= 100:
            print('100%', end='', flush=True)
        print()
        elapsed = datetime.now() - self.start_time
        print(f'  Duration: {elapsed}')
        return elapsed


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_columns(strings, width=DISPLAY_WIDTH,
                   gap="  ", prefix="    "):
    """Format strings into auto-computed columns."""
    if not strings:
        return []
    max_len = max(len(s) for s in strings)
    cols = max(1, (width - len(prefix)) // (max_len + len(gap)))
    rows = max(1, -(-len(strings) // cols))  # ceiling
    out = []
    for row in range(rows):
        parts = strings[row::rows]
        line = prefix + gap.join(
            f'{s:{max_len}}' for s in parts
        )
        out.append(line)
    return out


def print_scored_list(pairs, method=None, limit=20):
    """Print ranked (word, score) pairs in columns."""
    def fmt(s):
        if method:
            return method.format_score(s)
        return f'{s:0.4f}'
    items = [f'{w}: {fmt(s)}' for w, s in pairs[:limit]]
    if len(pairs) > limit:
        items.append('...')
    if items:
        print('\n'.join(format_columns(items)))


def print_word_list(words, limit=20):
    """Print plain words in columns."""
    items = list(words[:limit])
    if len(words) > limit:
        items.append(f'... ({len(words)} total)')
    if items:
        print('\n'.join(format_columns(items)))


def print_guesses(soln):
    """Print guess history with colored output."""
    console.set_color()
    if not soln.guesses:
        return
    print("  Prior guesses:")
    for word, response in soln.guesses:
        print(f'    {word}  ', end='')
        print_colored_pattern(response)
        print('  ', end='')
        print_colored_word(word, response)
        print()
    console.set_color()


# ---------------------------------------------------------------------------
# Pickle cache
# ---------------------------------------------------------------------------

def _cache_path(prefix, n, method_name=None):
    if method_name:
        return f"{prefix}-{n}-{method_name}.p"
    return f"{prefix}-{n}.p"


def load_cache(prefix, n, method_name=None):
    filepath = _cache_path(prefix, n, method_name)
    try:
        cache_mtime = os.path.getmtime(filepath)
        engine_mtime = os.path.getmtime(ENGINE_PATH)
        if cache_mtime < engine_mtime:
            return None
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, OSError):
        return None


def save_cache(data, prefix, n, method_name=None):
    with open(_cache_path(prefix, n, method_name),
              "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """Shared state for all command handlers."""

    def __init__(self, all_answers, all_guesses):
        self.all_answers = all_answers
        self.all_guesses = all_guesses
        self.n_answers = len(all_answers)
        self.n_guesses = len(all_guesses)
        self.solutions = [Solution(all_answers)]
        self.columns = 1
        self.input_set = InputSet.ALL_GUESSES
        self.volume = 10

    def reset_all(self):
        self.solutions = [Solution(self.all_answers)]
        self.columns = 1
        self.input_set = InputSet.ALL_GUESSES

    @property
    def single(self):
        return len(self.solutions) == 1


# ---------------------------------------------------------------------------
# Solution pickers
# ---------------------------------------------------------------------------

def pick_one(gs, prefix=""):
    """Pick a solution. Auto-selects when N=1."""
    if gs.single:
        return 0, gs.solutions[0]
    print(f'{prefix}Which solution? ', end='')
    try:
        sn = int(input()) - 1
        return sn, gs.solutions[sn]
    except (ValueError, IndexError):
        print_error("Invalid solution number.")
        return None


def pick_one_or_all(gs, prefix=""):
    """Pick a solution or all. Auto-selects when N=1."""
    if gs.single:
        return 0, gs.solutions[0]
    print(f'{prefix}Which solution or (a)ll? ', end='')
    ans = input().strip()
    if ans.lower() == 'a':
        return 'all', None
    try:
        sn = int(ans) - 1
        return sn, gs.solutions[sn]
    except (ValueError, IndexError):
        print_error("Invalid response.")
        return None


# ---------------------------------------------------------------------------
# Command: Guess
# ---------------------------------------------------------------------------

def cmd_guess(gs):
    if gs.single:
        local_solns = [(0, gs.solutions[0])]
    else:
        result = pick_one_or_all(gs, "Guess. ")
        if result is None:
            return
        key, val = result
        if key == 'all':
            local_solns = list(enumerate(gs.solutions))
        else:
            local_solns = [(key, val)]

    print("Word to guess? ", end="")
    try_word = input().strip().lower()
    if len(try_word) != 5:
        print_error("Word must be 5 letters.")
        return

    for i, soln in local_solns:
        if not gs.single:
            print(f'\nSolution {i + 1}')
            col = i % gs.columns
            print(f'  Row:{(i // gs.columns) + 1} '
                  f'Col:{col + 1} ', end='')
            for c in range(gs.columns):
                print('X' if c == col else '-', end='')
            print()

        if len(soln.current_words) == 1:
            print_success(
                f'  Already solved: '
                f'{soln.current_words[0]}'
            )
            sound.play_effect("Jump", 1, 0.3)
            continue
        elif len(soln.current_words) == 0:
            print_error("  No remaining words!")
            continue
        else:
            n = len(soln.current_words)
            print(f'  {n} words before guess')
            print_guesses(soln)

        if soln.answer_word:
            response = calculate_response(
                try_word, soln.answer_word
            )
            print(f'  -> {try_word}  ', end='')
            print_colored_pattern(response)
            print('  ', end='')
            print_colored_word(try_word, response)
            print()
        else:
            label = ("Response" if gs.single
                     else f"Response for #{i + 1}")
            print(f"{label} to {try_word}? ", end="")
            response_str = input().strip()
            if not response_str or response_str == 'stop':
                print_error("Stopped.")
                break
            response = parse_response(response_str)
            if response is None:
                continue

        soln.apply_guess(try_word, response)
        cw = soln.current_words
        print(f'  {len(cw)} words remaining', end='')
        if len(cw) == 0:
            print_error(": No words remaining!")
        elif len(cw) == 1:
            sound.play_effect("Coin_1")
            print_success(f': {cw[0]}')
        else:
            print()


# ---------------------------------------------------------------------------
# Command: Solve
# ---------------------------------------------------------------------------

def cmd_solve(gs):
    if gs.single:
        soln = gs.solutions[0]
    else:
        result = pick_one_or_all(gs, "Solve. ")
        if result is None:
            return
        key, val = result
        if key == 'all':
            soln = Solution.join(gs.solutions)
        else:
            soln = val

    # Input word set
    iset = gs.input_set
    if not gs.single:
        print('Input words? '
              '(h)ard, (a)ll, (s)olved? ', end='')
        ch = input().strip().lower()
        if ch == 'h':
            iset = InputSet.CURRENT_WORDLIST
        elif ch == 'a':
            iset = InputSet.ALL_GUESSES
        elif ch == 's':
            iset = InputSet.SOLVED_WORDS
        else:
            print_error("Invalid choice.")
            return
    else:
        is_hard = (iset == InputSet.CURRENT_WORDLIST)
        label = ("current words" if is_hard
                 else "all guesses")
        print(f'Input: {label}')

    if iset == InputSet.CURRENT_WORDLIST:
        wordlist = soln.current_words
    elif iset == InputSet.SOLVED_WORDS:
        wordlist = [
            s.current_words[0] for s in gs.solutions
            if len(s.current_words) == 1
        ]
        if not wordlist:
            print_error("No solutions found yet!")
            return
    else:
        wordlist = gs.all_guesses

    # Scoring method
    methods = list(ScoringMethod)
    print("\nScoring method:")
    for i, m in enumerate(methods):
        arrow = "^" if m.higher_is_better else "v"
        print(f"  {i + 1}. {m.label} ({arrow})")
    print(f"Choose (1-{len(methods)})? ", end='')
    try:
        method = methods[int(input().strip()) - 1]
    except (ValueError, IndexError):
        print_error("Invalid choice.")
        return

    is_full = (len(soln.current_words) == gs.n_answers
               and iset == InputSet.ALL_GUESSES)
    mname = method.name.lower()

    # Try cache
    if is_full:
        cached = load_cache("weights", gs.n_guesses, mname)
        if cached:
            soln.scores = sorted(cached,
                                 key=method.sort_key())
            soln.scores_method = method
            soln.scores_updated = True
            print(f"\nCached ({gs.n_guesses} words, "
                  f"{method.label}).")
            print("Best guesses:")
            print_scored_list(soln.scores, method)
            return

    n_in = len(wordlist)
    n_rem = len(soln.current_words)
    print(f"\nScoring {n_in:,} guesses vs "
          f"{n_rem:,} words.")
    print(f"Method: {method.label}")

    tracker = ProgressTracker(n_in)
    results = soln.compute_scores(
        wordlist, method, progress_callback=tracker.update
    )
    tracker.finish()

    if is_full:
        save_cache(results, "weights", gs.n_guesses,
                   mname)

    print(f"\n{method.label}:")
    print("Best guesses:")
    print_scored_list(results, method)


# ---------------------------------------------------------------------------
# Command: Display
# ---------------------------------------------------------------------------

def cmd_display(gs):
    result = pick_one(gs, "Display. ")
    if result is None:
        return
    _, soln = result

    n = len(soln.current_words)
    print(f"\n{n:,} words remaining:")
    if soln.scores_updated:
        filtered = [
            (w, s) for w, s in soln.scores
            if w in soln.current_words
        ]
        print_scored_list(filtered, soln.scores_method)
    else:
        print_word_list(soln.current_words)
    print_guesses(soln)


# ---------------------------------------------------------------------------
# Command: Test
# ---------------------------------------------------------------------------

def _explain_conflict(pos, guess_word, recorded, hypothetical):
    """Return an English explanation of a conflict."""
    letter = guess_word[pos].upper()
    rec = recorded[pos]
    hyp = hypothetical[pos]

    has_other = any(
        recorded[j] != 'gray'
        for j in range(5)
        if j != pos
        and guess_word[j] == guess_word[pos]
    )

    if rec == 'gray':
        if has_other:
            return (f"no extra {letter} beyond "
                    "those found")
        return f"{letter} is not in the answer"
    elif rec == 'green':
        return f"position {pos + 1} must be {letter}"
    elif rec == 'yellow' and hyp == 'gray':
        return f"{letter} must be in the answer"
    elif rec == 'yellow' and hyp == 'green':
        return (f"{letter} can't be at "
                f"position {pos + 1}")
    return f"{letter}: expected {rec}, got {hyp}"


def cmd_test(gs):
    result = pick_one(gs, "Test. ")
    if result is None:
        return
    _, soln = result

    print("Word to test? ", end="")
    try:
        word = input().strip().lower()
        assert len(word) == 5

        # Show pattern if answer is set
        if soln.answer_word:
            resp = calculate_response(
                word, soln.answer_word
            )
            print(f'\n  vs {soln.answer_word.upper()}: ',
                  end='')
            print_colored_pattern(resp)
            print('  ', end='')
            print_colored_word(word, resp)
            print()

        # Consistency check
        if soln.guesses:
            consistent = True
            for gw, recorded in soln.guesses:
                hyp = calculate_response(gw, word)
                if hyp != recorded:
                    if consistent:
                        print("  Conflicts:")
                        consistent = False
                    print(f'    {gw.upper()}: ', end='')
                    print_colored_pattern(recorded)
                    print(' vs ', end='')
                    print_colored_pattern(hyp)
                    print()
                    for pos in range(5):
                        if recorded[pos] != hyp[pos]:
                            reason = _explain_conflict(
                                pos, gw, recorded, hyp
                            )
                            print(f'      -> {reason}')
            if consistent:
                print('  Consistent with all guesses.')
            else:
                print('  Not a valid candidate.')

        # Scores
        groups = calculate_group_counts(
            word, soln.current_words
        )
        n = len(soln.current_words)
        print(f'\n  {word.upper()} vs {n} words:')
        for m in ScoringMethod:
            s = score_groups(groups, m)
            print(f'    {m.label}: {m.format_score(s)}')
        print(f'    Groups: {len(groups)}')
    except Exception as e:
        print_error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Command: Include / Exclude
# ---------------------------------------------------------------------------

def cmd_include(gs):
    result = pick_one(gs, "Include. ")
    if result is None:
        return
    _, soln = result
    print("Letters to include? ", end="")
    soln.include_letters(input().strip().lower())


def cmd_exclude(gs):
    result = pick_one(gs, "Exclude. ")
    if result is None:
        return
    _, soln = result
    print("Letters to exclude? ", end="")
    soln.exclude_letters(input().strip().lower())


# ---------------------------------------------------------------------------
# Command: Reset
# ---------------------------------------------------------------------------

def cmd_reset(gs):
    if gs.single:
        print("Reset? (y/n) ", end="")
        if input().strip().lower() == 'y':
            gs.reset_all()
            print("Reset.")
        else:
            print("Cancelled.")
    else:
        result = pick_one_or_all(gs, "Reset. ")
        if result is None:
            return
        key, val = result
        if key == 'all':
            gs.reset_all()
            print("All reset.")
        else:
            val.reset()
            print(f"Solution {key + 1} reset.")


# ---------------------------------------------------------------------------
# Command: Answer (simulation mode)
# ---------------------------------------------------------------------------

def cmd_answer(gs):
    result = pick_one_or_all(gs, "Answer. ")
    if result is None:
        return
    key, val = result

    if key == 'all':
        local_solns = list(enumerate(gs.solutions))
    else:
        local_solns = [(key, val)]

    for i, soln in local_solns:
        if not gs.single:
            print(f'  Solution {i + 1}: ', end='')

        if soln.answer_word:
            print(f"{soln.answer_word.upper()}. "
                  "Clear? (y/n) ", end="")
            ans = input().strip().lower()
            if ans == 'y':
                soln.answer_word = None
                print("  Simulation off.")
            else:
                print("  New word (or blank): ", end="")
                new = input().strip().lower()
                if new and len(new) == 5:
                    soln.answer_word = new
                    print(f"  Answer: {new.upper()}")
        else:
            print("Answer word? ", end="")
            new = input().strip().lower()
            if len(new) != 5:
                print_error("Must be 5 letters.")
            else:
                soln.answer_word = new
                print(f"  Simulation on: {new.upper()}")


# ---------------------------------------------------------------------------
# Command: Wordlist count
# ---------------------------------------------------------------------------

def cmd_wordcount(gs):
    print("How many games? ", end="")
    try:
        wc = int(input().strip())
        if wc < 1:
            raise ValueError
        gs.solutions = [
            Solution(gs.all_answers) for _ in range(wc)
        ]
        if wc > 1:
            print("How many per row? ", end="")
            gs.columns = int(input().strip())
        else:
            gs.columns = 1
        label = "game" if wc == 1 else "games"
        print(f"Set up {wc} {label}.")
    except (ValueError, TypeError):
        print_error("Invalid number.")


# ---------------------------------------------------------------------------
# Command: Hard mode toggle
# ---------------------------------------------------------------------------

def cmd_hardmode(gs):
    if gs.input_set == InputSet.ALL_GUESSES:
        gs.input_set = InputSet.CURRENT_WORDLIST
        print("  Hard mode on")
    else:
        gs.input_set = InputSet.ALL_GUESSES
        print("  Hard mode off")


# ---------------------------------------------------------------------------
# Command: Volume
# ---------------------------------------------------------------------------

def cmd_volume(gs):
    print(f"Volume (0-10, now {gs.volume})? ",
          end="")
    try:
        gs.volume = int(input().strip())
        sound.set_volume(gs.volume / 10)
        print(f"  Volume: {gs.volume}")
    except (ValueError, TypeError):
        print_error("Invalid volume.")


# ---------------------------------------------------------------------------
# Command: Help
# ---------------------------------------------------------------------------

def cmd_help(gs):
    hard = gs.input_set.name
    if gs.single:
        aw = gs.solutions[0].answer_word
        sim = aw.upper() if aw else "off"
    else:
        sim_count = sum(
            1 for s in gs.solutions if s.answer_word
        )
        sim = f"{sim_count}/{len(gs.solutions)} set"
    print(f"""
  g = Guess a word
  s = Solve (find best guess)
  d = Display remaining words
  t = Test a word (all methods)
  i = Include letters (filter)
  x = eXclude letters (filter)
  r = Reset
  a = Answer for simulation ({sim})
  w = Game count (quordle, etc.)
  h = Hard mode ({hard})
  v = Volume ({gs.volume})
  ? = This help
""")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

COMMANDS = {
    'g': cmd_guess,
    's': cmd_solve,
    'd': cmd_display,
    't': cmd_test,
    'i': cmd_include,
    'x': cmd_exclude,
    'r': cmd_reset,
    'a': cmd_answer,
    'w': cmd_wordcount,
    'h': cmd_hardmode,
    'v': cmd_volume,
    '?': cmd_help,
}


def print_status(gs):
    """Print current game status."""
    print(f'\n{"=" * DISPLAY_WIDTH}')
    if gs.single:
        soln = gs.solutions[0]
        if soln.answer_word:
            with colored_text("yellow"):
                print(f"Sim: {soln.answer_word.upper()}")
        words = soln.current_words
        n = len(words)
        if n == 0:
            with colored_text("red"):
                print("No words remaining!")
        elif n == 1:
            print_success(f"Solved: {words[0]}")
        else:
            print(f"{n:,} words remaining")
    else:
        n = len(gs.solutions)
        print(f'{n} wordlists')
        for i, soln in enumerate(gs.solutions):
            words = soln.current_words
            print(f'{i + 1:3d}: ', end='')
            if len(words) == 0:
                with colored_text("red"):
                    print('0 remaining')
            elif len(words) == 1:
                print('1 remaining', end='')
                print_success(f'  {words[0]}')
            else:
                print(f'{len(words):,} remaining', end='')
                if soln.answer_word:
                    with colored_text("yellow"):
                        print(f'  sim:{soln.answer_word}',
                              end='')
                print()


def main():
    all_answers = load_word_list(ANSWER_FILE)
    all_guesses = load_word_list(GUESS_FILE)
    print(f"Loaded {len(all_answers):,} answers, "
          f"{len(all_guesses):,} guesses.")

    gs = GameState(all_answers, all_guesses)

    while True:
        print_status(gs)
        print(f"\nCommand (gsdtixrawhv?)? ",
              end="")
        cmd = input().strip()
        if not cmd:
            continue
        handler = COMMANDS.get(cmd[0])
        if handler:
            try:
                handler(gs)
            except Exception as e:
                print_error(f"Error: {e}")
                raise
        else:
            print_error(f"Unknown: {cmd}")


if __name__ == '__main__':
    main()
