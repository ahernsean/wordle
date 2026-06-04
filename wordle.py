"""
wordle.py - Interactive Wordle solver (Pythonista on iOS, Linux-friendly).

Supports single-game and multi-game (quordle, etc.) modes.
When running a single game, redundant prompts are skipped
for a streamlined experience.
"""

import os
import sys
import shutil
import time
from collections import defaultdict
from datetime import datetime
import contextlib

try:
    import console  # Pythonista
except ImportError:
    console = None

import wordle_engine
from adaptive_cache_sqlite import LookaheadCache
from wordle_engine import (
    Solution, ScoringMethod, InputSet, ResponseCache,
    load_word_list, calculate_response,
    calculate_group_counts, score_groups,
    decode_response, max_entropy,
    answer_to_restriction,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANSWER_FILE = "NYT_wordlist.txt"
GUESS_FILE = "wordle.txt"
ENGINE_PATH = wordle_engine.__file__
BUILD = "b24"


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

IS_PYTHONISTA = console is not None
IS_LINUX = sys.platform.startswith("linux")
SUPPORTS_COLOR = (
    IS_PYTHONISTA
    or (
        IS_LINUX
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )
)

ANSI_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
}
ANSI_RESET = "\033[0m"
ANSI_BOLD  = "\033[1m" if SUPPORTS_COLOR else ""

if console is not None:
    console.set_color()


def reset_color():
    if IS_PYTHONISTA and console is not None:
        console.set_color()
    elif SUPPORTS_COLOR:
        print(ANSI_RESET, end="")


_width_cache: list = [None]  # [cols] — refreshed once per REPL cycle

def get_display_width() -> int:
    """Return cached display width. Never re-detects on its own."""
    if _width_cache[0] is None:
        _width_cache[0] = _detect_display_width()
    return _width_cache[0]


def refresh_display_width() -> None:
    """Re-detect display width. Called once before each prompt."""
    _width_cache[0] = _detect_display_width()


def _detect_display_width() -> int:
    """Detect console width; called at most once per REPL cycle."""
    env = os.environ.get('COLUMNS', '').strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass

    if IS_PYTHONISTA and console is not None:
        try:
            import ui
            w_points = None

            # L1: console.get_size()
            try:
                w_points, _ = console.get_size()
            except Exception:
                pass

            # L2: os.get_terminal_size() on each fd
            if w_points is None:
                for fd in range(3):
                    try:
                        sz = os.get_terminal_size(fd)
                        if sz.columns > 10:
                            return sz.columns
                    except OSError:
                        pass

            # L3: Walk ObjC view hierarchy to find OMTextView
            if w_points is None:
                try:
                    from objc_util import ObjCClass

                    def _cls(v):
                        try:
                            return v._get_objc_classname().decode('utf-8', errors='ignore')
                        except Exception:
                            return ''

                    all_views = []
                    def _walk_all(v, depth=0):
                        if depth > 12:
                            return
                        if 'OMTextView' in _cls(v):
                            sv = v.superview()
                            all_views.append((v, _cls(sv) if sv else ''))
                        try:
                            for sv in v.subviews():
                                _walk_all(sv, depth + 1)
                        except Exception:
                            pass

                    app = ObjCClass('UIApplication').sharedApplication()
                    win = app.keyWindow()
                    kw = win.frame().size.width
                    _walk_all(win.rootViewController().view())

                    # Prefer the OMTextView whose superview is plain UIView
                    # (the console pane); among those take the tallest.
                    # Fall back to narrowest sub-window view, then first found.
                    cv = None
                    w = 0
                    if all_views:
                        console_cands = [
                            (v, v.frame().size.width, v.frame().size.height)
                            for v, svcls in all_views if svcls == 'UIView'
                        ]
                        if console_cands:
                            cv, w, _ = max(console_cands, key=lambda x: x[2])
                        else:
                            sub_cands = [
                                (v, v.frame().size.width)
                                for v, _ in all_views
                                if v.frame().size.width < kw * 0.95
                            ]
                            if sub_cands:
                                cv, w = min(sub_cands, key=lambda x: x[1])
                            else:
                                cv, _ = all_views[0]
                                w = cv.frame().size.width

                    # Subtract internal UITextView padding.
                    # Routes A and B attempt ObjC reads; c=14 is the
                    # empirically correct fallback for iPhone and iPad.
                    if cv is not None:
                        adjusted = False
                        try:  # Route A: textStorage → NSLayoutManager → NSTextContainer
                            tc_w = (cv.textStorage().layoutManagers()
                                    .firstObject().textContainers()
                                    .firstObject().size().width)
                            if 10 < tc_w < w * 0.99:
                                w = tc_w
                                adjusted = True
                        except Exception:
                            pass
                        if not adjusted:
                            try:  # Route B: adjustedContentInset
                                ins = cv.adjustedContentInset()
                                horiz = ins.left + ins.right
                                if horiz > 0.5:
                                    w -= horiz
                                    adjusted = True
                            except Exception:
                                pass
                        if not adjusted:
                            w -= 14

                    if cv is not None and w > 10:
                        w_points = w
                except Exception:
                    pass

            # L4: Key window frame
            if w_points is None:
                try:
                    from objc_util import ObjCClass
                    win = ObjCClass('UIApplication').sharedApplication().keyWindow()
                    w = win.frame().size.width
                    if w > 10:
                        w_points = w - 16
                except Exception:
                    pass

            # L5: Screen size with cap
            if w_points is None:
                sw, _ = ui.get_screen_size()
                w_points = min(sw - 16, 720)

            try:
                tw, _ = ui.measure_string('M' * 20, font=('Menlo', 14))
                cols = int(w_points / (tw / 20))
                if cols >= 20:
                    return cols
            except Exception:
                pass
        except Exception:
            pass

        return 42

    try:
        return shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        return 80



@contextlib.contextmanager
def colored_text(color):
    pythonista_colors = {
        "red":    (1, 0, 0),
        "green":  (0, 0.6, 0),
        "yellow": (0.6, 0.6, 0),
    }
    if IS_PYTHONISTA and console is not None:
        if isinstance(color, list):
            console.set_color(*color)
        elif color in pythonista_colors:
            console.set_color(*pythonista_colors[color])
        else:
            reset_color()
    elif SUPPORTS_COLOR and color in ANSI_COLORS:
        print(ANSI_COLORS[color], end="")
    try:
        yield
    finally:
        reset_color()


def print_error(msg):
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

    Prints dots for each percentage point,
    milestone labels at 25% intervals, and
    inline ETA every ETA_INTERVAL seconds.
    """

    ETA_INTERVAL = 10  # seconds between ETA reports

    def __init__(self, total):
        self.count = 0
        self.total = max(total, 1)
        self.start_time = datetime.now()
        self.last_eta = self.start_time
        self.chars_printed = 0
        self.next_milestone = 25
        print('  ', end='', flush=True)
        self.chars_printed = 2

    @staticmethod
    def _fmt_eta(td):
        """Format a timedelta compactly."""
        secs = int(td.total_seconds())
        if secs < 60:
            return f'{secs}s'
        mins = secs // 60
        secs = secs % 60
        if secs == 0:
            return f'{mins}m'
        return f'{mins}m{secs:02d}s'

    def _emit(self, token):
        """Print token; pad with dots and wrap if it would overflow the margin."""
        margin = get_display_width() - 2
        if self.chars_printed + len(token) > margin:
            pad = margin - self.chars_printed
            if pad > 0:
                print('.' * pad, end='', flush=True)
            print('\n  ', end='', flush=True)
            self.chars_printed = 2
        print(token, end='', flush=True)
        self.chars_printed += len(token)
        if self.chars_printed >= margin:
            print('\n  ', end='', flush=True)
            self.chars_printed = 2

    def update(self):
        self.count += 1
        pct = (self.count * 100) // self.total
        prev = ((self.count - 1) * 100) // self.total
        if pct > prev:
            if pct >= self.next_milestone:
                self._emit(f'{self.next_milestone}%')
                self.next_milestone += 25
            else:
                self._emit('.')
        now = datetime.now()
        if ((now - self.last_eta).total_seconds()
                >= self.ETA_INTERVAL):
            self.last_eta = now
            frac = self.count / self.total
            if 0 < frac < 1:
                elapsed = now - self.start_time
                remaining = elapsed * (1 - frac) / frac
                self._emit(self._fmt_eta(remaining))

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

_display_answer_set = set()
_display_max_ent = 0.0


def set_display_context(soln):
    """Set the display context from a solution."""
    global _display_answer_set, _display_max_ent
    if soln is not None:
        _display_answer_set = soln.answer_set
        _display_max_ent = max_entropy(
            len(soln.current_words)
        )
    else:
        _display_answer_set = set()
        _display_max_ent = 0.0


def _mark(word):
    """Return '*' if word is in the current answer set, ' ' otherwise."""
    return '*' if word in _display_answer_set else ' '


def _is_max_ent(score):
    """True if score equals the theoretical max entropy."""
    return (_display_max_ent > 0
            and abs(score - _display_max_ent) < 1e-9)


def format_columns(strings, width=None,
                   gap="  ", prefix="    "):
    """Format strings into auto-computed columns."""
    if not strings:
        return []
    if width is None:
        width = get_display_width()
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
    """Print ranked (word, score) pairs in columns.
    Words in the answer set are marked with *.
    Max-entropy scores are marked with =.
    """
    def fmt(s):
        if method:
            fs = method.format_score(s)
        else:
            fs = f'{s:0.4f}'
        if (method == ScoringMethod.ENTROPY_GAIN
                and _is_max_ent(s)):
            fs += '='
        return fs
    items = [
        f'{w}{_mark(w)}: {fmt(s)}'
        for w, s in pairs[:limit]
    ]
    if len(pairs) > limit:
        items.append('...')
    if items:
        print('\n'.join(format_columns(items)))


def print_word_list(words, limit=20):
    """Print plain words in columns.
    Words in the answer set are marked with *.
    """
    items = [f'{w}{_mark(w)}' for w in words[:limit]]
    if items:
        print('\n'.join(format_columns(items)))
    if len(words) > limit:
        print(f'    ... ({len(words)} total)')


def print_guesses(soln):
    """Print guess history with colored output."""
    reset_color()
    if not soln.guesses:
        return
    print("  Prior guesses:")
    for word, response in soln.guesses:
        print(f'    {word}  ', end='')
        print_colored_pattern(response)
        print('  ', end='')
        print_colored_word(word, response)
        print()
    reset_color()


# ---------------------------------------------------------------------------
# Pickle cache
# ---------------------------------------------------------------------------



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
        self.cache = ResponseCache(all_answers)
        self.lookahead_cache_path = os.path.abspath("lookahead_cache.sqlite3")
        self.lookahead_cache = LookaheadCache(
            self.lookahead_cache_path,
            all_answers,
        )
        print(f"Lookahead cache: {self.lookahead_cache_path}")
        self.solutions = [Solution(all_answers,
                                   all_guesses,
                                   self.cache,
                                   self.lookahead_cache)]
        self.columns = 1
        self.input_set = InputSet.ALL_GUESSES

    def reset_all(self):
        self.solutions = [Solution(self.all_answers,
                                   self.all_guesses,
                                   self.cache,
                                   self.lookahead_cache)]
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
        if soln.fallback_active:
            with colored_text("yellow"):
                print(f'\n  Answer list exhausted. '
                      f'Fell back to full guess '
                      f'vocabulary.')
        print(f'  {len(cw)} words remaining', end='')
        if len(cw) == 0:
            print_error(": No words remaining!")
        elif len(cw) == 1:
            print_success(f': {cw[0]}')
        else:
            print()


# ---------------------------------------------------------------------------
# Command: Solve
# ---------------------------------------------------------------------------

def _input_wordlist(gs, soln, iset):
    """Resolve InputSet to the appropriate word list."""
    if iset == InputSet.HARD_MODE:
        return soln.hard_mode_words(gs.all_guesses)
    if iset == InputSet.CURRENT_WORDLIST:
        return soln.current_words
    if iset == InputSet.SOLVED_WORDS:
        return [
            s.current_words[0] for s in gs.solutions
            if len(s.current_words) == 1
        ]
    return gs.all_guesses


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

    set_display_context(soln)

    iset = gs.input_set
    if not gs.single:
        print('Input words? '
              '(h)ard mode, (a)ll, (s)olved? ', end='')
        ch = input().strip().lower()
        if ch == 'h':
            iset = InputSet.HARD_MODE
        elif ch == 'a':
            iset = InputSet.ALL_GUESSES
        elif ch == 's':
            iset = InputSet.SOLVED_WORDS
        else:
            print_error("Invalid choice.")
            return
    else:
        labels = {
            InputSet.ALL_GUESSES:     "all guesses",
            InputSet.HARD_MODE:       "hard mode",
            InputSet.CURRENT_WORDLIST: "answers only",
        }
        print(f'Input: {labels.get(iset, iset.name)}')

    wordlist = _input_wordlist(gs, soln, iset)
    if not wordlist:
        print_error("No words in input set!")
        return

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

    if is_full:
        cached = gs.lookahead_cache.read_scores(mname)
        if cached:
            results = sorted(cached, key=method.sort_key())
            soln.scores = results
            soln.scores_method = method
            soln.scores_updated = True
            for word, score in cached:
                soln.word_scores.setdefault(word, {})[method] = score
            print(f"\nCached ({gs.n_guesses} words, "
                  f"{method.label}).")
            if method == ScoringMethod.ENTROPY_GAIN:
                print(f"(= = max entropy "
                      f"{_display_max_ent:.4f})")
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
        gs.lookahead_cache.write_scores(results, mname)

    print(f"\n{method.label}:")
    if method == ScoringMethod.ENTROPY_GAIN:
        print(f"(= = max entropy "
              f"{_display_max_ent:.4f})")
    print("Best guesses:")
    print_scored_list(results, method)


# ---------------------------------------------------------------------------
# Command: Grid (entropy vs max group size, Pareto-filtered)
# ---------------------------------------------------------------------------

def _compute_pareto(word_ent_mx):
    """
    Return the Pareto frontier: words where no other word has both
    strictly higher entropy AND strictly smaller max group size.

    Returns list of (word, ent, mx) in ascending mx order,
    only including mx levels that are not dominated.
    """
    from collections import defaultdict as _dd
    mx_best = {}  # mx -> best entropy at that level
    for word, ent, mx in word_ent_mx:
        if mx not in mx_best or ent > mx_best[mx]:
            mx_best[mx] = ent

    frontier_mxs = []
    running_max = -1.0
    for mx in sorted(mx_best):
        if mx_best[mx] > running_max:
            frontier_mxs.append(mx)
            running_max = mx_best[mx]

    return set(frontier_mxs)


def cmd_grid(gs):
    if gs.single:
        soln = gs.solutions[0]
    else:
        result = pick_one_or_all(gs, "Grid. ")
        if result is None:
            return
        key, val = result
        if key == 'all':
            soln = Solution.join(gs.solutions)
        else:
            soln = val

    set_display_context(soln)

    iset = gs.input_set
    if not gs.single:
        print('Input words? '
              '(h)ard mode, (a)ll, (s)olved? ', end='')
        ch = input().strip().lower()
        if ch == 'h':
            iset = InputSet.HARD_MODE
        elif ch == 'a':
            iset = InputSet.ALL_GUESSES
        elif ch == 's':
            iset = InputSet.SOLVED_WORDS
        else:
            print_error("Invalid choice.")
            return
    else:
        labels = {
            InputSet.ALL_GUESSES:     "all guesses",
            InputSet.HARD_MODE:       "hard mode",
            InputSet.CURRENT_WORDLIST: "answers only",
        }
        print(f'Input: {labels.get(iset, iset.name)}')

    wordlist = _input_wordlist(gs, soln, iset)
    if not wordlist:
        print_error("No words in input set!")
        return

    methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MINIMAX]
    n_in = len(wordlist)
    n_rem = len(soln.current_words)
    print(f"\nScoring {n_in:,} guesses vs "
          f"{n_rem:,} words.")
    print("Board: Entropy vs Max Group Size (Pareto view)")

    tracker = ProgressTracker(n_in)
    results = soln.compute_scores_multi(
        wordlist, methods,
        progress_callback=tracker.update
    )
    tracker.finish()

    # Build flat list for Pareto analysis
    word_ent_mx = [
        (word, scores[ScoringMethod.ENTROPY_GAIN],
         int(scores[ScoringMethod.MINIMAX]))
        for word, scores in results
    ]

    frontier_mxs = _compute_pareto(word_ent_mx)

    # Group by mx, sorted by entropy descending within each group
    from collections import defaultdict as _dd
    mx_groups = _dd(list)
    for word, ent, mx in word_ent_mx:
        mx_groups[mx].append((word, ent))
    for mx in mx_groups:
        mx_groups[mx].sort(key=lambda x: -x[1])

    NEIGHBORS_PER_LEVEL = 4

    def ent_fmt(e):
        s = ScoringMethod.ENTROPY_GAIN.format_score(e)
        if _is_max_ent(e):
            s += '='
        return s

    print(f"\n(* = in answer set)")
    print("(showing Pareto frontier + top words per group size)")

    any_shown = False
    for mx in sorted(frontier_mxs):
        entries = mx_groups[mx]
        if not entries:
            continue
        to_show = entries[:NEIGHBORS_PER_LEVEL]
        any_shown = True
        print(f"\n  Max group {mx}  "
              f"({len(entries)} words at this level)")
        items = [
            f'{w}{_mark(w)}: {ent_fmt(e)}'
            for w, e in to_show
        ]
        print('\n'.join(format_columns(items)))

    if not any_shown:
        print("  (no words to display)")


# ---------------------------------------------------------------------------
# Command: Lookahead (two-step entropy)
# ---------------------------------------------------------------------------

LOOKAHEAD_N = 20


def cmd_lookahead(gs):
    if gs.single:
        soln = gs.solutions[0]
    else:
        result = pick_one_or_all(gs, "Lookahead. ")
        if result is None:
            return
        key, val = result
        if key == 'all':
            soln = Solution.join(gs.solutions)
        else:
            soln = val

    set_display_context(soln)

    n_rem = len(soln.current_words)
    if n_rem <= 2:
        print("Two or fewer words remain, "
              "lookahead not needed.")
        return

    print(f"How many top words? ({LOOKAHEAD_N}) ", end="")
    n_input = input().strip()
    if n_input:
        try:
            count = int(n_input)
            if count < 1:
                raise ValueError
        except ValueError:
            print_error("Invalid number.")
            return
    else:
        count = LOOKAHEAD_N

    # Try loading full-game entropy ranking from disk cache
    is_full = (len(soln.current_words) == gs.n_answers
               and gs.input_set != InputSet.CURRENT_WORDLIST)
    if is_full and (not soln.scores_updated
                    or soln.scores_method != ScoringMethod.ENTROPY_GAIN):
        cached = gs.lookahead_cache.read_scores("entropy_gain")
        if cached:
            soln.scores = sorted(cached,
                                 key=ScoringMethod.ENTROPY_GAIN.sort_key())
            soln.scores_method = ScoringMethod.ENTROPY_GAIN
            soln.scores_updated = True
            for word, score in cached:
                soln.word_scores.setdefault(
                    word, {})[ScoringMethod.ENTROPY_GAIN] = score
            print(f"  (entropy loaded from cache, {gs.n_guesses} words)")

    # Auto-compute entropy ranking if not already done
    if (not soln.scores_updated
            or soln.scores_method != ScoringMethod.ENTROPY_GAIN):
        print("Computing entropy ranking for lookahead...")
        rank_words = (soln.current_words
                      if gs.input_set == InputSet.CURRENT_WORDLIST
                      else gs.all_guesses)
        tracker = ProgressTracker(len(rank_words))
        soln.compute_scores(
            rank_words,
            ScoringMethod.ENTROPY_GAIN,
            progress_callback=tracker.update,
        )
        tracker.finish()
        if is_full:
            gs.lookahead_cache.write_scores(soln.scores, "entropy_gain")

    top_n = soln.scores[:count]

    # Answers-only mode: restrict step-2 candidates to the subgroup
    is_answers_only = (gs.input_set == InputSet.CURRENT_WORDLIST)
    if is_answers_only:
        second_step_words = None
        mode_label = "answers only (subgroup)"
    else:
        step2_count = max(count * count, 100)
        second_step_words = [w for w, _s in soln.scores[:step2_count]]
        mode_label = f"top {len(second_step_words)} guesses"

    print(f"\nTwo-step lookahead on top {len(top_n)} words "
          f"vs {n_rem:,} remaining.")
    print(f"  Second step: {mode_label}")

    # Count work for progress tracker (skipping cache hits)
    total_work = 0
    for word, first_ent in top_n:
        if soln.cache:
            grouped = soln.cache.group_words(word, soln.current_words)
        else:
            grouped = defaultdict(list)
            for answer in soln.current_words:
                from wordle_engine import _encode_response
                pat = _encode_response(calculate_response(word, answer))
                grouped[pat].append(answer)
        policy = 'full' if second_step_words else 'hard'
        for subgroup in grouped.values():
            cnt = len(subgroup)
            if cnt <= 2:
                continue
            if soln.lookahead_cache:
                blob = LookaheadCache.encode_subset(subgroup)
                if soln.lookahead_cache.read(blob, policy) is not None:
                    continue
            total_work += (len(second_step_words)
                           if second_step_words else cnt)

    tracker = ProgressTracker(max(total_work, 1))

    start = time.perf_counter()
    results = soln.compute_lookahead(
        top_n,
        second_step_words=second_step_words,
        progress_callback=tracker.update,
    )
    tracker.finish()
    elapsed = time.perf_counter() - start
    print(f"  Elapsed: {elapsed:.1f}s")

    print(f"\nTwo-step entropy lookahead:")
    print(f"  {'Word':<7} {'Step1':>7}  "
          f"{'Step2':>7}  {'Total':>7}")
    print(f"  {'----':<7} {'-----':>7}  "
          f"{'-----':>7}  {'-----':>7}")
    for word, s1, s2, combined in results:
        m = _mark(word)
        me_flag = '=' if _is_max_ent(s1) else ' '
        print(f"  {word}{m} "
              f"{s1:7.4f}{me_flag} "
              f"{s2:7.4f}  "
              f"{combined:7.4f}")


# ---------------------------------------------------------------------------
# Command: Display
# ---------------------------------------------------------------------------

def cmd_display(gs):
    result = pick_one(gs, "Display. ")
    if result is None:
        return
    _, soln = result

    set_display_context(soln)
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


def _multistep_stats(word, soln, step2_pool=None, hard_mode=False,
                     all_guesses=None):
    """
    Compute 3-step expected entropy and group stats for a single word.
    Returns a dict with keys: step1, step2, step3, max_grp, max_grp2,
    wt_avg, prob_finish, buckets.

    step2_pool: candidate pool for step 2. If None and not hard_mode,
        uses only the subgroup (answers-only mode).
    hard_mode: if True, compute valid step-2 candidates per subgroup
        by applying the step-1 response constraints to all_guesses.
    Step 3 always uses subgroup candidates — sub-subgroups are tiny.
    """
    cache = soln.cache
    remaining = soln.current_words
    n = len(remaining)

    if cache:
        s1_groups = cache.group_words(word, remaining)
    else:
        s1_groups = defaultdict(list)
        for answer in remaining:
            pat = tuple(calculate_response(word, answer))
            s1_groups[pat].append(answer)

    group_counts = {p: len(g) for p, g in s1_groups.items()}
    step1     = score_groups(group_counts, ScoringMethod.ENTROPY_GAIN)
    wt_avg    = score_groups(group_counts, ScoringMethod.WEIGHTED_AVG)
    max_grp   = int(score_groups(group_counts, ScoringMethod.MINIMAX))
    prob_fin  = score_groups(group_counts, ScoringMethod.PROB_FINISH)

    buckets = [0, 0, 0, 0, 0]
    for g in s1_groups.values():
        k = len(g)
        if k == 1:    buckets[0] += 1
        elif k <= 4:  buckets[1] += 1
        elif k <= 9:  buckets[2] += 1
        elif k <= 49: buckets[3] += 1
        else:         buckets[4] += 1

    step2 = 0.0
    step3 = 0.0

    t0 = time.time()
    _prog = {'on': False, 'step3': False}

    for pat, subgroup in s1_groups.items():
        k = len(subgroup)
        if k <= 1:
            continue

        if hard_mode and all_guesses:
            resp = decode_response(pat)
            cands2 = answer_to_restriction(word, resp).apply(all_guesses)
        elif step2_pool is not None:
            cands2 = step2_pool
        else:
            cands2 = subgroup

        # Announce progress only if we've been computing more than 5 seconds
        if not _prog['on'] and time.time() - t0 > 5:
            suffix = '2, 3' if _prog['step3'] else '2'
            print(f'  Computing entropy...{suffix}', end='', flush=True)
            _prog['on'] = True

        best2_ent = 0.0
        best2_grps = None
        for c2 in cands2:
            if cache:
                sg = cache.group_words(c2, subgroup)
            else:
                sg = defaultdict(list)
                for ans in subgroup:
                    p2 = tuple(calculate_response(c2, ans))
                    sg[p2].append(ans)
            ent = score_groups(
                {p: len(g) for p, g in sg.items()},
                ScoringMethod.ENTROPY_GAIN,
            )
            if ent > best2_ent:
                best2_ent = ent
                best2_grps = sg

        step2 += (k / n) * best2_ent

        if best2_grps:
            if not _prog['step3']:
                _prog['step3'] = True
                if _prog['on']:
                    print(', 3', end='', flush=True)
            for sub_sub in best2_grps.values():
                kk = len(sub_sub)
                if kk <= 1:
                    continue
                best3_ent = 0.0
                for c3 in sub_sub:
                    if cache:
                        ssc = cache.group_counts(c3, sub_sub)
                    else:
                        ssc = calculate_group_counts(c3, sub_sub)
                    ent = score_groups(ssc, ScoringMethod.ENTROPY_GAIN)
                    if ent > best3_ent:
                        best3_ent = ent
                step3 += (kk / n) * best3_ent

    if _prog['on']:
        print()  # finish the "Computing entropy..." line

    return {
        'step1': step1, 'step2': step2, 'step3': step3,
        'max_grp': max_grp,
        'wt_avg': wt_avg, 'prob_finish': prob_fin,
        'buckets': buckets,
    }


def _compare_words(words, soln, step2_pool=None, hard_mode=False,
                   all_guesses=None):
    """Compare 2–4 words side by side."""
    n = len(soln.current_words)

    print(f'\n  Computing {", ".join(w.upper() for w in words)}...', flush=True)
    all_stats = []
    for i, w in enumerate(words):
        if len(words) > 1:
            print(f'  [{i + 1}/{len(words)}] {w.upper()}', flush=True)
        all_stats.append(_multistep_stats(w, soln, step2_pool, hard_mode, all_guesses))

    lw = 9  # "Entropy 1" = 9, "10-49:" = 6

    # Build all data rows up front so we can measure max column width
    totals = [s['step1'] + s['step2'] + s['step3'] for s in all_stats]
    data_rows = [
        ('Wt avg',    [s['wt_avg']      for s in all_stats], '{:.2f}', False),
        ('Max grp',   [s['max_grp']     for s in all_stats], '{:d}',   False),
        ('Solve%',    [s['prob_finish'] for s in all_stats], '{:.1%}', True),
        ('Entropy 1', [s['step1']       for s in all_stats], '{:.4f}', True),
    ]
    if n > 2:
        data_rows += [
            ('+ ent. 2', [s['step2'] for s in all_stats], '{:.4f}', True),
            ('+ ent. 3', [s['step3'] for s in all_stats], '{:.4f}', True),
            ('Total ent', totals,                          '{:.4f}', True),
        ]

    bucket_labels = ['1:', '2-4:', '5-9:', '10-49:', '50+:']
    bucket_higher = [True, False, False, False, False]
    bucket_rows = []
    for bi, (lbl, hb) in enumerate(zip(bucket_labels, bucket_higher)):
        vals = [s['buckets'][bi] for s in all_stats]
        if any(vals):
            bucket_rows.append((lbl, vals, '{:d}', hb))

    # Column width: wide enough for word names AND every formatted value.
    # Using a consistent format per row means right-justifying to cw
    # automatically aligns decimal points within each row.
    cw = max(len(w) for w in words)
    for _, values, fmt, _ in data_rows + bucket_rows:
        for v in values:
            cw = max(cw, len(fmt.format(v)))

    def print_row(label, values, fmt, higher_better=True):
        padded = [fmt.format(v).rjust(cw) for v in values]
        best = max(values) if higher_better else min(values)
        all_tied = all(v == best for v in values)
        print(f'  {label:<{lw}} ', end='')
        for i, (p, v) in enumerate(zip(padded, values)):
            if i:
                print('  ', end='')
            if not all_tied and v == best:
                with colored_text('green'):
                    print(p, end='')
            else:
                print(p, end='')
        print()

    print(f'\n  {n:,} words:')
    print(f'  {"":>{lw}} ' + '  '.join(w.upper().rjust(cw) for w in words))

    for label, values, fmt, hb in data_rows:
        print_row(label, values, fmt, hb)

    if bucket_rows:
        print()
        for label, values, fmt, hb in bucket_rows:
            print_row(label, values, fmt, hb)


def cmd_test(gs, inline=''):
    result = pick_one(gs, "Test. ")
    if result is None:
        return
    _, soln = result

    set_display_context(soln)
    if inline:
        line = inline
    else:
        print("Word(s) to test? ", end="")
        line = input().strip()

    # Derive step-2 pool and hard-mode flag from current input-set setting
    iset = gs.input_set
    if iset == InputSet.HARD_MODE:
        step2_pool = None
        hard_mode  = True
    elif iset == InputSet.CURRENT_WORDLIST:
        step2_pool = None
        hard_mode  = False
    else:  # ALL_GUESSES — cap at 200 top-entropy words; searching all 12k is ~65x slower
        hard_mode  = False
        if (soln.scores_updated
                and soln.scores_method == ScoringMethod.ENTROPY_GAIN):
            step2_pool = [w for w, _ in soln.scores[:200]]
        else:
            step2_pool = soln.all_answers

    words = line.lower().split()
    try:
        if 2 <= len(words) <= 4:
            assert all(len(w) == 5 for w in words)
            _compare_words(words, soln, step2_pool, hard_mode, gs.all_guesses)
            return
        assert len(words) == 1 and len(words[0]) == 5
        word = words[0]

        # Show pattern if answer is set
        if soln.answer_word:
            resp = calculate_response(word, soln.answer_word)
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
        if soln.cache:
            groups = soln.cache.group_counts(word, soln.current_words)
        else:
            groups = calculate_group_counts(word, soln.current_words)
        n = len(soln.current_words)
        in_answers = _mark(word).strip()
        label = f'{word.upper()}'
        if in_answers:
            label += ' (in answer set)'
        print(f'\n  {label} vs {n} words:')
        for m in ScoringMethod:
            s = score_groups(groups, m)
            extra = ''
            if (m == ScoringMethod.ENTROPY_GAIN
                    and _is_max_ent(s)):
                extra = ' = max'
            print(f'    {m.label}: '
                  f'{m.format_score(s)}{extra}')
        print(f'    Groups: {len(groups)}')

        # Multi-step lookahead for this word
        if n > 2:
            st = _multistep_stats(word, soln, step2_pool, hard_mode,
                                  gs.all_guesses)
            mode = ('hard mode' if hard_mode
                    else (f'top {len(step2_pool)}' if step2_pool else 'answers only'))
            print(f'\n  Multi-step lookahead ({mode}):')
            total = st['step1'] + st['step2'] + st['step3']
            _rows = [
                ('Entropy 1:', st['step1']),
                ('+ ent. 2:',  st['step2']),
                ('+ ent. 3:',  st['step3']),
                ('Total:',     total),
            ]
            _lw = max(len(r[0]) for r in _rows)
            _vw = max(len(f'{v:.4f}') for _, v in _rows)
            for lbl, val in _rows:
                print(f'    {lbl:<{_lw}}  {val:>{_vw}.4f}')

        # Group size distribution
        sorted_groups = sorted(groups.items(), key=lambda x: -x[1])
        b = [0, 0, 0, 0, 0]
        for _, cnt in sorted_groups:
            if cnt == 1:
                b[0] += 1
            elif cnt <= 4:
                b[1] += 1
            elif cnt <= 9:
                b[2] += 1
            elif cnt <= 49:
                b[3] += 1
            else:
                b[4] += 1
        labels = ['1', '2-4', '5-9', '10-49', '50+']
        pairs = [f'{lbl}:{n}' for lbl, n in zip(labels, b) if n]
        prefix = '  Subgroup sizes: '
        width = get_display_width()
        print()
        cur = prefix
        for i, pair in enumerate(pairs):
            sep = '  ' if cur != prefix else ''
            if len(cur) + len(sep) + len(pair) <= width:
                cur += sep + pair
            else:
                print(cur)
                cur = '    ' + pair
        print(cur)

    except AssertionError:
        print_error("Word must be 5 letters.")
    except Exception as e:
        print_error(f"Error: {e}")
        raise


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
# Command: Undo
# ---------------------------------------------------------------------------

def cmd_undo(gs):
    if gs.single:
        soln = gs.solutions[0]
    else:
        result = pick_one(gs, "Undo. ")
        if result is None:
            return
        _, soln = result

    if not soln.guesses:
        print("  Nothing to undo.")
        return

    last_word, last_resp = soln.guesses[-1]
    if soln.undo_guess():
        n = len(soln.current_words)
        print(f"  Undid: {last_word.upper()} "
              f"({format_response(last_resp)})")
        print(f"  {n:,} words remaining.")
    else:
        print("  Nothing to undo.")


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
            Solution(gs.all_answers, gs.all_guesses,
                     gs.cache, gs.lookahead_cache)
            for _ in range(wc)
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
    cycle = {
        InputSet.ALL_GUESSES:      InputSet.HARD_MODE,
        InputSet.HARD_MODE:        InputSet.CURRENT_WORDLIST,
        InputSet.CURRENT_WORDLIST: InputSet.ALL_GUESSES,
    }
    labels = {
        InputSet.ALL_GUESSES:      "all guesses (normal)",
        InputSet.HARD_MODE:        "hard mode (satisfies constraints)",
        InputSet.CURRENT_WORDLIST: "answers only (strictest)",
    }
    gs.input_set = cycle.get(gs.input_set, InputSet.ALL_GUESSES)
    print(f"  Input set: {labels[gs.input_set]}")


# ---------------------------------------------------------------------------
# Command: Cache info
# ---------------------------------------------------------------------------

def cmd_cacheinfo(gs):
    lc = gs.lookahead_cache
    rows, mtime = lc.stats()
    if mtime:
        ts = datetime.utcfromtimestamp(mtime).isoformat() + "Z"
    else:
        ts = "n/a"
    print("\nLookahead cache:")
    print(f"  db path:    {gs.lookahead_cache_path}")
    print(f"  entries:    {rows:,}")
    print(f"  last write: {ts}")


# ---------------------------------------------------------------------------
# Command: Help
# ---------------------------------------------------------------------------

def cmd_help(gs):
    iset_labels = {
        InputSet.ALL_GUESSES:      "all guesses",
        InputSet.HARD_MODE:        "hard mode",
        InputSet.CURRENT_WORDLIST: "answers only",
    }
    iset = iset_labels.get(gs.input_set, gs.input_set.name)
    if gs.single:
        aw = gs.solutions[0].answer_word
        sim = aw.upper() if aw else "off"
        nguesses = len(gs.solutions[0].guesses)
    else:
        sim_count = sum(
            1 for s in gs.solutions if s.answer_word
        )
        sim = f"{sim_count}/{len(gs.solutions)} set"
        nguesses = "?"
    print(f"""
  g = Guess a word
  s = Solve (find best guess)
  b = Board (Pareto entropy vs max group)
  l = Lookahead (two-step entropy)
  d = Display remaining words
  t = Test a word (all methods + lookahead)
  i = Include letters (filter)
  x = eXclude letters (filter)
  u = Undo last guess  ({nguesses} guesses so far)
  r = Reset
  a = Answer for simulation ({sim})
  w = Game count (quordle, etc.)
  h = Input set: {iset}
  c = Cache info
  ? = This help
""")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

COMMANDS = {
    'g': cmd_guess,
    's': cmd_solve,
    'b': cmd_grid,
    'l': cmd_lookahead,
    'd': cmd_display,
    't': cmd_test,
    'i': cmd_include,
    'x': cmd_exclude,
    'u': cmd_undo,
    'r': cmd_reset,
    'a': cmd_answer,
    'w': cmd_wordcount,
    'h': cmd_hardmode,
    'c': cmd_cacheinfo,
    '?': cmd_help,
}


def print_status(gs):
    """Print current game status."""
    refresh_display_width()
    print(f'\n{"=" * get_display_width()}')
    if gs.single:
        soln = gs.solutions[0]
        if soln.answer_word:
            with colored_text("yellow"):
                print(f"Sim: {soln.answer_word.upper()}")
        if soln.fallback_active:
            with colored_text("yellow"):
                print("(using full guess vocabulary)")
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
                if soln.fallback_active:
                    with colored_text("yellow"):
                        print(' [fallback]', end='')
                if soln.answer_word:
                    with colored_text("yellow"):
                        print(f'  sim:{soln.answer_word}',
                              end='')
                print()


def main():
    print(f"wordle.py {BUILD}")
    all_answers = load_word_list(ANSWER_FILE)
    all_guesses = load_word_list(GUESS_FILE)
    print(f"Loaded {len(all_answers):,} answers, "
          f"{len(all_guesses):,} guesses.")

    gs = GameState(all_answers, all_guesses)

    print_status(gs)
    while True:
        print(f"\nCommand (gsbldtixurаwhc?)? ", end="")
        try:
            cmd = input().strip()
        except EOFError:
            print()
            print("Exiting.")
            break
        except KeyboardInterrupt:
            print()
            print("Interrupted.")
            break

        if not cmd:
            print_status(gs)
            continue
        handler = COMMANDS.get(cmd[0])
        if handler:
            try:
                inline = cmd[1:].strip()
                if inline and handler is cmd_test:
                    cmd_test(gs, inline)
                else:
                    handler(gs)
            except Exception as e:
                print_error(f"Error: {e}")
                raise
        else:
            print_error(f"Unknown: {cmd}")
        print_status(gs)


if __name__ == '__main__':
    main()
