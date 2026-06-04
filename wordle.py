"""
wordle.py - Interactive Wordle solver (Pythonista on iOS, Linux-friendly).

Supports single-game and multi-game (quordle, etc.) modes.
When running a single game, redundant prompts are skipped
for a streamlined experience.
"""

import os
import sys
import pickle
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
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANSWER_FILE = "NYT_wordlist.txt"
GUESS_FILE = "wordle.txt"
ENGINE_PATH = wordle_engine.__file__
BUILD = "b12"


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
    """Detect console width (slow path — called at most every 0.5 s)."""
    # Explicit override always wins.
    env = os.environ.get('COLUMNS', '').strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass

    if IS_PYTHONISTA and console is not None:
        try:
            import ui

            dbg = lambda msg: None
            w_points = None
            layer = None
            content_view_used = False

            # Layer 1: console.get_size()
            try:
                w_points, _ = console.get_size()
                layer = 'console.get_size()'
                dbg(f'L1 console.get_size(): {w_points:.1f}pt')
            except AttributeError:
                dbg('L1 console.get_size(): AttributeError')
            except Exception as e:
                dbg(f'L1 console.get_size(): {e}')

            # Layer 2: os.get_terminal_size() on each fd
            if w_points is None:
                for fd in range(3):
                    try:
                        sz = os.get_terminal_size(fd)
                        if sz.columns > 10:
                            dbg(f'L2 os.get_terminal_size({fd}): {sz.columns} cols')
                            return sz.columns
                        else:
                            dbg(f'L2 os.get_terminal_size({fd}): {sz.columns} (ignored)')
                    except OSError as e:
                        dbg(f'L2 os.get_terminal_size({fd}): OSError {e}')

            # Layer 3: Walk ObjC view hierarchy to find OMTextView
            if w_points is None:
                try:
                    from objc_util import ObjCClass

                    def _cls(v):
                        try:
                            return v._get_objc_classname().decode('utf-8', errors='ignore')
                        except Exception:
                            return ''

                    def _walk(v, depth=0):
                        if depth > 10:
                            return None
                        if 'OMTextView' in _cls(v):
                            sv = v.superview()
                            sv_name = _cls(sv) if sv else ''
                            if 'Editor' not in sv_name:
                                return v
                        try:
                            for sv in v.subviews():
                                hit = _walk(sv, depth + 1)
                                if hit is not None:
                                    return hit
                        except Exception:
                            pass
                        return None

                    # Collect ALL OMTextViews so we can pick the right one
                    all_views = []
                    def _walk_all(v, depth=0):
                        if depth > 12:
                            return
                        if 'OMTextView' in _cls(v):
                            sv = v.superview()
                            all_views.append((v, _cls(v), _cls(sv) if sv else ''))
                        try:
                            for sv in v.subviews():
                                _walk_all(sv, depth + 1)
                        except Exception:
                            pass

                    app = ObjCClass('UIApplication').sharedApplication()
                    win = app.keyWindow()
                    kw = win.frame().size.width
                    dbg(f'L3 keyWindow.frame.width={kw:.1f}pt')
                    root = win.rootViewController().view()
                    _walk_all(root)
                    dbg(f'L3 found {len(all_views)} OMTextView(s):')
                    for i, (v, vcls, svcls) in enumerate(all_views):
                        fw = v.frame().size.width
                        fh = v.frame().size.height
                        dbg(f'L3   [{i}] w={fw:.0f}pt h={fh:.0f}pt '
                            f'sv={svcls}')

                    # Pick: prefer OMTextViews whose superview is plain UIView.
                    # In Pythonista the console pane's OMTextView has UIView as
                    # its direct superview; editor panes use OMTextEditorView.
                    # Among UIView-superview candidates take the tallest (full
                    # console height > any toolbar widget).
                    # Fall back to narrowest sub-window view, then first found.
                    cv = None
                    w = 0
                    if all_views:
                        console_cands = [
                            (v, v.frame().size.width, v.frame().size.height)
                            for v, _, svcls in all_views
                            if svcls == 'UIView'
                        ]
                        if console_cands:
                            cv, w, h = max(console_cands, key=lambda x: x[2])
                            dbg(f'L3 chose UIView-sv view: '
                                f'w={w:.0f}pt h={h:.0f}pt')
                        else:
                            sub_cands = [(v, v.frame().size.width)
                                         for v, _, _ in all_views
                                         if v.frame().size.width < kw * 0.95]
                            if sub_cands:
                                cv, w = min(sub_cands, key=lambda x: x[1])
                                dbg(f'L3 chose narrowest sub-window (fallback): '
                                    f'{w:.0f}pt')
                            else:
                                cv, _, _ = all_views[0]
                                w = cv.frame().size.width
                                dbg(f'L3 chose first view (final fallback): '
                                    f'{w:.0f}pt')

                    # Try to get the true text area width, which is smaller than
                    # the OMTextView frame due to internal padding (UITextView
                    # lineFragmentPadding + textContainerInset). Attempt several
                    # ObjC routes; fall back to subtracting the iOS default 10pt.
                    if cv is not None:

                        # Route A: textStorage → NSLayoutManager → NSTextContainer
                        try:
                            ts = cv.textStorage()
                            lm = ts.layoutManagers().firstObject()
                            tc = lm.textContainers().firstObject()
                            tc_w = tc.size().width
                            dbg(f'L3   textContainer.size.width={tc_w:.1f}pt')
                            if 10 < tc_w < w * 0.99:
                                w = tc_w
                                content_view_used = True
                                dbg(f'L3   -> using textContainer width')
                        except Exception as e:
                            dbg(f'L3   textStorage chain: {e}')

                        # Route B: UIScrollView adjustedContentInset
                        if not content_view_used:
                            try:
                                ins = cv.adjustedContentInset()
                                horiz = ins.left + ins.right
                                dbg(f'L3   adjustedContentInset: '
                                    f'left={ins.left:.1f} right={ins.right:.1f}')
                                if horiz > 0.5:
                                    w -= horiz
                                    content_view_used = True
                                    dbg(f'L3   -> subtracted {horiz:.1f}pt '
                                        f'→ {w:.1f}pt')
                            except Exception as e:
                                dbg(f'L3   adjustedContentInset: {e}')

                        # Route C: direct subview with narrower width
                        if not content_view_used:
                            try:
                                for sv in cv.subviews():
                                    sn = _cls(sv)
                                    sw2 = sv.frame().size.width
                                    if 'TextContent' in sn and 10 < sw2 < w * 0.99:
                                        dbg(f'L3   subview {sn} w={sw2:.0f}pt '
                                            f'→ using it')
                                        w = sw2
                                        content_view_used = True
                            except Exception as e:
                                dbg(f'L3   subviews: {e}')

                        # Fallback: subtract UITextView default padding
                        # (lineFragmentPadding 5pt/side + rounding margin).
                        # c=14 is empirically correct for both iPhone and iPad.
                        if not content_view_used:
                            w -= 14

                    if cv is not None and w > 10:
                        w_points = w
                        layer = 'OMTextView.frame'
                    else:
                        dbg('L3 OMTextView: no usable view found')
                except Exception as e:
                    dbg(f'L3 OMTextView walk: exception {e}')

            # Layer 4: Key window frame
            if w_points is None:
                try:
                    from objc_util import ObjCClass
                    app = ObjCClass('UIApplication').sharedApplication()
                    win = app.keyWindow()
                    w = win.frame().size.width
                    dbg(f'L4 keyWindow.frame.width={w:.1f}pt -> using {w-16:.1f}pt')
                    if w > 10:
                        w_points = w - 16
                        layer = 'keyWindow'
                except Exception as e:
                    dbg(f'L4 keyWindow: {e}')

            # Layer 5: Screen size with cap
            if w_points is None:
                sw, sh = ui.get_screen_size()
                capped = min(sw - 16, 720)
                dbg(f'L5 screen={sw:.1f}x{sh:.1f}pt -> capped to {capped:.1f}pt')
                w_points = capped
                layer = 'screen(capped)'

            dbg(f'w_points={w_points:.1f}pt via {layer}')

            # Pythonista's console uses Menlo 14pt. Measure it directly
            # instead of a best-fit search, which is unreliable when the
            # inset correction doesn't land on a near-integer multiple.
            try:
                tw, _ = ui.measure_string('M' * 20, font=('Menlo', 14))
                adv = tw / 20
                cols = int(w_points / adv)
                if cols >= 20:
                    dbg(f'Menlo 14pt adv={adv:.3f}pt -> {cols} cols')
                    return cols
            except Exception as e:
                dbg(f'font: {e}')
        except Exception as e:
            print(f'[width] exception in detection: {e}')

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
        cached = load_cache("weights", gs.n_guesses, mname)
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
        save_cache(results, "weights", gs.n_guesses, mname)

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
        cached = load_cache("weights", gs.n_guesses, "entropy_gain")
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

    top_n = soln.scores[:count]

    # Full mode: search top N² words as second-step candidates
    is_hard = (gs.input_set == InputSet.CURRENT_WORDLIST)
    if is_hard:
        second_step_words = None
        mode_label = "hard mode (subgroup only)"
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


def cmd_test(gs):
    result = pick_one(gs, "Test. ")
    if result is None:
        return
    _, soln = result

    set_display_context(soln)
    print("Word to test? ", end="")
    try:
        word = input().strip().lower()
        assert len(word) == 5

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

        # Two-step lookahead for this word
        ent_score = score_groups(groups, ScoringMethod.ENTROPY_GAIN)
        if n > 2:
            print(f'\n  Two-step lookahead (hard mode):')
            la = soln.compute_lookahead([(word, ent_score)])
            if la:
                _, s1, s2, combined = la[0]
                print(f'    Step 1: {s1:.4f}')
                print(f'    Step 2: {s2:.4f}')
                print(f'    Total:  {combined:.4f}')

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
                handler(gs)
            except Exception as e:
                print_error(f"Error: {e}")
                raise
        else:
            print_error(f"Unknown: {cmd}")
        print_status(gs)


if __name__ == '__main__':
    main()
