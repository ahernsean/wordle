"""
wordle.py - Interactive Wordle solver (Pythonista on iOS, Linux-friendly).

Supports single-game and multi-game (quordle, etc.) modes.
When running a single game, redundant prompts are skipped
for a streamlined experience.
"""

import os
import sys
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime
import contextlib

try:
    import console  # Pythonista
except ImportError:
    console = None

import wordle_engine
from cache_sqlite import ScoreCache, MemoryScoreCache
from wordle_engine import (
    Solution, ScoringMethod, GuessUniverse, ComplianceFilter, ResponseCache,
    load_word_list, calculate_response,
    calculate_group_counts, score_groups, score_groups_multi,
    decode_response, max_entropy,
    answer_to_restriction,
    min_expected_guesses,
    ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED, ERD_ANSWERS_UNFILTERED,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANSWER_FILE = "NYT_wordlist.txt"
WORDS_FILE = "wordle.txt"
ENGINE_PATH = wordle_engine.__file__
BUILD = "b79"


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

    def __init__(self, all_answers, all_words):
        self.all_answers = all_answers
        self.all_words = all_words
        self.score_cache_path = os.path.abspath("wordle_cache.sqlite3")
        self.score_cache = ScoreCache(
            self.score_cache_path,
            all_answers,
        )
        self.cache = ResponseCache(all_answers, self.score_cache)
        print(f"Score cache: {self.score_cache_path}")
        # Long-lived: hard-mode ERD results are namespaced internally by a
        # fingerprint of the eligible-guess vocabulary (see set_scope), so
        # entries from earlier positions/games simply become unreachable
        # rather than needing to be reset — and become reusable again "for
        # free" if that exact vocabulary recurs (e.g. via undo/replay).
        self.constrained_erd_cache = MemoryScoreCache()
        self._start_new_game()

    def reset_all(self):
        """Abandon the current game(s) and start fresh — same word lists,
        same persistent caches, but every other piece of per-game state
        (solutions, mode, ERD progress/cache) goes back to its initial value.

        Centralized here and in __init__ so a freshly-started game and a
        reset game can never drift out of sync: any new per-game field only
        needs to be added in _start_new_game to be reset correctly too.
        """
        self._start_new_game()

    def _start_new_game(self):
        self.solutions = [Solution(self.all_answers,
                                   self.all_words,
                                   self.cache,
                                   self.score_cache)]
        self.columns = 1
        self.universe = GuessUniverse.ALL_WORDS
        self.compliance = ComplianceFilter.UNFILTERED
        self.last_erd_root_progress = (0, 0, None)  # (done, total, (best_word, best_erd))
        # constrained_erd_cache is intentionally NOT reset here — it is
        # long-lived and self-scoping (see __init__ and MemoryScoreCache).

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

# Candidate-mode selection grid: which static word list a guess may be drawn
# from (GuessUniverse) is independent of whether it must satisfy every clue
# revealed so far (ComplianceFilter) — see their docstrings in wordle_engine.
# Every (universe, compliance) pair is a meaningful, selectable mode; the
# dicts below are total over all four cells, by construction, rather than
# naming three of them and leaving the fourth to be discovered later.

_CANDIDATE_LABELS = {
    (GuessUniverse.ALL_WORDS,   ComplianceFilter.UNFILTERED): "any word",
    (GuessUniverse.ALL_WORDS,   ComplianceFilter.COMPLIANT):  "hard mode",
    (GuessUniverse.ALL_ANSWERS, ComplianceFilter.COMPLIANT):  "possible answers",
    (GuessUniverse.ALL_ANSWERS, ComplianceFilter.UNFILTERED): "answer-shaped words (unfiltered)",
}


def _candidate_label(universe, compliance):
    return _CANDIDATE_LABELS[(universe, compliance)]


def _is_hard_mode(gs):
    """True for real Wordle hard mode: all words, filtered to clue-compliant.

    The only cell of the grid whose eligible-guess vocabulary is
    path-dependent (see _warmer_branch_key) — the other COMPLIANT cell
    (answers-only) narrows to soln.current_words, which is already
    keyed by position and needs no extra vocabulary fingerprint.
    """
    return (gs.universe is GuessUniverse.ALL_WORDS
            and gs.compliance is ComplianceFilter.COMPLIANT)


def _is_possible_answers_mode(gs):
    """True for the (ALL_ANSWERS, COMPLIANT) cell — guesses narrowed to the
    live remaining-answer set, exactly soln.current_words."""
    return (gs.universe is GuessUniverse.ALL_ANSWERS
            and gs.compliance is ComplianceFilter.COMPLIANT)


def _input_wordlist(gs, soln, universe, compliance):
    """Resolve a (universe, compliance) grid cell to its candidate word list."""
    base = gs.all_words if universe is GuessUniverse.ALL_WORDS else gs.all_answers
    if compliance is ComplianceFilter.UNFILTERED:
        return base
    if universe is GuessUniverse.ALL_ANSWERS:
        # Not constraint_compliant_words(base): current_words is already
        # clue-compliant *and* narrowed by any manual include/exclude-letter
        # filters, which constraint_compliant_words doesn't apply.
        return soln.current_words
    return soln.constraint_compliant_words(base)


def _solved_words(gs):
    """Answers of completed sub-boards — a transient multi-board display
    filter, independent of either grid axis (it isn't a candidate-guess
    vocabulary at all, just "what's already been solved")."""
    return [s.current_words[0] for s in gs.solutions if len(s.current_words) == 1]


def _resolve_candidate_selection(gs):
    """Determine the candidate word-source for this solve/grid invocation.

    Single-board games use the persistent (gs.universe, gs.compliance)
    selection set via the 'c' command. Multi-board games prompt per
    invocation instead — there's no single "current position" for a
    persistent choice to describe — and additionally offer (s)olved, a
    transient display filter with no place on the grid (see _solved_words).

    Returns (universe, compliance, solved_only), or None on invalid input
    (the caller should treat that exactly like any other bad-input abort).
    """
    if gs.single:
        print(f'Candidates: {_candidate_label(gs.universe, gs.compliance)}')
        return gs.universe, gs.compliance, False
    print('Input words? '
          '(h)ard mode, (a)ll, (s)olved? ', end='')
    ch = input().strip().lower()
    if ch == 'h':
        return GuessUniverse.ALL_WORDS, ComplianceFilter.COMPLIANT, False
    if ch == 'a':
        return GuessUniverse.ALL_WORDS, ComplianceFilter.UNFILTERED, False
    if ch == 's':
        return None, None, True
    print_error("Invalid choice.")
    return None


def _resolve_candidate_wordlist(gs, soln):
    """Resolve this invocation's candidate word list, or None to abort.

    Wraps _resolve_candidate_selection: turns its (universe, compliance,
    solved_only) result into the actual word list and folds in the "No
    words in input set!" guard — the exact sequence both cmd_solve and
    cmd_grid need and nothing else, so they can share it verbatim.
    """
    selection = _resolve_candidate_selection(gs)
    if selection is None:
        return None
    universe, compliance, solved_only = selection
    wordlist = (_solved_words(gs) if solved_only
                else _input_wordlist(gs, soln, universe, compliance))
    if not wordlist:
        print_error("No words in input set!")
        return None
    return wordlist


class _ERDModeConfig:
    """ERD cache/policy/guess-vocabulary mapping for one grid cell.

    Single source of truth for the (universe, compliance) → (cache, policy,
    persist, guesses) relationship, referenced by _erd_cache_and_policy,
    cmd_solve, and the warmer-init block in main() so that mapping lives in
    exactly one place.
    """
    __slots__ = ('policy', 'persist', 'cache_attr', 'guesses_fn')

    def __init__(self, policy, persist, cache_attr, guesses_fn):
        self.policy = policy
        self.persist = persist
        self.cache_attr = cache_attr  # 'score_cache' or 'constrained_erd_cache'
        self.guesses_fn = guesses_fn  # (gs, soln) -> guess vocabulary

    def cache(self, gs, soln):
        return (gs.constrained_erd_cache if self.cache_attr == 'constrained_erd_cache'
                else soln.score_cache)


_ERD_MODE_CONFIG = {
    (GuessUniverse.ALL_WORDS, ComplianceFilter.UNFILTERED): _ERDModeConfig(
        policy=ERD_ALL, persist=True, cache_attr='score_cache',
        guesses_fn=lambda gs, soln: gs.all_words),
    (GuessUniverse.ALL_WORDS, ComplianceFilter.COMPLIANT): _ERDModeConfig(
        policy=ERD_CONSTRAINED, persist=False, cache_attr='constrained_erd_cache',
        guesses_fn=lambda gs, soln: soln.constraint_compliant_words(gs.all_words)),
    (GuessUniverse.ALL_ANSWERS, ComplianceFilter.COMPLIANT): _ERDModeConfig(
        policy=ERD_ANSWERS, persist=True, cache_attr='score_cache',
        guesses_fn=lambda gs, soln: soln.current_words),
    (GuessUniverse.ALL_ANSWERS, ComplianceFilter.UNFILTERED): _ERDModeConfig(
        policy=ERD_ANSWERS_UNFILTERED, persist=True, cache_attr='score_cache',
        guesses_fn=lambda gs, soln: gs.all_answers),
}


def _erd_mode_config(gs):
    """ERD config for the current (universe, compliance) grid cell.

    The dict above is total over all four cells, so this is a direct
    lookup — no fallback is needed (there is no fifth combination gs.universe
    / gs.compliance could hold).
    """
    return _ERD_MODE_CONFIG[(gs.universe, gs.compliance)]


def _warmer_branch_key(gs, soln, effective_guesses):
    """Identifies the (position, guess-vocabulary) a warmer is working on.

    Two snapshots compare equal exactly when a running warmer is still doing
    useful work for the branch the user is now looking at — i.e. when it is
    safe to just let it keep going rather than tearing it down and starting
    an identical one from scratch (which would race the old one to compute
    and announce the very same root result).

    current_words alone is not enough in hard mode: the eligible-guess
    vocabulary is path-dependent, so the same remaining-word set can arise
    from genuinely different guess histories (e.g. via undo) with different
    effective_guesses — hence the vocabulary fingerprint.
    """
    vocab_fp = (MemoryScoreCache.fingerprint_vocabulary(effective_guesses)
                if _is_hard_mode(gs) else None)
    return (tuple(sorted(soln.current_words)), (gs.universe, gs.compliance), vocab_fp)


def _erd_cache_and_policy(gs, soln):
    """Return (score_cache, policy) appropriate for the current grid cell.

    (ALL_WORDS,   UNFILTERED) → SQLite,           ERD_ALL
    (ALL_WORDS,   COMPLIANT)  → MemoryScoreCache, ERD_CONSTRAINED
    (ALL_ANSWERS, COMPLIANT)  → SQLite,           ERD_ANSWERS
    (ALL_ANSWERS, UNFILTERED) → SQLite,           ERD_ANSWERS_UNFILTERED
    """
    cfg = _erd_mode_config(gs)
    return cfg.cache(gs, soln), cfg.policy


def _erd_root_progress_tag(warmer, soln):
    """Status-line fragment for ERD warmup progress on the current position.

    Reads directly off the live warmer rather than a snapshot in `gs`:
    a snapshot taken before the current command ran (e.g. a guess that
    just narrowed current_words) would describe a now-superseded branch.
    The word-set check below catches that: if the warmer's word set doesn't
    match the position we're about to display, its numbers aren't about
    this position, so there's nothing honest to show yet.

    Phases in order:
      ordering    root_total == 0 (_ranked_root_guesses running)
      scanning    root_total > 0
    """
    if warmer is None or set(warmer._words) != set(soln.current_words):
        return ''
    if warmer.root_total < 0:
        return ''  # trivial position (≤2 words); no scan runs
    if warmer.root_total == 0:
        return '  [ERD: ordering candidates...]'
    if warmer.root_best is not None:
        bw, bs = warmer.root_best
        return (f'  [ERD: scanning {warmer.root_done}/{warmer.root_total} '
                f'— best so far {bw.upper()} {bs:.3f}]')
    return f'  [ERD: scanning {warmer.root_done}/{warmer.root_total}]'


def _erd_solve_scores(soln, score_cache=None, policy=ERD_ALL, guesses=None):
    """
    Rank candidates by ERD using only cached subgroup values.
    Returns sorted (word, erd_cost) list, lowest first, or None if any
    subgroup for any candidate is missing from the cache.

    score_cache: cache to read from; defaults to soln.score_cache (SQLite).
    policy:      cache namespace (ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED).
    guesses:     candidate words to rank.  Defaults to soln.current_words
                 (answer words only).  Pass the full word list to include
                 non-answer candidates in any-word mode.
    """
    from wordle_engine import _encode_response
    n = len(soln.current_words)
    sc = score_cache if score_cache is not None else soln.score_cache
    cache = soln.cache
    candidates = guesses if guesses is not None else soln.current_words
    answer_set = set(soln.current_words)
    results = []
    for word in candidates:
        if cache and word in cache.answer_words:
            groups = cache.group_words(word, soln.current_words)
        else:
            groups = defaultdict(list)
            for answer in soln.current_words:
                pat = _encode_response(calculate_response(word, answer))
                groups[pat].append(answer)
        cost = 1.0
        ok = True
        for sg in groups.values():
            k = len(sg)
            if k == 0:
                continue
            if k == 1 and sg[0] == word:
                continue  # all-green branch: solved, 0 extra guesses
            if k == 1:
                cost += 1.0 / n  # base case: one word left = 1 more guess
                continue
            hit = sc.read(ScoreCache.encode_subset(sg), policy)
            if hit is None:
                ok = False
                break
            cost += (k / n) * hit[1]
        if not ok:
            if word in answer_set:
                return None  # answer-word miss means warmer not ready
            continue        # non-answer miss: skip silently, try others
        results.append((word, cost))
    results.sort(key=lambda x: x[1])
    return results


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

    # ERD option: available when the full tree for this position is cached
    # for the current input mode.
    erd_root = None
    erd_sc, erd_policy = _erd_cache_and_policy(gs, soln)
    if not soln._is_full_game() and erd_sc is not None:
        root_key = ScoreCache.encode_subset(soln.current_words)
        erd_root = erd_sc.read(root_key, erd_policy)

    methods = list(ScoringMethod)
    erd_idx = len(methods) + 1
    print("\nScoring method:")
    for i, m in enumerate(methods):
        arrow = "^" if m.higher_is_better else "v"
        print(f"  {i + 1}. {m.label} ({arrow})")
    if erd_root is not None:
        print(f"  {erd_idx}. ERD: expected remaining depth (v) [{erd_root[1]:.3f}]")
    else:
        root_done, root_total, root_best = gs.last_erd_root_progress
        if root_total < 0:
            prog = 'not ready'
        elif root_total == 0:
            prog = 'ordering candidates...'
        elif root_best is not None:
            bw, bs = root_best
            prog = f'scanning root {root_done}/{root_total} — best so far {bw.upper()} {bs:.3f}'
        else:
            prog = f'scanning root {root_done}/{root_total}'
        print(f"  {erd_idx}. ERD: expected remaining depth  ({prog})")
    print(f"Choose (1-{erd_idx})? ", end='')
    try:
        raw = int(input().strip())
    except ValueError:
        print_error("Invalid choice.")
        return

    # ERD branch: candidates come from the effective guess vocabulary.
    if raw == erd_idx:
        if erd_root is None:
            print_error("ERD tree not ready yet. Wait for the [ERD ready] notification.")
            return
        erd_guesses = _erd_mode_config(gs).guesses_fn(gs, soln)
        scores = _erd_solve_scores(soln, erd_sc, erd_policy, guesses=erd_guesses)
        if scores is None:
            print_error("ERD cache incomplete — some subgroups missing.")
            return
        print("\nERD:")
        print("Best guesses:")
        print_scored_list(scores, method=None)
        return

    try:
        method = methods[raw - 1]
    except IndexError:
        print_error("Invalid choice.")
        return

    wordlist = _resolve_candidate_wordlist(gs, soln)
    if wordlist is None:
        return

    # Fast path: already computed this session
    if soln.scores_updated and soln.scores_method == method:
        print(f"\n{method.label} (cached):")
        if method == ScoringMethod.ENTROPY_GAIN:
            print(f"(= = max entropy {_display_max_ent:.4f})")
        print("Best guesses:")
        print_scored_list(soln.scores, method)
        return

    n_in = len(wordlist)
    n_rem = len(soln.current_words)
    print(f"\nScoring {n_in:,} guesses vs {n_rem:,} words.")
    print(f"Method: {method.label}")

    tracker = ProgressTracker(n_in)
    results = soln.compute_scores(
        wordlist, method, progress_callback=tracker.update
    )
    tracker.finish()

    print(f"\n{method.label}:")
    if method == ScoringMethod.ENTROPY_GAIN:
        print(f"(= = max entropy {_display_max_ent:.4f})")
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

    wordlist = _resolve_candidate_wordlist(gs, soln)
    if wordlist is None:
        return

    methods = [ScoringMethod.ENTROPY_GAIN, ScoringMethod.MAX_GROUP_SIZE]
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
         int(scores[ScoringMethod.MAX_GROUP_SIZE]))
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

    # Compute entropy ranking if needed (transparently loads/saves SQLite)
    if (not soln.scores_updated
            or soln.scores_method != ScoringMethod.ENTROPY_GAIN):
        print("Computing entropy ranking for lookahead...")
        rank_words = (soln.current_words
                      if _is_possible_answers_mode(gs)
                      else gs.all_words)
        tracker = ProgressTracker(len(rank_words))
        soln.compute_scores(
            rank_words,
            ScoringMethod.ENTROPY_GAIN,
            progress_callback=tracker.update,
        )
        tracker.finish()

    top_n = soln.scores[:count]

    # Possible-answers mode: restrict step-2 candidates to the subgroup
    is_possible_answers = _is_possible_answers_mode(gs)
    if is_possible_answers:
        second_step_words = None
        mode_label = "possible answers (subgroup)"
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
            if soln.score_cache:
                subset_key = ScoreCache.encode_subset(subgroup)
                if soln.score_cache.read(subset_key, policy) is not None:
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
        print(f"{soln.scores_method.label}:")
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


def _multistep_stats(word, soln, step2_pool=None, constraint_compliant=False,
                     all_words=None, erd_cache=None):
    """
    Compute 3-step expected entropy and group stats for a single word.
    Returns a dict with keys: step1, step2, step3, max_grp, max_grp2,
    wt_avg, prob_finish, buckets.

    step2_pool: candidate pool for step 2. If None and not constraint_compliant,
        uses only the subgroup (answers-only mode).
    constraint_compliant: if True, compute valid step-2 candidates per subgroup
        by applying the step-1 response constraints to all_words.  Also
        switches the ERD computation to the hard-mode guess vocabulary,
        cached under ERD_CONSTRAINED.  Those values are path-dependent and
        meaningless to other games/positions, so they must never reach the
        persisted cross-game SQLite cache — erd_cache must be a
        MemoryScoreCache (the caller's long-lived, vocabulary-scoped
        gs.constrained_erd_cache when available, else a throwaway one).
    Step 3 always uses subgroup candidates — sub-subgroups are tiny.
    """
    cache = soln.cache
    remaining = soln.current_words
    n = len(remaining)

    _step1_methods = (ScoringMethod.ENTROPY_GAIN, ScoringMethod.MAX_GROUP_SIZE,
                      ScoringMethod.WEIGHTED_AVG, ScoringMethod.PROB_FINISH)
    for m in _step1_methods:
        soln._ensure_scores_loaded(m)

    if cache:
        s1_groups = cache.group_words(word, remaining)
    else:
        s1_groups = defaultdict(list)
        for answer in remaining:
            pat = tuple(calculate_response(word, answer))
            s1_groups[pat].append(answer)

    _cached = soln.word_scores.get(word, {})
    if all(m in _cached for m in _step1_methods):
        step1    = _cached[ScoringMethod.ENTROPY_GAIN]
        max_grp  = int(_cached[ScoringMethod.MAX_GROUP_SIZE])
        wt_avg   = _cached[ScoringMethod.WEIGHTED_AVG]
        prob_fin = _cached[ScoringMethod.PROB_FINISH]
    else:
        group_counts = {p: len(g) for p, g in s1_groups.items()}
        step1    = score_groups(group_counts, ScoringMethod.ENTROPY_GAIN)
        wt_avg   = score_groups(group_counts, ScoringMethod.WEIGHTED_AVG)
        max_grp  = int(score_groups(group_counts, ScoringMethod.MAX_GROUP_SIZE))
        prob_fin = score_groups(group_counts, ScoringMethod.PROB_FINISH)
        soln.word_scores.setdefault(word, {})[ScoringMethod.ENTROPY_GAIN] = step1
        soln.word_scores[word][ScoringMethod.MAX_GROUP_SIZE]      = float(max_grp)
        soln.word_scores[word][ScoringMethod.WEIGHTED_AVG] = wt_avg
        soln.word_scores[word][ScoringMethod.PROB_FINISH]  = prob_fin

    buckets = [0, 0, 0, 0, 0]
    for g in s1_groups.values():
        k = len(g)
        if k == 1:    buckets[0] += 1
        elif k <= 4:  buckets[1] += 1
        elif k <= 9:  buckets[2] += 1
        elif k <= 49: buckets[3] += 1
        else:         buckets[4] += 1

    # For step-2 SQLite caching, hard mode can't be keyed by subgroup key alone
    # because cands2 depends on the specific (word, pattern) constraint set.
    lc = soln.score_cache
    policy = None if constraint_compliant else ('full' if step2_pool is not None else 'hard')

    step2 = 0.0
    step3 = 0.0

    t0 = time.time()
    _prog = {'on': False, 'step3': False}

    for pat, subgroup in s1_groups.items():
        k = len(subgroup)
        if k <= 1:
            continue

        if constraint_compliant and all_words:
            resp = decode_response(pat)
            cands2 = answer_to_restriction(word, resp).apply(all_words)
        elif step2_pool is not None:
            cands2 = step2_pool
        else:
            cands2 = subgroup

        # Check SQLite subgroup cache (shares entries with compute_lookahead)
        subset_key = ScoreCache.encode_subset(subgroup) if (lc and policy) else None
        cache_hit = lc.read(subset_key, policy) if subset_key else None

        if cache_hit is not None:
            best2_word, best2_ent = cache_hit
            # One group computation to get sub-subgroups for step 3
            if cache:
                best2_grps = cache.group_words(best2_word, subgroup)
            else:
                best2_grps = defaultdict(list)
                for ans in subgroup:
                    p2 = tuple(calculate_response(best2_word, ans))
                    best2_grps[p2].append(ans)
        else:
            # Announce progress only if we've been computing more than 5 seconds
            if not _prog['on'] and time.time() - t0 > 5:
                suffix = '2, 3' if _prog['step3'] else '2'
                print(f'  Computing entropy...{suffix}', end='', flush=True)
                _prog['on'] = True

            best2_ent = 0.0
            best2_word = None
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
                    best2_word = c2
                    best2_grps = sg

            if subset_key and best2_word:
                lc.write(subset_key, policy, best2_word, best2_ent)

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

    for m in _step1_methods:
        soln._persist_scores(m)

    # ERD: exact expected guesses for the current candidates mode, surfaced
    # purely from cache — never computed here. Exact ERD search over a large
    # guess vocabulary is combinatorially expensive (a single subgroup can
    # take tens of seconds), and that cost belongs solely to the background
    # ERDWarmer, which uses its own short deadlines to detect when it has
    # bitten off more than it can chew. The foreground must never block on
    # it: each subgroup is an instant cache read, exactly mirroring how
    # print_status's ERD tag already works. A miss means "not cached yet" —
    # reported as unavailable (None) — not a cue to compute it live. As the
    # warmer keeps populating the cache in the background, more of these
    # lookups become hits for free.
    erd = None
    if not soln._is_full_game():
        if constraint_compliant:
            # Hard-mode ERD values are path-dependent (the eligible guess
            # vocabulary depends on the exact constraints accumulated so
            # far) and must never reach the persisted, cross-game SQLite
            # cache. erd_cache (when supplied) is the long-lived,
            # vocabulary-scoped MemoryScoreCache shared with the warmer for
            # this position — surfacing from it lets try-mode see whatever
            # the warmer has already found. A throwaway empty cache is the
            # fallback for callers that don't have one (e.g. tests), which
            # simply means nothing is surfaceable yet.
            erd_policy = ERD_CONSTRAINED
            erd_score_cache = erd_cache if erd_cache is not None else MemoryScoreCache()
        elif all_words:
            erd_policy = ERD_ALL
            erd_score_cache = soln.score_cache
        else:
            erd_policy = ERD_ANSWERS
            erd_score_cache = soln.score_cache
        erd_cost = 1.0
        erd_ok = True
        for subgroup in s1_groups.values():
            k = len(subgroup)
            if k == 0:
                continue
            if k == 1 and subgroup[0] == word:
                continue  # all-green branch: already solved, 0 more guesses
            if k == 1:
                sub_erd = 1.0  # singleton: exactly one more guess, always cached-equivalent
            else:
                hit = erd_score_cache.read(ScoreCache.encode_subset(subgroup), erd_policy)
                sub_erd = hit[1] if hit is not None else None
            if sub_erd is None:
                erd_ok = False
                break
            erd_cost += (k / n) * sub_erd
        if erd_ok:
            erd = erd_cost

    return {
        'step1': step1, 'step2': step2, 'step3': step3,
        'max_grp': max_grp,
        'wt_avg': wt_avg, 'prob_finish': prob_fin,
        'buckets': buckets,
        'erd': erd,
    }


def _compare_words(words, soln, step2_pool=None, constraint_compliant=False,
                   all_words=None, erd_cache=None):
    """Compare 2–4 words side by side."""
    n = len(soln.current_words)

    print(f'\n  Computing {", ".join(w.upper() for w in words)}...', flush=True)
    all_stats = []
    for i, w in enumerate(words):
        if len(words) > 1:
            print(f'  [{i + 1}/{len(words)}] {w.upper()}', flush=True)
        all_stats.append(_multistep_stats(w, soln, step2_pool, constraint_compliant,
                                           all_words, erd_cache))

    lw = 9  # "Entropy 1" = 9, "10-49:" = 6

    # Build all data rows up front so we can measure max column width
    totals = [s['step1'] + s['step2'] + s['step3'] for s in all_stats]
    erd_vals = [s.get('erd') for s in all_stats]
    data_rows = [
        ('Wt avg',    [s['wt_avg']      for s in all_stats], '{:.2f}', False),
        ('Max grp',   [s['max_grp']     for s in all_stats], '{:d}',   False),
        ('Solve%',    [s['prob_finish'] for s in all_stats], '{:.1%}', True),
    ]
    if any(v is not None for v in erd_vals):
        data_rows.append(('ERD', erd_vals, '{:.3f}', False))
    data_rows.append(('Entropy 1', [s['step1'] for s in all_stats], '{:.4f}', True))
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
    cw = max(len(w) + 1 for w in words)  # +1 for the answer-set '*' marker
    for _, values, fmt, _ in data_rows + bucket_rows:
        for v in values:
            if v is not None:
                cw = max(cw, len(fmt.format(v)))

    def print_row(label, values, fmt, higher_better=True):
        padded = [('—'.rjust(cw) if v is None else fmt.format(v).rjust(cw))
                  for v in values]
        valid = [v for v in values if v is not None]
        if len(valid) < 2 or all(v == valid[0] for v in valid):
            all_tied = True
            best = None
        else:
            best = max(valid) if higher_better else min(valid)
            all_tied = False
        print(f'  {label:<{lw}}  ', end='')
        for i, (p, v) in enumerate(zip(padded, values)):
            if i:
                print('  ', end='')
            if not all_tied and v is not None and v == best:
                with colored_text('green'):
                    print(p, end='')
            else:
                print(p, end='')
        print()

    print(f'\n  {n:,} words:')
    print(f'  {"":>{lw}}  '
          + '  '.join((w.upper() + _mark(w)).rjust(cw) for w in words))

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

    # Derive step-2 pool and hard-mode flag from the current grid selection
    erd_cache = None
    if _is_hard_mode(gs):
        step2_pool = None
        constraint_compliant = True
        # Reuse the long-lived, vocabulary-scoped hard-mode ERD cache so
        # try-mode and the background warmer trade sub-results for free —
        # both use the exact same eligible-guess vocabulary for this position.
        eligible = soln.constraint_compliant_words(gs.all_words)
        gs.constrained_erd_cache.set_scope(
            MemoryScoreCache.fingerprint_vocabulary(eligible))
        erd_cache = gs.constrained_erd_cache
    elif _is_possible_answers_mode(gs):
        step2_pool = None
        constraint_compliant = False
    else:  # unfiltered (all words, or answer-shaped words) — cap at 200
           # top-entropy; searching all 12k is ~65x slower, and even the
           # ~3,200-word answer-shaped vocabulary benefits from the cap
        constraint_compliant = False
        if (soln.scores_updated
                and soln.scores_method == ScoringMethod.ENTROPY_GAIN):
            step2_pool = [w for w, _ in soln.scores[:200]]
        else:
            step2_pool = soln.all_answers

    words = line.lower().split()
    try:
        if 2 <= len(words) <= 4:
            assert all(len(w) == 5 for w in words)
            _compare_words(words, soln, step2_pool, constraint_compliant, gs.all_words, erd_cache)
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
            st = _multistep_stats(word, soln, step2_pool, constraint_compliant,
                                  gs.all_words, erd_cache)
            mode = ('hard mode' if constraint_compliant
                    else (f'top {len(step2_pool)}' if step2_pool else 'possible answers'))
            print(f'\n  Multi-step lookahead ({mode}):')
            total = st['step1'] + st['step2'] + st['step3']
            erd   = st.get('erd')
            chain_vals = [st['step1'], st['step2'], st['step3'], total]
            _vw = max(len(f'{v:.4f}') for v in chain_vals)
            _rows = []
            if erd is not None:
                _rows.append(('ERD:', f'{erd:>{_vw}.3f} exp remaining depth'))
            _rows += [
                ('Entropy 1:', f'{st["step1"]:>{_vw}.4f}'),
                ('+ ent. 2:',  f'{st["step2"]:>{_vw}.4f}'),
                ('+ ent. 3:',  f'{st["step3"]:>{_vw}.4f}'),
                ('Total:',     f'{total:>{_vw}.4f}'),
            ]
            _lw = max(len(lbl) for lbl, _ in _rows)
            for lbl, val in _rows:
                print(f'    {lbl:<{_lw}}  {val}')

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
            Solution(gs.all_answers, gs.all_words,
                     gs.cache, gs.score_cache)
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

def cmd_candidates(gs):
    universe_options = [
        (GuessUniverse.ALL_WORDS,   "all words (~12,972)"),
        (GuessUniverse.ALL_ANSWERS, "possible answers (~3,200)"),
    ]
    compliance_options = [
        (ComplianceFilter.UNFILTERED, "unfiltered"),
        (ComplianceFilter.COMPLIANT,  "must satisfy revealed clues"),
    ]
    print(f"\nCandidates mode (current: {_candidate_label(gs.universe, gs.compliance)}):")

    print("Draw guesses from:")
    for i, (_, label) in enumerate(universe_options, 1):
        print(f"  {i}. {label}")
    print("Choose (1-2)? ", end="")
    try:
        universe = universe_options[int(input().strip()) - 1][0]
    except (ValueError, IndexError):
        print_error("Invalid choice.")
        return

    print("Must they satisfy every clue revealed so far?")
    for i, (_, label) in enumerate(compliance_options, 1):
        print(f"  {i}. {label}")
    print("Choose (1-2)? ", end="")
    try:
        compliance = compliance_options[int(input().strip()) - 1][0]
    except (ValueError, IndexError):
        print_error("Invalid choice.")
        return

    gs.universe, gs.compliance = universe, compliance
    print(f"  Candidates: {_candidate_label(universe, compliance)}")


# ---------------------------------------------------------------------------
# Command: Help
# ---------------------------------------------------------------------------

def cmd_help(gs):
    cand = _candidate_label(gs.universe, gs.compliance)
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
    lc = gs.score_cache
    la_rows, ws_rows, rd_rows, mtime = lc.stats()
    cache_ts = (datetime.utcfromtimestamp(mtime).isoformat() + "Z") if mtime else "n/a"
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
  c = Candidates: {cand}
  ? = This help

  Cache: {gs.score_cache_path}
    {la_rows:,} subgroup picks, {ws_rows:,} word scores, {rd_rows:,} decompositions, last write {cache_ts}
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
    'c': cmd_candidates,
    '?': cmd_help,
}


def print_status(gs, warmer=None):
    """Print current game status.

    warmer: the live ERDWarmer for the current position, if any — passed
            through so the ERD progress tag can be validated against the
            position actually being displayed (see _erd_root_progress_tag).
    """
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
            erd_tag = ''
            if not soln._is_full_game():
                erd_sc, erd_pol = _erd_cache_and_policy(gs, soln)
                if erd_sc is not None:
                    hit = erd_sc.read(ScoreCache.encode_subset(words), erd_pol)
                    if hit is not None:
                        erd_tag = f'  [ERD: {hit[1]:.3f}]'
                    else:
                        erd_tag = _erd_root_progress_tag(warmer, soln)
            print(f"{n:,} words remaining")
            if erd_tag:
                print(erd_tag.strip())
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


# ---------------------------------------------------------------------------
# ERD background warmer
# ---------------------------------------------------------------------------

class ERDWarmer(threading.Thread):
    """Daemon thread: fills the ERD cache proactively for the current position.

    Runs while the main thread blocks on input(), giving background ERD
    computation at zero latency cost to the user.

    persist=True  (ANY_WORD / POSSIBLE_ANSWERS): uses a private SQLite
                  ScoreCache connection; results survive across sessions.
    persist=False (CONSTRAINT_COMPLIANT): uses the caller-supplied
                  seed_mem_cache (a long-lived, vocabulary-scoped
                  MemoryScoreCache — see GameState.constrained_erd_cache).
                  Results are transient because the eligible guess set is
                  path-dependent, but the cache itself survives across runs:
                  its internal scoping makes stale entries unreachable and
                  recurring vocabularies reusable, with no caller bookkeeping.

    policy must be passed explicitly: ERD_ALL, ERD_ANSWERS, or ERD_CONSTRAINED.
    """

    def __init__(self, current_words, all_answers, effective_guesses,
                 score_cache_path, response_cache,
                 policy=ERD_ALL, persist=True, seed_mem_cache=None):
        super().__init__(daemon=True, name='ERDWarmer')
        self._words = list(current_words)        # snapshot
        self._all_answers = all_answers
        self._effective_guesses = effective_guesses
        self._cache_path = score_cache_path
        self._rcache = response_cache
        self._policy = policy
        self._persist = persist
        self._seed_mem_cache = seed_mem_cache
        self._cancel = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # initially running (not paused)
        self.root_done = 0        # candidates fully scored so far in the root scan
        self.root_total = 0       # size of the root scan; 0 means not started yet; -1 = trivial
        self.root_best = None     # (word, erd) — best candidate found so far

    def stop(self):
        self._cancel.set()
        self._paused.set()  # unblock any paused wait so the thread can see _cancel

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def run(self):
        if len(self._words) <= 2:
            self.root_total = -1  # sentinel: trivial position, no scan needed
            return
        if self._persist:
            score_cache = ScoreCache(self._cache_path, self._all_answers)
        else:
            score_cache = self._seed_mem_cache
        # The response_cache supplied at construction (gs.cache) wraps a
        # SQLite connection opened on the main thread — sqlite3 connections
        # cannot be shared across threads. Replace it with one private to
        # this thread, backed by this thread's own connection (persist=True)
        # or none at all (persist=False — hard mode's eligible-guess
        # subgroups are small enough that recomputing decompositions
        # in-memory costs nothing noticeable).
        self._rcache = ResponseCache(self._all_answers,
                                     score_cache if self._persist else None)
        try:
            self._warm(score_cache)
        finally:
            if self._persist:
                score_cache.close()

    def _ranked_root_guesses(self):
        """Order root-scan candidates so a still-incomplete scan's running
        best is actually meaningful, instead of an artifact of the guess
        list's (alphabetical) file order.

        Sorted by max-group-size ascending, then entropy descending: a
        small worst-case group bounds how badly a guess can turn out, which
        is the property ERD itself is fundamentally about, so this ordering
        tracks ERD quality far more closely than alphabetical position does.
        Entropy breaks ties among equal worst-cases — a sharper distribution
        elsewhere in the partition tends to retire more words sooner. With
        this order, "best so far" early in the scan is a real signal, not
        a coin flip, and the true winner tends to surface early too.
        """
        methods = [ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN]
        scored = []
        for word in self._effective_guesses:
            if self._cancel.is_set():
                return self._effective_guesses
            if not self._paused.is_set():
                self._paused.wait()      # block while main thread is busy
            if self._cancel.is_set():
                return self._effective_guesses
            counts = self._rcache.group_counts(word, self._words)
            scores = score_groups_multi(counts, methods)
            scored.append((scores[ScoringMethod.MAX_GROUP_SIZE],
                           -scores[ScoringMethod.ENTROPY_GAIN],
                           word))
        scored.sort()
        return [word for _, _, word in scored]

    def _on_root_progress(self, done, total, best_word, best_erd):
        if self._cancel.is_set():
            return
        self.root_done = done
        self.root_total = total
        self.root_best = (best_word, best_erd)

    def _warm(self, score_cache):
        policy = self._policy
        root_key = ScoreCache.encode_subset(self._words)
        if score_cache.read(root_key, policy) is not None:
            return  # already done from a prior session

        ranked_guesses = self._ranked_root_guesses()
        if self._cancel.is_set():
            return

        # Run the root scan directly.  Uncached subgroups are computed
        # on-the-fly and written to the cache as a side effect (via
        # cache_all_scores), so the cache fills progressively during the
        # scan.  No pre-caching phase needed: the root scan is its own
        # bottom-up fill, and progress is visible immediately.
        #
        # The scan may be paused mid-run when the main thread starts a
        # foreground operation.  cancel_check returns True on both true
        # cancellation and on pause, causing min_expected_guesses to abort
        # and return None.  After a pause, we wait for resume and retry
        # from the top — partial results already written to score_cache
        # survive the abort, so the retry picks up from cache rather than
        # starting over.
        def _cancel_or_paused():
            return self._cancel.is_set() or not self._paused.is_set()

        while True:
            self.root_done = 0
            self.root_total = 0
            self.root_best = None
            result = min_expected_guesses(
                self._words, self._rcache, score_cache,
                guesses=ranked_guesses,
                policy=policy,
                progress_callback=self._on_root_progress,
                cancel_check=_cancel_or_paused,
            )
            if result is not None:
                if not self._cancel.is_set():
                    print(f'\n  [ERD ready: {result:.3f} expected remaining depth]',
                          flush=True)
                return
            if self._cancel.is_set():
                return
            # Paused — wait for the main thread to finish its operation.
            self._paused.wait()
            if self._cancel.is_set():
                return


def main():
    print(f"wordle.py {BUILD}")
    all_answers = load_word_list(ANSWER_FILE)
    all_words = load_word_list(WORDS_FILE)
    print(f"Loaded {len(all_answers):,} answers, "
          f"{len(all_words):,} guesses.")

    gs = GameState(all_answers, all_words)

    print_status(gs)
    _warmer = None
    _warmer_key = None
    while True:
        # Keep a warmer running for the current game position across REPL
        # turns — tearing it down and rebuilding it on every keystroke (even
        # when the branch hasn't changed) starves it of the very thing that
        # makes it useful: minutes of uninterrupted background time to build
        # out the ERD cache.  It also raced a fresh warmer against the old
        # one's still-finishing root computation, producing duplicate
        # "[ERD ready]" announcements for the same value.  Replace it only
        # when the branch actually changes (or the game ends); otherwise let
        # it keep going while input() blocks.
        if gs.single and not gs.solutions[0]._is_full_game():
            soln0 = gs.solutions[0]
            cfg = _erd_mode_config(gs)
            effective_guesses = cfg.guesses_fn(gs, soln0)
            branch_key = _warmer_branch_key(gs, soln0, effective_guesses)
            if _warmer is None or not _warmer.is_alive() or branch_key != _warmer_key:
                if _warmer is not None:
                    _warmer.stop()
                seed_cache = None
                if _is_hard_mode(gs):
                    # Hard-mode ERD results are valid only for the exact
                    # eligible-guess vocabulary that produced them.  Scope the
                    # long-lived cache to that vocabulary's fingerprint:
                    # entries from other vocabularies become invisible (no
                    # false hits across undo / reset / replay), while a
                    # recurring vocabulary's entries become reusable again
                    # automatically — no eviction needed.
                    gs.constrained_erd_cache.set_scope(branch_key[2])
                    seed_cache = gs.constrained_erd_cache
                _warmer = ERDWarmer(
                    soln0.current_words,
                    gs.all_answers,
                    effective_guesses,
                    gs.score_cache_path,
                    gs.cache,
                    policy=cfg.policy,
                    persist=cfg.persist,
                    seed_mem_cache=seed_cache,
                )
                _warmer.start()
                _warmer_key = branch_key
        elif _warmer is not None:
            _warmer.stop()
            _warmer = None
            _warmer_key = None

        print(f"\nCommand (gsbldtixurawc?)? ", end="")
        try:
            cmd = input().strip()
        except EOFError:
            if _warmer is not None:
                _warmer.stop()
            print()
            print("Exiting.")
            break
        except KeyboardInterrupt:
            if _warmer is not None:
                _warmer.stop()
            print()
            print("Interrupted.")
            break

        if _warmer is not None:
            gs.last_erd_root_progress = (_warmer.root_done, _warmer.root_total,
                                         _warmer.root_best)

        if not cmd:
            print_status(gs, _warmer)
            continue
        handler = COMMANDS.get(cmd[0])
        if handler:
            try:
                inline = cmd[1:].strip()
                if _warmer is not None:
                    _warmer.pause()
                try:
                    if inline and handler is cmd_test:
                        cmd_test(gs, inline)
                    else:
                        handler(gs)
                finally:
                    if _warmer is not None:
                        _warmer.resume()
            except Exception as e:
                print_error(f"Error: {e}")
                raise
        else:
            print_error(f"Unknown: {cmd}")
        print_status(gs, _warmer)


if __name__ == '__main__':
    main()
