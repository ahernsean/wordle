"""
wordle_engine.py - Core algorithms for Wordle solving.

No UI dependencies. All display/interaction is handled by the caller.
"""

import math
import time
from collections import defaultdict, OrderedDict
from enum import Enum, auto

from cache_sqlite import ScoreCache
from erd_lattice import ERD_LATTICE_NOISE_MARGIN, erd_numerator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoringMethod(Enum):
    WEIGHTED_AVG = auto()     # sum(n_i^2) / N  — lower is better
    ENTROPY_GAIN = auto()     # Shannon entropy (bits of info gained)
    MAX_GROUP_SIZE = auto()   # max(n_i)
    PROB_FINISH = auto()      # P(next guess solves it)

    @property
    def label(self):
        labels = {
            ScoringMethod.WEIGHTED_AVG:   "Weighted avg remaining",
            ScoringMethod.ENTROPY_GAIN:   "Entropy gain (bits)",
            ScoringMethod.MAX_GROUP_SIZE:        "Worst-case group size",
            ScoringMethod.PROB_FINISH:    "P(finish next turn)",
        }
        return labels[self]

    @property
    def higher_is_better(self):
        return self in (ScoringMethod.ENTROPY_GAIN,
                        ScoringMethod.PROB_FINISH)

    def sort_key(self):
        """Return a sort key function: best scores first."""
        if self.higher_is_better:
            return lambda x: -x[1]
        return lambda x: x[1]

    def format_score(self, value):
        """Format a score value for display."""
        if self == ScoringMethod.MAX_GROUP_SIZE:
            return str(int(value))
        if self == ScoringMethod.PROB_FINISH:
            return f'{value:.2%}'
        return f'{value:0.4f}'


class GuessUniverse(Enum):
    """Which static word list a candidate guess is drawn from.

    Independent of ComplianceFilter below: a player can pick a guess from
    either list whether or not it satisfies the clues revealed so far
    (e.g. a recognizable answer-shaped word that the clues have already
    eliminated is still a legal — if unwise — guess in normal play).
    """
    ALL_WORDS = 'words'      # every word the puzzle accepts
    ALL_ANSWERS = 'answers'  # words that can ever be a Wordle answer


class ComplianceFilter(Enum):
    """Whether a candidate must satisfy every clue revealed so far.

    Independent of GuessUniverse above — see its docstring. COMPLIANT is
    real Wordle hard mode's rule (green letters fixed in position, yellow
    letters present somewhere) applied to whichever universe is selected.
    """
    UNFILTERED = 'unfiltered'
    COMPLIANT = 'compliant'


# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

GAME_GUESSES = 6   # a Wordle game allows 6 guesses total; the root's budget


# ---------------------------------------------------------------------------
# ERD cache policy names — distinguish the guess-vocabulary namespaces under
# which min_expected_guesses results are stored. Each name spells out both
# the GuessUniverse and ComplianceFilter it was computed under, so namespaces
# can be told apart on sight rather than by tribal knowledge of which axis a
# bare word like "answers" or "constrained" was meant to encode.
# ---------------------------------------------------------------------------

def erd_policy_name(universe, compliance):
    """Derive the cache-namespace string for a (universe, compliance) pair."""
    return f'erd_{universe.value}_{compliance.value}'


ERD_ALL = erd_policy_name(GuessUniverse.ALL_WORDS, ComplianceFilter.UNFILTERED)
# any word may be guessed, no clue filter (SQLite, persisted)
ERD_CONSTRAINED = erd_policy_name(GuessUniverse.ALL_WORDS, ComplianceFilter.COMPLIANT)
# Wordle hard mode (in-memory, transient — path-dependent, never persisted)
ERD_ANSWERS = erd_policy_name(GuessUniverse.ALL_ANSWERS, ComplianceFilter.COMPLIANT)
# guesses restricted to possible answers (SQLite, persisted)
ERD_ANSWERS_UNFILTERED = erd_policy_name(GuessUniverse.ALL_ANSWERS, ComplianceFilter.UNFILTERED)
# guesses drawn from the answer-shaped list, no clue filter (SQLite, persisted)

VALID_ERD_POLICIES = frozenset(
    {ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED, ERD_ANSWERS_UNFILTERED})


# ---------------------------------------------------------------------------
# Result status codes for _solve_subset and evaluate_candidate
#
# Module-level int constants (not IntEnum) avoid dict-lookup overhead on the
# hot path, where these are compared millions of times per solve.
# ---------------------------------------------------------------------------

DEADLINE_EXCEEDED = 0   # wall-clock deadline passed
CANCEL_RECVD      = 1   # external stop signal received
SOLVED            = 2   # exact optimum found
OVER_DEPTH_BUDGET = 3   # depth budget too small — proven no winning strategy
OVER_ERD_LIMIT    = 4   # every candidate >= ERD bound — lower bound only, do not cache
NO_INFORMATION_GAINED = 5  # a response group is the whole branch: split gained nothing

_ABORT_STATUSES = frozenset({DEADLINE_EXCEEDED, CANCEL_RECVD})


# Admissible ceiling on how many answer words any strategy can guarantee to
# resolve within a given guess budget.  A guess yields at most 3**5 = 243
# distinct response patterns; the all-green pattern resolves the guessed word
# itself, and each of the other 242 patterns leads to a group that must be
# resolved within budget-1.  Hence M(b) = 1 + 242*M(b-1), M(0) = 0.  A branch
# holding more words than M(budget) cannot be solved within budget — a
# never-false loss certificate.  The bound is sound but weak: M(2) = 243 and
# M(3) = 58,807, so it fires only for very large branches and never at
# budget >= 3 over a few-thousand-word universe.  Structural losses (specific
# near-identical words a single guess cannot separate) sit far below it and
# can only be certified by exhaustive search.
_MAX_SOLVABLE_WITHIN = [0]


def max_solvable_within(budget):
    """Upper bound on answer words solvable within `budget` guesses (see above)."""
    while len(_MAX_SOLVABLE_WITHIN) <= budget:
        _MAX_SOLVABLE_WITHIN.append(1 + 242 * _MAX_SOLVABLE_WITHIN[-1])
    return _MAX_SOLVABLE_WITHIN[budget]


# ---------------------------------------------------------------------------
# Word list loading
# ---------------------------------------------------------------------------

def load_word_list(filepath):
    """Load a newline-delimited word list from a file."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Response calculation
# ---------------------------------------------------------------------------

def calculate_response(test_word, answer_word):
    """
    Compute Wordle color response for a guess against an answer.

    Returns a list of 5 strings: 'green', 'yellow', or 'gray'.

    Uses two passes (greens first, then yellow/gray) to correctly
    handle duplicate letters.
    """
    work = list(answer_word)
    response = []

    # Pass 1: mark greens and consume matched answer positions
    for pos, letter in enumerate(test_word):
        if work[pos] == letter:
            work[pos] = "_"

    # Pass 2: yellow and gray
    for pos, letter in enumerate(test_word):
        if work[pos] == "_":
            response.append("green")
        elif letter in work:
            response.append("yellow")
            ind = work.index(letter)
            work[ind] = " "
        else:
            response.append("gray")

    return response


_RESPONSE_VALUES = {'gray': 0, 'yellow': 1, 'green': 2}
_RESPONSE_NAMES = {0: 'gray', 1: 'yellow', 2: 'green'}


def _encode_response(response):
    """Encode a response list as an integer 0-242 (base 3)."""
    code = 0
    for r in response:
        code = code * 3 + _RESPONSE_VALUES[r]
    return code


def decode_response(code):
    """Decode an integer 0-242 back to a response list."""
    result = []
    for _ in range(5):
        result.append(_RESPONSE_NAMES[code % 3])
        code //= 3
    return result[::-1]


# All-green ("GGGGG") is the unique pattern produced only when guess==answer,
# so a group keyed on this pattern is always exactly {guess} when present —
# used by min_expected_guesses to detect the "self" group at O(1).
_ALL_GREEN_PATTERN = _encode_response(['green'] * 5)

# Best-first candidate ordering kicks in only for nodes at least this large.
# Ordering scans the full candidate vocab once (O(|vocab|*n)) per node; below
# this size the subtree it would prune is too small to repay that scan.  Tuned
# (=8) from a repeated sweep over branch sizes 48/81/146: a low gate helps cheap
# branches slightly but *hurts* hard ones 11-14% (they recurse through many
# small nodes, so the per-node scan overhead dominates).  8 minimises total wall
# across the mix; the optimum genuinely varies by branch difficulty (see notes).
ORDER_MIN_N = 8

# Alpha-beta ceiling slack.  When a branch is solved with a derived ceiling we
# add this tiny margin so floating-point rounding in the ceiling arithmetic can
# never cut off a candidate that would in fact (just barely) beat the bound — a
# false cutoff would only cost a missed cache write, never correctness, but the
# margin makes "never wrongly discard a true winner" exact rather than probable.
_CEIL_EPS = 1e-9


def erd_display_numerator(value, n_answers, *, ceiling=False):
    """Return the unreduced ERD lattice numerator suitable for display.

    Exact ERDs must lie on the branch's ``1 / n_answers`` lattice.  A derived
    alpha-beta ceiling carries ``_CEIL_EPS`` above its semantic lattice value,
    so its display check also admits that explicitly bounded offset.
    """
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    allowed_noise = ERD_LATTICE_NOISE_MARGIN
    if ceiling:
        allowed_noise += n_answers * _CEIL_EPS
    return erd_numerator(value, n_answers, noise_margin=allowed_noise)


# ---------------------------------------------------------------------------
# Restriction system
# ---------------------------------------------------------------------------

class Restriction:
    """Encapsulates the filtering constraints derived from a guess+response."""

    def __init__(self):
        self.letters = ['', '', '', '', '']
        self.count = [0, 0, 0, 0, 0]
        self.color = [None, None, None, None, None]

    def __setitem__(self, index, tup):
        letter, count, color = tup
        self.letters[index] = letter
        self.count[index] = count
        self.color[index] = color

    def __getitem__(self, index):
        return [self.letters[index], self.count[index], self.color[index]]

    def apply(self, words):
        """Apply all five position constraints to filter a word list."""
        for i in range(5):
            words = _perform_restriction(
                words, i, self.letters[i], self.count[i], self.color[i]
            )
        return words


def answer_to_restriction(guess, answer):
    """
    Build a Restriction from a guess word and its color response.

    Processes green, then yellow, then gray to correctly track
    how many instances of each letter have been accounted for.
    """
    restriction = Restriction()
    count = defaultdict(int)
    for color in ["green", "yellow", "gray"]:
        for pos, (letter, ans) in enumerate(zip(guess, answer)):
            if ans == color:
                restriction[pos] = [letter, count[letter], color]
                count[letter] += 1
    return restriction


def _ignore_word(_word, _pos, _letter, _ignore):
    """
    Mask out `_ignore` occurrences of `_letter` in `_word`, skipping
    position `_pos`. Used to handle duplicate-letter logic.
    """
    new = list(_word)
    letter_count = _ignore
    for pos, letter in enumerate(_word):
        if pos == _pos:
            continue
        if letter == _letter and letter_count > 0:
            new[pos] = "_"
            letter_count -= 1
    return ''.join(new)


def _perform_restriction(words, pos, letter, ignore, answer):
    """Filter words by a single position's constraint."""
    new_words = []
    for word in words:
        ignored = _ignore_word(word, pos, letter, ignore)
        keep = False
        if answer == "green":
            keep = (ignored[pos] == letter)
        elif answer == "gray":
            keep = (letter not in ignored)
        elif answer == "yellow":
            keep = (letter in ignored and ignored[pos] != letter)
        if keep:
            new_words.append(word)
    return new_words


def apply_guess(words, try_word, response):
    """Filter a word list by applying a guess and its color response."""
    restriction = answer_to_restriction(try_word, response)
    return restriction.apply(words)


def enumerate_branches(words, guess_word):
    """Partition `words` by every non-winning response to `guess_word`.

    Returns [(response_code, branch_words), ...] for response codes 0-241
    (242 == all-green == the win, excluded) whose branch has >= 2 words —
    smaller branches need no ERD entry (a singleton is solved next guess
    by definition).
    """
    branches = []
    for code in range(242):
        branch_words = apply_guess(words, guess_word, decode_response(code))
        if len(branch_words) >= 2:
            branches.append((code, branch_words))
    return branches


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def calculate_group_counts(test_word, words):
    """
    Simulate guessing `test_word` against every word in `words`.
    Returns a dict mapping response pattern codes to their counts.
    """
    groups = defaultdict(int)
    for word in words:
        response = calculate_response(test_word, word)
        groups[_encode_response(response)] += 1
    return groups


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

# ResponseCache.group_words's fast-path promotion threshold: a cold guess (no
# decoded blob yet) reused this many times via the fast path pays the one-time
# decode cost and switches to the warm loop for good. Derived from the ratio
# of a decode's one-time cost (~5ms) to the fast path's per-call NumPy
# overhead over an already-decoded guess (~15-25us): a threshold near that
# ratio bounds the worst case (single-use vs. heavily-reused guesses) at
# roughly twice the cost either extreme would incur alone (a ski-rental
# problem). Verified against diag_kernel_bench.py at both small (n=30, low
# per-guess reuse) and large, deep-recursion (n=500, extremely high per-guess
# reuse) branch sizes — controlled A/B sweeps across 40-1000 and an uncapped
# fast path all perform equivalently in both regimes, so this sits
# comfortably in that range without needing finer tuning.
_GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD = 250


# Branch ERD floors held per cache instance. Each entry is one key plus a
# float, so the ceiling is a bounded few tens of MB at the branch sizes the
# bound is worth computing for.
_BRANCH_FLOOR_CAPACITY = 200_000


class ResponseCache:
    """Lazily caches word-to-pattern mappings for each guess word.

    For each guess word G, stores a compact bytes blob giving every answer
    word's encoded response pattern (int 0-242), one byte per answer in
    canonical (self.answer_words) order. This is built once per guess word
    on first access, then all subsequent scoring against any subset is just
    an index lookup (via _answer_index) and counting.

    A {answer: pattern} dict per guess costs ~100KB (Python dict/str/int
    overhead dominates the single byte of real information per entry).
    Scoring the full guess vocabulary (~13,000 words) against any position
    would then build a ~1.3GB cache — enough to OOM a memory-constrained
    host. The bytes blob costs ~3.2KB per guess instead, ~32x smaller, with
    the same O(1) lookup via the shared _answer_index.
    """

    def __init__(self, answer_words, score_cache=None):
        self.answer_words = answer_words
        self.score_cache = score_cache
        self._answer_index = {word: i for i, word in enumerate(answer_words)}
        self._cache = {}   # guess → bytes (pattern_int per answer, canonical order)
        self._fast_path_use_counts = defaultdict(int)  # cold guess → group_words fast-path tally

    def branch_cost_lower_bound(self, subset, candidate_pool):
        """Admissible floor on subset's ERD over candidate_pool — the reference
        implementation of PatternMatrix.branch_cost_lower_bound.

        Every candidate costs at least 3 - (group_count + has_self)/k, so the
        widest split any pool word achieves gives a floor no strategy playing
        from that pool can beat.  Never below the all-singletons floor.

        Pure: the result depends only on the arguments.  Memoizing belongs to
        BranchFloorTable, which holds one pool for its lifetime — a floor
        cached here could otherwise be served to a search allowed to play words
        the pool it was computed from did not contain.
        """
        size = len(subset)
        if size <= 1:
            return float(size)
        subset_words = set(subset)
        widest_split = 0
        for guess in candidate_pool:
            split = len(self.group_counts(guess, subset))
            if guess in subset_words:
                split += 1
            if split > widest_split:
                widest_split = split
        return max(all_singletons_floor(size), 3.0 - widest_split / size)

    def _ensure(self, guess):
        """Build the mapping for guess if not cached, persisting/reloading via SQLite.

        The decomposition (guess -> pattern bytes) is the same for every
        position in every session against this answer universe, so it is
        cached to disk as a compact byte blob — one byte per answer word, in
        canonical (self.answer_words) order — keyed only by guess+universe.
        """
        if guess in self._cache:
            return
        blob = self._load_decomposition(guess)
        if blob is None:
            blob = bytes(
                _encode_response(calculate_response(guess, answer))
                for answer in self.answer_words
            )
            self._store_decomposition(guess, blob)
        self._cache[guess] = blob

    def _load_decomposition(self, guess):
        if not self.score_cache:
            return None
        return self.score_cache.read_decomposition(guess)

    def _store_decomposition(self, guess, blob):
        if not self.score_cache:
            return
        self.score_cache.write_decomposition(guess, blob)

    def group_counts(self, guess, subset):
        """Return {pattern_int: count} for guess vs subset."""
        self._ensure(guess)
        blob = self._cache[guess]
        counts = defaultdict(int)
        for word in subset:
            idx = self._answer_index.get(word)
            if idx is not None:
                counts[blob[idx]] += 1
            else:
                resp = calculate_response(guess, word)
                counts[_encode_response(resp)] += 1
        return counts

    def group_words(self, guess, subset, *, pattern_matrix=None, branch_indices=None):
        """Return {pattern_int: [words]} for guess vs subset.

        When pattern_matrix and branch_indices are both supplied (branch_indices
        index-aligned with subset via pattern_matrix.answer_indices) and guess
        is not already warm (is_cached), takes PatternMatrix.group_words's fast
        path — one matrix row read plus a stable argsort — for a dict identical
        to the loop below in keys, values, and iteration order. A warm guess
        always takes the loop instead: its decode cost is already sunk, so the
        loop beats the fast path's per-call NumPy overhead at every branch
        size. A cold guess reused past
        _GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD fast-path calls is promoted
        — decoded once via _ensure, then warm (and on the loop) for every
        subsequent call — so a guess genuinely used only a handful of times
        (the common case) keeps the fast path's full ~250x edge over a cold
        decode, while a guess deep recursion reuses thousands of times across
        many small sub-branches doesn't keep paying the fast path's per-call
        overhead indefinitely (see PR #94 review).

        Interactive-mode words outside the answer universe can't be expressed
        as branch_indices, so those callers pass neither and take the loop; a
        guess outside the matrix's guess vocabulary falls through to the loop
        the same way.
        """
        if (pattern_matrix is not None and branch_indices is not None
                and guess not in self._cache):
            self._fast_path_use_counts[guess] += 1
            if (self._fast_path_use_counts[guess]
                    <= _GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD):
                try:
                    return pattern_matrix.group_words(guess, subset, branch_indices)
                except KeyError:
                    pass  # guess outside the matrix vocabulary: the loop handles any word
        self._ensure(guess)
        blob = self._cache[guess]
        groups = defaultdict(list)
        for word in subset:
            idx = self._answer_index.get(word)
            if idx is not None:
                groups[blob[idx]].append(word)
            else:
                resp = calculate_response(guess, word)
                groups[_encode_response(resp)].append(word)
        return groups

    def is_cached(self, guess):
        """Check if a guess word's decoded response blob is already cached.

        This is group_words's own warm/cold dispatch predicate: a guess
        resolved so far entirely through group_words's fast path (see
        _GROUP_WORDS_FAST_PATH_PROMOTION_THRESHOLD) does not populate
        self._cache and so is not "cached" by this method until it warms.
        """
        return guess in self._cache


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_groups(groups, method=ScoringMethod.ENTROPY_GAIN):
    """
    Score a word's group partition.

    - WEIGHTED_AVG: sum(n_i^2)/N. Lower is better.
    - ENTROPY_GAIN: Shannon entropy in bits. Higher is better.
    - MAX_GROUP_SIZE: max(n_i). Lower is better.
    - PROB_FINISH: fraction of remaining words in size-1
      groups (game ends next turn). Higher is better.
    """
    if not groups:
        if method.higher_is_better:
            return 0.0
        return float('inf')

    sizes = list(groups.values())
    n = sum(sizes)

    if method == ScoringMethod.WEIGHTED_AVG:
        return sum(s * s for s in sizes) / n

    elif method == ScoringMethod.ENTROPY_GAIN:
        entropy = 0.0
        for s in sizes:
            p = s / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    elif method == ScoringMethod.MAX_GROUP_SIZE:
        return max(sizes)

    elif method == ScoringMethod.PROB_FINISH:
        singles = sum(1 for s in sizes if s == 1)
        return singles / n

    raise ValueError(f"Unknown scoring method: {method}")


def score_groups_multi(groups, methods):
    """Score a word's group partition under multiple methods at once."""
    return {m: score_groups(groups, m) for m in methods}


def score_word(word, remaining_words, method=ScoringMethod.ENTROPY_GAIN,
               progress_callback=None, cache=None):
    """Score a single candidate guess against the remaining answer words."""
    if progress_callback:
        progress_callback()
    if cache:
        groups = cache.group_counts(word, remaining_words)
    else:
        groups = calculate_group_counts(word, remaining_words)
    return score_groups(groups, method)


def score_word_multi(word, remaining_words, methods,
                     progress_callback=None, cache=None):
    """
    Score a single candidate guess under multiple methods.
    Computes group counts once, then scores with each method.
    """
    if progress_callback:
        progress_callback()
    if cache:
        groups = cache.group_counts(word, remaining_words)
    else:
        groups = calculate_group_counts(word, remaining_words)
    return score_groups_multi(groups, methods)


def cache_all_scores(candidate, branch_words, score_cache, branch_key, cache=None):
    """Derive and persist every ScoringMethod's view of `candidate` against `branch_words`.

    score_groups' methods all read off the same group-count partition, so
    once that partition is in hand for one method, the rest are nearly free
    to derive — persisting only the method an algorithm happened to be
    ranking by would be an arbitrary cutoff.

    This is the ONE place that enumerates ScoringMethod for caching purposes.
    Callers (compute_lookahead, min_expected_guesses, ...) just say "this
    candidate, for this branch, is worth remembering comprehensively" — they
    don't need to know what the full roster is, or to change when it grows.

    Hard-mode searches pass a MemoryScoreCache, whose minimal read/write
    interface deliberately omits write_scores: those ERD values are
    path-dependent and must never reach the persisted cross-game cache.
    Skip silently rather than require every transient cache to stub it out.
    """
    if not score_cache or not hasattr(score_cache, 'write_scores'):
        return
    for method, value in score_word_multi(candidate, branch_words, list(ScoringMethod),
                                           cache=cache).items():
        score_cache.write_scores(branch_key, [(candidate, value)], method.name.lower())


def rank_candidates_by_max_group_size_then_entropy_gain(words, candidates, rcache, score_cache,
                                        cancel_check=None):
    """Order `candidates` by 1-level max-group-size (asc) then entropy gain
    (desc) for the position `words` — reads/writes the SAME word_scores
    cache Solution.compute_scores_multi uses (ScoreCache.encode_subset(words)
    + method.name.lower()), so work here and work done by 's' at this exact
    position are mutually reusable.

    cancel_check: optional zero-arg callable, checked once per candidate; if
    it returns True, ranking stops and `candidates` is returned unchanged
    (any word_scores already read/written stays valid).
    """
    methods = (ScoringMethod.MAX_GROUP_SIZE, ScoringMethod.ENTROPY_GAIN)
    has_read = score_cache and hasattr(score_cache, 'read_scores')
    has_write = score_cache and hasattr(score_cache, 'write_scores')
    word_scores = {}
    if has_read:
        branch_key = ScoreCache.encode_subset(words)
        for method in methods:
            cached = score_cache.read_scores(branch_key, method.name.lower())
            if cached:
                for w, s in cached:
                    word_scores.setdefault(w, {})[method] = s

    to_write = {m: [] for m in methods}
    scored = []
    for word in candidates:
        if cancel_check is not None and cancel_check():
            return list(candidates)
        cached = word_scores.get(word, {})
        if all(m in cached for m in methods):
            scores = cached
        else:
            groups = rcache.group_counts(word, words)
            scores = score_groups_multi(groups, methods)
            for m in methods:
                if m not in cached:
                    to_write[m].append((word, scores[m]))
        scored.append((scores[ScoringMethod.MAX_GROUP_SIZE],
                       -scores[ScoringMethod.ENTROPY_GAIN], word))

    if has_write:
        branch_key = ScoreCache.encode_subset(words)
        for method, rows in to_write.items():
            if rows:
                score_cache.write_scores(branch_key, rows, method.name.lower())

    scored.sort()
    return [w for _, _, w in scored]


def max_entropy(n):
    """Theoretical maximum entropy for n remaining words: log2(n)."""
    if n <= 1:
        return 0.0
    return math.log2(n)


# ---------------------------------------------------------------------------
# Solution class
# ---------------------------------------------------------------------------

class Solution:
    """
    Tracks the state of a single Wordle game: remaining candidate
    answers, guess history, cached scores, and (optionally) a known
    answer word for simulation mode.

    If all_words is provided, falls back to it when the answer
    list is exhausted (word not in answer list).
    """

    def __init__(self, answer_words, all_words=None,
                 cache=None, score_cache=None):
        self.all_answers = answer_words
        self.all_words = all_words
        self.cache = cache
        self.score_cache = score_cache
        self.reset()

    def reset(self):
        self.current_words = self.all_answers[:]
        self._answer_set = None
        self.guesses = []
        self.word_scores = {}        # {word: {ScoringMethod: score}}
        self._db_loaded_methods = set()
        self.scores = []             # last sorted (word, score) list
        self.scores_method = None
        self.scores_updated = False
        self.answer_word = None
        self.fallback_active = False

    def _invalidate_scores(self):
        self.word_scores = {}
        self._db_loaded_methods = set()
        self.scores = []
        self.scores_method = None
        self.scores_updated = False

    def _is_full_game(self):
        return len(self.guesses) == 0

    def _ensure_scores_loaded(self, method):
        """Transparently load this position's scores from SQLite into word_scores.

        Cached entries are scoped to the current remaining-word subset (not
        just the full answer set), so any position that recurs — not only
        the opening guess — benefits from the cache.
        """
        if not self.score_cache:
            return
        if method in self._db_loaded_methods:
            return
        self._db_loaded_methods.add(method)
        branch_key = ScoreCache.encode_subset(self.current_words)
        cached = self.score_cache.read_scores(branch_key, method.name.lower())
        if cached:
            for w, s in cached:
                self.word_scores.setdefault(w, {})[method] = s

    def _persist_scores(self, method):
        """Transparently write this position's scores from word_scores to SQLite."""
        if not self.score_cache:
            return
        scores = [(w, s[method]) for w, s in self.word_scores.items()
                  if method in s]
        if scores:
            branch_key = ScoreCache.encode_subset(self.current_words)
            self.score_cache.write_scores(branch_key, scores, method.name.lower())

    @property
    def answer_set(self):
        """Cached set of current_words for O(1) membership tests."""
        if self._answer_set is None:
            self._answer_set = set(self.current_words)
        return self._answer_set

    @property
    def solved(self):
        """True once the most recently played guess was itself the answer.

        Narrowing to one remaining candidate by deduction is not the same
        thing: that candidate still has to be played as a real Wordle guess
        before the game is actually won.
        """
        return bool(self.guesses) and self.guesses[-1][1] == ['green'] * 5

    def apply_guess(self, try_word, response):
        """
        Apply a guess and its response, filtering the word list.

        If the result is empty and all_words is available, replays
        all guesses against the full guess vocabulary as a fallback.
        Returns the number of remaining words (caller should check
        for fallback_active).
        """
        self.guesses.append([try_word, list(response)])
        self.current_words = apply_guess(
            self.current_words, try_word, response
        )
        self._answer_set = None
        self._invalidate_scores()

        # Fallback: replay all guesses against full vocabulary
        if (len(self.current_words) == 0
                and self.all_words
                and not self.fallback_active):
            words = self.all_words[:]
            for gw, gr in self.guesses:
                words = apply_guess(words, gw, gr)
            if words:
                self.current_words = words
                self._answer_set = None
                self.fallback_active = True

        return len(self.current_words)

    def undo_guess(self):
        """
        Remove the last guess and recompute current_words from scratch.
        Returns True if a guess was undone, False if history was empty.
        """
        if not self.guesses:
            return False
        self.guesses.pop()
        self.current_words = self.all_answers[:]
        self._answer_set = None
        self.fallback_active = False
        self._invalidate_scores()
        for gw, gr in self.guesses:
            self.current_words = apply_guess(self.current_words, gw, gr)
            self._answer_set = None
        return True

    def constraint_compliant_words(self, all_words):
        """
        Return words from all_words consistent with all prior responses.
        This is real Wordle hard mode: must satisfy green/yellow constraints
        but is not restricted to remaining answers.
        """
        words = list(all_words)
        for gw, gr in self.guesses:
            words = apply_guess(words, gw, gr)
        return words

    def include_letters(self, letters):
        """Keep only words containing all specified letters."""
        for letter in letters:
            self.current_words = [
                w for w in self.current_words if letter in w
            ]
        self._answer_set = None
        self._invalidate_scores()

    def exclude_letters(self, letters):
        """Remove words containing any of the specified letters."""
        for letter in letters:
            self.current_words = [
                w for w in self.current_words if letter not in w
            ]
        self._answer_set = None
        self._invalidate_scores()

    @staticmethod
    def join(solutions):
        """
        Merge unsolved word lists from multiple solutions into a new
        Solution. Used to find a single guess that's good across all
        active boards.
        """
        if not solutions:
            return None
        first = solutions[0]
        out = Solution(first.all_answers,
                       first.all_words,
                       first.cache,
                       first.score_cache)
        combined = set()
        for soln in solutions:
            if len(soln.current_words) > 1:
                combined.update(soln.current_words)
        out.current_words = sorted(combined)
        return out

    def compute_scores(self, input_wordlist,
                       method=ScoringMethod.ENTROPY_GAIN,
                       progress_callback=None):
        """
        Score every word in input_wordlist against current_words.

        Uses per-word cache (word_scores) to skip words already scored
        under this method. Transparently loads from and saves to SQLite
        for full-game state. Returns a sorted list of (word, score) tuples.
        """
        self._ensure_scores_loaded(method)
        results = []
        for word in input_wordlist:
            cached = self.word_scores.get(word)
            if cached is not None and method in cached:
                s = cached[method]
                if progress_callback:
                    progress_callback()
            else:
                s = score_word(
                    word, self.current_words, method,
                    progress_callback, cache=self.cache
                )
                if cached is None:
                    self.word_scores[word] = {method: s}
                else:
                    cached[method] = s
            results.append((word, s))
        results.sort(key=method.sort_key())
        self.scores = results
        self.scores_method = method
        self.scores_updated = True
        self._persist_scores(method)
        return results

    def compute_scores_multi(self, input_wordlist, methods,
                             progress_callback=None):
        """
        Score every word under multiple methods in a single pass.
        Computes group counts once per word for any missing methods.
        Transparently loads from and saves to SQLite for full-game state.
        Returns a list of (word, {method: score}) tuples,
        sorted by the first method in the list.
        """
        for method in methods:
            self._ensure_scores_loaded(method)
        results = []
        for word in input_wordlist:
            cached = self.word_scores.get(word, {})
            missing = [m for m in methods if m not in cached]
            if not missing:
                if progress_callback:
                    progress_callback()
                scores = {m: cached[m] for m in methods}
            else:
                new_scores = score_word_multi(
                    word, self.current_words, missing,
                    progress_callback, cache=self.cache
                )
                merged = dict(cached)
                merged.update(new_scores)
                self.word_scores[word] = merged
                scores = {m: merged[m] for m in methods}
            results.append((word, scores))
        primary = methods[0]
        results.sort(key=lambda x: primary.sort_key()(
            (x[0], x[1][primary])
        ))
        for method in methods:
            self._persist_scores(method)
        return results

    def compute_lookahead(self, top_words,
                          second_step_words=None,
                          progress_callback=None):
        """
        Two-step entropy lookahead on (word, first_entropy) pairs.

        For each candidate first guess, computes the weighted average
        of the best second-step entropy across all response groups.

        second_step_words: word list to search for best second guess.
            If None, uses hard mode (branch words only).
            If provided, searches that list against each branch.

        Completed branch results are cached in the SQLite lookahead
        cache and reused across sessions.

        progress_callback(): called per work unit.

        Returns sorted list of (word, step1, step2, combined)
        tuples, best combined score first.
        """
        method = ScoringMethod.ENTROPY_GAIN
        n = len(self.current_words)
        full_mode = second_step_words is not None
        cache = self.cache
        lc = self.score_cache
        policy = 'full' if full_mode else 'hard'

        # Phase 1: compute group partitions
        word_data = []
        for word, first_ent in top_words:
            grouped = cache.group_words(word, self.current_words)
            word_data.append((word, first_ent, grouped))

        # Phase 2: second-step evaluation
        results = []
        for word, first_ent, grouped in word_data:
            weighted_second = 0.0

            for _pat, branch_words in grouped.items():
                cnt = len(branch_words)
                if cnt <= 1:
                    continue

                if cnt == 2:
                    weighted_second += (2 / n) * 1.0
                    continue

                # Check SQLite cache first
                best = None
                if lc:
                    branch_key = ScoreCache.encode_subset(branch_words)
                    hit = lc.read(branch_key, policy)
                    if hit is not None:
                        _best_guess, best = hit

                if best is None:
                    branch_key = ScoreCache.encode_subset(branch_words) if lc else None
                    candidates = (second_step_words
                                  if full_mode else branch_words)
                    best = 0.0
                    best_guess = None
                    for candidate in candidates:
                        if progress_callback:
                            progress_callback()
                        s = score_word(
                            candidate, branch_words, method,
                            cache=cache
                        )
                        if s > best:
                            best = s
                            best_guess = candidate
                    if lc and best_guess is not None and branch_key is not None:
                        lc.write(branch_key, policy, best_guess, best)
                        cache_all_scores(best_guess, branch_words, lc, branch_key,
                                         cache=cache)

                weighted_second += (cnt / n) * best

            combined = first_ent + weighted_second
            results.append((word, first_ent, weighted_second, combined))

        results.sort(key=lambda x: -x[3])
        return results


def _cache_reuse(entry, budget):
    """Decide whether a cached entry is valid at `budget` (None = unlimited).

    entry is (best_guess, best_score, max_depth, solve_budget) or None.
    Returns (cost, max_depth, tainted) to reuse, or None to recompute.

    Reuse rules (see cache_sqlite schema):
      unlimited (budget None): reuse legacy (max_depth None) and untainted
        (solve_budget None) entries — both are unconstrained optima; reject a
        tainted (solve_budget set) entry, which is budget-specific.
      budgeted: reject legacy (unknown depth); an untainted entry is valid
        when max_depth <= budget; a tainted entry only at solve_budget==budget.
    A reused tainted entry propagates tainted=True (its subtree hit the floor).
    """
    if entry is None:
        return None
    _bw, score, max_remaining_depth, sb = entry
    if budget is None:
        if sb is not None:
            return None
        return (score, max_remaining_depth, False)
    if max_remaining_depth is None:
        return None
    if sb is None:
        return (score, max_remaining_depth, False) if max_remaining_depth <= budget else None
    return (score, max_remaining_depth, True) if sb == budget else None


def _by_group_size(item):
    return len(item[1])


def _candidate_cost_lower_bound(group_sizes, has_self, n):
    """The engine's admissible lower bound on a candidate's ERD, shared by
    evaluate_candidate and the cost-estimator functions below: cost >=
    3 - (number_of_groups + has_self) / n.

    Why that holds.  Playing this candidate on a branch of `n` words sends each
    answer into one response group.  Only one answer can be finished by the
    guess itself — the candidate, when it is one of the branch words — and that
    line costs 1.  Within any other group, the next guess can finish at most one
    more answer, at a cost of 2; every remaining answer in that group needs at
    least a third guess.  So a group of `m` answers costs at least
    `2 + 3(m - 1) = 3m - 1`, and summing over the `G - has_self` groups that are
    not the candidate's own gives at least `3(n - has_self) - (G - has_self)`.
    Adding the candidate's own line, `1 = 3 - 2`, the total over all answers is
    at least `3n - G - has_self`.  Dividing by `n` gives the bound.

    Its ceiling is what makes the branch floor necessary: `G + has_self` cannot
    exceed `n + 1`, so this can never reach 3.0, and on a branch whose true ERD
    is 3.0 or more it prunes nothing at all.

    group_sizes need only support len() — evaluate_candidate passes
    groups.values() directly rather than materializing a list of sizes.
    """
    return 3.0 - (len(group_sizes) + (1 if has_self else 0)) / n


def all_singletons_floor(size):
    """ERD floor for a branch of `size` words attained by a perfect split: one
    guess wins outright, every other word is named on the guess after.

    Admissible for the same reason and one level weaker: at most one answer is
    found in one guess, so the total over `size` answers is at least
    `1 + 2(size - 1)`.  It asks nothing about which words may be played, which
    is why it holds for any pool and why the branch floor — which does ask —
    is the larger of the two on every branch no guess actually shatters.
    """
    return 2.0 - 1.0 / size


class BranchFloorTable:
    """Branch ERD floors for ONE candidate pool, fixed at construction.

    The floor for a branch is the best `_candidate_cost_lower_bound` any word
    in the pool can be held to: since every candidate costs at least
    `3 - (G_c + has_self_c) / k`, the branch costs at least that minimized over
    the pool, which is `3 - max_c(G_c + has_self_c) / k`.  Taking the maximum
    of that with `all_singletons_floor(k)` is admissible because each is
    separately a lower bound on the same quantity, and the larger of two lower
    bounds is a lower bound.

    Both terms are minimized over the pool the search will draw candidates
    from, and that is the invariant the whole thing rests on: the pool this
    table prices against and the pool the search may play must be the same
    words, and must stay the same words for the entire solve.  Widening either
    one alone breaks it.  A search allowed to play a word the table did not
    price may have a strategy cheaper than the floor claims possible, and the
    floor then prunes it — which raises the reported ERD with nothing to mark
    that a better line was discarded.

    A floor is only a floor over the words that may actually be played: a wider
    pool splits a branch more finely and so lowers it.  Serving a floor
    computed against a narrower pool than the search may play would claim a
    floor above reachable strategies, which prunes the optimum and silently
    raises the reported ERD.  The pool is therefore a property of this object
    rather than an argument to its methods — there is no per-call pool to
    disagree with the memo, and none to pass wrongly.

    Construct one per solve, from the same word list the search draws its
    candidates from (_solve_subset's candidate_list), and share it across the
    whole recursion: reuse across parents is what makes the floors cheaper than
    the search they replace.  The memo is bounded and insertion-ordered,
    evicting oldest-first, because a branch under evaluation revisits its own
    sub-branches across consecutive candidates.

    The vectorized path is used only when the matrix's rows ARE the pool;
    otherwise the matrix would answer for a vocabulary the search cannot play
    and the reference implementation is used instead.  Both compute the same
    value for the same pool.
    """

    def __init__(self, candidate_pool, cache=None, pattern_matrix=None,
                 capacity=_BRANCH_FLOOR_CAPACITY):
        # A tuple is kept as-is so a caller that snapshotted its pool holds the
        # very object this table answers for, and the per-candidate check
        # stays a pointer comparison.
        self._candidate_pool = (candidate_pool
                                if isinstance(candidate_pool, tuple)
                                else tuple(candidate_pool))
        self._cache = cache
        self._pattern_matrix = (
            pattern_matrix
            if pattern_matrix is not None
            and pattern_matrix.is_guess_pool(self._candidate_pool)
            else None)
        self._capacity = capacity
        self._floors = OrderedDict()
        self._verified_pool = None
        self.hits = 0
        self.misses = 0

    @property
    def candidate_pool(self):
        """The words this table prices strategies over, fixed at construction.

        Read-only: the memo holds floors computed against these words, and a
        pool swapped in afterwards would leave every stored floor priced for
        words the table no longer claims — approval granted for one vocabulary,
        answers given for another.  That is the same inadmissible pairing a
        mutable pool produces, reached from the other side.  The memo stays
        mutable; only what it is a memo *of* is frozen.
        """
        return self._candidate_pool

    def matches_pool(self, candidate_pool):
        """True when candidate_pool is the word set this table was built for.

        A table answers only for its own pool, so a caller that supplies one
        must be holding it against the same words the search draws candidates
        from.

        An approval is remembered only against a pool that cannot change.  A
        list that passed once can gain a word afterwards, and a remembered
        approval would go on holding: the table would keep answering with
        floors priced for the smaller pool, which sit above what the larger
        one can reach, and the search would prune strategies it can actually
        play.  A tuple cannot gain a word, so its approval is safe to keep —
        which is why callers snapshot their pool as a tuple rather than
        carrying a list forward.

        Word-set equality is deliberate: order is the search's business, not
        this table's.  Two pools of the same words split a branch identically,
        so they have the same floors, while the order among equal-cost
        candidates decides which one the search records.  A table therefore
        approves an ordering it does not share, and must never impose its own.
        """
        if candidate_pool is None:
            return False
        if (candidate_pool is self._candidate_pool
                or candidate_pool is self._verified_pool):
            return True
        if set(candidate_pool) != set(self._candidate_pool):
            return False
        if isinstance(candidate_pool, tuple):
            self._verified_pool = candidate_pool
        return True

    def branch_cost_lower_bound(self, sub_branch):
        key = tuple(sorted(sub_branch))
        cached = self._floors.get(key)
        if cached is not None:
            self.hits += 1
            self._floors.move_to_end(key)
            return cached
        self.misses += 1
        floor = None
        if self._pattern_matrix is not None:
            branch_indices = self._pattern_matrix.answer_indices_or_none(sub_branch)
            if branch_indices is not None:
                floor = self._pattern_matrix.branch_cost_lower_bound(branch_indices)
        if floor is None:
            if self._cache is None:
                return all_singletons_floor(len(sub_branch))
            floor = self._cache.branch_cost_lower_bound(sub_branch,
                                                        self._candidate_pool)
        self._floors[key] = floor
        if len(self._floors) > self._capacity:
            self._floors.popitem(last=False)
        return floor

    def sub_branch_cost_lower_bound(self, sub_branch, candidate):
        """Admissible floor on what `sub_branch` costs the parent that reached it.

        A singleton is one guess, or none at all when it is the candidate just
        played.  Otherwise the floor is `all_singletons_floor` raised to the
        sub-branch's own floor.

        `all_singletons_floor` assumes some guess shatters the sub-branch into
        singletons and is itself a member — true only for small, well-separated
        sets.  The pool-wide floor instead asks what the guess words can
        actually do to these words, and is the larger of the two on every
        branch the split does not genuinely shatter.  Both are admissible, so
        the maximum is too.

        Every sub-branch gets the computed floor.  A size threshold looks
        attractive — a floor costs one pass over the pool whatever the
        sub-branch's size, and a sub-branch of k words moves its parent's bound
        with weight k/n — but that weight argument is a function of the parent
        size, so no single cutoff holds across the range of parents the search
        walks.  The small sub-branches are also the numerous ones, so skipping
        them forfeits most of the tightening; measured on the strong-scaling
        workload, a cutoff of 8 doubled the drain time it was meant to protect.
        """
        size = len(sub_branch)
        if size == 1:
            return 0.0 if sub_branch[0] == candidate else 1.0
        return max(all_singletons_floor(size),
                   self.branch_cost_lower_bound(sub_branch))


def sub_branch_cost_lower_bound(sub_branch, candidate, branch_floor_table=None):
    """`branch_floor_table`'s floor for this sub-branch, or the all-singletons
    floor when no table is supplied — a weaker bound, never a wrong one.

    Searches that vary their candidate pool per node (the answers-only policy,
    where each node's pool is its own branch words) pass no table: a pool that
    changes as the recursion descends is exactly what a table may not hold.
    """
    if branch_floor_table is None:
        size = len(sub_branch)
        if size == 1:
            return 0.0 if sub_branch[0] == candidate else 1.0
        return all_singletons_floor(size)
    return branch_floor_table.sub_branch_cost_lower_bound(sub_branch, candidate)


def _candidate_response_groups(branch_words, candidate, cache,
                               pattern_matrix=None, branch_indices=None):
    """Response groups for one candidate, using the same kernel dispatch as
    evaluate_candidate.

    Kept separate so a worker can prove a whole claimed bundle against the
    two-level entry bound before it performs any per-candidate queue writes.
    """
    if cache:
        if (pattern_matrix is not None and branch_indices is None
                and not cache.is_cached(candidate)):
            branch_indices = pattern_matrix.answer_indices_or_none(branch_words)
        return cache.group_words(
            candidate, branch_words,
            pattern_matrix=pattern_matrix,
            branch_indices=branch_indices,
        )
    groups = defaultdict(list)
    for answer in branch_words:
        pattern = _encode_response(calculate_response(candidate, answer))
        groups[pattern].append(answer)
    return groups


def _remaining_groups_cost_lower_bounds(ordered_groups, candidate,
                                        branch_size, branch_floor_table):
    """Suffix sums used by both the two-level entry gate and sub-ceilings."""
    remaining_groups_cost_lower_bound = [0.0] * (len(ordered_groups) + 1)
    for index in range(len(ordered_groups) - 1, -1, -1):
        sub_branch = ordered_groups[index][1]
        remaining_groups_cost_lower_bound[index] = (
            remaining_groups_cost_lower_bound[index + 1]
            + (len(sub_branch) / branch_size) * sub_branch_cost_lower_bound(
                sub_branch, candidate, branch_floor_table)
        )
    return remaining_groups_cost_lower_bound


def candidate_two_level_cost_lower_bound(
        branch_words, candidate, cache, guesses=None,
        pattern_matrix=None, branch_indices=None, branch_floor_table=None):
    """Admissible two-level ERD lower bound for one candidate.

    This is evaluate_candidate's entry proof without recursion.  It performs
    the candidate's response partition and prices each response group through
    the same BranchFloorTable the search uses.  A caller may therefore batch
    several proofs outside a queue transaction and persist only the candidates
    whose returned bound reaches the branch's current best ERD or ceiling.

    The candidate pool is snapshotted and checked against branch_floor_table
    for the same reason as evaluate_candidate: a floor priced against a
    different pool can exceed a strategy the search is allowed to play.
    """
    if guesses is not None:
        guesses = tuple(guesses)
    if (branch_floor_table is not None
            and not branch_floor_table.matches_pool(guesses)):
        raise ValueError(
            "branch_floor_table's candidate pool is not this search's guess "
            "words; its floors would price strategies the search cannot play")
    branch_size = len(branch_words)
    groups = _candidate_response_groups(
        branch_words, candidate, cache,
        pattern_matrix=pattern_matrix,
        branch_indices=branch_indices,
    )
    has_self = _ALL_GREEN_PATTERN in groups
    closed_form_cost_lower_bound = _candidate_cost_lower_bound(
        groups.values(), has_self, branch_size)
    ordered_groups = sorted(groups.items(), key=_by_group_size, reverse=True)
    remaining_groups_cost_lower_bound = _remaining_groups_cost_lower_bounds(
        ordered_groups, candidate, branch_size, branch_floor_table)
    two_level_cost_lower_bound = (
        1.0 + remaining_groups_cost_lower_bound[0]
    )
    return max(closed_form_cost_lower_bound, two_level_cost_lower_bound)


def estimate_candidate_work(group_sizes, has_self, n, best_erd, budget, typical):
    """Predicted search work (in nodes), UNCUT sum over recursed groups.

    Returns 0.0 when the candidate is provably ERD-pruned
    (candidate_cost_lower_bound >= best_erd): the engine cuts it for free, so
    this is exact.  Otherwise it sums the cost model's typical node count for
    EVERY response group the candidate could recurse into.

    This is the naive estimate §4 warns about: it ignores the accumulated-cost
    cutoff, so it over-predicts weak splitters that pass the entry gate but are cut
    almost immediately.  Kept for comparison against estimate_candidate_work_cutoff
    in the metric-validation gate; the packer should prefer the cutoff-aware form.

    `typical(k, sub_budget)` returns the model's geometric-mean node count for a
    sub-branch of size k at the given budget, or None when cold; a cold size falls
    back to k itself.  A scheduling estimate only — it never sets a pruning bound.
    """
    if (best_erd is not None
            and _candidate_cost_lower_bound(group_sizes, has_self, n) >= best_erd):
        return 0.0
    sub_budget = None if budget is None else budget - 1
    total = 0.0
    for k in group_sizes:
        if k <= 1 or k >= n:
            continue
        t = typical(k, sub_budget)
        total += t if t is not None else float(k)
    return total


def estimate_candidate_work_cutoff(group_sizes, has_self, n, best_erd, budget,
                                   typical):
    """Predicted search work, CUTOFF-AWARE — the §4 metric.

    Mirrors evaluate_candidate's sub-branch loop: process groups largest-first,
    accumulating a per-group ERD proxy (3 - 2/k), and sum the cost model's node
    estimate only for the groups reached before the accumulated cost meets
    best_erd.  The proxy deliberately exceeds the admissible lower bound
    (2 - 1/k): the admissible bound sums to ~candidate_cost_lower_bound, which
    by construction never reaches best_erd for a non-ERD-pruned candidate, so
    a cutoff built on it would never fire.  The price is that this modeled
    cutoff can fire EARLIER than the engine's, under-counting reached groups —
    acceptable for a scheduling estimate.

    The effect: a weak splitter (few large groups, candidate_cost_lower_bound
    just under best_erd) busts the bound after one or two groups and predicts
    ~0; a strong splitter (candidate_cost_lower_bound far below best_erd)
    reaches many groups and predicts large.  This is the direction the uncut
    estimate gets backwards.  Scheduling only; never a bound.
    """
    if best_erd is None or best_erd == float('inf'):
        # No real bound yet: nothing is provably cut, so fall back to the uncut
        # sum (the conservative, finer-granularity direction per §4).
        return estimate_candidate_work(group_sizes, has_self, n, best_erd, budget,
                                       typical)
    if _candidate_cost_lower_bound(group_sizes, has_self, n) >= best_erd:
        return 0.0
    sub_budget = None if budget is None else budget - 1
    cost = 1.0                      # the candidate's own guess
    self_pending = has_self         # one size-1 group is the candidate itself (0 cost)
    total = 0.0
    for k in sorted(group_sizes, reverse=True):   # engine order: largest first
        if k >= n:
            continue                # no information gained; engine rejects
        if k == 1:
            if self_pending:
                self_pending = False
                continue            # the self singleton: 0 further guesses
            cost += 1.0 / n         # a non-self singleton: ~0 search, small cost
        else:
            t = typical(k, sub_budget)
            total += t if t is not None else float(k)
            # Accumulate an ESTIMATE of the sub-branch's real ERD, not its
            # admissible lower bound: the lower bound (2 - 1/k) ~=
            # candidate_cost_lower_bound summed, so it never trips for a
            # non-ERD-pruned candidate and the cutoff would never fire.
            # `3 - 2/k` is a monotone imperfect-split proxy in [2, 3)
            # — bigger groups cost more — and is the dial to calibrate offline
            # against the group_sizes corpus.
            cost += (k / n) * (3.0 - 2.0 / k)
        if cost >= best_erd:
            break                   # engine cuts here; later groups never reached
    return total


def evaluate_candidate(branch_words, candidate, cache, score_cache, *,
                   n=None, best_erd=float('inf'),
                   deadline=None, guesses=None, policy=ERD_ALL,
                   cancel_check=None, heartbeat=None,
                   note_depth=None, budget=None,
                   subbranch_solver=None, bound_provider=None,
                   mid_loop_publisher=None, metric_observer=None,
                   pattern_matrix=None, branch_indices=None,
                   branch_floor_table=None):
    """Evaluate one `candidate`'s exact ERD for solving `branch_words`.

    This is the body of the top-level candidate loop, extracted so a parallel
    coordinator can distribute candidates of the SAME `branch_words` across
    workers while sharing one running `best_erd` as the branch-and-bound bound.
    Recursion stays single-threaded and writes each sub-result to score_cache.

    budget: guesses available to solve `branch_words` from this point (None =
    unlimited).  A sub-branch that can't be solved within budget-1 makes this
    candidate infeasible (cost inf) and marks floor_hit — the depth cap fired.

    bound_provider: optional callable returning the current global best ERD.
    Called once before evaluation starts and once per sub-branch to tighten
    the alpha-beta bound dynamically, so a worker mid-evaluation benefits from
    improvements found by other workers without waiting for the next candidate.

    branch_indices: optional, branch_words expressed as pattern_matrix column
    indices (pattern_matrix.answer_indices(branch_words)). Callers that already
    hold this for `branch_words` (the _solve_subset candidate loop) should pass
    it to avoid recomputing it once per candidate; a caller that only has
    pattern_matrix (a single-candidate claim) may leave it None and this
    function derives it, falling back to None when branch_words contains a
    word outside the answer universe (the interactive fallback mode).

    Returns (status, cost, max_depth, floor_hit):
      ('ok', cost, max_remaining_depth, floor)  fully evaluated; cost < best_erd; max_remaining_depth is this
                               strategy's worst-case line length (None when
                               unlimited — not tracked).
      ('pruned', None, max_remaining_depth, floor)   can't beat best_erd, OR infeasible within
                               budget (a sub-branch hit the floor).
      ('useless', None, None, floor) a response group is all of `branch_words`.
      ('abort', None, None, floor)  deadline/cancel fired; caller must stop.
    floor_hit is always returned (even on early returns) so the caller can
    aggregate taint across all candidates it tries, including discarded ones.
    """
    # Snapshot the caller's pool before anything else can see it, keeping ITS
    # order (which breaks ties among equal-cost candidates).  Validating the
    # object the caller still holds only establishes what the pool was at that
    # instant: `heartbeat` runs below, and a callback that appends to a list
    # handed in here would widen the search while the table goes on quoting
    # floors priced for the narrower pool — floors that sit above what the
    # wider pool can reach, so the search prunes strategies it can play.  A
    # caller already holding a tuple hands over that same object, so a table
    # built for it still matches by identity.
    if guesses is not None:
        guesses = tuple(guesses)
    if (branch_floor_table is not None
            and not branch_floor_table.matches_pool(guesses)):
        raise ValueError(
            "branch_floor_table's candidate pool is not this search's guess "
            "words; its floors would price strategies the search cannot play")
    if n is None:
        n = len(branch_words)
    # Liveness tick: fire once per candidate evaluation (the dominant work
    # unit).  Guarantees a progress signal even through a long run of guesses
    # that all prune below without recursing.  Observation only.
    if heartbeat is not None:
        heartbeat()
    groups = _candidate_response_groups(
        branch_words, candidate, cache,
        pattern_matrix=pattern_matrix,
        branch_indices=branch_indices,
    )

    # Tighten the bound from any external update before starting evaluation.
    if bound_provider is not None:
        best_erd = min(best_erd, bound_provider())

    # Admissible lower bound on this candidate's cost — cost >= 3 - (G + has_self)/n.
    has_self = _ALL_GREEN_PATTERN in groups
    candidate_cost_lower_bound = _candidate_cost_lower_bound(
        groups.values(), has_self, n)
    # Report the work-metric inputs (group sizes, the closed-form lower bound,
    # the bound actually in force after any external tightening) so a
    # measurement layer can log predicted-vs-actual without recomputing them.
    # It fires exactly once per candidate.
    #
    # The bound reported is the closed form `3 - (G + has_self)/n` and not the
    # tighter two-level bound the gate below may act on.  analyze_swarm_telemetry
    # inverts that identity to recover has_self from the stored value, so a
    # tighter number in this field decodes as a different candidate: at n=40
    # with 17 groups and has_self true, 2.55 is the closed form and reads back
    # correctly, while the effective 2.575 reads back as has_self false.
    # `pruned` carries which gate actually cut the candidate, so predicted work
    # still reflects the two-level gate without moving what this field means.
    def _observe(pruned):
        if metric_observer is not None:
            metric_observer([len(g) for g in groups.values()], has_self,
                            candidate_cost_lower_bound, best_erd, pruned)

    if candidate_cost_lower_bound >= best_erd:
        # Provably can't beat the bound (but may well be feasible) — a cutoff,
        # not infeasibility.  See OVER_ERD_LIMIT in _solve_subset.
        _observe(True)
        return (OVER_ERD_LIMIT, None, None, False)

    cost = 1.0
    cand_max_remaining_depth = 1 if budget is not None else None
    floor = False
    sub_budget = None if budget is None else budget - 1
    # Largest sub-branches first: highest weight (k/n), pushes cost up fastest
    # so the branch-and-bound check fires after as few sub-evaluations as possible.
    ordered = sorted(groups.items(), key=_by_group_size, reverse=True)

    # Alpha-beta: solve each sub-branch under a derived ceiling so a deep node
    # prunes from a tight bound instead of inf.  remaining_groups_cost_lower_bound[i]
    # is an admissible lower bound on the weighted cost of the sub-branches
    # *after* position i (each sub-branch of size k costs >= lb(k)).  The self
    # singleton contributes 0.
    remaining_groups_cost_lower_bound = _remaining_groups_cost_lower_bounds(
        ordered, candidate, n, branch_floor_table)

    def _sub_lb(sub_branch):
        return sub_branch_cost_lower_bound(
            sub_branch, candidate, branch_floor_table)

    # Second gate, on the same sum the ceilings are derived from.  Position 0
    # holds every sub-branch's floor, so 1 + that is this candidate's cost
    # lower bound priced per sub-branch rather than by group count alone.  It
    # is never below candidate_cost_lower_bound and rises above 3.0 where that
    # bound cannot, which is the range large branches live in.
    two_level_cost_lower_bound = 1.0 + remaining_groups_cost_lower_bound[0]
    effective_cost_lower_bound = max(candidate_cost_lower_bound,
                                     two_level_cost_lower_bound)
    erd_lower_bound_pruned = effective_cost_lower_bound >= best_erd
    _observe(erd_lower_bound_pruned)
    if erd_lower_bound_pruned:
        return (OVER_ERD_LIMIT, None, None, False)

    for i, (pattern_code, sub_branch) in enumerate(ordered):
        # Tighten the bound from any inter-worker improvement before computing
        # this sub-branch's ceiling and the accumulated-cost cutoff check.
        if bound_provider is not None:
            best_erd = min(best_erd, bound_provider())
        k = len(sub_branch)
        if k == 1 and sub_branch[0] == candidate:
            continue  # self: solved by playing this candidate, 0 further guesses
        if k >= n:
            return (NO_INFORMATION_GAINED, None, None, floor)  # split gained nothing
        # Max ERD this sub-branch may have for the candidate to still beat the
        # bound, assuming the remaining siblings achieve only their lower bound.
        if best_erd == float('inf'):
            sub_ceiling = float('inf')
        else:
            sub_ceiling = (best_erd - cost
                          - remaining_groups_cost_lower_bound[i + 1]) * (n / k) + _CEIL_EPS
            if sub_ceiling <= _sub_lb(sub_branch):
                # Dead ceiling: every sub-branch of size k costs >= lb(k)
                # (admissible lower bound), so even a floor-attaining sub-tree
                # prices this candidate at >= best_erd.  The refutation is
                # pure arithmetic — never recurse, and never offer a foregone
                # conclusion to the swarm as a sub-branch.
                return (OVER_ERD_LIMIT, None, cand_max_remaining_depth, floor)
        sub = _solve_subset(
            sub_branch, cache, score_cache, sub_budget, deadline, guesses,
            policy, cancel_check, heartbeat, note_depth, None,
            subbranch_solver, ceiling=sub_ceiling,
            entry_guess=candidate, entry_pattern=pattern_code,
            mid_loop_publisher=mid_loop_publisher, pattern_matrix=pattern_matrix,
            branch_floor_table=branch_floor_table)
        if sub in _ABORT_STATUSES:
            return (sub, None, None, False)
        sub_status, sub_cost, sub_max_remaining_depth, sub_budget_tainted = sub
        floor = floor or sub_budget_tainted
        if sub_status == OVER_ERD_LIMIT:
            # Sub-branch search stopped at >= its ceiling: this candidate's cost
            # is therefore >= best_erd.  Discard it (sub_cost is only a lower
            # bound), keeping the sub-branch's floor taint: a floored child's
            # ceiling refutation holds only among budget-feasible strategies,
            # so an untainted result here would falsely claim no unconstrained
            # strategy was pruned.
            return (OVER_ERD_LIMIT, None, cand_max_remaining_depth, floor)
        if sub_status == OVER_DEPTH_BUDGET:
            # Sub-branch unsolvable within budget — this candidate is infeasible.
            return (OVER_DEPTH_BUDGET, None, None, True)
        cost += (k / n) * sub_cost
        if budget is not None:
            cand_max_remaining_depth = max(cand_max_remaining_depth, 1 + sub_max_remaining_depth)
        if cost >= best_erd:
            return (OVER_ERD_LIMIT, None, cand_max_remaining_depth, floor)
    return (SOLVED, cost, cand_max_remaining_depth, floor)


def _solve_subset(branch_words, cache, score_cache, budget, deadline, guesses,
                  policy, cancel_check, heartbeat, note_depth,
                  progress_callback, subbranch_solver=None,
                  branch_floor_table=None,
                  ceiling=float('inf'), entry_guess=None, entry_pattern=None,
                  mid_loop_publisher=None, pattern_matrix=None):
    """Budget-aware core of min_expected_guesses.

    Returns (cost, max_depth, floor_hit, cutoff), or None on deadline/cancel
    abort.

    cutoff distinguishes the two ways a search can fail to return an exact
    optimum, which the cache MUST treat differently:
      cutoff=False  cost is exact (the true optimum), OR a definite budget
                    floor: cost == inf with floor_hit means `branch_words` was
                    *proven* unsolvable within `budget`.  Either way reusable.
      cutoff=True   alpha-beta gave up early — every candidate priced out at
                    >= `ceiling`, so all we know is cost >= ceiling SO FAR;
                    solvability was never determined.  cost is only a lower
                    bound (= ceiling); the caller must NOT cache it.

    ceiling seeds the branch-and-bound bound (best_erd) so a deep node prunes
    from a tight value instead of inf.  Default inf = no alpha-beta pressure
    (legacy behaviour); a cutoff can then never occur (best_erd stays inf).

    budget None = unlimited (legacy: floor never fires, max_depth None,
    results cached as unconstrained).
    """
    n = len(branch_words)
    if heartbeat is not None:
        heartbeat()
    if note_depth is not None:
        note_depth(budget, n, entry_guess, entry_pattern)
    if budget is not None and budget < 1:
        return (OVER_DEPTH_BUDGET, float('inf'), None, True)   # no guess available at all
    if n == 1:
        return (SOLVED, 1.0, 1, False)
    if budget is not None and n > max_solvable_within(budget):
        # More words than any strategy can resolve within budget — certain loss.
        # Subsumes the budget < 2 case (M(1) = 1): >=2 words need >=2 guesses.
        return (OVER_DEPTH_BUDGET, float('inf'), None, True)

    if policy is None:
        policy = ERD_ALL if guesses is not None else ERD_ANSWERS
    elif policy not in VALID_ERD_POLICIES:
        raise ValueError(
            f"Unknown ERD policy {policy!r}; expected one of "
            f"{sorted(VALID_ERD_POLICIES)}"
        )
    candidate_list = guesses if guesses is not None else branch_words

    branch_key = ScoreCache.encode_subset(branch_words)
    if score_cache:
        reuse = _cache_reuse(
            score_cache.read_with_depth(branch_key, policy), budget)
        if reuse is not None:
            return (SOLVED, *reuse)   # cached values are exact optima
        # A loss proven within b guesses is a loss within any budget <= b.
        # Reuse it instead of re-disproving every candidate from scratch.
        if budget is not None:
            loss_budget = score_cache.read_loss(branch_key, policy)
            if loss_budget is not None and budget <= loss_budget:
                return (OVER_DEPTH_BUDGET, float('inf'), None, True)

    if deadline is not None and time.time() > deadline:
        return DEADLINE_EXCEEDED
    if cancel_check is not None and cancel_check():
        return CANCEL_RECVD

    # Recursive parallelism: on a cache miss, offer this sub-branch to the
    # swarm.  The solver decides (by size) whether to solve it cooperatively
    # across workers (returns a result) or decline (None) so we solve inline.
    # The frame's alpha-beta ceiling travels with the offer: a cooperative
    # solve prunes against it exactly as this frame would inline, and may
    # return a cut (OVER_ERD_LIMIT, bound, None, floor) — the same tuple an
    # inline all-cutoff loop produces below.  Correctness-neutral either way:
    # a solver that inlines yields the identical ERD.
    if subbranch_solver is not None:
        delegated = subbranch_solver(branch_words, budget, ceiling)
        if delegated is not None:
            if note_depth is not None:
                note_depth(budget, -n, None, None)  # sentinel: this level was promoted
            return delegated

    # Best-first ordering: evaluate the strongest splitter first (key = expected
    # remaining set size, Σ k²; smaller is better) so the branch-and-bound bound
    # (best_erd) is tight from the 2nd candidate on, letting evaluate_candidate's
    # partial-sum cutoff prune the rest before they recurse.  Order-only — the
    # minimum, and therefore every cached result, is unchanged.
    #
    # candidate_cost_lower_bounds, when set, is the vectorized twin of
    # evaluate_candidate's own admissible candidate_cost_lower_bound bound
    # (C2.1), aligned
    # index-for-index with the (already reordered) candidate_list — see the
    # ERD-pruning check in the candidate loop below.
    # branch_indices lets both the ordering below and every evaluate_candidate
    # call in the loop use the group_words fast path without each recomputing
    # branch_words' column indices — None when pattern_matrix is absent or a
    # branch word falls outside the answer universe (interactive fallback).
    branch_indices = None
    if pattern_matrix is not None:
        branch_indices = pattern_matrix.answer_indices_or_none(branch_words)

    candidate_cost_lower_bounds = None
    if cache and n >= ORDER_MIN_N and len(candidate_list) > 1:
        vectorized = False
        if branch_indices is not None:
            try:
                candidate_rows = [pattern_matrix.guess_index(c) for c in candidate_list]
            except KeyError:
                candidate_rows = None  # exotic interactive candidate: fall back below
            if candidate_rows is not None:
                stats = pattern_matrix.candidate_stats(branch_indices)
                # Python's sort is stable, so this reproduces exactly the order
                # np.argsort(kind='mergesort') would give on the same keys —
                # equal-Σk² candidates keep their pre-sort relative order (C2.2).
                order = sorted(
                    range(len(candidate_list)),
                    key=lambda i: stats.sum_squared_group_sizes[candidate_rows[i]],
                )
                candidate_list = [candidate_list[i] for i in order]
                candidate_cost_lower_bounds = [
                    stats.cost_lower_bound[candidate_rows[i]] for i in order]
                vectorized = True
        if not vectorized:
            candidate_list = sorted(
                candidate_list,
                key=lambda c: sum(
                    k * k for k in cache.group_counts(c, branch_words).values()),
            )

    # Seed the bound with the alpha-beta ceiling: any candidate that can't beat
    # it is a cutoff, and if none can we report a cutoff (lower bound) rather
    # than a spurious optimum.
    best_erd = ceiling
    best_guess = None
    best_max_remaining_depth = None
    node_floor = False
    cutoff_occurred = False
    token = mid_loop_publisher.enter(branch_words, budget) if mid_loop_publisher is not None else None

    for i, candidate in enumerate(candidate_list):
        # Checked every iteration, not only inside evaluate_candidate: a
        # candidate answered entirely from cache returns before any deeper
        # cancel check runs, so a loop of cache hits would otherwise never
        # observe a stop request.
        if cancel_check is not None and cancel_check():
            return CANCEL_RECVD
        if (candidate_cost_lower_bounds is not None
                and candidate_cost_lower_bounds[i] >= best_erd):
            # ERD-lower-bound pruned: the same admissible bound
            # evaluate_candidate would compute (C2.1) already proves this
            # candidate cannot beat best_erd. Falls through to the same
            # dispatch below as an OVER_ERD_LIMIT from evaluate_candidate
            # itself — in particular the status == OVER_ERD_LIMIT check just
            # below still sets cutoff_occurred, so a node where every
            # candidate is ERD-pruned is never mistaken for a proven loss.
            #
            # The heartbeat tick mirrors evaluate_candidate's own unconditional
            # first action: the node counter it drives (erd_swarm.py's
            # _BranchWorker._nodes) is documented to count every candidate
            # considered, ERD-pruned or not, and the swarm's cost-model
            # calibration (nodes_spent) depends on that count staying exact.
            if heartbeat is not None:
                heartbeat()
            status, cost, max_remaining_depth, budget_tainted = (
                OVER_ERD_LIMIT, None, None, False)
        else:
            status, cost, max_remaining_depth, budget_tainted = evaluate_candidate(
                branch_words, candidate, cache, score_cache,
                n=n, best_erd=best_erd, deadline=deadline, guesses=guesses,
                policy=policy, cancel_check=cancel_check, heartbeat=heartbeat,
                note_depth=note_depth, budget=budget,
                subbranch_solver=subbranch_solver,
                mid_loop_publisher=mid_loop_publisher,
                pattern_matrix=pattern_matrix,
                branch_indices=branch_indices,
                branch_floor_table=branch_floor_table,
            )
        if status in _ABORT_STATUSES:
            return status
        node_floor = node_floor or budget_tainted
        # Overrun check: runs every iteration (before status continues) so
        # expensive cutoff-tail candidates are caught, not just 'ok' ones.
        if token is not None:
            pub_result = mid_loop_publisher.check(
                token, candidate_list, i,
                best_guess, best_erd, best_max_remaining_depth, budget)
            if pub_result is not None:
                return pub_result
        if status == OVER_ERD_LIMIT:
            cutoff_occurred = True
            continue
        if status != SOLVED:
            continue

        if cost < best_erd:
            best_erd = cost
            best_guess = candidate
            best_max_remaining_depth = max_remaining_depth

        if progress_callback is not None:
            progress_callback(i + 1, len(candidate_list), best_guess, best_erd)

    if best_guess is None:
        if cutoff_occurred:
            # Every candidate priced out at >= ceiling but none was proven
            # infeasible: a lower bound only (= ceiling), solvability unknown.
            # Do NOT cache — OVER_ERD_LIMIT so the caller discards it.
            return (OVER_ERD_LIMIT, ceiling, None, node_floor)
        # No feasible guess and no cutoff: proven unsolvable within budget.
        # Persist the loss so this exhaustive disproof is never repeated for
        # this branch at this (or any smaller) budget.
        if score_cache and budget is not None:
            score_cache.write_loss(branch_key, policy, budget)
        return (OVER_DEPTH_BUDGET, float('inf'), None, True)

    if score_cache:
        # Untainted (floor never fired) => unconstrained optimum, reusable at
        # any budget >= max_depth: solve_budget NULL.  Tainted => valid only at
        # this budget: solve_budget = budget.  best_erd here is the EXACT
        # optimum: finding a candidate below the ceiling means the ceiling only
        # pruned provably-worse candidates, so the value is universally valid.
        solve_budget = None if (budget is None or not node_floor) else budget
        score_cache.write(branch_key, policy, best_guess, best_erd,
                          max_depth=best_max_remaining_depth, solve_budget=solve_budget)
        cache_all_scores(best_guess, branch_words, score_cache, branch_key, cache=cache)

    if mid_loop_publisher is not None and token is not None:
        mid_loop_publisher.record_inline(token)

    return (SOLVED, best_erd, best_max_remaining_depth, node_floor)


def min_expected_guesses(branch_words, cache, score_cache,
                          deadline=None, guesses=None,
                          policy=None, progress_callback=None,
                          cancel_check=None, heartbeat=None,
                          note_depth=None, budget=None,
                          subbranch_solver=None, mid_loop_publisher=None,
                          pattern_matrix=None, branch_floor_table=None):
    """
    Exact expected guesses to solve branch_words, playing optimally.

    With budget=None this is the classic unlimited-depth ERD.  With an integer
    budget it is depth-limited: the minimum expected guesses among strategies
    that are *guaranteed to win within `budget` guesses* — a candidate with a
    better average but a branch that can't finish in time is rejected.

    guesses: vocabulary of allowed guess words. None means answers-only.
    policy:  cache namespace (see VALID_ERD_POLICIES).  Defaults to ERD_ALL
             when guesses is supplied, ERD_ANSWERS otherwise.
    progress_callback: progress_callback(done, total, best_guess, best_erd),
             once per fully-evaluated top-level candidate (top level only).
    cancel_check / heartbeat / note_depth: threaded into every recursive
             call (see _solve_subset / evaluate_candidate).
    pattern_matrix: optional PatternMatrix (pattern_matrix.py) shared across
             the whole solve; when supplied, best-first ordering and
             ERD-lower-bound pruning run vectorized instead of per-candidate
             Python scans. None (the default) keeps the pure-Python path.

    Returns the expected-guesses cost, or None if the deadline/cancel fired or
    (when budgeted) `branch_words` is unsolvable within budget.  Partial results
    already written to score_cache are kept and valid either way.
    """
    # One table for the whole solve: the pool is fixed here, where the
    # search's own candidate list is known.  guesses=None means each node
    # draws candidates from its own branch words, a pool that changes as the
    # recursion descends and so cannot back a table.
    # Snapshot the caller's pool, keeping ITS order: order breaks ties among
    # equal-cost candidates and so decides which guess the search records,
    # whereas a table's order says nothing about its floors.  The snapshot is
    # what the whole recursion then draws from, so a later change to the list
    # the caller still holds cannot put the search and the floors out of step.
    if guesses is not None:
        guesses = tuple(guesses)
    if branch_floor_table is None:
        if guesses is not None:
            branch_floor_table = BranchFloorTable(
                guesses, cache=cache, pattern_matrix=pattern_matrix)
    elif not branch_floor_table.matches_pool(guesses):
        raise ValueError(
            "branch_floor_table's candidate pool is not this search's guess "
            "words; its floors would price strategies the search cannot play")
    res = _solve_subset(branch_words, cache, score_cache, budget, deadline,
                        guesses, policy, cancel_check, heartbeat,
                        note_depth, progress_callback, subbranch_solver,
                        branch_floor_table=branch_floor_table,
                        mid_loop_publisher=mid_loop_publisher,
                        pattern_matrix=pattern_matrix)
    if res in _ABORT_STATUSES:
        return None
    status, cost, _md, _budget_tainted = res
    if status == OVER_DEPTH_BUDGET or (budget is not None and cost == float('inf')):
        return None   # unsolvable within budget
    return cost


def verify_erd_cache(words, cache, score_cache, policy, max_nodes=2000):
    """Spot-check a cached ERD entry against its own cached subtree.

    For `words` and (recursively) every cached branch reachable through
    best_guess's response partition, recompute
    1 + sum_i (k_i/n) * sub_best_score_i
    from whatever branch entries are themselves cached, and compare it to
    the entry's own best_score. Every cached sub_best_score is >= 1, so a
    partial sum (some branches uncached) can only be <= the true total —
    if it already exceeds best_score, best_score itself must be wrong.

    This catches a cached entry that is internally inconsistent with its
    own subtree (e.g. left over from a different/older computation) without
    needing to recompute anything from scratch. It cannot prove a
    self-consistent value is the *correct* one, but a contradiction proves
    it is *wrong*.

    Returns a list of dicts (BFS order, root first), each with keys
    n, best_guess, best_score, reconstructed, complete, status, branch_key.
    status is one of:
      'uncached'   - no entry for this branch (only possible for the root)
      'match'      - reconstruction matches best_score and every k>=2
                     sub-branch was cached
      'incomplete' - some k>=2 sub-branch is uncached, but the partial
                     reconstruction is still consistent (<= best_score)
      'mismatch'   - reconstruction (partial or complete) contradicts
                     best_score
    Capped at max_nodes entries.
    """
    root_key = ScoreCache.encode_subset(words)
    hit = score_cache.read(root_key, policy)
    if hit is None:
        return [{'n': len(words), 'branch_key': root_key, 'status': 'uncached'}]

    visited = {root_key}
    queue = [(words, root_key, hit[0], hit[1])]
    report = []
    while queue and len(report) < max_nodes:
        cur_words, cur_key, cur_word, cur_score = queue.pop(0)
        n = len(cur_words)
        groups = cache.group_words(cur_word, cur_words)
        reconstructed = 1.0
        complete = True
        for sg in groups.values():
            k = len(sg)
            if k == 0:
                continue
            if k == 1 and sg[0] == cur_word:
                continue  # self: solved, contributes 0
            if k == 1:
                reconstructed += 1.0 / n
                continue
            sg_key = ScoreCache.encode_subset(sg)
            sub_hit = score_cache.read(sg_key, policy)
            if sub_hit is None:
                complete = False
                continue
            reconstructed += (k / n) * sub_hit[1]
            if sg_key not in visited and len(report) + len(queue) < max_nodes:
                visited.add(sg_key)
                queue.append((sg, sg_key, sub_hit[0], sub_hit[1]))

        if complete:
            status = 'match' if abs(reconstructed - cur_score) < 1e-9 else 'mismatch'
        else:
            status = 'mismatch' if reconstructed > cur_score + 1e-9 else 'incomplete'

        report.append({
            'n': n, 'best_guess': cur_word, 'best_score': cur_score,
            'reconstructed': reconstructed, 'complete': complete,
            'status': status, 'branch_key': cur_key,
        })
    return report
