"""
wordle_engine.py - Core algorithms for Wordle solving.

No UI dependencies. All display/interaction is handled by the caller.
"""

import math
import time
from collections import defaultdict
from enum import Enum, auto

from cache_sqlite import ScoreCache


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoringMethod(Enum):
    WEIGHTED_AVG = auto()     # sum(n_i^2) / N  — lower is better
    ENTROPY_GAIN = auto()     # Shannon entropy (bits of info gained)
    MINIMAX = auto()          # max(n_i)
    PROB_FINISH = auto()      # P(next guess solves it)

    @property
    def label(self):
        labels = {
            ScoringMethod.WEIGHTED_AVG:   "Weighted avg remaining",
            ScoringMethod.ENTROPY_GAIN:   "Entropy gain (bits)",
            ScoringMethod.MINIMAX:        "Worst-case group size",
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
        if self == ScoringMethod.MINIMAX:
            return str(int(value))
        if self == ScoringMethod.PROB_FINISH:
            return f'{value:.1%}'
        return f'{value:0.4f}'


class InputSet(Enum):
    ALL_GUESSES = auto()
    HARD_MODE = auto()        # real Wordle hard mode: satisfies all constraints
    CURRENT_WORDLIST = auto() # restrict to remaining possible answers (strictest)
    SOLVED_WORDS = auto()


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

class ResponseCache:
    """Lazily caches word-to-pattern mappings for each guess word.

    For each guess word G, stores a dict mapping every answer word
    to its encoded response pattern (int 0-242). This is built
    once per guess word on first access, then all subsequent
    scoring against any subset is just dict lookups and counting.
    """

    def __init__(self, answer_words):
        self.answer_words = answer_words
        self._cache = {}   # guess → {answer → pattern_int}

    def _ensure(self, guess):
        """Build the mapping for guess if not cached."""
        if guess not in self._cache:
            mapping = {}
            for answer in self.answer_words:
                resp = calculate_response(guess, answer)
                mapping[answer] = _encode_response(resp)
            self._cache[guess] = mapping

    def group_counts(self, guess, subset):
        """Return {pattern_int: count} for guess vs subset."""
        self._ensure(guess)
        mapping = self._cache[guess]
        counts = defaultdict(int)
        for word in subset:
            if word in mapping:
                counts[mapping[word]] += 1
            else:
                resp = calculate_response(guess, word)
                counts[_encode_response(resp)] += 1
        return counts

    def group_words(self, guess, subset):
        """Return {pattern_int: [words]} for guess vs subset."""
        self._ensure(guess)
        mapping = self._cache[guess]
        groups = defaultdict(list)
        for word in subset:
            if word in mapping:
                groups[mapping[word]].append(word)
            else:
                resp = calculate_response(guess, word)
                groups[_encode_response(resp)].append(word)
        return groups

    def is_cached(self, guess):
        """Check if a guess word is already cached."""
        return guess in self._cache


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_groups(groups, method=ScoringMethod.ENTROPY_GAIN):
    """
    Score a word's group partition.

    - WEIGHTED_AVG: sum(n_i^2)/N. Lower is better.
    - ENTROPY_GAIN: Shannon entropy in bits. Higher is better.
    - MINIMAX: max(n_i). Lower is better.
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

    elif method == ScoringMethod.MINIMAX:
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

    If all_guesses is provided, falls back to it when the answer
    list is exhausted (word not in answer list).
    """

    def __init__(self, answer_words, all_guesses=None,
                 cache=None, score_cache=None):
        self.all_answers = answer_words
        self.all_guesses = all_guesses
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
        """Transparently load full-game scores from SQLite into word_scores."""
        if not self.score_cache or not self._is_full_game():
            return
        if method in self._db_loaded_methods:
            return
        self._db_loaded_methods.add(method)
        cached = self.score_cache.read_scores(method.name.lower())
        if cached:
            for w, s in cached:
                self.word_scores.setdefault(w, {})[method] = s

    def _persist_scores(self, method):
        """Transparently write full-game scores from word_scores to SQLite."""
        if not self.score_cache or not self._is_full_game():
            return
        scores = [(w, s[method]) for w, s in self.word_scores.items()
                  if method in s]
        if scores:
            self.score_cache.write_scores(scores, method.name.lower())

    @property
    def answer_set(self):
        """Cached set of current_words for O(1) membership tests."""
        if self._answer_set is None:
            self._answer_set = set(self.current_words)
        return self._answer_set

    def apply_guess(self, try_word, response):
        """
        Apply a guess and its response, filtering the word list.

        If the result is empty and all_guesses is available, replays
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
                and self.all_guesses
                and not self.fallback_active):
            words = self.all_guesses[:]
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

    def hard_mode_words(self, all_guesses):
        """
        Return words from all_guesses consistent with all prior responses.
        This is real Wordle hard mode: must satisfy green/yellow constraints
        but is not restricted to remaining answers.
        """
        words = list(all_guesses)
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
                       first.all_guesses,
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
                          total_callback=None,
                          progress_callback=None):
        """
        Two-step entropy lookahead on (word, first_entropy) pairs.

        For each candidate first guess, computes the weighted average
        of the best second-step entropy across all response groups.

        second_step_words: word list to search for best second guess.
            If None, uses hard mode (subgroup words only).
            If provided, searches that list against each subgroup.

        Completed subgroup results are cached in the SQLite lookahead
        cache and reused across sessions.

        total_callback(n): called once with total work units.
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

        # Phase 1: compute group partitions, count work units
        word_data = []
        total_work = 0
        for word, first_ent in top_words:
            if cache:
                grouped = cache.group_words(word, self.current_words)
            else:
                grouped = defaultdict(list)
                for answer in self.current_words:
                    pat = _encode_response(calculate_response(word, answer))
                    grouped[pat].append(answer)

            work = 0
            for subgroup in grouped.values():
                cnt = len(subgroup)
                if cnt <= 2:
                    continue
                if lc:
                    subset_key = ScoreCache.encode_subset(subgroup)
                    if lc.read(subset_key, policy) is not None:
                        continue  # cache hit — no scan needed
                work += (len(second_step_words)
                         if full_mode else len(subgroup))
            total_work += work
            word_data.append((word, first_ent, grouped))

        if total_callback:
            total_callback(total_work)

        # Phase 2: second-step evaluation
        results = []
        for word, first_ent, grouped in word_data:
            weighted_second = 0.0

            for _pat, subgroup in grouped.items():
                cnt = len(subgroup)
                if cnt <= 1:
                    continue

                if cnt == 2:
                    weighted_second += (2 / n) * 1.0
                    continue

                # Check SQLite cache first
                best = None
                if lc:
                    subset_key = ScoreCache.encode_subset(subgroup)
                    hit = lc.read(subset_key, policy)
                    if hit is not None:
                        _best_word, best = hit

                if best is None:
                    subset_key = ScoreCache.encode_subset(subgroup) if lc else None
                    candidates = (second_step_words
                                  if full_mode else subgroup)
                    best = 0.0
                    best_word = None
                    for candidate in candidates:
                        if progress_callback:
                            progress_callback()
                        s = score_word(
                            candidate, subgroup, method,
                            cache=cache
                        )
                        if s > best:
                            best = s
                            best_word = candidate
                    if lc and best_word is not None and subset_key is not None:
                        lc.write(subset_key, policy, best_word, best)

                weighted_second += (cnt / n) * best

            combined = first_ent + weighted_second
            results.append((word, first_ent, weighted_second, combined))

        results.sort(key=lambda x: -x[3])
        return results
