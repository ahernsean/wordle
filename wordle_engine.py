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
            return f'{value:.1%}'
        return f'{value:0.4f}'


class GuessUniverse(Enum):
    """Which static word list a candidate guess is drawn from.

    Independent of ComplianceFilter below: a player can pick a guess from
    either list whether or not it satisfies the clues revealed so far
    (e.g. a recognizable answer-shaped word that the clues have already
    eliminated is still a legal — if unwise — guess in normal play).
    """
    ALL_WORDS = 'words'      # ~12,972 — every guessable word
    ALL_ANSWERS = 'answers'  # ~3,200 — words that can ever be a Wordle answer


class ComplianceFilter(Enum):
    """Whether a candidate must satisfy every clue revealed so far.

    Independent of GuessUniverse above — see its docstring. COMPLIANT is
    real Wordle hard mode's rule (green letters fixed in position, yellow
    letters present somewhere) applied to whichever universe is selected.
    """
    UNFILTERED = 'unfiltered'
    COMPLIANT = 'compliant'


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

    def __init__(self, answer_words, score_cache=None):
        self.answer_words = answer_words
        self.score_cache = score_cache
        self._cache = {}   # guess → {answer → pattern_int}

    def _ensure(self, guess):
        """Build the mapping for guess if not cached, persisting/reloading via SQLite.

        The decomposition (guess -> {answer: pattern}) is the same for every
        position in every session against this answer universe, so it is
        cached to disk as a compact byte blob — one byte per answer word, in
        canonical (self.answer_words) order — keyed only by guess+universe.
        """
        if guess in self._cache:
            return
        mapping = self._load_decomposition(guess)
        if mapping is None:
            mapping = {}
            for answer in self.answer_words:
                resp = calculate_response(guess, answer)
                mapping[answer] = _encode_response(resp)
            self._store_decomposition(guess, mapping)
        self._cache[guess] = mapping

    def _load_decomposition(self, guess):
        if not self.score_cache:
            return None
        blob = self.score_cache.read_decomposition(guess)
        if blob is None:
            return None
        return {answer: blob[i] for i, answer in enumerate(self.answer_words)}

    def _store_decomposition(self, guess, mapping):
        if not self.score_cache:
            return
        blob = bytes(mapping[answer] for answer in self.answer_words)
        self.score_cache.write_decomposition(guess, blob)

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


def cache_all_scores(word, subgroup, score_cache, subset_key, cache=None):
    """Derive and persist every ScoringMethod's view of `word` against `subgroup`.

    score_groups' methods all read off the same group-count partition, so
    once that partition is in hand for one method, the rest are nearly free
    to derive — persisting only the method an algorithm happened to be
    ranking by would be an arbitrary cutoff.

    This is the ONE place that enumerates ScoringMethod for caching purposes.
    Callers (compute_lookahead, min_expected_guesses, ...) just say "this
    word, for this subgroup, is worth remembering comprehensively" — they
    don't need to know what the full roster is, or to change when it grows.

    Hard-mode searches pass a MemoryScoreCache, whose minimal read/write
    interface deliberately omits write_scores: those ERD values are
    path-dependent and must never reach the persisted cross-game cache.
    Skip silently rather than require every transient cache to stub it out.
    """
    if not score_cache or not hasattr(score_cache, 'write_scores'):
        return
    for method, value in score_word_multi(word, subgroup, list(ScoringMethod),
                                           cache=cache).items():
        score_cache.write_scores(subset_key, [(word, value)], method.name.lower())


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
        subset_key = ScoreCache.encode_subset(self.current_words)
        cached = self.score_cache.read_scores(subset_key, method.name.lower())
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
            subset_key = ScoreCache.encode_subset(self.current_words)
            self.score_cache.write_scores(subset_key, scores, method.name.lower())

    @property
    def answer_set(self):
        """Cached set of current_words for O(1) membership tests."""
        if self._answer_set is None:
            self._answer_set = set(self.current_words)
        return self._answer_set

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
            If None, uses hard mode (subgroup words only).
            If provided, searches that list against each subgroup.

        Completed subgroup results are cached in the SQLite lookahead
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
                        cache_all_scores(best_word, subgroup, lc, subset_key,
                                         cache=cache)

                weighted_second += (cnt / n) * best

            combined = first_ent + weighted_second
            results.append((word, first_ent, weighted_second, combined))

        results.sort(key=lambda x: -x[3])
        return results


def min_expected_guesses(remaining, cache, score_cache,
                          deadline=None, guesses=None,
                          policy=None, progress_callback=None,
                          cancel_check=None):
    """
    Exact expected guesses to solve remaining words, playing optimally.

    guesses: vocabulary of allowed guess words. None means answers-only
             (restrict guesses to `remaining`).
    policy:  cache namespace under which the result is stored — one of
             ERD_ALL, ERD_ANSWERS, ERD_CONSTRAINED, ERD_ANSWERS_UNFILTERED
             (see VALID_ERD_POLICIES).  Defaults to ERD_ALL when guesses
             is supplied, ERD_ANSWERS otherwise.  Pass explicitly when the
             caller needs a different namespace (e.g. ERD_CONSTRAINED for
             constraint-compliant/hard mode).  An
             unrecognized policy raises ValueError — silently writing
             results into the wrong namespace would corrupt that mode's
             cache for every future game.
    progress_callback: optional progress_callback(done, total, best_word,
             best_erd), invoked once per fully-evaluated top-level
             candidate. Deliberately NOT threaded into the recursive
             calls below — passing it down would fire it once per
             candidate at every depth of the search tree, drowning the
             one signal a caller actually wants (how far the *requested*
             scan has gotten) in noise from scans the caller never asked
             to watch. A caller that wants visibility into a recursive
             scan should call min_expected_guesses on that subgroup
             directly and supply its own callback.
    cancel_check: optional zero-arg callable returning True once the
             caller has abandoned this computation (e.g. the user moved
             to a different branch). Unlike progress_callback, this IS
             threaded into every recursive call: cancellation needs to
             stop the search promptly at whatever depth it has reached,
             not just between top-level candidates. Checked alongside
             deadline — both are "stop early and return None" signals,
             but they answer different questions. deadline bounds *this*
             attempt's running time regardless of why it's running;
             cancel_check answers "is this attempt's answer even still
             wanted?" and can fire well before any deadline would.

    Returns None if the deadline is exceeded or cancel_check fires
    mid-computation; partial results already written to score_cache
    are kept and valid either way.
    """
    n = len(remaining)
    if n == 1:
        return 1.0

    if policy is None:
        policy = ERD_ALL if guesses is not None else ERD_ANSWERS
    elif policy not in VALID_ERD_POLICIES:
        raise ValueError(
            f"Unknown ERD policy {policy!r}; expected one of "
            f"{sorted(VALID_ERD_POLICIES)}"
        )
    guess_list = guesses if guesses is not None else remaining

    subset_key = ScoreCache.encode_subset(remaining)
    if score_cache:
        hit = score_cache.read(subset_key, policy)
        if hit is not None:
            return hit[1]

    if deadline is not None and time.time() > deadline:
        return None
    if cancel_check is not None and cancel_check():
        return None

    best_erd = float('inf')
    best_word = None

    for i, guess in enumerate(guess_list):
        # cache.group_words decomposes guess vs remaining via a persisted
        # per-guess pattern lookup table (~0.6us/word) instead of recomputing
        # calculate_response (~30us/word). Every new (cache-miss) subgroup
        # re-runs this loop over the full guess_list at every recursion
        # depth, so for non-answer guesses this difference is the dominant
        # cost of evaluating a new subgroup. The decomposition for any guess
        # is built once and persisted, so this is a one-time cost overall.
        if cache:
            groups = cache.group_words(guess, remaining)
        else:
            groups = defaultdict(list)
            for answer in remaining:
                pat = _encode_response(calculate_response(guess, answer))
                groups[pat].append(answer)

        # Admissible lower bound on this guess's cost — no recursion needed.
        # For any subgroup of size k, sub_erd >= 2 - 1/k: an oracle guess
        # that splits k words into k singletons (one of which is "self",
        # contributing 0) needs 1 + (k-1)/k = 2 - 1/k expected guesses, and
        # since sub_erd >= 1 for every part, no other partition of k can give
        # a lower weighted sum. Summing (k_i/n)*(2 - 1/k_i) over this guess's
        # groups (sizes sum to n) telescopes to 2 - G/n, where G = len(groups);
        # the "self" group {guess}, if present, contributes 0 instead of 1/n.
        # So cost >= 3 - (G + has_self)/n. If even this best case can't beat
        # best_erd, skip the guess entirely — exact for k <= 243 (a perfect
        # all-singleton split is achievable within the 243 response patterns),
        # and a valid-but-looser bound for k > 243.
        has_self = _ALL_GREEN_PATTERN in groups
        cost_lb = 3.0 - (len(groups) + (1 if has_self else 0)) / n
        if cost_lb >= best_erd:
            continue

        cost = 1.0
        aborted = False  # subscan returned None — deadline or cancel_check fired
        skip_guess = False
        # Largest subgroups first: they carry the highest weight (k/n) and
        # push `cost` up fastest, so the pruning check below fires after as
        # few subgroup evaluations as possible.
        for subgroup in sorted(groups.values(), key=len, reverse=True):
            k = len(subgroup)
            # When guess is the answer, the all-green response produces a
            # singleton {guess}. We've already solved it with this guess —
            # 0 additional guesses needed for that branch.
            if k == 1 and subgroup[0] == guess:
                continue
            if k >= n:
                # All remaining words gave the same response — this guess
                # provides zero information and cannot make progress.
                # Skip it to prevent infinite recursion.
                skip_guess = True
                break
            sub_erd = min_expected_guesses(
                subgroup, cache, score_cache, deadline, guesses,
                policy=policy, cancel_check=cancel_check,
            )
            if sub_erd is None:
                aborted = True
                break
            cost += (k / n) * sub_erd
            # Branch-and-bound: cost is non-decreasing (every remaining
            # subgroup contributes a positive amount), so if it already
            # meets or beats the best known result, no later subgroup can
            # make this guess competitive.
            if cost >= best_erd:
                break

        if skip_guess:
            continue
        if aborted:
            return None

        if cost < best_erd:
            best_erd = cost
            best_word = guess

        if progress_callback is not None:
            progress_callback(i + 1, len(guess_list), best_word, best_erd)

    if score_cache and best_word is not None:
        score_cache.write(subset_key, policy, best_word, best_erd)
        cache_all_scores(best_word, remaining, score_cache, subset_key, cache=cache)

    return best_erd
