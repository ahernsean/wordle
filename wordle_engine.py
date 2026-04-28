"""
wordle_engine.py - Core algorithms for Wordle solving.

No UI dependencies. All display/interaction is handled by the caller.
"""

import math
import time
from collections import defaultdict
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoringMethod(Enum):
    UNWEIGHTED_AVG = auto()   # N/k
    WEIGHTED_AVG = auto()     # sum(n_i^2) / N
    ENTROPY_GAIN = auto()     # Shannon entropy (bits of info gained)
    MINIMAX = auto()          # max(n_i)
    PROB_FINISH = auto()      # P(next guess solves it)

    @property
    def label(self):
        labels = {
            ScoringMethod.UNWEIGHTED_AVG: "Unweighted avg group size",
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
    CURRENT_WORDLIST = auto()
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


def _decode_response(code):
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
    Returns a dict mapping response patterns to their counts.
    """
    groups = defaultdict(int)
    for word in words:
        response = calculate_response(test_word, word)
        key = ",".join(response)
        groups[key] += 1
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
                # Uncached word (e.g. fallback mode)
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


class AdaptivePartitionAdapter:
    """Thin adaptive-search wrapper around ResponseCache.

    Provides machine-stable partition payloads keyed by pattern id,
    where each child subgroup is represented as a bitset over the
    global answer-word index space.
    """

    def __init__(self, answer_words, response_cache):
        self.answer_words = answer_words
        self.response_cache = response_cache
        self._word_to_index = {
            word: idx for idx, word in enumerate(answer_words)
        }
        self._subset_word_cache = {}      # subset_bits -> [words]
        self._partition_cache = {}        # (guess, subset_bits) -> payload

    def bits_to_words(self, subset_bits):
        """Convert subset bitset to list of words (memoized)."""
        cached = self._subset_word_cache.get(subset_bits)
        if cached is not None:
            return cached

        words = []
        bits = subset_bits
        while bits:
            low_bit = bits & -bits
            idx = low_bit.bit_length() - 1
            words.append(self.answer_words[idx])
            bits ^= low_bit

        self._subset_word_cache[subset_bits] = words
        return words

    def words_to_bits(self, words):
        """Convert words iterable to subset bitset."""
        bits = 0
        for word in words:
            idx = self._word_to_index.get(word)
            if idx is not None:
                bits |= (1 << idx)
        return bits

    def partition(self, guess_word, subset_bits):
        """Return adaptive partition payload for (guess, subset_bits).

        Stable machine format:
        {
            "pattern_to_subset_bits": {pattern_id: child_subset_bits},
            "pattern_counts": {pattern_id: child_count},
            "pattern_weights": {pattern_id: child_count / parent_count},
            "immediate_entropy": float,
            "total_count": int,
        }
        """
        cache_key = (guess_word, subset_bits)
        cached = self._partition_cache.get(cache_key)
        if cached is not None:
            return cached

        subset_words = self.bits_to_words(subset_bits)
        groups = self.response_cache.group_words(
            guess_word, subset_words
        )

        pattern_to_subset_bits = {}
        pattern_counts = {}
        total_count = len(subset_words)

        for pattern_id in sorted(groups):
            child_words = groups[pattern_id]
            child_bits = self.words_to_bits(child_words)
            pattern_to_subset_bits[pattern_id] = child_bits
            pattern_counts[pattern_id] = len(child_words)

        if total_count == 0:
            pattern_weights = {
                pattern_id: 0.0 for pattern_id in pattern_counts
            }
        else:
            pattern_weights = {
                pattern_id: pattern_counts[pattern_id] / total_count
                for pattern_id in pattern_counts
            }

        immediate_entropy = score_groups(
            pattern_counts, ScoringMethod.ENTROPY_GAIN
        )

        payload = {
            "pattern_to_subset_bits": pattern_to_subset_bits,
            "pattern_counts": pattern_counts,
            "pattern_weights": pattern_weights,
            "immediate_entropy": immediate_entropy,
            "total_count": total_count,
        }
        self._partition_cache[cache_key] = payload
        return payload


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_groups(groups, method=ScoringMethod.UNWEIGHTED_AVG):
    """
    Score a word's group partition.

    - UNWEIGHTED_AVG: N/k. Lower is better.
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
    k = len(sizes)

    if method == ScoringMethod.UNWEIGHTED_AVG:
        return n / k

    elif method == ScoringMethod.WEIGHTED_AVG:
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


def score_word(word, remaining_words, method=ScoringMethod.UNWEIGHTED_AVG,
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
    """Theoretical maximum entropy for n remaining words: log2(n).
    A guess achieves this only if it perfectly partitions all
    n words into n singleton groups.
    """
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

    If all_guesses is provided, falling back to it when the answer
    list is exhausted (word not in answer list).
    """

    def __init__(self, answer_words, all_guesses=None,
                 cache=None):
        self.all_answers = answer_words
        self.all_guesses = all_guesses
        self.cache = cache
        self.reset()

    def reset(self):
        self.current_words = self.all_answers[:]
        self._answer_set = None
        self.guesses = []
        self.scores = []
        self.scores_method = None
        self.scores_updated = False
        self.answer_word = None
        self.fallback_active = False

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
        self.scores_updated = False

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

    def include_letters(self, letters):
        """Keep only words containing all specified letters."""
        for letter in letters:
            self.current_words = [
                w for w in self.current_words if letter in w
            ]
        self._answer_set = None
        self.scores_updated = False

    def exclude_letters(self, letters):
        """Remove words containing any of the specified letters."""
        for letter in letters:
            self.current_words = [
                w for w in self.current_words if letter not in w
            ]
        self._answer_set = None
        self.scores_updated = False

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
                       first.cache)
        combined = set()
        for soln in solutions:
            if len(soln.current_words) > 1:
                combined.update(soln.current_words)
        out.current_words = sorted(combined)
        return out

    def compute_scores(self, input_wordlist,
                       method=ScoringMethod.UNWEIGHTED_AVG,
                       progress_callback=None):
        """
        Score every word in input_wordlist against current_words.
        Returns a sorted list of (word, score) tuples (best first).
        """
        results = []
        for word in input_wordlist:
            s = score_word(
                word, self.current_words, method,
                progress_callback, cache=self.cache
            )
            results.append((word, s))
        results.sort(key=method.sort_key())
        self.scores = results
        self.scores_method = method
        self.scores_updated = True
        return results

    def compute_scores_multi(self, input_wordlist, methods,
                             progress_callback=None):
        """
        Score every word under multiple methods in a single pass.
        Computes group counts once per word, then scores with each
        method. Returns a list of (word, {method: score}) tuples,
        sorted by the first method in the list.
        """
        results = []
        for word in input_wordlist:
            scores = score_word_multi(
                word, self.current_words, methods,
                progress_callback, cache=self.cache
            )
            results.append((word, scores))
        primary = methods[0]
        results.sort(key=lambda x: primary.sort_key()(
            (x[0], x[1][primary])
        ))
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

        total_callback(n): called once with total work units.
        progress_callback(): called per work unit.

        Returns sorted list of (word, step1, step2, combined)
        tuples, best combined score first.
        """
        method = ScoringMethod.ENTROPY_GAIN
        n = len(self.current_words)
        full_mode = second_step_words is not None
        cache = self.cache

        # Phase 1: compute group partitions, count work
        word_data = []
        total_work = 0
        for word, first_ent in top_words:
            if cache:
                grouped = cache.group_words(
                    word, self.current_words
                )
            else:
                groups = calculate_group_counts(
                    word, self.current_words
                )
                grouped = {}
                for pattern, cnt in groups.items():
                    resp = pattern.split(",")
                    sub = apply_guess(
                        self.current_words, word, resp
                    )
                    grouped[pattern] = sub

            big_groups = sum(
                1 for ws in grouped.values()
                if len(ws) > 2
            )
            if full_mode:
                work = big_groups * len(second_step_words)
            else:
                work = sum(
                    len(ws) for ws in grouped.values()
                    if len(ws) > 2
                )
            total_work += work
            word_data.append((word, first_ent, grouped))

        if total_callback:
            total_callback(total_work)

        # Phase 2: second-step evaluation
        results = []
        for word, first_ent, grouped in word_data:
            weighted_second = 0.0

            for pattern, subgroup in grouped.items():
                count = len(subgroup)
                if count <= 1:
                    continue

                if count == 2:
                    weighted_second += (2 / n) * 1.0
                    continue

                candidates = (second_step_words
                              if full_mode else subgroup)
                best = 0.0
                for candidate in candidates:
                    if progress_callback:
                        progress_callback()
                    s = score_word(
                        candidate, subgroup, method,
                        cache=cache
                    )
                    best = max(best, s)

                weighted_second += (count / n) * best

            combined = first_ent + weighted_second
            results.append(
                (word, first_ent, weighted_second, combined)
            )

        results.sort(key=lambda x: -x[3])
        return results

    def compute_deep_lookahead(self, top_words,
                               global_candidates=None,
                               max_depth=3,
                               time_budget=300,
                               top_k=20,
                               progress_callback=None):
        """
        Adaptive-depth entropy lookahead with pruning.

        Recursively evaluates up to max_depth levels,
        using branch-and-bound pruning: for a subgroup of
        size k, the max remaining contribution is log2(k).

        global_candidates: top-N² words from step-1 solve.
            At each level, candidates = union of subgroup
            + global_candidates.
            If None, hard mode (subgroup only).
        max_depth: max levels beyond step 1 (default 3).
        time_budget: seconds before stopping (default 300).
        top_k: keep this many best results (default 20).
        progress_callback: called per first-guess word
            completed, with (index, total, word, score).

        Returns sorted list of (word, step1, deeper,
        combined) tuples, best combined score first.
        Also returns a status string ('complete' or
        'timeout after N of M').
        """
        cache = self.cache
        n = len(self.current_words)
        deadline = time.time() + time_budget
        global_set = (set(global_candidates)
                      if global_candidates else set())
        adaptive_partitions = None
        if cache:
            adaptive_partitions = AdaptivePartitionAdapter(
                self.all_answers, cache
            )

        full_bits = 0
        if adaptive_partitions:
            full_bits = adaptive_partitions.words_to_bits(
                self.current_words
            )

        # Pruning threshold: 20th best score found.
        # Starts at 0 (no pruning until we have top_k).
        results = []
        prune_threshold = 0.0

        def _update_threshold():
            nonlocal prune_threshold
            if len(results) >= top_k:
                results.sort(key=lambda x: -x[3])
                prune_threshold = results[top_k - 1][3]

        def _best_from_subgroup(subgroup, depth):
            """Return the best total weighted entropy
            achievable from this subgroup over `depth`
            remaining levels."""
            k = len(subgroup)
            if k <= 1:
                return 0.0
            if k == 2:
                return 1.0
            if depth <= 0:
                return 0.0

            # Build candidate list: subgroup + globals
            sg_set = set(subgroup)
            candidates = list(subgroup)
            for w in global_set:
                if w not in sg_set:
                    candidates.append(w)

            best_total = 0.0
            for candidate in candidates:
                # Get this candidate's partition
                if adaptive_partitions:
                    subgroup_bits = adaptive_partitions.words_to_bits(
                        subgroup
                    )
                    partition_data = adaptive_partitions.partition(
                        candidate, subgroup_bits
                    )
                    groups = {
                        pattern_id: adaptive_partitions.bits_to_words(
                            child_bits
                        )
                        for pattern_id, child_bits in
                        partition_data["pattern_to_subset_bits"].items()
                    }
                else:
                    gc = calculate_group_counts(
                        candidate, subgroup
                    )
                    groups = {}
                    for pat, cnt in gc.items():
                        resp = pat.split(",")
                        sub = apply_guess(
                            subgroup, candidate, resp
                        )
                        groups[pat] = sub

                # Entropy at this level
                sizes = [len(ws) for ws in
                         groups.values()]
                ent = 0.0
                for s in sizes:
                    p = s / k
                    if p > 0:
                        ent -= p * math.log2(p)

                # Upper bound: ent + sum of log2
                # bounds on children
                if depth > 1:
                    upper = ent
                    for ws in groups.values():
                        m = len(ws)
                        if m > 1:
                            upper += (m / k) * math.log2(m)
                    if upper <= best_total:
                        continue  # prune candidate

                # Recurse into children
                recursive = 0.0
                for pat, sub_sub in groups.items():
                    m = len(sub_sub)
                    if m <= 1:
                        continue
                    if m == 2:
                        recursive += (2 / k) * 1.0
                        continue
                    sub_score = _best_from_subgroup(
                        sub_sub, depth - 1
                    )
                    recursive += (m / k) * sub_score

                total = ent + recursive
                best_total = max(best_total, total)

            return best_total

        # Main loop over first-guess candidates
        total_words = len(top_words)
        completed = 0
        timed_out = False

        for idx, (word, first_ent) in enumerate(
                top_words):
            # Time check
            if time.time() > deadline:
                timed_out = True
                break

            # Get step-1 groups
            if adaptive_partitions:
                partition_data = adaptive_partitions.partition(
                    word, full_bits
                )
                grouped = {
                    pattern_id: adaptive_partitions.bits_to_words(
                        child_bits
                    )
                    for pattern_id, child_bits in
                    partition_data["pattern_to_subset_bits"].items()
                }
            else:
                gc = calculate_group_counts(
                    word, self.current_words
                )
                grouped = {}
                for pat, cnt in gc.items():
                    resp = pat.split(",")
                    sub = apply_guess(
                        self.current_words, word, resp
                    )
                    grouped[pat] = sub

            # Upper bound for this word
            upper = first_ent
            for ws in grouped.values():
                m = len(ws)
                if m > 1:
                    upper += (m / n) * math.log2(m)

            if (len(results) >= top_k
                    and upper <= prune_threshold):
                completed += 1
                if progress_callback:
                    progress_callback(
                        idx, total_words, word, None
                    )
                continue  # prune entire word

            # Evaluate each group
            weighted_deeper = 0.0
            # Track partial upper bound for mid-word
            # pruning: replace remaining groups' actual
            # scores with log2 bounds
            remaining_upper = upper - first_ent
            actual_so_far = 0.0
            pruned_mid = False

            # Sort groups largest first (evaluate the
            # most informative groups first for better
            # pruning)
            sorted_groups = sorted(
                grouped.items(),
                key=lambda x: -len(x[1])
            )

            for pat, subgroup in sorted_groups:
                count = len(subgroup)
                if count <= 1:
                    continue

                # Remove this group's upper bound
                # contribution
                if count > 1:
                    remaining_upper -= (
                        (count / n) * math.log2(count)
                    )

                if count == 2:
                    contrib = (2 / n) * 1.0
                    actual_so_far += contrib
                    weighted_deeper += contrib
                else:
                    sub_score = _best_from_subgroup(
                        subgroup, max_depth
                    )
                    contrib = (count / n) * sub_score
                    actual_so_far += contrib
                    weighted_deeper += contrib

                # Mid-word prune check
                refined = (first_ent + actual_so_far
                           + remaining_upper)
                if (len(results) >= top_k
                        and refined <= prune_threshold):
                    pruned_mid = True
                    break

            if not pruned_mid:
                combined = first_ent + weighted_deeper
                results.append(
                    (word, first_ent,
                     weighted_deeper, combined)
                )
                _update_threshold()

            completed += 1
            if progress_callback:
                score = (None if pruned_mid
                         else first_ent + weighted_deeper)
                progress_callback(
                    idx, total_words, word, score
                )

        results.sort(key=lambda x: -x[3])
        results = results[:top_k]

        if timed_out:
            status = (f'timeout after {completed}'
                      f' of {total_words}')
        else:
            status = 'complete'

        return results, status
