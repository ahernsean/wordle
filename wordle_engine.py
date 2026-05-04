"""
wordle_engine.py - Core algorithms for Wordle solving.

No UI dependencies. All display/interaction is handled by the caller.
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

from adaptive_cache_sqlite import AdaptiveCacheSQLite
from adaptive_frontier import AdaptiveFrontier, WorkItemType


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



@dataclass(frozen=True)
class StateKey:
    """Canonical key for adaptive lookahead subproblems."""

    subset_bits: int
    remaining_depth: int
    policy: str


@dataclass
class ChildEdge:
    parent_guess_id: int
    child_state_key: StateKey
    path_weight: float


@dataclass
class CandidateRecord:
    guess_id: int
    immediate_entropy: float
    lower_bound: float
    upper_bound: float
    is_exact: bool
    children: list[ChildEdge]
    dirty_flags: set[str]


@dataclass
class StateNode:
    subset_bits: int
    remaining_depth: int
    policy: str
    lower_bound: float
    upper_bound: float
    best_guess_id: int | None
    is_exact: bool
    candidate_records: dict[int, CandidateRecord]
    dependents: set[tuple[StateKey, int]]
    retained_local_ranking: list[int]
    dirty_flags: set[str]
    generation: int


class AdaptiveFrontierSearch:
    """Stateful adaptive lookahead runner extracted from Solution."""

    def __init__(self, solution, top_words,
                 root_expansion_words=None,
                 global_candidates=None,
                 max_depth=3,
                 time_budget=300,
                 top_k=20,
                 persistence_policy='entropy_deep_v1',
                 progress_callback=None,
                 status_callback=None):
        self.solution = solution
        self.top_words = top_words
        self.root_expansion_words = (
            root_expansion_words
            if root_expansion_words is not None
            else top_words
        )
        self.global_candidates = global_candidates
        self.max_depth = max_depth
        self.time_budget = time_budget
        self.start_time = time.time()
        self.top_k = top_k
        self.persistence_policy = persistence_policy
        self.progress_callback = progress_callback
        self.status_callback = status_callback

        self.cache = solution.cache
        self.persistence = solution.adaptive_cache
        self.n = len(solution.current_words)
        self.deadline = time.time() + time_budget
        self.global_set = set(global_candidates) if global_candidates else set()
        self.answer_to_index = {w: i for i, w in enumerate(solution.all_answers)}
        self.index_to_answer = list(solution.all_answers)
        universe = list(solution.all_answers)
        if solution.all_guesses:
            for word in solution.all_guesses:
                if word not in self.answer_to_index:
                    universe.append(word)
        self.guess_to_id = {w: i for i, w in enumerate(universe)}
        self.policy = 'global' if self.global_set else 'hard'
        self.persistence_key = self.policy
        self.memo = {}

        self.root_subset_bits = 0
        self.results = []
        self.prune_threshold = 0.0
        self.prepared = {}
        self.generations = {}
        self.frontier = None
        self.intern_table = {}
        self.state_intern = {}
        self.pending = set()
        self.dirty = set()
        self.completed = 0
        self.timed_out = False
        self.total_words = len(top_words)
        self.activated_root_words = 0
        self.root_candidate_records = {}
        self.dormant_root_keys = []
        self.exploration_rate = 0.04
        self.expansion_batch_size = 3
        self.id_to_guess = {i: w for w, i in self.guess_to_id.items()}
        self.status_interval_s = 10.0
        self._last_status_emit = 0.0

    def words_to_bits(self, words):
        bits = 0
        for word in words:
            idx = self.answer_to_index.get(word)
            if idx is not None:
                bits |= (1 << idx)
        return bits

    def bits_to_words(self, bits):
        words = []
        while bits:
            lsb = bits & -bits
            idx = lsb.bit_length() - 1
            words.append(self.index_to_answer[idx])
            bits ^= lsb
        return words

    def partition_to_bits(self, candidate, subgroup_bits):
        subgroup_words = self.bits_to_words(subgroup_bits)
        subset_blob = AdaptiveCacheSQLite.encode_subset(subgroup_words)
        if self.persistence:
            cached = self.persistence.read_partition(
                subset_blob,
                candidate,
                self.persistence_key,
            )
            if cached is not None:
                return {int(pat): bits for pat, bits in cached.items()}
        if self.cache:
            grouped = self.cache.group_words(candidate, subgroup_words)
        else:
            grouped = defaultdict(list)
            for answer in subgroup_words:
                pat = _encode_response(calculate_response(candidate, answer))
                grouped[pat].append(answer)
        payload = {
            pat: self.words_to_bits(words)
            for pat, words in grouped.items()
        }
        if self.persistence:
            self.persistence.write_partition(
                subset_blob,
                candidate,
                self.persistence_key,
                {str(pat): bits for pat, bits in payload.items()},
            )
        return payload

    def get_or_create_state(self, subset_bits, remaining_depth, policy):
        key = StateKey(subset_bits, remaining_depth, policy)
        existing = self.state_intern.get(key)
        if existing is not None:
            return key, existing

        n = subset_bits.bit_count()
        if n <= 1 or remaining_depth <= 0:
            lower = upper = 0.0
            is_exact = True
        elif n == 2 and remaining_depth >= 1:
            lower = upper = 1.0
            is_exact = True
        else:
            lower = 0.0
            upper = remaining_depth * math.log2(n)
            is_exact = False

        node = StateNode(
            subset_bits=subset_bits,
            remaining_depth=remaining_depth,
            policy=policy,
            lower_bound=lower,
            upper_bound=upper,
            best_guess_id=None,
            is_exact=is_exact,
            candidate_records={},
            dependents=set(),
            retained_local_ranking=[],
            dirty_flags=set(),
            generation=0,
        )
        self.state_intern[key] = node
        return key, node

    def build_child_edges(self, parent_state_key, parent_guess_id, grouped):
        parent_size = parent_state_key.subset_bits.bit_count()
        edges = []
        for _pat, child_subset_bits in grouped.items():
            child_key, child_state = self.get_or_create_state(
                child_subset_bits,
                parent_state_key.remaining_depth - 1,
                parent_state_key.policy,
            )
            weight = 0.0
            if parent_size > 0:
                weight = child_subset_bits.bit_count() / parent_size
            edges.append(ChildEdge(parent_guess_id, child_key, weight))
            child_state.dependents.add((parent_state_key, parent_guess_id))
        return edges

    def _init_root_candidates(self):
        self.root_subset_bits = self.words_to_bits(self.solution.current_words)
        self.get_or_create_state(
            self.root_subset_bits,
            self.max_depth,
            self.policy,
        )
        for idx, (word, first_ent) in enumerate(
                self.root_expansion_words):
            grouped = self.partition_to_bits(word, self.root_subset_bits)
            upper = first_ent + sum(
                (bits.bit_count() / self.n) * math.log2(bits.bit_count())
                for bits in grouped.values()
                if bits.bit_count() > 1
            )
            key = (idx, word)
            self.prepared[key] = (word, first_ent, grouped, upper)
            self.generations[key] = 0
            self.intern_table[key] = StateKey(self.root_subset_bits,
                                              self.max_depth,
                                              self.policy)
            if idx >= len(self.top_words):
                self.dormant_root_keys.append(key)

    def _enqueue_initial_activation(self):
        def cutoff():
            return (self.prune_threshold if len(self.results) >= self.top_k
                    else float('-inf'))

        self.frontier = AdaptiveFrontier(
            get_generation=lambda k: self.generations[k],
            get_cutoff=cutoff,
        )
        for key, (_w, _f, _g, upper) in self.prepared.items():
            if key in self.dormant_root_keys:
                continue
            self.frontier.enqueue(
                WorkItemType.ACTIVATE_TOP_WORD,
                key,
                upper,
                0,
                upper,
            )
            self.pending.add(key)

    def _maybe_expand_roots(self):
        if not self.dormant_root_keys:
            return
        if random.random() >= self.exploration_rate:
            return
        random.shuffle(self.dormant_root_keys)
        batch = min(self.expansion_batch_size,
                    len(self.dormant_root_keys))
        for _ in range(batch):
            key = self.dormant_root_keys.pop()
            _w, _f, _g, upper = self.prepared[key]
            self.frontier.enqueue(
                WorkItemType.ACTIVATE_TOP_WORD,
                key,
                upper,
                self.generations[key],
                upper,
            )
            self.pending.add(key)

    def _emit_status(self, force=False):
        if not self.status_callback:
            return
        now = time.time()
        if (not force and self._last_status_emit
                and (now - self._last_status_emit) < self.status_interval_s):
            return
        top_rows = [
            (
                self.id_to_guess.get(guess_id, str(guess_id)),
                rec.lower_bound,
                rec.upper_bound,
                rec.is_exact,
            )
            for guess_id, rec in self.root_candidate_records.items()
        ]
        top_rows.sort(key=lambda row: (-row[1], row[0]))
        info = {
            'elapsed': now - self.start_time,
            'time_budget': self.time_budget,
            'frontier_size': len(self.frontier) if self.frontier else 0,
            'queued_items': len(self.pending),
            'activated_root_words': self.activated_root_words,
            'top_rows': top_rows,
            'prune_cutoff': self.prune_threshold,
        }
        self.status_callback(info)
        self._last_status_emit = now

    def best_from_subgroup(self, subgroup_bits, depth):
        key = StateKey(subgroup_bits, depth, self.policy)
        state_key, state_node = self.get_or_create_state(
            subgroup_bits, depth, self.policy
        )
        if key in self.memo:
            return self.memo[key]
        k = subgroup_bits.bit_count()
        if k <= 1 or depth <= 0:
            self.memo[key] = 0.0
            return 0.0
        if k == 2:
            self.memo[key] = 1.0
            return 1.0

        subgroup_words = self.bits_to_words(subgroup_bits)
        subset_blob = AdaptiveCacheSQLite.encode_subset(subgroup_words)
        if self.persistence:
            cached = self.persistence.read_state(subset_blob, depth, self.persistence_key)
            if cached and cached.is_exact:
                self.memo[key] = cached.lower_bound
                return cached.lower_bound

        candidates = list(subgroup_words)
        sg_set = set(subgroup_words)
        for word in self.global_set:
            if word not in sg_set:
                candidates.append(word)

        best_total = 0.0
        best_word = None
        for candidate in candidates:
            self._emit_status()
            guess_id = self.guess_to_id.get(candidate, -1)
            groups = self.partition_to_bits(candidate, subgroup_bits)
            edges = self.build_child_edges(state_key, guess_id, groups)
            ent = 0.0
            for edge in edges:
                s = edge.child_state_key.subset_bits.bit_count()
                if s:
                    p = edge.path_weight
                    ent -= p * math.log2(p)
            recursive = 0.0
            for edge in edges:
                m = edge.child_state_key.subset_bits.bit_count()
                if m <= 1:
                    continue
                if m == 2:
                    recursive += (2 / k)
                else:
                    recursive += edge.path_weight * self.best_from_subgroup(
                        edge.child_state_key.subset_bits,
                        edge.child_state_key.remaining_depth,
                    )
            total = ent + recursive
            is_exact = all(
                self.state_intern[edge.child_state_key].is_exact
                for edge in edges
            )
            state_node.candidate_records[guess_id] = CandidateRecord(
                guess_id=guess_id,
                immediate_entropy=ent,
                lower_bound=total,
                upper_bound=total,
                is_exact=is_exact,
                children=edges,
                dirty_flags=set(),
            )
            if best_word is None or total > best_total:
                best_total = total
                best_word = candidate

        if self.persistence:
            self.persistence.write_state(
                subset_blob,
                depth,
                self.persistence_key,
                lower_bound=best_total,
                upper_bound=best_total,
                best_guess_id=best_word,
                is_exact=True,
            )
        self.memo[key] = best_total
        state_node.lower_bound = best_total
        state_node.upper_bound = best_total
        state_node.is_exact = True
        return best_total

    def _recompute_candidate_bounds(self, candidate):
        lower = candidate.immediate_entropy
        upper = candidate.immediate_entropy
        is_exact = True
        for edge in candidate.children:
            child = self.state_intern[edge.child_state_key]
            lower += edge.path_weight * child.lower_bound
            upper += edge.path_weight * child.upper_bound
            is_exact = is_exact and child.is_exact
        candidate.lower_bound = lower
        candidate.upper_bound = upper
        candidate.is_exact = is_exact

    def _enqueue_refine_child(self, parent_state_key, parent_guess_id, edge, parent_upper):
        child_item_key = ('child', parent_state_key, parent_guess_id, edge.child_state_key)
        if child_item_key not in self.generations:
            self.generations[child_item_key] = 0
        self.frontier.enqueue(
            WorkItemType.REFINE_CHILD,
            child_item_key,
            edge.path_weight * parent_upper,
            self.generations[child_item_key],
            parent_upper,
        )
        self.pending.add(child_item_key)

    def _handle_activate_top_word(self, item):
        idx, word = item.item_key
        self.pending.discard(item.item_key)
        _word, first_ent, grouped, upper = self.prepared[item.item_key]
        root_key = StateKey(self.root_subset_bits, self.max_depth, self.policy)
        root_node = self.state_intern[root_key]
        guess_id = self.guess_to_id[word]
        edges = self.build_child_edges(root_key, guess_id, grouped)
        record = CandidateRecord(
            guess_id=guess_id,
            immediate_entropy=first_ent,
            lower_bound=first_ent,
            upper_bound=first_ent,
            is_exact=False,
            children=edges,
            dirty_flags=set(),
        )
        self._recompute_candidate_bounds(record)
        root_node.candidate_records[guess_id] = record
        self.root_candidate_records[guess_id] = record
        for edge in edges:
            child = self.state_intern[edge.child_state_key]
            if not child.is_exact:
                self._enqueue_refine_child(root_key, guess_id, edge, record.upper_bound)
        self.completed += 1
        self.activated_root_words += 1
        if self.progress_callback:
            self.progress_callback(idx, self.total_words, word, record.lower_bound)

    def _handle_refine_child(self, item):
        self.pending.discard(item.item_key)
        _kind, _parent_key, _guess_id, child_state_key = item.item_key
        child = self.state_intern[child_state_key]
        if child.is_exact:
            return
        old_lower = child.lower_bound
        old_upper = child.upper_bound
        exact_value = self.best_from_subgroup(
            child.subset_bits, child.remaining_depth
        )
        child.lower_bound = max(child.lower_bound, exact_value)
        child.upper_bound = min(child.upper_bound, exact_value)
        child.is_exact = abs(child.upper_bound - child.lower_bound) < 1e-12
        if child.lower_bound != old_lower or child.upper_bound != old_upper:
            for parent_state_key, parent_guess_id in child.dependents:
                parent_state = self.state_intern[parent_state_key]
                parent_record = parent_state.candidate_records.get(parent_guess_id)
                if parent_record:
                    self._recompute_candidate_bounds(parent_record)

    def _process_work_item(self, item):
        if item.item_type == WorkItemType.ACTIVATE_TOP_WORD:
            self._handle_activate_top_word(item)
        elif item.item_type == WorkItemType.REFINE_CHILD:
            self._handle_refine_child(item)

    def run(self):
        self._init_root_candidates()
        self._enqueue_initial_activation()

        while True:
            if time.time() > self.deadline:
                self.timed_out = True
                break
            item = self.frontier.pop()
            if item is None:
                break
            self._process_work_item(item)
            self._maybe_expand_roots()
            ranked = sorted(
                self.root_candidate_records.values(),
                key=lambda rec: -rec.lower_bound
            )
            if len(ranked) >= self.top_k:
                self.prune_threshold = ranked[self.top_k - 1].lower_bound
            self._emit_status()

        # Always emit one final snapshot at completion/timeout so the
        # caller sees end-state bounds even if interval throttling would
        # otherwise suppress it.
        self._emit_status(force=True)

        self.results = []
        for guess_id, rec in self.root_candidate_records.items():
            word = self.id_to_guess.get(guess_id, str(guess_id))
            deeper = max(0.0, rec.lower_bound - rec.immediate_entropy)
            self.results.append((word, rec.immediate_entropy, deeper, rec.lower_bound))
        self.results.sort(key=lambda row: -row[3])
        if len(self.results) > self.top_k:
            self.results = self.results[:self.top_k]

        status = (
            f'timeout after {self.completed} of {self.total_words}'
            if self.timed_out else
            'complete'
        )
        return self.results, status

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
                 cache=None, adaptive_cache=None):
        self.all_answers = answer_words
        self.all_guesses = all_guesses
        self.cache = cache
        self.adaptive_cache = adaptive_cache
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
                       first.cache,
                       first.adaptive_cache)
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

    def compute_adaptive_lookahead(self, top_words,
                                   root_expansion_words=None,
                                   global_candidates=None,
                                   max_depth=3,
                                   time_budget=300,
                                   top_k=20,
                                   persistence_policy='entropy_deep_v1',
                                   progress_callback=None,
                                   status_callback=None):
        """
        Adaptive-depth entropy lookahead with pruning.

        Recursively evaluates up to max_depth levels,
        using branch-and-bound pruning: for a subgroup of
        size k, the max remaining contribution is log2(k).

        root_expansion_words: optional broader pool of
            root candidates eligible for stochastic activation
            beyond initial top_words.
        global_candidates: top-N² words from step-1 solve.
            At each level, candidates = union of subgroup
            + global_candidates.
            If None, hard mode (subgroup only).
        max_depth: max levels beyond step 1 (default 3).
        time_budget: seconds before stopping (default 300).
        top_k: keep this many best results (default 20).
        progress_callback: called per first-guess word
            completed, with (index, total, word, score).
            Preserved for backward compatibility.
        status_callback(info): optional adaptive status
            callback. Receives a stable dict payload:
            {
              'elapsed', 'time_budget',
              'frontier_size', 'queued_items',
              'activated_root_words',
              'top_rows', 'prune_cutoff'
            }
            where top_rows is a list of
            (word, lower_bound, upper_bound, exact_flag).

        Returns sorted list of (word, step1, deeper,
        combined) tuples, best combined score first.
        Also returns a status string ('complete' or
        'timeout after N of M').
        """
        runner = AdaptiveFrontierSearch(
            solution=self,
            top_words=top_words,
            root_expansion_words=root_expansion_words,
            global_candidates=global_candidates,
            max_depth=max_depth,
            time_budget=time_budget,
            top_k=top_k,
            persistence_policy=persistence_policy,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        return runner.run()

    def compute_deep_lookahead(self, *args, **kwargs):
        """
        Deprecated compatibility shim.

        Use compute_adaptive_lookahead(...) instead.
        """
        return self.compute_adaptive_lookahead(
            *args, **kwargs
        )
