"""Pattern matrix: precomputed response patterns for all (guess, answer) pairs.

This module is the sole NumPy import point in the engine. NumPy is a hard
requirement on every deployment target (Pythonista bundles 1.22.3, the API
floor). The engine's pure-Python implementations remain permanently — not as
a runtime fallback but as the reference implementation the vectorized path is
tested against; selecting them is a caller choice (pattern_matrix=None).
"""
import collections
import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger("wordle")


# Parallel arrays returned by candidate_stats(), one entry per guess-word (matrix row).
# All fields share a common axis-0 length of n_guesses; consumers index them by the
# guess-word's matrix row number, not by position in any candidate_list.
CandidateStats = collections.namedtuple(
    'CandidateStats',
    ['group_count', 'has_self', 'cost_lower_bound',
     'sum_squared_group_sizes', 'max_group_size', 'entropy_gain'],
)


# Guess rows processed per bincount call. Bounds the per-call transient at
# 2 × _COUNT_CHUNK_ROWS × n × 4 bytes regardless of branch size (n = branch word count).
_COUNT_CHUNK_ROWS = 1024


def _compute_answer_list_id(answer_words):
    """SHA-256 identity of the answer universe (matches ScoreCache._ensure_answer_list)."""
    return hashlib.sha256("\n".join(answer_words).encode()).hexdigest()


def _compute_guess_vocabulary_id(guess_words):
    """SHA-256 identity of the guess vocabulary, order-sensitive like
    _compute_answer_list_id (row index assignment depends on guess order)."""
    return hashlib.sha256("\n".join(guess_words).encode()).hexdigest()


class PatternMatrix:
    """Response patterns for every (guess, answer) pair.

    matrix[g, a] is the encoded response pattern (0-242) of guess word g
    (row, canonical guess-list order) against answer word a (column,
    canonical answer-list order) — exactly the byte ResponseCache stores
    per guess, all guesses stacked.
    """

    def __init__(self, matrix, guess_words, answer_words):
        self.matrix = matrix
        self.n_guesses = matrix.shape[0]
        self.n_answers = matrix.shape[1]
        self.answer_list_id = _compute_answer_list_id(answer_words)
        self._guess_index = {w: i for i, w in enumerate(guess_words)}
        self._answer_index = {w: i for i, w in enumerate(answer_words)}

    @classmethod
    def build(cls, guess_words, answer_words, score_cache=None):
        """Build the matrix from score_cache decomposition blobs, computing and
        writing back any missing row via calculate_response + _encode_response."""
        from wordle_engine import calculate_response, _encode_response
        n_guesses = len(guess_words)
        n_answers = len(answer_words)
        matrix = np.empty((n_guesses, n_answers), dtype=np.uint8)
        for i, guess in enumerate(guess_words):
            blob = score_cache.read_decomposition(guess) if score_cache else None
            if blob is None:
                blob = bytes(
                    _encode_response(calculate_response(guess, answer))
                    for answer in answer_words
                )
                if score_cache:
                    score_cache.write_decomposition(guess, blob)
            matrix[i] = np.frombuffer(blob, dtype=np.uint8)
        return cls(matrix, guess_words, answer_words)

    def save(self, path):
        """Persist the matrix as a .npy file at path.

        Appends '.npy' if the path does not already end with it, matching
        the normalization np.save applies internally, so save(p) and
        load(p, ...) always agree on the filename regardless of whether the
        caller includes the extension.
        """
        path = str(path)
        if not path.endswith('.npy'):
            path = path + '.npy'
        np.save(path, self.matrix)

    @classmethod
    def load(cls, path, guess_words, answer_words):
        """Load from path with mmap_mode='r'; return None on file-not-found or shape mismatch.

        Applies the same '.npy' suffix normalization as save(), so load(p, ...)
        finds whatever save(p) wrote. Shape mismatch means the file was built
        for a different vocabulary; callers should rebuild and re-save.
        """
        path = str(path)
        if not path.endswith('.npy'):
            path = path + '.npy'
        try:
            matrix = np.load(path, mmap_mode='r')
        except (FileNotFoundError, ValueError, OSError):
            return None
        if matrix.shape != (len(guess_words), len(answer_words)):
            return None
        return cls(matrix, guess_words, answer_words)

    @classmethod
    def load_or_build(cls, cache_path, guess_words, answer_words, score_cache):
        """This process's PatternMatrix: load()ed from disk, or build()+save()d
        on a miss.

        The .npy path sits alongside cache_path and embeds both the
        answer-list identity and the guess-vocabulary identity, so neither a
        different answer universe nor a different (or reordered) guess list
        of the same length ever loads a stale matrix — the load() shape
        check alone only catches a different row/column *count*, not two
        same-size, different-content vocabularies. Multiple processes (swarm
        workers, an interactive session) may race to build on a cold start or
        a rebuild; each writes to its own PID-suffixed temp file and renames
        into place, so a racing build only wastes CPU — it can never
        truncate a file another process still has mmap'd, which an in-place
        save() could.
        """
        matrix_dir = os.path.dirname(os.path.abspath(cache_path))
        matrix_path = os.path.join(
            matrix_dir,
            f'pattern_matrix_{score_cache.answer_list_id}'
            f'_{_compute_guess_vocabulary_id(guess_words)}')
        matrix = cls.load(matrix_path, guess_words, answer_words)
        if matrix is not None:
            return matrix
        matrix = cls.build(guess_words, answer_words, score_cache=score_cache)
        tmp_path = f'{matrix_path}.tmp{os.getpid()}'
        tmp_npy_path = f'{tmp_path}.npy'
        try:
            matrix.save(tmp_path)
            os.replace(tmp_npy_path, f'{matrix_path}.npy')
        except OSError as exc:
            # Persisting is an optimization, not a durability requirement:
            # the matrix just built is still returned and usable this
            # session, so a write failure here (disk full, or on iOS a
            # transient iCloud File Provider Storage lock — see
            # ScoreCache.checkpoint) only costs a rebuild next time, not
            # this one. Clean up any partial temp file rather than leaking
            # it across every future worker restart.
            logger.warning("PatternMatrix persist failed for %s: %s",
                           matrix_path, exc)
            try:
                os.unlink(tmp_npy_path)
            except OSError:
                pass
        return matrix

    def guess_index(self, word):
        """Row index of word in the guess vocabulary; KeyError if word is unknown."""
        return self._guess_index[word]

    def answer_indices(self, words):
        """Branch words as a column-index int32 array.

        Raises KeyError if any word is not in the answer universe. Swarm
        branches are always answer subsets, so this is never reached there;
        the fallback for interactive mode (unknown words) lives in §5.
        """
        return np.array([self._answer_index[w] for w in words], dtype=np.int32)

    def counts_for_all_candidates(self, branch_indices):
        """(n_guesses, 243) int32: counts[g, p] = number of branch words whose
        response to guess-word g encodes to pattern p.

        Chunked over guess rows so the per-call transient stays bounded at
        ~2 × _COUNT_CHUNK_ROWS × n × 4 bytes regardless of branch size.
        The int32 offset trick (row * 243 + pattern) maps every (row, pattern)
        pair to a unique 1-D bin, letting np.bincount handle a full chunk in
        one C-speed pass.
        """
        counts = np.empty((self.n_guesses, 243), dtype=np.int32)
        row_offsets = np.arange(_COUNT_CHUNK_ROWS, dtype=np.int32)[:, None] * 243
        for start in range(0, self.n_guesses, _COUNT_CHUNK_ROWS):
            stop = min(start + _COUNT_CHUNK_ROWS, self.n_guesses)
            rows = stop - start
            branch_patterns = self.matrix[start:stop, branch_indices].astype(np.int32)
            offset_patterns = branch_patterns + row_offsets[:rows]
            counts[start:stop] = np.bincount(
                offset_patterns.ravel(), minlength=rows * 243
            ).reshape(rows, 243)
        return counts

    def patterns_for_candidates(self, candidate_indices, branch_indices):
        """Raw (len(candidate_indices), n) uint8 slice of response pattern values."""
        rows = np.asarray(candidate_indices, dtype=np.int32)
        return self.matrix[rows][:, branch_indices]

    def group_words(self, guess, branch_words, branch_indices):
        """{pattern_int: [words]} for guess against branch_words — the
        vectorized twin of ResponseCache.group_words, identical in keys,
        values, and iteration order.

        branch_indices must be index-aligned with branch_words (as returned
        by answer_indices(branch_words)). One matrix row read plus a stable
        argsort replaces the per-word Python loop; a stable sort on the
        already-ascending-by-pattern boundaries preserves branch order
        within each group, so each present pattern's first sorted position
        is also its first-appearance position while walking branch_words —
        exactly the insertion order the Python loop produces.
        """
        guess_row = self.guess_index(guess)
        branch_patterns = self.matrix[guess_row, branch_indices]
        order = np.argsort(branch_patterns, kind='mergesort')
        sorted_patterns = branch_patterns[order]

        # Ascending sort means every diff is >= 0, so uint8 subtraction
        # never wraps around here.
        boundaries = np.flatnonzero(np.diff(sorted_patterns)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(order)]))

        segments = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            segment_order = order[start:end]
            pattern = int(sorted_patterns[start])
            first_index = int(segment_order[0])
            words = [branch_words[i] for i in segment_order.tolist()]
            segments.append((first_index, pattern, words))
        segments.sort(key=lambda segment: segment[0])

        return {pattern: words for _, pattern, words in segments}

    def candidate_stats(self, branch_indices):
        """Vectorized per-candidate statistics for every guess word against one branch.

        Requires len(branch_indices) >= 1; an empty branch yields NaN for all
        float fields (silent IEEE 754 division by zero, no exception raised).

        Calls counts_for_all_candidates once and derives all fields from the result.
        Returns a CandidateStats of parallel arrays in guess-word (matrix-row) space:
        index g corresponds to self.matrix row g, not to any position in a candidate_list.

        Fields and dtypes (branch_size = len(branch_indices)):
          group_count             int32    number of non-empty response groups
          has_self                bool     candidate is in the branch (all-green pattern present)
          cost_lower_bound        float64  3.0 - (group_count + has_self) / branch_size
          sum_squared_group_sizes int64    Σk² over all response groups; sort key for best-first order
          max_group_size          int32    size of the largest response group
          entropy_gain            float64  Shannon entropy in bits; 1e-12 tolerance vs scalar path
        """
        branch_size = len(branch_indices)
        counts = self.counts_for_all_candidates(branch_indices)

        # Number of non-empty response groups per candidate (includes self group when present).
        group_count = (counts > 0).sum(axis=1).astype(np.int32)

        # Candidate is in the branch iff the all-green pattern (242) is non-empty.
        has_self = counts[:, 242] > 0

        # Admissible lower bound: same formula as evaluate_candidate's cost_lb.
        # Integer numerator cast to float64 before division so the result is
        # bit-identical to the scalar computation 3.0 - (G + s) / branch_size.
        cost_lower_bound = (
            np.float64(3.0)
            - (group_count.astype(np.float64) + has_self.astype(np.float64))
            / np.float64(branch_size)
        )

        # Σk² sort key; int64 output. Worst case is n² (all words in one group);
        # actual max ≈ 3185² ≈ 10M, which fits int32, but int64 is the spec type.
        # Zero entries contribute 0, so summing the full row is correct.
        sum_squared_group_sizes = (counts ** 2).sum(axis=1, dtype=np.int64)

        max_group_size = counts.max(axis=1).astype(np.int32)

        # Shannon entropy: -Σ p·log2(p) over non-empty groups, p = k/branch_size.
        # The double-where avoids log2(0): the inner where substitutes 1.0
        # (log2(1.0) = 0.0) for zero-count entries so log2 never sees zero,
        # and the outer where explicitly zeroes those contributions.
        group_probabilities = counts.astype(np.float64) / np.float64(branch_size)
        nonzero = counts > 0
        log2_group_probabilities = np.where(
            nonzero,
            np.log2(np.where(nonzero, group_probabilities, np.float64(1.0))),
            np.float64(0.0),
        )
        entropy_gain = -(group_probabilities * log2_group_probabilities).sum(axis=1)

        return CandidateStats(
            group_count=group_count,
            has_self=has_self,
            cost_lower_bound=cost_lower_bound,
            sum_squared_group_sizes=sum_squared_group_sizes,
            max_group_size=max_group_size,
            entropy_gain=entropy_gain,
        )
