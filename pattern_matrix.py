"""Pattern matrix: precomputed response patterns for all (guess, answer) pairs.

This module is the sole NumPy import point in the engine. The import is guarded
so all other modules work — and pass their full test suites — with NumPy absent.
Check available() before constructing or using PatternMatrix.
"""
import hashlib

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def available():
    """True iff NumPy is importable; every caller must check this before using PatternMatrix."""
    return _NUMPY_AVAILABLE


# Guess rows processed per bincount call. Bounds the per-call transient at
# 2 × _COUNT_CHUNK_ROWS × n × 4 bytes regardless of branch size (n = branch word count).
_COUNT_CHUNK_ROWS = 1024


def _compute_answer_list_id(answer_words):
    """SHA-256 identity of the answer universe (matches ScoreCache._ensure_answer_list)."""
    return hashlib.sha256("\n".join(answer_words).encode()).hexdigest()


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
