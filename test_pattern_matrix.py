"""Tests for pattern_matrix.py — §2 acceptance criteria.

Acceptance bullets:
  (a) ~200 random (guess, answer) pairs: matrix[g, a] matches _encode_response.
  (b) ~20 random (candidate, branch) pairs: nonzero entries of
      counts_for_all_candidates match ResponseCache.group_counts — including
      branches that exercise multiple guess-row chunks and the final partial chunk.
  (c) Save/load round-trip equals the built matrix; shape mismatch → None.
  (d) With NumPy absent, available() is False and import still succeeds.
"""
import importlib
import os
import random
import sys
import tempfile
import unittest

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

import pattern_matrix
from pattern_matrix import PatternMatrix, _COUNT_CHUNK_ROWS, _compute_answer_list_id
from wordle_engine import ResponseCache, calculate_response, _encode_response


def _load_words(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# Paths relative to the test file's directory (runner cwd is the repo root).
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_ALL_GUESS_WORDS = _load_words(os.path.join(_REPO_DIR, "wordle.txt"))
_ALL_ANSWER_WORDS = _load_words(os.path.join(_REPO_DIR, "NYT_wordlist.txt"))


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestAvailable(unittest.TestCase):
    def test_available_true_with_numpy(self):
        self.assertTrue(pattern_matrix.available())


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestMatrixCorrectness(unittest.TestCase):
    """Acceptance (a): matrix[g, a] == _encode_response(calculate_response(g, a))."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(0)
        # Small vocabulary for fast construction (~50 guesses × 50 answers).
        cls.guess_words = rng.sample(_ALL_GUESS_WORDS, 50)
        cls.answer_words = rng.sample(_ALL_ANSWER_WORDS, 50)
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def test_200_random_pairs(self):
        rng = random.Random(1)
        for _ in range(200):
            guess = rng.choice(self.guess_words)
            answer = rng.choice(self.answer_words)
            g = self.pm.guess_index(guess)
            a = self.pm._answer_index[answer]
            expected = _encode_response(calculate_response(guess, answer))
            self.assertEqual(
                int(self.pm.matrix[g, a]), expected,
                f"matrix[{g}, {a}] wrong for guess={guess!r}, answer={answer!r}",
            )

    def test_guess_index_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.pm.guess_index("zzzzz")

    def test_answer_indices_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.pm.answer_indices(["zzzzz"])

    def test_answer_indices_dtype(self):
        indices = self.pm.answer_indices(self.answer_words[:5])
        self.assertEqual(indices.dtype, np.int32)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestCountsForAllCandidates(unittest.TestCase):
    """Acceptance (b): counts match ResponseCache.group_counts for ~20 random pairs.

    The matrix has _COUNT_CHUNK_ROWS + 17 guess words, producing exactly two
    guess-row chunks: one full chunk (rows 0-1023) and one partial chunk (17
    rows, 1024-1040). All ~20 test cases exercise both chunks, satisfying the
    "larger than one chunk" and "final partial chunk" requirements.
    """

    @classmethod
    def setUpClass(cls):
        # _COUNT_CHUNK_ROWS + 17 guesses → 2 chunks; last chunk is 17 rows.
        cls.guess_words = _ALL_GUESS_WORDS[:_COUNT_CHUNK_ROWS + 17]
        cls.answer_words = _ALL_ANSWER_WORDS  # full answer universe
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)
        cls.rc = ResponseCache(cls.answer_words)

    def _assert_counts_match(self, candidate, branch_words):
        branch_indices = self.pm.answer_indices(branch_words)
        counts = self.pm.counts_for_all_candidates(branch_indices)
        g = self.pm.guess_index(candidate)

        # Nonzero entries from counts_for_all_candidates row g.
        matrix_counts = {
            p: int(counts[g, p]) for p in range(243) if counts[g, p] > 0
        }

        # Reference from ResponseCache.
        rc_counts = dict(self.rc.group_counts(candidate, branch_words))

        self.assertEqual(matrix_counts, rc_counts,
                         f"counts mismatch for candidate={candidate!r}, "
                         f"branch_size={len(branch_words)}")

    def test_20_random_pairs_various_branch_sizes(self):
        rng = random.Random(2)
        # Mix of branch sizes: small, medium, large (all > _COUNT_CHUNK_ROWS
        # in terms of the guess-row dimension — two chunks are always exercised).
        branch_sizes = [8, 8, 30, 30, 100, 100, 200, 300, 500, 1000,
                        1000, 1500, 2000, 3000, 50, 80, 150, 400, 700, 1200]
        self.assertEqual(len(branch_sizes), 20)
        for size in branch_sizes:
            candidate = rng.choice(self.guess_words)
            # Sample branch_words from answer_words so answer_indices works.
            branch_words = rng.sample(self.answer_words, min(size, len(self.answer_words)))
            self._assert_counts_match(candidate, branch_words)

    def test_candidate_in_last_chunk(self):
        # Explicitly pick a candidate in the final partial chunk (row >= _COUNT_CHUNK_ROWS).
        candidate = self.guess_words[_COUNT_CHUNK_ROWS + 5]
        rng = random.Random(3)
        branch_words = rng.sample(self.answer_words, 80)
        self._assert_counts_match(candidate, branch_words)

    def test_candidate_in_first_chunk(self):
        candidate = self.guess_words[10]
        rng = random.Random(4)
        branch_words = rng.sample(self.answer_words, 80)
        self._assert_counts_match(candidate, branch_words)

    def test_shape(self):
        branch_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        counts = self.pm.counts_for_all_candidates(branch_indices)
        self.assertEqual(counts.shape, (len(self.guess_words), 243))
        self.assertEqual(counts.dtype, np.int32)

    def test_single_chunk_matches_response_cache(self):
        # Vocabulary smaller than _COUNT_CHUNK_ROWS exercises the single-chunk
        # path (one partial chunk, no full chunk preceding it).
        rng = random.Random(11)
        small_guess_words = _ALL_GUESS_WORDS[:50]
        small_pm = PatternMatrix.build(small_guess_words, self.answer_words)
        small_rc = ResponseCache(self.answer_words)
        candidate = rng.choice(small_guess_words)
        branch_words = rng.sample(self.answer_words, 80)
        branch_indices = small_pm.answer_indices(branch_words)
        counts = small_pm.counts_for_all_candidates(branch_indices)
        g = small_pm.guess_index(candidate)
        matrix_counts = {p: int(counts[g, p]) for p in range(243) if counts[g, p] > 0}
        rc_counts = dict(small_rc.group_counts(candidate, branch_words))
        self.assertEqual(matrix_counts, rc_counts)

    def test_counts_sum_to_branch_size(self):
        rng = random.Random(5)
        branch_words = rng.sample(self.answer_words, 37)
        branch_indices = self.pm.answer_indices(branch_words)
        counts = self.pm.counts_for_all_candidates(branch_indices)
        # Every row must sum to exactly the branch size.
        row_sums = counts.sum(axis=1)
        self.assertTrue(np.all(row_sums == len(branch_words)),
                        f"row sums not all {len(branch_words)}: {row_sums[row_sums != len(branch_words)]}")


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestPatternsForCandidates(unittest.TestCase):
    """patterns_for_candidates returns the correct uint8 slice."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(6)
        cls.guess_words = rng.sample(_ALL_GUESS_WORDS, 20)
        cls.answer_words = rng.sample(_ALL_ANSWER_WORDS, 30)
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def test_shape_and_dtype(self):
        candidate_indices = [0, 3, 7]
        branch_indices = np.array([1, 5, 10, 15], dtype=np.int32)
        result = self.pm.patterns_for_candidates(candidate_indices, branch_indices)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, np.uint8)

    def test_values_match_matrix_rows(self):
        candidate_indices = [2, 5, 9]
        branch_indices = np.array([0, 2, 4, 6, 8], dtype=np.int32)
        result = self.pm.patterns_for_candidates(candidate_indices, branch_indices)
        for out_row, g in enumerate(candidate_indices):
            expected = self.pm.matrix[g][branch_indices]
            np.testing.assert_array_equal(result[out_row], expected)


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestSaveLoad(unittest.TestCase):
    """Acceptance (c): save/load round-trip; shape mismatch → None."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(7)
        cls.guess_words = rng.sample(_ALL_GUESS_WORDS, 40)
        cls.answer_words = rng.sample(_ALL_ANSWER_WORDS, 25)
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "matrix.npy")
            self.pm.save(path)
            pm2 = PatternMatrix.load(path, self.guess_words, self.answer_words)
            self.assertIsNotNone(pm2)
            np.testing.assert_array_equal(self.pm.matrix, pm2.matrix)
            self.assertEqual(pm2.answer_list_id, self.pm.answer_list_id)

    def test_save_and_load_without_npy_extension_agree(self):
        # save(p) and load(p, ...) must find the same file whether or not
        # the caller includes the '.npy' suffix — np.save appends it silently
        # but np.load does not, so both methods must normalize consistently.
        with tempfile.TemporaryDirectory() as tmp:
            path_no_ext = os.path.join(tmp, "matrix")  # no '.npy'
            self.pm.save(path_no_ext)
            pm2 = PatternMatrix.load(path_no_ext, self.guess_words, self.answer_words)
            self.assertIsNotNone(pm2, "load without .npy suffix should find the file save() wrote")
            np.testing.assert_array_equal(self.pm.matrix, pm2.matrix)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nonexistent.npy")
            result = PatternMatrix.load(path, self.guess_words, self.answer_words)
            self.assertIsNone(result)

    def test_shape_mismatch_returns_none_and_triggers_rebuild(self):
        # The file has shape (40, 25); loading it claiming 24 answer words → mismatch.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "matrix.npy")
            self.pm.save(path)
            # Claim one fewer answer word → (40, 24) expected vs (40, 25) on disk.
            wrong_answer_words = self.answer_words[:-1]
            result = PatternMatrix.load(path, self.guess_words, wrong_answer_words)
            self.assertIsNone(result,
                              "load should return None on shape mismatch so caller can rebuild")

    def test_shape_mismatch_different_guess_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "matrix.npy")
            self.pm.save(path)
            # Claim one fewer guess word → mismatch.
            result = PatternMatrix.load(path, self.guess_words[:-1], self.answer_words)
            self.assertIsNone(result)

    def test_mmap_load_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "matrix.npy")
            self.pm.save(path)
            pm2 = PatternMatrix.load(path, self.guess_words, self.answer_words)
            self.assertIsNotNone(pm2)
            # mmap_mode='r' → matrix is not writable.
            with self.assertRaises((ValueError, TypeError)):
                pm2.matrix[0, 0] = 0


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy not available")
class TestBuildWithScoreCache(unittest.TestCase):
    """build() reads from score_cache when available and writes back misses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_build_reads_and_writes_decomposition(self):
        from cache_sqlite import ScoreCache
        rng = random.Random(8)
        guess_words = rng.sample(_ALL_GUESS_WORDS, 5)
        answer_words = rng.sample(_ALL_ANSWER_WORDS, 10)

        db_path = os.path.join(self._tmp.name, "cache.sqlite3")
        cache = ScoreCache(db_path, answer_words)
        try:
            pm = PatternMatrix.build(guess_words, answer_words, score_cache=cache)

            # All 5 blobs should now be in the cache.
            for guess in guess_words:
                blob = cache.read_decomposition(guess)
                self.assertIsNotNone(blob, f"blob for {guess!r} not written to cache")
                self.assertEqual(len(blob), len(answer_words))

            # Building a second time from the populated cache gives the same matrix.
            pm2 = PatternMatrix.build(guess_words, answer_words, score_cache=cache)
            np.testing.assert_array_equal(pm.matrix, pm2.matrix)
        finally:
            cache.close()

    def test_build_without_score_cache(self):
        rng = random.Random(9)
        guess_words = rng.sample(_ALL_GUESS_WORDS, 5)
        answer_words = rng.sample(_ALL_ANSWER_WORDS, 10)
        pm = PatternMatrix.build(guess_words, answer_words, score_cache=None)
        self.assertEqual(pm.matrix.shape, (5, 10))
        self.assertEqual(pm.matrix.dtype, np.uint8)


class TestNumpyAbsent(unittest.TestCase):
    """Acceptance (d): available() False and imports succeed without NumPy."""

    def test_available_false_and_import_succeeds_without_numpy(self):
        orig_numpy = sys.modules.get("numpy")
        orig_pm = sys.modules.get("pattern_matrix")
        try:
            # None in sys.modules causes ImportError when the module executes
            # 'import numpy' — the documented sentinel for a failed import.
            sys.modules["numpy"] = None
            # Remove pattern_matrix so it reimports fresh without numpy.
            sys.modules.pop("pattern_matrix", None)
            import pattern_matrix as pm_no_numpy
            self.assertFalse(pm_no_numpy.available(),
                             "available() must be False when numpy is absent")
        finally:
            # Restore original state.
            if orig_numpy is not None:
                sys.modules["numpy"] = orig_numpy
            elif sys.modules.get("numpy") is None:
                del sys.modules["numpy"]
            if orig_pm is not None:
                sys.modules["pattern_matrix"] = orig_pm
            else:
                sys.modules.pop("pattern_matrix", None)
            # Reload the real pattern_matrix so subsequent tests work.
            importlib.reload(pattern_matrix)

    def test_engine_imports_without_numpy(self):
        # wordle_engine does not import numpy, so it always loads cleanly.
        # This verifies that engine import is independent of numpy availability.
        orig_numpy = sys.modules.get("numpy")
        orig_wm = sys.modules.get("wordle_engine")
        try:
            sys.modules["numpy"] = None
            sys.modules.pop("wordle_engine", None)
            import wordle_engine  # noqa: F401 — import succeeds is the assertion
        finally:
            if orig_numpy is not None:
                sys.modules["numpy"] = orig_numpy
            elif sys.modules.get("numpy") is None:
                del sys.modules["numpy"]
            if orig_wm is not None:
                sys.modules["wordle_engine"] = orig_wm
            else:
                sys.modules.pop("wordle_engine", None)
            importlib.reload(pattern_matrix)


class TestAnswerListId(unittest.TestCase):
    def test_matches_score_cache_id(self):
        from cache_sqlite import ScoreCache
        import tempfile
        rng = random.Random(10)
        answer_words = rng.sample(_ALL_ANSWER_WORDS, 20)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.sqlite3")
            sc = ScoreCache(db_path, answer_words)
            computed = _compute_answer_list_id(answer_words)
            self.assertEqual(computed, sc.answer_list_id)
            sc.close()


if __name__ == "__main__":
    unittest.main()
