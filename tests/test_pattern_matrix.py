"""Tests for pattern_matrix.py — §2, §3, and §5 acceptance criteria.

Acceptance bullets:
  (a) ~200 random (guess, answer) pairs: matrix[g, a] matches _encode_response.
  (b) ~20 random (candidate, branch) pairs: nonzero entries of
      counts_for_all_candidates match ResponseCache.group_counts — including
      branches that exercise multiple guess-row chunks and the final partial chunk.
  (c) Save/load round-trip equals the built matrix; shape mismatch → None.
  (d) §3 candidate_stats: for branch sizes {8, 30, 100, 500}, every field matches
      the per-candidate pure-Python computation — integers exactly,
      cost_lower_bound exactly, entropy within 1e-12.
  (e) §5 group_words fast path: ~50 random (guess, branch) pairs across varied
      sizes match ResponseCache.group_words in keys, values, and iteration
      order (list(d.items()), not just dict equality).
"""
import math
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

import pattern_matrix
from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH
from pattern_matrix import (
    CandidateStats, PatternMatrix, _COUNT_CHUNK_ROWS, _compute_answer_list_id,
)
from wordle_engine import ResponseCache, calculate_response, _encode_response


def _load_words(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# Paths relative to the repo root (runner cwd is the repo root).
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALL_GUESS_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_CANDIDATE_LIST_PATH))
_ALL_ANSWER_WORDS = _load_words(os.path.join(_REPO_DIR, DEFAULT_ANSWER_LIST_PATH))


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


class TestLoadOrBuild(unittest.TestCase):
    """load_or_build(): the one path erd_swarm.py and wordle.py both use —
    load a cached matrix, or build and atomically persist one on a miss."""

    def setUp(self):
        from cache_sqlite import ScoreCache
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        rng = random.Random(10)
        self.guess_words = rng.sample(_ALL_GUESS_WORDS, 12)
        self.answer_words = rng.sample(_ALL_ANSWER_WORDS, 8)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.score_cache = ScoreCache(self.cache_path, self.answer_words)
        self.addCleanup(self.score_cache.close)

    def _expected_matrix_path(self):
        from pattern_matrix import _compute_guess_vocabulary_id
        return os.path.join(
            self._tmp.name,
            f"pattern_matrix_{self.score_cache.answer_list_id}"
            f"_{_compute_guess_vocabulary_id(self.guess_words)}.npy")

    def test_cold_start_builds_and_persists(self):
        self.assertFalse(os.path.exists(self._expected_matrix_path()))
        pm = PatternMatrix.load_or_build(
            self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        self.assertIsNotNone(pm)
        self.assertTrue(os.path.exists(self._expected_matrix_path()))
        # No leftover per-PID temp file: the atomic tmp+replace left only the
        # final path behind.
        leftovers = [f for f in os.listdir(self._tmp.name) if ".tmp" in f]
        self.assertEqual(leftovers, [])

    def test_warm_start_loads_without_rebuilding(self):
        built = PatternMatrix.load_or_build(
            self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        with mock.patch.object(PatternMatrix, "build",
                               side_effect=AssertionError("must not rebuild on a warm load")):
            loaded = PatternMatrix.load_or_build(
                self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(built.matrix, loaded.matrix)

    def test_different_answer_universe_does_not_reuse_stale_file(self):
        from cache_sqlite import ScoreCache
        PatternMatrix.load_or_build(
            self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        rng = random.Random(11)
        other_answers = rng.sample(_ALL_ANSWER_WORDS, 8)
        other_cache_path = os.path.join(self._tmp.name, "cache2.sqlite3")
        other_score_cache = ScoreCache(other_cache_path, other_answers)
        self.addCleanup(other_score_cache.close)
        self.assertNotEqual(self.score_cache.answer_list_id,
                            other_score_cache.answer_list_id)
        pm_other = PatternMatrix.load_or_build(
            other_cache_path, self.guess_words, other_answers, other_score_cache)
        self.assertIsNotNone(pm_other)
        expected = PatternMatrix.build(self.guess_words, other_answers)
        np.testing.assert_array_equal(pm_other.matrix, expected.matrix)

    def test_different_guess_vocabulary_of_same_length_does_not_reuse_stale_file(self):
        # Same answer universe (so answer_list_id alone can't distinguish),
        # same guess *count*, different guess *words* -- exactly the case
        # load()'s shape check can't catch: only the guess-vocabulary
        # identity in the filename can.
        PatternMatrix.load_or_build(
            self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        rng = random.Random(12)
        other_pool = [w for w in _ALL_GUESS_WORDS if w not in set(self.guess_words)]
        other_guess_words = rng.sample(other_pool, len(self.guess_words))
        self.assertNotEqual(sorted(other_guess_words), sorted(self.guess_words))

        pm_other = PatternMatrix.load_or_build(
            self.cache_path, other_guess_words, self.answer_words, self.score_cache)
        self.assertIsNotNone(pm_other)
        expected = PatternMatrix.build(other_guess_words, self.answer_words)
        np.testing.assert_array_equal(pm_other.matrix, expected.matrix)

    def _leftover_tmp_files(self):
        return [f for f in os.listdir(self._tmp.name) if ".tmp" in f]

    def test_save_failure_still_returns_matrix_and_leaves_no_tmp_file(self):
        # Persisting is an optimization: a disk-full/permission error writing
        # the .npy must not crash construction (erd_swarm.py's
        # _BranchWorker.__init__ or wordle.py's GameState.__init__ call this
        # synchronously) and must not leak the per-PID temp file.
        with mock.patch.object(PatternMatrix, "save",
                               side_effect=OSError("disk full")):
            pm = PatternMatrix.load_or_build(
                self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        self.assertIsNotNone(pm)
        expected = PatternMatrix.build(self.guess_words, self.answer_words)
        np.testing.assert_array_equal(pm.matrix, expected.matrix)
        self.assertEqual(self._leftover_tmp_files(), [])

    def test_replace_failure_still_returns_matrix_and_leaves_no_tmp_file(self):
        with mock.patch("pattern_matrix.os.replace",
                        side_effect=OSError("cross-device link")):
            pm = PatternMatrix.load_or_build(
                self.cache_path, self.guess_words, self.answer_words, self.score_cache)
        self.assertIsNotNone(pm)
        expected = PatternMatrix.build(self.guess_words, self.answer_words)
        np.testing.assert_array_equal(pm.matrix, expected.matrix)
        self.assertEqual(self._leftover_tmp_files(), [])


class TestCandidateStats(unittest.TestCase):
    """Acceptance §3: candidate_stats returns correct parallel arrays for all branch sizes.

    For fixed branches of sizes {8, 30, 100, 500} — at least one branch whose words
    include candidates (has_self=True for some rows) and at least one whose words
    exclude all candidates (has_self=False for all rows) — every field matches a
    per-candidate pure-Python computation: integers exactly, cost_lower_bound exactly,
    entropy within 1e-12.
    """

    @classmethod
    def setUpClass(cls):
        # 150 guess words keeps build time short while verifying all rows.
        cls.guess_words = _ALL_GUESS_WORDS[:150]
        cls.answer_words = _ALL_ANSWER_WORDS
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)
        guess_word_set = set(cls.guess_words)
        # Answer words that are also in our 150-word guess vocabulary — these produce
        # has_self=True when included in the branch.
        cls.answer_also_guess = [w for w in cls.answer_words if w in guess_word_set]
        # Answer words NOT in our guess vocabulary — no candidate produces has_self=True
        # when the branch is drawn only from these.
        cls.answer_not_guess = [w for w in cls.answer_words if w not in guess_word_set]

    @staticmethod
    def _python_stats_for_row(counts_row, branch_size):
        """Pure-Python reference for one candidate's row of response-group counts.

        counts_row is the 243-element int32 array from counts_for_all_candidates[g].
        Matches the scalar formulas in wordle_engine.evaluate_candidate and score_groups.
        """
        groups = {pattern: int(count) for pattern, count in enumerate(counts_row) if count > 0}
        group_count = len(groups)
        has_self = groups.get(242, 0) > 0
        cost_lower_bound = 3.0 - (group_count + (1 if has_self else 0)) / branch_size
        sum_squared = sum(count * count for count in groups.values())
        max_group = max(groups.values()) if groups else 0
        entropy = 0.0
        for count in groups.values():
            probability = count / branch_size
            entropy -= probability * math.log2(probability)
        return {
            'group_count': group_count,
            'has_self': has_self,
            'cost_lower_bound': cost_lower_bound,
            'sum_squared_group_sizes': sum_squared,
            'max_group_size': max_group,
            'entropy_gain': entropy,
        }

    def _verify_all_candidates(self, branch_words, label):
        """Check every field of candidate_stats against the per-row Python reference."""
        branch_indices = self.pm.answer_indices(branch_words)
        branch_size = len(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        # Fetch counts separately to feed _python_stats_for_row. Non-circularity
        # comes from the Python arithmetic in that helper, not from this fetch.
        counts = self.pm.counts_for_all_candidates(branch_indices)

        for g in range(len(self.guess_words)):
            expected = self._python_stats_for_row(counts[g], branch_size)
            word = self.guess_words[g]
            failure_message = f"{label}, candidate={word!r}"
            self.assertEqual(int(stats.group_count[g]), expected['group_count'], failure_message)
            self.assertEqual(bool(stats.has_self[g]), expected['has_self'], failure_message)
            self.assertEqual(
                float(stats.cost_lower_bound[g]), expected['cost_lower_bound'], failure_message)
            self.assertEqual(
                int(stats.sum_squared_group_sizes[g]), expected['sum_squared_group_sizes'],
                failure_message)
            self.assertEqual(int(stats.max_group_size[g]), expected['max_group_size'],
                             failure_message)
            self.assertAlmostEqual(
                float(stats.entropy_gain[g]), expected['entropy_gain'],
                delta=1e-12, msg=failure_message)

    def test_branch_size_8_no_candidate_in_branch(self):
        """Branch of size 8 drawn entirely from words outside the guess vocabulary.

        No candidate can produce the all-green pattern (242) for any branch word,
        so has_self is False for every row — the 'not containing candidate words' case.
        """
        rng = random.Random(200)
        branch_words = rng.sample(self.answer_not_guess, 8)
        self._verify_all_candidates(branch_words, "size-8 (no candidate in branch)")
        # Confirm the has_self=False invariant directly.
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        self.assertFalse(np.any(stats.has_self),
                         "has_self must be False for all candidates when branch words "
                         "are outside the guess vocabulary")

    def test_branch_size_30_candidates_in_branch(self):
        """Branch of size 30 that includes words present in the guess vocabulary.

        Those words produce has_self=True for their corresponding candidate rows —
        the 'containing candidate words' case.
        """
        rng = random.Random(201)
        # Seed with 5 answer words that are also in our guess vocabulary.
        in_both = self.answer_also_guess[:5]
        remainder = rng.sample(self.answer_not_guess, 25)
        branch_words = in_both + remainder
        self._verify_all_candidates(branch_words, "size-30 (candidates in branch)")
        # Confirm has_self=True for the candidates whose word is in the branch.
        branch_word_set = set(branch_words)
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        for g, word in enumerate(self.guess_words):
            if word in branch_word_set:
                self.assertTrue(bool(stats.has_self[g]),
                                f"has_self must be True for {word!r} (word is in branch)")

    def test_branch_size_100(self):
        rng = random.Random(202)
        branch_words = rng.sample(self.answer_words, 100)
        self._verify_all_candidates(branch_words, "size-100")

    def test_branch_size_500(self):
        rng = random.Random(203)
        branch_words = rng.sample(self.answer_words, 500)
        self._verify_all_candidates(branch_words, "size-500")

    def test_output_dtypes(self):
        """Field dtypes match the plan specification."""
        rng = random.Random(204)
        branch_words = rng.sample(self.answer_words, 20)
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        self.assertIsInstance(stats, CandidateStats)
        self.assertEqual(stats.group_count.dtype, np.int32)
        self.assertEqual(stats.has_self.dtype, np.bool_)
        self.assertEqual(stats.cost_lower_bound.dtype, np.float64)
        self.assertEqual(stats.sum_squared_group_sizes.dtype, np.int64)
        self.assertEqual(stats.max_group_size.dtype, np.int32)
        self.assertEqual(stats.entropy_gain.dtype, np.float64)

    def test_array_lengths(self):
        """All six parallel arrays have length n_guesses."""
        rng = random.Random(205)
        branch_words = rng.sample(self.answer_words, 15)
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        n_guesses = len(self.guess_words)
        for field in stats._fields:
            self.assertEqual(len(getattr(stats, field)), n_guesses,
                             f"{field} has wrong length")

    def test_cost_lower_bound_range(self):
        """cost_lower_bound is bounded by [0, 3] for all candidates."""
        rng = random.Random(206)
        branch_words = rng.sample(self.answer_words, 50)
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        self.assertTrue(np.all(stats.cost_lower_bound >= 0.0))
        self.assertTrue(np.all(stats.cost_lower_bound <= 3.0))

    def test_sum_squared_dtype_is_int64(self):
        """sum_squared_group_sizes is int64 — the spec type for exact comparison with Python.

        Response groups partition the branch (Σk = n), so Σk² is maximized when all
        words land in one group: n² ≤ 3185² ≈ 10M, which fits int32 comfortably. int64
        is the spec type so that int(stats.sum_squared_group_sizes[g]) compares exactly
        against Python's arbitrary-precision sum(k*k).
        """
        rng = random.Random(207)
        branch_words = rng.sample(self.answer_words, 500)
        branch_indices = self.pm.answer_indices(branch_words)
        stats = self.pm.candidate_stats(branch_indices)
        self.assertEqual(stats.sum_squared_group_sizes.dtype, np.int64)
        # All values must be non-negative integers.
        self.assertTrue(np.all(stats.sum_squared_group_sizes >= 0))


class TestGroupWordsFastPath(unittest.TestCase):
    """§5 acceptance: PatternMatrix.group_words (the vectorized fast path) is
    identical to ResponseCache.group_words (the pure-Python reference) in
    keys, values, and iteration order — not just as sets/dicts, since
    iteration order determines evaluate_candidate's float accumulation order
    (full_tree_plan.md §5).
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(500)
        cls.answer_words = sorted(rng.sample(_ALL_ANSWER_WORDS, 200))
        extra_guesses = rng.sample(
            [w for w in _ALL_GUESS_WORDS if w not in set(cls.answer_words)], 100)
        cls.guess_words = sorted(set(cls.answer_words) | set(extra_guesses))
        cls.pm = PatternMatrix.build(cls.guess_words, cls.answer_words)
        cls.rc = ResponseCache(cls.answer_words)

    def test_50_random_guess_branch_pairs(self):
        rng = random.Random(501)
        sizes = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
        for trial in range(50):
            size = rng.choice(sizes)
            branch_words = sorted(rng.sample(self.answer_words, size))
            guess = rng.choice(self.guess_words)
            branch_indices = self.pm.answer_indices(branch_words)
            slow = self.rc.group_words(guess, branch_words)
            fast = self.pm.group_words(guess, branch_words, branch_indices)
            with self.subTest(trial=trial, size=size, guess=guess):
                self.assertEqual(list(fast.items()), list(slow.items()))

    def test_iteration_order_with_many_ties(self):
        """A branch over few distinct letters produces many equal-size
        response groups (e.g. several singletons) against a strong
        splitter — exactly the case where emitting in ascending-pattern
        order (the canonical wrong answer) would diverge from the
        Python loop's first-appearance order.
        """
        rng = random.Random(502)
        branch_words = sorted(rng.sample(self.answer_words, 60))
        for guess in rng.sample(self.guess_words, 15):
            branch_indices = self.pm.answer_indices(branch_words)
            slow = self.rc.group_words(guess, branch_words)
            fast = self.pm.group_words(guess, branch_words, branch_indices)
            with self.subTest(guess=guess):
                self.assertEqual(list(fast.items()), list(slow.items()))

    def test_response_cache_delegates_only_with_both_supplied(self):
        """ResponseCache.group_words takes the fast path only when both
        pattern_matrix and branch_indices are supplied; either alone falls
        through to the loop (the interactive-fallback contract)."""
        branch_words = sorted(random.Random(503).sample(self.answer_words, 10))
        guess = self.guess_words[0]
        branch_indices = self.pm.answer_indices(branch_words)
        loop_only = self.rc.group_words(guess, branch_words)
        matrix_without_indices = self.rc.group_words(
            guess, branch_words, pattern_matrix=self.pm)
        indices_without_matrix = self.rc.group_words(
            guess, branch_words, branch_indices=branch_indices)
        both = self.rc.group_words(
            guess, branch_words, pattern_matrix=self.pm,
            branch_indices=branch_indices)
        self.assertEqual(list(matrix_without_indices.items()), list(loop_only.items()))
        self.assertEqual(list(indices_without_matrix.items()), list(loop_only.items()))
        self.assertEqual(list(both.items()), list(loop_only.items()))

    def test_empty_branch_returns_empty_dict(self):
        """PatternMatrix.group_words matches the loop on an empty branch
        instead of indexing into an empty sorted-patterns array."""
        branch_words = []
        branch_indices = self.pm.answer_indices(branch_words)
        slow = self.rc.group_words(self.guess_words[0], branch_words)
        fast = self.pm.group_words(self.guess_words[0], branch_words, branch_indices)
        self.assertEqual(fast, {})
        self.assertEqual(list(fast.items()), list(slow.items()))

    def test_size_one_branch(self):
        branch_words = [self.answer_words[0]]
        branch_indices = self.pm.answer_indices(branch_words)
        for guess in (self.answer_words[0], self.guess_words[1]):
            slow = self.rc.group_words(guess, branch_words)
            fast = self.pm.group_words(guess, branch_words, branch_indices)
            with self.subTest(guess=guess):
                self.assertEqual(list(fast.items()), list(slow.items()))

    def test_out_of_vocabulary_guess_falls_back_to_loop(self):
        """ResponseCache.group_words with both pattern_matrix and
        branch_indices supplied must still fall back to the loop for a
        guess outside the matrix's guess vocabulary, not raise."""
        branch_words = sorted(random.Random(504).sample(self.answer_words, 10))
        branch_indices = self.pm.answer_indices(branch_words)
        out_of_vocab_guess = next(
            w for w in _ALL_GUESS_WORDS if w not in set(self.guess_words))
        expected = self.rc.group_words(out_of_vocab_guess, branch_words)
        actual = self.rc.group_words(
            out_of_vocab_guess, branch_words,
            pattern_matrix=self.pm, branch_indices=branch_indices)
        self.assertEqual(list(actual.items()), list(expected.items()))


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
