"""Tests for diag_toplevel_census.py's census logic.

The census is checked against a brute-force reference partition built
directly from calculate_response over a small sampled vocabulary: same
distinct-branch count, same histogram, same over-300 count (trivially zero
at this scale), and byte-identical reports across two runs.
"""
import random
import unittest

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from wordle_engine import calculate_response, _encode_response, load_word_list

if _NUMPY_AVAILABLE:
    import diag_toplevel_census
    from pattern_matrix import PatternMatrix


def _reference_distinct_branches(opener_words, answer_words):
    """Brute-force distinct branches with >= 2 words, as frozensets of words."""
    distinct_branches = set()
    instance_count = 0
    for opener in opener_words:
        groups = {}
        for answer in answer_words:
            pattern = _encode_response(calculate_response(opener, answer))
            groups.setdefault(pattern, []).append(answer)
        for group_words in groups.values():
            if len(group_words) >= 2:
                instance_count += 1
                distinct_branches.add(frozenset(group_words))
    return distinct_branches, instance_count


@unittest.skipUnless(_NUMPY_AVAILABLE, 'NumPy required for the pattern matrix')
class CensusTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random_generator = random.Random(7)
        answer_words = load_word_list('NYT_wordlist.txt')
        opener_words = load_word_list('wordle.txt')
        cls.answer_words = sorted(random_generator.sample(answer_words, 60))
        cls.opener_words = sorted(
            set(random_generator.sample(opener_words, 25))
            | set(cls.answer_words[:5]))
        cls.matrix = PatternMatrix.build(cls.opener_words, cls.answer_words)

    def test_census_matches_brute_force(self):
        result = diag_toplevel_census.run_census(self.matrix)
        distinct_branches, instance_count = _reference_distinct_branches(
            self.opener_words, self.answer_words)
        self.assertEqual(result.distinct_count, len(distinct_branches))
        self.assertEqual(result.instance_count, instance_count)
        self.assertEqual(result.largest_branch_size,
                         max(len(branch) for branch in distinct_branches))
        self.assertEqual(
            result.over_count,
            sum(1 for branch in distinct_branches
                if len(branch) > diag_toplevel_census.OVER_WORD_COUNT))

        expected_histogram = [0] * len(diag_toplevel_census.HISTOGRAM_BUCKETS)
        for branch in distinct_branches:
            for bucket_index, (low, high) in enumerate(
                    diag_toplevel_census.HISTOGRAM_BUCKETS):
                if len(branch) >= low and (high is None or len(branch) <= high):
                    expected_histogram[bucket_index] += 1
                    break
        self.assertEqual(result.histogram, expected_histogram)

    def test_branch_segments_indices_are_sorted_and_correct(self):
        pattern_values = np.asarray(self.matrix.matrix[0])
        segments = list(diag_toplevel_census.branch_segments(pattern_values))
        self.assertTrue(segments)
        for column_indices in segments:
            self.assertGreaterEqual(len(column_indices), 2)
            self.assertTrue(np.all(np.diff(column_indices) > 0))
            segment_patterns = set(pattern_values[column_indices].tolist())
            self.assertEqual(len(segment_patterns), 1)
        # Segments plus size-1 groups partition all answer columns exactly.
        covered = sum(len(column_indices) for column_indices in segments)
        _, group_sizes = np.unique(pattern_values, return_counts=True)
        self.assertEqual(covered, int(group_sizes[group_sizes >= 2].sum()))

    def test_report_is_deterministic(self):
        first = diag_toplevel_census.format_report(
            diag_toplevel_census.run_census(self.matrix))
        second = diag_toplevel_census.format_report(
            diag_toplevel_census.run_census(self.matrix))
        self.assertEqual(first, second)
        self.assertIn(f'distinct branches over '
                      f'{diag_toplevel_census.OVER_WORD_COUNT} words:', first)


if __name__ == '__main__':
    unittest.main()
