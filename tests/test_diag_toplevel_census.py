"""Tests for diag_toplevel_census.py's census logic.

The census is checked against a brute-force reference partition built
directly from calculate_response over a small sampled vocabulary: same
distinct-branch count, same histogram, same over-300 count (trivially zero
at this scale), and byte-identical reports across two runs.
"""
import random
import unittest

import numpy as np

import diag_toplevel_census
from erd_queue import cost_size_bucket
from pattern_matrix import PatternMatrix
from wordle_engine import calculate_response, _encode_response, load_word_list


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
        # Pins that the fixture actually exercises cross-opener dedupe --
        # without a duplicate branch present, the digest-dedupe path below
        # would never be differentially tested by the equality checks above.
        self.assertGreater(result.instance_count, result.distinct_count)
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

        expected_cost_model_histogram = {}
        for branch in distinct_branches:
            bucket_index = cost_size_bucket(len(branch))
            expected_cost_model_histogram[bucket_index] = (
                expected_cost_model_histogram.get(bucket_index, 0) + 1)
        self.assertEqual(result.cost_model_histogram,
                         expected_cost_model_histogram)

    def test_cost_model_bucket_ranges_partition_the_size_axis(self):
        max_branch_size = len(self.answer_words)
        ranges = diag_toplevel_census.cost_model_bucket_ranges(max_branch_size)
        for bucket_index, (low, high) in ranges.items():
            self.assertLessEqual(low, high)
            self.assertEqual(cost_size_bucket(low), bucket_index)
            self.assertEqual(cost_size_bucket(high), bucket_index)
        # Contiguous cover of 2..max_branch_size with no overlap.
        boundaries = sorted(ranges.values())
        self.assertEqual(boundaries[0][0], 2)
        self.assertEqual(boundaries[-1][1], max_branch_size)
        for (_, previous_high), (next_low, _) in zip(boundaries, boundaries[1:]):
            self.assertEqual(next_low, previous_high + 1)

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
