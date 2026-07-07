"""diag_toplevel_census.py -- §7a top-level branch census (resolves U7 exactly).

For every opener in wordle.txt, partitions the NYT answer list by response
pattern using the pattern matrix, then counts the distinct branches (answer
subsets with >= 2 words) across ALL openers: the same subset reached from two
different openers counts once. Reports the aggregate distinct count, two
branch-size histograms (human-scale buckets, and the cost model's
erd_queue.cost_size_bucket binning for the §7c census-count x per-regime-cost
arithmetic), and the count of branches over 300 words (the never-queued
regime). Per-opener totals are deliberately not reported.

Each branch is keyed by the SHA-256 digest of its sorted answer-column
indices (int32, little-endian), not by its word list -- ~1.5-2M distinct
branches are expected and digests keep the dedupe set to a few hundred MB.

Matrix acquisition: pass --matrix for a prebuilt .npy (must match the
vocabulary shapes; content is trusted). Without it the matrix is built with
PatternMatrix.build; pass --cache to read warm decomposition blobs from a
ScoreCache (build writes back any missing blob -- its designed behavior --
and the database is touched in no other way).

Deterministic: the same inputs produce byte-identical stdout. Progress goes
to stderr and carries no wall-clock content either.

Read-only diagnostic, same family as the other diag_*.py scripts -- NOT part
of the unittest suite. Sanity-check on a small vocabulary with
--vocab-sample N (seeded, like diag_kernel_bench.py's DIAG_VOCAB_SAMPLE);
the full run is the owner's to launch.
"""
import argparse
import hashlib
import random
import sys

import numpy as np

import pattern_matrix
from erd_queue import cost_size_bucket
from wordle_engine import load_word_list

OPENER_LIST_PATH = 'wordle.txt'
ANSWER_LIST_PATH = 'NYT_wordlist.txt'

# (low, high) inclusive branch-size buckets; high=None means unbounded.
# The 201-300 / 301-500 boundary is aligned to the over-300 regime line.
HISTOGRAM_BUCKETS = [
    (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100),
    (101, 200), (201, 300), (301, 500), (501, 1000), (1001, None),
]

OVER_WORD_COUNT = 300


class CensusResult:
    """Aggregate census over all openers.

    instance_count counts every (opener, pattern) branch with >= 2 words;
    distinct_count and the histogram count each distinct answer subset once.
    """

    def __init__(self, opener_count, answer_count):
        self.opener_count = opener_count
        self.answer_count = answer_count
        self.instance_count = 0
        self.distinct_count = 0
        self.over_count = 0
        self.largest_branch_size = 0
        self.histogram = [0] * len(HISTOGRAM_BUCKETS)
        # {cost_size_bucket index: distinct-branch count} — the §7c memo
        # multiplies these by measured per-regime costs.
        self.cost_model_histogram = {}

    def record_distinct(self, branch_size):
        self.distinct_count += 1
        if branch_size > OVER_WORD_COUNT:
            self.over_count += 1
        if branch_size > self.largest_branch_size:
            self.largest_branch_size = branch_size
        cost_model_bucket = cost_size_bucket(branch_size)
        self.cost_model_histogram[cost_model_bucket] = (
            self.cost_model_histogram.get(cost_model_bucket, 0) + 1)
        for bucket_index, (low, high) in enumerate(HISTOGRAM_BUCKETS):
            if branch_size >= low and (high is None or branch_size <= high):
                self.histogram[bucket_index] += 1
                return


def cost_model_bucket_ranges(max_branch_size):
    """{cost_size_bucket index: (smallest, largest) word count} over branch
    sizes 2..max_branch_size, derived by scanning the imported function so
    the printed ranges can never drift from the cost model's binning."""
    ranges = {}
    for branch_size in range(2, max_branch_size + 1):
        bucket_index = cost_size_bucket(branch_size)
        if bucket_index in ranges:
            low, high = ranges[bucket_index]
            ranges[bucket_index] = (min(low, branch_size), max(high, branch_size))
        else:
            ranges[bucket_index] = (branch_size, branch_size)
    return ranges


def branch_segments(pattern_values):
    """Yield the sorted answer-column indices (int32 array) of every response
    group with >= 2 words in one opener's pattern row.

    Segments come from pattern_matrix._stable_pattern_segments, already the
    sorted index list the branch digest is keyed on. The all-green self group
    has at most 1 word (the opener itself) and falls out of the >= 2 filter
    without special-casing.
    """
    sorted_order, _, segment_starts, segment_stops = (
        pattern_matrix._stable_pattern_segments(pattern_values))
    for start, stop in zip(segment_starts, segment_stops):
        if stop - start >= 2:
            yield sorted_order[start:stop]


def run_census(matrix, progress_stream=None):
    """Census every opener row of a PatternMatrix; returns a CensusResult."""
    result = CensusResult(matrix.n_guesses, matrix.n_answers)
    distinct_branch_digests = set()
    for opener_row in range(matrix.n_guesses):
        pattern_values = np.asarray(matrix.matrix[opener_row])
        for column_indices in branch_segments(pattern_values):
            result.instance_count += 1
            branch_digest = hashlib.sha256(column_indices.tobytes()).digest()
            if branch_digest not in distinct_branch_digests:
                distinct_branch_digests.add(branch_digest)
                result.record_distinct(len(column_indices))
        if progress_stream and (opener_row + 1) % 1000 == 0:
            print(f'{opener_row + 1}/{matrix.n_guesses} openers, '
                  f'{result.distinct_count} distinct branches so far',
                  file=progress_stream, flush=True)
    return result


def format_report(result):
    lines = [
        f'openers: {result.opener_count}  answers: {result.answer_count}',
        f'branch instances (>= 2 words, all openers): {result.instance_count}',
        f'distinct branches (>= 2 words): {result.distinct_count}',
        '',
        'distinct-branch size histogram:',
    ]
    for (low, high), count in zip(HISTOGRAM_BUCKETS, result.histogram):
        if high is None:
            label = f'{low}+'
        elif low == high:
            label = f'{low}'
        else:
            label = f'{low}-{high}'
        lines.append(f'  {label:>9} : {count}')
    lines += [
        '',
        'distinct-branch count per cost-model size bucket'
        ' (erd_queue.cost_size_bucket):',
    ]
    for bucket_index, (low, high) in sorted(
            cost_model_bucket_ranges(result.answer_count).items()):
        word_range = f'{low}' if low == high else f'{low}-{high}'
        count = result.cost_model_histogram.get(bucket_index, 0)
        lines.append(f'  bucket {bucket_index:>2} ({word_range + " words":>15})'
                     f' : {count}')
    lines += [
        '',
        f'distinct branches over {OVER_WORD_COUNT} words: {result.over_count}',
        f'largest branch: {result.largest_branch_size} words',
    ]
    return '\n'.join(lines) + '\n'


def acquire_matrix(arguments, opener_words, answer_words):
    if arguments.matrix:
        matrix = pattern_matrix.PatternMatrix.load(
            arguments.matrix, opener_words, answer_words)
        if matrix is None:
            sys.exit(f'--matrix {arguments.matrix}: missing or shape mismatch '
                     f'for {len(opener_words)} openers x {len(answer_words)} answers')
        return matrix
    score_cache = None
    if arguments.cache:
        from cache_sqlite import ScoreCache
        score_cache = ScoreCache(arguments.cache, answer_words,
                                 checkpoint_on_close=False)
    print('building pattern matrix'
          + (' (warm decomposition blobs)' if score_cache else ' (no cache; slow)'),
          file=sys.stderr, flush=True)
    try:
        return pattern_matrix.PatternMatrix.build(
            opener_words, answer_words, score_cache=score_cache)
    finally:
        if score_cache:
            score_cache.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--matrix',
                        help='prebuilt pattern-matrix .npy (skips the build)')
    parser.add_argument('--cache',
                        help='ScoreCache sqlite3 path; warm decomposition '
                             'blobs for the matrix build')
    parser.add_argument('--vocab-sample', type=int, default=0,
                        help='sample N words from each list for a fast '
                             'sanity run (0 = full lists)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for --vocab-sample')
    arguments = parser.parse_args(argv)

    if arguments.vocab_sample and arguments.cache:
        # A sampled answer universe would register a new answer_list_id in
        # the database on open; the census must not touch it beyond warm
        # decomposition blobs, so the combination is refused.
        sys.exit('--vocab-sample cannot be combined with --cache.')

    opener_words = load_word_list(OPENER_LIST_PATH)
    answer_words = load_word_list(ANSWER_LIST_PATH)
    if arguments.vocab_sample:
        random_generator = random.Random(arguments.seed)
        answer_words = sorted(random_generator.sample(
            answer_words, min(arguments.vocab_sample, len(answer_words))))
        opener_words = sorted(
            set(random_generator.sample(
                opener_words, min(arguments.vocab_sample, len(opener_words))))
            | set(answer_words))

    matrix = acquire_matrix(arguments, opener_words, answer_words)
    result = run_census(matrix, progress_stream=sys.stderr)
    sys.stdout.write(format_report(result))


if __name__ == '__main__':
    main()
