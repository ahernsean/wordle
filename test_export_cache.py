"""Unit tests for export_cache.cmd_export.

candidate_scores is exported in full: a phone without a cached ERD result
for its current position needs entropy/max-group-size scores to rank
candidates regardless of how deep into the game that position is, so the
export isn't limited to the opening guess.
"""
import os
import sqlite3
import tempfile
import types
import unittest

import export_cache
from cache_sqlite import ScoreCache

ANSWER_WORDS = ["crane", "slate", "trace", "stale", "tales"]
OTHER_BRANCH_WORDS = ["crane", "slate", "trace", "stale"]


def _make_args(tmp_dir, **overrides):
    args = types.SimpleNamespace(
        cache=os.path.join(tmp_dir, 'cache.sqlite3'),
        output=os.path.join(tmp_dir, 'export.sqlite3'),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestExportCandidateScores(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _seed_candidate_scores(self, cache_path):
        sc = ScoreCache(cache_path, ANSWER_WORDS)
        root_key = ScoreCache.encode_subset(ANSWER_WORDS)
        sc.write_scores(root_key, [('crane', 3.2)], 'entropy_gain')
        other_key = ScoreCache.encode_subset(OTHER_BRANCH_WORDS)
        sc.write_scores(other_key, [('slate', 1.1)], 'entropy_gain')
        sc.close()

    def test_export_carries_every_branch_position_scores(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)

        conn = sqlite3.connect(args.output)
        rows = conn.execute(
            "SELECT word FROM candidate_scores ORDER BY word").fetchall()
        conn.close()

        self.assertEqual(rows, [('crane',), ('slate',)])

    def test_export_is_idempotent(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)
        export_cache.cmd_export(args)

        conn = sqlite3.connect(args.output)
        n = conn.execute(
            "SELECT COUNT(*) FROM candidate_scores").fetchone()[0]
        conn.close()
        self.assertEqual(n, 2)

    def test_export_reads_back_through_score_cache(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)

        sc = ScoreCache(args.output, ANSWER_WORDS, checkpoint_on_close=False)
        root_key = ScoreCache.encode_subset(ANSWER_WORDS)
        other_key = ScoreCache.encode_subset(OTHER_BRANCH_WORDS)
        root_scores = sc.read_scores(root_key, 'entropy_gain')
        other_scores = sc.read_scores(other_key, 'entropy_gain')
        sc.close()
        self.assertEqual(root_scores, [('crane', 3.2)])
        self.assertEqual(other_scores, [('slate', 1.1)])


if __name__ == '__main__':
    unittest.main()
