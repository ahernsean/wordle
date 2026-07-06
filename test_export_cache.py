"""Unit tests for export_cache.cmd_export.

Covers the root-scoped candidate_scores export: the table is far too bulky
to export in full (a row per candidate per method for every branch position
the swarm has touched), but the root position — the full answer list,
before any guess — is worth carrying so a phone missing an ERD lookup can
still rank candidates by entropy/max-group-size.
"""
import os
import sqlite3
import tempfile
import types
import unittest

import export_cache
from cache_sqlite import ScoreCache
from wordle_engine import load_word_list

OTHER_BRANCH_WORDS = ["crane", "slate", "trace", "stale", "tales"]


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
        self.all_answers = load_word_list(export_cache.ANSWER_FILE)

    def _seed_candidate_scores(self, cache_path):
        sc = ScoreCache(cache_path, self.all_answers)
        root_key = ScoreCache.encode_subset(self.all_answers)
        sc.write_scores(root_key, [('crane', 3.2)], 'entropy_gain')
        other_key = ScoreCache.encode_subset(OTHER_BRANCH_WORDS)
        sc.write_scores(other_key, [('slate', 1.1)], 'entropy_gain')
        sc.close()

    def test_export_carries_only_root_position_scores(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)

        conn = sqlite3.connect(args.output)
        rows = conn.execute(
            "SELECT subset_hash, word FROM candidate_scores").fetchall()
        conn.close()

        root_hash = ScoreCache._subset_hash(
            ScoreCache.encode_subset(self.all_answers))
        self.assertEqual(rows, [(root_hash, 'crane')])

    def test_export_is_idempotent(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)
        export_cache.cmd_export(args)

        conn = sqlite3.connect(args.output)
        n = conn.execute(
            "SELECT COUNT(*) FROM candidate_scores").fetchone()[0]
        conn.close()
        self.assertEqual(n, 1)

    def test_export_reads_back_through_score_cache(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        export_cache.cmd_export(args)

        sc = ScoreCache(args.output, self.all_answers,
                        checkpoint_on_close=False)
        root_key = ScoreCache.encode_subset(self.all_answers)
        scores = sc.read_scores(root_key, 'entropy_gain')
        sc.close()
        self.assertEqual(scores, [('crane', 3.2)])


if __name__ == '__main__':
    unittest.main()
