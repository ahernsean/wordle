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
        since=None,
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

    def test_table_missing_from_source_is_skipped_not_fatal(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        conn = sqlite3.connect(args.cache)
        conn.execute("DROP TABLE response_decomposition")
        conn.close()

        export_cache.cmd_export(args)

        out_conn = sqlite3.connect(args.output)
        tables = {r[0] for r in out_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        out_conn.close()
        self.assertNotIn("response_decomposition", tables)
        self.assertIn("candidate_scores", tables)

    def test_the_obsolete_candidate_erd_memo_does_not_travel(self):
        # A cache old enough to still hold the table must not put it on the
        # wire: the value is derived where it is displayed, so a stale row
        # would be a wrong number syncing to the phone.
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        conn = sqlite3.connect(args.cache)
        conn.execute("""
            CREATE TABLE candidate_erd_by_policy (
                subset_hash TEXT NOT NULL, candidate_word TEXT NOT NULL,
                policy TEXT NOT NULL, answer_list_id TEXT NOT NULL,
                erd REAL NOT NULL, max_remaining_depth INTEGER NOT NULL,
                response_group_count INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, candidate_word, policy, answer_list_id)
            )
        """)
        conn.execute(
            "INSERT INTO candidate_erd_by_policy VALUES "
            "('h', 'crane', 'erd_all', 'x', 3.5, 5, 120, 100)")
        conn.commit()
        conn.close()

        export_cache.cmd_export(args)

        out_conn = sqlite3.connect(args.output)
        tables = {r[0] for r in out_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        out_conn.close()
        self.assertNotIn("candidate_erd_by_policy", tables)
        self.assertIn("candidate_scores", tables)

    def test_since_excludes_rows_not_updated_after_watermark(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        conn = sqlite3.connect(args.cache)
        conn.execute(
            "UPDATE candidate_scores SET updated_at = 100 WHERE word = 'crane'")
        conn.execute(
            "UPDATE candidate_scores SET updated_at = 200 WHERE word = 'slate'")
        conn.commit()
        conn.close()

        args.since = 100
        export_cache.cmd_export(args)

        out_conn = sqlite3.connect(args.output)
        rows = out_conn.execute(
            "SELECT word FROM candidate_scores ORDER BY word").fetchall()
        out_conn.close()
        self.assertEqual(rows, [('slate',)])

    def test_since_still_copies_answer_list_in_full(self):
        # answer_list has no updated_at column, so it is always copied whole
        # regardless of --since -- it's the tiny namespace key row, not a
        # growing result table.
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)

        args.since = 2**31  # far future: no candidate_scores row qualifies
        export_cache.cmd_export(args)

        out_conn = sqlite3.connect(args.output)
        answer_rows = out_conn.execute(
            "SELECT COUNT(*) FROM answer_list").fetchone()[0]
        score_rows = out_conn.execute(
            "SELECT COUNT(*) FROM candidate_scores").fetchone()[0]
        out_conn.close()
        self.assertEqual(answer_rows, 1)
        self.assertEqual(score_rows, 0)

    def test_since_none_is_a_full_export(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        conn = sqlite3.connect(args.cache)
        conn.execute("UPDATE candidate_scores SET updated_at = 100")
        conn.commit()
        conn.close()

        export_cache.cmd_export(args)

        out_conn = sqlite3.connect(args.output)
        n = out_conn.execute(
            "SELECT COUNT(*) FROM candidate_scores").fetchone()[0]
        out_conn.close()
        self.assertEqual(n, 2)

    def test_second_delta_only_carries_rows_newer_than_first_watermark(self):
        # Simulates two successive deltas: the second, using the first
        # delta's watermark, must not re-carry rows the first already sent.
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        conn = sqlite3.connect(args.cache)
        conn.execute(
            "UPDATE candidate_scores SET updated_at = 100 WHERE word = 'crane'")
        conn.execute(
            "UPDATE candidate_scores SET updated_at = 200 WHERE word = 'slate'")
        conn.commit()
        conn.close()

        first_output = os.path.join(self._tmp.name, 'export1.sqlite3')
        args.output = first_output
        args.since = 0
        export_cache.cmd_export(args)

        second_output = os.path.join(self._tmp.name, 'export2.sqlite3')
        args.output = second_output
        args.since = 150
        export_cache.cmd_export(args)

        first_conn = sqlite3.connect(first_output)
        first_words = {r[0] for r in first_conn.execute(
            "SELECT word FROM candidate_scores")}
        first_conn.close()
        second_conn = sqlite3.connect(second_output)
        second_words = {r[0] for r in second_conn.execute(
            "SELECT word FROM candidate_scores")}
        second_conn.close()

        self.assertEqual(first_words, {'crane', 'slate'})
        self.assertEqual(second_words, {'slate'})

    def test_error_mid_export_rolls_back_and_reraises(self):
        args = _make_args(self._tmp.name)
        self._seed_candidate_scores(args.cache)
        # An export file whose answer_list table already exists under an
        # incompatible (extra-column) definition makes the copied INSERT
        # reference a column that isn't there, forcing a genuine failure
        # partway through the transaction.
        conn = sqlite3.connect(args.output)
        conn.execute("CREATE TABLE answer_list (bogus_column TEXT)")
        conn.close()

        with self.assertRaises(sqlite3.OperationalError):
            export_cache.cmd_export(args)

        # The failed run must not wedge the file against a later, correct
        # export attempt (proves DETACH/close ran despite the exception).
        os.remove(args.output)
        export_cache.cmd_export(args)
        out_conn = sqlite3.connect(args.output)
        n = out_conn.execute(
            "SELECT COUNT(*) FROM candidate_scores").fetchone()[0]
        out_conn.close()
        self.assertEqual(n, 2)


if __name__ == '__main__':
    unittest.main()
