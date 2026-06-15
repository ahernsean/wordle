"""Tests for the parallel-ERD-infra correctness fixes:

- liveness-gated chunk reclaim (a slow-but-alive worker is never reclaimed;
  a dead worker's chunk is freed),
- per-worker chunk reclaim used by the supervisor on respawn,
- merge_cache preferring an untainted entry over a tainted one,
- backfill_max_depth not clobbering a worker's fresh solve_budget,
- _multistep_stats keying response groups consistently with/without a cache.
"""
import os
import sqlite3
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache, MemoryScoreCache
from wordle_engine import ResponseCache, ERD_ALL, Solution
from erd_queue import ErdQueue
import merge_cache


WORDS = ["crane", "slate", "trace", "stale", "tales"]


class _TmpDB:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def path(self, name):
        return os.path.join(self._tmp.name, name)


class TestReclaimLiveness(_TmpDB, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.q = ErdQueue(self.path("q.sqlite3"))
        self.addCleanup(self.q.close)
        self.key = ScoreCache.encode_subset(WORDS)

    def _insert_chunk(self, worker, claimed_at, done=0):
        self.q._conn.execute(
            "INSERT INTO branch_chunks (subset_key, idx, claimed_by, claimed_at, done) "
            "VALUES (?, 0, ?, ?, ?)", (self.key, worker, claimed_at, done))

    def _n_chunks(self):
        return self.q._conn.execute(
            "SELECT COUNT(*) FROM branch_chunks").fetchone()[0]

    def test_live_worker_chunk_not_reclaimed(self):
        now = int(time.time())
        self._insert_chunk("worker-1", claimed_at=now - 1000)  # an old claim
        # ...but the worker is alive: it heartbeat just now.
        self.q.heartbeat("worker-1", pid=1, current_subset_key=self.key,
                         n_words=5, started_at=now, chunks_done=0)
        freed = self.q.reclaim_stale_chunks(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 0)
        self.assertEqual(self._n_chunks(), 1)

    def test_dead_worker_chunk_reclaimed(self):
        now = int(time.time())
        self._insert_chunk("worker-2", claimed_at=now - 1000)
        # Worker heartbeat is stale (>120s old) → presumed dead.
        self.q.heartbeat("worker-2", pid=1, current_subset_key=self.key,
                         n_words=5, started_at=now, chunks_done=0)
        self.q._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = 'worker-2'",
            (now - 1000,))
        freed = self.q.reclaim_stale_chunks(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 1)
        self.assertEqual(self._n_chunks(), 0)

    def test_no_heartbeat_chunk_reclaimed(self):
        now = int(time.time())
        self._insert_chunk("worker-3", claimed_at=now - 1000)  # never heartbeat
        freed = self.q.reclaim_stale_chunks(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 1)

    def test_done_chunk_never_reclaimed(self):
        now = int(time.time())
        self._insert_chunk("worker-4", claimed_at=now - 1000, done=1)
        freed = self.q.reclaim_stale_chunks(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 0)
        self.assertEqual(self._n_chunks(), 1)

    def test_reclaim_chunks_of_worker(self):
        now = int(time.time())
        self._insert_chunk("worker-5", claimed_at=now, done=0)
        self.q._conn.execute(
            "INSERT INTO branch_chunks (subset_key, idx, claimed_by, claimed_at, done) "
            "VALUES (?, 1, 'worker-5', ?, 1)", (self.key, now))  # done=1, keep
        freed = self.q.reclaim_chunks_of_worker("worker-5")
        self.assertEqual(freed, 1)  # only the done=0 row
        self.assertEqual(self._n_chunks(), 1)


class TestMergeUntaintedWins(_TmpDB, unittest.TestCase):
    def _make_cache(self, name, solve_budget):
        sc = ScoreCache(self.path(name), WORDS)
        key = ScoreCache.encode_subset(WORDS)
        sc.write(key, ERD_ALL, "crane", 1.5, max_depth=3,
                 solve_budget=solve_budget)
        sc.close()
        return key

    def _merge(self, target, source):
        conn = sqlite3.connect(self.path(target), isolation_level=None)
        conn.execute(f"ATTACH DATABASE '{self.path(source)}' AS src")
        cols = merge_cache._all_cols(conn, "subgroup_best_by_policy")
        col_list = ", ".join(cols)
        insert_sql = merge_cache._insert_sql("subgroup_best_by_policy", cols)
        rows = conn.execute(
            f"SELECT {col_list} FROM src.subgroup_best_by_policy").fetchall()
        conn.executemany(insert_sql, rows)
        conn.execute("DETACH DATABASE src")
        conn.close()

    def _read_budget(self, name, key):
        sc = ScoreCache(self.path(name), WORDS, checkpoint_on_close=False)
        row = sc.read_with_depth(key, ERD_ALL)
        sc.close()
        return row[3]  # solve_budget

    def test_untainted_source_upgrades_tainted_target(self):
        key = self._make_cache("target.sqlite3", solve_budget=5)   # tainted
        self._make_cache("source.sqlite3", solve_budget=None)      # untainted
        self._merge("target.sqlite3", "source.sqlite3")
        self.assertIsNone(self._read_budget("target.sqlite3", key))  # upgraded

    def test_tainted_source_does_not_downgrade_untainted_target(self):
        key = self._make_cache("target.sqlite3", solve_budget=None)  # untainted
        self._make_cache("source.sqlite3", solve_budget=5)           # tainted
        self._merge("target.sqlite3", "source.sqlite3")
        self.assertIsNone(self._read_budget("target.sqlite3", key))  # kept


class TestBackfillGuard(_TmpDB, unittest.TestCase):
    """The backfill UPDATE must no-op on any row a worker already filled in."""

    UPDATE = ("UPDATE subgroup_best_by_policy SET max_depth=?, solve_budget=NULL "
              "WHERE subset_key=? AND policy=? AND universe_id=? "
              "AND max_depth IS NULL")

    def setUp(self):
        super().setUp()
        self.sc = ScoreCache(self.path("c.sqlite3"), WORDS)
        self.addCleanup(self.sc.close)
        self.uid = self.sc.universe_id
        self.key = ScoreCache.encode_subset(WORDS)

    def _db_row(self):
        # Read straight from the DB (not the in-memory mirror) so we observe
        # exactly what the UPDATE did.
        return self.sc._conn.execute(
            "SELECT max_depth, solve_budget FROM subgroup_best_by_policy "
            "WHERE subset_key=? AND policy=? AND universe_id=?",
            (self.key, ERD_ALL, self.uid)).fetchone()

    def test_guard_skips_row_with_known_depth(self):
        # A worker's fresh tainted entry (real max_depth, solve_budget=5).
        self.sc.write(self.key, ERD_ALL, "crane", 1.5, max_depth=3, solve_budget=5)
        self.sc._conn.execute(self.UPDATE, (9, self.key, ERD_ALL, self.uid))
        md, sb = self._db_row()
        self.assertEqual(md, 3)      # max_depth unchanged
        self.assertEqual(sb, 5)      # solve_budget NOT clobbered to NULL

    def test_guard_fills_legacy_null_depth(self):
        self.sc.write(self.key, ERD_ALL, "crane", 1.5, max_depth=None,
                      solve_budget=None)
        self.sc._conn.execute(self.UPDATE, (4, self.key, ERD_ALL, self.uid))
        md, _sb = self._db_row()
        self.assertEqual(md, 4)      # backfilled


class TestMultistepKeyConsistency(unittest.TestCase):
    def test_no_cache_constraint_compliant_does_not_crash(self):
        from wordle import _multistep_stats
        answers = ["crane", "slate", "trace", "stale", "tales", "least"]
        guesses = answers + ["brain", "stove"]
        soln = Solution(answers, guesses, cache=None, score_cache=None)
        # Hard-mode (constraint_compliant) + no cache exercised the tuple-vs-int
        # key bug at decode_response(pat); it must now run cleanly.
        st = _multistep_stats("crane", soln, step2_pool=None,
                              constraint_compliant=True, all_words=guesses,
                              erd_cache=MemoryScoreCache())
        self.assertIn("step1", st)


if __name__ == "__main__":
    unittest.main()
