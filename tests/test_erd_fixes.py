"""Tests for the parallel-ERD-infra correctness fixes:

- liveness-gated candidate claim reclaim (a slow-but-alive worker is never
  reclaimed; a dead worker's claim is freed),
- per-worker claim reclaim used by the supervisor on respawn,
- import_cache preferring an untainted entry over a tainted one,
- backfill_max_depth not clobbering a worker's fresh solve_budget,
- _multistep_stats keying response groups consistently with/without a cache.
"""
import math
import os
import sqlite3
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache, MemoryScoreCache
from wordle_engine import ResponseCache, ERD_ALL, Solution, erd_ge, erd_numerator
from erd_queue import ERDQueue
import import_cache


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
        self.q = ERDQueue(self.path("q.sqlite3"))
        self.addCleanup(self.q.close)
        self.key = ScoreCache.encode_subset(WORDS)

    def _insert_claim(self, worker, claimed_at, done=0, idx=0):
        self.q._conn.execute(
            "INSERT INTO candidate_claims (branch_id, idx, claimed_by, claimed_at, done) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.q._intern_branch(self.key, create=True), idx, worker,
             claimed_at, done))

    def _n_claims(self):
        return self.q._conn.execute(
            "SELECT COUNT(*) FROM candidate_claims").fetchone()[0]

    def test_live_worker_claim_not_reclaimed(self):
        now = int(time.time())
        self._insert_claim("worker-1", claimed_at=now - 1000)  # an old claim
        # ...but the worker is alive: it heartbeat just now.
        self.q.heartbeat("worker-1", pid=1, current_branch_key=self.key,
                         n_words=5, started_at=now, claims_done=0)
        freed = self.q.reclaim_stale_claims(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 0)
        self.assertEqual(self._n_claims(), 1)

    def test_dead_worker_claim_reclaimed(self):
        now = int(time.time())
        self._insert_claim("worker-2", claimed_at=now - 1000)
        # Worker heartbeat is stale (>120s old) → presumed dead.
        self.q.heartbeat("worker-2", pid=1, current_branch_key=self.key,
                         n_words=5, started_at=now, claims_done=0)
        self.q._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = 'worker-2'",
            (now - 1000,))
        freed = self.q.reclaim_stale_claims(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 1)
        self.assertEqual(self._n_claims(), 0)

    def test_no_heartbeat_claim_reclaimed(self):
        now = int(time.time())
        self._insert_claim("worker-3", claimed_at=now - 1000)  # never heartbeat
        freed = self.q.reclaim_stale_claims(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 1)

    def test_done_claim_never_reclaimed(self):
        now = int(time.time())
        self._insert_claim("worker-4", claimed_at=now - 1000, done=1)
        freed = self.q.reclaim_stale_claims(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, 0)
        self.assertEqual(self._n_claims(), 1)

    def test_reclaim_claims_of_worker(self):
        now = int(time.time())
        self._insert_claim("worker-5", claimed_at=now, done=0, idx=0)
        self._insert_claim("worker-5", claimed_at=now, done=1, idx=1)  # done=1, keep
        freed = self.q.reclaim_claims_of_worker("worker-5")
        self.assertEqual(freed, 1)  # only the done=0 row
        self.assertEqual(self._n_claims(), 1)


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
        cols = import_cache._all_cols(conn, "branch_best_by_policy")
        col_list = ", ".join(cols)
        insert_sql = import_cache._insert_sql("branch_best_by_policy", cols)
        rows = conn.execute(
            f"SELECT {col_list} FROM src.branch_best_by_policy").fetchall()
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

    UPDATE = ("UPDATE branch_best_by_policy SET max_depth=?, solve_budget=NULL "
              "WHERE branch_key=? AND policy=? AND answer_list_id=? "
              "AND max_depth IS NULL")

    def setUp(self):
        super().setUp()
        self.sc = ScoreCache(self.path("c.sqlite3"), WORDS)
        self.addCleanup(self.sc.close)
        self.uid = self.sc.answer_list_id
        self.key = ScoreCache.encode_subset(WORDS)

    def _db_row(self):
        # Read straight from the DB (not the in-memory mirror) so we observe
        # exactly what the UPDATE did.
        return self.sc._conn.execute(
            "SELECT max_depth, solve_budget FROM branch_best_by_policy "
            "WHERE branch_key=? AND policy=? AND answer_list_id=?",
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


class TestLastWriteTs(_TmpDB, unittest.TestCase):
    def test_none_when_empty_then_timestamp_after_write(self):
        sc = ScoreCache(self.path("c.sqlite3"), WORDS)
        self.addCleanup(sc.close)
        self.assertIsNone(sc.last_write_ts())  # no ERD rows yet
        sc.write(ScoreCache.encode_subset(WORDS), ERD_ALL, "crane", 1.5,
                 max_depth=2, solve_budget=None)
        self.assertIsNotNone(sc.last_write_ts())


class TestLRUDict(unittest.TestCase):
    """_LRUDict: eviction order, capacity boundary, interface contract."""

    def _make(self, size):
        from cache_sqlite import _LRUDict
        return _LRUDict(max_size=size)

    def test_get_on_miss_returns_default(self):
        d = self._make(4)
        self.assertIsNone(d.get("x"))
        self.assertEqual(d.get("x", 99), 99)

    def test_basic_set_and_get(self):
        d = self._make(4)
        d["a"] = 1
        self.assertEqual(d.get("a"), 1)
        self.assertIn("a", d)

    def test_overwrite_updates_value(self):
        d = self._make(4)
        d["a"] = 1
        d["a"] = 2
        self.assertEqual(d.get("a"), 2)
        self.assertEqual(len(d), 1)

    def test_evicts_lru_on_capacity(self):
        d = self._make(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        # "a" is the LRU; inserting "d" should evict it.
        d["d"] = 4
        self.assertNotIn("a", d)
        self.assertIn("b", d)
        self.assertIn("c", d)
        self.assertIn("d", d)

    def test_access_updates_lru_order(self):
        d = self._make(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        # Access "a" so "b" becomes the LRU.
        _ = d.get("a")
        d["d"] = 4
        self.assertNotIn("b", d)
        self.assertIn("a", d)

    def test_write_updates_lru_order(self):
        d = self._make(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        # Overwrite "a" so "b" becomes the LRU.
        d["a"] = 10
        d["d"] = 4
        self.assertNotIn("b", d)
        self.assertIn("a", d)

    def test_pop_removes_entry(self):
        d = self._make(4)
        d["a"] = 1
        d.pop("a", None)
        self.assertNotIn("a", d)
        self.assertEqual(len(d), 0)

    def test_pop_with_default_on_miss(self):
        d = self._make(4)
        self.assertEqual(d.pop("missing", 99), 99)

    def test_len_reflects_evictions(self):
        d = self._make(2)
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(len(d), 2)
        d["c"] = 3           # evicts "a"
        self.assertEqual(len(d), 2)

    def test_unbounded_when_max_size_none(self):
        from cache_sqlite import _LRUDict
        d = _LRUDict(max_size=None)
        for i in range(1000):
            d[i] = i
        self.assertEqual(len(d), 1000)


class TestScoreCacheLRU(_TmpDB, unittest.TestCase):
    """ScoreCache._mem_cache evicts via LRU when max_mem_entries is set."""

    def test_lru_eviction_limits_mem_cache_size(self):
        sc = ScoreCache(self.path("cache.db"), WORDS, max_mem_entries=2)
        self.addCleanup(sc.close)

        k1 = ScoreCache.encode_subset(WORDS[:2])
        k2 = ScoreCache.encode_subset(WORDS[1:3])
        k3 = ScoreCache.encode_subset(WORDS[2:4])

        sc.write(k1, ERD_ALL, "crane", 1.5, max_depth=2)
        sc.write(k2, ERD_ALL, "slate", 2.0, max_depth=2)
        # Cache has 2 entries; inserting k3 should evict k1.
        sc.write(k3, ERD_ALL, "trace", 1.8, max_depth=2)

        # k1 was evicted from _mem_cache; read should still hit SQLite.
        result = sc.read(k1, ERD_ALL)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "crane")

    def test_no_limit_when_max_mem_entries_none(self):
        sc = ScoreCache(self.path("cache.db"), WORDS, max_mem_entries=None)
        self.addCleanup(sc.close)
        for i, w in enumerate(WORDS):
            key = ScoreCache.encode_subset([w])
            sc.write(key, ERD_ALL, w, float(i), max_depth=1)
        self.assertEqual(len(sc._mem_cache), len(WORDS))


class TestERDQueueManagement(_TmpDB, unittest.TestCase):
    """queue-clear, set_priority, remove_pending."""

    def _make_queue(self):
        q = ERDQueue(self.path("queue.db"))
        self.addCleanup(q.close)
        return q

    def _add_branch(self, q, words, priority=0):
        key = ScoreCache.encode_subset(words)
        q.add_pending_many([(key, len(words), priority, "crane", 0)])
        return key

    def test_clear_wipes_all_tables(self):
        q = self._make_queue()
        self._add_branch(q, WORDS[:2])
        self._add_branch(q, WORDS[1:3])
        counts_before = q.counts_by_status()
        self.assertGreater(counts_before.get("pending", 0), 0)

        q.clear()

        counts_after = q.counts_by_status()
        self.assertEqual(counts_after.get("pending", 0), 0)
        self.assertEqual(counts_after.get("done", 0), 0)

    def test_set_priority_updates_pending_branch(self):
        q = self._make_queue()
        key = self._add_branch(q, WORDS[:2], priority=0)
        updated = q.set_priority(key, 5)
        self.assertTrue(updated)
        row = q.get_pending_branch(key)
        self.assertEqual(row["priority"], 5)

    def test_set_priority_returns_false_for_unknown_branch(self):
        q = self._make_queue()
        fake_key = ScoreCache.encode_subset(["zzzzz"])
        self.assertFalse(q.set_priority(fake_key, 3))

    def test_remove_pending_deletes_pending_branch(self):
        q = self._make_queue()
        key = self._add_branch(q, WORDS[:2])
        removed = q.remove_pending(key)
        self.assertTrue(removed)
        self.assertIsNone(q.get_pending_branch(key))

    def test_remove_pending_returns_false_for_unknown_branch(self):
        q = self._make_queue()
        fake_key = ScoreCache.encode_subset(["zzzzz"])
        self.assertFalse(q.remove_pending(fake_key))

    def test_get_pending_branch_returns_row(self):
        q = self._make_queue()
        key = self._add_branch(q, WORDS[:3], priority=2)
        row = q.get_pending_branch(key)
        self.assertIsNotNone(row)
        self.assertEqual(row["priority"], 2)
        self.assertEqual(row["n_words"], 3)

    def test_priority_upgrade_on_duplicate_add(self):
        q = self._make_queue()
        key = self._add_branch(q, WORDS[:2], priority=0)
        # add_pending_many upgrades priority on conflict
        q.add_pending_many([(key, len(WORDS[:2]), 5, "crane", 0)])
        row = q.get_pending_branch(key)
        self.assertEqual(row["priority"], 5)

    def test_priority_not_downgraded_on_duplicate_add(self):
        q = self._make_queue()
        key = self._add_branch(q, WORDS[:2], priority=5)
        q.add_pending_many([(key, len(WORDS[:2]), 1, "crane", 0)])
        row = q.get_pending_branch(key)
        self.assertEqual(row["priority"], 5)


class TestErdGe(unittest.TestCase):
    """erd_numerator/erd_ge compare k/N-grid ERD values at exact rational
    precision, so two float64 images of the same rational never spuriously
    compare unequal."""

    def test_same_rational_via_different_arithmetic_compares_equal(self):
        # (n_answers, k) pairs confirmed to produce two distinct float64
        # images of k/n_answers via direct division vs. repeated addition.
        for n_answers, k in [(7, 5), (19, 13), (45, 5), (3209, 3)]:
            direct = k / n_answers
            accumulated = 0.0
            for _ in range(k):
                accumulated += 1 / n_answers
            self.assertNotEqual(direct, accumulated)  # float noise is real
            self.assertTrue(erd_ge(direct, accumulated, n_answers))
            self.assertTrue(erd_ge(accumulated, direct, n_answers))
            self.assertEqual(erd_numerator(direct, n_answers), k)
            self.assertEqual(erd_numerator(accumulated, n_answers), k)

    def test_adjacent_rationals_compare_strictly(self):
        for n_answers, k in [(7, 3), (19, 49), (45, 61), (3209, 8123)]:
            lower = k / n_answers
            higher = (k + 1) / n_answers
            self.assertTrue(erd_ge(higher, lower, n_answers))
            self.assertFalse(erd_ge(lower, higher, n_answers))

    def test_ulp_noise_does_not_flip_the_comparison(self):
        # The exact float64-noise scenario from the swarm: two images of the
        # same k/N value differing by one ULP must compare as equal (>= both
        # ways), never as a spurious re-solve trigger.
        value = 13 / 5
        nudged = math.nextafter(value, math.inf)
        self.assertNotEqual(value, nudged)
        self.assertTrue(erd_ge(value, nudged, 5))
        self.assertTrue(erd_ge(nudged, value, 5))

    def test_off_grid_value_returns_no_numerator(self):
        # 2.5 is not on the 5-word k/5 grid (12.5 is not near an integer).
        self.assertIsNone(erd_numerator(2.5, 5))
        self.assertEqual(erd_numerator(2.4, 5), 12)

    def test_off_grid_operand_falls_back_to_raw_float_compare(self):
        # An off-grid operand must never raise: erd_ge degrades to the
        # pre-fix float compare instead of crashing the caller, since this
        # comparison only ever gates reuse of already-proven work.
        self.assertTrue(erd_ge(3.0, 2.5, 5))     # raw: 3.0 >= 2.5
        self.assertFalse(erd_ge(2.0, 2.5, 5))    # raw: 2.0 >= 2.5
        self.assertTrue(erd_ge(2.5, 2.5, 5))     # both off-grid, equal raw


if __name__ == "__main__":
    unittest.main()
