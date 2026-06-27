"""End-to-end test of the parallel ERD swarm algorithm.

Registers one branch and actually runs the swarm workers (cooperating on
disjoint candidate claims, sharing the running-best bound, finalizing exactly
once) and checks that the result is correct: it must equal both a single-worker
run and the ground-truth depth-limited ERD from min_expected_guesses.  This is
the regression guard that the parallelization hasn't broken the algorithm.

Workers are driven in threads (via _BranchWorker.solve_branch_focused, the same
loop the process entry point uses) with small word lists, so it runs in well
under a second without spawning processes.
"""
import os
import tempfile
import threading
import unittest
from unittest import mock

from cache_sqlite import ScoreCache
from wordle_engine import (
    ResponseCache, min_expected_guesses, ERD_ALL,
)
import erd_swarm
from erd_swarm import _BranchWorker, ROOT_BUDGET
from erd_queue import ERDQueue, encode_subset

# A branch of 8 answers and 15 candidate guesses: with single-candidate claiming,
# two workers genuinely interleave candidate evaluation and cooperate.
BRANCH = ["crane", "slate", "trace", "stale", "tales", "least", "heart", "share"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train", "grade", "shine"]


class TestParallelSwarmSolve(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._path("answers.txt", BRANCH)
        self.words_file = self._path("words.txt", CANDIDATES)
        # _BranchWorker loads these module-level paths in __init__.
        p1 = mock.patch.object(erd_swarm, "ANSWER_FILE", self.answer_file)
        p2 = mock.patch.object(erd_swarm, "WORDS_FILE", self.words_file)
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        self.branch_key = encode_subset(BRANCH)

    def _path(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _fresh_db(self, tag):
        return (os.path.join(self._tmp.name, f"cache_{tag}.sqlite3"),
                os.path.join(self._tmp.name, f"queue_{tag}.sqlite3"))

    def _swarm_solve(self, tag, n_workers):
        """Register the branch and run n_workers cooperating threads; return
        the finalized (best_guess, best_erd) the swarm wrote to the cache."""
        cache_path, queue_path = self._fresh_db(tag)
        # Apply schema migrations once before the worker threads open the cache
        # concurrently (production always has a single pre-open first).
        ScoreCache(cache_path, BRANCH).close()
        q = ERDQueue(queue_path)
        q.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.close()

        def worker(wid):
            w = _BranchWorker(wid, cache_path, queue_path, None)
            try:
                w.solve_branch_focused(self.branch_key)
            finally:
                w.close()

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertFalse(any(t.is_alive() for t in threads), "worker hung")

        # The branch must have finalized (its coordination rows are deleted).
        q = ERDQueue(queue_path)
        self.assertIsNone(q.get_branch(self.branch_key),
                          "branch was not finalized")
        q.close()

        sc = ScoreCache(cache_path, BRANCH, checkpoint_on_close=False)
        result = sc.read(self.branch_key, ERD_ALL)
        sc.close()
        self.assertIsNotNone(result, "no ERD written to the cache")
        return result

    def _ground_truth(self):
        cache_path, _ = self._fresh_db("truth")
        sc = ScoreCache(cache_path, BRANCH)
        rc = ResponseCache(BRANCH, sc)
        erd = min_expected_guesses(
            BRANCH, rc, sc, guesses=CANDIDATES, policy=ERD_ALL,
            budget=ROOT_BUDGET)
        sc.close()
        return erd

    def test_parallel_matches_serial_and_ground_truth(self):
        truth = self._ground_truth()
        self.assertIsNotNone(truth)

        serial_word, serial_erd = self._swarm_solve("serial", n_workers=1)
        par_word, par_erd = self._swarm_solve("parallel", n_workers=3)

        # Parallel result equals the single-worker result...
        self.assertAlmostEqual(par_erd, serial_erd, places=6)
        # ...and both equal the depth-limited ground-truth ERD.
        self.assertAlmostEqual(par_erd, truth, places=6)
        # The chosen guess must be a real candidate.
        self.assertIn(par_word, CANDIDATES)
        self.assertIn(serial_word, CANDIDATES)


class TestClaimSkipsCachedBranch(unittest.TestCase):
    """claim_one() must not re-claim candidates for a pending branch whose ERD
    is already reusable in ScoreCache at the worker's budget: it should mark
    the pending row done directly instead of promoting it for real work."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._path("answers.txt", BRANCH)
        self.words_file = self._path("words.txt", CANDIDATES)
        p1 = mock.patch.object(erd_swarm, "ANSWER_FILE", self.answer_file)
        p2 = mock.patch.object(erd_swarm, "WORDS_FILE", self.words_file)
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        self.branch_key = encode_subset(BRANCH)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _path(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _queue_branch(self):
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(self.branch_key, len(BRANCH), 0, "salet", 0)])
        q.close()

    def _worker(self):
        return _BranchWorker(0, self.cache_path, self.queue_path, None,
                             root_budget=ROOT_BUDGET)

    def test_skips_branch_with_reusable_cache_entry(self):
        sc = ScoreCache(self.cache_path, BRANCH)
        # Untainted entry, reusable at any budget >= max_depth.
        sc.write(self.branch_key, ERD_ALL, "crane", 1.5,
                 max_depth=2, solve_budget=None)
        sc.close()
        self._queue_branch()

        w = self._worker()
        try:
            self.assertIsNone(w.claim_one())
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        self.assertEqual(q.counts_by_status().get("done"), 1)
        self.assertIsNone(q.get_branch(self.branch_key),
                          "no active_branches row should have been created")
        q.close()

    def test_promotes_branch_with_unreusable_cache_entry(self):
        # A legacy entry (max_depth=None) isn't reusable at a finite budget.
        sc = ScoreCache(self.cache_path, BRANCH)
        sc.write(self.branch_key, ERD_ALL, "crane", 1.5)
        sc.close()
        self._queue_branch()

        w = self._worker()
        try:
            result = w.claim_one()
        finally:
            w.close()

        self.assertIsNotNone(result,
                             "branch should have been promoted for real work")
        branch, idx = result
        self.assertEqual(branch["branch_key"], self.branch_key)
        self.assertIsNotNone(idx)


class TestReportingInvariants(unittest.TestCase):
    """MaxD and path have different reporting scopes, neither tied to heartbeat
    timing.  MaxD resets at candidate start; path resets at screen refresh."""

    def _bare_worker(self):
        w = _BranchWorker.__new__(_BranchWorker)
        w._spine = {}
        w._hb_max_spine = {}
        w._log_max_spine = {}
        w._cand_max_depth = 0
        w._cur_depth = 0
        return w

    def test_cand_max_depth_resets_per_candidate(self):
        """_cand_max_depth is zeroed before each candidate so MaxD reflects
        only the current candidate's recursion."""
        w = self._bare_worker()

        w._note_depth(5, 50)
        w._note_depth(4, 12)
        w._note_depth(3, 4)
        self.assertEqual(w._cand_max_depth, 3)

        # evaluate_claim resets before candidate 2.
        w._cand_max_depth = 0

        w._note_depth(5, 30)
        self.assertEqual(w._cand_max_depth, 1)  # not 3 from candidate 1

    def test_spine_windows(self):
        """_hb_max_spine resets after each heartbeat write (2s window for the
        status display); _log_max_spine resets after each 120s progress log
        write.  Both reset when a new candidate claim starts."""
        w = self._bare_worker()

        w._note_depth(5, 50)
        w._note_depth(4, 12)
        w._note_depth(3, 4)

        # Both accumulators capture the depth-3 path.
        self.assertEqual(w._hb_spine_str().count('→'), 2)
        self.assertEqual(w._log_spine_str().count('→'), 2)

        # Heartbeat fires: hb resets, log accumulator is untouched.
        w._hb_max_spine = {}
        w._note_depth(5, 20)
        self.assertEqual(w._hb_spine_str().count('→'), 0)   # fresh 2s window
        self.assertEqual(w._log_spine_str().count('→'), 2)  # still has depth-3

        # Progress log fires: log resets.
        w._log_max_spine = {}
        w._note_depth(5, 15)
        self.assertEqual(w._log_spine_str().count('→'), 0)  # fresh 120s window

        # New candidate claim: both reset.
        w._note_depth(4, 8)
        w._hb_max_spine = {}
        w._log_max_spine = {}
        w._note_depth(5, 10)
        self.assertEqual(w._hb_spine_str().count('→'), 0)
        self.assertEqual(w._log_spine_str().count('→'), 0)

    def test_display_path_resets_each_screen_refresh(self):
        """max_paths in _print_status is rebuilt from scratch on each call so
        the path shown reflects only the current screen refresh, not history."""
        def one_refresh(cur_path):
            # Mirrors what _print_status does: start empty, take cur_path.
            max_paths = {}
            prev = max_paths.get('w', '')
            if cur_path.count('→') >= prev.count('→'):
                max_paths['w'] = cur_path
            return max_paths['w']

        shown1 = one_refresh('50→12→4')
        self.assertEqual(shown1, '50→12→4')

        # Next refresh starts empty: a shallower path is NOT displaced by history.
        shown2 = one_refresh('30→8')
        self.assertEqual(shown2, '30→8')

    def test_display_path_tied_depth_takes_most_recent(self):
        """Within one refresh, a path tied in depth with the current max
        overwrites it so the displayed path is the most recent at that depth."""
        max_paths = {}
        for path in ('50→12→4', '45→9→2'):
            prev = max_paths.get('w', '')
            if path.count('→') >= prev.count('→'):
                max_paths['w'] = path
        self.assertEqual(max_paths['w'], '45→9→2')


class TestBranchSpinePersistence(unittest.TestCase):
    """active_branches.spine round-trips through create_branch / reads, and its
    migration is additive and idempotent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def test_spine_round_trips(self):
        q = ERDQueue(self.queue_path)
        key = encode_subset(BRANCH)
        q.create_branch(key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, spine="SALET -g-g- CRANE bb-y-")
        row = q.get_branch(key)
        self.assertEqual(row["spine"], "SALET -g-g- CRANE bb-y-")
        # branches_in_progress (SELECT *) also surfaces it for the display.
        listed = {bytes(b["branch_key"]): b for b in q.branches_in_progress()}
        self.assertEqual(listed[key]["spine"], "SALET -g-g- CRANE bb-y-")
        q.close()

    def test_spine_defaults_to_null(self):
        q = ERDQueue(self.queue_path)
        key = encode_subset(BRANCH)
        q.create_branch(key, len(BRANCH), len(CANDIDATES), budget=ROOT_BUDGET)
        self.assertIsNone(q.get_branch(key)["spine"])
        q.close()

    def test_migration_is_idempotent_on_legacy_db(self):
        # Simulate a pre-spine database: pre-create active_branches WITHOUT the
        # spine column so ERDQueue's CREATE TABLE IF NOT EXISTS is a no-op and
        # _migrate is the only thing that can add the column.
        import sqlite3
        raw = sqlite3.connect(self.queue_path)
        raw.execute("""
            CREATE TABLE active_branches (
                branch_key BLOB PRIMARY KEY, n_words INTEGER NOT NULL,
                n_candidates INTEGER NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
                source_word TEXT, source_pattern INTEGER, best_erd REAL,
                best_guess TEXT, status TEXT NOT NULL DEFAULT 'open',
                created_at INTEGER, finalized_at INTEGER, budget INTEGER,
                best_max_depth INTEGER, tainted INTEGER NOT NULL DEFAULT 0,
                depth INTEGER NOT NULL DEFAULT 0, nodes_spent INTEGER NOT NULL DEFAULT 0)
        """)
        raw.commit()
        raw.close()

        q = ERDQueue(self.queue_path)            # _migrate adds the column
        cols = {r["name"] for r in
                q._conn.execute("PRAGMA table_info(active_branches)")}
        self.assertIn("spine", cols)
        q._migrate()                              # second run must not raise/dup
        cols2 = {r["name"] for r in
                 q._conn.execute("PRAGMA table_info(active_branches)")}
        self.assertEqual(cols, cols2)
        q.close()


if __name__ == "__main__":
    unittest.main()
