"""End-to-end test of the parallel ERD swarm algorithm.

Registers one branch and actually runs the swarm workers (cooperating on
disjoint candidate chunks, sharing the running-best bound, finalizing exactly
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
from erd_queue import ErdQueue, encode_subset

# A branch of 8 answers and 15 candidate guesses → 3 chunks (divisor 3), so two
# workers genuinely split the candidate list and cooperate.
BRANCH = ["crane", "slate", "trace", "stale", "tales", "least", "heart", "share"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train", "grade", "shine"]
DIVISOR = 3
MAX_CHUNKS = 256


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
        self.subset_key = encode_subset(BRANCH)

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
        the finalized (best_word, best_erd) the swarm wrote to the cache."""
        cache_path, queue_path = self._fresh_db(tag)
        # Apply schema migrations once before the worker threads open the cache
        # concurrently (production always has a single pre-open first).
        ScoreCache(cache_path, BRANCH).close()
        chunk_size = ErdQueue.chunk_size_for(
            len(BRANCH), len(CANDIDATES), DIVISOR, MAX_CHUNKS)
        q = ErdQueue(queue_path)
        q.create_branch(self.subset_key, len(BRANCH), len(CANDIDATES),
                        chunk_size, budget=ROOT_BUDGET)
        n_chunks = ErdQueue.n_chunks_for(len(CANDIDATES), chunk_size)
        q.close()
        self.assertGreaterEqual(n_chunks, 2, "test needs a multi-chunk branch")

        def worker(wid):
            w = _BranchWorker(wid, cache_path, queue_path, None,
                              DIVISOR, MAX_CHUNKS)
            try:
                w.solve_branch_focused(self.subset_key)
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
        q = ErdQueue(queue_path)
        self.assertIsNone(q.get_branch(self.subset_key),
                          "branch was not finalized")
        q.close()

        sc = ScoreCache(cache_path, BRANCH, checkpoint_on_close=False)
        result = sc.read(self.subset_key, ERD_ALL)
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


if __name__ == "__main__":
    unittest.main()
