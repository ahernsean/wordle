"""Scaling guard for the parallel ERD swarm.

Two checks across 1, 2, and 4 workers:

1. Work amplification (deterministic, thread-driven): the TOTAL number of
   candidate evaluations across all workers must stay equal to the candidate
   count regardless of how many workers run.  If coordination regressed and
   workers redid each other's chunks, total work would balloon — this catches
   that without depending on wall-clock timing.  Each run must also still
   produce the correct ERD.  This is the primary performance regression guard.

2. Real multi-process run (fork only): actually spawn 1/2/4 worker processes on
   one branch and confirm they all produce the correct result.  A 30-second
   per-run timeout catches deadlocks and livelocks.  No wall-clock comparison
   is made: on a branch this small, process spawn overhead (~100 ms/fork)
   dominates solver time (~10 ms), making timing comparisons meaningful; and
   with SQLite chunk claims serializing writers, even 4 workers cooperating
   over many branches shows only modest wall-clock speedup.  The work-count
   invariant in test 1 is the reliable regression guard for parallelism.
"""
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from cache_sqlite import ScoreCache
from wordle_engine import ResponseCache, min_expected_guesses, ERD_ALL
import erd_swarm
from erd_swarm import _BranchWorker, ROOT_BUDGET, run_branch_solve
from erd_queue import ErdQueue, encode_subset

# 12-word branch -> ceil(12/3) = 4 chunks, so up to 4 workers each take a chunk.
BRANCH = ["crane", "slate", "trace", "stale", "tales", "least",
          "heart", "share", "rates", "earth", "brave", "cleat"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train", "grade",
                       "shine", "mount", "frost", "plumb", "dwarf", "gawky"]
DIVISOR = 3
MAX_CHUNKS = 256
WORKER_COUNTS = (1, 2, 4)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr in ("ANSWER_FILE", "WORDS_FILE"):
            p = mock.patch.object(
                erd_swarm, attr,
                self.answer_file if attr == "ANSWER_FILE" else self.words_file)
            p.start()
            self.addCleanup(p.stop)
        self.subset_key = encode_subset(BRANCH)

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _db(self, tag):
        return (os.path.join(self._tmp.name, f"cache_{tag}.sqlite3"),
                os.path.join(self._tmp.name, f"queue_{tag}.sqlite3"))

    def _register_branch(self, queue_path):
        chunk_size = ErdQueue.chunk_size_for(
            len(BRANCH), len(CANDIDATES), DIVISOR, MAX_CHUNKS)
        q = ErdQueue(queue_path)
        q.create_branch(self.subset_key, len(BRANCH), len(CANDIDATES),
                        chunk_size, budget=ROOT_BUDGET)
        q.close()
        return ErdQueue.n_chunks_for(len(CANDIDATES), chunk_size)

    def _ground_truth(self):
        cache_path, _ = self._db("truth")
        sc = ScoreCache(cache_path, BRANCH)
        erd = min_expected_guesses(BRANCH, ResponseCache(BRANCH, sc), sc,
                                   guesses=CANDIDATES, policy=ERD_ALL,
                                   budget=ROOT_BUDGET)
        sc.close()
        return erd

    def _read(self, cache_path):
        sc = ScoreCache(cache_path, BRANCH, checkpoint_on_close=False)
        res = sc.read(self.subset_key, ERD_ALL)
        sc.close()
        return res


class TestWorkDoesNotAmplify(_Base):
    def _solve_counting_work(self, n_workers):
        cache_path, queue_path = self._db(f"work{n_workers}")
        # Apply schema migrations once before the worker threads open the cache
        # concurrently (production always has a single pre-open: bootstrap, or
        # run_branch_solve's parent).
        ScoreCache(cache_path, BRANCH).close()
        n_chunks = self._register_branch(queue_path)
        self.assertGreaterEqual(n_chunks, 2, "need a multi-chunk branch")
        evaluated = []
        lock = threading.Lock()

        def worker(wid):
            w = _BranchWorker(wid, cache_path, queue_path, None,
                              DIVISOR, MAX_CHUNKS)
            try:
                w.solve_branch_focused(self.subset_key)
                with lock:
                    evaluated.append(w.n_ok + w.n_pruned + w.n_useless)
            finally:
                w.close()

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertFalse(any(t.is_alive() for t in threads), "a worker hung")
        return self._read(cache_path), sum(evaluated)

    def test_total_work_constant_across_worker_counts(self):
        truth = self._ground_truth()
        self.assertIsNotNone(truth)
        for nw in WORKER_COUNTS:
            with self.subTest(workers=nw):
                result, total_evaluated = self._solve_counting_work(nw)
                self.assertIsNotNone(result, "branch did not finalize")
                self.assertAlmostEqual(result[1], truth, places=6)
                # The whole point: each candidate is evaluated exactly once in
                # total, no matter how many workers split the chunks.  More
                # workers must not multiply the work.
                self.assertEqual(
                    total_evaluated, len(CANDIDATES),
                    f"{nw} workers evaluated {total_evaluated} candidates, "
                    f"expected {len(CANDIDATES)} (work amplification!)")


@unittest.skipUnless(
    "fork" in mp.get_all_start_methods() and (os.cpu_count() or 1) >= 2,
    "needs fork start method and >=2 CPUs")
class TestProcessScalingSmoke(_Base):
    def _solve_processes(self, n_workers):
        cache_path, queue_path = self._db(f"proc{n_workers}")
        t0 = time.time()
        result = run_branch_solve(
            self.subset_key, BRANCH, n_workers=n_workers,
            cache_path=cache_path, queue_path=queue_path,
            divisor=DIVISOR, max_chunks=MAX_CHUNKS,
            source_word="crane", source_pattern=0,
            timeout=30)
        return result, time.time() - t0

    def test_runs_and_agrees_at_1_2_4_workers(self):
        truth = self._ground_truth()
        results, times = {}, {}
        for nw in WORKER_COUNTS:
            with self.subTest(workers=nw):
                res, elapsed = self._solve_processes(nw)
                # timeout=30 in run_branch_solve means a deadlock/livelock
                # manifests as None rather than a hung test.
                self.assertIsNotNone(res, f"{nw}-worker run timed out or produced no result")
                self.assertAlmostEqual(res[1], truth, places=6)
                results[nw] = res
                times[nw] = elapsed
        erds = {round(r[1], 6) for r in results.values()}
        self.assertEqual(len(erds), 1, f"worker counts disagreed: {results}")
        sys.stderr.write(f"\n[scaling] wall times by workers: "
                         f"{ {k: round(v, 3) for k, v in times.items()} }\n")


if __name__ == "__main__":
    unittest.main()
