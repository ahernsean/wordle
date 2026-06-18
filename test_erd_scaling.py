"""Scaling guard for the parallel ERD swarm.

Three checks:

1. Work amplification (deterministic, thread-driven): the TOTAL number of
   candidate evaluations across all workers must stay equal to the candidate
   count regardless of how many workers run.  If coordination regressed and
   workers redid each other's chunks, total work would balloon — this catches
   that without depending on wall-clock timing.  Each run must also still
   produce the correct ERD.

2. Real multi-process run (fork only): actually spawn 1/2/4 worker processes on
   one branch and confirm they all produce the correct result.  A 30-second
   per-run timeout catches deadlocks and livelocks.  No wall-clock comparison
   is made: on a branch this small, process spawn overhead (~100 ms/fork)
   dominates solver time (~10 ms), making timing comparisons meaningless.

3. Cooperative drain timing (fork only): spawn 1 vs 4 swarm_workers, drain 80
   disjoint branches from a shared queue, and assert 4 workers finish in < 80%
   of 1-worker time.  Key design constraints that make the comparison
   meaningful are documented on TestCooperativeDrainSmoke.
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
from erd_queue import ERDQueue, encode_subset

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
        self.branch_key = encode_subset(BRANCH)

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _db(self, tag):
        return (os.path.join(self._tmp.name, f"cache_{tag}.sqlite3"),
                os.path.join(self._tmp.name, f"queue_{tag}.sqlite3"))

    def _register_branch(self, queue_path):
        chunk_size = ERDQueue.chunk_size_for(
            len(BRANCH), len(CANDIDATES), DIVISOR, MAX_CHUNKS)
        q = ERDQueue(queue_path)
        q.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                        chunk_size, budget=ROOT_BUDGET)
        q.close()
        return ERDQueue.n_chunks_for(len(CANDIDATES), chunk_size)

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
        res = sc.read(self.branch_key, ERD_ALL)
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
                w.solve_branch_focused(self.branch_key)
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
            self.branch_key, BRANCH, n_workers=n_workers,
            cache_path=cache_path, queue_path=queue_path,
            min_words_per_chunk=DIVISOR, max_chunk_count=MAX_CHUNKS,
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


@unittest.skipUnless(
    "fork" in mp.get_all_start_methods() and (os.cpu_count() or 1) >= 2,
    "needs fork start method and >=2 CPUs")
class TestCooperativeDrainSmoke(unittest.TestCase):
    # -------------------------------------------------------------------------
    # PURPOSE: verify that N cooperative swarm workers drain a shared queue
    # faster than 1 worker.  This is a PARALLELISM REGRESSION GUARD — if
    # coordination overhead grows (lock contention, redundant work, etc.),
    # the speedup shrinks and the test catches it.
    #
    # DO NOT replace the timing assertion with a pure correctness check.
    # Correctness is covered by TestProcessScalingSmoke and TestWorkDoesNotAmplify.
    # This test's only job is to confirm that parallelism HELPS.
    #
    # Worker count is min(4, cpu_count) so the comparison is always honest:
    # N workers on N CPUs should each get a full core, giving near-linear
    # speedup.  On a 2-CPU CI runner this runs 2 workers vs 1; on Rocky it
    # runs 4 workers vs 1.  The 80% threshold is achievable on any of these.
    # -------------------------------------------------------------------------

    _BRANCH_SIZE = 12
    _N_BRANCHES = 80         # 80 × 12 = 960 unique answer words
    _DRAIN_DIVISOR = 100     # ceil(12/100)=1 → 1 chunk per branch
    _N_CANDIDATES = 400      # must dominate SQLite overhead so parallelism shows
    _SPEEDUP_RATIO = 0.80    # N workers must complete in < 80% of 1-worker time

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        with open("NYT_wordlist.txt") as f:
            nyt = [l.strip() for l in f if l.strip()]
        with open("wordle.txt") as f:
            wl = [l.strip() for l in f if l.strip()]
        self._pool = nyt[:self._N_BRANCHES * self._BRANCH_SIZE]
        self._branches = [self._pool[i * self._BRANCH_SIZE:(i + 1) * self._BRANCH_SIZE]
                          for i in range(self._N_BRANCHES)]
        self._candidates = wl[:self._N_CANDIDATES]
        af = os.path.join(self._tmp.name, "answers.txt")
        wf = os.path.join(self._tmp.name, "words.txt")
        with open(af, "w") as f:
            f.write("\n".join(self._pool) + "\n")
        with open(wf, "w") as f:
            f.write("\n".join(self._candidates) + "\n")
        for attr, path in [("ANSWER_FILE", af), ("WORDS_FILE", wf)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self._af = af
        self._wf = wf

    def _drain(self, n_workers, tag, timeout=120):
        """Run n_workers swarm workers to drain all branches; return (cache_path, queue_path)."""
        from erd_swarm import swarm_worker
        cache_path = os.path.join(self._tmp.name, f"cache_{tag}.sqlite3")
        queue_path = os.path.join(self._tmp.name, f"queue_{tag}.sqlite3")
        ScoreCache(cache_path, self._pool).close()
        chunk_size = ERDQueue.chunk_size_for(
            self._BRANCH_SIZE, len(self._candidates),
            self._DRAIN_DIVISOR, MAX_CHUNKS)
        q = ERDQueue(queue_path)
        for bw in self._branches:
            q.create_branch(encode_subset(bw), self._BRANCH_SIZE,
                            len(self._candidates), chunk_size,
                            budget=ROOT_BUDGET)
        q.close()
        stop_event = mp.Event()
        # Suppress erd_worker_N.log creation: mock is inherited by forked children.
        with mock.patch("erd_swarm._setup_logging", lambda *_: None):
            procs = [mp.Process(target=swarm_worker,
                                args=(w, cache_path, queue_path, stop_event,
                                      self._DRAIN_DIVISOR, MAX_CHUNKS))
                     for w in range(n_workers)]
            t0 = time.time()
            for p in procs:
                p.start()
        deadline = time.time() + timeout
        q = ERDQueue(queue_path)
        try:
            while time.time() < deadline:
                if not q.branches_in_progress():
                    break
                time.sleep(0.05)
            elapsed = time.time() - t0
        finally:
            q.close()
        stop_event.set()
        for p in procs:
            p.join(timeout=15)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join()
        return elapsed, cache_path

    def test_Nworkers_faster_than_1worker(self):
        n = min(4, os.cpu_count() or 1)
        t1, _         = self._drain(1, "w1")
        tN, cache_path = self._drain(n, f"w{n}")
        sys.stderr.write(
            f"\n[drain] 1-worker: {t1:.3f}s, {n}-worker: {tN:.3f}s "
            f"({t1 / tN:.2f}x speedup)\n")
        self.assertLess(
            tN, t1 * self._SPEEDUP_RATIO,
            f"{n} workers ({tN:.3f}s) not fast enough vs "
            f"1 worker ({t1:.3f}s); expected < {t1 * self._SPEEDUP_RATIO:.3f}s")
        # Sanity: every branch must also have produced a valid cache entry.
        sc = ScoreCache(cache_path, self._pool)
        missing = [bw for bw in self._branches
                   if sc.read(encode_subset(bw), ERD_ALL) is None]
        sc.close()
        self.assertEqual(missing, [],
                         f"{len(missing)} branches missing from cache")


if __name__ == "__main__":
    unittest.main()
