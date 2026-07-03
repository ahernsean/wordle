"""Scaling guard for the parallel ERD swarm.

Three checks:

1. Work amplification (deterministic, thread-driven): the TOTAL number of
   candidate evaluations across all workers must stay equal to the candidate
   count regardless of how many workers run.  If coordination regressed and
   workers redid each other's candidates, total work would balloon — this catches
   that without depending on wall-clock timing.  Each run must also still
   produce the correct ERD.

2. Multi-process correctness (fork only): spawn 1, 2, and 4 real swarm_worker
   processes on one branch and confirm they all produce the same ERD as the
   serial result.  Catches races in inter-process coordination (update_branch_best,
   complete_chunk) that thread-based tests cannot exercise.

3. Cooperative drain timing (fork only): spawn 1 vs 4 swarm_workers, drain 80
   disjoint branches from a shared queue, and assert 4 workers finish in < 80%
   of 1-worker time.  Key design constraints that make the comparison
   meaningful are documented on TestCooperativeDrainSmoke.
"""
import json
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
from erd_swarm import _BranchWorker, ROOT_BUDGET
from erd_queue import ERDQueue, encode_subset

# 12-word branch with 24 candidates: up to 24 workers each claim one candidate.
BRANCH = ["crane", "slate", "trace", "stale", "tales", "least",
          "heart", "share", "rates", "earth", "brave", "cleat"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train", "grade",
                       "shine", "mount", "frost", "plumb", "dwarf", "gawky"]
WORKER_COUNTS = (1, 2, 4)

# The multi-process scaling smoke tests need a fork start method and >=2 CPUs.
# Where those aren't available the tests skip, and a skipped scaling test looks
# identical to a passing suite in the CI summary — a green check would then hide
# the fact that the swarm was never actually exercised across processes. Emit a
# loud line to stderr so the skip is visible in the run log.
SCALING_SMOKE_REQS_MET = "fork" in mp.get_all_start_methods() and (os.cpu_count() or 1) >= 2
SCALING_SMOKE_SKIP_REASON = "needs fork start method and >=2 CPUs"

if not SCALING_SMOKE_REQS_MET:
    print(
        f"WARNING: skipping multi-process scaling smoke tests "
        f"(TestProcessScalingSmoke, TestCooperativeDrainSmoke): "
        f"{SCALING_SMOKE_SKIP_REASON} "
        f"(fork={'fork' in mp.get_all_start_methods()}, cpus={os.cpu_count()})",
        file=sys.stderr, flush=True)


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
        q = ERDQueue(queue_path)
        q.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.close()

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
        # concurrently (production always has a single pre-open: queue-add).
        ScoreCache(cache_path, BRANCH).close()
        self._register_branch(queue_path)
        evaluated = []
        lock = threading.Lock()

        def worker(wid):
            # Pure-partition path: the "each candidate evaluated exactly once"
            # invariant holds only with adaptive decomposition off, since
            # publishing a frame's remainder re-routes candidates through
            # cooperative claims rather than this worker's own loop.
            w = _BranchWorker(wid, cache_path, queue_path, None,
                              enable_adaptive_decomposition=False)
            try:
                w.solve_branch_focused(self.branch_key)
                with lock:
                    evaluated.append(w.n_ok + w.n_cutoff + w.n_pruned + w.n_useless)
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
                # total, no matter how many workers cooperate.  More workers
                # must not multiply the work.
                self.assertEqual(
                    total_evaluated, len(CANDIDATES),
                    f"{nw} workers evaluated {total_evaluated} candidates, "
                    f"expected {len(CANDIDATES)} (work amplification!)")


@unittest.skipUnless(SCALING_SMOKE_REQS_MET, SCALING_SMOKE_SKIP_REASON)
class TestProcessScalingSmoke(_Base):
    """Verify that ERD results from real forked swarm_worker processes agree.

    Spawns 1, 2, and 4 processes via swarm_worker (the same entry point as
    production workers) and confirms each produces the same ERD as the serial
    ground truth.  This is the multi-process correctness guard: it catches
    races in update_branch_best, complete_candidate, and claim/release that
    thread-based tests cannot exercise.
    """

    def _solve_processes(self, n_workers, timeout=30):
        from erd_swarm import swarm_worker
        cache_path, queue_path = self._db(f"proc{n_workers}")
        ScoreCache(cache_path, BRANCH).close()
        q = ERDQueue(queue_path)
        q.add_pending_many([(self.branch_key, len(BRANCH), 0, "crane", 0)])
        q.close()

        stop_event = mp.Event()
        with mock.patch("erd_swarm._setup_logging", lambda *_: None):
            # Pure-partition path: this guard catches races in update_branch_best,
            # complete_candidate, and claim/release — primitives that sit below the
            # adaptive layer and are exercised identically without it.  Adaptive-on
            # correctness is covered by test_erd_parallel.
            procs = [mp.Process(target=swarm_worker,
                                args=(w, cache_path, queue_path, stop_event,
                                      1, False))
                     for w in range(n_workers)]
            for p in procs:
                p.start()

        deadline = time.time() + timeout
        q = ERDQueue(queue_path)
        try:
            while time.time() < deadline:
                if q.counts_by_status().get('done', 0) >= 1:
                    break
                time.sleep(0.1)
        finally:
            stop_event.set()
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
            q.close()

        return self._read(cache_path)

    def test_erd_agrees_across_worker_counts(self):
        truth = self._ground_truth()
        self.assertIsNotNone(truth)
        n_cpu = os.cpu_count() or 1
        for nw in [w for w in WORKER_COUNTS if w <= n_cpu]:
            with self.subTest(workers=nw):
                result = self._solve_processes(nw)
                self.assertIsNotNone(result,
                                     f"{nw}-worker run timed out or produced no result")
                self.assertAlmostEqual(result[1], truth, places=6)


@unittest.skipUnless(SCALING_SMOKE_REQS_MET, SCALING_SMOKE_SKIP_REASON)
class TestCooperativeDrainSmoke(unittest.TestCase):
    # -------------------------------------------------------------------------
    # PURPOSE: verify that N cooperative swarm workers drain a shared queue
    # faster than 1 worker.  This is a PARALLELISM REGRESSION GUARD — if
    # coordination overhead grows (lock contention, redundant work, etc.),
    # the speedup shrinks and the test catches it.
    #
    # DO NOT replace the timing assertion with a pure correctness check.
    # Correctness is covered by TestProcessScalingSmoke (multi-process ERD
    # agreement) and TestWorkDoesNotAmplify (no candidate duplication).
    # This test's only job is to confirm that parallelism HELPS.
    #
    # Worker count is min(4, cpu_count) so the comparison is always honest:
    # N workers on N CPUs should each get a full core, giving near-linear
    # speedup.  On a 2-CPU CI runner this runs 2 workers vs 1; on Rocky it
    # runs 4 workers vs 1.  The 80% threshold is achievable on any of these.
    # -------------------------------------------------------------------------

    _BRANCH_SIZE = 12
    _N_BRANCHES = 80         # 80 × 12 = 960 unique answer words
    _N_CANDIDATES = 100
    _SPEEDUP_RATIO = 0.80    # N workers must complete in < 80% of 1-worker time

    def setUp(self):
        # Use /dev/shm (RAM-backed tmpfs) when available so SQLite I/O doesn't
        # serialize workers and mask the parallelism signal.  Falls back to a
        # normal tempdir on macOS or systems without /dev/shm.
        shm = '/dev/shm'
        tmp_dir = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) else None
        self._tmp = tempfile.TemporaryDirectory(dir=tmp_dir)
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
        from erd_swarm import swarm_worker
        cache_path = os.path.join(self._tmp.name, f"cache_{tag}.sqlite3")
        queue_path = os.path.join(self._tmp.name, f"queue_{tag}.sqlite3")

        t_setup0 = time.time()
        ScoreCache(cache_path, self._pool).close()
        q = ERDQueue(queue_path)
        for bw in self._branches:
            q.create_branch(encode_subset(bw), self._BRANCH_SIZE,
                            len(self._candidates), budget=ROOT_BUDGET)
        q.close()
        t_setup = time.time() - t_setup0

        stop_event = mp.Event()
        # Suppress erd_worker_N.log creation: mock is inherited by forked children.
        with mock.patch("erd_swarm._setup_logging", lambda *_: None):
            # Disable the adaptive-decomposition layer (cost model, probing,
            # overrun): this test measures pure candidate-partition scaling.
            procs = [mp.Process(target=swarm_worker,
                                args=(w, cache_path, queue_path, stop_event,
                                      1, False))
                     for w in range(n_workers)]
            t0 = time.time()
            for p in procs:
                p.start()
            t_spawn = time.time() - t0

        deadline = time.time() + timeout
        q = ERDQueue(queue_path)
        try:
            while time.time() < deadline:
                if not q.branches_in_progress():
                    break
                time.sleep(0.05)
            t_drain = time.time() - t0
        finally:
            q.close()
        stop_event.set()
        for p in procs:
            p.join(timeout=15)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join()
        t_total = time.time() - t0

        return {
            'wall':    t_total,
            'drain':   t_drain,
            'spawn':   t_spawn,
            'setup':   t_setup,
            'workers': n_workers,
            'cache':   cache_path,
        }

    @staticmethod
    def _publish_summary(rows, n, t1, tN, passed):
        """Write a markdown timing table to $GITHUB_STEP_SUMMARY if available."""
        summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if not summary_path:
            return
        status = '✅ PASSED' if passed else '❌ FAILED'
        lines = [
            f'## Parallelism drain test — {status}',
            '',
            f'**{n} workers vs 1 worker** | '
            f'speedup: **{t1/tN:.2f}x** | '
            f'1-worker: {t1:.3f}s | {n}-worker: {tN:.3f}s',
            '',
            '| run | workers | setup (s) | spawn (s) | drain (s) | total (s) | ms/branch |',
            '|-----|---------|-----------|-----------|-----------|-----------|-----------|',
        ]
        n_branches = rows[0].get('n_branches', 80)
        for r in rows:
            ms_b = r['drain'] * 1000 / n_branches
            lines.append(
                f'| {r["tag"]} | {r["workers"]} '
                f'| {r["setup"]:.3f} | {r["spawn"]:.3f} '
                f'| {r["drain"]:.3f} | {r["wall"]:.3f} '
                f'| {ms_b:.1f} |'
            )
        lines += [
            '',
            f'**cpu_count:** {os.cpu_count()}  '
            f'**branches:** {n_branches}  '
            f'**candidates:** {rows[0].get("n_candidates", "?")}  '
            f'**branch_size:** {rows[0].get("branch_size", "?")}',
        ]
        with open(summary_path, 'a') as f:
            f.write('\n'.join(lines) + '\n')

    @staticmethod
    def _write_result_file(n, t1, tN, passed, ratio):
        """Record the measured drain speedup as JSON so a dedicated CI step
        can display and re-check it, the way `coverage report` surfaces the
        coverage number in its own step. Path is $SCALING_RESULT_PATH (default
        scaling_result.json in the cwd)."""
        path = os.environ.get('SCALING_RESULT_PATH', 'scaling_result.json')
        with open(path, 'w') as f:
            json.dump({
                'workers':      n,
                't1_drain':     t1,
                'tN_drain':     tN,
                'speedup':      t1 / tN,
                'min_speedup':  1 / ratio,
                'threshold_ratio': ratio,
                'passed':       passed,
                'cpu_count':    os.cpu_count(),
            }, f)

    def test_Nworkers_faster_than_1worker(self):
        n = min(4, os.cpu_count() or 1)
        r1 = self._drain(1, "w1")
        rN = self._drain(n, f"w{n}")
        t1, tN = r1['drain'], rN['drain']

        for r, tag in [(r1, '1-worker'), (rN, f'{n}-worker')]:
            r['tag'] = tag
            r['n_branches'] = self._N_BRANCHES
            r['n_candidates'] = self._N_CANDIDATES
            r['branch_size'] = self._BRANCH_SIZE

        passed = tN < t1 * self._SPEEDUP_RATIO
        self._publish_summary([r1, rN], n, t1, tN, passed)
        self._write_result_file(n, t1, tN, passed, self._SPEEDUP_RATIO)

        sys.stderr.write(
            f"\n[drain] cpu_count={os.cpu_count()}  "
            f"branches={self._N_BRANCHES}  candidates={self._N_CANDIDATES}\n"
            f"  1-worker : {t1:.3f}s drain  {r1['spawn']:.3f}s spawn  "
            f"{t1*1000/self._N_BRANCHES:.1f}ms/branch\n"
            f"  {n}-worker: {tN:.3f}s drain  {rN['spawn']:.3f}s spawn  "
            f"{tN*1000/self._N_BRANCHES:.1f}ms/branch  "
            f"({t1/tN:.2f}x speedup)\n"
        )
        self.assertLess(
            tN, t1 * self._SPEEDUP_RATIO,
            f"{n} workers ({tN:.3f}s) not fast enough vs "
            f"1 worker ({t1:.3f}s); expected < {t1 * self._SPEEDUP_RATIO:.3f}s")
        # Sanity: every branch must also have produced a valid cache entry.
        sc = ScoreCache(rN['cache'], self._pool)
        missing = [bw for bw in self._branches
                   if sc.read(encode_subset(bw), ERD_ALL) is None]
        sc.close()
        self.assertEqual(missing, [],
                         f"{len(missing)} branches missing from cache")


if __name__ == "__main__":
    unittest.main()
