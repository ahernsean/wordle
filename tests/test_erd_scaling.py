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

3. Coordination-floor drain smoke (fork only): spawn 1 vs 4 swarm_workers,
   drain 80 tiny disjoint branches from a shared queue, and assert 4 workers
   finish in < 90% of 1-worker time.  The branches carry near-zero solve work,
   so this measures the coordination fabric alone: it guards against total
   serialization, NOT strong scaling.  Key design constraints are documented
   on TestCooperativeDrainSmoke.

4. Solve-dominated strong scaling (fork only): 1 vs 4 swarm_workers over
   trap-family branches heavy enough that solving dominates coordination,
   with adaptive decomposition ON and a seeded cost model — the production
   configuration.  Asserts a real speedup; documented on
   TestSolveDominatedStrongScaling.
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
from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH
from wordle_engine import ResponseCache, min_expected_guesses, ERD_ALL
import erd_swarm
from erd_swarm import _BranchWorker, ROOT_BUDGET
from erd_queue import ERDQueue, encode_subset
from wordle_engine import (BranchFloorTable, ResponseCache,
                           sub_branch_cost_lower_bound)
from tests.trap_workloads import (build_trap_branches, build_candidates,
                                  seed_cost_model)

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
            # Pure-partition path: adaptive publishing cannot re-route
            # candidates through additional cooperative claims.
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
                # Exact lower-bound proofs complete candidates without an
                # evaluation; cooperation must never evaluate more than the
                # complete candidate list.
                self.assertLessEqual(
                    total_evaluated, len(CANDIDATES),
                    f"{nw} workers evaluated {total_evaluated} candidates, "
                    f"more than the {len(CANDIDATES)} available "
                    f"(work amplification!)")


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
    # PURPOSE: drain smoke — verify that N cooperative swarm workers drain a
    # queue of MANY SMALL branches faster than 1 worker.  This guards against
    # total serialization (a global lock, a serialized queue); it cannot
    # certify that parallelism is worth running.  Strong scaling on a
    # solve-dominated workload is TestSolveDominatedStrongScaling; overhead vs
    # the plain serial engine is tests/test_swarm_vs_engine_overhead.py.
    #
    # The branch COUNT is what this test contributes: 80 claims against the
    # strong-scaling guard's 4, so a queue that serializes claim handout fails
    # here and passes there.  What it cannot isolate is coordination cost on
    # its own.  That would need the per-claim cost — ~1.5ms, so ~0.12s across
    # the whole queue — to stand clear of the ~1.9s of process spawn and
    # worker startup, and no fixture the answer list can build gets near that.
    # Branches therefore carry enough solve work to be measurable at all, and
    # the number below is a drain speedup, not a coordination floor.
    #
    # DO NOT replace the timing assertion with a pure correctness check.
    # Correctness is covered by TestProcessScalingSmoke (multi-process ERD
    # agreement) and TestWorkDoesNotAmplify (no candidate duplication).
    #
    # Worker count is min(4, cpu_count) so the comparison is always honest:
    # N workers on N CPUs should each get a full core, giving near-linear
    # speedup.  On a 2-CPU CI runner this runs 2 workers vs 1; on Rocky it
    # runs 4 workers vs 1.  The 90% threshold is achievable on any of these.
    # -------------------------------------------------------------------------

    # Branch size sets how much work there is to share out.  Wall time here
    # splits into a fixed cost that no worker count changes — process spawn
    # and worker startup, measured at ~1.9s — and the drain itself, which is
    # the only part a second worker can help with.  The ratio below is
    # therefore a measurement of the drain against that fixed cost, and it
    # goes UNMEASURABLE as the drain shrinks: at 12 words the engine solves a
    # branch so fast that 0.25s of drain sits under 1.9s of startup and no
    # worker count can move the total by 10%.  Solve cost climbs steeply with
    # branch size (20 words: 4.7s at one worker; 24 words: 16.6s), so 24 puts
    # the drain well clear of startup — measured 0.53, against a 0.90 bar.
    #
    # A future speedup will erode this the same way.  The response is to
    # restore the margin by giving the fixture more to do, never to relax
    # _SPEEDUP_RATIO: the bar is what distinguishes a working queue from a
    # serialized one, and a fixture too small to clear it is the thing that
    # is wrong.
    _BRANCH_SIZE = 24
    _N_BRANCHES = 80         # 80 × 24 = 1,920 unique answer words
    _N_CANDIDATES = 100
    # N workers must complete in < 90% of 1-worker time.  A serialized queue
    # measures ~1.0 and fails; the fixture above measures ~0.53 on an idle
    # 8-core box.  The bar stays at 0.90 rather than tightening toward what
    # the fixture achieves, because it is calibrated against the failure it
    # names — serialization — and not against the current fixture's margin.
    _SPEEDUP_RATIO = 0.90

    def setUp(self):
        # Use /dev/shm (RAM-backed tmpfs) when available so SQLite I/O doesn't
        # serialize workers and mask the parallelism signal.  Falls back to a
        # normal tempdir on macOS or systems without /dev/shm.
        shm = '/dev/shm'
        tmp_dir = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) else None
        self._tmp = tempfile.TemporaryDirectory(dir=tmp_dir)
        self.addCleanup(self._tmp.cleanup)
        # A frozen word sample (tests/fixtures/drain_smoke/), not the live
        # lists: this measures the coordination floor, which is a constant, so
        # the workload must stay identical across vocabulary changes rather
        # than resampling by list position. Regenerate the fixture only to
        # deliberately refresh the sample, never on a routine list update.
        fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "fixtures", "drain_smoke")
        with open(os.path.join(fixture_dir, "pool.txt")) as f:
            pool = [l.strip() for l in f if l.strip()]
        with open(os.path.join(fixture_dir, "candidates.txt")) as f:
            candidates = [l.strip() for l in f if l.strip()]
        need_pool = self._N_BRANCHES * self._BRANCH_SIZE
        assert len(pool) >= need_pool and len(candidates) >= self._N_CANDIDATES, (
            "drain_smoke fixture is smaller than the test's parameters; "
            "regenerate tests/fixtures/drain_smoke/")
        self._pool = pool[:need_pool]
        self._branches = [self._pool[i * self._BRANCH_SIZE:(i + 1) * self._BRANCH_SIZE]
                          for i in range(self._N_BRANCHES)]
        self._candidates = candidates[:self._N_CANDIDATES]
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
            # A leg that hits the deadline must fail the test, never feed the
            # ratio: a timed-out 1-worker leg would INFLATE the measured
            # speedup.
            drained = not q.branches_in_progress()
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
            'drained': drained,
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
            f'## Coordination-floor drain smoke — {status}',
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
        coverage number in its own step. Path is $COORDINATION_FLOOR_RESULT_PATH
        (default coordination_floor_result.json in the cwd)."""
        path = os.environ.get('COORDINATION_FLOOR_RESULT_PATH',
                              'coordination_floor_result.json')
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
        # A speedup assertion is only meaningful on a box that can actually
        # run the workers in parallel. CI (the CI env var, set by GitHub
        # Actions) always runs it — catching algorithmic scalability
        # regressions is this test's purpose — but on a shared dev box an
        # external load (e.g. the production swarm) turns the assertion into
        # a load measurement, so it skips with a reason instead of failing.
        # Load average is a trailing signal, which suits the loads that
        # matter here: a swarm that runs for hours, not a momentary spike.
        if not os.environ.get('CI'):
            idle_cores = (os.cpu_count() or 1) - os.getloadavg()[0]
            if idle_cores < n + 1:
                self.skipTest(
                    f'insufficient idle cores for a parallel speedup '
                    f'measurement ({idle_cores:.1f} idle of '
                    f'{os.cpu_count()}, need {n + 1}); set CI=1 to force')
        r1 = self._drain(1, "w1")
        rN = self._drain(n, f"w{n}")
        for r, tag in [(r1, '1-worker'), (rN, f'{n}-worker')]:
            self.assertTrue(r['drained'],
                            f"{tag} leg did not drain within its deadline; "
                            f"timing is meaningless")
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
        # Every branch must have resolved to a best row or a proven loss; a
        # branch with neither is a genuine gap the drain left behind. A loss is
        # a valid terminal result (a branch unsolvable within budget from this
        # test's deliberately small candidate set), not a missing entry.
        sc = ScoreCache(rN['cache'], self._pool)
        unresolved = [bw for bw in self._branches
                      if sc.read(encode_subset(bw), ERD_ALL) is None
                      and sc.read_loss(encode_subset(bw), ERD_ALL) is None]
        sc.close()
        self.assertEqual(unresolved, [],
                         f"{len(unresolved)} branches unresolved "
                         f"(neither best row nor proven loss)")


@unittest.skipUnless(SCALING_SMOKE_REQS_MET, SCALING_SMOKE_SKIP_REASON)
class TestSolveDominatedStrongScaling(unittest.TestCase):
    # -------------------------------------------------------------------------
    # PURPOSE: strong-scaling guard on a workload where SOLVING dominates
    # coordination — the regime the swarm exists for.  Trap-family branches
    # (see tests/trap_workloads.py) are heavy enough that per-claim overhead
    # is a small fraction of wall time, adaptive decomposition is ON, and the
    # cost model is seeded so entry-gate promotion fires as it does in
    # production.  If coordination overhead grows to dominate solve time —
    # the epoch-4 disease (tiny-claim storms, dropped ceilings) — the speedup
    # collapses toward the coordination floor (~1.25x) and this fails.
    #
    # The complementary guard is tests/test_swarm_vs_engine_overhead.py: this
    # test pins the SLOPE (more workers help), that one pins the CONSTANT
    # FACTOR (the swarm is not paying a large tax over the plain engine).
    # A uniform tax at every worker count passes here and fails there;
    # serialized workers pass there and fail here.
    # -------------------------------------------------------------------------

    # Workload shape is load-bearing, not arbitrary.
    #
    # Branch COUNT is the balance lever: with fewer branches than workers, a
    # worker has nothing to claim until decomposition publishes a sub-claim,
    # so the fixture caps its own parallelism before any engine behaviour is
    # measured.  One branch per worker is the floor, hence 4.
    #
    # Branch SIZE is the work lever, and it is what keeps solving dominant.
    # Every worker pays a fixed startup — pattern matrix, cache and queue open
    # — that does not shrink with worker count, so a short drain measures
    # startup rather than scaling.  At this size the 1-worker leg runs ~90s
    # against a startup of ~14s.  Do not shrink the workload to speed the
    # suite up: it lowers the measured ratio without any regression behind it.
    #
    # Work is neither linear nor monotone in either lever, because the
    # candidate vocabulary is built from the branches (build_candidates): more
    # or wider branches mean more guess words, which find a strong guess
    # sooner and prune harder.  Re-measure after changing either number rather
    # than extrapolating.
    _N_BRANCHES = 4
    _BRANCH_SIZE = 80
    _BASE_CANDIDATES = 150
    _BUDGET = 4
    # Required t1/tN speedup by worker count.  These sit far above the ~1.25x
    # coordination floor, so a regression into coordination-bound behaviour
    # fails unambiguously.  Do NOT lower these to make a regressed run pass —
    # that buries the signal this guard exists to raise.
    _MIN_SPEEDUP = {2: 1.3, 3: 1.5, 4: 1.7}

    def setUp(self):
        shm = '/dev/shm'
        tmp_dir = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) else None
        self._tmp = tempfile.TemporaryDirectory(dir=tmp_dir)
        self.addCleanup(self._tmp.cleanup)
        with open(DEFAULT_ANSWER_LIST_PATH) as f:
            answers = [l.strip() for l in f if l.strip()]
        with open(DEFAULT_CANDIDATE_LIST_PATH) as f:
            guesses = [l.strip() for l in f if l.strip()]
        self._branches = build_trap_branches(answers, self._N_BRANCHES,
                                             self._BRANCH_SIZE)
        self._pool = [w for branch in self._branches for w in branch]
        self._candidates = build_candidates(guesses, self._branches,
                                            self._BASE_CANDIDATES)
        answer_file = os.path.join(self._tmp.name, "answers.txt")
        words_file = os.path.join(self._tmp.name, "words.txt")
        with open(answer_file, "w") as f:
            f.write("\n".join(self._pool) + "\n")
        with open(words_file, "w") as f:
            f.write("\n".join(self._candidates) + "\n")
        for attr, path in [("ANSWER_FILE", answer_file),
                           ("WORDS_FILE", words_file)]:
            patcher = mock.patch.object(erd_swarm, attr, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _drain(self, n_workers, tag, timeout=600):
        from erd_swarm import swarm_worker
        cache_path = os.path.join(self._tmp.name, f"cache_{tag}.sqlite3")
        queue_path = os.path.join(self._tmp.name, f"queue_{tag}.sqlite3")
        ScoreCache(cache_path, self._pool).close()
        seed_cost_model(queue_path)
        q = ERDQueue(queue_path)
        for branch_words in self._branches:
            q.create_branch(encode_subset(branch_words), self._BRANCH_SIZE,
                            len(self._candidates), budget=self._BUDGET)
        q.close()

        stop_event = mp.Event()
        with mock.patch("erd_swarm._setup_logging", lambda *_: None):
            procs = [mp.Process(target=swarm_worker,
                                args=(w, cache_path, queue_path, stop_event,
                                      n_workers, True))
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
                time.sleep(0.1)
            # A leg that hits the deadline must fail the test, never feed the
            # ratio: a timed-out 1-worker leg would INFLATE the measured
            # speedup.
            drained = not q.branches_in_progress()
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
        return {'drain': t_drain, 'drained': drained,
                'workers': n_workers, 'cache': cache_path}

    @staticmethod
    def _publish(result):
        path = os.environ.get('STRONG_SCALING_RESULT_PATH',
                              'strong_scaling_result.json')
        with open(path, 'w') as f:
            json.dump(result, f)
        summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_path:
            status = '✅ PASSED' if result['passed'] else '❌ FAILED'
            with open(summary_path, 'a') as f:
                f.write(
                    f"## Solve-dominated strong scaling — {status}\n\n"
                    f"**{result['workers']} workers vs 1** | "
                    f"speedup: **{result['speedup']:.2f}x** "
                    f"(min {result['min_speedup']:.2f}x) | "
                    f"1-worker: {result['t1_drain']:.3f}s | "
                    f"{result['workers']}-worker: {result['tN_drain']:.3f}s | "
                    f"cpu_count: {result['cpu_count']}\n")

    def test_workload_exercises_the_branch_floor(self):
        """Fixture precondition for the scaling guard below.

        Floor construction is per-process work, so it is a place a change
        could quietly flatten multi-worker scaling.  This pins that the
        workload actually builds floors, rather than the speedup assertion
        passing because the machinery never ran.
        """
        table = BranchFloorTable(self._candidates,
                                 cache=ResponseCache(self._pool))
        multi_word_groups = 0
        for branch in self._branches:
            for candidate in self._candidates[:40]:
                for group in table._cache.group_words(candidate, branch).values():
                    if len(group) > 1:
                        sub_branch_cost_lower_bound(group, candidate, table)
                        multi_word_groups += 1
        self.assertGreater(multi_word_groups, 0,
                           "every response group is a singleton: this "
                           "workload cannot exercise the branch floor")
        self.assertGreater(table.misses, 0, "no floor was actually built")

    def test_workers_scale_when_solving_dominates_coordination(self):
        n = min(4, os.cpu_count() or 1)
        if n < 2:
            self.skipTest("needs >=2 CPUs for a speedup measurement")
        # Same idle-core guard as the drain smoke: on a loaded shared box the
        # assertion measures the external load, not the swarm.
        if not os.environ.get('CI'):
            idle_cores = (os.cpu_count() or 1) - os.getloadavg()[0]
            if idle_cores < n + 1:
                self.skipTest(
                    f'insufficient idle cores for a parallel speedup '
                    f'measurement ({idle_cores:.1f} idle of '
                    f'{os.cpu_count()}, need {n + 1}); set CI=1 to force')

        r1 = self._drain(1, "strong1")
        rN = self._drain(n, f"strong{n}")
        for r, tag in [(r1, '1-worker'), (rN, f'{n}-worker')]:
            self.assertTrue(r['drained'],
                            f"{tag} leg did not drain within its deadline; "
                            f"timing is meaningless")
        t1, tN = r1['drain'], rN['drain']
        speedup = t1 / tN if tN > 0 else float('inf')
        min_speedup = self._MIN_SPEEDUP[n]
        passed = speedup >= min_speedup

        self._publish({
            'workers': n,
            't1_drain': t1,
            'tN_drain': tN,
            'speedup': speedup,
            'min_speedup': min_speedup,
            'passed': passed,
            'cpu_count': os.cpu_count(),
            'n_branches': self._N_BRANCHES,
            'branch_size': self._BRANCH_SIZE,
            'n_candidates': len(self._candidates),
            'budget': self._BUDGET,
        })
        sys.stderr.write(
            f"\n[strong-scaling] cpu_count={os.cpu_count()}  "
            f"1-worker: {t1:.3f}s  {n}-worker: {tN:.3f}s  "
            f"speedup {speedup:.2f}x (min {min_speedup:.2f}x)\n")

        # Every root branch, in BOTH legs, must have resolved to a best row
        # or a proven loss; a drain that ended with work outstanding must not
        # pass on a fast-but-wrong technicality.
        for r, tag in [(r1, '1-worker'), (rN, f'{n}-worker')]:
            sc = ScoreCache(r['cache'], self._pool, checkpoint_on_close=False)
            unresolved = [
                bw for bw in self._branches
                if sc.read(encode_subset(bw), ERD_ALL) is None
                and sc.read_loss(encode_subset(bw), ERD_ALL) is None]
            sc.close()
            self.assertEqual(unresolved, [],
                             f"{len(unresolved)} branches never resolved "
                             f"in the {tag} leg")
        self.assertGreaterEqual(
            speedup, min_speedup,
            f"{n} workers achieved only {speedup:.2f}x over 1 worker "
            f"({tN:.3f}s vs {t1:.3f}s); expected >= {min_speedup:.2f}x on a "
            f"solve-dominated workload — coordination is eating the "
            f"parallelism")


if __name__ == "__main__":
    unittest.main()
