"""Swarm-overhead guard: one swarm worker vs the direct engine, same branch.

The swarm exists to beat the single-core engine.  Its characteristic failure
mode is not wrong answers but waste, in two classes:

1. Redundant compute — e.g. a dropped alpha-beta ceiling exact-solving
   promoted sub-branches nobody needed exactly.  Visible in node counts and
   wall clock.
2. Coordination tax — claim/finalize round-trips, SQLite serialization,
   polling.  Zero extra engine work; visible ONLY in wall clock.

The asserted metric is therefore wall clock: every waste mechanism, present
or future, must eventually surface there.  Node and coordination figures are
recorded alongside as diagnostics so a failure classifies itself — swarm
node count exploded means a pruning/ceiling regression (class 1); nodes flat
with time exploded means coordination tax (class 2).

The threshold is a generous multiple of a healthy measured ratio, not a tight
percentage: the regressions this guards against are measured in multiples
(the epoch-4 promoted-trap grind was ~80x the direct engine), so noise
headroom costs nothing in detection power.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

from cache_sqlite import ScoreCache
from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH
from wordle_engine import ResponseCache, min_expected_guesses, ERD_ALL
import pattern_matrix as pattern_matrix_module
import erd_swarm
from erd_swarm import _BranchWorker
from erd_queue import ERDQueue, encode_subset
from tests.trap_workloads import (build_trap_branches, build_candidates,
                                  seed_cost_model)


class TestSwarmVsDirectEngineOverhead(unittest.TestCase):
    _BRANCH_SIZE = 25
    _BASE_CANDIDATES = 200
    _BUDGET = 4
    # Max allowed (1-worker swarm wall) / (direct engine wall).  Healthy ratio
    # measured ~2.2x on an idle 8-core box (the swarm solves FEWER nodes than
    # the engine thanks to cooperative caching, but pays ~24k claims of
    # coordination on this branch); the failure this catches starts at ~10x,
    # so 5x holds wide noise margins on both sides.
    _MAX_OVERHEAD_FACTOR = 5.0

    def setUp(self):
        shm = '/dev/shm'
        tmp_dir = shm if (os.path.isdir(shm) and os.access(shm, os.W_OK)) else None
        self._tmp = tempfile.TemporaryDirectory(dir=tmp_dir)
        self.addCleanup(self._tmp.cleanup)
        with open(DEFAULT_ANSWER_LIST_PATH) as f:
            answers = [l.strip() for l in f if l.strip()]
        with open(DEFAULT_CANDIDATE_LIST_PATH) as f:
            guesses = [l.strip() for l in f if l.strip()]
        [self._branch] = build_trap_branches(answers, 1, self._BRANCH_SIZE)
        self._candidates = build_candidates(guesses, [self._branch],
                                            self._BASE_CANDIDATES)
        answer_file = os.path.join(self._tmp.name, "answers.txt")
        words_file = os.path.join(self._tmp.name, "words.txt")
        with open(answer_file, "w") as f:
            f.write("\n".join(self._branch) + "\n")
        with open(words_file, "w") as f:
            f.write("\n".join(self._candidates) + "\n")
        for attr, path in [("ANSWER_FILE", answer_file),
                           ("WORDS_FILE", words_file)]:
            patcher = mock.patch.object(erd_swarm, attr, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self._branch_key = encode_subset(self._branch)

    def _direct_engine_leg(self):
        """Solve the branch with a plain engine call: fresh caches, pattern
        matrix on, no swarm machinery.  This is the serial baseline the
        swarm's overhead is measured against."""
        cache_path = os.path.join(self._tmp.name, "cache_engine.sqlite3")
        score_cache = ScoreCache(cache_path, self._branch)
        response_cache = ResponseCache(self._branch, score_cache)
        matrix = pattern_matrix_module.PatternMatrix.load_or_build(
            cache_path, self._candidates, self._branch, score_cache)
        nodes = [0]

        def heartbeat():
            nodes[0] += 1

        t0 = time.time()
        erd = min_expected_guesses(self._branch, response_cache, score_cache,
                                   guesses=self._candidates, policy=ERD_ALL,
                                   budget=self._BUDGET, heartbeat=heartbeat,
                                   pattern_matrix=matrix)
        wall = time.time() - t0
        score_cache.close()
        return wall, erd, nodes[0]

    def _swarm_leg(self):
        """Solve the same branch through the full swarm path: queue claims,
        adaptive decomposition on, cost model seeded so entry-gate promotion
        fires as it would mid-production.  Worker construction (word lists,
        pattern matrix) is excluded from the timing; the engine leg's setup
        is likewise outside its window."""
        cache_path = os.path.join(self._tmp.name, "cache_swarm.sqlite3")
        queue_path = os.path.join(self._tmp.name, "queue_swarm.sqlite3")
        ScoreCache(cache_path, self._branch).close()
        seed_cost_model(queue_path)
        queue = ERDQueue(queue_path)
        queue.create_branch(self._branch_key, len(self._branch),
                            len(self._candidates), budget=self._BUDGET)
        queue.close()

        worker = _BranchWorker(0, cache_path, queue_path, None,
                               enable_adaptive_decomposition=True)
        try:
            t0 = time.time()
            worker.solve_branch_focused(self._branch_key)
            wall = time.time() - t0
            nodes = worker._nodes
        finally:
            worker.close()

        score_cache = ScoreCache(cache_path, self._branch,
                                 checkpoint_on_close=False)
        # A solve whose floor fired records its result under the branch's
        # budget rather than as the unrestricted optimum, so read it back the
        # way a search at that budget would.
        row = score_cache.read_for_budget(self._branch_key, ERD_ALL, self._BUDGET)
        score_cache.close()
        erd = row[1] if row is not None else None

        queue = ERDQueue(queue_path)
        claims, coordination_millis = queue._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(coordination_millis), 0) "
            "FROM telemetry.claim_telemetry").fetchone()
        queue.close()
        return wall, erd, nodes, claims, coordination_millis

    @staticmethod
    def _publish(result):
        path = os.environ.get('SWARM_OVERHEAD_RESULT_PATH',
                              'swarm_overhead_result.json')
        with open(path, 'w') as f:
            json.dump(result, f)
        summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_path:
            status = '✅ PASSED' if result['passed'] else '❌ FAILED'
            with open(summary_path, 'a') as f:
                f.write(
                    f"## Swarm vs direct engine overhead — {status}\n\n"
                    f"overhead: **{result['overhead_ratio']:.2f}x** "
                    f"(max {result['max_overhead_factor']:.1f}x) | "
                    f"engine: {result['engine_wall']:.3f}s | "
                    f"swarm: {result['swarm_wall']:.3f}s\n\n"
                    f"engine nodes: {result['engine_nodes']} | "
                    f"swarm nodes: {result['swarm_nodes']} | "
                    f"claims: {result['claim_count']} | "
                    f"coordination: {result['coordination_millis']}ms\n")

    def test_one_worker_swarm_within_overhead_factor_of_direct_engine(self):
        engine_wall, engine_erd, engine_nodes = self._direct_engine_leg()
        self.assertIsNotNone(engine_erd, "baseline branch became unsolvable; "
                             "the workload needs recalibrating")
        swarm_wall, swarm_erd, swarm_nodes, claims, coordination_millis = \
            self._swarm_leg()

        ratio = swarm_wall / engine_wall
        passed = ratio <= self._MAX_OVERHEAD_FACTOR
        self._publish({
            'engine_wall': engine_wall,
            'swarm_wall': swarm_wall,
            'overhead_ratio': ratio,
            'max_overhead_factor': self._MAX_OVERHEAD_FACTOR,
            'passed': passed,
            'engine_nodes': engine_nodes,
            'swarm_nodes': swarm_nodes,
            'claim_count': claims,
            'coordination_millis': coordination_millis,
            'cpu_count': os.cpu_count(),
        })
        sys.stderr.write(
            f"\n[overhead] engine {engine_wall:.3f}s/{engine_nodes} nodes  "
            f"swarm {swarm_wall:.3f}s/{swarm_nodes} nodes  "
            f"ratio {ratio:.2f}x  claims={claims}  "
            f"coordination={coordination_millis}ms\n")

        # Both paths must agree on the exact optimum before timing means
        # anything.
        self.assertIsNotNone(swarm_erd, "swarm leg produced no cached result")
        self.assertAlmostEqual(swarm_erd, engine_erd, places=6)
        # The claim count doubles as proof the adaptive layer actually
        # engaged: without promotion the whole branch is a handful of
        # root-level claims.
        self.assertGreater(claims, 1, "no cooperative claims recorded — "
                           "promotion never fired, the guard tested nothing")
        self.assertLessEqual(
            ratio, self._MAX_OVERHEAD_FACTOR,
            f"swarm took {ratio:.2f}x the direct engine "
            f"({swarm_wall:.3f}s vs {engine_wall:.3f}s); "
            f"swarm_nodes={swarm_nodes} vs engine_nodes={engine_nodes} "
            f"(node explosion = pruning/ceiling regression; flat nodes = "
            f"coordination tax), coordination={coordination_millis}ms")


if __name__ == "__main__":
    unittest.main()
