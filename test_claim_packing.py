"""Unit/integration tests for binary claim packing (issue #67,
adaptive_claim_packing.md): the exact-elimination packer (_pack_bundle),
ERDQueue.claim_next_bundle, republish-on-overrun, and the erd_swarm.py
integration (evaluate_bundle, _BranchWorker._claim_bundle/_packing_stats).

See test_erd_scaling.py for the pre-existing scaling/correctness guards
(TestCooperativeDrainSmoke, TestWorkDoesNotAmplify, TestProcessScalingSmoke)
and test_claim_packing_measurement.py for the measurement-layer tests that
predate the packer.  This file covers the packer itself.
"""
import inspect
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock

import erd_queue
import erd_swarm
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue, encode_subset, _pack_bundle
from erd_swarm import _BranchWorker, ROOT_BUDGET
from wordle_engine import ResponseCache, min_expected_guesses, ERD_ALL


class TestPackBundle(unittest.TestCase):
    """_pack_bundle: the exact-elimination packer (adaptive_claim_packing.md §5)."""

    def test_bulk_bundle_coalesces_consecutive_eliminated(self):
        order = list(range(10))
        cost_lower_bound = [5.0] * 10   # all >= bound: provably eliminated
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound, 1.0,
                                        small_count=2, count_cap=4)
        self.assertEqual(bundle, [0, 1, 2, 3])
        self.assertEqual(next_pos, 4)

    def test_bulk_bundle_stops_at_first_survivor(self):
        order = list(range(10))
        cost_lower_bound = [5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound, 1.0,
                                        small_count=2, count_cap=100)
        self.assertEqual(bundle, [0, 1, 2])
        self.assertEqual(next_pos, 3)

    def test_small_bundle_absorbs_interleaved_eliminated_candidates(self):
        # Survivors at 0, 2, 4; eliminated candidates at 1, 3 ride along free.
        order = list(range(6))
        cost_lower_bound = [0.0, 5.0, 0.0, 5.0, 0.0, 0.0]
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound, 1.0,
                                        small_count=3, count_cap=100)
        self.assertEqual(bundle, [0, 1, 2, 3, 4])   # stops once 3 survivors taken
        self.assertEqual(next_pos, 5)

    def test_count_cap_bounds_small_bundle(self):
        order = list(range(10))
        cost_lower_bound = [0.0] * 10   # nothing eliminated
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound, 1.0,
                                        small_count=8, count_cap=3)
        self.assertEqual(len(bundle), 3)
        self.assertEqual(next_pos, 3)

    def test_count_cap_bounds_bulk_bundle(self):
        order = list(range(10))
        cost_lower_bound = [5.0] * 10
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound, 1.0,
                                        small_count=2, count_cap=3)
        self.assertEqual(len(bundle), 3)
        self.assertEqual(next_pos, 3)

    def test_start_past_end_returns_empty(self):
        bundle, next_pos = _pack_bundle([0, 1, 2], 3, [0.0, 0.0, 0.0], 1.0, 2, 5)
        self.assertEqual(bundle, [])
        self.assertEqual(next_pos, 3)

    def test_loose_bound_eliminates_nothing(self):
        # An unset best_erd is passed as +inf: cost_lower_bound is always
        # finite (<= 3.0), so `>= inf` is never true — every candidate packs
        # as a survivor until small_count is reached.
        order = list(range(5))
        cost_lower_bound = [2.9] * 5
        bundle, next_pos = _pack_bundle(order, 0, cost_lower_bound,
                                        float('inf'), small_count=2, count_cap=100)
        self.assertEqual(bundle, [0, 1])
        self.assertEqual(next_pos, 2)


N_CANDIDATES = 40
_ORDER = list(range(N_CANDIDATES))
_ZERO_LOWER_BOUND = [0.0] * N_CANDIDATES


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.path)
        self.addCleanup(self.q.close)
        self.key = encode_subset(["crane", "slate", "trace", "stale", "tales"])
        self.q.create_branch(self.key, 5, N_CANDIDATES)


class TestClaimNextBundle(_TmpQueue):
    def test_loose_bound_packs_small_bundles(self):
        # No best_erd set yet -> B reads as +inf -> nothing eliminated.
        bundle_id, indices, forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.9] * N_CANDIDATES,
            small_count=5, count_cap=500)
        self.assertEqual(len(indices), 5)
        self.assertEqual(forced, frozenset())

    def test_tight_bound_packs_the_whole_eliminated_tail_in_one_bulk_bundle(self):
        self.q.update_branch_best(self.key, "salet", 1.0)
        bundle_id, indices, forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES,
            small_count=5, count_cap=500)
        self.assertEqual(len(indices), N_CANDIDATES)   # one bundle, whole branch

    def test_returns_none_for_finalized_branch(self):
        self.q.try_finalize_branch(self.key)
        claim = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND)
        self.assertIsNone(claim)

    def test_returns_none_once_fully_claimed(self):
        while True:
            claim = self.q.claim_next_bundle(
                self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=10, count_cap=10)
            if claim is None:
                break
        self.assertIsNone(self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND))

    def test_bundle_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.q.claim_next_bundle(self.key, "w0", N_CANDIDATES, [0, 1, 2],
                                     _ZERO_LOWER_BOUND)

    def test_counts_real_busy_retries_under_write_lock_contention(self):
        # Hold the write lock from a second connection just long enough that
        # claim_next_bundle's short per-attempt busy_timeout
        # (_BUNDLE_CLAIM_RETRY_MILLIS) must fail and retry at least once
        # before the lock is released.
        # check_same_thread=False: released from the timer thread below, not
        # the thread that opened it — sqlite3 forbids that by default.
        blocker = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        blocker.execute("BEGIN IMMEDIATE")
        blocker.execute("SELECT 1")

        def release():
            time.sleep(0.25)
            blocker.commit()
            blocker.close()
        t = threading.Thread(target=release)
        t.start()
        try:
            claim = self.q.claim_next_bundle(
                self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=5, count_cap=5)
        finally:
            t.join(timeout=5)
        self.assertIsNotNone(claim)
        self.assertGreater(self.q._last_claim_retries, 0)
        self.assertGreater(self.q._last_claim_busy_millis, 0)

    def test_no_two_calls_return_overlapping_indices(self):
        seen = set()
        while True:
            claim = self.q.claim_next_bundle(
                self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=6, count_cap=6)
            if claim is None:
                break
            _bundle_id, indices, _forced = claim
            self.assertTrue(seen.isdisjoint(indices), "overlapping bundle claim")
            seen.update(indices)
        self.assertEqual(seen, set(range(N_CANDIDATES)))


class TestRepublishRemainder(_TmpQueue):
    def test_republish_deletes_done0_rows_and_bumps_count(self):
        _bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        counts = self.q.republish_remainder(self.key, indices)
        self.assertEqual(set(counts), set(indices))
        self.assertTrue(all(c == 1 for c in counts.values()))
        rows = self.q.claims_for_branch(self.key)
        self.assertEqual([r for r in rows if r["idx"] in indices], [])

    def test_republished_candidates_are_reclaimable_via_holes_pass(self):
        _bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=N_CANDIDATES, count_cap=N_CANDIDATES)
        self.assertEqual(len(indices), N_CANDIDATES)   # cursor now == N_CANDIDATES
        self.q.republish_remainder(self.key, indices[:3])
        _bundle_id, reclaimed, _forced = self.q.claim_next_bundle(
            self.key, "w1", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        self.assertEqual(set(reclaimed), set(indices[:3]))

    def test_forced_set_after_republish_limit_reached(self):
        # Drain the whole order in one bundle so pack_cursor reaches
        # N_CANDIDATES; every further claim then goes through holes_pass,
        # which re-surfaces a republished subset as the same indices each
        # time (the forward path would instead hand out fresh, never-seen
        # positions and never re-visit the republished ones).
        self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=N_CANDIDATES, count_cap=N_CANDIDATES)
        target = [3, 4, 5]
        forced = frozenset()
        for i in range(3):
            self.q.republish_remainder(self.key, target)
            _bundle_id, indices, forced = self.q.claim_next_bundle(
                self.key, f"w{i + 1}", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=5, count_cap=5, republish_limit=3)
            self.assertEqual(set(indices), set(target))
        self.assertEqual(forced, frozenset(target))

    def test_republish_of_empty_indices_is_noop(self):
        self.assertEqual(self.q.republish_remainder(self.key, []), {})


class TestBundleStatsAndFinalizeLog(_TmpQueue):
    def test_finalize_bundle_stats_aggregates_and_clears(self):
        self.q.record_bundle_stats(self.key, "b1", nodes=10, coord_millis=5)
        self.q.record_bundle_stats(self.key, "b2", nodes=40, coord_millis=7,
                                   censored=True)
        n_bundles, max_bundle_nodes, total_coord_millis, censored_units = (
            self.q.finalize_bundle_stats(self.key))
        self.assertEqual(n_bundles, 2)
        self.assertEqual(max_bundle_nodes, 40)
        self.assertEqual(total_coord_millis, 12)
        self.assertEqual(censored_units, 1)
        # Cleared: a second call sees nothing.
        self.assertEqual(self.q.finalize_bundle_stats(self.key),
                         (None, None, None, None))

    def test_finalize_bundle_stats_empty_when_branch_never_claimed_a_bundle(self):
        self.assertEqual(self.q.finalize_bundle_stats(self.key),
                         (None, None, None, None))


class TestMultiWorkerNoOverlap(unittest.TestCase):
    """Concurrent claim_next_bundle callers (real separate connections, real
    threads) never pack overlapping bundles — adaptive_claim_packing.md §12's
    correctness constraint."""

    def test_concurrent_claims_never_overlap_or_duplicate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "q.sqlite3")
        n_candidates = 400
        key = encode_subset([f"w{i:04d}" for i in range(50)])
        q0 = ERDQueue(path)
        q0.create_branch(key, 50, n_candidates)
        q0.close()

        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates   # forces small bundles: max contention
        all_indices = []
        lock = threading.Lock()

        def worker():
            q = ERDQueue(path)
            local = []
            try:
                while True:
                    claim = q.claim_next_bundle(
                        key, threading.current_thread().name, n_candidates,
                        order, cost_lower_bound, small_count=4, count_cap=20)
                    if claim is None:
                        break
                    local.append(claim[1])
            finally:
                q.close()
            with lock:
                all_indices.extend(local)

        threads = [threading.Thread(target=worker, name=f"worker-{i}")
                  for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), "a worker hung")

        flat = [idx for bundle in all_indices for idx in bundle]
        self.assertEqual(len(flat), len(set(flat)),
                         "overlapping/duplicated bundle claim across workers")
        self.assertEqual(set(flat), set(range(n_candidates)))


class TestNoWorkEstimateInClaimPath(unittest.TestCase):
    """adaptive_claim_packing.md §11: estimate_candidate_work(_cutoff) must
    appear nowhere in the packer or the claim path — telemetry/analysis only."""

    def test_erd_queue_module_never_imports_wordle_engine(self):
        source = inspect.getsource(erd_queue)
        self.assertNotIn("wordle_engine", source)

    def test_claim_path_functions_never_reference_the_work_estimator(self):
        functions = [
            erd_queue._pack_bundle,
            erd_queue.ERDQueue.claim_next_bundle,
            erd_queue.ERDQueue.republish_remainder,
            _BranchWorker._claim_bundle,
            _BranchWorker._packing_stats,
            _BranchWorker.evaluate_bundle,
        ]
        for fn in functions:
            source = inspect.getsource(fn)
            self.assertNotIn("estimate_candidate_work", source,
                             f"{fn.__qualname__} must never call the work estimator")


# -- integration: erd_swarm.py drains a real branch through the packer --------

BRANCH = ["crane", "slate", "trace", "stale", "tales", "least",
          "heart", "share", "rates", "earth", "brave", "cleat"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train", "grade",
                       "shine", "mount", "frost", "plumb", "dwarf", "gawky"]


class _SwarmBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attr, path in [("ANSWER_FILE", self.answer_file),
                           ("WORDS_FILE", self.words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
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

    def _ground_truth(self):
        cache_path, _ = self._db("truth")
        sc = ScoreCache(cache_path, BRANCH)
        erd = min_expected_guesses(BRANCH, ResponseCache(BRANCH, sc), sc,
                                   guesses=CANDIDATES, policy=ERD_ALL,
                                   budget=ROOT_BUDGET)
        sc.close()
        return erd

    def _solve(self, tag, **worker_kwargs):
        cache_path, queue_path = self._db(tag)
        ScoreCache(cache_path, BRANCH).close()
        q = ERDQueue(queue_path)
        q.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.close()
        w = _BranchWorker(0, cache_path, queue_path, None, **worker_kwargs)
        try:
            w.solve_branch_focused(self.branch_key)
        finally:
            w.close()
        sc = ScoreCache(cache_path, BRANCH, checkpoint_on_close=False)
        result = sc.read_with_depth(self.branch_key, ERD_ALL)
        sc.close()
        return result


class TestPackingEquivalence(_SwarmBase):
    """Bundled claiming reaches the same optimum as single-candidate-equivalent
    claiming (small_count=1, count_cap=1 — every bundle degenerates to one
    candidate): same winning guess, same max_remaining_depth, ERD within
    ±1e-5 (adaptive_claim_packing.md §8/§11 — float summation order differs,
    the optimum does not)."""

    def test_bundled_matches_single_candidate_equivalent(self):
        baseline = self._solve("baseline", small_count=1, count_cap=1)
        bundled = self._solve("bundled", small_count=8, count_cap=500)
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(bundled)
        base_guess, base_erd, base_depth, _ = baseline
        bund_guess, bund_erd, bund_depth, _ = bundled
        self.assertEqual(base_guess, bund_guess)
        self.assertEqual(base_depth, bund_depth)
        self.assertAlmostEqual(base_erd, bund_erd, delta=1e-5)


class TestMidSweepReclaimEquivalence(unittest.TestCase):
    """#68's retained equivalence test: an injected mid-sweep reclaim (a
    worker's bundle is freed as if it crashed) yields identical coverage —
    every candidate ends up done exactly once, none skipped or duplicated."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.path)
        self.addCleanup(self.q.close)
        self.key = encode_subset(["crane", "slate", "trace", "stale", "tales"])
        self.q.create_branch(self.key, 5, N_CANDIDATES)

    def test_injected_mid_sweep_reclaim_yields_full_unique_coverage(self):
        _bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "worker-A", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=6, count_cap=6)
        self.assertTrue(indices)   # worker-A "crashes" — never completes or heartbeats

        # Backdate the claim so it reads as stale regardless of how much wall
        # time actually elapsed during the test (mirrors test_erd_fixes.py's
        # TestReclaimLiveness pattern).
        self.q._conn.execute(
            "UPDATE candidate_claims SET claimed_at = 0 WHERE branch_key = ?",
            (self.key,))
        freed = self.q.reclaim_stale_claims(heartbeat_timeout_seconds=120)
        self.assertEqual(freed, len(indices))

        done = set()
        while True:
            claim = self.q.claim_next_bundle(
                self.key, "worker-B", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=6, count_cap=6)
            if claim is None:
                break
            _bundle_id, claimed_indices, _forced = claim
            self.assertTrue(done.isdisjoint(claimed_indices),
                            "reclaim must never hand out a candidate twice")
            for idx in claimed_indices:
                self.q.complete_candidate(self.key, idx)
                done.add(idx)
        self.assertEqual(done, set(range(N_CANDIDATES)))


class TestStructuralClaimReduction(_SwarmBase):
    """Draining a branch produces far fewer claim transactions than
    candidates, and no code path hands out a lone candidate as a dedicated
    behavior (a size-1 bundle may still occur as an emergent tail outcome)."""

    def test_claim_transactions_far_fewer_than_candidates(self):
        cache_path, queue_path = self._db("structural")
        ScoreCache(cache_path, BRANCH).close()
        q = ERDQueue(queue_path)
        q.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.close()
        w = _BranchWorker(0, cache_path, queue_path, None)
        bundle_sizes = []
        original_claim = w.queue.claim_next_bundle

        def counting_claim(*args, **kwargs):
            result = original_claim(*args, **kwargs)
            if result is not None:
                bundle_sizes.append(len(result[1]))
            return result
        w.queue.claim_next_bundle = counting_claim
        try:
            w.solve_branch_focused(self.branch_key)
        finally:
            w.close()

        n_candidates = len(CANDIDATES)
        n_bundles = len(bundle_sizes)
        self.assertGreater(n_bundles, 0)
        # Concrete reduction factor: this fixture (24 candidates, default
        # small_count=8/count_cap=500) collapses the ERD-pruned tail into a
        # single bulk bundle once the first candidate solves, so the claim
        # count drops well below one-per-candidate.
        self.assertLessEqual(n_bundles, n_candidates // 2,
                             f"{n_bundles} claims for {n_candidates} candidates "
                             f"— packing did not reduce claim count")
        singleton_bundles = sum(1 for s in bundle_sizes if s == 1)
        self.assertLessEqual(
            singleton_bundles, 1,
            "no path should hand out a lone candidate as standard behavior "
            "(a size-1 bundle may occur only as an emergent tail outcome)")


class TestRepublishOnOverrunConverges(_SwarmBase):
    """An aggressive node cap forces republish on nearly every bundle; the
    branch must still converge to the correct optimum (adaptive_claim_
    packing.md §7's bounded-republish-depth guardrail prevents thrashing)."""

    def test_forced_republish_still_reaches_ground_truth(self):
        truth = self._ground_truth()
        result = self._solve("republish", small_count=4, count_cap=4,
                             bundle_node_cap=0, republish_limit=2)
        self.assertIsNotNone(result, "branch did not finalize under forced republish")
        self.assertAlmostEqual(result[1], truth, places=6)


if __name__ == "__main__":
    unittest.main()
