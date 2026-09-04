"""Unit tests for _BranchWorker behaviors not covered by the integration tests.

The integration tests (test_erd_parallel, test_erd_scaling) exercise the swarm
via solve_branch_focused(), which covers the core evaluate/finalize path.  What
they miss:

- _heartbeat(): node-counter throttling (counter increments on every call even
  when the 2-second DB-write gate suppresses the actual write).
- evaluate_claim(): cancellation path (stop_event set → returns False without
  completing the claim).
- _subbranch_solver(): the inline-vs-promote branching decision.
- _maybe_checkpoint(): force=True always triggers checkpoint; force=False with a
  recently-set timestamp does not.
- cooperative_solve(): the cached-result fast path (result already in cache →
  returns immediately without evaluating any candidates).
"""
import math
import multiprocessing
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

import erd_queue
import erd_swarm
from erd_swarm import (_BranchWorker, WorkContext, ROOT_BUDGET,
                       PROMOTE_MIN_SIZE, MAX_WORKERS_PER_BRANCH,
                       decode_subset)
from cache_sqlite import ScoreCache
from wordle_engine import ERD_ALL, SOLVED, OVER_DEPTH_BUDGET, OVER_ERD_LIMIT
from erd_queue import guess_depth_from_spine, SCHEDULING_ROLE_PREFERRED
from tests.queue_invariants import OpenerWorkInvariantCheckMixin

BRANCH = ["crane", "slate", "trace", "stale", "tales"]
CANDIDATES = BRANCH + ["brain", "stove", "cloud", "piano", "train"]


ProductionERDQueue = erd_queue.ERDQueue


class InvariantCheckedERDQueue(OpenerWorkInvariantCheckMixin,
                               ProductionERDQueue):
    pass


ERDQueue = InvariantCheckedERDQueue


def setUpModule():
    erd_queue.ERDQueue = InvariantCheckedERDQueue
    erd_swarm.ERDQueue = InvariantCheckedERDQueue


def tearDownModule():
    erd_queue.ERDQueue = ProductionERDQueue
    erd_swarm.ERDQueue = ProductionERDQueue


def _bare_worker():
    """Skeleton _BranchWorker for unit tests: no DB connections, no word files.

    Sets only the attributes required for the method under test; other tests
    mock queue and cache so the unit stays fast and isolated.
    """
    w = _BranchWorker.__new__(_BranchWorker)
    w.name = "worker-0"
    w.stop_event = None
    w._stop_requested = False
    w.root_budget = ROOT_BUDGET
    w.all_words = CANDIDATES
    w.n_candidates = len(CANDIDATES)
    w.claims_done = 0
    w.n_ok = w.n_cutoff = w.n_pruned = w.n_useless = 0
    w._nodes = 0
    w._nodes_at_last_hb = 0
    w._last_hb = 0.0
    w._last_progress_log = 0.0
    w._last_util_log = 0.0
    w._eval_seconds = 0.0
    w._last_claim_complete = 0.0
    w._last_checkpoint = 0.0
    w._last_wal_traffic = ({}, {})
    w._last_wal_traffic_log = 0.0
    w._checkpoint_interval = erd_swarm.CHECKPOINT_SECONDS
    w._last_disk_check = 0.0
    w._last_pause_check = 0.0
    w._pause_active = False
    w._last_wal_ceiling_check = 0.0
    w._wal_ceiling_hit = False
    w._cand_max_depth = 0
    w._cur_candidate = None
    w._cur_depth = 0
    w._spine = {}
    w._hb_max_spine = {}
    w._log_max_spine = {}
    w.started = 0
    w._work_context = WorkContext.empty()
    w._help_recursion_depth = 0
    w._pending_scheduling_millis = 0
    w._adaptive = True
    w._erd_lower_bound_pruned_accuracy_n = 0
    w._typical_cache = {}
    w._cost_model_buffer = {}
    w._word_idx = {word: i for i, word in enumerate(w.all_words)}
    w._coord_ema = erd_swarm._LogEMA()
    w._node_time_ema = erd_swarm._LogEMA()
    w._mid_loop_publisher = erd_swarm._MidLoopPublisher(w)
    w.n_workers = 1
    w.small_count = erd_swarm.BUNDLE_SMALL_COUNT
    w.count_cap = erd_swarm.BUNDLE_COUNT_CAP
    w.bundle_node_cap = erd_swarm.BUNDLE_NODE_CAP
    w.bundle_wall_cap_seconds = erd_swarm.BUNDLE_WALL_CAP_SECONDS
    w.republish_limit = erd_swarm.BUNDLE_REPUBLISH_LIMIT
    w._packing_stats_cache = {}
    w.rcache = mock.MagicMock()
    # No hint artifact: the default a run without --hint-cache produces.
    w.hint_cache = None
    w.pattern_matrix = None
    w.branch_floor_table = None
    w.score_cache = mock.MagicMock()
    w.score_cache.read_hits = 0
    w.score_cache.read_misses = 0
    w.queue = mock.MagicMock()
    w.queue.read_branch_best.return_value = (None, None, None)
    w.queue.branch_bulk_done_candidates.return_value = 0
    w.queue.branch_erd_pruned_candidate_counts.return_value = (0, 0)
    w.queue.get_cost_typical.return_value = None  # cold model by default
    w.queue.checkpoint_paused.return_value = False
    w.queue.wal_traffic_snapshot.return_value = ({}, {})
    w.queue.wal_size_bytes.return_value = 0
    return w


def _context(branch_key=b"branch", spine=None, opener_work_id=None,
             opener_priority=0, opener=None, opener_pattern=None,
             scheduling_role=None):
    return WorkContext(
        opener_work_id, opener_priority, opener, opener_pattern,
        branch_key, spine, scheduling_role)


class TestHeartbeatThrottling(unittest.TestCase):
    """_nodes increments on every call; the DB write is gated by HB_SECONDS."""

    def test_node_counter_increments_even_when_db_write_is_throttled(self):
        w = _bare_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)

        # First call: force=True bypasses the time gate — DB write happens.
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertEqual(w._nodes, 1)
        self.assertEqual(w.queue.heartbeat.call_count, 1)

        # Second call: force=False, but _last_hb was just set so the gate fires.
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None)
        self.assertEqual(w._nodes, 2)          # counter still incremented
        self.assertEqual(w.queue.heartbeat.call_count, 1)  # still only one DB write

    def test_hb_max_spine_reset_after_each_db_write(self):
        """_hb_max_spine is cleared after each DB write so the 2-second window
        starts fresh — the next heartbeat builds a new spine from scratch."""
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, 12)
        branch_key = ScoreCache.encode_subset(BRANCH)
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertEqual(w._hb_max_spine, {})

    def test_heartbeat_reports_worker_cur_candidate(self):
        """The DB write carries the worker's in-flight candidate — a heartbeat
        reports current state, so a coordination heartbeat (cur_candidate
        cleared) records None even while the worker still holds its branch."""
        w = _bare_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)

        w._cur_candidate = "crane"
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertEqual(
            w.queue.heartbeat.call_args.kwargs["cur_candidate"], "crane")

        w._cur_candidate = None
        w._heartbeat(branch_key, len(BRANCH), 0, 0, None, None, force=True)
        self.assertIsNone(
            w.queue.heartbeat.call_args.kwargs["cur_candidate"])


class TestPromotedSpine(unittest.TestCase):
    """_promoted_spine composes base + descent without overlap.

    The live descent dict keeps shallow entries until a shallower frame
    overwrites them, so after the base is advanced it still holds the guesses
    that reach the base.  Those belong to the base, not the descent; appending
    them re-emits the base's own reaching guess and over-counts guess_depth.
    """

    def test_descent_at_or_above_base_depth_is_not_reappended(self):
        w = _bare_worker()
        # Base reaches a guess_depth-2 branch via SALET then ABORT.
        w._work_context = _context(spine='SALET -g-g- ABORT y----')
        # Live descent still carries the reaching guess (depth 2, budget 4) AND
        # a genuine child below it (depth 3, budget 3).
        w._note_depth(4, 12, 'abort', 'y----')   # guess_depth 2 — the base tail
        w._note_depth(3, 5, 'dogma', 'y---y')     # guess_depth 3 — the real child
        spine = w._promoted_spine()
        self.assertEqual(spine, 'SALET -g-g- ABORT y---- DOGMA y---y')
        self.assertEqual(guess_depth_from_spine(spine), 3)

    def test_different_entry_at_base_depth_preserves_budget_invariant(self):
        w = _bare_worker()
        w._work_context = _context(
            spine='CRANE -yy-y SATED -g-g- DARGS -gy--')
        w._note_depth(3, 5, 'gamps', '-g---')

        spine = w._promoted_spine()

        self.assertEqual(3 + guess_depth_from_spine(spine), ROOT_BUDGET)
        self.assertEqual(spine, 'CRANE -yy-y SATED -g-g- DARGS -gy--')

    def test_stale_entry_at_base_depth_dropped_but_descent_kept(self):
        w = _bare_worker()
        w._work_context = _context(spine='ALIBI y---- EARNT yg---')
        w._note_depth(4, 30, 'story', '-yy-y')
        w._note_depth(3, 12, 'coups', '-----')

        spine = w._promoted_spine()

        self.assertEqual(spine, 'ALIBI y---- EARNT yg--- COUPS -----')
        self.assertEqual(3 + guess_depth_from_spine(spine), ROOT_BUDGET)

    def test_cooperative_entry_spines_match_promotion_budgets(self):
        cases = (
            (5, 'CRANE -yy-y', ((5, 'gamps', '-g---'),)),
            (4, 'CRANE -yy-y', ((4, 'sated', '-g-g-'),)),
            (3, 'CRANE -yy-y SATED -g-g-',
             ((3, 'dargs', '-gy--'),)),
        )
        for budget, base_spine, descent_entries in cases:
            with self.subTest(budget=budget):
                w = _bare_worker()
                w._work_context = _context(spine=base_spine)
                for entry_budget, guess, pattern in descent_entries:
                    w._note_depth(entry_budget, 5, guess, pattern)
                w.score_cache.read_with_depth.return_value = None
                w.score_cache.read_for_budget.return_value = None
                w.score_cache.read_loss.return_value = None
                w.queue.read_cut_result.return_value = []
                w.queue.has_pending_row.return_value = False
                w.queue.create_branch.return_value = True
                w.queue.get_branch.return_value = None

                w.cooperative_solve(BRANCH, budget)

                promoted_spine = w.queue.create_branch.call_args.kwargs['spine']
                self.assertEqual(
                    budget + guess_depth_from_spine(promoted_spine),
                    ROOT_BUDGET)

    def test_nested_cooperative_branch_records_immediate_parent(self):
        w = _bare_worker()
        outer_words = BRANCH
        inner_words = BRANCH[:4]
        outer_key = ScoreCache.encode_subset(outer_words)
        root_key = b"root-branch"
        w._work_context = _context(branch_key=root_key)
        w.score_cache.read_with_depth.return_value = None
        w.score_cache.read_for_budget.return_value = None
        w.score_cache.read_loss.return_value = None
        w.queue.read_cut_result.return_value = []
        w.queue.has_pending_row.return_value = False
        w.queue.create_branch.return_value = True
        w.queue.get_branch.side_effect = (
            lambda branch_key: {} if branch_key == outer_key else None)
        w._claim_bundle = mock.MagicMock(return_value=(1, [0], False))

        def evaluate_outer_branch(*_args, **_kwargs):
            self.assertEqual(w._work_context.branch_key, outer_key)
            w.cooperative_solve(inner_words, ROOT_BUDGET - 1)
            self.assertEqual(w._work_context.branch_key, outer_key)
            w._stop_requested = True
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=evaluate_outer_branch)

        w.cooperative_solve(outer_words, ROOT_BUDGET)

        nested_create = w.queue.create_branch.call_args_list[1]
        self.assertEqual(nested_create.kwargs["parent_branch_key"], outer_key)
        self.assertEqual(w._work_context.branch_key, root_key)

    def test_pure_descent_below_base_is_appended(self):
        w = _bare_worker()
        w._work_context = _context(spine='SALET -g-g-')
        w._note_depth(4, 5, 'dogma', 'y---y')     # guess_depth 2 — below the base
        self.assertEqual(w._promoted_spine(), 'SALET -g-g- DOGMA y---y')

    def test_no_base_yields_none(self):
        w = _bare_worker()
        w._work_context = WorkContext.empty()
        w._note_depth(4, 5, 'dogma', 'y---y')
        self.assertIsNone(w._promoted_spine())


class TestHeartbeatSpineStrFilter(unittest.TestCase):
    """_hb_spine_str omits outer-frame entries at or shallower than the claimed branch.

    _solve_subset fires note_depth before calling subbranch_solver, so the
    entry at the claimed branch's own guess_depth is written by the outer frame
    and persists into inner cooperative_solve sessions.  It belongs to the base
    spine, not to the live descent, and must be filtered out of the heartbeat
    string so the worker detail panel does not display the reaching guess twice.
    """

    def test_outer_frame_entry_at_claimed_depth_is_excluded(self):
        w = _bare_worker()
        # Outer frame set _spine[3] = (44, 'peaze', '-yy--') before calling
        # subbranch_solver, then cooperative_solve advanced _claimed_branch_spine
        # to include PEAZE.  The outer frame's entry should not appear as a live
        # descent step.
        w._work_context = _context(
            spine='SALET -g-g- ABOVE y---y PEAZE -yy--')
        w._note_depth(3, 44, 'peaze', '-yy--')   # outer frame's entry at guess_depth 3
        w._note_depth(2, 16, 'nurdy', '---y-')   # real descent at guess_depth 4
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        self.assertNotIn('PEAZE', result)
        self.assertIn('NURDY', result)

    def test_descent_below_claimed_branch_is_included(self):
        w = _bare_worker()
        w._work_context = _context(spine='SALET -g-g-')
        w._note_depth(4, 30, 'crane', '-yg--')   # guess_depth 2
        w._note_depth(3, 8, 'dogma', 'y---y')    # guess_depth 3
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        self.assertIn('CRANE', result)
        self.assertIn('DOGMA', result)

    def test_tokens_carry_explicit_guess_depth_prefix(self):
        w = _bare_worker()
        w._work_context = _context(spine='SALET -g-g-')
        w._note_depth(4, 30, 'crane', '-yg--')   # guess_depth 2
        w._note_depth(3, 8, 'dogma', 'y---y')    # guess_depth 3
        w._hb_max_spine = dict(w._spine)
        result = w._hb_spine_str()
        # Each token is prefixed 'guess_depth:GUESS:...' so the display can label
        # and filter entries without positional inference.
        self.assertIn('2:CRANE', result)
        self.assertIn('3:DOGMA', result)


class TestCancelPath(unittest.TestCase):
    """evaluate_claim returns False without completing the claim when cancelled."""

    def _make_cancel_worker(self):
        stop = multiprocessing.Event()
        stop.set()
        w = _bare_worker()
        w.stop_event = stop
        return w

    def test_evaluate_claim_returns_false_when_stop_event_set(self):
        w = self._make_cancel_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        result = w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        self.assertFalse(result)

    def test_claim_not_marked_complete_when_cancelled(self):
        w = self._make_cancel_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        # complete_candidate must NOT have been called (claim left done=0 for reclaim).
        w.queue.complete_candidate.assert_not_called()

    def test_request_stop_cancels_without_shared_event(self):
        # A worker's own SIGTERM/SIGINT handler calls request_stop(); this must
        # cancel the worker via a process-local flag, never touching the shared
        # stop_event (so recycling one worker does not stop the pool).
        w = _bare_worker()           # stop_event is None
        self.assertFalse(w.cancel())
        w.request_stop()
        self.assertTrue(w.cancel())
        self.assertIsNone(w.stop_event)

    def test_evaluate_bundle_returns_false_and_records_censored_bundle_on_cancel(self):
        # Cancelled before the first candidate in the bundle is even attempted.
        w = self._make_cancel_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        result = w.evaluate_bundle(branch_key, BRANCH, len(BRANCH), "bundle-1",
                                   [0, 1], frozenset())
        self.assertFalse(result)
        w.queue.record_bundle_stats.assert_called_once_with(
            branch_key, "bundle-1", 0, mock.ANY, censored=True)

    def test_evaluate_bundle_returns_false_when_evaluate_claim_fails_mid_bundle(self):
        # Not cancelled at the loop level, but evaluate_claim itself reports
        # cancellation (or an abort status) partway through the bundle.
        w = _bare_worker()
        w.evaluate_claim = mock.MagicMock(return_value=False)
        branch_key = ScoreCache.encode_subset(BRANCH)
        result = w.evaluate_bundle(branch_key, BRANCH, len(BRANCH), "bundle-2",
                                   [0, 1], frozenset())
        self.assertFalse(result)
        w.evaluate_claim.assert_called_once()
        w.queue.record_bundle_stats.assert_called_once_with(
            branch_key, "bundle-2", 0, mock.ANY, censored=True)

    def test_forced_candidate_cost_does_not_leak_into_sibling_cap_check(self):
        # A (forced) does 5000 nodes of work; B (not forced) does 1. The
        # bundle_node_cap is small enough that B's own cost would never trip
        # it alone -- if A's cost leaked into the cumulative counter, B would
        # be spuriously treated as having overrun.
        w = _bare_worker()
        w.bundle_node_cap = 100
        w.bundle_wall_cap_seconds = 999
        seen = []

        def fake_evaluate_claim(branch_key, words, n_words, idx, budget=None, **kwargs):
            seen.append(idx)
            w._nodes += 5000 if idx == 0 else 1
            return True
        w.evaluate_claim = fake_evaluate_claim
        branch_key = ScoreCache.encode_subset(BRANCH)

        result = w.evaluate_bundle(branch_key, BRANCH, len(BRANCH), "bundle-3",
                                   [0, 1], frozenset({0}))
        self.assertTrue(result)
        self.assertEqual(seen, [0, 1])   # both evaluated
        w.queue.republish_remainder.assert_not_called()

    def test_forced_candidate_after_overrun_is_evaluated_not_republished(self):
        # X (not forced) overruns the cap; Y (forced) sits right after it in
        # best-first order. Y must still be evaluated in this bundle, not
        # swept into the republished remainder -- bouncing an already-
        # republish-limited candidate through another cycle is exactly what
        # `forced` exists to prevent.
        w = _bare_worker()
        w.bundle_node_cap = 100
        w.bundle_wall_cap_seconds = 999
        seen = []

        def fake_evaluate_claim(branch_key, words, n_words, idx, budget=None, **kwargs):
            seen.append(idx)
            w._nodes += 5000 if idx == 0 else 1
            return True
        w.evaluate_claim = fake_evaluate_claim
        branch_key = ScoreCache.encode_subset(BRANCH)

        result = w.evaluate_bundle(branch_key, BRANCH, len(BRANCH), "bundle-4",
                                   [0, 1, 2], forced=frozenset({1}))
        self.assertTrue(result)
        self.assertEqual(seen, [0, 1])   # 1 (forced) evaluated despite the overrun at 0
        w.queue.republish_remainder.assert_called_once_with(branch_key, "bundle-4", [2])

    def test_cancel_during_forced_remainder_evaluation_aborts_bundle(self):
        # X overruns; Y (forced) is evaluated in the post-overrun sweep, but
        # cancellation fires during that sweep -- evaluate_bundle must abort
        # (not silently finish the bundle) exactly as it would mid-main-loop.
        w = _bare_worker()
        w.bundle_node_cap = 100
        w.bundle_wall_cap_seconds = 999
        seen = []

        def fake_evaluate_claim(branch_key, words, n_words, idx, budget=None, **kwargs):
            seen.append(idx)
            if idx == 0:
                w._nodes += 5000
            return True
        w.evaluate_claim = fake_evaluate_claim
        w.cancel = mock.MagicMock(side_effect=[False, True])
        branch_key = ScoreCache.encode_subset(BRANCH)

        result = w.evaluate_bundle(branch_key, BRANCH, len(BRANCH), "bundle-5",
                                   [0, 1, 2], forced=frozenset({1}))
        self.assertFalse(result)
        self.assertEqual(seen, [0])   # cancelled before forced candidate 1 runs
        w.queue.republish_remainder.assert_not_called()
        w.queue.record_bundle_stats.assert_called_once_with(
            branch_key, "bundle-5", 5000, mock.ANY, censored=True)


class TestEvaluateClaimPatternMatrix(unittest.TestCase):
    """evaluate_claim threads self.pattern_matrix into evaluate_candidate.

    _bare_worker() must keep mirroring _BranchWorker.__init__: an
    unconditional field read added to evaluate_claim (like this one) needs
    the matching field added to _bare_worker(), or any non-cancelled call
    to evaluate_claim on a bare worker raises AttributeError (see
    reference_heartbeat_test_skeleton.md for the same class of gap in
    _heartbeat).
    """

    def test_evaluate_candidate_receives_worker_pattern_matrix(self):
        # Deliberately relies on _bare_worker()'s own construction for
        # self.pattern_matrix rather than setting it here afterward --
        # setting it here would mask the exact gap this test exists to catch.
        w = _bare_worker()           # stop_event is None -> not cancelled
        branch_key = ScoreCache.encode_subset(BRANCH)
        with mock.patch('erd_swarm.evaluate_candidate',
                        return_value=(SOLVED, 1.5, 1, False)) as mock_eval:
            result = w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        self.assertTrue(result)
        self.assertIsNone(mock_eval.call_args.kwargs['pattern_matrix'])

    def test_nonadaptive_worker_records_candidate_eta_telemetry(self):
        w = _bare_worker()
        w._adaptive = False
        branch_key = ScoreCache.encode_subset(BRANCH)
        with mock.patch('erd_swarm.evaluate_candidate',
                        return_value=(SOLVED, 1.5, 1, False)):
            result = w.evaluate_claim(branch_key, BRANCH, len(BRANCH), idx=0)
        self.assertTrue(result)
        w.queue.add_claim_telemetry.assert_called_once()
        self.assertGreaterEqual(
            w.queue.add_claim_telemetry.call_args.kwargs[
                'candidate_evaluation_millis'], 0)

    def test_candidate_accuracy_carries_identity_and_lifecycle_fields(self):
        w = _bare_worker()
        w._work_context = _context(opener="salet")
        branch_key = ScoreCache.encode_subset(BRANCH)

        def evaluate_with_metric(*args, **kwargs):
            kwargs["metric_observer"]([3, 2], False, 2.5, 3.0, False)
            return (SOLVED, 1.5, 1, False)

        with mock.patch("erd_swarm.evaluate_candidate",
                        side_effect=evaluate_with_metric):
            self.assertTrue(w.evaluate_claim(
                branch_key, BRANCH, len(BRANCH), idx=0,
                bundle_id="worker-0:99:1", bundle_start_idx=0,
                bundle_end_idx=3))
        call = w.queue.add_candidate_accuracy.call_args
        self.assertEqual(call.kwargs["candidate_word"], CANDIDATES[0])
        self.assertEqual(call.kwargs["worker_id"], "worker-0")
        self.assertEqual(call.kwargs["bundle_id"], "worker-0:99:1")
        self.assertEqual(call.kwargs["idx"], 0)
        self.assertEqual(call.kwargs["outcome"], "exact")
        self.assertEqual(call.kwargs["opener"], "salet")
        self.assertIsInstance(call.kwargs["started_at"], int)
        self.assertIsInstance(call.kwargs["evaluation_millis"], int)
        self.assertGreaterEqual(call.kwargs["evaluation_millis"], 0)


class TestSubbranchSolver(unittest.TestCase):
    """_subbranch_solver returns None for small/unbudgeted branches (inline);
    delegates to cooperative_solve for large branches with a real budget."""

    def test_returns_none_for_branch_below_promote_threshold(self):
        w = _bare_worker()
        words = BRANCH[:3]  # well below PROMOTE_MIN_SIZE (60)
        self.assertIsNone(w._subbranch_solver(words, budget=5))

    def test_returns_none_when_budget_is_none(self):
        # budget=None always inlines regardless of branch size.
        w = _bare_worker()
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        self.assertIsNone(w._subbranch_solver(words, budget=None))

    def test_calls_cooperative_solve_for_large_budgeted_branch(self):
        w = _bare_worker()
        expected = (1.5, 2, False, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        w.cooperative_solve.assert_called_once_with(words, 5, float('inf'))
        self.assertEqual(result, expected)


class TestNoteDepthPromotionSentinel(unittest.TestCase):
    """_note_depth with n<0 marks a cooperative-promoted sub-branch with '•'.

    _note_depth's first arg is the engine's `budget`; the spine is keyed by
    guess_depth = GAME_GUESSES (6) - budget.  Budgets 5/4/3 are guess_depths
    1/2/3 — a worker descending one guess at a time.
    """

    def test_sentinel_marks_spine_with_bullet(self):
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, 12)
        # n=-1 is the cooperative-promotion sentinel.
        w._note_depth(4, -1)
        size, guess, pattern = w._spine[2]
        self.assertEqual(size, '•')
        self.assertIn(1, w._spine)  # parent depth untouched

    def test_none_budget_is_ignored(self):
        w = _bare_worker()
        original_spine = dict(w._spine)

        w._note_depth(None, 12)

        self.assertEqual(w._spine, original_spine)

    def test_fmt_spine_entry_accepts_non_tuple_sentinel(self):
        self.assertEqual(_BranchWorker._fmt_spine_entry('•'), '•')

    def test_sentinel_updates_hb_and_log_max_spine(self):
        w = _bare_worker()
        w._note_depth(5, 50)
        w._note_depth(4, -1)
        self.assertIn(2, w._hb_max_spine)
        self.assertIn(2, w._log_max_spine)

    def test_sentinel_does_not_update_max_spine_when_spine_is_shorter(self):
        """When the new spine is shorter than the accumulated max, neither
        _hb_max_spine nor _log_max_spine is overwritten (depth NOT extended)."""
        w = _bare_worker()
        # Build a 3-level max spine.
        w._note_depth(5, 100)
        w._note_depth(4, 50)
        w._note_depth(3, 20)
        # Reset spine to a single-level view.
        w._spine = {1: 100}
        # n<0 with a shorter spine: len(spine)=2 (after adding depth 2) may still
        # be < len(hb_max_spine=3) depending on prior history, so this exercises
        # the False branch of the len comparison.
        pre_hb = dict(w._hb_max_spine)
        w._note_depth(4, -1)  # spine becomes {1:100, 2:('•',…)}, len=2 < 3
        # If hb_max_spine was already size 3, it should NOT be overwritten.
        if len(pre_hb) > len(w._spine):
            self.assertEqual(w._hb_max_spine, pre_hb)


class TestSpineComposition(unittest.TestCase):
    """_opener_spine / _promoted_spine build the absolute root -> branch path."""

    def test_opener_spine_formats_word_and_pattern(self):
        from wordle_ui import fmt_pattern
        w = _bare_worker()
        self.assertEqual(w._opener_spine('salet', 0),
                         f'SALET {fmt_pattern(0)}')

    def test_opener_spine_none_without_opener(self):
        w = _bare_worker()
        self.assertIsNone(w._opener_spine(None, 0))
        self.assertIsNone(w._opener_spine('salet', None))

    def test_promoted_spine_composes_base_and_descent_edges(self):
        w = _bare_worker()
        w._work_context = _context(spine='SALET -g-g-')
        # Pattern strings pass through unchanged (only ints are fmt_pattern'd).
        w._note_depth(4, 50, 'crane', 'bb-y-')
        w._note_depth(3, 12, 'pound', 'g--y-')
        self.assertEqual(w._promoted_spine(),
                         'SALET -g-g- CRANE bb-y- POUND g--y-')

    def test_promoted_spine_none_without_base(self):
        w = _bare_worker()
        w._work_context = WorkContext.empty()
        w._note_depth(5, 50, 'crane', 'bb-y-')
        self.assertIsNone(w._promoted_spine())

    def test_promoted_spine_skips_sentinel_and_size_only_levels(self):
        w = _bare_worker()
        w._work_context = _context(spine='SALET -g-g-')
        w._note_depth(4, 50, 'crane', 'bb-y-')
        w._note_depth(3, 12)            # size-only level: no guess/pattern
        # Promotion sentinel preserves the guess but sets size to '•'; the edge
        # is still a real edge and must be kept.
        self.assertEqual(w._promoted_spine(), 'SALET -g-g- CRANE bb-y-')


class TestMaybeCheckpoint(unittest.TestCase):
    """_maybe_checkpoint(force=True) always checkpoints both the score cache
    and the queue; force=False respects the CHECKPOINT_SECONDS timer."""

    def test_force_true_always_checkpoints(self):
        import time
        w = _bare_worker()
        w._last_checkpoint = time.time()   # just checkpointed — timer not yet expired
        w._maybe_checkpoint(force=True)
        w.score_cache.checkpoint.assert_called_once()
        # Workers may only checkpoint the queue PASSIVE: TRUNCATE takes the
        # writer lock and stalls every other worker while it waits on readers.
        w.queue.checkpoint.assert_called_once_with("PASSIVE")

    def test_no_checkpoint_before_interval_without_force(self):
        import time
        w = _bare_worker()
        w._last_checkpoint = time.time()   # timer still fresh
        w._maybe_checkpoint(force=False)
        w.score_cache.checkpoint.assert_not_called()
        w.queue.checkpoint.assert_not_called()


class TestWALTrafficLogThrottle(unittest.TestCase):
    """The WAL traffic log runs on its own short timer (not the 5-minute
    checkpoint), so a fast runaway is attributed before the hard ceiling
    latches down; force bypasses the throttle for the shutdown flush."""

    def test_throttled_within_window_does_not_snapshot(self):
        w = _bare_worker()
        w._last_wal_traffic_log = 1000.0
        w._log_wal_traffic(1000.0 + 5)   # dt=5s < WAL_TRAFFIC_LOG_SECONDS
        w.queue.wal_traffic_snapshot.assert_not_called()
        self.assertEqual(w._last_wal_traffic_log, 1000.0)

    def test_fires_after_window(self):
        w = _bare_worker()
        w._last_wal_traffic_log = 1000.0
        fire_at = 1000.0 + erd_swarm.WAL_TRAFFIC_LOG_SECONDS + 1
        w._log_wal_traffic(fire_at)
        w.queue.wal_traffic_snapshot.assert_called_once()
        self.assertEqual(w._last_wal_traffic_log, fire_at)

    def test_force_bypasses_throttle_for_shutdown_flush(self):
        w = _bare_worker()
        w._last_wal_traffic_log = 1000.0
        w._log_wal_traffic(1000.0 + 1, force=True)   # dt=1s but forced
        w.queue.wal_traffic_snapshot.assert_called_once()


class TestWorkerDiskAndPause(unittest.TestCase):
    """_check_disk latches the swarm down at DISK_STOP_FRACTION;
    _respect_checkpoint_pause keeps the worker off the queue database while
    the supervisor's quiesce flag is set."""

    def test_check_disk_latches_and_stops_at_threshold(self):
        w = _bare_worker()
        with mock.patch.object(erd_swarm, "disk_stats",
                               return_value={"used_fraction": 0.95}):
            w._check_disk()
        w.queue.set_disk_stop.assert_called_once()
        self.assertTrue(w._stop_requested)

    def test_check_disk_quiet_below_threshold(self):
        w = _bare_worker()
        with mock.patch.object(erd_swarm, "disk_stats",
                               return_value={"used_fraction": 0.5}):
            w._check_disk()
        w.queue.set_disk_stop.assert_not_called()
        self.assertFalse(w._stop_requested)

    def test_check_disk_is_throttled(self):
        w = _bare_worker()
        w._last_disk_check = time.time()
        with mock.patch.object(erd_swarm, "disk_stats") as stats:
            w._check_disk()
        stats.assert_not_called()

    def test_check_disk_latch_write_failure_still_stops(self):
        # A 100%-full disk can make the latch write itself fail; the worker
        # must still stop cleanly rather than crash out of its run loop.
        w = _bare_worker()
        w.queue.set_disk_stop.side_effect = sqlite3.OperationalError(
            "database or disk is full")
        with mock.patch.object(erd_swarm, "disk_stats",
                               return_value={"used_fraction": 0.95}):
            w._check_disk()
        self.assertTrue(w._stop_requested)

    def test_respect_checkpoint_pause_waits_until_cleared(self):
        w = _bare_worker()
        # One cached entry read, then direct polls: True, True, False.
        w.queue.checkpoint_paused.side_effect = [True, True, True, False]
        with mock.patch.object(erd_swarm.time, "sleep") as sleep:
            w._respect_checkpoint_pause()
        self.assertEqual(sleep.call_count, 2)
        # The cache is reset on exit so the next hot-path check doesn't see a
        # stale pause for a poll interval.
        self.assertFalse(w._pause_active)

    def test_respect_checkpoint_pause_returns_immediately_when_clear(self):
        w = _bare_worker()
        with mock.patch.object(erd_swarm.time, "sleep") as sleep:
            w._respect_checkpoint_pause()
        sleep.assert_not_called()

    def test_pause_flag_read_is_cached(self):
        # The flag check is itself a queue read; hot paths (heartbeat, bound
        # refresh) must not poll it more than once per PAUSE_POLL_SECONDS.
        w = _bare_worker()
        w.queue.checkpoint_paused.return_value = True
        self.assertTrue(w._checkpoint_pause_active())
        self.assertTrue(w._checkpoint_pause_active())
        self.assertEqual(w.queue.checkpoint_paused.call_count, 1)


class TestRivalFinalizeRecovery(unittest.TestCase):
    """A finalizer killed between try_finalize_branch and delete_branch must
    not wedge waiting siblings: the wait heartbeats, and past
    FINALIZE_TAKEOVER_SECONDS the row is reopened and completed."""

    def test_maybe_finalize_returns_false_when_rival_holds(self):
        w = _bare_worker()
        w.queue.branch_done_candidates.return_value = w.n_candidates
        w.queue.try_finalize_branch.return_value = False
        self.assertFalse(w.maybe_finalize(b"k", BRANCH, w.n_candidates))

    def test_await_rival_reopens_stale_finalize_and_completes_it(self):
        w = _bare_worker()
        w.queue.reclaim_stale_finalize.return_value = True
        with mock.patch.object(w, "maybe_finalize") as finalize, \
                mock.patch.object(erd_swarm.time, "sleep") as sleep:
            w._await_rival_finalize(b"k", BRANCH, len(BRANCH), 10)
        finalize.assert_called_once()
        sleep.assert_not_called()
        # The wait stays visible: liveness was written before the takeover.
        w.queue.heartbeat.assert_called_once()

    def test_await_rival_sleeps_while_finalizer_is_fresh(self):
        w = _bare_worker()
        w.queue.reclaim_stale_finalize.return_value = False
        with mock.patch.object(erd_swarm.time, "sleep") as sleep:
            w._await_rival_finalize(b"k", BRANCH, len(BRANCH), 10)
        sleep.assert_called_once()


class TestSolveBranchFocusedLostFinalizeRace(unittest.TestCase):
    """_solve_branch_focused_in_context has two call sites where a rival
    worker may finalize the branch first (maybe_finalize returns False).
    Both must retry rather than abandoning the branch — driven directly
    (not through the mocked-out integration path) so the retry actually
    runs under a controlled interleaving."""

    def _branch_row(self, w, branch_key):
        return {
            'branch_key': branch_key,
            'n_candidates': w.n_candidates,
            'n_words': len(BRANCH),
            'budget': ROOT_BUDGET,
            'opener_work_id': None,
        }

    def test_lost_race_with_claim_exhausted_retries_via_await_rival(self):
        w = _bare_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        branch = self._branch_row(w, branch_key)
        # First pass: every candidate is already claimed elsewhere, and this
        # worker loses the finalize race to a rival.  Second pass: the rival
        # has since deleted the (now-finalized) branch row.
        w.queue.get_branch.side_effect = [branch, None]
        w._claim_bundle = mock.MagicMock(return_value=None)
        w.queue.branch_done_candidates.return_value = w.n_candidates
        w.maybe_finalize = mock.MagicMock(return_value=False)
        w._await_rival_finalize = mock.MagicMock()

        w._solve_branch_focused_in_context(branch)

        w._await_rival_finalize.assert_called_once_with(
            branch_key, sorted(BRANCH), len(BRANCH), w.n_candidates)
        self.assertEqual(w.queue.get_branch.call_count, 2)

    def test_lost_race_after_evaluate_bundle_loops_back_without_await(self):
        w = _bare_worker()
        branch_key = ScoreCache.encode_subset(BRANCH)
        branch = self._branch_row(w, branch_key)
        # First pass: this worker wins a claim, completes it, and the branch
        # is now fully done — but a rival finalizes first.  This call site
        # loops straight back to the top rather than awaiting the rival.
        # Second pass: the rival has since deleted the branch row.
        w.queue.get_branch.side_effect = [branch, None]
        w._claim_bundle = mock.MagicMock(return_value=(1, [0, 1], False))
        w.evaluate_bundle = mock.MagicMock(return_value=True)
        w.queue.branch_done_candidates.return_value = w.n_candidates
        w.maybe_finalize = mock.MagicMock(return_value=False)
        w._await_rival_finalize = mock.MagicMock()

        w._solve_branch_focused_in_context(branch)

        w.maybe_finalize.assert_called_once()
        w._await_rival_finalize.assert_not_called()
        self.assertEqual(w.queue.get_branch.call_count, 2)


class TestCooperativeSolveCachedPath(unittest.TestCase):
    """cooperative_solve returns the cached result immediately when the branch
    is already solved in ScoreCache, without claiming or evaluating any candidate."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_returns_cached_result_without_evaluating_chunks(self):
        words = BRANCH[:3]
        branch_key = ScoreCache.encode_subset(words)
        # Pre-populate with an untainted result (solve_budget=None → reusable
        # at any budget >= max_depth).
        sc = ScoreCache(self.cache_path, BRANCH)
        sc.write(branch_key, ERD_ALL, "crane", 1.5, max_depth=2, solve_budget=None)
        sc.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        self.assertIsNotNone(result)
        status, cost, max_depth, budget_tainted = result
        self.assertEqual(status, SOLVED)
        self.assertAlmostEqual(cost, 1.5)
        self.assertEqual(max_depth, 2)

        # No candidate claims should have been created — the cache hit short-circuited.
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()


class TestTransferredTaintFinalization(unittest.TestCase):
    """A published branch's queue taint reaches its permanent cache row."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.answer_file = self._write("answers.txt", BRANCH)
        self.words_file = self._write("words.txt", CANDIDATES)
        for attribute, path in [("ANSWER_FILE", self.answer_file),
                                ("WORDS_FILE", self.words_file)]:
            patcher = mock.patch.object(erd_swarm, attribute, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as stream:
            stream.write("\n".join(words) + "\n")
        return path

    def _finalize(self, *, transfer_taint):
        worker = _BranchWorker(0, self.cache_path, self.queue_path, None)
        branch_words = BRANCH[:3]
        branch_key = ScoreCache.encode_subset(branch_words)
        try:
            worker.queue.create_branch(
                branch_key, len(branch_words), worker.n_candidates, budget=4)
            if transfer_taint:
                worker.queue.mark_branch_tainted(branch_key)
            worker.queue.mark_claims_done(
                branch_key, range(worker.n_candidates))
            worker.queue.update_branch_best(
                branch_key, "crane", 1.8, max_depth=2)
            with mock.patch.object(erd_swarm, "cache_all_scores"):
                self.assertTrue(worker.maybe_finalize(
                    branch_key, branch_words, worker.n_candidates))
            # A transferred taint records the result under the branch's
            # budget rather than as the unrestricted optimum, so read it
            # back the way a search at that budget would.
            return worker.score_cache.read_for_budget(branch_key, ERD_ALL, 4)
        finally:
            worker.close()

    def test_transferred_taint_limits_cache_row_to_branch_budget(self):
        self.assertEqual(self._finalize(transfer_taint=True),
                         ("crane", 1.8, 2, 4))

    def test_untainted_branch_keeps_unbounded_cache_reuse(self):
        self.assertEqual(self._finalize(transfer_taint=False),
                         ("crane", 1.8, 2, None))


class TestSolveBranchFocusedPrecompletedCandidates(unittest.TestCase):
    """solve_branch_focused finalizes correctly when all candidates are already
    done by other workers: claim_next_bundle returns None,
    branch_done_candidates >= n_candidates, so the worker finalizes from the
    claim-is-None path (not via evaluate_bundle)."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_finalizes_when_all_candidates_pre_completed(self):
        from erd_queue import ERDQueue
        # Pre-mark all candidate claims as done (simulating other workers that
        # have already evaluated every candidate).
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()  # initialise schema
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        q.create_branch(branch_key, len(BRANCH), n_candidates)
        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates
        bundle_id, indices, _forced = q.claim_next_bundle(
            branch_key, "other", n_candidates, order, cost_lower_bound,
            small_count=n_candidates, count_cap=n_candidates)
        for idx in indices:
            q.complete_candidate(branch_key, idx)
        q.close()

        # Our worker enters solve_branch_focused: claim_next_bundle → None (all
        # done), branch_done_candidates >= n_candidates → maybe_finalize.  Since
        # no best_guess was set, maybe_finalize treats it as a loss and deletes
        # the branch without writing a cache entry.
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        # Branch must be gone (finalized + deleted).
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()


class TestSolveBranchFocusedMultiBundleDrain(unittest.TestCase):
    """solve_branch_focused loops back for another bundle when the branch
    isn't yet fully covered — a small_count/count_cap forcing several bundles
    to drain len(CANDIDATES) exercises that loop-continuation path, not just
    the single-bundle and precompleted-candidates cases covered elsewhere."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_drains_all_candidates_across_several_bundles(self):
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, len(BRANCH), len(CANDIDATES), budget=ROOT_BUDGET)
        q.close()

        # small_count=2, count_cap=2: len(CANDIDATES)=10 needs >= 5 bundles,
        # so the loop must claim, evaluate, and loop back repeatedly before
        # coverage is complete and it finalizes.
        w = _BranchWorker(0, self.cache_path, self.queue_path, None,
                          small_count=2, count_cap=2)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))   # finalized + deleted
        q.close()
        sc = ScoreCache(self.cache_path, BRANCH, checkpoint_on_close=False)
        cached = sc.read(branch_key, ERD_ALL)
        sc.close()
        self.assertIsNotNone(cached)

    def test_stops_without_finalizing_when_evaluate_bundle_reports_cancellation(self):
        # evaluate_bundle returning False (cancelled mid-evaluation) must stop
        # the drain loop without finalizing, exactly like a single-bundle
        # cancellation — this is the multi-bundle loop's own cancel exit, not
        # evaluate_claim's.
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, len(BRANCH), len(CANDIDATES), budget=ROOT_BUDGET)
        q.close()

        stop = multiprocessing.Event()
        w = _BranchWorker(0, self.cache_path, self.queue_path, stop)

        def fake_evaluate_bundle(*args, **kwargs):
            stop.set()
            return False
        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate_bundle)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        w.evaluate_bundle.assert_called_once()
        q = ERDQueue(self.queue_path)
        self.assertIsNotNone(q.get_branch(branch_key))   # not finalized
        q.close()

    def test_waits_and_reclaims_when_siblings_hold_remaining_claims(self):
        # Every candidate already claimed (done=0) by another worker: claim
        # is None, but branch_done_candidates is 0 < n_candidates, so this
        # must take the "wait and reclaim" branch rather than finalizing.
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        q.create_branch(branch_key, len(BRANCH), n_candidates, budget=ROOT_BUDGET)
        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates
        q.claim_next_bundle(branch_key, "other", n_candidates, order,
                            cost_lower_bound, small_count=n_candidates,
                            count_cap=n_candidates)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w.cancel = mock.MagicMock(side_effect=[False, True])
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        self.assertGreaterEqual(w.cancel.call_count, 2)
        q = ERDQueue(self.queue_path)
        self.assertIsNotNone(q.get_branch(branch_key))   # not finalized
        q.close()


class TestSolveBranchFocusedClaimTelemetryAttribution(unittest.TestCase):
    """solve_branch_focused's claim_telemetry rows carry branch/bundle
    attribution end to end (issue #197): branch_id, spine, and worker_id are
    populated, and idx falls within [bundle_start_idx, bundle_end_idx] for a
    bundle claim.  This is the only path today that exercises the full
    evaluate_claim -> add_claim_telemetry write path, so it is what catches a
    claim whose work context (and so its spine) never reached the telemetry."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_claim_telemetry_rows_carry_branch_and_bundle_attribution(self):
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, spine="CRANE -----")
        q.close()

        w = _BranchWorker(1, self.cache_path, self.queue_path, None)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        rows = q._conn.execute(
            "SELECT branch_id, spine, worker_id, bundle_id, idx, "
            "bundle_start_idx, bundle_end_idx FROM claim_telemetry "
            "ORDER BY id").fetchall()
        # branch_id is the branches-registry surrogate, not the raw
        # branch_key: resolve it back to confirm the row actually points at
        # the branch this worker solved, not just some non-NULL id.
        expected_branch_id = q._conn.execute(
            "SELECT branch_id FROM branches WHERE branch_key = ?",
            (branch_key,)).fetchone()["branch_id"]
        q.close()
        self.assertTrue(rows)
        # Every row is a candidate evaluation -- the finalize does not write
        # here -- so all of them carry full branch and bundle attribution.
        for row in rows:
            self.assertEqual(row["branch_id"], expected_branch_id)
            self.assertEqual(row["spine"], "CRANE -----")
            self.assertEqual(row["worker_id"], "worker-1")
            self.assertIsNotNone(row["bundle_id"])
            self.assertIsNotNone(row["idx"])
            self.assertLessEqual(row["bundle_start_idx"], row["idx"])
            self.assertLessEqual(row["idx"], row["bundle_end_idx"])

    def test_phases_never_exceed_coordination_millis(self):
        # idle_millis is computed as the remainder, so the five phases sum to
        # coordination_millis by construction -- EXCEPT when the other four
        # already exceed it, where the max(0, ...) clamp floors idle at 0 and
        # the identity breaks.  That is the case worth guarding: a phase
        # counting time from outside the coordination window (queue work done
        # during a candidate's own evaluation) would land here.  This does NOT
        # detect coordination work that simply has no phase -- that inflates
        # idle_millis while keeping the sum exact; see
        # test_scan_time_is_attributed_to_scheduling_not_idle for that.
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, spine="CRANE -----")
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        rows = q._conn.execute(
            "SELECT coordination_millis, candidate_evaluation_millis, "
            "claim_transaction_millis, "
            "claim_commit_millis, busy_wait_millis, scheduling_millis, "
            "idle_millis FROM claim_telemetry ORDER BY id").fetchall()
        q.close()
        self.assertTrue(rows)
        for row in rows:
            self.assertIsNotNone(row["candidate_evaluation_millis"])
            self.assertEqual(
                row["claim_transaction_millis"] + row["claim_commit_millis"]
                + row["busy_wait_millis"] + row["scheduling_millis"]
                + row["idle_millis"],
                row["coordination_millis"])

    def test_scan_time_is_attributed_to_scheduling_not_idle(self):
        # The point of the scheduling phase: work-selection time must be
        # visible as scheduling_millis rather than falling into idle_millis,
        # where a large value reads as "workers are starved" -- the opposite
        # of the truth when work selection is what consumed the window.
        #
        # Drives claim_one (the only path with a scan) with a known delay
        # injected into it, then evaluates a candidate so a telemetry row is
        # written, and asserts where the delay landed.  Deleting the
        # scheduling computation from claim_one fails this test.
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH)
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, spine="CRANE -----")
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        real_scan = w.queue.direct_branches_in_progress

        def slow_scan(*args, **kwargs):
            time.sleep(0.05)          # inside claim_one's scan, no lock held
            return real_scan(*args, **kwargs)
        w.queue.direct_branches_in_progress = slow_scan
        try:
            work = w.claim_one()
            self.assertIsNotNone(work)
            context, branch, _bundle_id, indices, _forced = work
            with w._entered(context):
                w.evaluate_claim(branch_key, decode_subset(branch_key),
                                 branch['n_words'], indices[0],
                                 budget=ROOT_BUDGET)
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        row = q._conn.execute(
            "SELECT scheduling_millis, idle_millis, coordination_millis "
            "FROM claim_telemetry ORDER BY id LIMIT 1").fetchone()
        q.close()
        self.assertGreaterEqual(row["scheduling_millis"], 40)
        # The scan is the bulk of the window, so idle must not have absorbed it.
        self.assertLess(row["idle_millis"], row["scheduling_millis"])

    def test_finalize_cost_lands_on_its_own_branch_and_not_the_next_one(self):
        # A worker that finalizes branch1 then moves on to branch2 must record
        # branch1's finalize cost against branch1, and must not let that span
        # reappear anywhere in branch2's telemetry: the finalize always runs
        # strictly after branch1's own candidates are done, so both a "fold it
        # into the next claim" scheme and a telescoped coordination window
        # that isn't restarted would silently bill it to branch2.
        branch1_words = BRANCH
        branch2_words = BRANCH[:2]
        branch1_key = ScoreCache.encode_subset(branch1_words)
        branch2_key = ScoreCache.encode_subset(branch2_words)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.create_branch(branch1_key, len(branch1_words), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.create_branch(branch2_key, len(branch2_words), len(CANDIDATES),
                        budget=ROOT_BUDGET)
        q.close()

        w = _BranchWorker(1, self.cache_path, self.queue_path, None)
        real_write = w.score_cache.write
        # Slow only branch1's own finalize write, so a large span anywhere
        # else can only be a leak, never branch2's own (fast) finalize cost.
        call_count = {"n": 0}

        def slow_once_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(0.05)
            return real_write(*args, **kwargs)
        w.score_cache.write = slow_once_write
        try:
            w.solve_branch_focused(branch1_key)
            w.solve_branch_focused(branch2_key)
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        finalize_rows = q._conn.execute(
            "SELECT branch_key, cache_write_millis FROM branch_finalize_log "
            "ORDER BY id").fetchall()
        # claim_telemetry stores branch_id, not branch_key -- resolve it back
        # through the branches registry so the rest of this test can compare
        # against branch1_key/branch2_key like the finalize-log rows above.
        claim_rows = q._conn.execute(
            "SELECT b.branch_key AS branch_key, t.coordination_millis "
            "FROM claim_telemetry t JOIN branches b ON t.branch_id = b.branch_id "
            "ORDER BY t.id").fetchall()
        q.close()

        # The slowed finalize is billed to branch1's own finalize row.
        by_branch = {bytes(r["branch_key"]): r["cache_write_millis"]
                     for r in finalize_rows}
        self.assertGreaterEqual(by_branch[branch1_key], 40)
        self.assertLess(by_branch[branch2_key], 40)

        # And it is nowhere in branch2's claim telemetry: the coordination
        # window restarts past the finalize, so branch2's first claim does not
        # inherit branch1's finalize span as idle time.
        branch2_claims = [r for r in claim_rows
                          if bytes(r["branch_key"]) == branch2_key]
        self.assertTrue(branch2_claims)
        for row in branch2_claims:
            self.assertLess(row["coordination_millis"], 40)


class TestClaimOneJoinsInProgressBranch(unittest.TestCase):
    """claim_one() joins a branch that is already in-progress (created by
    another worker) rather than promoting a new one from the pending queue."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_solve_branch_focused_returns_early_for_missing_branch(self):
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH)
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Branch was never registered — should return without error.
            w.solve_branch_focused(branch_key)
        finally:
            w.close()

    def test_solve_branch_focused_records_preferred_role_for_opener_owned_branch(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH[:3])
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(branch_key, 3, 1, "crane", 0)])
        claimed = q.claim_next("peer")
        q.create_branch(
            branch_key, 3, len(CANDIDATES), budget=ROOT_BUDGET,
            priority=claimed["priority"], opener=claimed["opener"],
            opener_pattern=claimed["opener_pattern"],
            opener_work_id=claimed["opener_work_id"])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        recorded = {}

        def capture(*_args, **_kwargs):
            recorded['opener_work_id'] = w._work_context.opener_work_id
            recorded['scheduling_role'] = w._work_context.scheduling_role

        w._solve_branch_focused_in_context = mock.MagicMock(side_effect=capture)
        try:
            w.solve_branch_focused(branch_key)
            self.assertEqual(recorded['opener_work_id'], claimed["opener_work_id"])
            # opener_work_id is not NULL: must be preferred/fallback, never
            # direct (check_opener_work_invariants rejects that pairing).
            self.assertEqual(recorded['scheduling_role'], "preferred")
            self.assertEqual(w.queue.check_opener_work_invariants(), [])
        finally:
            w.close()

    def test_solve_branch_focused_records_direct_role_for_unowned_branch(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH[:3])
        q = ERDQueue(self.queue_path)
        q.create_branch(branch_key, 3, len(CANDIDATES), budget=ROOT_BUDGET)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        recorded = {}

        def capture(*_args, **_kwargs):
            recorded['opener_work_id'] = w._work_context.opener_work_id
            recorded['scheduling_role'] = w._work_context.scheduling_role

        w._solve_branch_focused_in_context = mock.MagicMock(side_effect=capture)
        try:
            w.solve_branch_focused(branch_key)
            self.assertIsNone(recorded['opener_work_id'])
            self.assertEqual(recorded['scheduling_role'], "direct")
        finally:
            w.close()

    def test_claim_one_skips_fully_claimed_in_progress_branch_and_joins_next(self):
        """If the first in-progress branch has all candidates claimed, claim_one
        must continue iterating and claim a candidate from the next branch."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Branch A: all candidates pre-claimed by another worker.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates
        q.claim_next_bundle(key_a, "other-worker", n_candidates, order,
                           cost_lower_bound, small_count=n_candidates,
                           count_cap=n_candidates)

        # Branch B: has a free candidate for our worker to claim.
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q.create_branch(key_b, len(words_b), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            with mock.patch.object(w.queue, 'opener_work_candidates',
                                   wraps=w.queue.opener_work_candidates) as opener_rows:
                result = w.claim_one()
            self.assertEqual(opener_rows.call_count, 0)
        finally:
            w.close()

        # Worker must have skipped branch A (fully claimed) and joined branch B.
        self.assertIsNotNone(result)
        _context_value, branch, bundle_id, indices, forced = result
        self.assertEqual(branch['branch_key'], key_b)
        self.assertTrue(indices)

    def test_claim_one_reads_only_the_top_opener_from_a_large_ladder(self):
        """A claimable top rung must not scan the lower queued openers."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        branch_key = ScoreCache.encode_subset(BRANCH)
        q = ERDQueue(self.queue_path)
        q.add_pending_many([
            (branch_key, len(BRANCH), 512 - number,
             f"opener-{number}", number)
            for number in range(512)
        ])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            with mock.patch.object(w.queue, "opener_work_candidates",
                                   wraps=w.queue.opener_work_candidates) as rows:
                result = w.claim_one()
        finally:
            w.close()

        self.assertIsNotNone(result)
        self.assertEqual(rows.call_args_list,
                         [mock.call(limit=1, after=None)])

    def test_claim_one_records_scan_time_net_of_the_queue_phases(self):
        # The work-selection scan must be charged to scheduling_millis rather
        # than falling into idle_millis, and must exclude the lock wait and
        # claim transaction the queue already accounts for -- otherwise the
        # phases would double-count and overshoot coordination_millis.
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        key = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key, len(BRANCH), len(CANDIDATES), budget=ROOT_BUDGET)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        # Sleep inside claim_next_bundle's open transaction, so the delay is
        # charged to claim_transaction_millis by the queue's own timing.
        real_commit = w.queue._commit_claim_transaction

        def slow_commit(txn_t0):
            time.sleep(0.05)
            return real_commit(txn_t0)
        w.queue._commit_claim_transaction = slow_commit
        try:
            result = w.claim_one()
            attributed = w._queue_attributed_millis()
        finally:
            w.close()

        self.assertIsNotNone(result)
        # The queue booked the 50ms as its own phase ...
        self.assertGreaterEqual(attributed, 40)
        # ... so the scan figure must not also contain it.
        self.assertLess(w._pending_scheduling_millis, 40)

    def test_claim_one_clears_scan_time_when_nothing_is_claimable(self):
        # A scan that selects no branch must not carry its cost forward onto
        # whichever unrelated branch this worker claims later.
        ScoreCache(self.cache_path, BRANCH).close()
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w._pending_scheduling_millis = 999
            self.assertIsNone(w.claim_one())      # empty queue
            self.assertEqual(w._pending_scheduling_millis, 0)
        finally:
            w.close()

    def test_claim_one_discovers_opener_work_after_unclaimable_direct_branch(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        direct_key = ScoreCache.encode_subset(BRANCH)
        q.create_branch(direct_key, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET)
        q.claim_next_bundle(direct_key, "other-worker", n_candidates,
                            list(range(n_candidates)), [0.0] * n_candidates,
                            small_count=n_candidates, count_cap=n_candidates)
        opener_key = ScoreCache.encode_subset(BRANCH[:4])
        q.add_pending_many([(opener_key, 4, 7, "crane", 0)])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        self.assertIsNotNone(result)
        _context_value, branch, _bundle_id, indices, _forced = result
        self.assertEqual(branch['branch_key'], opener_key)
        self.assertEqual(branch['opener'], "crane")
        self.assertTrue(indices)

    def test_stale_worker_preserves_owner_when_joining_opener_active_branch(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        stale_worker = _BranchWorker(0, self.cache_path, self.queue_path, None)
        q = ERDQueue(self.queue_path)
        branch_key = ScoreCache.encode_subset(BRANCH)
        q.add_pending_many([(branch_key, len(BRANCH), 7, "crane", 0)])
        opener_work_id = q.opener_work_rows()[0]["opener_work_id"]
        q.close()

        promoting_worker = _BranchWorker(1, self.cache_path, self.queue_path, None)
        try:
            self.assertIsNotNone(promoting_worker.claim_one())
        finally:
            promoting_worker.close()
        try:
            result = stale_worker.claim_one()
        finally:
            stale_worker.close()

        self.assertIsNotNone(result)
        context, branch, _bundle_id, indices, _forced = result
        self.assertEqual(branch['branch_key'], branch_key)
        self.assertEqual(context.opener_work_id, opener_work_id)
        self.assertTrue(indices)

    def test_claim_one_joins_existing_in_progress_branch(self):
        from erd_queue import ERDQueue
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        # Register branch as in-progress (created, not pending).
        q.create_branch(branch_key, len(BRANCH), n_candidates, budget=ROOT_BUDGET)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        # claim_one should have joined the in-progress branch via
        # branches_in_progress() → claim_next_bundle() → return
        # (context, branch, bundle_id, indices, forced).
        self.assertIsNotNone(result)
        _context_value, branch, bundle_id, indices, forced = result
        self.assertEqual(branch['branch_key'], branch_key)
        self.assertTrue(indices)

    def test_claim_one_joins_active_opener_work_branch(self):
        from erd_queue import ERDQueue
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(branch_key, len(BRANCH), 3, "crane", 0)])
        q.close()

        first = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            self.assertIsNotNone(first.claim_one())
        finally:
            first.close()

        second = _BranchWorker(1, self.cache_path, self.queue_path, None)
        try:
            result = second.claim_one()
        finally:
            second.close()

        self.assertIsNotNone(result)
        _context_value, branch, _bundle_id, indices, _forced = result
        self.assertEqual(branch['branch_key'], branch_key)
        self.assertTrue(indices)

    def test_claim_active_branch_returns_selected_owner_metadata(self):
        from erd_queue import ERDQueue
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(branch_key, len(BRANCH), 1, "crane", 7)])
        q.add_pending_many([(branch_key, len(BRANCH), 9, "slate", 42)])
        openers = {row["opener"]: row for row in q.opener_work_rows()}
        crane = q.claim_next("peer", openers["crane"]["opener_work_id"])
        q.create_branch(
            branch_key, len(BRANCH), len(CANDIDATES), budget=ROOT_BUDGET,
            priority=crane["priority"], opener=crane["opener"],
            opener_pattern=crane["opener_pattern"],
            opener_work_id=crane["opener_work_id"])
        slate_rows = q.branches_in_progress(openers["slate"]["opener_work_id"])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w._claim_active_branch(
                slate_rows, openers["slate"]["opener_work_id"])
        finally:
            w.close()
        context, _branch, _bundle_id, indices, _forced = result
        self.assertEqual(context.opener, "slate")
        self.assertEqual(context.opener_pattern, 42)
        self.assertEqual(context.opener_priority, 9)
        self.assertTrue(indices)

    def test_claim_one_finalizes_completed_direct_branch(self):
        from erd_queue import ERDQueue
        branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        q.create_branch(branch_key, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET)
        order = list(range(n_candidates))
        q.claim_next_bundle(branch_key, "other-worker", n_candidates, order,
                            [0.0] * n_candidates, small_count=n_candidates,
                            count_cap=n_candidates)
        for index in order:
            q.complete_candidate(branch_key, index)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            self.assertIsNone(w.claim_one())
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        try:
            self.assertIsNone(q.get_branch(branch_key))
        finally:
            q.close()

    def test_claim_one_records_fallback_role_when_preferred_opener_is_held(self):
        from erd_queue import ERDQueue, SCHEDULING_ROLE_FALLBACK
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        # crane (priority 9) is the preferred opener, but its only branch is
        # already fully held by another worker: no claimable bundle.
        crane_key = ScoreCache.encode_subset(BRANCH[:3])
        q.add_pending_many([(crane_key, 3, 9, "crane", 0)])
        crane_id = q.opener_work_rows()[0]["opener_work_id"]
        claimed = q.claim_next("other-worker", crane_id)
        q.create_branch(claimed['branch_key'], claimed['n_words'], n_candidates,
                        priority=claimed['priority'],
                        opener=claimed['opener'],
                        opener_pattern=claimed['opener_pattern'],
                        opener_work_id=claimed['opener_work_id'])
        q.claim_next_bundle(
            crane_key, "other-worker", n_candidates, list(range(n_candidates)),
            [0.0] * n_candidates, small_count=n_candidates, count_cap=n_candidates,
            expected_opener_work_id=crane_id, expected_opener_priority=9)
        # slate (priority 1) has claimable work: the only eligible fallback.
        slate_key = ScoreCache.encode_subset(BRANCH[1:4])
        q.add_pending_many([(slate_key, 3, 1, "slate", 0)])
        slate_id = next(row["opener_work_id"] for row in q.opener_work_rows()
                        if row["opener"] == "slate")
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        self.assertIsNotNone(result)
        context, branch, _bundle_id, indices, _forced = result
        self.assertEqual(branch['branch_key'], slate_key)
        self.assertEqual(context.opener_work_id, slate_id)
        self.assertEqual(context.scheduling_role, SCHEDULING_ROLE_FALLBACK)
        self.assertTrue(indices)

    def test_claim_one_records_preferred_role_for_equal_priority_tiebreak_winner(self):
        """Two openers at the SAME requested priority: opener_work_candidates()
        picks one via its tiebreak (state, then opener_work_id), but neither was
        skipped in favor of a higher-priority peer, so both are 'preferred', not
        just the admission-order-first one."""
        from erd_queue import ERDQueue, SCHEDULING_ROLE_PREFERRED
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        crane_key = ScoreCache.encode_subset(BRANCH[:3])
        slate_key = ScoreCache.encode_subset(BRANCH[1:4])
        q.add_pending_many([(crane_key, 3, 5, "crane", 0)])
        q.add_pending_many([(slate_key, 3, 5, "slate", 0)])
        admission_order = [row["opener"] for row in q.opener_work_candidates()]
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.claim_one()
        finally:
            w.close()

        self.assertIsNotNone(result)
        context, branch, _bundle_id, indices, _forced = result
        # Whichever of the two equal-priority openers the tiebreak orders
        # first is the one claim_one() actually claims; either way it must
        # be reported preferred — nothing higher-priority was skipped.
        self.assertIn(admission_order[0], ("crane", "slate"))
        self.assertEqual(context.scheduling_role, SCHEDULING_ROLE_PREFERRED)
        self.assertTrue(indices)

    def test_claim_one_returns_to_preferred_role_once_it_is_claimable_again(self):
        from erd_queue import (ERDQueue, SCHEDULING_ROLE_PREFERRED,
                               SCHEDULING_ROLE_FALLBACK)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        crane_key = ScoreCache.encode_subset(BRANCH[:3])
        q.add_pending_many([(crane_key, 3, 9, "crane", 0)])
        crane_id = q.opener_work_rows()[0]["opener_work_id"]
        claimed = q.claim_next("other-worker", crane_id)
        q.create_branch(claimed['branch_key'], claimed['n_words'], n_candidates,
                        priority=claimed['priority'],
                        opener=claimed['opener'],
                        opener_pattern=claimed['opener_pattern'],
                        opener_work_id=claimed['opener_work_id'])
        q.claim_next_bundle(
            crane_key, "other-worker", n_candidates, list(range(n_candidates)),
            [0.0] * n_candidates, small_count=n_candidates, count_cap=n_candidates,
            expected_opener_work_id=crane_id, expected_opener_priority=9)
        slate_key = ScoreCache.encode_subset(BRANCH[1:4])
        q.add_pending_many([(slate_key, 3, 1, "slate", 0)])
        q.close()

        first = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            first_result = first.claim_one()
        finally:
            first.close()
        self.assertEqual(first_result[0].scheduling_role, SCHEDULING_ROLE_FALLBACK)

        # "other-worker" never heartbeat, so its held crane claims are stale:
        # once reclaimed, crane is claimable again at the next boundary.
        q = ERDQueue(self.queue_path)
        q._conn.execute(
            "UPDATE candidate_claims SET claimed_at = 0 WHERE claimed_by = ?",
            ("other-worker",))
        self.assertEqual(q.reclaim_stale_claims(120), n_candidates)
        q.close()

        second = _BranchWorker(1, self.cache_path, self.queue_path, None)
        try:
            second_result = second.claim_one()
        finally:
            second.close()

        self.assertIsNotNone(second_result)
        context, branch, _bundle_id, indices, _forced = second_result
        self.assertEqual(branch['branch_key'], crane_key)
        self.assertEqual(context.scheduling_role, SCHEDULING_ROLE_PREFERRED)
        self.assertTrue(indices)


class TestCooperativeSolveFullPath(unittest.TestCase):
    """cooperative_solve evaluates all candidates and returns the correct result
    when the branch is NOT pre-cached.  This exercises the main cooperative
    loop (create branch, evaluate candidates, finalize, read cache) and the
    maybe_finalize early-return path (called after each candidate, returns early
    until the last one makes all candidates done)."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_solves_uncached_branch_and_returns_result(self):
        # Use all 5 BRANCH words so cooperative_solve evaluates all 10 CANDIDATES
        # one at a time.  maybe_finalize() returns early on each until the last
        # candidate makes branch_done_candidates == n_candidates and finalizes.
        words = BRANCH
        branch_key = ScoreCache.encode_subset(words)

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        self.assertIsNotNone(result)
        status, cost, max_depth, budget_tainted = result
        self.assertEqual(status, SOLVED)
        self.assertGreater(cost, 0)

        # The branch must be finalized — rows deleted from the queue.
        from erd_queue import ERDQueue
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()

        # The result must be persisted to the cache.
        sc = ScoreCache(self.cache_path, BRANCH, checkpoint_on_close=False)
        from wordle_engine import ERD_ALL
        cached = sc.read(branch_key, ERD_ALL)
        sc.close()
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(cached[1], cost, places=6)

    def test_cooperative_solve_claims_after_opener_reprioritization(self):
        from erd_queue import ERDQueue

        root_key = ScoreCache.encode_subset(BRANCH[:3])
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(root_key, 3, 1, "crane", 0)])
        opener_work_id = q.opener_work_rows()[0]["opener_work_id"]
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w._work_context = WorkContext(
            opener_work_id, 1, "crane", 0, root_key, None,
            SCHEDULING_ROLE_PREFERRED)
        original_claim_bundle = w._claim_bundle
        reprioritized = False
        # A worker locked out of its own reprioritized branch never claims
        # again and spins in the wait loop, so the guard bounds the claim
        # attempts rather than the wall clock: exhausting them stops the
        # worker and fails with the reason instead of hanging the job.  The
        # healthy descent claims twice.
        claim_attempt_limit = 20
        claim_attempts = 0

        def reprioritize_then_claim(*args, **kwargs):
            nonlocal reprioritized, claim_attempts
            claim_attempts += 1
            if claim_attempts > claim_attempt_limit:
                w._stop_requested = True
                return None
            if not reprioritized:
                reprioritized = True
                updated = ERDQueue(self.queue_path)
                try:
                    self.assertTrue(updated.set_opener_work_priority(
                        opener_work_id, 9))
                finally:
                    updated.close()
            return original_claim_bundle(*args, **kwargs)

        w._claim_bundle = mock.MagicMock(side_effect=reprioritize_then_claim)
        try:
            result = w.cooperative_solve(BRANCH, ROOT_BUDGET)
        finally:
            w.close()

        self.assertTrue(reprioritized)
        self.assertLessEqual(
            claim_attempts, claim_attempt_limit,
            "cooperative_solve made no progress after "
            f"{claim_attempt_limit} claim attempts")
        self.assertEqual(result[0], SOLVED)

    def test_does_not_finalize_when_evaluate_bundle_reports_cancellation(self):
        words = BRANCH
        branch_key = ScoreCache.encode_subset(words)
        stop = multiprocessing.Event()
        w = _BranchWorker(0, self.cache_path, self.queue_path, stop)

        def fake_evaluate_bundle(*args, **kwargs):
            stop.set()
            return False
        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate_bundle)
        try:
            w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        w.evaluate_bundle.assert_called_once()
        from erd_queue import ERDQueue
        q = ERDQueue(self.queue_path)
        self.assertIsNotNone(q.get_branch(branch_key))   # not finalized
        q.close()

    def test_finalizes_when_all_candidates_already_done_by_other_workers(self):
        # Mirrors solve_branch_focused's precompleted-candidates case: another
        # cooperative worker already evaluated every candidate before this
        # worker joins, so _claim_bundle returns None but coverage is already
        # complete — cooperative_solve must finalize from that branch, not
        # via evaluate_bundle.
        from erd_queue import ERDQueue
        words = BRANCH
        branch_key = ScoreCache.encode_subset(words)
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        q.create_branch(branch_key, len(words), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates
        bundle_id, indices, _forced = q.claim_next_bundle(
            branch_key, "other", n_candidates, order, cost_lower_bound,
            small_count=n_candidates, count_cap=n_candidates)
        for idx in indices:
            q.complete_candidate(branch_key, idx)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w.cooperative_solve(words, ROOT_BUDGET)
        finally:
            w.close()

        # No feasible guess was ever recorded, so this finalizes as a proven
        # loss: the branch is gone, but there is no cache entry to read.
        self.assertIsNotNone(result)
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(branch_key))
        q.close()


class TestHelpOtherBranch(unittest.TestCase):
    """_help_other_branch claims and evaluates one candidate from a branch other
    than the excluded branch, returning True if work was found, False if not."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_help_other_branch_returns_true_when_candidate_evaluated(self):
        """When a candidate is available in another branch, help_other_branch
        claims and evaluates it, then returns True."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create branch A (the one being excluded).
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)

        # Create branch B with a free candidate claim.
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q.create_branch(key_b, len(words_b), n_candidates,
                        budget=ROOT_BUDGET, priority=1)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            result = w._help_other_branch(key_a)
        finally:
            w.close()

        # Should have evaluated a candidate and returned True.
        self.assertTrue(result)

    def test_help_other_branch_reads_active_branches_once(self):
        """Helping must not issue one active-branch query per opener."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        excluded_key = ScoreCache.encode_subset(BRANCH)
        other_key = ScoreCache.encode_subset(BRANCH[:4])
        q.create_branch(excluded_key, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, priority=0)
        q.create_branch(other_key, len(BRANCH) - 1, len(CANDIDATES),
                        budget=ROOT_BUDGET, priority=1)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            with mock.patch.object(w.queue, "branches_in_progress",
                                   wraps=w.queue.branches_in_progress) as branches:
                with mock.patch.object(w.queue, "direct_branches_in_progress",
                                       wraps=w.queue.direct_branches_in_progress) as direct:
                    result = w._help_other_branch(excluded_key)
        finally:
            w.close()

        self.assertTrue(result)
        branches.assert_called_once_with()
        direct.assert_not_called()

    def test_returns_true_without_finalizing_when_evaluate_bundle_reports_cancellation(self):
        # _help_other_branch reports True (a bundle WAS claimed) even when
        # evaluation itself was cancelled — it never finalizes in that case.
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q.create_branch(key_b, len(words_b), n_candidates,
                        budget=ROOT_BUDGET, priority=1)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w.evaluate_bundle = mock.MagicMock(return_value=False)
        try:
            result = w._help_other_branch(key_a)
        finally:
            w.close()

        self.assertTrue(result)
        q = ERDQueue(self.queue_path)
        self.assertIsNotNone(q.get_branch(key_b))   # not finalized
        q.close()

    def test_help_other_branch_switches_and_restores_opener_context(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q = ERDQueue(self.queue_path)
        q.add_pending_many([(key_b, len(words_b), 9, "slate", 42)])
        claimed = q.claim_next("peer")
        q.create_branch(
            key_b, len(words_b), len(CANDIDATES), budget=ROOT_BUDGET,
            priority=claimed["priority"], opener=claimed["opener"],
            opener_pattern=claimed["opener_pattern"],
            opener_work_id=claimed["opener_work_id"])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        waiting_context = _context(
            branch_key=b"waiting", opener_work_id=999, opener_priority=1,
            opener="crane", opener_pattern=7)
        w._work_context = waiting_context

        def evaluate(*_args, **_kwargs):
            self.assertEqual(w._work_context.branch_key, key_b)
            self.assertEqual(
                w._work_context.opener_work_id, claimed["opener_work_id"])
            self.assertEqual(w._work_context.opener_priority, 9)
            self.assertEqual(w._work_context.opener, "slate")
            self.assertEqual(w._work_context.opener_pattern, 42)
            # Opener-owned: fallback, regardless of which opener owns it —
            # the worker's own branch is blocked, so this is fallback work.
            self.assertEqual(w._work_context.scheduling_role, "fallback")
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=evaluate)
        try:
            self.assertTrue(w._help_other_branch(b"excluded"))
            self.assertEqual(w._work_context, waiting_context)
        finally:
            w.close()

    def test_help_other_branch_preserves_direct_branch_metadata(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        words_b = BRANCH[:4]
        key_b = ScoreCache.encode_subset(words_b)
        q = ERDQueue(self.queue_path)
        q.create_branch(
            key_b, len(words_b), len(CANDIDATES), budget=ROOT_BUDGET,
            priority=9, opener="crane", opener_pattern=7)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)

        def evaluate(*_args, **_kwargs):
            self.assertIsNone(w._work_context.opener_work_id)
            self.assertEqual(w._work_context.opener_priority, 9)
            self.assertEqual(w._work_context.opener, "crane")
            self.assertEqual(w._work_context.opener_pattern, 7)
            # No live opener-work ownership: direct, not fallback — no
            # opener-first admission decision was made for this branch, and
            # opener_work_id IS NULL pairs only with DIRECT
            # (check_opener_work_invariants rejects opener_work_id IS NULL
            # paired with preferred/fallback).
            self.assertEqual(w._work_context.scheduling_role, "direct")
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=evaluate)
        try:
            self.assertTrue(w._help_other_branch(b"excluded"))
            self.assertEqual(
                w.queue.check_opener_work_invariants(), [])
        finally:
            w.close()

    def test_help_other_branch_returns_false_when_no_candidates_available(self):
        """When no other branches have available candidate claims, help_other_branch
        returns False without claiming anything."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create only one branch and fully claim all its candidates.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        order = list(range(n_candidates))
        cost_lower_bound = [0.0] * n_candidates
        q.claim_next_bundle(key_a, "other-worker", n_candidates, order,
                           cost_lower_bound, small_count=n_candidates,
                           count_cap=n_candidates)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Exclude a different branch key (none exist, so effectively no branches to help).
            fake_exclude_key = ScoreCache.encode_subset(BRANCH[:3])
            result = w._help_other_branch(fake_exclude_key)
        finally:
            w.close()

        # Should return False (no candidates to claim).
        self.assertFalse(result)

    def test_help_other_branch_skips_excluded_branch(self):
        """When the only available branch matches exclude_branch_key,
        help_other_branch skips it and returns False."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Create one branch with free candidates.
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), n_candidates,
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            # Exclude the only branch — nothing else to help.
            result = w._help_other_branch(key_a)
        finally:
            w.close()

        # Should return False (the only available branch was excluded).
        self.assertFalse(result)


class TestHelpOtherBranchPromotesHigherPriority(unittest.TestCase):
    """issue #214: an opener with only pending branches must be able to start
    while a lower-priority opener has active branches available to help
    with — and the reverse must not happen (a lower-priority pending opener
    must not preempt a higher-priority branch already in progress)."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _opener_work_id(self, q, opener):
        return next(
            row["opener_work_id"] for row in q.opener_work_candidates()
            if row["opener"] == opener)

    def test_promotes_pending_higher_priority_opener_over_active_lower_priority_one(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Opener "low" (priority 1) has an active branch with free candidates —
        # the branch this call would otherwise fall back to joining.
        words_low = BRANCH[:4]
        key_low = ScoreCache.encode_subset(words_low)
        q.add_pending_many([(key_low, len(words_low), 1, "low", 10)])
        claimed_low = q.claim_next("setup", self._opener_work_id(q, "low"))
        q.create_branch(
            key_low, len(words_low), n_candidates, budget=ROOT_BUDGET,
            priority=claimed_low["priority"],
            opener=claimed_low["opener"],
            opener_pattern=claimed_low["opener_pattern"],
            opener_work_id=claimed_low["opener_work_id"])

        # Opener "high" (priority 9) has only a pending branch: 74-pending/
        # 0-active AUDIO from the issue.  It must start without being promoted
        # by a setup helper first.
        words_high = BRANCH
        key_high = ScoreCache.encode_subset(words_high)
        q.add_pending_many([(key_high, len(words_high), 9, "high", 20)])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        served = []

        def fake_evaluate(branch_key, *_args, **_kwargs):
            served.append(bytes(branch_key))
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate)
        try:
            result = w._help_other_branch(b"unrelated-exclude")
            row = w.queue.get_branch(key_high)
        finally:
            w.close()

        self.assertTrue(result)
        self.assertEqual(served, [key_high])
        self.assertNotIn(key_low, served)
        # "high" is now promoted into an active branch.
        self.assertIsNotNone(row)

    def test_does_not_promote_lower_priority_pending_opener_over_active_higher_priority_one(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # Opener "high" (priority 9) already has an active branch.
        words_high = BRANCH[:4]
        key_high = ScoreCache.encode_subset(words_high)
        q.add_pending_many([(key_high, len(words_high), 9, "high", 10)])
        claimed_high = q.claim_next("setup", self._opener_work_id(q, "high"))
        q.create_branch(
            key_high, len(words_high), n_candidates, budget=ROOT_BUDGET,
            priority=claimed_high["priority"],
            opener=claimed_high["opener"],
            opener_pattern=claimed_high["opener_pattern"],
            opener_work_id=claimed_high["opener_work_id"])

        # Opener "low" (priority 1) has only a pending branch: lower priority
        # than the branch already in progress, so it must not be promoted.
        words_low = BRANCH
        key_low = ScoreCache.encode_subset(words_low)
        q.add_pending_many([(key_low, len(words_low), 1, "low", 20)])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        served = []

        def fake_evaluate(branch_key, *_args, **_kwargs):
            served.append(bytes(branch_key))
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate)
        try:
            result = w._help_other_branch(b"unrelated-exclude")
            row = w.queue.get_branch(key_low)
        finally:
            w.close()

        self.assertTrue(result)
        self.assertEqual(served, [key_high])
        # "low" was never promoted: it stays pending, not active.
        self.assertIsNone(row)

    def test_does_not_promote_lower_priority_pending_opener_over_branch_already_being_served(self):
        """A worker blocked deep inside a higher-priority branch's dependency
        (no OTHER active branches exist at all) must not abandon it for a
        lower-priority pending opener just because nothing else is active."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)

        words_low = BRANCH
        key_low = ScoreCache.encode_subset(words_low)
        q.add_pending_many([(key_low, len(words_low), 1, "low", 20)])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        # Simulate being blocked inside a priority-9 branch: no other active
        # branch exists, but the context this call is nested under outranks
        # "low".
        w._work_context = _context(
            branch_key=b"served-branch", opener_priority=9,
            opener="served", opener_pattern=1)
        w.evaluate_bundle = mock.MagicMock(return_value=False)
        try:
            result = w._help_other_branch(b"served-branch")
            row = w.queue.get_branch(key_low)
        finally:
            w.close()

        self.assertFalse(result)
        self.assertIsNone(row)
        w.evaluate_bundle.assert_not_called()

    def test_widens_its_own_opener_when_only_active_branch_is_the_one_excluded(self):
        """The production shape from the issue: a worker blocked on AUDIO's
        only active branch, with AUDIO also holding a pending branch under
        the SAME opener_work_id and a lower-priority opener (SCOPE) holding
        an active one.  The worker must widen AUDIO — promote its own
        pending branch — rather than serve SCOPE, and rather than treat
        AUDIO's active branch (the one excluded, since it's the one being
        waited on) as already covering the opener.

        Both AUDIO branches are added in a single add_pending_many call so
        they share one opener_work_id (add_pending_many creates a fresh
        opener_work row per call, even for a repeated opener/priority
        pair) — otherwise the pending branch would land under a second,
        merely coincidentally-tied opener_work_id whose own
        branches_in_progress() is trivially empty, which would exercise the
        equal-priority fix but not the joinable_opener_ids fix.
        """
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # AUDIO (priority 9): one active branch (the one this call excludes,
        # standing in for "already being served") plus one pending branch,
        # both under the same opener_work_id.  n_words picks which pending
        # row claim_next promotes first (largest n_words wins), so the
        # larger branch becomes "active" and the smaller stays pending.
        words_audio_active = BRANCH
        key_audio_active = ScoreCache.encode_subset(words_audio_active)
        words_audio_pending = BRANCH[:3]
        key_audio_pending = ScoreCache.encode_subset(words_audio_pending)
        q.add_pending_many([
            (key_audio_active, len(words_audio_active), 9, "audio", 10),
            (key_audio_pending, len(words_audio_pending), 9, "audio", 10),
        ])
        audio_opener_work_id = self._opener_work_id(q, "audio")
        claimed_audio = q.claim_next("setup", audio_opener_work_id)
        self.assertEqual(claimed_audio["branch_key"], key_audio_active)
        q.create_branch(
            key_audio_active, len(words_audio_active), n_candidates,
            budget=ROOT_BUDGET, priority=claimed_audio["priority"],
            opener=claimed_audio["opener"],
            opener_pattern=claimed_audio["opener_pattern"],
            opener_work_id=claimed_audio["opener_work_id"])

        # SCOPE (priority 1): one active branch, joinable and lower priority.
        # A BRANCH subset (not CANDIDATES-only words like "brain"): branch
        # words must be valid answers, and the fixture's answer file is
        # BRANCH alone — the join loop below does try to claim this branch
        # pre-fix (promotion is wrongly skipped), so an out-of-answer word
        # here would fail on pattern_matrix.answer_indices regardless of the
        # fix under test.
        words_penis = BRANCH[-2:]
        key_penis = ScoreCache.encode_subset(words_penis)
        q.add_pending_many([(key_penis, len(words_penis), 1, "scope", 20)])
        claimed_penis = q.claim_next("setup", self._opener_work_id(q, "scope"))
        q.create_branch(
            key_penis, len(words_penis), n_candidates, budget=ROOT_BUDGET,
            priority=claimed_penis["priority"],
            opener=claimed_penis["opener"],
            opener_pattern=claimed_penis["opener_pattern"],
            opener_work_id=claimed_penis["opener_work_id"])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w._work_context = _context(
            branch_key=key_audio_active, opener_work_id=audio_opener_work_id,
            opener_priority=9, opener="audio",
            opener_pattern=claimed_audio["opener_pattern"])
        served = []

        def fake_evaluate(branch_key, *_args, **_kwargs):
            served.append(bytes(branch_key))
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate)
        try:
            result = w._help_other_branch(key_audio_active)
            audio_pending_row = w.queue.get_branch(key_audio_pending)
        finally:
            w.close()

        self.assertTrue(result)
        self.assertEqual(served, [key_audio_pending])
        self.assertNotIn(key_penis, served)
        # AUDIO's pending branch is now promoted (active), not left pending.
        self.assertIsNotNone(audio_pending_row)

    def test_prefers_joining_own_openers_other_active_branch_over_promoting(self):
        """When the worker's own opener already has ANOTHER joinable active
        branch (distinct from the one excluded), the promotion loop must
        skip it — 'prefer joining over promoting' — and leave its pending
        branch alone; the join loop below picks up the other active branch
        instead."""
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        n_candidates = len(CANDIDATES)

        # AUDIO (priority 9): the excluded active branch (being served),
        # a SECOND active branch with free candidates (joinable), and a
        # pending branch that must stay pending while the second active
        # branch remains available to join.
        words_excluded = BRANCH
        key_excluded = ScoreCache.encode_subset(words_excluded)
        words_joinable = BRANCH[:3]
        key_joinable = ScoreCache.encode_subset(words_joinable)
        words_pending = BRANCH[:2]
        key_pending = ScoreCache.encode_subset(words_pending)
        q.add_pending_many([
            (key_excluded, len(words_excluded), 9, "audio", 10),
            (key_joinable, len(words_joinable), 9, "audio", 10),
            (key_pending, len(words_pending), 9, "audio", 10),
        ])
        audio_opener_work_id = self._opener_work_id(q, "audio")
        claimed_excluded = q.claim_next("setup", audio_opener_work_id)
        self.assertEqual(claimed_excluded["branch_key"], key_excluded)
        q.create_branch(
            key_excluded, len(words_excluded), n_candidates,
            budget=ROOT_BUDGET, priority=claimed_excluded["priority"],
            opener=claimed_excluded["opener"],
            opener_pattern=claimed_excluded["opener_pattern"],
            opener_work_id=claimed_excluded["opener_work_id"])
        claimed_joinable = q.claim_next("setup", audio_opener_work_id)
        self.assertEqual(claimed_joinable["branch_key"], key_joinable)
        q.create_branch(
            key_joinable, len(words_joinable), n_candidates,
            budget=ROOT_BUDGET, priority=claimed_joinable["priority"],
            opener=claimed_joinable["opener"],
            opener_pattern=claimed_joinable["opener_pattern"],
            opener_work_id=claimed_joinable["opener_work_id"])
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w._work_context = _context(
            branch_key=key_excluded, opener_work_id=audio_opener_work_id,
            opener_priority=9, opener="audio",
            opener_pattern=claimed_excluded["opener_pattern"])
        served = []

        def fake_evaluate(branch_key, *_args, **_kwargs):
            served.append(bytes(branch_key))
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate)
        try:
            result = w._help_other_branch(key_excluded)
            pending_row = w.queue.get_branch(key_pending)
        finally:
            w.close()

        self.assertTrue(result)
        self.assertEqual(served, [key_joinable])
        # The still-pending branch was never promoted: joining took priority.
        self.assertIsNone(pending_row)


class TestHelpOtherBranchRecursionBound(unittest.TestCase):
    """issue #214: _help_other_branch must not let its own recursive
    evaluate_bundle chain grow the worker's stack without bound."""

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def test_short_circuits_at_the_cap_without_scanning_the_queue(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        w._help_recursion_depth = erd_swarm.MAX_HELP_RECURSION_DEPTH
        w.queue.opener_work_candidates = mock.MagicMock(
            wraps=w.queue.opener_work_candidates)
        try:
            result = w._help_other_branch(b"excluded")
        finally:
            w.close()

        self.assertFalse(result)
        w.queue.opener_work_candidates.assert_not_called()

    def test_recursion_never_exceeds_the_configured_cap(self):
        from erd_queue import ERDQueue
        ScoreCache(self.cache_path, BRANCH).close()
        q = ERDQueue(self.queue_path)
        key_a = ScoreCache.encode_subset(BRANCH)
        q.create_branch(key_a, len(BRANCH), len(CANDIDATES),
                        budget=ROOT_BUDGET, priority=0)
        q.close()

        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        # Every claim attempt "succeeds" against a fixed dummy bundle, so
        # sustained recursion is driven purely by the fake evaluate_bundle
        # below rather than by how many real candidates remain claimable.
        w._claim_bundle = mock.MagicMock(
            return_value=("bundle-x", [0], frozenset()))
        depths_seen = []

        def fake_evaluate_bundle(*_args, **_kwargs):
            depths_seen.append(w._help_recursion_depth)
            w._help_other_branch(b"nested-exclude")
            return False

        w.evaluate_bundle = mock.MagicMock(side_effect=fake_evaluate_bundle)
        try:
            w._help_other_branch(b"excluded")
        finally:
            w.close()

        self.assertEqual(depths_seen,
                         list(range(1, erd_swarm.MAX_HELP_RECURSION_DEPTH + 1)))
        # Fully unwound: no leaked recursion count on the worker.
        self.assertEqual(w._help_recursion_depth, 0)


class TestClaimBundleRetry(unittest.TestCase):
    """issue #214: claim_next_bundle's CLAIM_RETRY sentinel means the SAME
    branch remains claimable, so _claim_bundle — the single funnel every
    claim path goes through — must retry it transparently rather than
    reporting no work, which would divert a caller to a different branch."""

    def test_retries_transparently_and_returns_the_eventual_bundle(self):
        w = _bare_worker()
        w._packing_stats = mock.MagicMock(
            return_value=([0, 1, 2], [0.0, 0.0, 0.0]))
        real_bundle = ("bundle-1", [0, 1], frozenset())
        w.queue.claim_next_bundle = mock.MagicMock(
            side_effect=[erd_queue.CLAIM_RETRY, erd_queue.CLAIM_RETRY,
                        real_bundle])

        result = w._claim_bundle(b"branch", 3, ["a", "b", "c"], ROOT_BUDGET)

        self.assertEqual(result, real_bundle)
        self.assertEqual(w.queue.claim_next_bundle.call_count, 3)

    def test_gives_up_after_the_retry_cap_and_reports_no_work(self):
        w = _bare_worker()
        w._packing_stats = mock.MagicMock(
            return_value=([0, 1, 2], [0.0, 0.0, 0.0]))
        w.queue.claim_next_bundle = mock.MagicMock(
            return_value=erd_queue.CLAIM_RETRY)

        result = w._claim_bundle(b"branch", 3, ["a", "b", "c"], ROOT_BUDGET)

        self.assertIsNone(result)
        self.assertEqual(w.queue.claim_next_bundle.call_count,
                         erd_swarm.CLAIM_RETRY_ATTEMPTS)

    def test_plain_none_is_not_retried(self):
        w = _bare_worker()
        w._packing_stats = mock.MagicMock(
            return_value=([0, 1, 2], [0.0, 0.0, 0.0]))
        w.queue.claim_next_bundle = mock.MagicMock(return_value=None)

        result = w._claim_bundle(b"branch", 3, ["a", "b", "c"], ROOT_BUDGET)

        self.assertIsNone(result)
        self.assertEqual(w.queue.claim_next_bundle.call_count, 1)


class TestInfeasibleCandidateCounters(unittest.TestCase):
    def test_depth_infeasible_candidate_attributes_its_nodes(self):
        worker = _bare_worker()

        def evaluate(*_args, **_kwargs):
            worker._nodes += 7
            return (OVER_DEPTH_BUDGET, float('inf'), None, True)

        with mock.patch.object(
                erd_swarm, "evaluate_candidate", side_effect=evaluate):
            self.assertTrue(worker.evaluate_claim(
                b"branch", BRANCH, len(BRANCH), 0, budget=4))

        worker.queue.add_nodes_spent.assert_called_once_with(
            b"branch", 7, infeasible=True)
        worker.queue.mark_branch_tainted.assert_called_once_with(b"branch")


class TestTwoLevelERDPruneBundles(unittest.TestCase):
    def _worker(self, bound_erd=3.1):
        worker = _bare_worker()
        worker.pattern_matrix = mock.MagicMock()
        worker.pattern_matrix.answer_indices.return_value = list(range(len(BRANCH)))
        worker.branch_floor_table = object()
        worker.queue.read_branch_best.return_value = (
            "clart", bound_erd, None)
        worker.queue.complete_bundle_two_level_erd_prunes.side_effect = (
            lambda _branch_key, _bundle_id, candidate_indices, nodes_spent=0,
            wall_millis=0, bound_erd=None, worker_count=None, worker_id=None:
                len(candidate_indices))

        def count_heartbeat(*_args, **_kwargs):
            worker._nodes += 1

        worker._heartbeat = mock.MagicMock(side_effect=count_heartbeat)
        return worker

    def test_an_all_pruned_bundle_uses_one_completion_write(self):
        worker = self._worker()
        candidate_indices = [0, 1, 2]
        worker._evaluate_bundle_member = mock.MagicMock(return_value=True)

        with mock.patch.object(
                erd_swarm, "candidate_two_level_cost_lower_bound",
                return_value=3.2) as lower_bound:
            completed = worker.evaluate_bundle(
                b"branch", BRANCH, len(BRANCH), "bundle-1",
                candidate_indices, frozenset(), budget=5)

        self.assertTrue(completed)
        self.assertEqual(lower_bound.call_count, len(candidate_indices))
        worker.queue.complete_bundle_two_level_erd_prunes.assert_called_once_with(
            b"branch", "bundle-1", candidate_indices,
            nodes_spent=len(candidate_indices), wall_millis=mock.ANY,
            bound_erd=3.1, worker_count=1, worker_id=worker.name)
        worker.queue.add_nodes_spent.assert_not_called()
        worker._evaluate_bundle_member.assert_not_called()
        worker.queue.record_bundle_stats.assert_called_once()

    def test_only_survivors_reach_ordinary_candidate_evaluation(self):
        worker = self._worker()
        candidate_indices = [0, 1, 2]
        worker._evaluate_bundle_member = mock.MagicMock(return_value=True)

        with mock.patch.object(
                erd_swarm, "candidate_two_level_cost_lower_bound",
                side_effect=[3.2, 2.9, 3.3]):
            completed = worker.evaluate_bundle(
                b"branch", BRANCH, len(BRANCH), "bundle-1",
                candidate_indices, frozenset(), budget=5)

        self.assertTrue(completed)
        worker.queue.complete_bundle_two_level_erd_prunes.assert_called_once_with(
            b"branch", "bundle-1", [0, 2],
            nodes_spent=len(candidate_indices), wall_millis=mock.ANY,
            bound_erd=3.1, worker_count=1, worker_id=worker.name)
        self.assertEqual(worker._evaluate_bundle_member.call_count, 1)
        self.assertEqual(worker._evaluate_bundle_member.call_args.args[3], 1)

    def test_a_bound_below_three_uses_the_existing_evaluation_path(self):
        worker = self._worker(bound_erd=2.9)
        candidate_indices = [0, 1, 2]
        worker._evaluate_bundle_member = mock.MagicMock(return_value=True)

        with mock.patch.object(
                erd_swarm, "candidate_two_level_cost_lower_bound") as lower_bound:
            completed = worker.evaluate_bundle(
                b"branch", BRANCH, len(BRANCH), "bundle-1",
                candidate_indices, frozenset(), budget=5)

        self.assertTrue(completed)
        lower_bound.assert_not_called()
        worker.queue.complete_bundle_two_level_erd_prunes.assert_not_called()
        self.assertEqual(worker._evaluate_bundle_member.call_count,
                         len(candidate_indices))

    def test_missing_bound_skips_the_two_level_preflight(self):
        worker = self._worker()
        worker.queue.read_branch_best.return_value = (None, None, None)

        pruned, cancelled = worker._complete_bundle_two_level_erd_prunes(
            b"branch", BRANCH, len(BRANCH), "bundle-1", [0])

        self.assertEqual(pruned, frozenset())
        self.assertFalse(cancelled)
        worker.queue.complete_bundle_two_level_erd_prunes.assert_not_called()

    def test_preflight_cancellation_persists_prior_prunes(self):
        worker = self._worker()
        worker.cancel = mock.MagicMock(side_effect=[False, True])

        with mock.patch.object(
                erd_swarm, "candidate_two_level_cost_lower_bound",
                return_value=3.2):
            pruned, cancelled = worker._complete_bundle_two_level_erd_prunes(
                b"branch", BRANCH, len(BRANCH), "bundle-1", [0, 1])

        self.assertEqual(pruned, frozenset({0}))
        self.assertTrue(cancelled)
        worker.queue.complete_bundle_two_level_erd_prunes.assert_called_once_with(
            b"branch", "bundle-1", [0], nodes_spent=1,
            wall_millis=mock.ANY, bound_erd=3.1, worker_count=1,
            worker_id=worker.name)


class TestMidLoopPublisher(unittest.TestCase):
    """_MidLoopPublisher.enter() / check() / record_inline() unit tests."""

    def _pub(self, predicted=None):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        pub = erd_swarm._MidLoopPublisher(w)
        return pub, w

    def test_enter_returns_none_below_min_words(self):
        pub, _ = self._pub()
        # MIN_PUBLISH_BRANCH_WORDS is 2; a 1-word frame is a base case.
        self.assertIsNone(pub.enter(BRANCH[:1], budget=0))

    def test_enter_returns_token_at_min_words(self):
        pub, _ = self._pub(predicted=1000)
        words = BRANCH[:2]  # exactly MIN_PUBLISH_BRANCH_WORDS
        token = pub.enter(words, budget=0)
        self.assertIsNotNone(token)
        nodes_at_entry, predicted, entry_time, bw, depth = token[:5]
        self.assertEqual(bw, words)
        self.assertEqual(predicted, 1000)
        self.assertTrue(token[5])  # armed

    def test_enter_cold_model_still_returns_token(self):
        pub, _ = self._pub(predicted=None)  # cold model
        token = pub.enter(BRANCH[:6], budget=1)
        self.assertIsNotNone(token)
        _, predicted, _, _, _ = token[:5]
        self.assertIsNone(predicted)  # token carries None when model is cold

    def test_check_returns_none_for_none_token(self):
        pub, _ = self._pub()
        self.assertIsNone(pub.check(None, CANDIDATES, 0, None, None, None, 5))

    def test_check_cold_model_under_backstop_returns_none(self):
        # Cold model: the node-proportionate check can't arm, and the wall-clock
        # backstop hasn't elapsed on a fresh token, so the frame stays inline.
        pub, _ = self._pub(predicted=None)
        token = pub.enter(BRANCH[:6], budget=0)
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))

    def test_check_fires_on_wall_clock_backstop_when_cold(self):
        # Cold model, but the frame has run past COLD_BACKSTOP_SECONDS: the
        # backstop hands off the remainder even with no cost-model prediction.
        pub, w = self._pub(predicted=None)
        nodes_at_entry, _, _, words, entry_budget = \
            pub.enter(BRANCH[:6], budget=0)[:5]
        old_entry = time.time() - (erd_swarm.COLD_BACKSTOP_SECONDS + 1)
        token = [nodes_at_entry, None, old_entry, words, entry_budget, True]
        w._nodes = nodes_at_entry + 50   # some work done, no prediction to compare
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        result = pub.check(token, CANDIDATES, 0, None, None, None, 5)
        self.assertIsNotNone(result)
        w.cooperative_solve.assert_called_once()
        # The firing is recorded for offline tuning, with predicted=None (cold).
        w.queue.add_backstop_telemetry.assert_called_once()
        args = w.queue.add_backstop_telemetry.call_args[0]
        self.assertEqual(args[0], len(words))   # n_words
        self.assertIsNone(args[4])              # predicted_nodes is None when cold

    def test_check_warm_overrun_does_not_record_backstop(self):
        # The node-proportionate path is the model working as intended; only the
        # wall-clock backstop is recorded for tuning.
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:6], budget=0)
        # Past the proportionate trigger (OVERRUN_K * predicted) AND the
        # absolute break-even gate (_publish_threshold, bootstrap when cold).
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        pub.check(token, CANDIDATES, 0, None, None, None, 5)
        w.queue.add_backstop_telemetry.assert_not_called()

    def test_check_returns_none_when_under_overrun_threshold(self):
        pub, w = self._pub(predicted=100)
        token = pub.enter(BRANCH[:6], budget=0)
        # _nodes hasn't changed: delta = 0 <= OVERRUN_K * 100
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))

    def test_check_fires_on_overrun(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        # Simulate spending > OVERRUN_K * predicted nodes
        # Past the proportionate trigger (OVERRUN_K * predicted) AND the
        # absolute break-even gate (_publish_threshold, bootstrap when cold).
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        # last_index=0 of a 10-candidate list → 9 remaining (>= MIN_HANDOFF).
        result = pub.check(token, CANDIDATES, 0, None, None, None, 5)
        self.assertIsNotNone(result)
        w.cooperative_solve.assert_called_once()

    def test_check_returns_none_when_remaining_count_below_threshold(self):
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:6], budget=0)
        # Past the proportionate trigger (OVERRUN_K * predicted) AND the
        # absolute break-even gate (_publish_threshold, bootstrap when cold).
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        # last_index=7 of a 10-candidate list → only 2 remaining (< MIN_HANDOFF=4)
        self.assertIsNone(pub.check(token, CANDIDATES, 7, None, None, None, 5))

    def test_check_marks_prefix_done(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        # Past the proportionate trigger (OVERRUN_K * predicted) AND the
        # absolute break-even gate (_publish_threshold, bootstrap when cold).
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        # last_index=1 → the evaluated prefix is CANDIDATES[:2].
        pub.check(token, CANDIDATES, 1, None, None, None, 5)
        call_args = w.queue.mark_claims_done.call_args
        self.assertIsNotNone(call_args)
        marked_indices = call_args[0][1]  # second positional arg is the indices list
        # Each evaluated prefix word appears in the done list at its all_words index.
        for word in CANDIDATES[:2]:
            self.assertIn(w.all_words.index(word), marked_indices)

    def test_check_transfers_prefix_budget_taint_before_done_marks(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=3)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 2, False))

        result = pub.check(
            token, CANDIDATES, 1, None, None, None, 3,
            prefix_budget_tainted=True)

        branch_key = ScoreCache.encode_subset(words)
        taint_call = mock.call.mark_branch_tainted(branch_key)
        done_call = mock.call.mark_claims_done(branch_key, mock.ANY)
        self.assertLess(
            w.queue.mock_calls.index(taint_call),
            w.queue.mock_calls.index(done_call))
        self.assertEqual(result, (erd_swarm.SOLVED, 2.0, 2, True))

    def test_check_preserves_cancel_status_after_transferring_taint(self):
        pub, w = self._pub(predicted=10)
        words = BRANCH[:6]
        token = pub.enter(words, budget=3)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=erd_swarm.CANCEL_RECVD)

        result = pub.check(
            token, CANDIDATES, 1, None, None, None, 3,
            prefix_budget_tainted=True)

        branch_key = ScoreCache.encode_subset(words)
        w.queue.mark_branch_tainted.assert_called_once_with(branch_key)
        w.queue.mark_claims_done.assert_called_once()
        self.assertEqual(result, erd_swarm.CANCEL_RECVD)

    def test_record_inline_accumulates_buffer(self):
        import math
        pub, w = self._pub(predicted=100)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        w._nodes = 200   # 200 - 0 = 200 nodes for this frame
        pub.record_inline(token)
        # Buffer is keyed by (size, budget); this frame entered at budget=0.
        self.assertIn((len(words), 0), w._cost_model_buffer)
        s, sq, c = w._cost_model_buffer[(len(words), 0)]
        self.assertEqual(c, 1)
        self.assertAlmostEqual(s, math.log(200), places=6)
        self.assertAlmostEqual(sq, math.log(200) ** 2, places=6)

    def test_record_inline_is_noop_for_none_token(self):
        pub, w = self._pub()
        pub.record_inline(None)
        self.assertEqual(w._cost_model_buffer, {})


class TestPublishStormGuards(unittest.TestCase):
    """The latches that prevent a publish-per-iteration write storm: the
    absolute break-even gate, the disarm-on-decline token flag, and the
    stop/WAL-ceiling refusal.  Without them, a collapsed cost model (typical
    ~= cache-hit node counts) makes the proportionate trigger fire on every
    iteration of every frame, and an unjoinable raced row turns that into
    mark_claims_done per candidate — the WAL flood that filled the disk."""

    def _pub(self, predicted=None):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        return erd_swarm._MidLoopPublisher(w), w

    def test_overrun_below_break_even_does_not_publish(self):
        pub, w = self._pub(predicted=5)
        token = pub.enter(BRANCH[:5], budget=0)
        w._nodes = erd_swarm.OVERRUN_K * 5 + 1   # proportionate trigger true
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))
        w.queue.create_branch.assert_not_called()
        w.queue.mark_claims_done.assert_not_called()
        self.assertTrue(token[5])   # not disarmed: may fire later in the frame

    def test_unjoinable_budget_disarms_and_writes_nothing(self):
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:5], budget=0)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.queue.create_branch.return_value = False   # raced: row exists
        w.queue.get_branch.return_value = {'budget': 4, 'ceiling': None}
        w.cooperative_solve = mock.MagicMock()
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))
        self.assertFalse(token[5])
        w.queue.mark_claims_done.assert_not_called()
        w.queue.add_cost_sample.assert_not_called()
        w.cooperative_solve.assert_not_called()
        # Disarmed: later iterations return immediately, touching nothing.
        w.queue.get_branch.reset_mock()
        self.assertIsNone(pub.check(token, CANDIDATES, 1, None, None, None, 5))
        w.queue.get_branch.assert_not_called()

    def test_cooperative_decline_disarms(self):
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:5], budget=0)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(return_value=None)
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))
        self.assertFalse(token[5])

    def test_stop_requested_disarms_without_writing(self):
        pub, w = self._pub(predicted=10)
        token = pub.enter(BRANCH[:5], budget=0)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w._stop_requested = True
        self.assertIsNone(pub.check(token, CANDIDATES, 0, None, None, None, 5))
        self.assertFalse(token[5])
        w.queue.create_branch.assert_not_called()


class TestPromotedSpineDepthCap(unittest.TestCase):
    """_promoted_spine caps composition at the promoted branch's guess depth:
    the live descent map keeps edges from deeper frames the engine has already
    unwound out of, and composing them into a shallower promotion builds a
    spine whose guess_depth contradicts the branch's budget — which
    create_branch's budget + guess_depth = root_budget invariant rejects."""

    def _descended_worker(self):
        w = _bare_worker()
        w._work_context = _context(spine='ALIBI ----- ELOPE y-y--')
        w._spine = {3: (7, 'rends', '-y-y-'), 4: (3, 'motza', '-g---')}
        return w

    def test_stale_deeper_entries_are_excluded(self):
        w = self._descended_worker()
        self.assertEqual(w._promoted_spine(3),
                         'ALIBI ----- ELOPE y-y-- RENDS -y-y-')

    def test_uncapped_keeps_full_descent(self):
        w = self._descended_worker()
        self.assertEqual(
            w._promoted_spine(),
            'ALIBI ----- ELOPE y-y-- RENDS -y-y- MOTZA -g---')

    def test_publisher_spine_matches_budget_invariant(self):
        w = self._descended_worker()
        w.queue.get_cost_typical.return_value = 10
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        pub = erd_swarm._MidLoopPublisher(w)
        token = pub.enter(BRANCH[:5], budget=3)
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        pub.check(token, CANDIDATES, 0, None, None, None, 3)
        spine = w.queue.create_branch.call_args.kwargs['spine']
        self.assertEqual(spine, 'ALIBI ----- ELOPE y-y-- RENDS -y-y-')
        self.assertEqual(guess_depth_from_spine(spine) + 3, w.root_budget)


class TestWorkerWALCeiling(unittest.TestCase):
    """Worker-side backstop for the queue WAL hard ceiling: trips, latches,
    requests a stop, and is throttled between probes."""

    def test_trips_latches_and_requests_stop(self):
        w = _bare_worker()
        w.queue.wal_size_bytes.return_value = \
            erd_swarm.QUEUE_WAL_HARD_CEILING_BYTES
        self.assertTrue(w._wal_ceiling_tripped())
        self.assertTrue(w._stop_requested)
        # Latched: no further size probes once tripped.
        w.queue.wal_size_bytes.reset_mock()
        self.assertTrue(w._wal_ceiling_tripped())
        w.queue.wal_size_bytes.assert_not_called()

    def test_below_ceiling_probes_are_throttled(self):
        w = _bare_worker()
        self.assertFalse(w._wal_ceiling_tripped())
        self.assertFalse(w._stop_requested)
        # Within WAL_CEILING_CHECK_SECONDS the size is not re-probed.
        self.assertFalse(w._wal_ceiling_tripped())
        self.assertEqual(w.queue.wal_size_bytes.call_count, 1)


class TestSubbranchSolverCostModel(unittest.TestCase):
    """_subbranch_solver respects cost model: warm model gates on predicted
    nodes; cold model falls back to PROMOTE_MIN_SIZE size threshold."""

    def test_warm_model_below_threshold_inlines(self):
        w = _bare_worker()
        # Warm model predicts < PUBLISH_THRESHOLD_BOOTSTRAP nodes.
        w.queue.get_cost_typical.return_value = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP - 1
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        self.assertIsNone(w._subbranch_solver(words, budget=5))

    def test_warm_model_above_threshold_promotes(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        expected = (erd_swarm.SOLVED, 2.0, 3, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertEqual(result, expected)
        w.cooperative_solve.assert_called_once()

    def test_cold_model_small_branch_inlines(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = None  # cold model
        # A cold small branch inlines directly; the mid-loop publisher's
        # wall-clock backstop bounds it if it turns out to be a tarpit.
        w.cooperative_solve = mock.MagicMock()
        words_small = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE - 1)
        self.assertIsNone(w._subbranch_solver(words_small, budget=5))
        w.cooperative_solve.assert_not_called()

    def test_cold_model_above_size_threshold_promotes(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = None  # cold model
        expected = (erd_swarm.SOLVED, 2.5, 4, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertEqual(result, expected)

    def test_promotion_uses_frames_own_budget_inline_case(self):
        # The frame's own budget (5) predicts below threshold (inline); a
        # different budget (6) predicts above threshold (promote).  A correct
        # decision must query the cell for the frame's actual budget, not any
        # other budget's cell.
        w = _bare_worker()
        threshold = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP
        cell_for_frame_budget = threshold - 1
        cell_for_other_budget = threshold + 1

        def fake_get_cost_typical(policy, n_words, budget):
            return cell_for_frame_budget if budget == 5 else cell_for_other_budget

        w.queue.get_cost_typical.side_effect = fake_get_cost_typical
        w.cooperative_solve = mock.MagicMock()
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertIsNone(result)
        w.cooperative_solve.assert_not_called()
        w.queue.get_cost_typical.assert_called_once_with(ERD_ALL, len(words), 5)

    def test_promotion_uses_frames_own_budget_promote_case(self):
        # Mirror case: the frame's own budget (5) predicts above threshold
        # (promote); a different budget (6) predicts below threshold
        # (inline).  A correct decision must still query the frame's own
        # budget cell and promote.
        w = _bare_worker()
        threshold = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP
        cell_for_frame_budget = threshold + 1
        cell_for_other_budget = threshold - 1

        def fake_get_cost_typical(policy, n_words, budget):
            return cell_for_frame_budget if budget == 5 else cell_for_other_budget

        w.queue.get_cost_typical.side_effect = fake_get_cost_typical
        expected = (erd_swarm.SOLVED, 2.0, 3, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (erd_swarm.PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, budget=5)
        self.assertEqual(result, expected)
        w.queue.get_cost_typical.assert_called_once_with(ERD_ALL, len(words), 5)


class TestLogEMA(unittest.TestCase):
    """_LogEMA: edge cases for add() and value()."""

    def test_add_nonpositive_value_is_noop(self):
        ema = erd_swarm._LogEMA(tau=1.0, min_weight=1)
        ema.add(0)
        ema.add(-5)
        self.assertIsNone(ema.value())  # weight still 0

    def test_add_with_implicit_now_warms_ema(self):
        ema = erd_swarm._LogEMA(tau=1.0, min_weight=1)
        ema.add(10)   # now=None → time.time() called internally
        self.assertIsNotNone(ema.value())

    def test_add_twice_applies_decay(self):
        # Use a very short tau so decay is substantial and clearly exercises the
        # decay branch; min_weight=1 so value() is readable after each add.
        ema = erd_swarm._LogEMA(tau=0.1, min_weight=1)
        ema.add(100, now=0.0)
        ema.add(100, now=1.0)   # second call: self._last is not None → decay applied
        # Decayed weight < 2; value should still be computable with min_weight=1.
        self.assertIsNotNone(ema.value())

    def test_value_returns_geometric_mean_when_warm(self):
        import math
        ema = erd_swarm._LogEMA(tau=1e9, min_weight=2)  # huge tau → no decay
        ema.add(10, now=0.0)
        ema.add(1000, now=0.0)
        expected = math.exp((math.log(10) + math.log(1000)) / 2)
        self.assertAlmostEqual(ema.value(), expected, places=6)


class TestPublishThresholdWarmPath(unittest.TestCase):
    """_publish_threshold returns SAFETY_FACTOR * coord / node_time when warm."""

    def test_warm_threshold_exceeds_bootstrap(self):
        w = _bare_worker()
        # Use now=0.0 for all adds so decay = exp(0) = 1 and weight accumulates
        # cleanly to exactly min_weight, without floating-point decay undershooting.
        for _ in range(erd_swarm._PUBLISH_EMA_MIN_WEIGHT):
            w._coord_ema.add(0.01, now=0.0)      # 10 ms coordination
            w._node_time_ema.add(1e-6, now=0.0)  # 1 µs / node
        threshold = w._publish_threshold()
        self.assertGreater(threshold, erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP)


class TestFlushCostModelBuffer(unittest.TestCase):
    """_flush_cost_model_buffer propagates buffered inline samples to the queue."""

    def test_flush_calls_update_logsums_per_size(self):
        w = _bare_worker()
        # Buffer is keyed by (size, budget); flush forwards budget as a kwarg.
        w._cost_model_buffer = {(5, 0): (2.0, 4.0, 1), (10, 1): (3.0, 9.0, 2)}
        w._flush_cost_model_buffer()
        self.assertEqual(w.queue.update_cost_model_logsums.call_count, 2)
        budgets = {c.kwargs.get('budget')
                   for c in w.queue.update_cost_model_logsums.call_args_list}
        self.assertEqual(budgets, {0, 1})
        w._typical_cache.clear()  # confirmed cleared by flush
        self.assertEqual(w._cost_model_buffer, {})

    def test_flush_empty_buffer_clears_without_update(self):
        w = _bare_worker()
        w._flush_cost_model_buffer()
        w.queue.update_cost_model_logsums.assert_not_called()


class TestRecordInlineEdgeCases(unittest.TestCase):
    """record_inline edge cases: zero-node frame and duplicate size in buffer."""

    def _pub(self):
        w = _bare_worker()
        pub = erd_swarm._MidLoopPublisher(w)
        return pub, w

    def test_record_inline_noop_when_nodes_unchanged(self):
        pub, w = self._pub()
        token = pub.enter(BRANCH[:4], budget=0)
        # _nodes stays 0 → nodes_spent = 0 → early return, nothing buffered.
        pub.record_inline(token)
        self.assertEqual(w._cost_model_buffer, {})

    def test_record_inline_accumulates_second_sample_for_same_size(self):
        import math
        pub, w = self._pub()
        words = BRANCH[:4]
        token1 = pub.enter(words, budget=0)
        w._nodes = 100
        pub.record_inline(token1)
        token2 = pub.enter(words, budget=0)
        w._nodes = 300   # 200 more nodes for second frame of same size
        pub.record_inline(token2)
        s, sq, c = w._cost_model_buffer[(len(words), 0)]
        self.assertEqual(c, 2)
        self.assertAlmostEqual(s, math.log(100) + math.log(200), places=6)


class TestNoteDepthDeeperPruning(unittest.TestCase):
    """_note_depth prunes deeper spine entries when resetting a shallower depth."""

    def test_note_depth_prunes_deeper_entries(self):
        w = _bare_worker()
        w._note_depth(6, 200, guess="crane", pattern="00000")
        w._note_depth(5, 50, guess="slate", pattern="10000")
        w._note_depth(4, 10, guess="trace", pattern="22222")
        # Revisit depth 1: depths 2+ should be pruned.
        w._note_depth(5, 48, guess="tales", pattern="01000")
        self.assertNotIn(2, w._spine)
        self.assertIn(1, w._spine)

    def test_sentinel_prunes_deeper_entries(self):
        w = _bare_worker()
        w._note_depth(5, 50, guess="crane", pattern="00000")
        w._note_depth(4, 20, guess="slate", pattern="10000")
        w._note_depth(3, 5, guess="trace", pattern="22222")
        # Sentinel at depth 1 with n<0 should prune depths 2 and 3.
        w._note_depth(5, -1)
        self.assertNotIn(2, w._spine)
        self.assertNotIn(3, w._spine)


class TestMidLoopPublisherBranchEdgeCases(unittest.TestCase):
    """Publisher check() with best_guess and with empty done_indices."""

    def _pub_overrun(self, predicted=10, best_guess=None,
                     best_max_remaining_depth=3):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        pub = erd_swarm._MidLoopPublisher(w)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        # Past the proportionate trigger AND the absolute break-even gate.
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        result = pub.check(token, CANDIDATES, 0, best_guess, 1.5,
                           best_max_remaining_depth, 5)
        return result, w, pub

    def test_check_calls_update_branch_best_when_best_guess_known(self):
        result, w, _ = self._pub_overrun(best_guess="crane")
        self.assertIsNotNone(result)
        # The seed carries the winner's worst-case line, not just its cost: a
        # branch seeded with an unknown depth finalizes into a cache row no
        # budget can ever reuse, so it reads as unsolved forever.
        w.queue.update_branch_best.assert_called_once_with(
            ScoreCache.encode_subset(BRANCH[:6]), "crane", 1.5, 3)

    def test_check_skips_update_branch_best_when_no_best_guess(self):
        result, w, _ = self._pub_overrun(best_guess=None)
        self.assertIsNotNone(result)
        w.queue.update_branch_best.assert_not_called()

    def test_check_handles_empty_done_indices(self):
        # Use a candidate_list of words that are NOT in word_idx so done_indices
        # is empty — mark_claims_done should not be called.
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = 10
        pub = erd_swarm._MidLoopPublisher(w)
        words = BRANCH[:6]
        token = pub.enter(words, budget=0)
        # Past the proportionate trigger (OVERRUN_K * predicted) AND the
        # absolute break-even gate (_publish_threshold, bootstrap when cold).
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        w.cooperative_solve = mock.MagicMock(
            return_value=(erd_swarm.SOLVED, 2.0, 3, False))
        unknown_words = ["zzzzz"] * 10
        pub.check(token, unknown_words, 0, None, None, None, 5)
        w.queue.mark_claims_done.assert_not_called()


class TestFinalizeTelemetryFailureIsolation(unittest.TestCase):
    """A failing finalize-telemetry write must not kill the worker or skip
    cleanup: the branch result is published to the score cache before the
    telemetry calls, and mark_done/delete_branch must run regardless, or the
    branch's claim rows leak and its pending row is never retired."""

    def _finalizing_worker(self):
        w = _bare_worker()
        w.queue.branch_done_candidates.return_value = len(BRANCH)
        w.queue.try_finalize_branch.return_value = True
        w.queue.read_branch_meta.return_value = ("crane", 1.8, 2, False, 4, None, False)
        w.queue.get_branch.return_value = {
            "nodes_spent": 0,
            "infeasible_candidates": 0,
            "infeasible_nodes": 0,
            "created_at": 100,
            "finalized_at": 200,
            "first_best_at": 150,
            "nodes_at_first_best": 40,
            "spine": "SALET -g-g-",
        }
        w.queue.finalize_bundle_stats.return_value = (None, None, None, None)
        return w

    def test_telemetry_insert_failure_still_runs_cleanup(self):
        w = self._finalizing_worker()
        w.queue.add_branch_finalize_log.side_effect = RuntimeError(
            "no column named total_bundle_wall_millis")
        key = b"branch-key"
        # Real key shape: the packing order depends on the branch's budget, so
        # a branch finalized at one budget must not leave an order behind for
        # any budget it was cached under.  Another branch's entry is seeded too,
        # so the sweep is shown to be selective rather than a clear().
        other_key = b"other-branch-key"
        w._packing_stats_cache[(key, 4)] = object()
        w._packing_stats_cache[(key, 5)] = object()
        w._packing_stats_cache[(other_key, 4)] = object()
        with mock.patch.object(erd_swarm, "cache_all_scores"):
            w.maybe_finalize(key, BRANCH, len(BRANCH))
        w.score_cache.write.assert_called_once()
        w.queue.mark_done.assert_called_once_with(key)
        w.queue.delete_branch.assert_called_once_with(key)
        self.assertEqual(list(w._packing_stats_cache), [(other_key, 4)])

    def test_bundle_stats_aggregation_failure_still_runs_cleanup(self):
        w = self._finalizing_worker()
        w.queue.finalize_bundle_stats.side_effect = RuntimeError("boom")
        key = b"branch-key"
        with mock.patch.object(erd_swarm, "cache_all_scores"):
            w.maybe_finalize(key, BRANCH, len(BRANCH))
        w.queue.add_branch_finalize_log.assert_not_called()
        w.queue.mark_done.assert_called_once_with(key)
        w.queue.delete_branch.assert_called_once_with(key)

    def test_branch_cleanup_snapshots_the_opener_that_just_completed(self):
        w = self._finalizing_worker()
        key = b"branch-key"
        w.queue.delete_branch.return_value = ["salet"]
        w._snapshot_completed_openers = mock.MagicMock()
        with mock.patch.object(erd_swarm, "cache_all_scores"):
            w.maybe_finalize(key, BRANCH, len(BRANCH))
        w._snapshot_completed_openers.assert_called_once_with(["salet"])


class TestCompletedOpenerSnapshots(unittest.TestCase):
    def test_snapshot_skips_empty_completion_lists_without_queue_aggregation(self):
        worker = _bare_worker()

        worker._snapshot_completed_openers(None)

        worker.queue.completed_opener_timing.assert_not_called()
        worker.queue.opener_rows.assert_not_called()

    def test_snapshot_persists_the_completed_opener_timing(self):
        worker = _bare_worker()
        worker.queue.completed_opener_timing.return_value = {
            "first_created_at": 100,
            "completed_at": 160,
            "worker_millis": 2_000,
            "telemetry_epochs": "3",
        }

        worker._snapshot_completed_openers(["SALET"])

        worker.score_cache.write_completed_opener_summary.assert_called_once_with(
            "SALET", erd_swarm.ERD_ALL, 160, 60_000, 2_000, (3,))

    def test_snapshot_failure_does_not_abort_the_worker(self):
        worker = _bare_worker()
        worker.queue.completed_opener_timing.side_effect = [
            RuntimeError("locked"),
            {"first_created_at": 100, "completed_at": 160,
             "worker_millis": 2_000, "telemetry_epochs": "3"},
        ]

        with self.assertLogs("wordle", level="ERROR"):
            worker._snapshot_completed_openers(["salet", "crane"])

        worker.score_cache.write_completed_opener_summary.assert_called_once_with(
            "crane", erd_swarm.ERD_ALL, 160, 60_000, 2_000, (3,))


class TestSubbranchSolverForwardsCeiling(unittest.TestCase):
    """_subbranch_solver passes the frame's ceiling through to
    cooperative_solve on every promotion path."""

    def test_size_promotion_forwards_ceiling(self):
        w = _bare_worker()
        w._adaptive = False
        expected = (SOLVED, 2.0, 3, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, 5, 2.5)
        self.assertEqual(result, expected)
        w.cooperative_solve.assert_called_once_with(words, 5, 2.5)

    def test_warm_model_promotion_forwards_ceiling(self):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = 1e12  # far above threshold
        expected = (SOLVED, 2.0, 3, False)
        w.cooperative_solve = mock.MagicMock(return_value=expected)
        words = ["crane"] * (PROMOTE_MIN_SIZE + 1)
        result = w._subbranch_solver(words, 5, 2.5)
        self.assertEqual(result, expected)
        w.cooperative_solve.assert_called_once_with(words, 5, 2.5)


class TestCooperativeSolveCeiling(unittest.TestCase):
    """cooperative_solve's ceiling handling: the cut_results fast path, the
    reuse-miss ledger, the join rule, and pending-row ceiling suppression."""

    def _worker(self):
        w = _bare_worker()
        w.score_cache = mock.MagicMock()
        w.score_cache.read_with_depth.return_value = None
        w.score_cache.read_for_budget.return_value = None
        w.score_cache.read_loss.return_value = None
        w.queue.read_cut_result.return_value = []
        w.queue.has_pending_row.return_value = False
        w.queue.create_branch.return_value = True
        # Exit the help loop immediately wherever it is reached: these tests
        # exercise the decisions before it, not the claim/evaluate cycle.
        w._stop_requested = True
        return w

    def test_satisfying_cut_short_circuits(self):
        w = self._worker()
        w.queue.read_cut_result.return_value = [(3.0, 5, False)]
        result = w.cooperative_solve(BRANCH, 4, ceiling=2.4)
        self.assertEqual(result, (OVER_ERD_LIMIT, 3.0, None, False))
        w.queue.create_branch.assert_not_called()
        w.queue.add_cut_reuse_miss.assert_not_called()

    def test_satisfying_tainted_cut_joins_taint(self):
        # The cut's own proof involved the remaining-depth floor, so a
        # consumer that reuses it must not report an untainted result.
        w = self._worker()
        w.queue.read_cut_result.return_value = [(3.0, 5, True)]
        result = w.cooperative_solve(BRANCH, 4, ceiling=2.4)
        self.assertEqual(result, (OVER_ERD_LIMIT, 3.0, None, True))

    def test_satisfying_cut_survives_ulp_noise(self):
        # The stored cut bound and the wanted ceiling are the same rational
        # (k / len(BRANCH)) but differ by one ULP in float64; erd_ge must
        # treat them as equal rather than triggering a re-solve.
        w = self._worker()
        n_words = len(BRANCH)
        stored_bound = 13 / n_words
        wanted_ceiling = math.nextafter(stored_bound, math.inf)
        self.assertNotEqual(stored_bound, wanted_ceiling)
        w.queue.read_cut_result.return_value = [(stored_bound, 5, False)]
        result = w.cooperative_solve(BRANCH, 4, ceiling=wanted_ceiling)
        self.assertEqual(result, (OVER_ERD_LIMIT, stored_bound, None, False))
        w.queue.create_branch.assert_not_called()
        w.queue.add_cut_reuse_miss.assert_not_called()

    def test_cut_at_smaller_budget_is_a_miss(self):
        # The bound was proven at budget 3; at budget 4 more strategies exist,
        # so it proves nothing — logged as a reuse miss and re-solved.
        w = self._worker()
        w.queue.read_cut_result.return_value = [(3.0, 3, False)]
        w.cooperative_solve(BRANCH, 4, ceiling=2.5)
        w.queue.add_cut_reuse_miss.assert_called_once_with(
            mock.ANY, len(BRANCH), 4, 2.5, 3.0, 3)
        w.queue.create_branch.assert_called_once()

    def test_cut_below_wanted_ceiling_is_a_miss(self):
        w = self._worker()
        w.queue.read_cut_result.return_value = [(2.0, 5, False)]
        w.cooperative_solve(BRANCH, 4, ceiling=2.4)
        w.queue.add_cut_reuse_miss.assert_called_once_with(
            mock.ANY, len(BRANCH), 4, 2.4, 2.0, 5)

    def test_exact_consumer_never_satisfied_by_cut(self):
        w = self._worker()
        w.queue.read_cut_result.return_value = [(3.0, 5, False)]
        w.cooperative_solve(BRANCH, 4)   # no ceiling: exact required
        w.queue.add_cut_reuse_miss.assert_called_once_with(
            mock.ANY, len(BRANCH), 4, None, 3.0, 5)
        w.queue.create_branch.assert_called_once()

    def test_ceiling_stored_on_created_branch(self):
        w = self._worker()
        w.cooperative_solve(BRANCH, 4, ceiling=2.5)
        self.assertAlmostEqual(
            w.queue.create_branch.call_args.kwargs["ceiling"], 2.5)

    def test_pending_row_suppresses_ceiling(self):
        w = self._worker()
        w.queue.has_pending_row.return_value = True
        w.cooperative_solve(BRANCH, 4, ceiling=2.5)
        self.assertIsNone(w.queue.create_branch.call_args.kwargs["ceiling"])

    def test_join_refused_when_existing_ceiling_tighter(self):
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": 2.0, "budget": 4}
        self.assertIsNone(w.cooperative_solve(BRANCH, 4, ceiling=2.4))

    def test_join_refused_for_exact_consumer_on_ceilinged_branch(self):
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": 2.0, "budget": 4}
        self.assertIsNone(w.cooperative_solve(BRANCH, 4))

    def test_join_refused_when_existing_budget_smaller(self):
        # The waiter regression from review: a budget-5 consumer must not wait
        # on a budget-4 branch — its cut or loss proves nothing at budget 5,
        # and the deleted-branch exit would misreport a loss.
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 4}
        self.assertIsNone(w.cooperative_solve(BRANCH, 5, ceiling=2.5))
        self.assertIsNone(w.cooperative_solve(BRANCH, 5))

    def test_join_refused_when_existing_budget_larger(self):
        # A tainted exact solved at budget 5 is not reusable at budget 4, so
        # a budget-4 waiter on a budget-5 branch can also fall through the
        # deleted-branch exit to a spurious loss.
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 5}
        self.assertIsNone(w.cooperative_solve(BRANCH, 4, ceiling=2.5))

    def test_join_allowed_when_existing_ceiling_looser(self):
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": 3.0, "budget": 4}
        result = w.cooperative_solve(BRANCH, 4, ceiling=2.4)
        self.assertIsNotNone(result)   # proceeded to the loop (cancelled out)

    def test_join_allowed_when_existing_ceiling_is_ulp_apart(self):
        # The row's ceiling and ours are the same rational (k / len(BRANCH))
        # but differ by one ULP in float64; erd_ge must treat them as equal
        # rather than refusing a join that should succeed.
        w = self._worker()
        row_ceiling = 13 / len(BRANCH)
        ours = math.nextafter(row_ceiling, math.inf)
        self.assertNotEqual(row_ceiling, ours)
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": row_ceiling, "budget": 4}
        result = w.cooperative_solve(BRANCH, 4, ceiling=ours)
        self.assertIsNotNone(result)   # proceeded to the loop (cancelled out)

    def test_join_allowed_on_exact_branch(self):
        w = self._worker()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 4}
        result = w.cooperative_solve(BRANCH, 4, ceiling=2.5)
        self.assertIsNotNone(result)

    def test_compatible_raced_branch_adopts_selected_opener(self):
        w = self._worker()
        parent_key = b"parent-branch"
        w._work_context = _context(
            branch_key=parent_key, opener_work_id=41, opener_priority=1,
            opener="crane", opener_pattern=42)
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 4}

        w.cooperative_solve(BRANCH, 4, ceiling=2.5)

        w.queue.attach_branch_opener_work.assert_called_once_with(
            ScoreCache.encode_subset(BRANCH), 41, 4, 2.5, len(BRANCH), 42,
            parent_key)

    def test_waiter_refreshes_a_cached_loss_miss_before_branch_deletion(self):
        w = self._worker()
        w._stop_requested = False
        w.score_cache.read_loss.side_effect = (None, None, 3)
        w.queue.get_branch.return_value = None
        result = w.cooperative_solve(BRANCH, 3, ceiling=3.25)
        self.assertEqual(
            result, (OVER_DEPTH_BUDGET, float('inf'), None, True))
        self.assertEqual(
            w.score_cache.read_loss.call_args_list[1].kwargs,
            {"refresh": True})
        self.assertEqual(
            w.score_cache.read_loss.call_args_list[2].kwargs,
            {"refresh": True})


class TestMaybeFinalizeTriage(unittest.TestCase):
    """maybe_finalize's exact / cut / loss triage from the branch meta."""

    def _worker(self, meta, nodes_spent=500):
        w = _bare_worker()
        w.score_cache = mock.MagicMock()
        w.rcache = mock.MagicMock()
        w.queue.branch_done_candidates.return_value = len(CANDIDATES)
        w.queue.try_finalize_branch.return_value = True
        w.queue.read_branch_meta.return_value = meta
        w.queue.get_branch.return_value = {
            "nodes_spent": nodes_spent,
            "infeasible_candidates": 0,
            "infeasible_nodes": 0,
            "created_at": 100,
            "finalized_at": 200,
            "first_best_at": 150,
            "nodes_at_first_best": 40,
            "spine": "SALET -g-g-",
        }
        w.queue.get_pending_branch.return_value = None
        w.queue.finalize_bundle_stats.return_value = (None, None, None, None)
        return w

    def test_cut_publishes_bound_and_never_caches(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 4, 2.5, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.queue.add_cut_result.assert_called_once_with(
            key, 4, 2.5, tainted=False)
        w.score_cache.write.assert_not_called()
        w.score_cache.write_loss.assert_not_called()
        w.queue.requeue_pending.assert_called_once_with(key)
        w.queue.mark_done.assert_not_called()
        w.queue.delete_branch.assert_called_once_with(key)
        log_kwargs = w.queue.add_branch_finalize_log.call_args.kwargs
        self.assertEqual(log_kwargs["outcome"], "cut")
        self.assertAlmostEqual(log_kwargs["ceiling"], 2.5)
        # The exact solve never ran: the node count is right-censored and must
        # not fold into the completed-solve cost model.
        w.queue.update_cost_model.assert_not_called()
        sample = w.queue.add_cost_sample.call_args
        self.assertEqual(sample.args[3], "cut")
        self.assertEqual(sample.kwargs["censored"], 1)

    def test_tainted_branch_publishes_tainted_cut(self):
        # meta's tainted field (index 3) is the branch's own floor taint; a
        # ceilinged solve that hit the floor must publish a tainted cut so a
        # consumer joins it rather than treating the bound as unconstrained.
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, True, 4, 2.5, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.queue.add_cut_result.assert_called_once_with(
            key, 4, 2.5, tainted=True)

    def test_ceiling_above_budget_caches_loss_and_retires_pending_row(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 3, 3.111111112, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_called_once_with(key, ERD_ALL, 3)
        w.queue.add_cut_result.assert_not_called()
        w.queue.complete_pending_for_loss.assert_called_once_with(
            key, 3, ROOT_BUDGET)
        w.queue.requeue_pending.assert_not_called()
        log_kwargs = w.queue.add_branch_finalize_log.call_args.kwargs
        self.assertEqual(log_kwargs["outcome"], "loss")
        self.assertAlmostEqual(log_kwargs["ceiling"], 3.111111112)

    def test_ceiling_equal_to_budget_remains_an_ordinary_cut(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 3, 3.0, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_not_called()
        w.queue.add_cut_result.assert_called_once_with(key, 3, 3.0, tainted=False)
        w.queue.requeue_pending.assert_called_once_with(key)

    def test_tainted_ceiling_above_budget_still_caches_loss(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, True, 3, 3.111111112, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_called_once_with(key, ERD_ALL, 3)
        w.queue.add_cut_result.assert_not_called()

    def test_ceiling_proven_loss_completes_pending_work_by_budget(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 3, 3.111111112, True))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_called_once_with(key, ERD_ALL, 3)
        w.queue.complete_pending_for_loss.assert_called_once_with(
            key, 3, ROOT_BUDGET)

    def test_loss_path_unchanged_without_cut_flag(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 4, None, False))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_called_once()
        w.queue.add_cut_result.assert_not_called()
        w.queue.complete_pending_for_loss.assert_called_once_with(
            key, 4, ROOT_BUDGET)
        log_kwargs = w.queue.add_branch_finalize_log.call_args.kwargs
        self.assertEqual(log_kwargs["outcome"], "loss")
        self.assertIsNone(log_kwargs["ceiling"])

    def test_exhaustive_loss_completes_pending_work_by_budget(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 3, None, False))
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write_loss.assert_called_once_with(key, ERD_ALL, 3)
        w.queue.complete_pending_for_loss.assert_called_once_with(
            key, 3, ROOT_BUDGET)

    def test_pending_completion_failure_still_runs_cleanup(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker((None, None, None, False, 3, None, False))
        w.queue.complete_pending_for_loss.side_effect = RuntimeError("boom")
        w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.queue.requeue_pending.assert_called_once_with(key)
        w.queue.delete_branch.assert_called_once_with(key)

    def test_exact_below_ceiling_caches_as_usual(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker(("crane", 1.8, 2, False, 4, 2.5, False))
        with mock.patch.object(erd_swarm, "cache_all_scores"):
            w.maybe_finalize(key, BRANCH, len(CANDIDATES))
        w.score_cache.write.assert_called_once()
        w.queue.add_cut_result.assert_not_called()
        w.queue.mark_done.assert_called_once_with(key)
        log_kwargs = w.queue.add_branch_finalize_log.call_args.kwargs
        self.assertEqual(log_kwargs["outcome"], "exact")
        self.assertAlmostEqual(log_kwargs["ceiling"], 2.5)

    def test_exact_direct_completion_is_snapshotted_before_branch_cleanup(self):
        key = ScoreCache.encode_subset(BRANCH)
        w = self._worker(("crane", 1.8, 2, False, 4, None, False))
        w.queue.mark_done.return_value = ["salet"]
        w.queue.delete_branch.return_value = []
        w._snapshot_completed_openers = mock.MagicMock()

        with mock.patch.object(erd_swarm, "cache_all_scores"):
            w.maybe_finalize(key, BRANCH, len(CANDIDATES))

        w._snapshot_completed_openers.assert_called_once_with(["salet"])


class TestMidLoopPublisherCeiling(unittest.TestCase):
    """check()'s ceiling handling on overrun handoff: a frame still riding its
    entry ceiling publishes a ceilinged branch; prefix marks happen only when
    the surviving row's ceiling makes them sound."""

    def _overrunning(self, predicted=10):
        w = _bare_worker()
        w.queue.get_cost_typical.return_value = predicted
        w.queue.has_pending_row.return_value = False
        w.queue.create_branch.return_value = True
        w.cooperative_solve = mock.MagicMock(
            return_value=(SOLVED, 2.0, 3, False))
        pub = erd_swarm._MidLoopPublisher(w)
        token = pub.enter(BRANCH[:6], budget=5)
        # Past the proportionate trigger AND the absolute break-even gate.
        w._nodes = erd_swarm.PUBLISH_THRESHOLD_BOOTSTRAP + 1
        return pub, w, token

    def test_ceilinged_frame_publishes_ceilinged_branch(self):
        pub, w, token = self._overrunning()
        pub.check(token, CANDIDATES, 1, None, 2.4, None, 5)
        self.assertAlmostEqual(
            w.queue.create_branch.call_args.kwargs["ceiling"], 2.4)
        w.queue.mark_claims_done.assert_called_once()
        w.cooperative_solve.assert_called_once_with(
            BRANCH[:6], 5, ceiling=2.4)

    def test_exact_frame_publishes_exact_branch(self):
        pub, w, token = self._overrunning()
        pub.check(token, CANDIDATES, 1, None, float('inf'), None, 5)
        self.assertIsNone(w.queue.create_branch.call_args.kwargs["ceiling"])
        w.queue.mark_claims_done.assert_called_once()
        w.cooperative_solve.assert_called_once_with(
            BRANCH[:6], 5, ceiling=float('inf'))

    def test_achieved_best_seeds_and_publishes_exact(self):
        pub, w, token = self._overrunning()
        pub.check(token, CANDIDATES, 1, "crane", 1.8, 4, 5)
        self.assertIsNone(w.queue.create_branch.call_args.kwargs["ceiling"])
        w.queue.update_branch_best.assert_called_once_with(
            ScoreCache.encode_subset(BRANCH[:6]), "crane", 1.8, 4)
        w.queue.mark_claims_done.assert_called_once()
        w.cooperative_solve.assert_called_once_with(
            BRANCH[:6], 5, ceiling=float('inf'))

    def test_pending_row_suppresses_ceiling_and_prefix_marks(self):
        pub, w, token = self._overrunning()
        w.queue.has_pending_row.return_value = True
        pub.check(token, CANDIDATES, 1, None, 2.5, None, 5)
        self.assertIsNone(w.queue.create_branch.call_args.kwargs["ceiling"])
        # The prefix was priced against the ceiling; in an exact branch those
        # price-outs do not hold, so the marks are skipped and redone there.
        w.queue.mark_claims_done.assert_not_called()

    def test_skipped_prefix_marks_do_not_transfer_budget_taint(self):
        pub, w, token = self._overrunning()
        w.queue.has_pending_row.return_value = True

        result = pub.check(
            token, CANDIDATES, 1, None, 2.5, None, 5,
            prefix_budget_tainted=True)

        w.queue.mark_branch_tainted.assert_not_called()
        w.queue.mark_claims_done.assert_not_called()
        self.assertEqual(result, (SOLVED, 2.0, 3, False))

    def test_race_to_exact_branch_skips_prefix_marks(self):
        pub, w, token = self._overrunning()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 5}
        pub.check(token, CANDIDATES, 1, None, 2.5, None, 5)
        w.queue.mark_claims_done.assert_not_called()

    def test_compatible_race_adopts_selected_opener(self):
        pub, w, token = self._overrunning()
        parent_key = b"parent-branch"
        w._work_context = _context(
            branch_key=parent_key, opener_work_id=41, opener_priority=1,
            opener="crane", opener_pattern=42)
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": None, "budget": 5}

        pub.check(token, CANDIDATES, 1, "crane", 1.8, None, 5)

        w.queue.attach_branch_opener_work.assert_called_once_with(
            ScoreCache.encode_subset(BRANCH[:6]), 41, 5, None,
            len(BRANCH[:6]), 42, parent_key)

    def test_race_to_other_budget_skips_marks_and_seed(self):
        # Everything the prefix proved, it proved at the frame's budget: a
        # raced row at another budget takes neither the done-marks nor the
        # achieved-best seed.
        pub, w, token = self._overrunning()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": 2.5, "budget": 4}
        pub.check(token, CANDIDATES, 1, "crane", 1.8, None, 5)
        w.queue.mark_claims_done.assert_not_called()
        w.queue.update_branch_best.assert_not_called()

    def test_race_to_tighter_ceiling_declines_and_disarms(self):
        # A raced row with a strictly tighter ceiling proves too little to
        # join, so the frame solves inline — writing nothing.  The prefix
        # marks would be sound for that row in isolation, but pouring them
        # into a branch this frame will not join repeats on every later
        # iteration (the overrun stays true), which is the write storm the
        # armed flag exists to prevent.
        pub, w, token = self._overrunning()
        w.queue.create_branch.return_value = False
        w.queue.get_branch.return_value = {"ceiling": 2.0, "budget": 5}
        self.assertIsNone(pub.check(token, CANDIDATES, 1, None, 2.4, None, 5))
        self.assertFalse(token[5])
        w.queue.mark_claims_done.assert_not_called()
        w.cooperative_solve.assert_not_called()

    def test_race_to_ulp_apart_ceiling_joins(self):
        # The raced row's ceiling and ours are the same rational one ULP
        # apart in float64; erd_ge must treat them as equal so the frame
        # still joins rather than declining as though the row's ceiling
        # were strictly tighter.
        pub, w, token = self._overrunning()
        w.queue.create_branch.return_value = False
        row_ceiling = 13 / len(BRANCH[:6])
        ours = math.nextafter(row_ceiling, math.inf)
        self.assertNotEqual(row_ceiling, ours)
        w.queue.get_branch.return_value = {"ceiling": row_ceiling, "budget": 5}
        result = pub.check(token, CANDIDATES, 1, None, ours, None, 5)
        self.assertIsNotNone(result)
        w.cooperative_solve.assert_called_once_with(BRANCH[:6], 5, ceiling=ours)

    def test_sound_to_mark_survives_ulp_noise(self):
        # The raced row's ceiling and this frame's own best_erd are the same
        # rational one ULP apart in float64; erd_ge must treat them as equal
        # so the prefix marks are still recorded as sound rather than
        # skipped.
        pub, w, token = self._overrunning()
        w.queue.create_branch.return_value = False
        best_erd = 13 / len(BRANCH[:6])
        row_ceiling = math.nextafter(best_erd, math.inf)
        self.assertNotEqual(best_erd, row_ceiling)
        w.queue.get_branch.return_value = {"ceiling": row_ceiling, "budget": 5}
        pub.check(token, CANDIDATES, 1, None, best_erd, None, 5)
        w.queue.mark_claims_done.assert_called_once()


class TestCeilingFinalizeIntegration(unittest.TestCase):
    """End-to-end triage against real queue/cache files: a ceilinged branch
    whose candidates all priced out finalizes as a CUT (bound delivered via
    cut_results, nothing cached); one with a best below the ceiling finalizes
    exact (cached, no cut row)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        answer_file = os.path.join(self._tmp.name, "answers.txt")
        words_file = os.path.join(self._tmp.name, "words.txt")
        with open(answer_file, "w") as f:
            f.write("\n".join(BRANCH) + "\n")
        with open(words_file, "w") as f:
            f.write("\n".join(CANDIDATES) + "\n")
        for attr, path in [("ANSWER_FILE", answer_file),
                           ("WORDS_FILE", words_file)]:
            p = mock.patch.object(erd_swarm, attr, path)
            p.start()
            self.addCleanup(p.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        self.key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()

    def _complete_all(self, q, cut, budget=4, ceiling=2.5,
                      queue_user_request=False):
        n = len(CANDIDATES)
        q.create_branch(self.key, len(BRANCH), n, budget=budget, ceiling=ceiling)
        if queue_user_request:
            q.add_pending_many([(self.key, len(BRANCH), 0, "crane", 0)])
        opener_work_id = (q.opener_work_rows()[0]["opener_work_id"]
                          if queue_user_request else None)
        order = list(range(n))
        _, indices, _ = q.claim_next_bundle(
            self.key, "other", n, order, [0.0] * n,
            small_count=n, count_cap=n,
            expected_opener_work_id=opener_work_id,
            expected_opener_priority=0 if queue_user_request else None)
        for idx in indices:
            q.complete_candidate(self.key, idx)
        if cut:
            q.mark_branch_cut(self.key)

    def test_all_priced_out_finalizes_as_cut(self):
        q = ERDQueue(self.queue_path)
        self._complete_all(q, cut=True)
        q.close()
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.maybe_finalize(self.key, BRANCH, len(CANDIDATES))
        finally:
            w.close()
        q = ERDQueue(self.queue_path)
        self.assertEqual(q.read_cut_result(self.key), [(2.5, 4, False)])
        self.assertIsNone(q.get_branch(self.key))
        row = q._conn.execute(
            "SELECT outcome, ceiling FROM telemetry.branch_finalize_log"
        ).fetchone()
        self.assertEqual(row["outcome"], "cut")
        q.close()
        sc = ScoreCache(self.cache_path, BRANCH)
        self.assertIsNone(sc.read_with_depth(self.key, ERD_ALL))
        sc.close()

    def test_best_below_ceiling_finalizes_exact(self):
        q = ERDQueue(self.queue_path)
        self._complete_all(q, cut=False)
        q.update_branch_best(self.key, "crane", 1.5, max_depth=2)
        q.close()
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.maybe_finalize(self.key, BRANCH, len(CANDIDATES))
        finally:
            w.close()
        q = ERDQueue(self.queue_path)
        self.assertEqual(q.read_cut_result(self.key), [])
        row = q._conn.execute(
            "SELECT outcome, ceiling FROM telemetry.branch_finalize_log"
        ).fetchone()
        self.assertEqual(row["outcome"], "exact")
        self.assertAlmostEqual(row["ceiling"], 2.5)
        q.close()
        sc = ScoreCache(self.cache_path, BRANCH)
        best = sc.read_with_depth(self.key, ERD_ALL)
        self.assertIsNotNone(best)
        sc.close()

    def test_ceiling_proven_loss_reopens_as_a_durable_loss(self):
        q = ERDQueue(self.queue_path)
        self._complete_all(q, cut=True, budget=3, ceiling=3.25)
        q.close()
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.maybe_finalize(self.key, BRANCH, len(CANDIDATES))
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        self.assertEqual(q.read_cut_result(self.key), [])
        row = q._conn.execute(
            "SELECT outcome, ceiling, budget FROM telemetry.branch_finalize_log"
        ).fetchone()
        self.assertEqual((row["outcome"], row["ceiling"], row["budget"]),
                         ("loss", 3.25, 3))
        q.close()

        w = _BranchWorker(1, self.cache_path, self.queue_path, None)
        try:
            self.assertEqual(
                w.cooperative_solve(BRANCH, 3),
                (OVER_DEPTH_BUDGET, float("inf"), None, True),
            )
            self.assertEqual(
                w.cooperative_solve(BRANCH, 2),
                (OVER_DEPTH_BUDGET, float("inf"), None, True),
            )
        finally:
            w.close()
        q = ERDQueue(self.queue_path)
        self.assertIsNone(q.get_branch(self.key))
        q.close()

    def test_exhaustive_loss_preserves_a_larger_budget_user_request(self):
        q = ERDQueue(self.queue_path)
        self._complete_all(
            q, cut=False, budget=3, ceiling=None, queue_user_request=True)
        q.close()
        w = _BranchWorker(0, self.cache_path, self.queue_path, None)
        try:
            w.maybe_finalize(self.key, BRANCH, len(CANDIDATES))
        finally:
            w.close()

        q = ERDQueue(self.queue_path)
        self.assertEqual(q.get_pending_branch(self.key)["status"], "pending")
        self.assertIsNone(q.get_branch(self.key))
        q.close()
        sc = ScoreCache(self.cache_path, BRANCH)
        self.assertEqual(sc.read_loss(self.key, ERD_ALL), 3)
        sc.close()


class TestOneWorkerPerBranch(unittest.TestCase):
    """Work selection takes one worker to a branch: an unoccupied branch is
    always preferred, a branch with one worker is joined only as a last
    resort, and a branch nobody holds a claim on is claimable again.

    Occupancy throughout is unfinished candidate claims, which is what the
    scheduler reads.  Heartbeats are reporting state and decide nothing here.
    """

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
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        ScoreCache(self.cache_path, BRANCH).close()
        self.queue = erd_queue.ERDQueue(self.queue_path)
        self.addCleanup(self.queue.close)

    def _write(self, name, words):
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as f:
            f.write("\n".join(words) + "\n")
        return p

    def _worker(self, worker_id=0, **kwargs):
        w = _BranchWorker(worker_id, self.cache_path, self.queue_path, None,
                          **kwargs)
        self.addCleanup(w.close)
        return w

    def _queue_opener(self, words_per_branch, opener="crane", priority=0):
        """Queue one opener's branches as a single request, so every branch
        shares the opener work the way an opener's response groups do."""
        keys = [ScoreCache.encode_subset(w) for w in words_per_branch]
        self.queue.add_pending_many([
            (key, len(words), priority, opener, 0)
            for key, words in zip(keys, words_per_branch)])
        return keys

    def _promote(self, opener=None):
        """Open the next pending branch of an opener the way a worker does."""
        opener_work_id = None
        if opener is not None:
            opener_work_id = self.queue._conn.execute(
                "SELECT opener_work_id FROM opener_work WHERE opener = ?",
                (opener,)).fetchone()[0]
        claimed = self.queue.claim_next("promoter", opener_work_id)
        self.queue.create_branch(
            claimed["branch_key"], claimed["n_words"], len(CANDIDATES),
            budget=ROOT_BUDGET, priority=claimed["priority"],
            opener=claimed["opener"],
            opener_pattern=claimed["opener_pattern"],
            opener_work_id=claimed["opener_work_id"])
        return bytes(claimed["branch_key"]), claimed["opener_work_id"]

    def _occupy(self, branch_key, worker_id):
        """Put a worker on a branch the way production does: an unfinished
        candidate claim, written by the same transaction that hands out the
        bundle.  That row IS the occupancy the scheduler reads."""
        w = self._worker(worker_id, small_count=1, count_cap=1)
        owner = self.queue.owner_row_for_branch(branch_key)
        claim = w._claim_bundle(
            branch_key, owner["n_candidates"], decode_subset(branch_key),
            w._branch_budget(owner),
            expected_opener_work_id=owner["opener_work_id"],
            expected_opener_priority=owner["owner_priority"])
        self.assertIsNotNone(claim, "fixture worker took no claim to hold")
        _bundle_id, indices, _forced = claim
        return w, list(indices)

    def _drain_claims(self, branch_key, worker_id):
        """Claim every remaining candidate of a branch, leaving it with no
        bundle for anyone else — a branch that is occupied AND offers nothing
        to fall back on."""
        w = self._worker(worker_id, small_count=len(CANDIDATES),
                         count_cap=len(CANDIDATES))
        owner = self.queue.owner_row_for_branch(branch_key)
        while w._claim_bundle(
                branch_key, owner["n_candidates"], decode_subset(branch_key),
                w._branch_budget(owner),
                expected_opener_work_id=owner["opener_work_id"],
                expected_opener_priority=owner["owner_priority"]) is not None:
            pass
        return w

    def _claimed_key(self, work):
        self.assertIsNotNone(work)
        return bytes(work[1]["branch_key"])

    # -- occupancy -------------------------------------------------------

    def test_free_branch_is_preferred_over_an_occupied_one(self):
        self._queue_opener([BRANCH, BRANCH[:3]])
        busy_key, _ = self._promote()
        free_key, _ = self._promote()
        self._occupy(busy_key, 9)

        self.assertEqual(self._claimed_key(self._worker().claim_one()),
                         free_key)

    def test_selection_skips_an_occupied_branch_without_attempting_a_claim(self):
        """The two occupancy checks divide differently than they look.

        The claim transaction is what ENFORCES the cap — disable this filter
        and an occupied branch is still refused, just one write transaction
        later.  What the filter buys is that cost: a worker choosing among
        branches does not open a BEGIN IMMEDIATE against each occupied one
        before finding free work.  So the thing to assert is the attempt, not
        the outcome.
        """
        self._queue_opener([BRANCH, BRANCH[:3]])
        busy_key, _ = self._promote()
        free_key, _ = self._promote()
        self._occupy(busy_key, 9)
        w = self._worker()
        attempted = []
        claim_bundle = w._claim_bundle

        def spy(branch_key, *args, **kwargs):
            attempted.append(bytes(branch_key))
            return claim_bundle(branch_key, *args, **kwargs)

        w._claim_bundle = spy

        self.assertEqual(self._claimed_key(w.claim_one()), free_key)
        self.assertNotIn(busy_key, attempted)
        self.assertIn(free_key, attempted)

    def test_a_workers_own_claims_do_not_make_a_branch_look_occupied(self):
        """A worker part-way through a branch must keep claiming from it.
        Counting its own unfinished claims would push it off its own branch."""
        # Two free branches, the worker's own sorting first (more answer words
        # at equal priority), so a wrong answer is visible as a move.
        self._queue_opener([BRANCH, BRANCH[:3]])
        own_key, _ = self._promote()
        other_key, _ = self._promote()
        self.assertNotEqual(own_key, other_key)
        w, _ = self._occupy(own_key, 0)

        self.assertEqual(self._claimed_key(w.claim_one()), own_key)

    def test_branch_is_claimable_again_once_its_claims_are_released(self):
        """Occupancy is held by unfinished claims, not by a branch having ever
        been claimed.  A worker that finished its bundle and moved on leaves
        the branch immediately claimable — this is how a partly-solved branch
        resumes, and a filter that blocked it would starve the swarm while
        passing every other test here."""
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        holder, _ = self._occupy(only_key, 8)
        self._occupy(only_key, 9)
        # Two holders: at the cap, so nothing else may enter.
        self.assertIsNone(self._worker(3).claim_one())

        self.queue.reclaim_claims_of_worker(holder.name)

        self.assertEqual(self._claimed_key(self._worker(3).claim_one()),
                         only_key)

    def test_finished_claims_do_not_reserve_a_branch(self):
        """Only UNFINISHED claims are occupancy.  Counting completed ones
        would make every branch anyone ever worked look permanently taken."""
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        _holder, indices = self._occupy(only_key, 8)
        self.assertEqual(self.queue.claim_holders_by_branch()[only_key], 1)

        for idx in indices:
            self.queue.complete_candidate(only_key, idx)

        self.assertNotIn(only_key, self.queue.claim_holders_by_branch())

    def test_reclaiming_a_crashed_workers_claims_frees_its_branch(self):
        """Occupancy carries no liveness test of its own — it does not need
        one.  A crashed worker's unfinished claims are freed by the queue's
        existing reclaim paths, and the branch's occupancy falls with them.

        This pins the dependency rather than new code: if reclaim stopped
        freeing claims, branches held by dead workers would never reopen.
        """
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        self._occupy(only_key, 8)
        self._occupy(only_key, 9)
        self.assertEqual(self.queue.claim_holders_by_branch()[only_key], 2)

        # Claims taken an hour ago by workers that have not heartbeat since:
        # reclaim_stale_claims holds a floor on claim age, so a claim made in
        # this same second is deliberately not yet eligible.
        self.queue._conn.execute(
            "UPDATE candidate_claims SET claimed_at = ? WHERE done = 0",
            (int(time.time()) - 3600,))
        self.queue._conn.commit()
        freed = self.queue.reclaim_stale_claims(heartbeat_timeout_seconds=60)

        self.assertGreater(freed, 0)
        self.assertNotIn(only_key, self.queue.claim_holders_by_branch())

    # -- last-resort pairing ---------------------------------------------

    def test_second_worker_joins_when_no_branch_is_free(self):
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        self._occupy(only_key, 9)

        self.assertEqual(self._claimed_key(self._worker().claim_one()),
                         only_key)

    def test_third_worker_never_joins(self):
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        self._occupy(only_key, 8)
        self._occupy(only_key, 9)

        self.assertIsNone(self._worker(3).claim_one())

    def test_pairing_loses_to_a_promotable_pending_branch(self):
        """A second worker on a branch is worth a fraction of the same worker
        on a branch of its own, so every other path is tried first."""
        keys = self._queue_opener([BRANCH, BRANCH[:3]])
        busy_key, _ = self._promote()
        self._occupy(busy_key, 9)

        claimed = self._claimed_key(self._worker().claim_one())

        self.assertNotEqual(claimed, busy_key)
        self.assertIn(claimed, keys)

    def test_cap_holds_when_two_workers_select_the_same_branch_together(self):
        """Selection filters on occupancy read before the claim, so two
        workers can both see the same branch as free.  The claim transaction
        re-counts and only one of them gets in."""
        self._queue_opener([BRANCH])
        only_key, _ = self._promote()
        self._occupy(only_key, 8)
        third = self._worker(9, small_count=1, count_cap=1)
        owner = self.queue.owner_row_for_branch(only_key)
        # A stale reading: the branch looked free when this worker chose it.
        stale_occupancy = {}

        self.assertIsNone(third._claim_active_branch(
            self.queue.branches_in_progress(owner["opener_work_id"]),
            owner["opener_work_id"], SCHEDULING_ROLE_PREFERRED,
            occupancy=stale_occupancy, max_other_workers=0))

    # -- widening and preemption -----------------------------------------

    def test_occupied_opener_promotes_another_of_its_own_pending_branches(self):
        """Widen the opener already in flight rather than stacking onto the
        branch of it that is running."""
        self._queue_opener([BRANCH, BRANCH[:3]])
        busy_key, opener_work_id = self._promote()
        self._occupy(busy_key, 9)

        work = self._worker().claim_one()

        self.assertNotEqual(self._claimed_key(work), busy_key)
        self.assertEqual(work[1]["opener_work_id"], opener_work_id)

    def test_worker_takes_a_higher_priority_opener_at_the_next_claim(self):
        """Preemption needs no cancellation: selection reruns at every claim
        boundary and opener_work_candidates() is ordered by requested
        priority, so newly-queued higher-priority work wins the next claim."""
        self._queue_opener([BRANCH[:3]], opener="slate", priority=10)
        low_key, _ = self._promote()
        w = self._worker()
        self.assertEqual(self._claimed_key(w.claim_one()), low_key)

        high_keys = self._queue_opener([BRANCH[:4]], opener="crane",
                                       priority=500)

        self.assertEqual(self._claimed_key(w.claim_one()), high_keys[0])

    def test_branch_left_for_higher_priority_work_keeps_its_progress(self):
        """A worker that moves to higher-priority work abandons nothing: the
        branch stays open with its completed claims and its discovered
        best_erd, so whoever resumes it continues rather than restarting."""
        self._queue_opener([BRANCH[:3]], opener="slate", priority=10)
        low_key, _ = self._promote()
        # One candidate per bundle, so the move happens mid-branch.
        w = self._worker(small_count=1, count_cap=1)
        work = w.claim_one()
        self.assertEqual(self._claimed_key(work), low_key)
        with w._entered(erd_swarm.WorkContext.from_branch_row(
                work[1], SCHEDULING_ROLE_PREFERRED)):
            w.evaluate_bundle(low_key, decode_subset(low_key),
                              work[1]["n_words"], work[2], work[3], work[4],
                              budget=ROOT_BUDGET)
        done_claims = self.queue.branch_done_candidates(low_key)
        self.assertGreater(done_claims, 0)
        before = self.queue.owner_row_for_branch(low_key)
        self.assertIsNotNone(before["best_erd"])

        high_keys = self._queue_opener([BRANCH[:4]], opener="crane",
                                       priority=500)
        self.assertEqual(self._claimed_key(w.claim_one()), high_keys[0])

        after = self.queue.owner_row_for_branch(low_key)
        self.assertEqual(after["status"], "open")
        self.assertEqual(after["best_erd"], before["best_erd"])
        self.assertEqual(self.queue.branch_done_candidates(low_key),
                         done_claims)

    def test_higher_priority_pending_only_opener_still_starts(self):
        """Issue #214: an opener with no active branch must still be able to
        start while other openers hold active ones."""
        self._queue_opener([BRANCH], opener="slate", priority=10)
        self._promote()
        high_keys = self._queue_opener([BRANCH[:4]], opener="crane",
                                       priority=500)

        self.assertEqual(self._claimed_key(self._worker().claim_one()),
                         high_keys[0])

    # -- widening from a blocked worker ----------------------------------

    def test_blocked_worker_widens_a_opener_whose_branches_are_all_worked(self):
        """_help_other_branch treats an opener as covered only when it has an
        unoccupied branch.  A opener whose open branches all have workers
        still needs its pending branches promoted to absorb another worker,
        which is what leaves an opener running one response group at a time
        with the rest waiting."""
        # Three branches under one opener: the one this worker is blocked on,
        # one another worker holds, and one still pending.  The occupied
        # branch must not count as covering the opener — if it does, the
        # promote loop is skipped and the worker joins it instead of opening
        # the pending one, which is the failure being fixed.
        self._queue_opener([BRANCH, BRANCH[:3], BRANCH[:4]])
        own_key, opener_work_id = self._promote()
        busy_key, _ = self._promote()
        self._occupy(busy_key, 9)
        self.assertEqual(
            len(self.queue.branches_in_progress(opener_work_id)), 2)

        w = self._worker(small_count=1, count_cap=1)
        self.assertTrue(w._help_other_branch(own_key))

        open_after = self.queue.branches_in_progress(opener_work_id)
        self.assertEqual(len(open_after), 3)
        self.assertEqual({b["opener_work_id"] for b in open_after},
                         {opener_work_id})

    def test_promoter_beaten_to_its_own_branch_does_not_join_it(self):
        """create_branch commits before the promoter claims, so between the two
        the branch is visible and unoccupied to every other worker.  Creating a
        branch is therefore no exemption from the cap: by the time the promoter
        claims, someone may already be on it, and claiming anyway would seat a
        second worker on a branch while this one still had a free sibling to
        open instead.

        Deterministic rather than threaded: the interleaving is forced by
        claiming from another worker inside create_branch, at exactly the point
        the real race occurs.
        """
        keys = self._queue_opener([BRANCH, BRANCH[:3]])
        w = self._worker(small_count=1, count_cap=1)
        create_branch = self.queue.create_branch
        stolen = []

        def create_then_let_a_rival_in(branch_key, *args, **kwargs):
            result = create_branch(branch_key, *args, **kwargs)
            if not stolen:                     # only the first promotion
                stolen.append(bytes(branch_key))
                self._occupy(bytes(branch_key), 9)
            return result

        with mock.patch.object(w.queue, "create_branch",
                               side_effect=create_then_let_a_rival_in):
            work = w.claim_one()

        self.assertEqual(len(stolen), 1)
        holders = self.queue.claim_holders_by_branch()
        self.assertEqual(holders.get(stolen[0]), 1,
                         "promoter took a second seat on the branch it created")
        for branch_key, count in holders.items():
            self.assertLessEqual(count, 1, f"{branch_key!r} has {count} holders")
        if work is not None:
            claimed = self._claimed_key(work)
            self.assertNotEqual(claimed, stolen[0])
            self.assertIn(claimed, keys)

    # -- dependency waits obey the same last-resort rule ------------------

    def _wait_on_an_occupied_dependency(self, worker, parent_key, child_words):
        """Drive cooperative_solve until its wait path decides, with the child
        branch taken by another worker the instant it is created.

        Shaped the way production reaches this loop: the worker is inside its
        parent branch's context and asks the engine to solve a sub-branch, so
        the child is created here rather than pre-built.  A rival claims it in
        the window between create_branch committing and the solver's own claim
        — the same window the promoter race lives in.

        Returns the claim attempts made.  What is asserted is the decision
        (which branch, at which cap, granted or not), not work completed:
        stopping the worker cancels an in-flight evaluation, so a granted claim
        can legitimately leave nothing done behind it.
        """
        attempts = []
        claim_bundle = worker._claim_bundle
        help_other_branch = worker._help_other_branch
        create_branch = worker.queue.create_branch
        taken = []

        def create_then_let_a_rival_in(branch_key, *args, **kwargs):
            result = create_branch(branch_key, *args, **kwargs)
            if not taken:
                taken.append(bytes(branch_key))
                self._occupy(bytes(branch_key), 9)
            return result

        def record(branch_key, *args, **kwargs):
            result = claim_bundle(branch_key, *args, **kwargs)
            attempts.append({
                "branch_key": bytes(branch_key),
                "max_other_workers": kwargs.get("max_other_workers"),
                "granted": result is not None,
            })
            if kwargs.get("max_other_workers") == MAX_WORKERS_PER_BRANCH - 1:
                worker._stop_requested = True    # the pairing decision is made
            return result

        def help_then_stop_if_it_worked(*args, **kwargs):
            did_work = help_other_branch(*args, **kwargs)
            if did_work:
                worker._stop_requested = True    # it chose other work
            return did_work

        owner = self.queue.owner_row_for_branch(parent_key)
        context = erd_swarm.WorkContext.from_branch_row(
            dict(owner), SCHEDULING_ROLE_PREFERRED)
        with worker._entered(context):
            with mock.patch.object(worker.queue, "create_branch",
                                   side_effect=create_then_let_a_rival_in):
                worker._claim_bundle = record
                worker._help_other_branch = help_then_stop_if_it_worked
                # budget + guess_depth = GAME_GUESSES, so a child of this
                # branch sits one below it.
                worker.cooperative_solve(child_words, owner["budget"] - 1)
        self.assertEqual(len(taken), 1, "no child branch was created")
        return attempts, taken[0]

    def test_dependency_wait_tries_help_other_branch_before_any_write(self):
        """The write-heavy liveness backstop (heartbeat force=True, then
        reclaim_stale_claims) must stay on the RARE path: reached only when
        _help_other_branch finds nothing, not on every iteration a legitimate
        pair is simply progressing.  Getting this backwards converts a rare
        write into a per-iteration one — exactly the regression this pins.
        """
        self._queue_opener([BRANCH, BRANCH[:4]])
        parent_key, _ = self._promote()
        free_key, _ = self._promote()
        w = self._worker(small_count=1, count_cap=1)

        calls = []
        reclaim = w.queue.reclaim_stale_claims
        help_other_branch = w._help_other_branch
        claim_bundle = w._claim_bundle
        create_branch = w.queue.create_branch
        taken = []

        def create_then_occupy(branch_key, *args, **kwargs):
            result = create_branch(branch_key, *args, **kwargs)
            if not taken:
                taken.append(bytes(branch_key))
                self._occupy(bytes(branch_key), 9)
            return result

        def record_claim(*a, **kw):
            attempt = claim_bundle(*a, **kw)
            if kw.get("max_other_workers") == 0 and attempt is None:
                calls.append("sole_worker_refused")
            return attempt

        # reclaim_stale_claims has no other legitimate caller anywhere in this
        # trace, so it is the one unambiguous signal that the write-heavy
        # fallback ran.  (Ordinary per-candidate heartbeats fire regardless —
        # evaluating the bundle _help_other_branch finds calls them as part of
        # normal progress reporting, which is correct and not what this pins.)
        def record_reclaim(*a, **kw):
            calls.append("reclaim")
            return reclaim(*a, **kw)

        def record_help(*a, **kw):
            calls.append("help")
            result = help_other_branch(*a, **kw)
            w._stop_requested = True   # one pass through the wait loop is enough
            return result

        w._claim_bundle = record_claim
        w.queue.reclaim_stale_claims = record_reclaim
        w._help_other_branch = record_help

        owner = self.queue.owner_row_for_branch(parent_key)
        context = erd_swarm.WorkContext.from_branch_row(
            dict(owner), SCHEDULING_ROLE_PREFERRED)
        with w._entered(context):
            with mock.patch.object(w.queue, "create_branch",
                                   side_effect=create_then_occupy):
                w.cooperative_solve(BRANCH[:3], owner["budget"] - 1)

        self.assertEqual(len(taken), 1, "no child branch was created")
        # The dependency is occupied, so the sole-worker claim is refused —
        # and a free branch exists, so help finds it on the first call and the
        # loop stops there, never reaching the write-heavy fallback.
        self.assertEqual(calls[:2], ["sole_worker_refused", "help"])
        self.assertNotIn("reclaim", calls)

    def test_occupied_dependency_loses_to_a_free_branch(self):
        """A worker blocked on a dependency another worker holds must take free
        work rather than seat itself second.  The hard cap alone would permit
        the pair; the last-resort rule is what forbids it while anything else
        is claimable."""
        self._queue_opener([BRANCH, BRANCH[:4]])
        parent_key, _ = self._promote()
        free_key, _ = self._promote()

        w = self._worker(small_count=1, count_cap=1)
        attempts, child_key = self._wait_on_an_occupied_dependency(
            w, parent_key, BRANCH[:3])

        paired = [a for a in attempts
                  if a["branch_key"] == child_key
                  and a["max_other_workers"] == MAX_WORKERS_PER_BRANCH - 1]
        self.assertEqual(paired, [],
                         "asked to pair onto the dependency while a branch "
                         "was free")
        self.assertEqual(self.queue.claim_holders_by_branch().get(child_key), 1)
        # The rival holds its claim unfinished throughout, so any completed
        # candidate on the child could only be this worker's.
        self.assertEqual(self.queue.branch_done_candidates(child_key), 0,
                         "worked the occupied dependency anyway")
        self.assertIsNotNone(self.queue.owner_row_for_branch(free_key))

    def test_dependency_pair_forms_when_nothing_else_is_claimable(self):
        """The rule is last resort, not never: with no other branch to take,
        a second worker on the dependency beats idling."""
        self._queue_opener([BRANCH])
        parent_key, _ = self._promote()
        # Every candidate of the parent is claimed, so it offers no bundle to
        # fall back on — occupying it alone would leave a pairing slot open and
        # the worker would take that instead of ever reaching the child.
        self._drain_claims(parent_key, 8)

        w = self._worker(small_count=1, count_cap=1)
        attempts, child_key = self._wait_on_an_occupied_dependency(
            w, parent_key, BRANCH[:3])

        granted_pair = [a for a in attempts
                        if a["branch_key"] == child_key
                        and a["max_other_workers"] == MAX_WORKERS_PER_BRANCH - 1
                        and a["granted"]]
        self.assertEqual(len(granted_pair), 1,
                         "no free work anywhere, yet no pair formed")
        # And it was tried as the sole worker first, never straight to a pair.
        child_attempts = [a for a in attempts if a["branch_key"] == child_key]
        self.assertEqual(child_attempts[0]["max_other_workers"], 0)

    def test_recursion_capped_help_does_not_authorize_pairing(self):
        """_help_other_branch returns False for two different reasons: it
        scanned everything and found nothing, or it refused to scan at all
        because _help_recursion_depth is already at MAX_HELP_RECURSION_DEPTH
        — its own docstring says the caller should poll at that depth, not
        treat it as a completed empty scan.  Conflating the two lets a worker
        pair onto an occupied dependency while a free branch sits untouched,
        reached whenever the worker happens to already be deep in nested help
        calls (issue #214's blocked-worker chains)."""
        self._queue_opener([BRANCH, BRANCH[:4]])
        parent_key, _ = self._promote()
        free_key, _ = self._promote()
        w = self._worker(small_count=1, count_cap=1)
        w._help_recursion_depth = erd_swarm.MAX_HELP_RECURSION_DEPTH

        create_branch = w.queue.create_branch
        taken = []

        def create_then_occupy(branch_key, *args, **kwargs):
            result = create_branch(branch_key, *args, **kwargs)
            if not taken:
                taken.append(bytes(branch_key))
                self._occupy(bytes(branch_key), 9)
            return result

        claim_bundle = w._claim_bundle

        def stop_after_one_wait_pass(*a, **kw):
            result = claim_bundle(*a, **kw)
            if kw.get("max_other_workers") == 0 and result is None:
                # The sole-worker attempt was just refused: whatever this
                # loop iteration decides next is the thing under test, so one
                # more pass is enough.
                w._stop_requested = True
            return result

        w._claim_bundle = stop_after_one_wait_pass
        owner = self.queue.owner_row_for_branch(parent_key)
        context = erd_swarm.WorkContext.from_branch_row(
            dict(owner), SCHEDULING_ROLE_PREFERRED)
        with w._entered(context):
            with mock.patch.object(w.queue, "create_branch",
                                   side_effect=create_then_occupy):
                w.cooperative_solve(BRANCH[:3], owner["budget"] - 1)

        self.assertEqual(len(taken), 1, "no child branch was created")
        self.assertEqual(self.queue.claim_holders_by_branch().get(taken[0]), 1,
                         "paired onto a recursion-capped dependency while a "
                         "branch was free")

    # -- branches with no opener ownership --------------------------------

    def _direct_branch(self, words):
        """A branch with no opener-work owner, as a queue upgraded while work
        was in flight leaves behind."""
        branch_key = ScoreCache.encode_subset(words)
        self.queue.create_branch(branch_key, len(words), len(CANDIDATES),
                                 budget=ROOT_BUDGET)
        return branch_key

    def test_unowned_branch_is_claimed_when_no_opener_can_supply_work(self):
        self._queue_opener([BRANCH])
        owned_key, _ = self._promote()
        self._occupy(owned_key, 8)
        self._occupy(owned_key, 9)
        direct_key = self._direct_branch(BRANCH[:3])
        w = self._worker()
        # A worker that has already seen opener work does not re-check the
        # unowned branches until every opener has been tried.
        w._opener_work_enabled = True

        self.assertEqual(self._claimed_key(w.claim_one()), direct_key)

    def test_unowned_branch_can_be_paired_as_a_last_resort(self):
        direct_key = self._direct_branch(BRANCH)
        self._occupy(direct_key, 9)

        self.assertEqual(self._claimed_key(self._worker().claim_one()),
                         direct_key)


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()

    def test_recursion_capped_help_does_not_authorize_pairing(self):
        """_help_other_branch returns False for two different reasons: it
        scanned everything and found nothing, or it refused to scan at all
        because _help_recursion_depth is already at MAX_HELP_RECURSION_DEPTH
        — its own docstring says the caller should poll at that depth, not
        treat it as a completed empty scan.  Conflating the two lets a worker
        pair onto an occupied dependency while a free branch sits untouched,
        reached only when the worker happens to already be deep in nested
        help calls (issue #214's blocked-worker chains)."""
        self._queue_opener([BRANCH, BRANCH[:4]])
        parent_key, _ = self._promote()
        free_key, _ = self._promote()
        w = self._worker(small_count=1, count_cap=1)
        w._help_recursion_depth = erd_swarm.MAX_HELP_RECURSION_DEPTH

        create_branch = w.queue.create_branch
        taken = []

        def create_then_occupy(branch_key, *args, **kwargs):
            result = create_branch(branch_key, *args, **kwargs)
            if not taken:
                taken.append(bytes(branch_key))
                self._occupy(bytes(branch_key), 9)
            return result

        owner = self.queue.owner_row_for_branch(parent_key)
        context = erd_swarm.WorkContext.from_branch_row(
            dict(owner), SCHEDULING_ROLE_PREFERRED)
        with w._entered(context):
            with mock.patch.object(w.queue, "create_branch",
                                   side_effect=create_then_occupy):
                w._stop_requested_after = 1  # see below
                orig_claim_bundle = w._claim_bundle
                calls = []
                def stop_after_first_wait_iteration(*a, **kw):
                    result = orig_claim_bundle(*a, **kw)
                    calls.append((kw.get("max_other_workers"), result is not None))
                    if len(calls) >= 3:
                        w._stop_requested = True
                    return result
                w._claim_bundle = stop_after_first_wait_iteration
                w.cooperative_solve(BRANCH[:3], owner["budget"] - 1)

        self.assertEqual(len(taken), 1, "no child branch was created")
        self.assertEqual(self.queue.claim_holders_by_branch().get(taken[0]), 1,
                         "paired onto a recursion-capped dependency while a "
                         "branch was free")
