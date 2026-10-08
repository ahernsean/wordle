"""Where swarm time goes: the worker time account, the per-branch
coordination total, and the reclaim and checkpoint-pause records."""

import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

import erd_search
import erd_swarm
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
from erd_swarm import _BranchWorker, _WorkerTimeAccount, ROOT_BUDGET
from wordle_engine import SOLVED
from tests.test_erd_swarm_unit import BRANCH, CANDIDATES, _bare_worker

ACTIVITY_COLUMNS = tuple(f"{activity}_millis"
                         for activity in _WorkerTimeAccount.ACTIVITIES)


class _SteppingClock:
    """A nanosecond clock that moves only when told to, or by `step` on
    every read."""

    def __init__(self, step_millis=0):
        self.now = 0
        self.step = int(step_millis * 1_000_000)

    def __call__(self):
        self.now += self.step
        return self.now

    def advance(self, millis):
        self.now += int(millis * 1_000_000)


def _account(clock):
    return _WorkerTimeAccount(clock=clock, wall_clock=lambda: 1_000.0)


class TestWorkerTimeAccount(unittest.TestCase):

    def test_nested_activities_are_charged_to_the_innermost_only(self):
        clock = _SteppingClock()
        account = _account(clock)
        clock.advance(10)
        with account.activity("evaluation"):
            clock.advance(40)
            with account.activity("wait_dependency"):
                clock.advance(20)
                with account.activity("scheduling"):
                    clock.advance(5)
                clock.advance(5)
            clock.advance(30)
        clock.advance(2)
        _started_at, interval_millis, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(interval_millis, 112)
        self.assertEqual(figures["evaluation_millis"], 70)
        self.assertEqual(figures["wait_dependency_millis"], 25)
        self.assertEqual(figures["scheduling_millis"], 5)
        self.assertEqual(figures["other_millis"], 12)
        self.assertEqual(sum(figures[c] for c in ACTIVITY_COLUMNS),
                         interval_millis)

    def test_sub_millisecond_spans_still_partition_the_interval(self):
        # Each span floors to 0 ms on its own; their total does not.
        clock = _SteppingClock()
        account = _account(clock)
        for _ in range(9):
            with account.activity("claiming"):
                clock.advance(0.7)
            clock.advance(0.2)
        _started_at, interval_millis, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(interval_millis, 8)
        self.assertEqual(figures["claiming_millis"], 6)
        self.assertEqual(figures["other_millis"], 2)
        self.assertEqual(sum(figures[c] for c in ACTIVITY_COLUMNS),
                         interval_millis)

    def test_other_absorbs_the_flooring_and_never_goes_negative(self):
        # Two 0.6 ms spans and nothing else: rounding each would claim 2 ms
        # of a 1 ms interval.
        clock = _SteppingClock()
        account = _account(clock)
        for activity in ("claiming", "finalizing"):
            with account.activity(activity):
                clock.advance(0.6)
        _started_at, interval_millis, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(interval_millis, 1)
        self.assertEqual(figures["other_millis"], 1)
        self.assertTrue(all(figures[c] >= 0 for c in ACTIVITY_COLUMNS))

    def test_an_activity_left_by_an_exception_stops_charging(self):
        clock = _SteppingClock()
        account = _account(clock)
        with self.assertRaises(RuntimeError):
            with account.activity("finalizing"):
                clock.advance(3)
                raise RuntimeError("boom")
        clock.advance(7)
        self.assertEqual(account.current, "other")
        _started_at, _interval, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(figures["finalizing_millis"], 3)
        self.assertEqual(figures["other_millis"], 7)

    def test_an_unknown_activity_is_refused(self):
        account = _account(_SteppingClock())
        with self.assertRaises(ValueError):
            with account.activity("napping"):
                pass

    def test_the_figures_name_every_column_the_table_has(self):
        _started_at, _interval, figures = _account(
            _SteppingClock()).close_interval(0, (0, 0, 0, 0))
        self.assertEqual(set(figures), set(ERDQueue.WORKER_TIME_COLUMNS))

    def test_a_new_interval_starts_from_zero(self):
        clock = _SteppingClock()
        account = _account(clock)
        account.candidates_evaluated = 4
        account.fruitless_scans = 2
        account.heartbeats_deferred = 1
        with account.activity("evaluation"):
            clock.advance(9)
        account.close_interval(0, (0, 0, 0, 0))
        clock.advance(3)
        _started_at, interval_millis, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(interval_millis, 3)
        self.assertEqual(figures["evaluation_millis"], 0)
        self.assertEqual(figures["candidates_evaluated"], 0)
        self.assertEqual(figures["fruitless_scans"], 0)
        self.assertEqual(figures["heartbeats_deferred"], 0)

    def test_the_longest_tick_gap_names_the_activity_it_opened_in(self):
        clock = _SteppingClock()
        account = _account(clock)
        account.note_tick()
        clock.advance(2_000)
        with account.activity("evaluation"):
            account.note_tick()
            clock.advance(41_000)
        with account.activity("finalizing"):
            account.note_tick()
            clock.advance(3_000)
            account.note_tick()
        _started_at, _interval, figures = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(figures["max_tick_gap_millis"], 41_000)
        self.assertEqual(figures["max_tick_gap_activity"], "evaluation")

    def test_a_gap_is_reported_in_the_interval_it_closes(self):
        clock = _SteppingClock()
        account = _account(clock)
        account.note_tick()
        account.note_heartbeat_written()
        clock.advance(35_000)
        _started_at, _interval, first = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertIsNone(first["max_tick_gap_millis"])
        self.assertIsNone(first["max_heartbeat_gap_millis"])
        clock.advance(5_000)
        account.note_tick()
        account.note_heartbeat_written()
        _started_at, _interval, second = account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(second["max_tick_gap_millis"], 40_000)
        self.assertEqual(second["max_heartbeat_gap_millis"], 40_000)

    def test_counters_and_claim_timing_are_reported_as_given(self):
        account = _account(_SteppingClock())
        account.candidates_evaluated = 12
        account.fruitless_scans = 3
        account.heartbeats_deferred = 2
        _started_at, _interval, figures = account.close_interval(
            987, (5, 6, 7, 8))
        self.assertEqual(
            {name: figures[name] for name in (
                "candidates_evaluated", "fruitless_scans",
                "heartbeats_deferred", "nodes", "claim_lock_wait_millis",
                "claim_transaction_millis", "claim_commit_millis",
                "claim_retries")},
            {"candidates_evaluated": 12, "fruitless_scans": 3,
             "heartbeats_deferred": 2, "nodes": 987,
             "claim_lock_wait_millis": 5, "claim_transaction_millis": 6,
             "claim_commit_millis": 7, "claim_retries": 8})


class TestWorkerChargesItsActivities(unittest.TestCase):
    """The worker charges each kind of work where it does it.

    A clock that moves 1 ms on every read gives every activity the worker
    enters at least a millisecond, so a column left at zero is a site that
    never opened its activity.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for attr, words in (("ANSWER_FILE", BRANCH),
                            ("WORDS_FILE", CANDIDATES)):
            path = os.path.join(self._tmp.name, f"{attr}.txt")
            with open(path, "w") as f:
                f.write("\n".join(words) + "\n")
            patcher = mock.patch.object(erd_swarm, attr, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        self.branch_key = ScoreCache.encode_subset(BRANCH)
        ScoreCache(self.cache_path, BRANCH).close()
        queue = ERDQueue(self.queue_path)
        queue.create_branch(self.branch_key, len(BRANCH), len(CANDIDATES),
                            budget=ROOT_BUDGET)
        queue.close()

    def _worker(self, **kwargs):
        worker = _BranchWorker(0, self.cache_path, self.queue_path, None,
                               **kwargs)
        worker._time_account = _account(_SteppingClock(step_millis=1))
        return worker

    def _rows(self):
        queue = ERDQueue(self.queue_path)
        try:
            return [dict(row) for row in queue._conn.execute(
                "SELECT * FROM telemetry.worker_time ORDER BY id")]
        finally:
            queue.close()

    def test_a_focused_solve_charges_evaluation_claiming_and_finalizing(self):
        worker = self._worker(small_count=2, count_cap=2)
        try:
            worker.solve_branch_focused(self.branch_key)
        finally:
            worker.close()
        [row] = self._rows()
        self.assertGreater(row["evaluation_millis"], 0)
        self.assertGreater(row["claiming_millis"], 0)
        self.assertGreater(row["finalizing_millis"], 0)
        self.assertEqual(sum(row[c] for c in ACTIVITY_COLUMNS),
                         row["interval_millis"])
        self.assertEqual(row["worker_id"], "worker-0")
        self.assertGreater(row["candidates_evaluated"], 0)
        self.assertEqual(row["nodes"], worker._nodes)

    def test_candidates_evaluated_matches_the_claims_recorded(self):
        worker = self._worker(small_count=2, count_cap=2)
        try:
            worker.solve_branch_focused(self.branch_key)
        finally:
            worker.close()
        [row] = self._rows()
        queue = ERDQueue(self.queue_path)
        claims = queue._conn.execute(
            "SELECT COUNT(*) FROM telemetry.claim_telemetry").fetchone()[0]
        queue.close()
        self.assertEqual(row["candidates_evaluated"], claims)

    def test_work_selection_is_charged_to_scheduling(self):
        worker = self._worker()
        try:
            self.assertIsNotNone(worker.claim_one())
        finally:
            worker.close()
        [row] = self._rows()
        self.assertGreater(row["scheduling_millis"], 0)
        self.assertGreater(row["claiming_millis"], 0)

    def test_a_scan_that_selects_nothing_is_counted(self):
        worker = self._worker()
        try:
            worker._claim_one_uninstrumented = lambda: None
            self.assertIsNone(worker.claim_one())
        finally:
            worker.close()
        [row] = self._rows()
        self.assertEqual(row["fruitless_scans"], 1)


class TestWorkerTimeIsWrittenOnTheHeartbeat(unittest.TestCase):

    def _worker(self, clock):
        worker = _bare_worker()
        worker.queue.claim_timing_totals.return_value = (10, 20, 30, 4)
        worker._time_account = _account(clock)
        worker._time_account_claim_timing = (1, 2, 3, 1)
        return worker

    def _tick(self, worker):
        worker._liveness_tick(None, None, None, None, None, None, force=True)

    def test_no_row_before_the_interval_has_run(self):
        worker = self._worker(_SteppingClock())
        self._tick(worker)
        worker.queue.add_worker_time.assert_not_called()

    def test_a_row_is_written_once_the_interval_has_run(self):
        clock = _SteppingClock()
        worker = self._worker(clock)
        worker._nodes = 500
        clock.advance(erd_swarm.WORKER_TIME_INTERVAL_SECONDS * 1000)
        self._tick(worker)
        worker.queue.add_worker_time.assert_called_once()
        worker_id, _started_at, interval_millis, figures = (
            worker.queue.add_worker_time.call_args.args)
        self.assertEqual(worker_id, "worker-0")
        self.assertEqual(interval_millis,
                         erd_swarm.WORKER_TIME_INTERVAL_SECONDS * 1000)
        self.assertEqual(figures["nodes"], 500)
        self.assertEqual(
            (figures["claim_lock_wait_millis"],
             figures["claim_transaction_millis"],
             figures["claim_commit_millis"], figures["claim_retries"]),
            (9, 18, 27, 3))
        self.assertEqual(worker._time_account_nodes, 500)
        self.assertEqual(worker._time_account_claim_timing, (10, 20, 30, 4))

    def test_a_checkpoint_pause_defers_both_the_heartbeat_and_the_row(self):
        clock = _SteppingClock()
        worker = self._worker(clock)
        clock.advance(erd_swarm.WORKER_TIME_INTERVAL_SECONDS * 1000)
        worker._checkpoint_pause_active = mock.Mock(return_value=True)
        worker._liveness_tick(None, None, None, None, None, None)
        worker.queue.heartbeat.assert_not_called()
        worker.queue.add_worker_time.assert_not_called()
        self.assertEqual(worker._time_account.heartbeats_deferred, 1)

    def test_the_heartbeat_path_measures_both_gaps(self):
        clock = _SteppingClock()
        worker = self._worker(clock)
        self._tick(worker)
        clock.advance(61_000)
        self._tick(worker)
        _worker_id, _started_at, _interval, figures = (
            worker.queue.add_worker_time.call_args.args)
        self.assertEqual(figures["max_tick_gap_millis"], 61_000)
        self.assertEqual(figures["max_tick_gap_activity"], "other")
        self.assertEqual(figures["max_heartbeat_gap_millis"], 61_000)

    def test_a_candidate_evaluation_is_charged_to_evaluation(self):
        clock = _SteppingClock()
        worker = self._worker(clock)

        def evaluation_taking_30_millis(*args, **kwargs):
            clock.advance(30)
            return (SOLVED, 1.5, 1, False)

        with mock.patch("erd_swarm.evaluate_candidate",
                        side_effect=evaluation_taking_30_millis):
            self.assertTrue(worker.evaluate_claim(
                ScoreCache.encode_subset(BRANCH), BRANCH, len(BRANCH), idx=0))
        _started_at, _interval, figures = worker._time_account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(figures["evaluation_millis"], 30)
        self.assertEqual(figures["candidates_evaluated"], 1)

    def test_heartbeat_writes_inside_an_evaluation_are_not_evaluation(self):
        clock = _SteppingClock()
        worker = self._worker(clock)
        clock.advance(erd_swarm.WORKER_TIME_INTERVAL_SECONDS * 1000)
        worker.queue.heartbeat.side_effect = lambda *a, **k: clock.advance(20)
        worker.queue.add_worker_time.side_effect = (
            lambda *a, **k: clock.advance(7))
        with worker._time_account.activity("evaluation"):
            self._tick(worker)
        _started_at, _interval, figures = worker._time_account.close_interval(
            0, (0, 0, 0, 0))
        # The row insert lands in the interval that follows the one it closes.
        self.assertEqual(figures["evaluation_millis"], 0)
        self.assertEqual(figures["other_millis"], 7)
        _worker_id, _started_at, _interval, written = (
            worker.queue.add_worker_time.call_args.args)
        self.assertEqual(written["evaluation_millis"], 0)
        self.assertEqual(written["other_millis"],
                         erd_swarm.WORKER_TIME_INTERVAL_SECONDS * 1000 + 20)

    def test_closing_the_worker_writes_the_partial_interval(self):
        worker = self._worker(_SteppingClock())
        worker._maybe_write_worker_time(force=True)
        worker.queue.add_worker_time.assert_called_once()

    def test_a_wait_is_charged_to_the_reason_given(self):
        clock = _SteppingClock()
        worker = self._worker(clock)
        with mock.patch("erd_swarm.time.sleep",
                        side_effect=lambda seconds: clock.advance(
                            seconds * 1000)):
            worker._idle_wait(0.5, "no_work")
            worker._idle_wait(0.05, "rival_finalize")
        _started_at, _interval, figures = worker._time_account.close_interval(
            0, (0, 0, 0, 0))
        self.assertEqual(figures["wait_no_work_millis"], 500)
        self.assertEqual(figures["wait_rival_finalize_millis"], 50)


class TestBranchCoordinationReachesTheFinalizeLog(unittest.TestCase):

    def setUp(self):
        TestWorkerChargesItsActivities.setUp(self)

    def test_the_branch_total_is_the_sum_of_its_claims(self):
        worker = _BranchWorker(0, self.cache_path, self.queue_path, None,
                               small_count=2, count_cap=2)
        try:
            worker.solve_branch_focused(self.branch_key)
        finally:
            worker.close()
        self.assertEqual(worker._bundle_coordination_millis, {})
        queue = ERDQueue(self.queue_path)
        try:
            [logged] = queue._conn.execute(
                "SELECT coordination_millis "
                "FROM telemetry.branch_finalize_log").fetchone()
            claimed = queue._conn.execute(
                "SELECT SUM(coordination_millis) "
                "FROM telemetry.claim_telemetry").fetchone()[0]
        finally:
            queue.close()
        self.assertIsNotNone(logged)
        self.assertEqual(logged, claimed)

    def test_each_bundle_carries_only_its_own_members(self):
        worker = _bare_worker()
        worker.queue = mock.Mock()
        worker._bundle_coordination_millis = {"b1": 7, "b2": 4}
        worker._finish_bundle(b"key", "b1", 0, time.time(), censored=False)
        worker.queue.record_bundle_stats.assert_called_once_with(
            b"key", "b1", 0, mock.ANY, censored=False, coordination_millis=7)
        self.assertEqual(worker._bundle_coordination_millis, {"b2": 4})


class _QueueTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue = ERDQueue(os.path.join(self._tmp.name, "queue.sqlite3"))
        self.addCleanup(self.queue.close)


class TestReclaimsAreRecorded(_QueueTest):

    def setUp(self):
        super().setUp()
        self.branch_key = ScoreCache.encode_subset(BRANCH)
        self.queue.create_branch(self.branch_key, len(BRANCH),
                                 len(CANDIDATES), budget=ROOT_BUDGET)
        order = list(range(len(CANDIDATES)))
        self.queue.claim_next_bundle(
            self.branch_key, "worker-1", len(CANDIDATES), order,
            [0.0] * len(CANDIDATES), small_count=3, count_cap=3)

    def _reclaims(self):
        return [dict(row) for row in self.queue._conn.execute(
            "SELECT worker_id, cause, claims_freed, heartbeat_age_seconds "
            "FROM telemetry.claim_reclaim ORDER BY id")]

    def _age_claims(self, seconds):
        self.queue._conn.execute(
            "UPDATE candidate_claims SET claimed_at = claimed_at - ?",
            (seconds,))

    def test_a_stale_reclaim_records_the_heartbeat_it_found(self):
        self.queue.heartbeat("worker-1", 1, self.branch_key, len(BRANCH),
                             0, 0)
        self.queue._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = updated_at - 45")
        self._age_claims(60)
        freed = self.queue.reclaim_stale_claims(30)
        [reclaim] = self._reclaims()
        self.assertEqual(reclaim["claims_freed"], freed)
        self.assertGreater(freed, 0)
        self.assertEqual(reclaim["worker_id"], "worker-1")
        self.assertEqual(reclaim["cause"], "stale")
        self.assertGreaterEqual(reclaim["heartbeat_age_seconds"], 45)

    def test_a_worker_that_never_heartbeat_has_no_age(self):
        self._age_claims(60)
        self.queue.reclaim_stale_claims(30)
        [reclaim] = self._reclaims()
        self.assertIsNone(reclaim["heartbeat_age_seconds"])

    def test_a_reclaim_that_frees_nothing_records_nothing(self):
        self.queue.heartbeat("worker-1", 1, self.branch_key, len(BRANCH),
                             0, 0)
        self._age_claims(60)
        self.assertEqual(self.queue.reclaim_stale_claims(30), 0)
        self.assertEqual(self._reclaims(), [])

    def test_a_respawn_and_a_restart_say_so(self):
        self.queue.reclaim_claims_of_worker("worker-1")
        self.queue.claim_next_bundle(
            self.branch_key, "worker-2", len(CANDIDATES),
            list(range(len(CANDIDATES))), [0.0] * len(CANDIDATES),
            small_count=3, count_cap=3)
        self.queue.recover_active_branches()
        self.assertEqual(
            [(row["worker_id"], row["cause"]) for row in self._reclaims()],
            [("worker-1", "worker"), ("worker-2", "restart")])


class TestClaimTimingTotals(_QueueTest):

    def test_totals_survive_what_clears_the_claim_attribution(self):
        with mock.patch("erd_queue.time.perf_counter",
                        side_effect=[0.0, 0.005]):
            self.queue._begin_immediate_timed()
        self.queue._conn.execute("COMMIT")
        self.queue.discard_claim_attribution()
        self.assertEqual(self.queue.claim_timing_totals(), (5, 0, 0, 0))
        with mock.patch("erd_queue.time.perf_counter",
                        side_effect=[0.0, 0.003]):
            self.queue._begin_immediate_timed()
        self.queue._conn.execute("COMMIT")
        self.queue.add_claim_telemetry(5, 0, 0, 1)
        self.assertEqual(self.queue.claim_timing_totals(), (8, 0, 0, 0))

    def test_a_claim_adds_its_transaction_and_commit(self):
        with mock.patch("erd_queue.time.perf_counter",
                        side_effect=[1.004, 1.004, 1.006]):
            self.queue._conn.execute("BEGIN IMMEDIATE")
            self.queue._commit_claim_transaction(1.0)
        totals = self.queue.claim_timing_totals()
        self.assertEqual(totals[1:3], (4, 2))


class TestCheckpointPausesAreRecorded(_QueueTest):

    def test_a_truncate_records_its_pause(self):
        with mock.patch.object(erd_search, "QUEUE_WAL_QUIESCE_BYTES", 0):
            erd_search._maybe_quiesce_truncate(self.queue)
        [pause] = self.queue._conn.execute(
            "SELECT pause_millis, wal_bytes, truncated "
            "FROM telemetry.checkpoint_pause").fetchall()
        self.assertGreaterEqual(pause["pause_millis"], 0)
        self.assertEqual(pause["truncated"], 1)
        self.assertFalse(self.queue.checkpoint_paused())

    def test_a_truncate_that_never_wins_is_recorded_as_such(self):
        queue = mock.Mock()
        queue.wal_size_bytes.return_value = erd_search.QUEUE_WAL_QUIESCE_BYTES
        queue.checkpoint.return_value = (1, 0, 0)
        with mock.patch.object(erd_search.time, "time",
                               side_effect=[0, 1_000]):
            erd_search._maybe_quiesce_truncate(queue)
        started_at, _pause_millis, wal_bytes, truncated = (
            queue.add_checkpoint_pause.call_args.args)
        self.assertEqual((started_at, wal_bytes, truncated),
                         (0, erd_search.QUEUE_WAL_QUIESCE_BYTES, False))

    def test_the_retry_budget_starts_once_the_flag_has_landed(self):
        queue = mock.Mock()
        queue.wal_size_bytes.return_value = erd_search.QUEUE_WAL_QUIESCE_BYTES
        clock = [100.0]
        queue.set_checkpoint_pause.side_effect = (
            lambda paused: clock.__setitem__(0, clock[0] + 20.0)
            if paused else None)
        outcomes = iter([(1, 0, 0), (0, 0, 0)])
        queue.checkpoint.side_effect = lambda mode: next(outcomes)
        with mock.patch.object(erd_search.time, "time",
                               side_effect=lambda: clock[0]), \
                mock.patch.object(erd_search.time, "sleep"):
            erd_search._maybe_quiesce_truncate(queue)
        self.assertEqual(queue.checkpoint.call_count, 2)
        started_at, _pause_millis, _wal_bytes, truncated = (
            queue.add_checkpoint_pause.call_args.args)
        self.assertEqual(started_at, 120.0)
        self.assertTrue(truncated)

    def test_a_pause_that_cannot_be_recorded_still_ends(self):
        queue = mock.Mock()
        queue.wal_size_bytes.return_value = erd_search.QUEUE_WAL_QUIESCE_BYTES
        queue.checkpoint.return_value = (0, 0, 0)
        queue.add_checkpoint_pause.side_effect = sqlite3.OperationalError(
            "locked")
        with self.assertLogs(erd_search.logger, "WARNING"):
            erd_search._maybe_quiesce_truncate(queue)
        self.assertEqual(queue.set_checkpoint_pause.call_args_list[-1],
                         mock.call(False))


if __name__ == "__main__":
    unittest.main()
