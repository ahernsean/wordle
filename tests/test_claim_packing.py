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


class TestSchemaMigration(unittest.TestCase):
    """Opening a pre-existing (pre-packer) queue.sqlite3 must migrate
    cleanly, not crash — CLAUDE.md requires every schema change to be an
    idempotent migration, never a manual-SQL requirement."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "old_queue.sqlite3")
        # The pre-PR candidate_claims shape: no bundle_id column.  CREATE
        # TABLE IF NOT EXISTS in a fresh ERDQueue() is a no-op against this,
        # so any statement assuming bundle_id already exists (e.g. an index
        # on it) must not run before _migrate() adds the column.
        conn = sqlite3.connect(self.path)
        conn.executescript("""
            CREATE TABLE candidate_claims (
                branch_key BLOB    NOT NULL,
                idx        INTEGER NOT NULL,
                claimed_by TEXT,
                claimed_at INTEGER,
                done       INTEGER NOT NULL DEFAULT 0,
                done_at    INTEGER,
                PRIMARY KEY (branch_key, idx)
            );
        """)
        conn.commit()
        conn.close()

    def test_opening_pre_bundle_id_database_does_not_raise(self):
        q = ERDQueue(self.path)
        try:
            cols = {r["name"] for r in
                   q._conn.execute("PRAGMA table_info(candidate_claims)")}
            self.assertIn("bundle_id", cols)
            index_rows = q._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' AND name = 'idx_candidate_claims_bundle'"
            ).fetchall()
            self.assertEqual(len(index_rows), 1)
        finally:
            q.close()

    def test_migration_is_idempotent_on_reopen(self):
        ERDQueue(self.path).close()
        q = ERDQueue(self.path)   # second open must not fail or double-create
        q.close()

    def test_unmigratable_missing_column_refuses_to_open(self):
        # A pre-existing queue table missing a column that no _migrate() rule
        # adds: CREATE TABLE IF NOT EXISTS is a no-op against it, so without
        # the schema assertion the mismatch would surface only at the first
        # statement naming the column — long after open, killing a worker.
        # Keeps the columns the schema's index needs, so the failure lands in
        # the assertion rather than in index creation.
        conn = sqlite3.connect(self.path)
        conn.execute("""
            CREATE TABLE pending_branches (
                branch_key BLOB    NOT NULL,
                n_words    INTEGER NOT NULL,
                priority   INTEGER NOT NULL DEFAULT 0,
                status     TEXT    NOT NULL DEFAULT 'pending',
                PRIMARY KEY (branch_key)
            )
        """)
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError) as raised:
            ERDQueue(self.path)
        self.assertIn("pending_branches", str(raised.exception))
        self.assertIn("source_word", str(raised.exception))

    def test_unmigratable_telemetry_file_refuses_to_open(self):
        # The assertion covers the telemetry file too: a bad-shaped table
        # there refuses at open just like one in the queue file.
        telemetry_path = os.path.join(self._tmp.name,
                                      "old_queue_telemetry.sqlite3")
        conn = sqlite3.connect(telemetry_path)
        conn.execute("""
            CREATE TABLE claim_telemetry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at INTEGER
            )
        """)
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError) as raised:
            ERDQueue(self.path)
        self.assertIn("telemetry.claim_telemetry", str(raised.exception))
        self.assertIn("n_words", str(raised.exception))

    def test_extra_column_warns_but_opens(self):
        ERDQueue(self.path).close()
        telemetry_path = os.path.join(self._tmp.name,
                                      "old_queue_telemetry.sqlite3")
        conn = sqlite3.connect(telemetry_path)
        conn.execute("ALTER TABLE claim_telemetry ADD COLUMN stray INTEGER")
        conn.commit()
        conn.close()
        with self.assertLogs("erd_queue", level="WARNING") as captured:
            q = ERDQueue(self.path)
            q.close()
        self.assertTrue(any("stray" in message for message in captured.output))

    def test_pre_split_empty_telemetry_table_is_dropped(self):
        # A queue file from before the telemetry split carries telemetry
        # tables in the queue database, where the qualified telemetry.*
        # statements would silently bypass them.  An empty one is dropped at
        # open (with a warning) and the telemetry write path works normally.
        conn = sqlite3.connect(self.path)
        conn.execute("""
            CREATE TABLE branch_finalize_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
        """)
        conn.commit()
        conn.close()

        with self.assertLogs("erd_queue", level="WARNING") as captured:
            q = ERDQueue(self.path)
        self.assertTrue(any("pre-split" in message
                            for message in captured.output))
        try:
            in_queue_file = q._conn.execute(
                "SELECT 1 FROM main.sqlite_master "
                "WHERE type = 'table' AND name = 'branch_finalize_log'"
            ).fetchone()
            self.assertIsNone(in_queue_file)
            q.add_branch_finalize_log(
                b"key", "SALET -g-g-", 87, 4, 100, 200, 12345, 87,
                n_bundles=3, max_bundle_nodes=999,
                total_bundle_wall_millis=5000, censored_units=0)
            row = q._conn.execute(
                "SELECT total_bundle_wall_millis "
                "FROM telemetry.branch_finalize_log").fetchone()
            self.assertEqual(row["total_bundle_wall_millis"], 5000)
        finally:
            q.close()

    def test_pre_split_populated_telemetry_table_refuses_to_open(self):
        # A pre-split telemetry table that still holds rows refuses at open:
        # its data must be archived deliberately, never ignored silently.
        conn = sqlite3.connect(self.path)
        conn.execute("""
            CREATE TABLE claim_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                n_words INTEGER
            )
        """)
        conn.execute("INSERT INTO claim_telemetry (n_words) VALUES (42)")
        conn.commit()
        conn.close()

        with self.assertRaises(RuntimeError) as raised:
            ERDQueue(self.path)
        self.assertIn("predates the telemetry split", str(raised.exception))
        self.assertIn("claim_telemetry", str(raised.exception))

    def test_telemetry_file_is_created_alongside_queue(self):
        ERDQueue(self.path).close()
        telemetry_path = os.path.join(self._tmp.name,
                                      "old_queue_telemetry.sqlite3")
        self.assertTrue(os.path.exists(telemetry_path))
        conn = sqlite3.connect(telemetry_path)
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")}
        conn.close()
        self.assertEqual(names, {
            "bundle_stats", "cost_samples", "claim_telemetry",
            "branch_finalize_log", "candidate_accuracy",
            "backstop_telemetry", "cut_reuse_misses"})


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

    def test_tight_bound_completes_eliminated_candidates_without_bundle(self):
        self.q.update_branch_best(self.key, "salet", 1.0)
        claim = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES,
            small_count=5, count_cap=500)
        self.assertIsNone(claim)
        self.assertEqual(self.q.branch_done_candidates(self.key), N_CANDIDATES)
        self.assertEqual(self.q.branch_bulk_done_candidates(self.key),
                         N_CANDIDATES)

    def test_survivors_still_flow_through_normal_bundle(self):
        self.q.update_branch_best(self.key, "salet", 2.0)
        lower_bounds = [2.5] * N_CANDIDATES
        survivor_indices = [1, 4, 9]
        for idx in survivor_indices:
            lower_bounds[idx] = 1.5
        _bundle_id, indices, forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, lower_bounds,
            small_count=5, count_cap=500)
        self.assertEqual(indices, survivor_indices)
        self.assertEqual(forced, frozenset())
        self.assertEqual(self.q.branch_done_candidates(self.key),
                         N_CANDIDATES - len(survivor_indices))

    def test_holes_pass_bulk_completes_republished_eliminated_candidates(self):
        bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=N_CANDIDATES, count_cap=N_CANDIDATES)
        holes = indices[:4]
        for idx in indices[4:]:
            self.q.complete_candidate(self.key, idx)
        self.q.republish_remainder(self.key, bundle_id, holes)
        self.q.update_branch_best(self.key, "salet", 1.0)
        claim = self.q.claim_next_bundle(
            self.key, "w1", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES)
        self.assertIsNone(claim)
        self.assertEqual(self.q.branch_done_candidates(self.key), N_CANDIDATES)
        self.assertEqual(self.q.branch_bulk_done_candidates(self.key), len(holes))

    def test_bulk_completion_supersedes_in_flight_claim(self):
        _bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        self.q.update_branch_best(self.key, "salet", 1.0)
        self.assertIsNone(self.q.claim_next_bundle(
            self.key, "w1", N_CANDIDATES, _ORDER, [2.5] * N_CANDIDATES))
        rows = {row["idx"]: row for row in self.q.claims_for_branch(self.key)}
        for idx in indices:
            self.assertEqual(rows[idx]["done"], 1)
            self.assertEqual(rows[idx]["claimed_by"], "bulk-elimination")
        self.assertTrue(self.q.try_finalize_branch(self.key))

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

    def test_forward_path_does_not_collide_with_mark_claims_done(self):
        # mark_claims_done (within-candidate overrun promotion, erd_swarm.py's
        # _MidLoopPublisher) can insert done=1 rows for a fresh branch's
        # best-first prefix before the packer's cursor ever advances past
        # them -- the forward path must not try to re-INSERT those same
        # positions.
        self.q.mark_claims_done(self.key, [0, 1, 2])
        claim = self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        self.assertIsNotNone(claim)
        _bundle_id, indices, _forced = claim
        self.assertTrue(set(indices).isdisjoint({0, 1, 2}),
                        "must not re-claim positions mark_claims_done already covered")

    def test_forward_path_returns_none_when_whole_packed_bundle_already_done(self):
        # mark_claims_done covers positions 0..N_CANDIDATES-1 entirely (the
        # whole branch was evaluated inline before overrun fired): the
        # forward path's packed bundle filters down to empty, and the call
        # must report "nothing claimed" rather than a bundle of size 0.
        self.q.mark_claims_done(self.key, list(range(N_CANDIDATES)))
        claim = self.q.claim_next_bundle(
            self.key, "worker-0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        self.assertIsNone(claim)

    def test_full_drain_with_mark_claims_done_prefix_has_unique_coverage(self):
        self.q.mark_claims_done(self.key, [0, 1, 2])
        done = {0, 1, 2}
        while True:
            claim = self.q.claim_next_bundle(
                self.key, "worker-0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=5, count_cap=5)
            if claim is None:
                break
            _bundle_id, indices, _forced = claim
            self.assertTrue(done.isdisjoint(indices), "overlapping/duplicate claim")
            for idx in indices:
                self.q.complete_candidate(self.key, idx)
                done.add(idx)
        self.assertEqual(done, set(range(N_CANDIDATES)))


class TestRepublishRemainder(_TmpQueue):
    def test_republish_deletes_done0_rows_and_bumps_count(self):
        bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        counts = self.q.republish_remainder(self.key, bundle_id, indices)
        self.assertEqual(set(counts), set(indices))
        self.assertTrue(all(c == 1 for c in counts.values()))
        rows = self.q.claims_for_branch(self.key)
        self.assertEqual([r for r in rows if r["idx"] in indices], [])

    def test_republished_candidates_are_reclaimable_via_holes_pass(self):
        bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=N_CANDIDATES, count_cap=N_CANDIDATES)
        self.assertEqual(len(indices), N_CANDIDATES)   # cursor now == N_CANDIDATES
        self.q.republish_remainder(self.key, bundle_id, indices[:3])
        _bundle_id, reclaimed, _forced = self.q.claim_next_bundle(
            self.key, "w1", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        self.assertEqual(set(reclaimed), set(indices[:3]))

    def test_republish_only_deletes_rows_still_claimed_under_its_own_bundle_id(self):
        # A candidate reclaimed (or re-claimed by another worker) under a
        # DIFFERENT bundle_id after this bundle's claim must survive an
        # unrelated republish_remainder call for the stale bundle_id -- and
        # must not have its republish count bumped.
        bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        stolen_idx = indices[0]
        # Simulate reclaim_stale_claims freeing it, then another worker's
        # packer legitimately re-claiming it under a fresh bundle_id.
        self.q._conn.execute(
            "DELETE FROM candidate_claims WHERE branch_key = ? AND idx = ?",
            (self.key, stolen_idx))
        self.q._conn.execute("""
            INSERT INTO candidate_claims
                (branch_key, idx, claimed_by, claimed_at, done, bundle_id)
            VALUES (?, ?, 'other-worker', 0, 0, 'other-bundle')
        """, (self.key, stolen_idx))

        counts = self.q.republish_remainder(self.key, bundle_id, indices)
        self.assertNotIn(stolen_idx, counts)   # not bumped: not deleted by this call
        row = self.q._conn.execute(
            "SELECT bundle_id FROM candidate_claims WHERE branch_key = ? AND idx = ?",
            (self.key, stolen_idx)).fetchone()
        self.assertEqual(row["bundle_id"], "other-bundle")   # untouched
        republish_row = self.q._conn.execute(
            "SELECT count FROM candidate_republish WHERE branch_key = ? AND idx = ?",
            (self.key, stolen_idx)).fetchone()
        self.assertIsNone(republish_row)   # no spurious count bump

    def test_republish_returns_empty_when_nothing_still_claimed_under_bundle_id(self):
        bundle_id, indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=5, count_cap=5)
        # Every one of this bundle's rows already moved to a different
        # bundle_id (e.g. reclaimed and re-claimed elsewhere) before this
        # worker's own republish call runs.
        self.q._conn.executemany(
            "UPDATE candidate_claims SET bundle_id = 'elsewhere' "
            "WHERE branch_key = ? AND idx = ?",
            [(self.key, idx) for idx in indices])
        counts = self.q.republish_remainder(self.key, bundle_id, indices)
        self.assertEqual(counts, {})

    def test_forced_set_after_republish_limit_reached(self):
        # Drain the whole order in one bundle so pack_cursor reaches
        # N_CANDIDATES; every further claim then goes through holes_pass,
        # which re-surfaces a republished subset as the same indices each
        # time (the forward path would instead hand out fresh, never-seen
        # positions and never re-visit the republished ones).
        bundle_id, _indices, _forced = self.q.claim_next_bundle(
            self.key, "w0", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
            small_count=N_CANDIDATES, count_cap=N_CANDIDATES)
        target = [3, 4, 5]
        forced = frozenset()
        for i in range(3):
            self.q.republish_remainder(self.key, bundle_id, target)
            bundle_id, indices, forced = self.q.claim_next_bundle(
                self.key, f"w{i + 1}", N_CANDIDATES, _ORDER, _ZERO_LOWER_BOUND,
                small_count=5, count_cap=5, republish_limit=3)
            self.assertEqual(set(indices), set(target))
        self.assertEqual(forced, frozenset(target))

    def test_republish_of_empty_indices_is_noop(self):
        self.assertEqual(self.q.republish_remainder(self.key, "b0", []), {})


class TestBundleStatsAndFinalizeLog(_TmpQueue):
    def test_finalize_log_preserves_aggregate_bulk_done_count(self):
        self.q.add_branch_finalize_log(
            self.key, None, 5, 4, 10, 20, 30, 3,
            bulk_done_candidates=37)
        row = self.q._conn.execute(
            "SELECT n_claims, bulk_done_candidates "
            "FROM telemetry.branch_finalize_log").fetchone()
        self.assertEqual(row["n_claims"], 3)
        self.assertEqual(row["bulk_done_candidates"], 37)

    def test_finalize_bundle_stats_aggregates_and_clears(self):
        self.q.record_bundle_stats(self.key, "b1", nodes=10, wall_millis=5)
        self.q.record_bundle_stats(self.key, "b2", nodes=40, wall_millis=7,
                                   censored=True)
        n_bundles, max_bundle_nodes, total_bundle_wall_millis, censored_units = (
            self.q.finalize_bundle_stats(self.key))
        self.assertEqual(n_bundles, 2)
        self.assertEqual(max_bundle_nodes, 40)
        self.assertEqual(total_bundle_wall_millis, 12)
        self.assertEqual(censored_units, 1)
        # Cleared: a second call sees nothing.
        self.assertEqual(self.q.finalize_bundle_stats(self.key),
                         (None, None, None, None))

    def test_finalize_bundle_stats_empty_when_branch_never_claimed_a_bundle(self):
        self.assertEqual(self.q.finalize_bundle_stats(self.key),
                         (None, None, None, None))

    def test_record_bundle_stats_is_a_noop_once_branch_is_deleted(self):
        # A worker's own record_bundle_stats call can race behind another
        # worker's finalize_bundle_stats + delete_branch for the same
        # branch: the insert must not resurrect an orphaned row.
        self.q.delete_branch(self.key)
        self.q.record_bundle_stats(self.key, "late-bundle", nodes=99, wall_millis=1)
        row = self.q._conn.execute(
            "SELECT * FROM bundle_stats WHERE branch_key = ?", (self.key,)).fetchone()
        self.assertIsNone(row)


class TestBundleIdUniqueAcrossRespawn(unittest.TestCase):
    """A crashed-and-respawned worker reuses its fixed worker_id slot but
    never its pid, so record_bundle_stats can never silently clobber a
    still-open branch's earlier bundle under the same bundle_id."""

    def test_same_worker_id_different_pid_never_collide(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "q.sqlite3")
        key = encode_subset(["crane", "slate", "trace", "stale", "tales"])
        order = list(range(N_CANDIDATES))
        lb = [0.0] * N_CANDIDATES

        with mock.patch("os.getpid", return_value=1111):
            q1 = ERDQueue(path)
            q1.create_branch(key, 5, N_CANDIDATES)
            bundle_id_1, _indices, _forced = q1.claim_next_bundle(
                key, "worker-0", N_CANDIDATES, order, lb, small_count=5, count_cap=5)
            q1.close()

        with mock.patch("os.getpid", return_value=2222):
            q2 = ERDQueue(path)   # simulates a respawned worker-0 process
            bundle_id_2, _indices, _forced = q2.claim_next_bundle(
                key, "worker-0", N_CANDIDATES, order, lb, small_count=5, count_cap=5)
            q2.close()

        self.assertNotEqual(bundle_id_1, bundle_id_2)


class TestClaimTelemetryContentionAttribution(_TmpQueue):
    """claim_retries/busy_wait_millis must attribute to exactly the claim
    that produced them, never repeat on a later, unrelated candidate's row
    -- the failure mode a nested claim_next_bundle call (within-candidate
    sub-branch promotion, on the same connection) can otherwise trigger."""

    def test_contention_values_reset_after_being_logged_once(self):
        self.q._last_claim_busy_millis = 250
        self.q._last_claim_retries = 3
        self.q.add_claim_telemetry(10, 5, 1, 4)
        self.q.add_claim_telemetry(10, 1, 1, 4)   # a later, unrelated candidate
        rows = self.q._conn.execute(
            "SELECT busy_wait_millis, claim_retries FROM claim_telemetry "
            "ORDER BY id").fetchall()
        self.assertEqual((rows[0]["busy_wait_millis"], rows[0]["claim_retries"]), (250, 3))
        self.assertEqual((rows[1]["busy_wait_millis"], rows[1]["claim_retries"]), (0, 0))


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
