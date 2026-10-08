import os
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue


WORDS = ["crane", "slate", "trace", "stale", "tales"]


def _words(prefix, count):
    return [f"{prefix}{i:04d}"[:5] for i in range(count)]


class QueueVisibilityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(self.q.close)
        self.user_key = ScoreCache.encode_subset(WORDS)
        self.coop_key = ScoreCache.encode_subset(WORDS[:3])

    def test_pending_user_branch_row(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 7, "crane", 1)])
        rows = self.q.list_queue_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "user")
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(rows[0]["opener_pattern_text"], "----y")

    def test_report_telemetry_indexes_exist_and_are_idempotent(self):
        expected = {
            "idx_branch_finalize_log_branch_recorded_at",
            "idx_branch_finalize_log_epoch_recorded_id",
            "idx_branch_finalize_log_finalized_at",
            "idx_cut_reuse_misses_branch_recorded_at",
            "idx_cut_reuse_misses_epoch_recorded_id",
        }
        indexes = set()
        for table in ("branch_finalize_log", "cut_reuse_misses"):
            indexes.update(
                row["name"] for row in self.q._conn.execute(
                    f"PRAGMA telemetry.index_list({table})"
                )
            )
        self.assertTrue(expected.issubset(indexes))
        self.q._migrate()
        indexes_after = set()
        for table in ("branch_finalize_log", "cut_reuse_misses"):
            indexes_after.update(
                row["name"] for row in self.q._conn.execute(
                    f"PRAGMA telemetry.index_list({table})"
                )
            )
        self.assertEqual(indexes, indexes_after)

    def test_candidate_republish_count_defaults_to_zero_and_reads_a_republish(self):
        self.q.create_branch(self.user_key, len(WORDS), len(WORDS))
        self.assertEqual(self.q.candidate_republish_for_branch(self.user_key), [])
        branch_id = self.q._intern_branch(self.user_key)
        self.q._conn.execute("""
            INSERT INTO candidate_republish (branch_id, idx, count)
            VALUES (?, ?, ?)
        """, (branch_id, 1, 3))
        self.assertEqual(
            self.q.candidate_republish_for_branch(self.user_key),
            [{"idx": 1, "count": 3}],
        )

    def test_erd_prune_provenance_migration_backfills_legacy_counts_once(self):
        self.q.create_branch(self.user_key, len(WORDS), 10)
        branch_id = self.q._intern_branch(self.user_key)
        self.q._conn.execute("""
            UPDATE active_branches
            SET bulk_done_candidates = 7,
                one_level_erd_pruned_candidates = 0,
                two_level_erd_pruned_candidates = 0
            WHERE branch_id = ?
        """, (branch_id,))
        self.q.add_branch_finalize_log(
            self.user_key, "CRANE -----", 5, 5, 10, 20, 100, 3,
            bulk_done_candidates=5,
        )
        self.q._conn.execute("""
            UPDATE telemetry.branch_finalize_log
            SET one_level_erd_pruned_candidates = 0,
                two_level_erd_pruned_candidates = 0
        """)
        self.q._conn.execute(
            "DELETE FROM schema_migrations "
            "WHERE name = 'split_erd_prune_provenance'")

        self.q._migrate()
        active_counts = self.q.branch_erd_pruned_candidate_counts(self.user_key)
        finalize_counts = self.q._conn.execute("""
            SELECT one_level_erd_pruned_candidates,
                   two_level_erd_pruned_candidates
            FROM telemetry.branch_finalize_log
        """).fetchone()
        self.assertEqual(active_counts, (7, 0))
        self.assertEqual(tuple(finalize_counts), (5, 0))

        self.q._conn.execute("""
            UPDATE active_branches
            SET two_level_erd_pruned_candidates = 2
            WHERE branch_id = ?
        """, (branch_id,))
        self.q._migrate()
        self.assertEqual(
            self.q.branch_erd_pruned_candidate_counts(self.user_key), (7, 2))

    def test_branch_report_telemetry_is_bounded_and_preserves_outcomes(self):
        self.q.create_branch(self.user_key, len(WORDS), 10)
        self.q.record_bundle_stats(self.user_key, "bundle-1", 50, 20, censored=True)
        for outcome, evaluated, bulk in (
            ("exact", 7, 1), ("cut", 5, 2), ("loss", 3, 4),
        ):
            self.q.add_branch_finalize_log(
                self.user_key, "CRANE -----", 5, 5,
                10, 20, 100, evaluated,
                n_bundles=2, max_bundle_nodes=60,
                total_bundle_wall_millis=30, censored_units=1,
                ceiling=2.5 if outcome == "cut" else None,
                outcome=outcome, bulk_done_candidates=bulk,
                best_guess="crane" if outcome == "exact" else None,
                best_erd=1.5 if outcome == "exact" else None,
            )
        self.q.add_cut_reuse_miss(self.user_key, 5, 4, None, 2.5, 3)
        telemetry = self.q.report_branch_telemetry(self.user_key, limit=2)
        self.assertEqual(telemetry["bundle_summary"]["bundle_count"], 1)
        self.assertEqual(len(telemetry["recent_finalizations"]), 2)
        self.assertEqual(telemetry["finalization_total_count"], 3)
        self.assertEqual(
            {row["spine"] for row in telemetry["recent_finalizations"]},
            {"CRANE -----"},
        )
        self.assertEqual(
            {row["outcome"] for row in telemetry["recent_finalizations"]},
            {"cut", "loss"},
        )
        loss = telemetry["recent_finalizations"][0]
        self.assertNotEqual(
            loss["evaluated_candidate_count"],
            loss["bulk_completed_candidate_count"],
        )
        self.assertIsNone(loss["best_guess"])
        self.assertIsNone(loss["best_erd"])
        self.assertEqual(len(telemetry["cut_reuse_misses"]), 1)

    def test_branch_candidate_eta_sample_starts_at_latest_best_erd(self):
        self.q.create_branch(self.user_key, 100, 10)
        branch_id = self.q._intern_branch(self.user_key)
        self.q._conn.execute("""
            UPDATE active_branches
            SET best_erd = 3.0, best_updated_at = 900
            WHERE branch_id = ?
        """, (branch_id,))
        self.q._conn.execute("""
            INSERT INTO telemetry.two_level_prune_telemetry
                (branch_id, inspected_candidate_count, pruned_candidate_count,
                 bound_erd, wall_millis, epoch, recorded_at)
            VALUES (?, 8, 6, 2.5, 800, 0, 850),
                   (?, 10, 7, 3.0, 1000, 0, 950)
        """, (branch_id, branch_id))
        self.q._conn.execute("""
            INSERT INTO candidate_claims
                (branch_id, idx, claimed_by, done, done_at,
                 evaluation_millis, evaluation_bound_erd)
            VALUES (?, 0, 'worker-0', 1, 850, 9000, 2.5),
                   (?, 1, 'worker-0', 1, 950, 11000, 3.0),
                   (?, 2, 'one-level-erd-prune', 1, 960, NULL, NULL),
                   (?, 3, 'worker-1', 1, 970, 5000, 2.5)
        """, (branch_id, branch_id, branch_id, branch_id))

        sample = self.q.branch_candidate_eta_sample(
            self.user_key, window_seconds=600, now=1_000)

        self.assertEqual(sample["window_started_at"], 900)
        self.assertEqual(sample["inspected_candidate_count"], 10)
        self.assertEqual(sample["pruned_candidate_count"], 7)
        self.assertEqual(sample["inspection_worker_millis"], 1000)
        self.assertEqual(sample["evaluated_candidate_count"], 1)
        self.assertEqual(sample["evaluation_worker_millis"], 11000)

    def test_branch_candidate_eta_sample_without_best_erd_starts_at_creation(self):
        self.q.create_branch(self.user_key, 100, 10)
        branch_id = self.q._intern_branch(self.user_key)
        self.q._conn.execute("""
            UPDATE active_branches SET created_at = 900 WHERE branch_id = ?
        """, (branch_id,))
        self.q._conn.execute("""
            INSERT INTO candidate_claims
                (branch_id, idx, claimed_by, done, done_at,
                 evaluation_millis, evaluation_bound_erd)
            VALUES (?, 0, 'worker-0', 1, 950, 11000, NULL)
        """, (branch_id,))

        sample = self.q.branch_candidate_eta_sample(
            self.user_key, window_seconds=600, now=1_000)

        self.assertIsNone(sample["best_updated_at"])
        self.assertEqual(sample["window_started_at"], 900)
        self.assertEqual(sample["evaluated_candidate_count"], 1)
        self.assertEqual(sample["evaluation_worker_millis"], 11000)

    def test_candidate_eta_sample_counts_only_workers_on_its_branch(self):
        self.q.create_branch(self.user_key, 100, 10)
        self.q.create_branch(self.coop_key, 100, 10)
        now = int(time.time())
        for worker_id, branch_key in (
                ("worker-0", self.user_key), ("worker-1", self.user_key),
                ("worker-2", self.coop_key)):
            self.q.heartbeat(
                worker_id, 1, branch_key, 100, now, 0)
        bundle_id, [idx], _forced = self.q.claim_next_bundle(
            self.user_key, "worker-0", 10, list(range(10)), [0.0] * 10,
            small_count=1, count_cap=1)
        self.assertTrue(self.q.apply_candidate_result(
            self.user_key, idx, claimed_by="worker-0", bundle_id=bundle_id,
            evaluation_millis=1))

        row = self.q._conn.execute("""
            SELECT branch_worker_count FROM candidate_claims
            WHERE evaluation_millis IS NOT NULL
        """).fetchone()
        self.assertEqual(row["branch_worker_count"], 2)
        sample = self.q.branch_candidate_eta_sample(
            self.user_key, window_seconds=60, now=now)
        self.assertEqual(sample["evaluation_worker_count"], 2)
        self.assertEqual(sample["evaluation_worker_count_min"], 2)
        self.assertEqual(sample["evaluation_worker_count_max"], 2)

    def test_eta_migration_starts_existing_incumbent_sample_once(self):
        self.q.create_branch(self.user_key, 100, 10)
        branch_id = self.q._intern_branch(self.user_key)
        self.q._conn.execute("""
            UPDATE active_branches
            SET best_erd = 3.0, best_updated_at = NULL
            WHERE branch_id = ?
        """, (branch_id,))

        before = int(time.time())
        self.q._migrate()
        first_timestamp = self.q._conn.execute("""
            SELECT best_updated_at FROM active_branches WHERE branch_id = ?
        """, (branch_id,)).fetchone()["best_updated_at"]
        self.assertGreaterEqual(first_timestamp, before)

        self.q._migrate()
        second_timestamp = self.q._conn.execute("""
            SELECT best_updated_at FROM active_branches WHERE branch_id = ?
        """, (branch_id,)).fetchone()["best_updated_at"]
        self.assertEqual(second_timestamp, first_timestamp)

    def test_branch_report_telemetry_after_cursor_pages_past_the_first_window(self):
        self.q.create_branch(self.user_key, len(WORDS), 10)
        for outcome in ("exact", "cut", "loss"):
            self.q.add_branch_finalize_log(
                self.user_key, "CRANE -----", 5, 5,
                10, 20, 100, 7,
                n_bundles=2, max_bundle_nodes=60,
                total_bundle_wall_millis=30, censored_units=1,
                ceiling=2.5 if outcome == "cut" else None,
                outcome=outcome, bulk_done_candidates=1,
                best_guess="crane" if outcome == "exact" else None,
                best_erd=1.5 if outcome == "exact" else None,
            )
        first_page = self.q.report_branch_telemetry(self.user_key, limit=2)
        self.assertEqual(first_page["finalization_total_count"], 3)
        self.assertEqual(
            [row["outcome"] for row in first_page["recent_finalizations"]],
            ["loss", "cut"],
        )
        last_row = first_page["recent_finalizations"][-1]
        second_page = self.q.report_branch_telemetry(
            self.user_key, limit=2,
            after=(last_row["recorded_at"], last_row["finalization_id"]),
        )
        self.assertEqual(
            [row["outcome"] for row in second_page["recent_finalizations"]],
            ["exact"],
        )

    def test_branch_report_telemetry_after_cursor_is_stable_when_a_new_row_lands(self):
        # An OFFSET counts rows from the current head, so a row landing
        # between two page fetches shifts everything under it -- the next
        # page then repeats a row instead of continuing past it.  A cursor
        # tied to an actual already-seen row must not be affected by
        # whatever lands after it, since the branch is still being solved
        # while the user pages through its history.
        self.q.create_branch(self.user_key, len(WORDS), 10)
        for outcome in ("exact", "cut", "loss"):
            self.q.add_branch_finalize_log(
                self.user_key, "CRANE -----", 5, 5,
                10, 20, 100, 7, outcome=outcome, bulk_done_candidates=1,
            )
        first_page = self.q.report_branch_telemetry(self.user_key, limit=1)
        cursor_row = first_page["recent_finalizations"][0]
        self.assertEqual(cursor_row["outcome"], "loss")
        # A new finalization lands on the branch while the user is paging.
        self.q.add_branch_finalize_log(
            self.user_key, "CRANE -----", 5, 5,
            10, 20, 100, 7, outcome="exact", bulk_done_candidates=1,
            best_guess="crane", best_erd=2.0,
        )
        second_page = self.q.report_branch_telemetry(
            self.user_key, limit=1,
            after=(cursor_row["recorded_at"], cursor_row["finalization_id"]),
        )
        self.assertEqual(second_page["recent_finalizations"][0]["outcome"], "cut")

    def test_branch_report_telemetry_before_cursor_reverses_back_to_newest_first(self):
        self.q.create_branch(self.user_key, len(WORDS), 10)
        for outcome in ("exact", "cut", "loss"):
            self.q.add_branch_finalize_log(
                self.user_key, "CRANE -----", 5, 5,
                10, 20, 100, 7, outcome=outcome, bulk_done_candidates=1,
            )
        first_page = self.q.report_branch_telemetry(self.user_key, limit=1)
        first_row = first_page["recent_finalizations"][0]
        second_page = self.q.report_branch_telemetry(
            self.user_key, limit=1,
            after=(first_row["recorded_at"], first_row["finalization_id"]),
        )
        second_row = second_page["recent_finalizations"][0]
        self.assertEqual(second_row["outcome"], "cut")
        back_to_first = self.q.report_branch_telemetry(
            self.user_key, limit=1,
            before=(second_row["recorded_at"], second_row["finalization_id"]),
        )
        self.assertEqual(back_to_first["recent_finalizations"][0]["outcome"], "loss")

    def test_coordination_is_not_a_hotspot_field(self):
        # Coordination is recorded per branch and per worker, not per claim,
        # so there are no claim rows to bucket.
        with self.assertRaisesRegex(ValueError, "unsupported hotspot field"):
            self.q.report_hotspots(
                "coordination", 0, int(time.time()) - 60, 3, 2)

    def test_current_hotspots_support_queue_and_tree_populations(self):
        self.q.create_branch(
            self.user_key, len(WORDS), 10, priority=7, budget=4,
            spine="CRANE -----",
        )
        self.q.add_nodes_spent(self.user_key, 25)

        queue_result = self.q.report_hotspots(
            "nodes", epoch=0, since=0, sample_size=10, limit=1,
        )
        tree_result = self.q.report_hotspots(
            "size", epoch=0, since=0, sample_size=10, limit=1,
            spine_prefix="CRANE -----",
        )

        self.assertEqual(queue_result["population"], "current_queue_branches")
        self.assertEqual(queue_result["sample_size"], None)
        self.assertEqual(queue_result["rows"][0]["search_node_count"], 25)
        self.assertEqual(tree_result["sampled_row_count"], 1)
        self.assertEqual(tree_result["rows"][0]["spine"], "CRANE -----")

    def test_cut_reuse_and_erd_prune_hotspots_are_normalized(self):
        now = int(time.time())
        self.q.add_cut_reuse_miss(self.user_key, 5, 4, None, 2.5, 3)
        self.q.add_branch_finalize_log(
            self.user_key, "CRANE -----", 5, 4, now - 1, now,
            100, 3, outcome="cut", bulk_done_candidates=9,
            one_level_erd_pruned_candidates=5,
            two_level_erd_pruned_candidates=4,
        )

        cut_reuse = self.q.report_hotspots(
            "cut-reuse", epoch=0, since=now - 60,
            sample_size=10, limit=1,
        )
        one_level = self.q.report_hotspots(
            "one-level-erd-prunes", epoch=0, since=now - 60,
            sample_size=10, limit=1,
        )
        two_level = self.q.report_hotspots(
            "two-level-erd-prunes", epoch=0, since=now - 60,
            sample_size=10, limit=1,
        )

        self.assertEqual(cut_reuse["population"], "recent_cut_reuse_misses")
        self.assertEqual(cut_reuse["rows"][0]["cut_reuse_miss_count"], 1)
        self.assertEqual(
            one_level["rows"][0]["one_level_erd_pruned_candidate_count"], 5
        )
        self.assertEqual(
            two_level["rows"][0]["two_level_erd_pruned_candidate_count"], 4
        )
        with self.assertRaisesRegex(ValueError, "unsupported hotspot field"):
            self.q.report_hotspots("unknown", 0, now - 60, 10, 1)

    def test_cut_reuse_hotspot_scope_uses_exact_branch_key(self):
        now = int(time.time())
        self.q.add_cut_reuse_miss(self.user_key, 5, 4, None, 2.5, 3)
        self.q.add_cut_reuse_miss(self.coop_key, 3, 3, None, 2.0, 2)
        result = self.q.report_hotspots(
            "cut-reuse", epoch=0, since=now - 60,
            sample_size=10, limit=10, spine_prefix="CRANE -----",
            branch_key=self.user_key,
        )
        self.assertEqual(
            [row["branch_key"] for row in result["rows"]], [self.user_key]
        )
        with self.assertRaisesRegex(ValueError, "singular branch target"):
            self.q.report_hotspots(
                "cut-reuse", epoch=0, since=now - 60,
                sample_size=10, limit=10, spine_prefix="CRANE",
            )

    def test_legacy_finalization_does_not_claim_an_exact_outcome(self):
        now = int(time.time())
        self.q.add_branch_finalize_log(
            self.user_key, "CRANE -----", 5, 4, now - 2, now - 1,
            100, 3, outcome="loss",
        )
        self.q._conn.execute(
            "UPDATE telemetry.branch_finalize_log "
            "SET outcome = NULL, ceiling = NULL WHERE branch_key = ?",
            (self.user_key,),
        )
        telemetry = self.q.report_branch_telemetry(self.user_key, limit=1)
        self.assertEqual(telemetry["recent_finalizations"][0]["outcome"], "unknown")

    def test_report_normalizes_ceiling_above_budget_cut_as_loss(self):
        now = int(time.time())
        self.q.add_branch_finalize_log(
            self.user_key, "CRANE -----", 5, 3, now - 2, now - 1,
            100, 3, ceiling=3.25, outcome="cut",
        )
        row = self.q.report_branch_telemetry(self.user_key, limit=1)[
            "recent_finalizations"][0]
        self.assertEqual(row["outcome"], "loss")
        self.assertEqual(row["loss_proof"], "ceiling_above_budget")
        stored = self.q._conn.execute(
            "SELECT outcome FROM telemetry.branch_finalize_log"
        ).fetchone()["outcome"]
        self.assertEqual(stored, "cut")

    def test_report_queue_filters_accept_dicts_and_cover_each_bound(self):
        self.q.create_branch(
            self.user_key, len(WORDS), 10, priority=7, budget=4,
            spine="CRANE -----",
        )
        self.q.heartbeat(
            "worker-1", 1, self.user_key, len(WORDS), int(time.time()), 0
        )
        result = self.q.report_queue_rows({
            "branch_statuses": ("evaluating",),
            "branch_worker_statuses": ("active",),
            "minimum_answer_count": len(WORDS),
            "maximum_answer_count": len(WORDS),
            "budget": 4,
            "priority": 7,
            "limit": 1,
        })
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["rows"][0]["branch_status"], "evaluating")
        self.assertEqual(result["rows"][0]["branch_worker_status"], "active")
        self.q._conn.execute("DELETE FROM run_meta WHERE key = 'epoch'")
        self.assertIsNone(self.q.epoch_metadata())

    def test_queue_totals_and_rows_come_from_one_snapshot(self):
        # A limited page cannot be counted by measuring itself, so the totals
        # come from their own aggregate -- a second statement.  A swarm moves
        # branches between statuses continuously, so without one read snapshot
        # a status-filtered report can count matches the page no longer holds.
        keys = [ScoreCache.encode_subset(WORDS[:size]) for size in (2, 3, 4, 5)]
        self.q.add_pending_many([
            (key, len(WORDS), 7, "crane", 1) for key in keys
        ])
        rival = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(rival.close)

        class _FinishBetweenStatements:
            """Stand in for a worker that finishes every matched branch in the
            window between the aggregate and the page."""

            def __init__(self, connection):
                self._connection = connection
                self._fired = False

            def execute(self, statement, *arguments):
                if not self._fired and " ORDER BY " in statement:
                    self._fired = True
                    for key in keys:
                        rival.mark_done(key)
                return self._connection.execute(statement, *arguments)

            def __getattr__(self, name):
                return getattr(self._connection, name)

        real_connection = self.q._conn
        self.q._conn = _FinishBetweenStatements(real_connection)
        try:
            result = self.q.report_queue_rows({"branch_statuses": ("queued",)})
        finally:
            self.q._conn = real_connection

        self.assertEqual(result["matched_rows"], len(keys))
        self.assertEqual(len(result["rows"]), len(keys))
        self.assertEqual(
            result["summary"]["branch_count_by_status"], {"queued": len(keys)}
        )
        # The write itself landed; the report simply predates it.
        after = self.q.report_queue_rows({"branch_statuses": ("queued",)})
        self.assertEqual(after["matched_rows"], 0)

    def test_finalization_hotspot_metadata_uses_the_scoped_population(self):
        now = int(time.time())
        for index, spine in enumerate((
            "RAISE -----", "RAISE -----", "RAISE -----", "CRANE -----",
        )):
            self.q.add_branch_finalize_log(
                bytes([index]), spine, 5, 4, now - 10, now,
                100 + index, index + 1,
            )
        result = self.q.report_hotspots(
            "evaluated-candidates", epoch=0, since=now - 60,
            sample_size=2, limit=2, spine_prefix="RAISE -----",
        )
        self.assertEqual(result["sampled_row_count"], 2)
        self.assertTrue(result["sample_truncated"])
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(row["spine"] == "RAISE -----"
                            for row in result["rows"]))
        self.assertTrue(all(row["outcome"] == "unknown"
                            for row in result["rows"]))

    def test_historical_queries_use_bounded_report_indexes(self):
        finalize_plan = " ".join(
            row["detail"] for row in self.q._conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM telemetry.branch_finalize_log
                WHERE branch_key = ? ORDER BY recorded_at DESC LIMIT 5
            """, (self.user_key,))
        )
        cut_plan = " ".join(
            row["detail"] for row in self.q._conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM telemetry.cut_reuse_misses
                WHERE branch_key = ? ORDER BY recorded_at DESC LIMIT 5
            """, (self.user_key,))
        )
        finalization_sample_plan = " ".join(
            row["detail"] for row in self.q._conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM telemetry.branch_finalize_log
                WHERE epoch = ? AND recorded_at >= ?
                ORDER BY recorded_at DESC, id DESC LIMIT 5
            """, (0, 1))
        )
        cut_sample_plan = " ".join(
            row["detail"] for row in self.q._conn.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM telemetry.cut_reuse_misses
                WHERE epoch = ? AND recorded_at >= ?
                ORDER BY recorded_at DESC, id DESC LIMIT 5
            """, (0, 1))
        )
        self.assertIn("idx_branch_finalize_log_branch_recorded_at", finalize_plan)
        self.assertIn("idx_cut_reuse_misses_branch_recorded_at", cut_plan)
        self.assertIn(
            "idx_branch_finalize_log_epoch_recorded_id",
            finalization_sample_plan,
        )
        self.assertIn("idx_cut_reuse_misses_epoch_recorded_id", cut_sample_plan)

    def test_user_in_progress_joins_pending_and_active_state(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 5, "crane", 1)])
        self.q.claim_next("worker-0")
        self.q.create_branch(
            self.user_key, len(WORDS), 20, priority=5,
            opener="crane", opener_pattern=1,
            budget=5, spine="CRANE ----y")
        rows = self.q.list_queue_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "user")
        self.assertEqual(rows[0]["status"], "in_progress")
        self.assertEqual(rows[0]["budget"], 5)
        self.assertEqual(rows[0]["n_candidates"], 20)

    def test_cooperative_active_branch_has_no_pending_membership(self):
        self.q.create_branch(
            self.coop_key, 3, 10, priority=1_000_000,
            opener="alibi", opener_pattern=42,
            spine="CRANE -y--g ALIBI g-g--")
        rows = self.q.list_queue_rows()
        self.assertEqual(rows[0]["kind"], "coop")
        self.assertEqual(rows[0]["status"], "open")

    def test_done_rows_appear_when_filtering_done(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 0, "crane", 0)])
        self.q.claim_next("worker-0")
        self.q.mark_done(self.user_key)
        rows = self.q.list_queue_rows({"status": "done"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "done")

    def test_prefix_filter_matches_descendants(self):
        self.q.create_branch(
            self.coop_key, 3, 10, priority=1_000_000,
            spine="CRANE -y--g ALIBI g-g--")
        self.assertEqual(
            len(self.q.list_queue_rows({"prefix": "CRANE -y--g"})), 1)
        self.assertEqual(
            self.q.list_queue_rows({"prefix": "SLATE"}), [])

    def test_queue_top_excludes_pending_rows(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 0, "crane", 0)])
        self.q.create_branch(
            self.coop_key, 3, 10, priority=1_000_000,
            spine="CRANE -y--g ALIBI g-g--")
        rows = self.q.queue_top_rows("size", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["branch_key"], self.coop_key)

    def test_filters_cover_all_rejection_paths(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 7, "crane", 1)])
        self.q.create_branch(
            self.user_key, len(WORDS), 20, priority=7,
            opener="crane", opener_pattern=1, budget=4,
            spine="CRANE ----y")

        self.assertEqual(self.q.list_queue_rows({"status": "done"}), [])
        self.assertEqual(self.q.list_queue_rows({"min_words": 6}), [])
        self.assertEqual(self.q.list_queue_rows({"max_words": 4}), [])
        self.assertEqual(self.q.list_queue_rows({"budget": 3}), [])
        self.assertEqual(self.q.list_queue_rows({"priority": 8}), [])
        self.assertEqual(self.q.list_queue_rows({"opener": "slate"}), [])
        self.assertEqual(self.q.list_queue_rows({"prefix": "SLATE -----"}), [])
        self.assertEqual(
            len(self.q.list_queue_rows({
                "status": "pending",
                "min_words": 5,
                "max_words": 5,
                "budget": 4,
                "priority": 7,
                "opener": "crane",
                "prefix": "CRANE ----y",
            })),
            1)

    def test_sort_modes_and_limit_are_deterministic(self):
        small_key = ScoreCache.encode_subset(_words("a", 3))
        big_key = ScoreCache.encode_subset(_words("b", 12))
        self.q.create_branch(
            small_key, 3, 10, priority=1, budget=3,
            spine="CRANE ----- ALIBI -----")
        self.q.create_branch(
            big_key, 12, 10, priority=2, budget=3,
            spine="CRANE ----- ZONAL -----")
        self.q.add_nodes_spent(small_key, 100)
        self.q.add_nodes_spent(big_key, 50)
        now = 123456
        self.q.heartbeat("worker-0", 1, big_key, 12, now, 0)
        self.q.heartbeat("worker-1", 2, big_key, 12, now, 0)
        self.q.heartbeat("worker-2", 3, small_key, 3, now, 0)

        self.assertEqual(self.q.list_queue_rows(sort="nodes")[0]["branch_key"], small_key)
        self.assertEqual(self.q.list_queue_rows(sort="size")[0]["branch_key"], big_key)
        self.assertEqual(self.q.list_queue_rows(sort="workers")[0]["branch_key"], big_key)
        self.assertEqual(self.q.list_queue_rows(sort="priority")[0]["branch_key"], big_key)
        self.assertEqual(self.q.list_queue_rows(sort="slowest")[0]["branch_key"], small_key)
        self.assertEqual(len(self.q.list_queue_rows(sort="age", limit=1)), 1)

    def test_dashboard_tree_summary_and_detail_helpers(self):
        small_key = ScoreCache.encode_subset(_words("a", 3))
        mid_key = ScoreCache.encode_subset(_words("b", 50))
        large_key = ScoreCache.encode_subset(_words("c", 500))
        huge_key = ScoreCache.encode_subset(_words("d", 1000))

        self.q.add_pending_many([
            (mid_key, 50, 0, "crane", 0),
            (large_key, 500, 3, "slate", 0),
            (huge_key, 1000, 1, "trace", 0),
        ])
        self.q.create_branch(
            small_key, 3, 10, priority=1_000_000,
            opener="alibi", opener_pattern=42, budget=2,
            spine="CRANE -y--g ALIBI g-g--")
        self.q.add_nodes_spent(small_key, 123)
        self.q.update_branch_best(small_key, "crane", 1.25, max_depth=3)
        self.q.mark_branch_tainted(small_key)
        self.q.mark_claims_done(small_key, [0, 1])
        self.q.record_bundle_stats(small_key, "bundle-1", 100, 50)
        self.q._conn.execute(
            "INSERT INTO candidate_republish (branch_id, idx, count) "
            "VALUES (?, 2, 1)", (self.q._intern_branch(small_key, create=True),))
        self.q.add_branch_finalize_log(
            small_key, "CRANE -y--g ALIBI g-g--", 3, 2,
            10, 20, 123, 2)
        self.q.heartbeat("worker-0", 1, small_key, 3, 10, 0)

        dashboard = self.q.queue_dashboard(limit=1)
        self.assertEqual(len(dashboard["active"]), 1)
        self.assertEqual(len(dashboard["pending"]), 1)
        self.assertGreaterEqual(dashboard["summary"]["total"], 4)

        tree = self.q.queue_tree_rows(
            "CRANE -y--g", active_only=True, max_depth=2, limit=1)
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["branch_key"], small_key)

        summary = self.q.queue_summary()
        self.assertEqual(summary["by_priority"]["coop"], 0)
        self.assertGreaterEqual(summary["by_priority"]["0"], 2)
        self.assertGreaterEqual(summary["by_priority"]["1-999"], 2)
        self.assertEqual(summary["by_size"]["2-9"], 1)
        self.assertEqual(summary["by_size"]["10-99"], 1)
        self.assertEqual(summary["by_size"]["100-999"], 1)
        self.assertEqual(summary["by_size"]["1000+"], 1)
        self.assertIsNotNone(summary["largest_pending"])
        self.assertIsNotNone(summary["oldest_active"])

        detail = self.q.branch_detail(small_key, include_claims=True)
        self.assertEqual(len(detail["claims"]), 2)
        self.assertEqual(len(detail["bundle_stats"]), 1)
        self.assertEqual(len(detail["republish"]), 1)
        self.assertEqual(len(detail["finalize_log"]), 1)
        self.assertEqual(len(detail["workers"]), 1)
        self.assertTrue(detail["tainted"])

        self.assertIsNone(self.q.branch_detail(b"missing"))

    def test_branch_ref_resolution_variants(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 0, "crane", 0)])
        row = self.q.list_queue_rows()[0]
        self.assertEqual(self.q.resolve_branch_ref(""), [])
        self.assertEqual(
            self.q.resolve_branch_ref(row["branch_key_hex"][:12])[0]["branch_key"],
            self.user_key)
        self.assertEqual(
            self.q.resolve_branch_ref("CRANE")[0]["branch_key"],
            self.user_key)

    def test_row_spine_text_helper(self):
        self.assertEqual(
            self.q.row_spine_text({"spine": "CRANE -----"}),
            "CRANE -----")
        self.assertEqual(
            self.q.row_spine_text({
                "opener": "crane",
                "opener_pattern_text": "-----",
            }),
            "CRANE -----")
        self.assertEqual(self.q.row_spine_text({}), "")


# Two seconds and thirty seconds of worker time, matching the first two
# report_model band edges; the tests below build branches on either side.
BAND_EDGE_MILLIS = [2_000, 30_000]


class WorkDistributionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.q = ERDQueue(os.path.join(self._tmp.name, "q.sqlite3"))
        self.addCleanup(self.q.close)

    def _finalized(self, tag, size, claims, nodes, worker_millis,
                   coordination_millis=10, created_at=None):
        """Log a finalized branch the way maybe_finalize does."""
        finalized_at = int(time.time())
        self.q.add_branch_finalize_log(
            ScoreCache.encode_subset(_words(tag, 4)), f"{tag.upper()} -----",
            size, 3, finalized_at if created_at is None else created_at,
            finalized_at, nodes, claims, n_bundles=1,
            total_bundle_wall_millis=worker_millis,
            coordination_millis=coordination_millis)

    def _report(self, since=None, **answer_count_range):
        return self.q.report_work_distribution(
            self.q.epoch, since, BAND_EDGE_MILLIS, **answer_count_range)

    def _band(self, report, band_index):
        for band in report["bands"]:
            if band["band_index"] == band_index:
                return band
        return None

    def test_bands_come_from_worker_time_not_claim_count(self):
        # More claims than the costly branch, yet one second of work: the band
        # key is time.
        self._finalized("cheap", 4, claims=10, nodes=50, worker_millis=1_000)
        self._finalized("costl", 6, claims=2, nodes=1_800_000,
                        worker_millis=50_000)

        report = self._report()

        self.assertEqual(self._band(report, 0)["branch_count"], 1)
        self.assertEqual(self._band(report, 0)["worker_millis"], 1_000)
        self.assertEqual(self._band(report, 0)["claim_count"], 10)
        self.assertEqual(self._band(report, 2)["branch_count"], 1)
        self.assertEqual(self._band(report, 2)["worker_millis"], 50_000)
        self.assertIsNone(self._band(report, 1))

    def test_a_branch_that_waited_on_children_bands_by_its_own_work(self):
        # A promoting parent's finalize row spans hours of waiting on its
        # children, and its own work is a fraction of a second.
        finalized_at = int(time.time())
        self._finalized("paren", 8, claims=3, nodes=120, worker_millis=600,
                        created_at=finalized_at - 10_000)
        self._finalized("child", 5, claims=4, nodes=2_000_000,
                        worker_millis=80_000)

        report = self._report()

        self.assertEqual(self._band(report, 0)["branch_count"], 1)
        self.assertEqual(self._band(report, 0)["search_node_count"], 120)
        self.assertEqual(self._band(report, 2)["branch_count"], 1)
        self.assertEqual(self._band(report, 2)["search_node_count"], 2_000_000)

    def test_coordination_is_totalled_with_its_band(self):
        self._finalized("cheap", 4, claims=2, nodes=5, worker_millis=100,
                        coordination_millis=70)
        self._finalized("chea2", 4, claims=2, nodes=5, worker_millis=100,
                        coordination_millis=30)

        self.assertEqual(
            self._band(self._report(), 0)["coordination_millis"], 100)

    def test_a_branch_with_unrecorded_time_is_never_banded(self):
        # Coalescing an unknown to zero would seat a branch of unknown cost in
        # the cheapest band while still counting its nodes.
        self._finalized("measu", 4, claims=2, nodes=10, worker_millis=100)
        self._finalized("nowrk", 6, claims=3, nodes=2_700_000,
                        worker_millis=None)
        self._finalized("nocrd", 6, claims=1, nodes=5, worker_millis=100,
                        coordination_millis=None)

        report = self._report()

        self.assertEqual(sum(b["branch_count"] for b in report["bands"]), 1)
        self.assertEqual(self._band(report, 0)["search_node_count"], 10)
        self.assertEqual(report["unmeasured"]["branch_count"], 2)
        self.assertEqual(report["unmeasured"]["claim_count"], 4)
        self.assertEqual(report["unmeasured"]["search_node_count"], 2_700_005)

    def test_a_window_narrows_the_population_and_the_whole_epoch_is_default(self):
        self._finalized("older", 4, claims=4, nodes=5, worker_millis=100)
        self._finalized("newer", 4, claims=6, nodes=5, worker_millis=100)
        self.q._conn.execute(
            "UPDATE telemetry.branch_finalize_log SET recorded_at = 100 "
            "WHERE n_claims = 4")

        self.assertEqual(self._band(self._report(), 0)["claim_count"], 10)
        self.assertEqual(
            self._band(self._report(since=101), 0)["claim_count"], 6)

    def test_an_answer_count_range_narrows_the_banded_population(self):
        for size in (3, 7, 20):
            self._finalized(f"b{size:03d}", size, claims=size, nodes=5,
                            worker_millis=100)

        banded = self._report(minimum_answer_count=5,
                              maximum_answer_count=10)
        at_least = self._report(minimum_answer_count=5)
        at_most = self._report(maximum_answer_count=7)

        self.assertEqual(self._band(banded, 0)["claim_count"], 7)
        self.assertEqual(self._band(at_least, 0)["claim_count"], 27)
        self.assertEqual(self._band(at_most, 0)["claim_count"], 10)

    def test_rows_outside_the_epoch_are_excluded(self):
        self._finalized("cheap", 4, claims=3, nodes=5, worker_millis=100)

        other_epoch = self.q.report_work_distribution(
            self.q.epoch + 1, None, BAND_EDGE_MILLIS)

        self.assertEqual(other_epoch["bands"], [])
        self.assertEqual(other_epoch["unmeasured"]["branch_count"], 0)


if __name__ == "__main__":
    unittest.main()
