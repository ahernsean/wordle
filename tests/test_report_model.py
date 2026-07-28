"""Tests for the shared ERD swarm report model."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
import erd_search
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportSources,
    WORKER_LIVENESS_SECONDS,
    _opener_erd_summary,
    branch_reference,
    collect_overview_report,
    collect_report,
    normalize_worker_descent,
    parse_rich_spine,
    parse_report_branch_target,
)
from wordle_engine import ERD_ALL


ANSWERS = ["salet", "crane", "nurdy", "khaki"]


class ReportModelTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = self.temporary_directory.name
        self.queue_path = os.path.join(directory, "queue.sqlite3")
        self.telemetry_path = os.path.join(directory, "telemetry.sqlite3")
        self.cache_path = os.path.join(directory, "cache.sqlite3")
        self.answer_list_path = os.path.join(directory, "answers.txt")
        self.candidate_list_path = os.path.join(directory, "candidates.txt")
        with open(self.answer_list_path, "w") as answer_file:
            answer_file.write("\n".join(ANSWERS) + "\n")
        with open(self.candidate_list_path, "w") as candidate_file:
            candidate_file.write("\n".join(ANSWERS + ["raise"]) + "\n")
        self.sources = ReportSources(
            queue_path=self.queue_path,
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=self.telemetry_path,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _open_queue(self):
        return ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)

    @staticmethod
    def _group(pattern, answer_count, best_erd, max_remaining_depth):
        return {
            "pattern": pattern,
            "answer_count": answer_count,
            "best_erd": best_erd,
            "max_remaining_depth": max_remaining_depth,
        }

    def test_opener_erd_summary_folds_solved_groups_with_opener_guess(self):
        summary = _opener_erd_summary([
            self._group("-----", 8, 2.1, 3),
            self._group("----g", 2, 1.5, 2),
            self._group("ggggg", 1, None, None),
        ])
        # 1 (opener) + weighted mean of remaining depth; the all-green group is
        # the opener itself, contributing zero remaining guesses.
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["erd"], 1.0 + (8 * 2.1 + 2 * 1.5) / 11)
        self.assertEqual(summary["max_remaining_depth"], 4)
        self.assertEqual(summary["resolved_group_count"], 3)
        self.assertEqual(summary["response_group_count"], 3)

    def test_opener_erd_summary_is_incomplete_while_a_group_is_unsolved(self):
        summary = _opener_erd_summary([
            self._group("-----", 8, 2.1, 3),
            self._group("y----", 5, None, None),
            self._group("ggggg", 1, None, None),
        ])
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["erd"])
        self.assertIsNone(summary["max_remaining_depth"])
        self.assertEqual(summary["resolved_group_count"], 2)
        self.assertEqual(summary["response_group_count"], 3)

    def test_opener_erd_summary_solves_a_lone_survivor_in_one_more_guess(self):
        summary = _opener_erd_summary([self._group("----y", 1, None, None)])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["erd"], 2.0)
        self.assertEqual(summary["max_remaining_depth"], 2)

    def test_rich_spine_parser_preserves_legacy_tuple_contract(self):
        path = "3:KHAKI:--y--/33→4:NURDY:---y-/17"
        expected = [
            (3, "KHAKI", "--y--", "33"),
            (4, "NURDY", "---y-", "17"),
        ]
        self.assertEqual(parse_rich_spine(path), expected)
        self.assertEqual(erd_search._parse_spine(path), expected)

    def test_worker_descent_is_normalized_without_changing_parser(self):
        parsed = parse_rich_spine("3:KHAKI:--y--/33→4:RAISE:---y-/17")
        descent = normalize_worker_descent(parsed, set(ANSWERS))
        self.assertEqual(descent[0], {
            "guess_depth": 3,
            "word": "khaki",
            "pattern": "--y--",
            "answer_count_text": "33",
            "word_is_answer": True,
        })
        self.assertFalse(descent[1]["word_is_answer"])
        self.assertIsInstance(parsed[0], tuple)

    def test_branch_reference_is_stable_long_and_distinct(self):
        reference = branch_reference(b"branch-one")
        self.assertEqual(reference, branch_reference(b"branch-one"))
        self.assertEqual(len(reference), 12)
        self.assertNotEqual(reference, branch_reference(b"branch-two"))
        self.assertNotIn("@", reference)

    def test_empty_sources_produce_complete_json_report(self):
        report = collect_overview_report(self.sources)
        self.assertEqual(json.loads(json.dumps(report)), report)
        self.assertEqual(set(report), {
            "schema_version", "report_kind", "generated_at", "branch_target",
            "filters", "tree", "sources", "data",
        })
        self.assertFalse(report["tree"])
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertTrue(report["sources"]["telemetry"]["ok"])
        self.assertTrue(report["sources"]["cache"]["ok"])
        self.assertEqual(report["data"]["branches"], [])
        self.assertEqual(report["data"]["workers"], [])

    def test_queue_and_cache_fail_independently(self):
        missing_directory = os.path.join(self.temporary_directory.name, "missing")
        unavailable_queue = ReportSources(
            queue_path=os.path.join(missing_directory, "queue.sqlite3"),
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=os.path.join(missing_directory, "telemetry.sqlite3"),
        )
        report = collect_overview_report(unavailable_queue)
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertFalse(report["sources"]["telemetry"]["ok"])
        self.assertTrue(report["sources"]["cache"]["ok"])

        unavailable_cache = ReportSources(
            queue_path=self.queue_path,
            cache_path=os.path.join(missing_directory, "cache.sqlite3"),
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=self.telemetry_path,
        )
        report = collect_overview_report(unavailable_cache)
        self.assertTrue(report["sources"]["queue"]["ok"])
        self.assertFalse(report["sources"]["cache"]["ok"])

    def test_answer_list_failure_skips_queue_and_cache_normalization(self):
        with (
            patch("report_model.load_word_list", side_effect=OSError("missing answers")),
            patch("report_model.ERDQueue") as queue_class,
        ):
            report = collect_overview_report(self.sources)
        queue_class.assert_not_called()
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertIn("missing answers", report["sources"]["queue"]["error"])
        self.assertFalse(report["sources"]["cache"]["ok"])
        self.assertIn("missing answers", report["sources"]["cache"]["error"])
        self.assertEqual(report["data"]["branches"], [])
        self.assertEqual(report["data"]["workers"], [])

    def test_queue_collection_error_does_not_relabel_attached_telemetry(self):
        with patch.object(
            ERDQueue, "counts_by_status",
            side_effect=sqlite3.OperationalError("queue read failed"),
        ):
            report = collect_overview_report(self.sources)
        self.assertFalse(report["sources"]["queue"]["ok"])
        self.assertIn("queue read failed", report["sources"]["queue"]["error"])
        self.assertTrue(report["sources"]["telemetry"]["ok"])
        self.assertIsNone(report["sources"]["telemetry"]["error"])

    def test_programming_error_in_queue_collection_is_not_masked(self):
        with patch.object(ERDQueue, "counts_by_status", side_effect=KeyError("bug")):
            with self.assertRaises(KeyError):
                collect_overview_report(self.sources)

    def test_epoch_metadata_is_queue_source_metadata(self):
        queue = self._open_queue()
        queue.set_epoch(7, label="bounded-claims", git_sha="abcdef12")
        queue.close()
        report = collect_overview_report(self.sources)
        self.assertEqual(report["sources"]["queue"]["epoch"], 7)
        self.assertEqual(report["sources"]["queue"]["label"], "bounded-claims")
        self.assertEqual(report["sources"]["queue"]["git_sha"], "abcdef12")
        self.assertEqual(set(report["sources"]["telemetry"]), {
            "path", "ok", "error",
        })

    def test_custom_paths_are_retained_without_erd_search_ownership(self):
        self.assertEqual(self.sources.answer_list_path, self.answer_list_path)
        self.assertEqual(
            self.sources.candidate_list_path, self.candidate_list_path
        )
        report = collect_overview_report(self.sources)
        self.assertEqual(report["sources"]["queue"]["path"], self.queue_path)
        self.assertEqual(report["sources"]["cache"]["path"], self.cache_path)

    def test_active_branch_and_worker_normalize_domain_fields(self):
        now = int(time.time())
        branch_key = ScoreCache.encode_subset(ANSWERS[:3])
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 9, "SALET", 0)])
        queue.claim_next("worker-2")
        queue.create_branch(
            branch_key, 3, 10, priority=9, source_word="SALET",
            source_pattern=0, budget=4, spine="SALET ----- CRANE y----",
        )
        queue._conn.execute(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done, done_at) "
            "VALUES (?, 0, 'worker-2', ?, 1, ?)",
            (queue._intern_branch(branch_key), now, now),
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 2, "
            "best_guess = 'CRANE', best_erd = 2.25, best_max_depth = 3, "
            "nodes_spent = 1234 WHERE branch_id = ?",
            (queue._intern_branch(branch_key),),
        )
        queue.heartbeat(
            "worker-2", pid=42, current_branch_key=branch_key, n_words=3,
            started_at=now, claims_done=7, claim_idx=4,
            claim_started_at=now, cache_hits=5, cache_misses=2, n_ok=3,
            n_cutoff=4, n_pruned=1, best_guess="CRANE", best_erd=2.25,
            bound_erd=2.5, cur_candidate="NURDY", cur_max_depth=5,
            cur_nodes=900, node_rate=45.5,
            cur_path="3:KHAKI:--y--/3→4:NURDY:---y-/2",
        )
        queue.close()

        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        branch = report["data"]["branches"][0]
        worker = report["data"]["workers"][0]
        self.assertEqual(branch["branch_status"], "active")
        self.assertEqual(branch["branch_phase"], "evaluating")
        self.assertEqual(branch["raw_status"], "in_progress")
        self.assertFalse(branch["is_cooperative"])
        self.assertEqual(branch["guess_depth"], 2)
        self.assertEqual(branch["source_pattern"], "-----")
        self.assertEqual(branch["completed_candidate_count"], 1)
        self.assertEqual(branch["bulk_completed_candidate_count"], 2)
        self.assertEqual(branch["best_max_remaining_depth"], 3)
        self.assertNotIn("best_max_depth", branch)
        self.assertTrue(branch["best_guess_is_answer"])
        self.assertEqual(worker["candidate_index"], 4)
        self.assertNotIn("claim_idx", worker)
        self.assertEqual(worker["current_max_guess_depth"], 5)
        self.assertNotIn("cur_max_depth", worker)
        self.assertTrue(worker["current_candidate_is_answer"])
        self.assertTrue(worker["descent"][0]["word_is_answer"])
        self.assertEqual(report["data"]["worker_totals"]["cache_hit_count"], 5)
        self.assertGreater(report["data"]["disk"]["total_bytes"], 0)
        self.assertGreaterEqual(report["data"]["disk"]["queue_wal_bytes"], 0)

    def test_all_gray_legacy_spine_fallback_has_one_guess(self):
        branch_key = b"legacy-gray"
        queue = self._open_queue()
        queue.create_branch(
            branch_key, 2, 4, source_word="SALET", source_pattern=0
        )
        queue.close()
        report = collect_overview_report(self.sources)
        branch = report["data"]["branches"][0]
        self.assertEqual(branch["guess_depth"], 1)
        self.assertEqual(branch["spine"][0]["pattern"], "-----")

    def test_cooperative_branch_is_active_and_counted(self):
        queue = self._open_queue()
        queue.create_branch(b"cooperative", 3, 5)
        queue.close()
        report = collect_overview_report(self.sources)
        self.assertTrue(report["data"]["branches"][0]["is_cooperative"])
        self.assertEqual(
            report["data"]["queue_counts"]["evaluating_cooperative_branch_count"], 1
        )

    def test_branch_report_lists_live_off_branch_claim_holders(self):
        now = int(time.time())
        parent_key = b"parent-branch"
        child_key = b"child-branch"
        queue = self._open_queue()
        queue.create_branch(
            parent_key, 3, 10, source_word="SALET", source_pattern=0, budget=5
        )
        queue.create_branch(
            child_key, 2, 10, source_word="SALET", source_pattern=6, budget=4
        )
        parent_id = queue._intern_branch(parent_key)
        queue._conn.execute(
            "INSERT INTO candidate_claims "
            "(branch_id, idx, claimed_by, claimed_at, done) "
            "VALUES (?, 17, 'worker-5', ?, 0)",
            (parent_id, now),
        )
        queue.heartbeat(
            "worker-5", pid=55, current_branch_key=child_key, n_words=2,
            started_at=now, claims_done=9, claim_idx=3, claim_started_at=now,
            cur_candidate="BEEFY", cur_max_depth=5,
        )
        queue.close()
        report = collect_report(self.sources, ReportRequest(
            report_kind="auto",
            branch_target=parse_report_branch_target(
                "@" + branch_reference(parent_key)
            ),
            filters=ReportFilters(),
        ))
        ownership = report["data"]["branch_ownership"]
        self.assertEqual(ownership["live_workers"], [])
        self.assertEqual(
            [worker["worker_id"] for worker in ownership["claim_holders_off_branch"]],
            ["worker-5"],
        )
        off_branch_holder = ownership["claim_holders_off_branch"][0]
        self.assertEqual(off_branch_holder["branch_reference"], branch_reference(child_key))
        self.assertEqual(off_branch_holder["in_flight_claim_count"], 1)
        self.assertEqual(off_branch_holder["branch_context"][0]["word"], "salet")
        self.assertEqual(off_branch_holder["branch_context"][0]["pattern"], "---g-")

    def test_finalizing_branch_remains_visible_after_worker_departure(self):
        now = int(time.time())
        live_key = b"live-finalizing"
        dead_key = b"dead-finalizing"
        queue = self._open_queue()
        for key, worker_id in ((live_key, "worker-1"), (dead_key, "worker-3")):
            queue.create_branch(key, 2, 1)
            queue.try_finalize_branch(key)
            queue.heartbeat(worker_id, 10, key, 2, now, 0)
        queue._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = 'worker-3'",
            (now - 31,),
        )
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        self.assertEqual(len(report["data"]["branches"]), 2)
        branches = {
            row["branch_key_hex"]: row for row in report["data"]["branches"]
        }
        self.assertEqual(branches[live_key.hex()]["branch_status"], "active")
        self.assertEqual(branches[live_key.hex()]["branch_phase"], "finalizing")
        self.assertEqual(branches[dead_key.hex()]["branch_status"], "pending")
        self.assertEqual(branches[dead_key.hex()]["branch_phase"], "finalizing")
        self.assertEqual(
            set(branches), {live_key.hex(), dead_key.hex()}
        )
        self.assertEqual(report["data"]["queue_counts"]["finalizing_branch_count"], 2)
        self.assertEqual(len(report["data"]["workers"]), 2)
        self.assertEqual(sum(worker["is_live"] for worker in report["data"]["workers"]), 1)

    def test_worker_on_removed_branch_is_not_on_a_live_branch(self):
        now = int(time.time())
        live_key = b"live-branch-xx"
        removed_key = b"gone-branch-xx"
        queue = self._open_queue()
        queue.create_branch(live_key, 2, 4)
        queue.heartbeat("worker-1", 1, live_key, 2, now, 0)
        # worker-2 still names a branch that has no pending or active row, as
        # happens for one heartbeat interval after a branch is finalized and
        # removed while the worker moves on to its next claim.
        queue.heartbeat("worker-2", 2, removed_key, 2, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        branch_keys = {
            row["branch_key_hex"] for row in report["data"]["branches"]
        }
        self.assertEqual(branch_keys, {live_key.hex()})
        workers = {
            worker["worker_id"]: worker for worker in report["data"]["workers"]
        }
        self.assertTrue(workers["worker-1"]["on_active_branch"])
        self.assertFalse(workers["worker-2"]["on_active_branch"])

    def test_idle_worker_is_not_on_a_live_branch(self):
        now = int(time.time())
        queue = self._open_queue()
        queue.heartbeat("worker-1", 1, None, 0, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        worker = report["data"]["workers"][0]
        self.assertIsNone(worker["branch_key_hex"])
        self.assertFalse(worker["on_active_branch"])

    def test_branch_worker_without_candidate_is_coordinating(self):
        now = int(time.time())
        key = b"coord-branch-x"
        queue = self._open_queue()
        queue.create_branch(key, 2, 4)
        queue.heartbeat("worker-1", 1, key, 2, now, 0, cur_candidate="crane")
        queue.heartbeat("worker-2", 2, key, 2, now, 0)
        queue.close()
        with patch("report_model.time.time", return_value=now):
            report = collect_overview_report(self.sources)
        workers = {
            worker["worker_id"]: worker for worker in report["data"]["workers"]
        }
        self.assertEqual(workers["worker-1"]["state"], "working")
        self.assertEqual(workers["worker-2"]["state"], "coordinating")

    def test_overview_status_filter_tracks_worker_arrival_and_departure(self):
        now = int(time.time())
        working_key = b"working-branch"
        waiting_key = b"waiting-branch"
        queue = self._open_queue()
        queue.create_branch(working_key, 2, 4)
        queue.create_branch(waiting_key, 3, 4)
        queue.heartbeat("worker-1", 1, working_key, 2, now, 0)
        queue.close()

        active_request = ReportRequest(filters=ReportFilters(
            branch_statuses=("active",)
        ))
        with patch("report_model.time.time", return_value=now):
            active_report = collect_report(self.sources, active_request)
        self.assertEqual(
            [row["branch_key_hex"] for row in active_report["data"]["branches"]],
            [working_key.hex()],
        )

        queue = self._open_queue()
        queue._conn.execute(
            "UPDATE worker_heartbeat SET updated_at = ? WHERE worker_id = ?",
            (now - WORKER_LIVENESS_SECONDS - 1, "worker-1"),
        )
        queue.close()
        with patch("report_model.time.time", return_value=now):
            active_report = collect_report(self.sources, active_request)
            pending_report = collect_report(self.sources, ReportRequest(
                filters=ReportFilters(branch_statuses=("pending",), limit=1)
            ))
        self.assertEqual(active_report["data"]["branches"], [])
        self.assertEqual(len(pending_report["data"]["branches"]), 1)
        self.assertEqual(
            pending_report["data"]["branches"][0]["branch_status"], "pending"
        )

    def test_pending_overview_includes_scheduled_branch_before_evaluation(self):
        branch_key = b"queued-branch"
        queue = self._open_queue()
        queue.add_pending_many([(branch_key, 3, 5, "raise", 0)])
        queue.close()

        report = collect_report(self.sources, ReportRequest(
            filters=ReportFilters(branch_statuses=("pending",))
        ))
        self.assertEqual(len(report["data"]["branches"]), 1)
        branch = report["data"]["branches"][0]
        self.assertEqual(branch["branch_key_hex"], branch_key.hex())
        self.assertEqual(branch["branch_status"], "pending")
        self.assertEqual(branch["branch_phase"], "queued")
        self.assertIsNone(branch["candidate_count"])

    def test_candidate_progress_batch_handles_all_states(self):
        first_key = b"first"
        second_key = b"second"
        missing_key = b"missing"
        queue = self._open_queue()
        queue.create_branch(first_key, 2, 4)
        queue.create_branch(second_key, 2, 4)
        queue._conn.execute(
            "INSERT INTO candidate_claims (branch_id, idx, done) VALUES (?, 0, 1)",
            (queue._intern_branch(first_key),),
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 3 "
            "WHERE branch_id = ?", (queue._intern_branch(second_key),),
        )
        progress = queue.candidate_progress_by_branch_keys(
            [first_key, second_key, missing_key]
        )
        self.assertEqual(queue.candidate_progress_by_branch_keys([]), {})
        self.assertEqual(progress[first_key]["completed_candidate_count"], 1)
        self.assertEqual(progress[second_key]["bulk_completed_candidate_count"], 3)
        self.assertEqual(progress[missing_key], {
            "completed_candidate_count": 0,
            "bulk_completed_candidate_count": 0,
        })
        queue.close()

    def test_cache_summary_and_dispatch(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        now = int(time.time())
        cache._conn.execute(
            "INSERT INTO branch_best_by_policy "
            "(branch_key, policy, answer_list_id, best_guess, best_score, updated_at) "
            "VALUES (?, ?, ?, 'salet', 2.0, ?)",
            (b"exact", ERD_ALL, cache.answer_list_id, now),
        )
        cache._conn.execute(
            "INSERT INTO branch_loss_by_policy "
            "(branch_key, policy, answer_list_id, loss_budget, updated_at) "
            "VALUES (?, ?, ?, 3, ?)",
            (b"loss", ERD_ALL, cache.answer_list_id, now),
        )
        self.assertEqual(cache.erd_report_summary(ERD_ALL, now - 1), {
            "exact_branch_count": 1,
            "recent_exact_branch_count": 1,
            "loss_branch_count": 1,
        })
        cache.close()
        report = collect_report(self.sources, ReportRequest())
        self.assertEqual(report["data"]["cache_summary"]["exact_branch_count"], 1)
        self.assertEqual(ReportRequest().report_kind, "auto")
        with self.assertRaisesRegex(ValueError, "unsupported report kind: hotspot"):
            collect_report(self.sources, ReportRequest("hotspot"))


if __name__ == "__main__":
    unittest.main()
