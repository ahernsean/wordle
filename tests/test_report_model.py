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
    ReportRequest,
    ReportSources,
    branch_reference,
    collect_overview_report,
    collect_report,
    normalize_worker_descent,
    parse_rich_spine,
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
            "schema_version", "report_kind", "generated_at", "selector",
            "filters", "sources", "data",
        })
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
        self.assertEqual(branch["lifecycle"], "active")
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
            report["data"]["queue_counts"]["active_cooperative_branch_count"], 1
        )

    def test_only_live_heartbeat_retains_finalizing_branch(self):
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
        self.assertEqual(len(report["data"]["branches"]), 1)
        self.assertEqual(report["data"]["branches"][0]["lifecycle"], "finalizing")
        self.assertEqual(
            report["data"]["branches"][0]["branch_key_hex"], live_key.hex()
        )
        self.assertEqual(report["data"]["queue_counts"]["finalizing_branch_count"], 1)
        self.assertEqual(len(report["data"]["workers"]), 2)
        self.assertEqual(sum(worker["is_live"] for worker in report["data"]["workers"]), 1)

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
