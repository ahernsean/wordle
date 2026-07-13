"""Tests for semantic word and branch report objects."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportSources,
    branch_reference,
    collect_report,
    parse_report_selector,
    resolve_branch_reference,
    resolve_selector_branch,
)
from wordle_engine import ERD_ALL, ResponseCache
from wordle_ui import fmt_pattern


ANSWERS = [
    "cigar", "rebut", "sissy", "humph", "awake", "blush", "focal",
    "evade", "naval", "serve", "heath", "dwarf", "model", "karma",
    "stink", "grade", "quiet", "bench", "abate", "feign",
]


class SemanticReportTest(unittest.TestCase):
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
            candidate_file.write(
                "\n".join(ANSWERS + ["raise", "queue", "cache"]) + "\n"
            )
        self.sources = ReportSources(
            self.queue_path,
            self.cache_path,
            self.answer_list_path,
            self.candidate_list_path,
            self.telemetry_path,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _groups(self, word="raise"):
        return ResponseCache(ANSWERS, score_cache=None).group_words(word, ANSWERS)

    def _largest_group(self, word="raise"):
        pattern_code, answer_words = max(
            self._groups(word).items(), key=lambda item: len(item[1])
        )
        return pattern_code, answer_words, ScoreCache.encode_subset(answer_words)

    def test_selector_inference_and_normalization(self):
        root = parse_report_selector(None)
        word = parse_report_selector("CRANE")
        nested_word = parse_report_selector(["CRANE", ".y..g", "ALIBI"])
        branch = parse_report_selector("CRANE -y--g ALIBI g-g--")
        reference = parse_report_selector("@8B31")
        self.assertEqual(root.kind, "root")
        self.assertEqual(word.kind, "word")
        self.assertEqual(word.trailing_word, "crane")
        self.assertEqual(nested_word.steps[0].pattern, "-y--g")
        self.assertEqual(nested_word.trailing_word, "alibi")
        self.assertEqual(branch.kind, "branch")
        self.assertEqual(len(branch.steps), 2)
        self.assertEqual(reference.kind, "branch_reference")
        self.assertEqual(reference.branch_reference, "8b31")
        with self.assertRaisesRegex(ValueError, "five-character response pattern"):
            parse_report_selector("CRANE ALIBI")

    def test_queue_and_cache_are_unreserved_words(self):
        for word in ("QUEUE", "CACHE"):
            selector = parse_report_selector(word)
            self.assertEqual(selector.kind, "word")
            self.assertEqual(selector.trailing_word, word.lower())

    def test_word_report_combines_queue_cache_and_trivial_groups(self):
        pattern_code, answer_words, branch_key = self._largest_group()
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.add_pending_many([
            (branch_key, len(answer_words), 7, "raise", pattern_code)
        ])
        queue.close()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(
            branch_key, ERD_ALL, "cigar", 2.1,
            max_depth=2, solve_budget=None,
        )
        cache.close()

        request = ReportRequest(selector=parse_report_selector("RAISE"))
        report = collect_report(self.sources, request)
        row = next(
            row for row in report["data"]["response_groups"]
            if row["branch_key_hex"] == branch_key.hex()
        )
        self.assertEqual(report["report_kind"], "word")
        self.assertEqual(row["pattern"], fmt_pattern(pattern_code))
        self.assertEqual(row["lifecycle"], "pending")
        self.assertEqual(row["cache_state"], "exact")
        self.assertNotIn("answer_words", row)
        trivial_rows = [
            group for group in report["data"]["response_groups"]
            if group["answer_count"] < 2
        ]
        self.assertTrue(trivial_rows)
        self.assertTrue(all(
            group["cache_state"] == "not_applicable" for group in trivial_rows
        ))
        self.assertGreater(
            report["data"]["response_group_counts"]["trivial_response_group_count"],
            0,
        )

    def test_semantic_cache_only_branch_and_optional_answers(self):
        pattern_code, answer_words, branch_key = self._largest_group()
        selector = parse_report_selector(f"RAISE {fmt_pattern(pattern_code)}")
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(
            branch_key, ERD_ALL, "cigar", 2.2,
            max_depth=2, solve_budget=None,
        )
        cache.close()
        report = collect_report(
            self.sources,
            ReportRequest(selector=selector, include_answers=True),
        )
        self.assertEqual(report["report_kind"], "branch")
        self.assertIsNone(report["data"]["queue"])
        self.assertEqual(report["data"]["cache"]["cache_state"], "exact")
        self.assertEqual(
            set(report["data"]["branch"]["answer_words"]), set(answer_words)
        )
        self.assertIsNone(report["data"]["claims"])

    def test_semantic_resolution_never_opens_persistent_cache(self):
        selector = parse_report_selector("RAISE -----")
        with patch.object(ScoreCache, "__init__", side_effect=AssertionError):
            resolved = resolve_selector_branch(selector, ANSWERS)
        self.assertEqual(resolved.steps, selector.steps)

    def test_digest_resolution_rejects_zero_and_multiple_matches(self):
        no_matches = unittest.mock.Mock()
        no_matches.branch_rows_for_reference_prefix.return_value = []
        with self.assertRaisesRegex(ValueError, "not found"):
            resolve_branch_reference(no_matches, "1234")

        multiple = unittest.mock.Mock()
        multiple.branch_rows_for_reference_prefix.return_value = [
            {"branch_key": b"cigarrebut", "spine": "RAISE -----"},
            {"branch_key": b"awakeblush", "spine": "CRANE y----"},
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous") as raised:
            resolve_branch_reference(multiple, "1234")
        self.assertIn("@" + branch_reference(b"cigarrebut"), str(raised.exception))

    def test_queue_prefix_helper_has_bounded_candidate_contract(self):
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        first_key = b"cigarrebut"
        second_key = b"awakeblush"
        queue.add_pending_many([
            (first_key, 2, 0, "raise", 0),
            (second_key, 2, 0, "crane", 1),
        ])
        matches = queue.branch_rows_for_reference_prefix(
            branch_reference(first_key)[:6]
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(bytes(matches[0]["branch_key"]), first_key)
        self.assertEqual(
            resolve_branch_reference(queue, branch_reference(first_key)[:6]),
            first_key,
        )
        queue.close()

    def test_reference_collection_and_done_lifecycle(self):
        pattern_code, answer_words, branch_key = self._largest_group()
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.add_pending_many([
            (branch_key, len(answer_words), 3, "raise", pattern_code)
        ])
        queue.mark_done(branch_key)
        queue.close()
        selector = parse_report_selector("@" + branch_reference(branch_key)[:6])
        report = collect_report(
            self.sources,
            ReportRequest(selector=selector, include_answers=True),
        )
        self.assertEqual(report["report_kind"], "branch")
        self.assertEqual(report["data"]["queue"]["lifecycle"], "done")
        self.assertEqual(
            set(report["data"]["branch"]["answer_words"]), set(answer_words)
        )
        self.assertEqual(
            report["data"]["branch"]["spine"][0]["pattern"],
            fmt_pattern(pattern_code),
        )

    def test_branch_claim_provenance_and_optional_detail(self):
        now = int(time.time())
        pattern_code, answer_words, branch_key = self._largest_group()
        pattern = fmt_pattern(pattern_code)
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(
            branch_key, len(answer_words), 10, source_word="raise",
            source_pattern=pattern_code, budget=5, spine=f"RAISE {pattern}",
        )
        queue._conn.executemany(
            "INSERT INTO candidate_claims "
            "(branch_key, idx, claimed_by, claimed_at, done, done_at, bundle_id) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            [
                (branch_key, 1, "bulk-elimination", now, now, None),
                (branch_key, 2, "worker-3", now, now, "bundle-1"),
                (branch_key, 3, "legacy-holder", now, now, None),
            ],
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 1 "
            "WHERE branch_key = ?", (branch_key,),
        )
        queue._conn.execute(
            "INSERT INTO candidate_republish (branch_key, idx, count) "
            "VALUES (?, 2, 4)", (branch_key,),
        )
        queue.close()
        selector = parse_report_selector(f"RAISE {pattern}")
        report = collect_report(
            self.sources, ReportRequest(selector=selector, include_claims=True)
        )
        claims = {
            claim["candidate_index"]: claim for claim in report["data"]["claims"]
        }
        self.assertEqual(claims[1]["completion_kind"], "bulk_eliminated")
        self.assertIsNone(claims[1]["worker_id"])
        self.assertEqual(claims[2]["completion_kind"], "evaluated")
        self.assertEqual(claims[2]["worker_id"], "worker-3")
        self.assertEqual(claims[2]["republish_count"], 4)
        self.assertIsNone(claims[3]["completion_kind"])
        self.assertTrue(report["data"]["provenance_unknown"])
        self.assertNotIn("answer_words", report["data"]["branch"])
        self.assertEqual(
            report["data"]["queue"]["bulk_completed_candidate_count"], 1
        )

    def test_cache_state_obeys_budget_reuse_rules(self):
        branch_key = ScoreCache.encode_subset(ANSWERS[:3])
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(
            branch_key, ERD_ALL, "cigar", 2.0,
            max_depth=4, solve_budget=None,
        )
        self.assertEqual(
            cache.report_branch_state(branch_key, ERD_ALL, 4)["cache_state"],
            "exact",
        )
        self.assertEqual(
            cache.report_branch_state(branch_key, ERD_ALL, 3)["cache_state"],
            "missing",
        )
        tainted_key = ScoreCache.encode_subset(ANSWERS[3:6])
        cache.write(
            tainted_key, ERD_ALL, "awake", 2.2,
            max_depth=3, solve_budget=4,
        )
        self.assertEqual(
            cache.report_branch_state(tainted_key, ERD_ALL, 4)["cache_state"],
            "exact",
        )
        self.assertTrue(
            cache.report_branch_state(tainted_key, ERD_ALL, 4)["tainted"]
        )
        self.assertEqual(
            cache.report_branch_state(tainted_key, ERD_ALL, 5)["cache_state"],
            "missing",
        )
        loss_key = ScoreCache.encode_subset(ANSWERS[6:9])
        cache.write_loss(loss_key, ERD_ALL, 3)
        self.assertEqual(
            cache.report_branch_state(loss_key, ERD_ALL, 2)["cache_state"],
            "loss",
        )
        self.assertEqual(
            cache.report_branch_state(loss_key, ERD_ALL, 4)["cache_state"],
            "missing",
        )
        cache.close()

    def test_queue_filters_precede_limit_and_do_not_duplicate_user_work(self):
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        user_key = b"cigarrebut"
        queue.add_pending_many([
            (user_key, 2, 5, "raise", 0),
            (b"awakeblush", 20, 4, "crane", 1),
            (b"focalserve", 200, 3, "stink", 2),
        ])
        queue.claim_next("worker-1")
        queue.create_branch(user_key, 2, 10, priority=5)
        result = queue.report_queue_rows(ReportFilters(
            minimum_answer_count=10, limit=1, sort="size"
        ))
        self.assertEqual(result["matched_rows"], 2)
        self.assertEqual(result["summary"]["branch_count"], 2)
        self.assertEqual(len(result["rows"]), 1)
        all_rows = queue.report_queue_rows()
        self.assertEqual(
            sum(row["branch_key_hex"] == user_key.hex() for row in all_rows["rows"]),
            1,
        )
        queue.create_branch(
            b"modelkarma", 2, 4, budget=5, spine="RAISE -----"
        )
        fully_filtered = queue.report_queue_rows(ReportFilters(
            statuses=("active",),
            maximum_answer_count=2,
            budget=5,
            priority=0,
        ))
        self.assertEqual(
            [row["branch_key"] for row in fully_filtered["rows"]],
            [b"modelkarma"],
        )
        scoped = queue.report_tree_rows(
            "RAISE -----", ReportFilters(sort="size", limit=1)
        )
        self.assertEqual(scoped["matched_rows"], 1)
        self.assertEqual(scoped["rows"][0]["spine"], "RAISE -----")
        queue.close()

    def test_active_only_excludes_finalizing_lifecycle(self):
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        active_key = b"cigarrebut"
        finalizing_key = b"awakeblush"
        queue.create_branch(active_key, 2, 4)
        queue.create_branch(finalizing_key, 2, 4)
        queue.try_finalize_branch(finalizing_key)
        result = queue.report_queue_rows(ReportFilters(active_only=True))
        self.assertEqual([row["branch_key"] for row in result["rows"]], [active_key])
        queue.close()

    def test_root_word_and_branch_trees_use_only_queue_spines(self):
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(
            b"cigarrebut", 2, 4, budget=5, spine="RAISE -----"
        )
        queue.create_branch(
            b"awakeblush", 2, 4, budget=4,
            spine="RAISE ----- CRANE y----",
        )
        queue.create_branch(
            b"focalserve", 2, 4, budget=5, spine="STINK g----"
        )
        queue.close()
        for selector_text, expected_word in (
            ("", None), ("RAISE", "raise"), ("RAISE -----", "raise"),
        ):
            with self.subTest(selector=selector_text):
                report = collect_report(self.sources, ReportRequest(
                    selector=parse_report_selector(selector_text), tree=True
                ))
                self.assertTrue(report["data"]["tree_available"])
                node_ids = {node["node_id"] for node in report["data"]["nodes"]}
                for node in report["data"]["nodes"]:
                    if node["parent_node_id"] is not None:
                        self.assertIn(node["parent_node_id"], node_ids)
                    self.assertGreaterEqual(node["guess_depth"], 0)
                    if expected_word and node["step"] is not None:
                        self.assertEqual(node["node_id"].split(":", 1)[0], expected_word)

    def test_tree_keeps_filtered_context_and_drops_other_descendants(self):
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(
            b"cigarrebut", 2, 4, budget=5, spine="RAISE -----"
        )
        queue.create_branch(
            b"awakeblush", 150, 4, budget=4,
            spine="RAISE ----- CRANE y----",
        )
        queue.create_branch(
            b"focalserve", 3, 4, budget=4,
            spine="RAISE ----- STINK g----",
        )
        queue.close()
        report = collect_report(self.sources, ReportRequest(
            selector=parse_report_selector("RAISE -----"),
            tree=True,
            filters=ReportFilters(minimum_answer_count=100),
        ))
        branch_nodes = [
            node for node in report["data"]["nodes"]
            if node["branch_key_hex"] is not None
        ]
        self.assertEqual(len(branch_nodes), 2)
        self.assertTrue(any(node["is_context"] for node in branch_nodes))
        self.assertTrue(any(node["answer_count"] == 150 for node in branch_nodes))
        self.assertFalse(any(node["answer_count"] == 3 for node in branch_nodes))

    def test_cache_only_tree_is_unavailable_and_legacy_spine_is_unknown(self):
        pattern_code, answer_words, branch_key = self._largest_group()
        selector = parse_report_selector(f"RAISE {fmt_pattern(pattern_code)}")
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(branch_key, ERD_ALL, "cigar", 2.0, max_depth=2)
        cache.close()
        report = collect_report(
            self.sources, ReportRequest(selector=selector, tree=True)
        )
        self.assertFalse(report["data"]["tree_available"])
        self.assertEqual(report["data"]["nodes"], [])

        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(b"legacybranchkey", 3, 4, budget=3, spine=None)
        queue.close()
        legacy_tree = collect_report(
            self.sources, ReportRequest(tree=True)
        )
        unknown_nodes = [
            node for node in legacy_tree["data"]["nodes"]
            if node["node_id"].startswith("unknown:")
        ]
        self.assertEqual(len(unknown_nodes), 3)
        self.assertTrue(all(node["step"] is None for node in unknown_nodes))

    def test_cache_distribution_helpers_name_reuse_axes(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(
            ScoreCache.encode_subset(ANSWERS[:3]), ERD_ALL, "cigar", 2.0,
            max_depth=3, solve_budget=None,
        )
        cache.write(
            ScoreCache.encode_subset(ANSWERS[3:6]), ERD_ALL, "awake", 2.2,
            max_depth=4, solve_budget=4,
        )
        cache.write_loss(ScoreCache.encode_subset(ANSWERS[6:9]), ERD_ALL, 3)
        distributions = cache.report_cache_distributions(ERD_ALL)
        self.assertEqual(
            distributions["state_branch_counts"]["exact_branch_count"], 2
        )
        self.assertEqual(
            distributions["exact_branch_count_by_taint"],
            {"untainted": 1, "tainted": 1},
        )
        self.assertEqual(distributions["loss_branch_count_by_loss_budget"]["3"], 1)
        cache.close()

    def test_queue_worker_and_cache_collectors_dispatch(self):
        now = int(time.time())
        branch_key = b"cigarrebut"
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(
            branch_key, 2, 4, budget=5, spine="RAISE -----"
        )
        queue.heartbeat("worker-2", 42, branch_key, 2, now, 0)
        queue.close()
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(branch_key, ERD_ALL, "cigar", 2.0, max_depth=2)
        cache.close()

        queue_report = collect_report(
            self.sources, ReportRequest(report_kind="queue")
        )
        workers_report = collect_report(
            self.sources, ReportRequest(report_kind="workers", worker_id="2")
        )
        cache_report = collect_report(
            self.sources, ReportRequest(report_kind="cache")
        )
        self.assertEqual(queue_report["report_kind"], "queue")
        self.assertEqual(queue_report["data"]["matched_rows"], 1)
        self.assertEqual(workers_report["report_kind"], "workers")
        self.assertEqual(workers_report["data"]["rows"][0]["worker_number"], "2")
        self.assertEqual(cache_report["report_kind"], "cache")
        self.assertEqual(cache_report["data"]["summary"]["exact_branch_count"], 1)

    def test_active_only_has_same_meaning_for_word_groups(self):
        pattern_code, answer_words, branch_key = self._largest_group()
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.create_branch(
            branch_key, len(answer_words), 4, source_word="raise",
            source_pattern=pattern_code, budget=5,
            spine=f"RAISE {fmt_pattern(pattern_code)}",
        )
        queue.close()
        report = collect_report(self.sources, ReportRequest(
            selector=parse_report_selector("RAISE"),
            filters=ReportFilters(active_only=True),
        ))
        self.assertEqual(report["data"]["matched_rows"], 1)
        self.assertTrue(all(
            row["lifecycle"] == "active"
            for row in report["data"]["response_groups"]
        ))


if __name__ == "__main__":
    unittest.main()
