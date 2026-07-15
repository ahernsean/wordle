"""Tests for semantic word and branch report objects."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue
from report_model import (
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

    def test_semantic_queue_programming_errors_propagate(self):
        with patch.object(
            ERDQueue, "status_by_branch_keys", side_effect=KeyError("bug")
        ):
            with self.assertRaises(KeyError):
                collect_report(self.sources, ReportRequest(
                    selector=parse_report_selector("RAISE")
                ))

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
        with self.assertRaisesRegex(ValueError, "not found") as raised:
            resolve_branch_reference(no_matches, "1234")
        self.assertIn("cache-only state", str(raised.exception))

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
            bytes(resolve_branch_reference(
                queue, branch_reference(first_key)[:6]
            )["branch_key"]),
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
        legacy_key = ScoreCache.encode_subset(ANSWERS[9:12])
        cache._conn.execute(
            """INSERT INTO branch_best_by_policy
               (branch_key, policy, answer_list_id, best_guess, best_score,
                updated_at, max_depth, solve_budget)
               VALUES (?, ?, ?, 'heath', 2.4, ?, NULL, NULL)""",
            (legacy_key, ERD_ALL, cache.answer_list_id, int(time.time())),
        )
        self.assertEqual(
            cache.report_branch_state(legacy_key, ERD_ALL)["cache_state"],
            "missing",
        )
        cache.close()


if __name__ == "__main__":
    unittest.main()
