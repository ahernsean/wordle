import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import erd_search
from cache_sqlite import ScoreCache
from erd_queue import ERDQueue


WORDS = ["crane", "slate", "trace", "stale", "tales"]


def _words(prefix, count):
    return [f"{prefix}{i:04d}"[:5] for i in range(count)]


class TestQueueBranchRefParser(unittest.TestCase):
    def test_accepts_partial_and_full_spines(self):
        self.assertEqual(
            erd_search.parse_queue_branch_ref("CRANE"),
            [("CRANE", None)])
        self.assertEqual(
            erd_search.parse_queue_branch_ref("CRANE -y--g"),
            [("CRANE", "-y--g")])
        self.assertEqual(
            erd_search.parse_queue_branch_ref("CRANE -y--g ALIBI"),
            [("CRANE", "-y--g"), ("ALIBI", None)])
        self.assertEqual(
            erd_search.parse_queue_branch_ref("CRANE -y--g ALIBI g-g--"),
            [("CRANE", "-y--g"), ("ALIBI", "g-g--")])

    def test_normalizes_lowercase_and_gray_chars(self):
        self.assertEqual(
            erd_search.parse_queue_branch_ref("crane .yxxg alibi 00000"),
            [("CRANE", "-y--g"), ("ALIBI", "-----")])
        self.assertEqual(
            erd_search.parse_queue_branch_ref("CRANE xxxxx ALIBI gyxgg"),
            [("CRANE", "-----"), ("ALIBI", "gy-gg")])

    def test_rejects_malformed_refs(self):
        bad = ["CRAN", "CRANE ALIBI", "CRANE -y--g DOG", "CRANE -y--"]
        for ref in bad:
            with self.subTest(ref=ref):
                with self.assertRaises(erd_search.QueueRefError):
                    erd_search.parse_queue_branch_ref(ref)


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
        self.assertEqual(rows[0]["source_pattern_text"], "----y")

    def test_user_in_progress_joins_pending_and_active_state(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 5, "crane", 1)])
        self.q.claim_next("worker-0")
        self.q.create_branch(
            self.user_key, len(WORDS), 20, priority=5,
            source_word="crane", source_pattern=1,
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
            source_word="alibi", source_pattern=42,
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

    def test_short_branch_id_resolves(self):
        self.q.add_pending_many([(self.user_key, len(WORDS), 0, "crane", 0)])
        row = self.q.list_queue_rows()[0]
        bid = erd_search._branch_id(row["branch_key"])
        matches = self.q.resolve_branch_ref(bid)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["branch_key"], self.user_key)

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
            source_word="crane", source_pattern=1, budget=4,
            spine="CRANE ----y")

        self.assertEqual(self.q.list_queue_rows({"status": "done"}), [])
        self.assertEqual(self.q.list_queue_rows({"min_words": 6}), [])
        self.assertEqual(self.q.list_queue_rows({"max_words": 4}), [])
        self.assertEqual(self.q.list_queue_rows({"budget": 3}), [])
        self.assertEqual(self.q.list_queue_rows({"priority": 8}), [])
        self.assertEqual(self.q.list_queue_rows({"source_word": "slate"}), [])
        self.assertEqual(self.q.list_queue_rows({"prefix": "SLATE -----"}), [])
        self.assertEqual(
            len(self.q.list_queue_rows({
                "status": "pending",
                "min_words": 5,
                "max_words": 5,
                "budget": 4,
                "priority": 7,
                "source_word": "crane",
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
            source_word="alibi", source_pattern=42, budget=2,
            spine="CRANE -y--g ALIBI g-g--")
        self.q.add_nodes_spent(small_key, 123)
        self.q.update_branch_best(small_key, "crane", 1.25, max_depth=3)
        self.q.mark_branch_tainted(small_key)
        self.q.mark_claims_done(small_key, [0, 1])
        self.q.record_bundle_stats(small_key, "bundle-1", 100, 50)
        self.q._conn.execute(
            "INSERT INTO candidate_republish (branch_key, idx, count) "
            "VALUES (?, 2, 1)", (small_key,))
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
        self.assertEqual(summary["by_priority"]["coop"], 1)
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
                "source_word": "crane",
                "source_pattern_text": "-----",
            }),
            "CRANE -----")
        self.assertEqual(self.q.row_spine_text({}), "")

    def test_queue_table_columns_accommodate_rendered_values(self):
        rows = [
            {
                "branch_key": b"first",
                "kind": "coop",
                "status": "open",
                "priority": 1_000_000,
                "n_words": 60,
                "done_candidates": 12616,
                "n_candidates": 12972,
                "worker_count": 4,
                "nodes_spent": 1732478,
                "spine": "ALIBI -----",
            },
            {
                "branch_key": b"second",
                "kind": "user",
                "status": "in_progress",
                "priority": 170000,
                "n_words": 841,
                "done_candidates": 26,
                "n_candidates": 12972,
                "worker_count": 0,
                "nodes_spent": 2748659,
                "spine": "CRANE -----",
            },
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            erd_search._print_queue_table(rows)
        lines = output.getvalue().splitlines()

        self.assertEqual(
            lines,
            [
                "ID   Kind Status         Pri Words        Done W   Nodes  Spine",
                f"{erd_search._branch_id(b'first')} coop open          COOP    60 "
                "12616/12972 4 1732478  ALIBI -----",
                f"{erd_search._branch_id(b'second')} user in_progress 170000   841 "
                "   26/12972 0 2748659  CRANE -----",
            ],
        )


class QueueCliArgparseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "q.sqlite3")
        q = ERDQueue(self.queue_path)
        try:
            q.add_pending_many([
                (ScoreCache.encode_subset(WORDS), len(WORDS), 0, "crane", 0),
                (ScoreCache.encode_subset(WORDS[:3]), 3, 0, "slate", 0),
            ])
        finally:
            q.close()

    def _run_main(self, *argv):
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["erd_search.py", *argv]):
            with redirect_stdout(buf):
                erd_search.main()
        return buf.getvalue()

    def test_queue_global_json_applies_to_child_command(self):
        out = self._run_main(
            "queue", "--queue", self.queue_path, "--json", "ls")
        rows = json.loads(out)
        self.assertEqual(len(rows), 2)

    def test_queue_global_limit_applies_to_child_command(self):
        out = self._run_main(
            "queue", "--queue", self.queue_path, "--limit", "1", "ls", "--json")
        rows = json.loads(out)
        self.assertEqual(len(rows), 1)

    def test_child_queue_option_overrides_queue_global(self):
        other_path = os.path.join(self._tmp.name, "other.sqlite3")
        out = self._run_main(
            "queue", "--queue", other_path, "ls",
            "--queue", self.queue_path, "--json")
        rows = json.loads(out)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
