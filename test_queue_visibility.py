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
