"""Cross-layer contract tests: report_model's collectors piped straight into
report_terminal's renderers.

report_model and report_terminal are each tested in isolation elsewhere
(tests/test_report_model.py, tests/test_report_objects.py,
tests/test_report_terminal.py), against fixtures hand-built to match
whatever shape each side currently assumes. That isolation is exactly what
let a real regression through once: collect_queue_report's row["spine"]
changed from a flat "WORD pattern" string to a structured step list, and
report_terminal's queue renderer still called .split() on it, but every
existing test on both sides kept passing because neither side's fixture
was ever produced by the other side's code.

These tests close that gap by calling collect_report/collect_overview_report
for real and feeding the actual result into report_terminal.render_report,
so a shape mismatch between the two modules raises here.
"""

import os
import tempfile
import time
import unittest

from cache_sqlite import ScoreCache, branch_reference
from erd_queue import ERDQueue
from report_model import (
    ReportFilters,
    ReportRequest,
    ReportSources,
    collect_overview_report,
    collect_report,
    parse_report_branch_target,
)
from report_terminal import render_report
from wordle_engine import ERD_ALL


ANSWERS = ["salet", "crane", "nurdy", "khaki"]


class ReportPipelineTest(unittest.TestCase):
    """Builds one non-trivial queue + cache and renders every report kind."""

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
            candidate_file.write("\n".join(ANSWERS) + "\n")
        self.sources = ReportSources(
            queue_path=self.queue_path,
            cache_path=self.cache_path,
            answer_list_path=self.answer_list_path,
            candidate_list_path=self.candidate_list_path,
            telemetry_path=self.telemetry_path,
        )

        now = int(time.time())
        self.branch_key = ScoreCache.encode_subset(ANSWERS[:3])
        queue = ERDQueue(self.queue_path, telemetry_path=self.telemetry_path)
        queue.add_pending_many([(self.branch_key, 3, 9, "salet", 0)])
        queue.claim_next("worker-2")
        # A real two-step spine and a best_guess that is itself an answer
        # word -- the two fields whose shape broke report_terminal.py.
        queue.create_branch(
            self.branch_key, 3, 10, priority=9, source_word="salet",
            source_pattern=0, budget=4, spine="salet ----- crane y----",
        )
        queue._conn.execute(
            "UPDATE active_branches SET bulk_done_candidates = 1, "
            "best_guess = 'crane', best_erd = 2.25, best_max_depth = 3, "
            "nodes_spent = 500 WHERE branch_id = ?",
            (queue._intern_branch(self.branch_key),),
        )
        queue.heartbeat(
            "worker-2", pid=42, current_branch_key=self.branch_key, n_words=3,
            started_at=now, claims_done=1, claim_idx=0, claim_started_at=now,
            best_guess="crane", best_erd=2.25, cur_candidate="nurdy",
            cur_max_depth=5, cur_nodes=10, node_rate=5.0,
        )
        queue.close()

        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write(self.branch_key, ERD_ALL, "crane", 2.25, max_depth=3)
        cache.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _render(self, request):
        report = collect_report(self.sources, request)
        return report, render_report(report)

    def test_overview_report_renders(self):
        report = collect_overview_report(self.sources)
        text = render_report(report)
        self.assertIn("CRANE*", text)
        self.assertIn("w2", text)

    def test_queue_report_renders_with_a_real_spine_and_best_guess(self):
        report, text = self._render(ReportRequest(report_kind="queue"))
        self.assertEqual(report["report_kind"], "queue")
        self.assertIn("guess_depth=2", text)

    def test_workers_report_renders(self):
        report, text = self._render(ReportRequest(report_kind="workers"))
        self.assertEqual(report["report_kind"], "workers")
        self.assertIn("w2", text)

    def test_cache_report_renders(self):
        report, text = self._render(ReportRequest(report_kind="cache"))
        self.assertEqual(report["report_kind"], "cache")
        self.assertIn("exact=1", text)

    def test_hotspots_report_renders(self):
        report, text = self._render(ReportRequest(report_kind="hotspots"))
        self.assertEqual(report["report_kind"], "hotspots")
        self.assertIsInstance(text, str)

    def test_leaderboard_report_renders(self):
        report, text = self._render(ReportRequest(report_kind="leaderboard"))
        self.assertEqual(report["report_kind"], "leaderboard")
        self.assertIsInstance(text, str)

    def test_word_report_renders(self):
        report, text = self._render(ReportRequest(
            branch_target=parse_report_branch_target("salet"),
        ))
        self.assertEqual(report["report_kind"], "word")
        self.assertIn("SALET*", text)

    def test_branch_report_renders_with_its_queued_spine(self):
        report, text = self._render(ReportRequest(
            branch_target=parse_report_branch_target(
                "@" + branch_reference(self.branch_key)
            ),
            filters=ReportFilters(),
        ))
        self.assertEqual(report["report_kind"], "branch")
        self.assertIn("SALET", text)
        self.assertIn("CRANE", text)

    def test_tree_report_renders(self):
        report, text = self._render(ReportRequest(tree=True))
        self.assertTrue(report["tree"])
        self.assertIn("SALET", text)


if __name__ == "__main__":
    unittest.main()
