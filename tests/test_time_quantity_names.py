"""Columns that hold a duration name the time they measure, not only its unit."""

import os
import re
import sqlite3
import tempfile
import unittest

from cache_sqlite import ScoreCache
from erd_queue import ERDQueue, derive_telemetry_path

ANSWERS = ["crane", "slate", "trace"]

# A unit with no quantity in front of it: `wall_millis`, but not
# `wall_time_millis`, `duration_millis` or `timeout_millis`.
_BARE_UNIT = re.compile(r"(?<!_time)(?<!duration)(?<!timeout)(?<!delay)"
                        r"(?<!interval)_millis$")


def _columns(path):
    connection = sqlite3.connect(path)
    try:
        return {
            (table, column)
            for (table,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")
            for column in (row[1] for row in connection.execute(
                f"PRAGMA table_info({table})"))
        }
    finally:
        connection.close()


class _Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.queue_path = os.path.join(self._tmp.name, "queue.sqlite3")
        self.telemetry_path = derive_telemetry_path(self.queue_path)
        self.cache_path = os.path.join(self._tmp.name, "cache.sqlite3")


class TestNoColumnNamesOnlyItsUnit(_Files):

    def test_every_schema_names_its_time_quantities(self):
        ERDQueue(self.queue_path).close()
        ScoreCache(self.cache_path, ANSWERS,
                   checkpoint_on_close=False).close()
        bare = sorted(
            f"{os.path.basename(path)}: {table}.{column}"
            for path in (self.queue_path, self.telemetry_path,
                         self.cache_path)
            for table, column in _columns(path)
            if _BARE_UNIT.search(column))
        self.assertEqual(bare, [])

    def test_every_renamed_column_has_a_migration(self):
        # A column the map leaves out keeps its old name on every existing
        # file.  Only these were created with the name they have now.
        created_named = {
            ("bundle_stats", "evaluation_time_millis"),
            ("branch_finalize_log", "evaluation_time_millis"),
        }
        ERDQueue(self.queue_path).close()
        ScoreCache(self.cache_path, ANSWERS,
                   checkpoint_on_close=False).close()
        queue_named = {
            (table, column)
            for path in (self.queue_path, self.telemetry_path)
            for table, column in _columns(path)
            if column.endswith("_time_millis")}
        queue_migrated = {
            (table, new) for _schema, table, renames
            in ERDQueue.TIME_QUANTITY_RENAMES for new in renames.values()}
        self.assertEqual(queue_named - created_named, queue_migrated)
        cache_named = {
            (table, column) for table, column in _columns(self.cache_path)
            if column.endswith("_time_millis")}
        cache_migrated = {
            (table, new) for table, renames
            in ScoreCache.TIME_QUANTITY_RENAMES for new in renames.values()}
        self.assertEqual(cache_named - {
            ("completed_opener_summaries", "evaluation_time_millis"),
            ("completed_opener_summaries", "coordination_time_millis"),
        }, cache_migrated)

    def test_the_pattern_tells_a_bare_unit_from_a_named_time(self):
        self.assertTrue(_BARE_UNIT.search("wall_millis"))
        self.assertFalse(_BARE_UNIT.search("wall_time_millis"))
        self.assertFalse(_BARE_UNIT.search("duration_millis"))


class TestQueueRenamesItsTimeColumns(_Files):

    def _revert(self):
        """Give every renamed column its old name back, as a file written by
        code from before the rename carries them."""
        connection = sqlite3.connect(self.queue_path)
        connection.execute(
            f"ATTACH DATABASE '{self.telemetry_path}' AS telemetry")
        try:
            for schema, table, renames in ERDQueue.TIME_QUANTITY_RENAMES:
                for old, new in renames.items():
                    connection.execute(
                        f"ALTER TABLE {schema}.{table} "
                        f"RENAME COLUMN {new} TO {old}")
            connection.commit()
        finally:
            connection.close()

    def test_an_old_file_is_renamed_and_keeps_its_rows(self):
        queue = ERDQueue(self.queue_path)
        queue.add_branch_finalize_log(
            b"key", "CRANE -----", 3, 4, 10, 20, 30, 2,
            total_bundle_wall_time_millis=500, cache_write_time_millis=7,
            coordination_time_millis=40, evaluation_time_millis=450)
        queue.add_checkpoint_pause(100.0, 1_500, 4096, True)
        queue.close()
        self._revert()
        reverted = _columns(self.telemetry_path)
        self.assertIn(("branch_finalize_log", "cache_write_millis"), reverted)
        self.assertIn(("worker_time", "interval_millis"), reverted)

        queue = ERDQueue(self.queue_path)
        try:
            finalized = queue._conn.execute(
                "SELECT total_bundle_wall_time_millis, cache_write_time_millis,"
                " coordination_time_millis "
                "FROM telemetry.branch_finalize_log").fetchone()
            paused = queue._conn.execute(
                "SELECT pause_time_millis FROM telemetry.checkpoint_pause"
            ).fetchone()
        finally:
            queue.close()
        self.assertEqual(tuple(finalized), (500, 7, 40))
        self.assertEqual(paused[0], 1_500)
        renamed = _columns(self.queue_path) | _columns(self.telemetry_path)
        for schema, table, renames in ERDQueue.TIME_QUANTITY_RENAMES:
            for old, new in renames.items():
                with self.subTest(table=table, column=old):
                    self.assertIn((table, new), renamed)
                    self.assertNotIn((table, old), renamed)

    def test_reopening_a_renamed_file_changes_nothing(self):
        ERDQueue(self.queue_path).close()
        self._revert()
        ERDQueue(self.queue_path).close()
        before = _columns(self.queue_path) | _columns(self.telemetry_path)
        ERDQueue(self.queue_path).close()
        self.assertEqual(
            _columns(self.queue_path) | _columns(self.telemetry_path), before)


class TestCacheRenamesItsSummaryColumns(_Files):

    def test_an_old_cache_is_renamed_and_keeps_its_rows(self):
        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        cache.write_completed_opener_summary(
            "crane", "erd_all", 1_000, 60_000, 42_000, (22,))
        cache.add_opener_response_group_summary(
            "crane", "-----", "erd_all", 10, 9_000, 100, 200, 22)
        cache.close()
        connection = sqlite3.connect(self.cache_path)
        for table, renames in ScoreCache.TIME_QUANTITY_RENAMES:
            for old, new in renames.items():
                connection.execute(
                    f"ALTER TABLE {table} RENAME COLUMN {new} TO {old}")
        connection.execute(
            "DELETE FROM schema_migrations WHERE name = 'name_time_quantities'")
        connection.commit()
        connection.close()

        cache = ScoreCache(self.cache_path, ANSWERS, checkpoint_on_close=False)
        try:
            summary = cache._conn.execute(
                "SELECT elapsed_time_millis, worker_time_millis "
                "FROM completed_opener_summaries").fetchone()
            group = cache._conn.execute(
                "SELECT worker_time_millis "
                "FROM opener_response_group_summaries").fetchone()
        finally:
            cache.close()
        self.assertEqual(tuple(summary), (60_000, 42_000))
        self.assertEqual(group[0], 9_000)


if __name__ == "__main__":
    unittest.main()
