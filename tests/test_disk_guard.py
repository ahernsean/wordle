"""Disk watermark guard and queue WAL checkpointing.

Covers the pieces that keep the queue WAL bounded and the disk from filling:
ERDQueue.checkpoint's result row (SQLite reports checkpoint contention in the
row, not as an exception), the checkpoint_pause quiesce flag, the disk-stop
latch, the supervisor's _disk_guard / _supervisor_checkpoint, and the status
display's disk line.
"""
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

import erd_search
from erd_queue import (ERDQueue, disk_stats, DISK_SAMPLE_KEEP,
                       DISK_WARN_FRACTION, DISK_STOP_FRACTION)


class _TmpQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.q = ERDQueue(self.path, timeout=0.2)
        self.addCleanup(self.q.close)


class TestDiskStats(unittest.TestCase):
    def test_fields_and_range(self):
        st = disk_stats(tempfile.gettempdir())
        self.assertGreater(st["total_bytes"], 0)
        self.assertGreaterEqual(st["used_bytes"], 0)
        self.assertGreaterEqual(st["avail_bytes"], 0)
        self.assertGreaterEqual(st["used_fraction"], 0.0)
        self.assertLessEqual(st["used_fraction"], 1.0)


class TestCheckpointResult(_TmpQueue):
    def test_uncontended_truncate_reports_not_busy(self):
        self.q.set_meta("k", "v")   # ensure the WAL has frames
        busy, log_frames, checkpointed = self.q.checkpoint("TRUNCATE")
        self.assertEqual(busy, 0)
        self.assertEqual(self.q.wal_size_bytes(), 0)

    def test_pinned_reader_reports_busy_without_raising(self):
        self.q.set_meta("k", "v")
        reader = sqlite3.connect(self.path, timeout=0.2)
        self.addCleanup(reader.close)
        # An unfinalized SELECT holds a read snapshot into the WAL, which
        # blocks the restart-and-truncate phase.
        cursor = reader.execute("SELECT key FROM run_meta")
        cursor.fetchone()
        self.q.set_meta("k2", "v2")     # WAL frames past the reader's snapshot
        result = self.q.checkpoint("TRUNCATE")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1)
        # Releasing the reader lets the next TRUNCATE complete.
        cursor.fetchall()
        reader.rollback()
        busy, _, _ = self.q.checkpoint("TRUNCATE")
        self.assertEqual(busy, 0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.q.checkpoint("truncate; DROP TABLE run_meta")


class TestCheckpointPause(_TmpQueue):
    def test_set_and_clear(self):
        self.assertFalse(self.q.checkpoint_paused())
        self.q.set_checkpoint_pause(True)
        self.assertTrue(self.q.checkpoint_paused())
        self.q.set_checkpoint_pause(False)
        self.assertFalse(self.q.checkpoint_paused())

    def test_stale_flag_is_ignored(self):
        # A flag from a supervisor that died before clearing it must not
        # wedge the swarm.
        self.q.set_meta("checkpoint_pause", str(int(time.time()) - 3600))
        self.assertFalse(self.q.checkpoint_paused())


class TestDiskStopLatch(_TmpQueue):
    def test_roundtrip_and_clear(self):
        self.assertIsNone(self.q.disk_stop())
        self.q.set_disk_stop("supervisor: disk 91.0% full")
        latch = self.q.disk_stop()
        self.assertEqual(latch["reason"], "supervisor: disk 91.0% full")
        self.assertIsInstance(latch["at"], int)
        self.q.clear_disk_stop()
        self.assertIsNone(self.q.disk_stop())

    def test_samples_ring_is_bounded(self):
        for i in range(DISK_SAMPLE_KEEP + 5):
            self.q.record_disk_sample(1000 - i)
        samples = self.q.disk_samples()
        self.assertEqual(len(samples), DISK_SAMPLE_KEEP)
        # Oldest entries fell off the front; the newest is last.
        self.assertEqual(samples[-1][1], 1000 - (DISK_SAMPLE_KEEP + 4))


class TestDiskGuard(_TmpQueue):
    def _stats(self, used_fraction):
        return {"total_bytes": 100, "used_bytes": 0, "avail_bytes": 100,
                "used_fraction": used_fraction}

    def test_below_threshold_is_quiet(self):
        with mock.patch.object(erd_search, "disk_stats",
                               return_value=self._stats(0.5)):
            self.assertFalse(erd_search._disk_guard(self.q, self.path))
        self.assertIsNone(self.q.disk_stop())

    def test_at_threshold_latches(self):
        with mock.patch.object(erd_search, "disk_stats",
                               return_value=self._stats(DISK_STOP_FRACTION)):
            self.assertTrue(erd_search._disk_guard(self.q, self.path))
        latch = self.q.disk_stop()
        self.assertIsNotNone(latch)
        self.assertIn("supervisor", latch["reason"])


class TestSupervisorCheckpoint(_TmpQueue):
    def test_quiesce_truncates_and_clears_pause(self):
        self.q.set_meta("k", "v")
        wal_before = self.q.wal_size_bytes()
        with mock.patch.object(erd_search, "QUEUE_WAL_QUIESCE_BYTES", 0):
            erd_search._maybe_quiesce_truncate(self.q)
        # Clearing the pause flag writes a frame into the fresh WAL, so the
        # file is small but not zero after the truncate.
        self.assertLess(self.q.wal_size_bytes(), wal_before)
        self.assertIsNone(self.q.get_meta("checkpoint_pause"))

    def test_below_quiesce_threshold_never_pauses(self):
        self.q.set_meta("k", "v")
        wal_before = self.q.wal_size_bytes()
        self.assertGreater(wal_before, 0)
        with mock.patch.object(self.q, "set_checkpoint_pause") as pause:
            erd_search._maybe_quiesce_truncate(self.q)
        pause.assert_not_called()
        self.assertEqual(self.q.wal_size_bytes(), wal_before)

    def test_periodic_checkpoint_is_passive_only(self):
        self.q.set_meta("k", "v")
        wal_before = self.q.wal_size_bytes()
        erd_search._supervisor_checkpoint(self.q)
        # PASSIVE backfills but never truncates the file.
        self.assertEqual(self.q.wal_size_bytes(), wal_before)
        self.assertIsNone(self.q.get_meta("checkpoint_pause"))


class TestDiskStatusLine(_TmpQueue):
    def _stats(self, used_fraction, avail=100 * 10 ** 9):
        used = int(avail * used_fraction / (1 - used_fraction))
        return {"total_bytes": used + avail, "used_bytes": used,
                "avail_bytes": avail, "used_fraction": used_fraction}

    def test_plain_below_warn(self):
        with mock.patch.object(erd_search, "disk_stats",
                               return_value=self._stats(0.5)):
            line = erd_search._disk_status_line(self.path, [])
        self.assertIn("Disk:", line)
        self.assertIn("(50%)", line)
        self.assertNotIn("\033[31m", line)

    def test_red_at_warn_threshold(self):
        with mock.patch.object(erd_search, "disk_stats",
                               return_value=self._stats(DISK_WARN_FRACTION)):
            line = erd_search._disk_status_line(self.path, [])
        self.assertIn("\033[31m", line)

    def test_fill_rate_and_eta(self):
        now = int(time.time())
        st = self._stats(0.5)
        # 30 MB/s of fill over the last 60 seconds.
        samples = [[now - 60, st["avail_bytes"] + 60 * 30_000_000],
                   [now, st["avail_bytes"]]]
        with mock.patch.object(erd_search, "disk_stats", return_value=st):
            line = erd_search._disk_status_line(self.path, samples)
        self.assertIn("filling 30.0 MB/s: 90% in ~", line)

    def test_freeing_disk_is_labelled_freeing(self):
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 60, st["avail_bytes"] - 60 * 30_000_000],
                   [now, st["avail_bytes"]]]
        with mock.patch.object(erd_search, "disk_stats", return_value=st):
            line = erd_search._disk_status_line(self.path, samples)
        self.assertIn("freeing 30.0 MB/s", line)
        self.assertNotIn("filling", line)

    def test_stale_samples_show_no_rate(self):
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 4000, st["avail_bytes"] + 10 ** 9],
                   [now - 3600, st["avail_bytes"]]]
        with mock.patch.object(erd_search, "disk_stats", return_value=st):
            line = erd_search._disk_status_line(self.path, samples)
        self.assertNotIn("filling", line)


if __name__ == "__main__":
    unittest.main()
