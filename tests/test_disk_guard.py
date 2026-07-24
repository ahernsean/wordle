"""Disk watermark guard and queue WAL checkpointing.

Covers the pieces that keep the queue WAL bounded and the disk from filling:
ERDQueue.checkpoint's result row (SQLite reports checkpoint contention in the
row, not as an exception), the checkpoint_pause quiesce flag, the disk-stop
latch, the supervisor's _disk_guard / _supervisor_checkpoint, and the report
view's disk line.
"""
import io
import os
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import erd_search
from erd_queue import (ERDQueue, disk_stats, DISK_SAMPLE_KEEP,
                       DISK_WARN_FRACTION, DISK_STOP_FRACTION)
from report_model import disk_fill_rate
from report_terminal import format_disk_size, render_disk_status


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

    def test_set_if_unset_preserves_existing_reason(self):
        self.assertTrue(self.q.set_disk_stop_if_unset("manual hold"))
        self.assertFalse(self.q.set_disk_stop_if_unset("later manual hold"))
        self.assertEqual(self.q.disk_stop()["reason"], "manual hold")


class TestSetDiskStopCommand(_TmpQueue):
    def _args(self, reason):
        return SimpleNamespace(queue=self.path, reason=reason)

    def test_sets_manual_latch(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            erd_search.cmd_queue_set_disk_stop(self._args("maintenance hold"))
        self.assertEqual(self.q.disk_stop()["reason"], "maintenance hold")
        self.assertEqual(stdout.getvalue(),
                         "Disk-stop latch set: maintenance hold.\n")

    def test_preserves_existing_latch_reason(self):
        self.q.set_disk_stop("supervisor: disk 91.0% full")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            erd_search.cmd_queue_set_disk_stop(self._args("maintenance hold"))
        self.assertEqual(self.q.disk_stop()["reason"],
                         "supervisor: disk 91.0% full")
        self.assertEqual(
            stdout.getvalue(),
            "Disk-stop latch is already set (supervisor: disk 91.0% full); "
            "it remains unchanged.\n")


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


class TestFmtSize(unittest.TestCase):
    def test_thousands_separator_on_large_values(self):
        self.assertEqual(format_disk_size(1234 * 2 ** 30), "1,234G")
        self.assertEqual(format_disk_size(1000 * 2 ** 20), "1,000M")

    def test_small_values_unchanged(self):
        self.assertEqual(format_disk_size(512 * 2 ** 20), "512M")
        self.assertEqual(format_disk_size(5 * 2 ** 30 // 2), "2.5G")

    def test_sub_megabyte_drops_to_kilobytes(self):
        # Below 1 MiB the value must not round to "0M"; it drops to K.
        self.assertEqual(format_disk_size(30 * 2 ** 10), "30K")
        self.assertEqual(format_disk_size(700 * 2 ** 10), "700K")


class TestDiskStatusLine(_TmpQueue):
    def _stats(self, used_fraction, avail=100 * 10 ** 9):
        used = int(avail * used_fraction / (1 - used_fraction))
        return {"total_bytes": used + avail, "used_bytes": used,
                "avail_bytes": avail, "used_fraction": used_fraction}

    def _render(self, used_fraction, samples, *, color=False):
        stats = self._stats(used_fraction)
        return render_disk_status({
            "total_bytes": stats["total_bytes"],
            "used_bytes": stats["used_bytes"],
            "available_bytes": stats["avail_bytes"],
            "used_fraction": stats["used_fraction"],
            "queue_wal_bytes": 0,
            "fill_rate_bytes_per_second": disk_fill_rate(samples, time.time()),
            "warning_fraction": DISK_WARN_FRACTION,
            "stop_fraction": DISK_STOP_FRACTION,
        }, color=color)

    def test_plain_below_warn(self):
        line = self._render(0.5, [])
        self.assertIn("Disk:", line)
        self.assertIn("(50%)", line)
        self.assertNotIn("\033[31m", line)

    def test_red_at_warn_threshold(self):
        line = self._render(DISK_WARN_FRACTION, [], color=True)
        self.assertIn("\033[31m", line)

    def test_fill_rate_and_eta(self):
        now = int(time.time())
        st = self._stats(0.5)
        # 30 MiB/s of fill over the last 60 seconds; the rate shares the
        # fullness/WAL binary units, so it reads "30M/s".
        samples = [[now - 60, st["avail_bytes"] + 60 * 30 * 2 ** 20],
                   [now, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertIn("filling 30M/s: 90% in ~", line)

    def test_rate_shares_units_with_sizes(self):
        # The rate must not be decimal MB (MB/s) beside binary GiB/MiB sizes.
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 60, st["avail_bytes"] + 60 * 30 * 2 ** 20],
                   [now, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertNotIn("MB/s", line)
        self.assertIn("M/s", line)

    def test_freeing_disk_is_labelled_freeing(self):
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 60, st["avail_bytes"] - 60 * 30 * 2 ** 20],
                   [now, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertIn("freeing 30M/s", line)
        self.assertNotIn("filling", line)

    def test_reportable_slow_rate_never_shows_zero(self):
        # A slow fill above the noise floor (~30 kB/s) is reported, and the
        # adaptive K/M/G unit keeps it a nonzero figure ("29K/s") instead of
        # rounding to the "0.0" that fixed decimal-MB formatting produced.
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 100, st["avail_bytes"] + 100 * 30_000],  # ~30 kB/s
                   [now, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertIn("filling 29K/s", line)
        self.assertNotIn("0.0", line)

    def test_sub_noise_floor_rate_reads_steady(self):
        # Below the noise floor the trend is indistinguishable from sampling
        # jitter, so it reads "steady" rather than a spurious rate.
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 100, st["avail_bytes"] + 100 * 5_000],   # ~5 kB/s
                   [now, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertIn("steady", line)
        self.assertNotIn("filling", line)

    def test_regression_ignores_one_noisy_endpoint(self):
        # A steady ~10 MB/s fill with one wildly noisy final reading still
        # reports the trend fitted across the full window.
        now = int(time.time())
        st = self._stats(0.5)
        n_steps, step_seconds, step_bytes = 12, 30, 10 * 2 ** 20
        start_avail = st["avail_bytes"] + n_steps * step_bytes
        # Chronological, oldest first: avail_bytes falls steadily (filling).
        samples = [[now - (n_steps - k) * step_seconds,
                    start_avail - k * step_bytes]
                   for k in range(n_steps + 1)]
        samples[-1][1] += 50 * 2 ** 20   # noisy last reading: a freeing burst
        line = self._render(0.5, samples)
        self.assertIn("filling", line)

    def test_stale_samples_show_no_rate(self):
        now = int(time.time())
        st = self._stats(0.5)
        samples = [[now - 4000, st["avail_bytes"] + 10 ** 9],
                   [now - 3600, st["avail_bytes"]]]
        line = self._render(0.5, samples)
        self.assertNotIn("filling", line)


class TestDiskFillRate(unittest.TestCase):
    """The fill rate fits a slope across every fresh sample."""

    def test_two_points_matches_secant(self):
        now = time.time()
        samples = [[now - 60, 1_000_000], [now, 400_000]]
        # avail fell 600,000 over 60 s -> filling 10,000 B/s.
        self.assertAlmostEqual(disk_fill_rate(samples, now), 10_000)

    def test_perfectly_linear_series_recovers_exact_slope(self):
        now = time.time()
        # Chronological, oldest first: avail falls 150,000 every 30 s = 5 kB/s.
        samples = [[now - (10 - k) * 30, 2_000_000 - k * 150_000]
                   for k in range(11)]
        self.assertAlmostEqual(disk_fill_rate(samples, now),
                               5_000.0, delta=1.0)

    def test_single_fresh_sample_has_no_rate(self):
        now = time.time()
        self.assertIsNone(disk_fill_rate([[now, 1_000_000]], now))

    def test_stale_samples_are_excluded(self):
        now = time.time()
        samples = [[now - 4000, 2_000_000], [now - 3600, 1_000_000]]
        self.assertIsNone(disk_fill_rate(samples, now))


if __name__ == "__main__":
    unittest.main()
