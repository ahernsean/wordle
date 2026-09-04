"""Unit tests for tests/webkit_container.py's stale-container reaper.

Exercises _reap_stale_containers against a fake subprocess.run so it needs
neither a real container runtime nor a real container.
"""

from datetime import datetime, timedelta, timezone
import subprocess
import unittest
from unittest import mock

from tests.webkit_container import _reap_stale_containers, STALE_CONTAINER_SECONDS


def _iso(age_seconds):
    started = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return started.strftime("%Y-%m-%dT%H:%M:%S%z")


class ReapStaleContainersTest(unittest.TestCase):
    def _run(self, ps_ids, inspect_stdouts):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[1] == "ps":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="\n".join(ps_ids), stderr="")
            if cmd[1] == "inspect":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=inspect_stdouts.get(cmd[2], ""), stderr="")
            if cmd[1] == "stop":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            raise AssertionError(f"unexpected command {cmd}")

        with mock.patch("tests.webkit_container.subprocess.run",
                         side_effect=fake_run):
            _reap_stale_containers("podman")
        return calls

    def test_stops_only_containers_older_than_the_threshold(self):
        calls = self._run(
            ["fresh", "stale"],
            {"fresh": _iso(STALE_CONTAINER_SECONDS - 60),
             "stale": _iso(STALE_CONTAINER_SECONDS + 60)},
        )
        stop_calls = [call for call in calls if call[1] == "stop"]
        self.assertEqual(stop_calls, [["podman", "stop", "stale"]])

    def test_no_containers_found_stops_nothing(self):
        calls = self._run([], {})
        self.assertFalse(any(call[1] == "stop" for call in calls))

    def test_unparseable_timestamp_is_skipped_not_stopped(self):
        calls = self._run(["mystery"], {"mystery": "not-a-timestamp"})
        self.assertFalse(any(call[1] == "stop" for call in calls))

    def test_missing_inspect_output_is_skipped_not_stopped(self):
        calls = self._run(["gone"], {})
        self.assertFalse(any(call[1] == "stop" for call in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
