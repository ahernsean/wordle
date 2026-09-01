"""Contract tests for the containerized Taildrop export handoff."""
import os
import fcntl
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class TestExportAndSend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name)
        shutil.copy2(ROOT / "export_and_send.sh", self.workdir)
        self.bin_dir = self.workdir / "bin"
        self.bin_dir.mkdir()
        self.log_path = self.workdir / "podman.log"
        self._write_fake("python3.13", """\
            #!/usr/bin/env bash
            touch wordle_erd_export.sqlite3
        """)
        self._write_fake("sleep", """\
            #!/usr/bin/env bash
            :
        """)
        self._write_fake("podman", """\
            #!/usr/bin/env bash
            printf '%s\\n' "$*" >> "$PODMAN_LOG"
            case "$1:$2" in
                run:*) echo relay-container ;;
                container:inspect)
                    if [[ "${PODMAN_RELAY_STOPS:-}" == 1 ]]; then
                        echo false
                    else
                        echo true
                    fi
                    ;;
                exec:*)
                    if [[ "$*" == *"tailscale status --json"* ]]; then
                        if [[ "${PODMAN_RELAY_STOPS:-}" == 1 ]]; then
                            echo 'Error: no container with name or ID "wordle-taildrop" found' >&2
                            exit 125
                        elif [[ "${PODMAN_RELAY_NEVER_READY:-}" == 1 ]]; then
                            echo '{"BackendState":"NeedsLogin"}'
                        else
                            echo '{"BackendState":"Running"}'
                        fi
                    elif [[ "$*" == *"tailscale file cp"* ]] && \\
                            [[ "${PODMAN_TRANSFER_FAIL:-}" == 1 ]]; then
                        echo '502 Bad Gateway:' >&2
                        exit 42
                    fi
                    ;;
                cp:*)
                    if [[ "${PODMAN_COPY_FAIL:-}" == 1 ]]; then
                        echo 'copy failed' >&2
                        exit 43
                    fi
                    ;;
                logs:*) echo 'authentication failed' >&2 ;;
                rm:*) exit 0 ;;
                *) exit 1 ;;
            esac
        """)

    def _write_fake(self, name, source):
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(source))
        path.chmod(0o755)

    def _run(self, *args, **environment):
        env = {key: value for key, value in os.environ.items()
               if key != "TAILSCALE_AUTHKEY"
               and not key.startswith("WORDLE_TAILDROP_")}
        env |= environment
        env["PATH"] = f"{self.bin_dir}:{env['PATH']}"
        env["PODMAN_LOG"] = str(self.log_path)
        env["TMPDIR"] = str(self.workdir)
        return subprocess.run(
            ["bash", "./export_and_send.sh", *args], cwd=self.workdir,
            env=env, capture_output=True, text=True, check=False)

    def test_starts_user_owned_relay_then_copies_and_removes_it(self):
        result = self._run(TAILSCALE_AUTHKEY="tskey-auth-bootstrap")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.workdir / "wordle_export_watermark").exists(), True)
        self.assertFalse((self.workdir / "wordle_erd_export.sqlite3").exists())
        commands = self.log_path.read_text()
        self.assertIn("--env TS_AUTH_ONCE=true", commands)
        self.assertIn("--env TS_USERSPACE=true", commands)
        self.assertIn("--env TS_AUTHKEY", commands)
        self.assertNotIn("TS_AUTHKEY=tskey-auth-bootstrap", commands)
        self.assertIn("docker.io/tailscale/tailscale:stable", commands)
        self.assertIn("cp wordle_erd_export.sqlite3", commands)
        self.assertIn("tailscale file cp /exports/wordle_erd_export.sqlite3 ios-app:", commands)
        self.assertIn("rm --force", commands)

    def test_saved_state_does_not_require_an_auth_key(self):
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.log_path.read_text()
        self.assertNotIn("TS_AUTHKEY=", commands)

    def test_ignores_an_ambient_bootstrap_key(self):
        with mock.patch.dict(os.environ, {"TAILSCALE_AUTHKEY": "ambient-key"}):
            result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.log_path.read_text()
        self.assertNotIn("TS_AUTHKEY", commands)

    def test_stopped_relay_reports_its_bootstrap_error_without_exporting(self):
        result = self._run(PODMAN_RELAY_STOPS="1")

        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.workdir / "wordle_erd_export.sqlite3").exists())
        self.assertFalse((self.workdir / "wordle_export_watermark").exists())
        self.assertIn("relay stopped before it connected", result.stderr)
        self.assertIn("authentication failed", result.stderr)
        self.assertIn("rm --force wordle-taildrop", self.log_path.read_text())

    def test_relay_timeout_preserves_the_export_and_prints_setup_hint(self):
        result = self._run(PODMAN_RELAY_NEVER_READY="1")

        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.workdir / "wordle_erd_export.sqlite3").exists())
        self.assertIn("did not connect within 30 seconds", result.stderr)
        self.assertIn("For first-time setup", result.stderr)

    def test_keeps_export_when_copying_it_into_the_relay_fails(self):
        result = self._run(PODMAN_COPY_FAIL="1")

        self.assertEqual(result.returncode, 43)
        self.assertTrue((self.workdir / "wordle_erd_export.sqlite3").exists())
        self.assertFalse((self.workdir / "wordle_export_watermark").exists())
        self.assertIn("could not prepare the export", result.stderr)
        self.assertIn("Details: copy failed", result.stderr)

    def test_keeps_export_and_explains_how_to_retry_after_transfer_failure(self):
        result = self._run(PODMAN_TRANSFER_FAIL="1")

        self.assertEqual(result.returncode, 42)
        self.assertTrue((self.workdir / "wordle_erd_export.sqlite3").exists())
        self.assertFalse((self.workdir / "wordle_export_watermark").exists())
        self.assertIn("Taildrop could not reach ios-app.", result.stderr)
        self.assertIn("retrying will resend it.", result.stderr)
        self.assertIn("wait for it to report synchronized", result.stderr)
        self.assertIn("Details: 502 Bad Gateway:", result.stderr)

    def test_rejects_a_second_export_while_the_first_holds_the_lock(self):
        lock_path = self.workdir / "export.lock"
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self._run(WORDLE_TAILDROP_LOCK_FILE=str(lock_path))

        self.assertEqual(result.returncode, 1)
        self.assertIn("Another Taildrop export is already running.", result.stderr)
        self.assertFalse(self.log_path.exists())
