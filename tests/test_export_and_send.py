"""Contract tests for the containerized Taildrop export handoff."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


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
            case "$1" in
                volume) exit 0 ;;
                run) echo relay-container ;;
                exec)
                    if [[ "$*" == *"tailscale status --json"* ]]; then
                        echo '{"BackendState":"Running"}'
                    fi
                    ;;
                cp|rm) exit 0 ;;
                *) exit 1 ;;
            esac
        """)

    def _write_fake(self, name, source):
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(source))
        path.chmod(0o755)

    def _run(self, *args, **environment):
        env = os.environ | environment
        env["PATH"] = f"{self.bin_dir}:{env['PATH']}"
        env["PODMAN_LOG"] = str(self.log_path)
        return subprocess.run(
            ["bash", "./export_and_send.sh", *args], cwd=self.workdir,
            env=env, capture_output=True, text=True, check=False)

    def test_starts_user_owned_relay_then_copies_and_removes_it(self):
        result = self._run(TAILSCALE_AUTHKEY="tskey-auth-bootstrap")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.workdir / "wordle_export_watermark").exists(), True)
        self.assertFalse((self.workdir / "wordle_erd_export.sqlite3").exists())
        commands = self.log_path.read_text()
        self.assertIn("volume create wordle-taildrop-state", commands)
        self.assertIn("--env TS_AUTH_ONCE=true", commands)
        self.assertIn("--env TS_USERSPACE=true", commands)
        self.assertIn("--env TS_AUTHKEY=tskey-auth-bootstrap", commands)
        self.assertIn("docker.io/tailscale/tailscale:stable", commands)
        self.assertIn("cp wordle_erd_export.sqlite3", commands)
        self.assertIn("tailscale file cp /exports/wordle_erd_export.sqlite3 ios-app:", commands)
        self.assertIn("rm --force", commands)

    def test_saved_state_does_not_require_an_auth_key(self):
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.log_path.read_text()
        self.assertNotIn("TS_AUTHKEY=", commands)
