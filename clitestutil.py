"""Shared harness for exercising wordle.py's interactive command handlers.

The cmd_* handlers read with input() and write with print(); this base class
builds a real GameState in a throwaway working directory (so its SQLite cache
lands in a temp dir), feeds canned input, and captures stdout.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import wordle
from wordle import GameState

# Small deterministic word sets, mirroring test_wordle.py so behavior matches
# the rest of the suite.
ANSWERS = ["crane", "slate", "trace", "stale", "tales",
           "least", "heart", "earth", "share", "rates"]
GUESSES = ANSWERS + ["brain", "stove", "cloud", "piano", "train"]


class _StopInput(AssertionError):
    """Raised when a handler reads more input lines than the test supplied."""


class CliTestCase(unittest.TestCase):
    """Base class providing a GameState fixture and a run_cmd() driver."""

    ANSWERS = ANSWERS
    GUESSES = GUESSES

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.gs = GameState(list(self.ANSWERS), list(self.GUESSES))
        self.addCleanup(self._teardown)

    def _teardown(self):
        # Reap any background threads a handler may have spawned so they do not
        # outlive the temp dir holding their cache file.
        for attr in ("precache_solver", "solver"):
            t = getattr(self.gs, attr, None)
            if t is not None and getattr(t, "is_alive", lambda: False)():
                if hasattr(t, "stop"):
                    t.stop()
                t.join(timeout=10)
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def run_cmd(self, fn, inputs=None, gs=None):
        """Invoke handler `fn(gs)` with `inputs` queued for input(); return stdout."""
        gs = gs if gs is not None else self.gs
        it = iter(list(inputs or []))

        def fake_input(*_a, **_k):
            try:
                return next(it)
            except StopIteration:
                raise _StopInput(
                    "handler requested more input than the test provided")

        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch("builtins.input", fake_input):
            fn(gs)
        return buf.getvalue()

    def soln(self, gs=None):
        return (gs if gs is not None else self.gs).solutions[0]
