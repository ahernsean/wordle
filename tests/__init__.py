"""Shared helpers for the test package."""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The repo targets 3.13 — wordle.py's shebang, every CI job, and the documented
# suite command all name it, and only 3.13 has the test dependencies installed
# on the development box.  Refuse to run rather than run degraded: an older
# interpreter has numpy but not playwright, so the browser suites would skip
# themselves and a run that never touched the report client would still report
# OK.  A suite that cannot test what it claims to must fail, not shrink.
MINIMUM_PYTHON = (3, 13)

if sys.version_info < MINIMUM_PYTHON:
    raise RuntimeError(
        f"This suite requires Python {'.'.join(map(str, MINIMUM_PYTHON))} or "
        f"newer; got {sys.version.split()[0]} from {sys.executable}. "
        f"Run it as: python3.13 -m unittest discover -s tests -t . "
        f"-p 'test_*.py'"
    )
