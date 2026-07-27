#!/bin/bash
# Install the test dependencies for Claude Code on the web.
#
# Those containers clone the repo into a bare image: requirements-dev.txt
# declares numpy and playwright but nothing installs them, so without this
# the suite cannot even import wordle_engine.  A local checkout is left
# alone -- its environment is whatever the developer set up.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# The hook event arrives as JSON on stdin.  A compact or a clear keeps the
# same container, whose dependencies are already installed; only a fresh
# start or a resume onto a rebuilt container needs the install.
payload=""
[ -t 0 ] || payload=$(cat)
event_source=$(printf '%s' "$payload" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("source",""))' \
    2>/dev/null) || event_source=""
case "$event_source" in
  compact|clear) exit 0 ;;
esac

# Run in the background so the session starts immediately.  The session can
# therefore reach for numpy or playwright before the install finishes; a
# failure that looks like a missing dependency is worth re-checking once.
echo '{"async": true, "asyncTimeout": 300000}'

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# 3.13 is what the repo targets: wordle.py's shebang, the CI matrix, and the
# test-suite command in AGENTS.md all name it.  The image's default python3
# is older, so ask for 3.13 by name and fall back only if it is absent.
PYTHON=python3.13
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

# coverage is not a runtime dependency, but CI gates on `coverage report
# --fail-under=98`, so a session needs it to check its own work.
"$PYTHON" -m pip install --quiet --disable-pip-version-check \
    -r requirements-dev.txt coverage

# Chromium is already in the image (PLAYWRIGHT_BROWSERS_PATH); the browser
# tests find it by path.  Requiring it turns a broken browser into a failure
# instead of a silent skip, matching CI.
echo 'export REQUIRE_PLAYWRIGHT_BROWSER=1' >> "${CLAUDE_ENV_FILE:-/dev/null}"

echo "Test dependencies installed for $PYTHON ($("$PYTHON" -V))."
echo "Run the suite with: $PYTHON -m unittest discover -s tests -t . -p 'test_*.py'"
