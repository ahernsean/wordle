"""Containerized WebKit for the browser contract tests.

Rocky's glibc is too old for Playwright's bundled WebKit build, so this
launches WebKit inside the official Microsoft Playwright container instead of
locally. The container runs in `run-server` mode, which exposes a `ws://`
endpoint that `playwright.webkit.connect()` drives exactly like a local
browser; only the browser process itself lives in the container, not the test
process. The `run-server` wire protocol requires the client and server to be
running the same Playwright version, so the image tag is derived from the
installed `playwright` package's version rather than pinned.

The container runs with `--network host`: the fixture HTTP server the browser
tests navigate to is bound to 127.0.0.1 on the host, which is only reachable
from a page running inside the container if the container shares the host's
network namespace rather than getting a port forwarded into an isolated one.
"""

from importlib.metadata import PackageNotFoundError, version
import shutil
import socket
import subprocess
import time


IMAGE_REPOSITORY = "mcr.microsoft.com/playwright"
READY_TIMEOUT_SECONDS = 60


def installed_playwright_version():
    try:
        return version("playwright")
    except PackageNotFoundError:
        return None


def _container_runtime():
    for runtime in ("podman", "docker"):
        if shutil.which(runtime):
            return runtime
    return None


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class WebKitContainerUnavailable(RuntimeError):
    """The WebKit run-server container could not be started."""


class WebKitContainerServer:
    """A running `playwright run-server` container exposing WebKit.

    `ws_endpoint` connects via `playwright.webkit.connect()`. Call `stop()`
    (or use as a context manager) to tear the container down.
    """

    def __init__(self, runtime, container_id, ws_endpoint):
        self.runtime = runtime
        self.container_id = container_id
        self.ws_endpoint = ws_endpoint

    def stop(self):
        subprocess.run(
            [self.runtime, "stop", self.container_id],
            capture_output=True, check=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.stop()


def start_webkit_server(timeout=READY_TIMEOUT_SECONDS):
    """Start a containerized `playwright run-server` exposing WebKit.

    Raises WebKitContainerUnavailable if no container runtime is on PATH, the
    playwright package's version can't be read, or the server does not report
    ready within `timeout` seconds. The container is stopped before raising
    on the timeout path.
    """
    runtime = _container_runtime()
    if runtime is None:
        raise WebKitContainerUnavailable("neither podman nor docker is on PATH")

    pw_version = installed_playwright_version()
    if pw_version is None:
        raise WebKitContainerUnavailable("the playwright package is not installed")

    image = f"{IMAGE_REPOSITORY}:v{pw_version}-noble"
    port = _free_port()
    launch = subprocess.run(
        [
            runtime, "run", "--rm", "-d", "--network", "host",
            image,
            "npx", "-y", f"playwright@{pw_version}",
            "run-server", "--port", str(port), "--host", "127.0.0.1",
        ],
        capture_output=True, text=True,
    )
    if launch.returncode != 0:
        raise WebKitContainerUnavailable(
            f"{runtime} run failed for {image}: {launch.stderr.strip()}"
        )
    container_id = launch.stdout.strip()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        logs = subprocess.run(
            [runtime, "logs", container_id],
            capture_output=True, text=True, check=False,
        )
        if "Listening on ws://" in logs.stdout + logs.stderr:
            return WebKitContainerServer(
                runtime, container_id, f"ws://127.0.0.1:{port}/"
            )
        time.sleep(1)

    subprocess.run([runtime, "stop", container_id], capture_output=True, check=False)
    raise WebKitContainerUnavailable(
        f"{image} run-server did not report ready within {timeout}s"
    )
