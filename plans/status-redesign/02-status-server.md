# Phase 2 — HTTP server: `status_server.py`

Read `00-overview.md` first. Requires phase 1 (`status_model.py`) merged.

## Goal

A small stdlib-only HTTP server that exposes the snapshot as JSON and serves
the (phase 3) browser client, plus a **fixture mode** that serves a canned
snapshot so the client can be developed and demoed without a live swarm.

## Non-goals

- No authentication, no TLS (LAN/tailnet-only service).
- No WebSockets/Server-Sent Events — the client polls.
- No changes to `erd_search.py`, `erd_queue.py`, or `status_model.py`.

## Files touched

- `status_server.py` — new
- `status_fixture.json` — new (hand-written sample snapshot)
- `tests/test_status_server.py` — new
- `status_client.html` — new, but as a **placeholder only** (single line of
  text such as `Status client arrives in phase 3.` inside a `<p>` tag); phase 3
  replaces it.

## Step 1 — `status_server.py`

Stdlib imports only: `argparse`, `json`, `os`, `sys`, `time`,
`http.server`, plus `from status_model import collect_status`.

```python
#!/usr/bin/env python3.13
"""status_server.py — HTTP access to the swarm status snapshot.

Endpoints
---------
GET /api/status   The snapshot from status_model.collect_status(), as JSON.
                  With --fixture, the fixture file's contents verbatim instead.
GET /             status_client.html, served from this file's directory.

Each request opens fresh SQLite connections (inside collect_status) and issues
only status reads after ERDQueue initialization. ERDQueue derives and attaches
the queue's sibling telemetry database automatically; the status service does
not need a separate telemetry option. The threading server keeps no shared
database state, and WAL mode permits concurrent reads while workers write.
"""
```

Defaults (module constants): `DEFAULT_PORT = 8765`, `DEFAULT_BIND = '0.0.0.0'`,
and reuse the queue/cache defaults by value:
`DEFAULT_CACHE = 'wordle_cache.sqlite3'`, `DEFAULT_QUEUE = 'erd_queue.sqlite3'`.

Handler:

```python
class StatusRequestHandler(http.server.BaseHTTPRequestHandler):
    # Configuration installed as class attributes by main() before serving.
    queue_path = DEFAULT_QUEUE
    cache_path = DEFAULT_CACHE
    fixture_path = None
```

`do_GET` behavior, exactly:

| Path | Response |
|---|---|
| `/api/status` | 200, `Content-Type: application/json; charset=utf-8`, `Cache-Control: no-store`. Body: fixture file bytes if `fixture_path` is set, else `json.dumps(collect_status(queue_path, cache_path))`. If snapshot assembly raises unexpectedly: 500 with body `{"error": "<str(exception)>"}` (also JSON content type). |
| `/` or `/index.html` | 200, `Content-Type: text/html; charset=utf-8`, `Cache-Control: no-store`. Body: `status_client.html` read from `os.path.dirname(os.path.abspath(__file__))`. If the file is missing: 404. |
| anything else | 404, `text/plain`, body `not found` |

Always set `Content-Length` and call `end_headers()` before writing the body.
Override `log_message` with a no-op so a 2-second poll doesn't flood the
terminal.

`main()`:

- `argparse` with `--port` (int, default `DEFAULT_PORT`), `--bind`
  (default `DEFAULT_BIND`), `--queue` (default `DEFAULT_QUEUE`), `--cache`
  (default `DEFAULT_CACHE`), `--fixture` (metavar `PATH`, default None,
  help: serve this file verbatim at /api/status instead of live data).
- Install the four values as `StatusRequestHandler` class attributes.
- `server = http.server.ThreadingHTTPServer((bind, port), StatusRequestHandler)`
- Print one line: `Serving swarm status on http://<bind>:<port>/` then
  `server.serve_forever()` inside `try/except KeyboardInterrupt`.
- Standard `if __name__ == '__main__': main()` guard.

## Step 2 — `status_fixture.json`

A hand-written snapshot conforming exactly to the phase 1 schema
(`schema_version` 1). It exists to exercise every client rendering path, so it
must contain at least:

- header data: non-zero `total_erd_branches`, `recent_5m_branches`, and
  `worker_totals`
- one **user-queued** branch (`is_cooperative` false) at `guess_depth` 2 with a
  two-guess `spine`, 3 of 8 candidates done, `worker_count` 2, a `best_guess`
  that is an answer, `status` `open`, and non-contiguous
  `done_candidate_indices` matching the count
- one **cooperative** branch (`priority` 1000000, `is_cooperative` true) with
  an empty `spine` but `guess_depth` 1 (unrecorded-spine rendering path),
  `status` `open`, and its required `done_candidate_indices`
- four workers: two live on the first branch (different `claim_idx`, non-empty
  `descent` including at least one entry with null `guess`/`pattern`), one
  **idle** (`branch_key_hex` null), one **dead** (`age_seconds` 120,
  `is_live` false)
- `generated_at` may be any fixed integer; clients must compute ages from the
  snapshot's own fields, never from wall-clock time, precisely so fixtures
  stay valid.

Verify validity: `python -m json.tool status_fixture.json` succeeds.

## Step 3 — Tests: `tests/test_status_server.py`

Start the real server in-process for each test class:

```python
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), StatusRequestHandler)
port = server.server_address[1]
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
# tearDown: server.shutdown(); thread.join()
```

Fetch with `urllib.request.urlopen(f'http://127.0.0.1:{port}/...')`.

Required cases:

1. **Fixture mode**: with `StatusRequestHandler.fixture_path =
   'status_fixture.json'`, GET `/api/status` returns 200, JSON content type,
   and a body that parses to a dict with `schema_version == 1`, non-empty
   `branches` and `workers`.
2. **Live mode**: with fixture unset and `queue_path`/`cache_path` pointing at
   fresh temp databases (create them the same way `tests/test_status_model.py`
   does), GET `/api/status` returns 200 and a dict whose `queue.ok` is true.
3. **Client page**: GET `/` returns 200 with HTML content type and the
   placeholder body.
4. **Unknown path**: GET `/nope` returns 404 (`urllib` raises `HTTPError`;
   assert `e.code == 404`).

## Acceptance checklist

- [ ] `python status_server.py --fixture status_fixture.json` starts, and
      `curl http://127.0.0.1:8765/api/status` returns the fixture JSON;
      `curl http://127.0.0.1:8765/` returns the placeholder page.
- [ ] `python status_server.py` (live mode) serves a valid snapshot with or
      without the SQLite files present (`queue.ok` false is fine — the HTTP
      response is still 200 with a complete snapshot).
- [ ] `python -m json.tool status_fixture.json` succeeds and the fixture
      includes every element listed in Step 2.
- [ ] Full test suite passes, including `tests/test_status_server.py`.
- [ ] `status_server.py` imports nothing outside the Python stdlib and
      `status_model`.
- [ ] No file outside the "Files touched" list is modified.
