# Phase 4 — HTTP report service

Read phases 00–03 and `AGENTS.md` first. Requires phase 3 merged.

## Goal

Expose every report request over a small read-only stdlib HTTP service and
serve a placeholder browser client:

    GET /api/view
    GET /api/view/queue
    GET /

The server parses transport parameters into the same `ReportRequest` used by
the terminal and calls `collect_report`. It contains no report SQL,
normalization, or presentation derivation.

## Non-goals

- No authentication or TLS. Deployment is loopback/LAN/tailnet policy.
- No mutations.
- No WebSocket or Server-Sent Events; the browser polls.
- No production HTML beyond a placeholder; phase 5 replaces it.

## Files touched

- `report_server.py` — new
- `report_client.html` — new placeholder
- `tests/fixtures/reports/overview.json` — new
- `tests/fixtures/reports/word.json` — new
- `tests/fixtures/reports/branch.json` — new
- `tests/fixtures/reports/tree.json` — new
- `tests/fixtures/reports/queue.json` — new
- `tests/fixtures/reports/workers.json` — new
- `tests/fixtures/reports/cache.json` — new
- `tests/fixtures/reports/hotspots.json` — new
- `tests/test_report_server.py` — new

## Server CLI

    python report_server.py
        [--bind ADDRESS]
        [--port PORT]
        [--queue-path PATH]
        [--cache-path PATH]
        [--fixture-directory PATH]

Defaults:

- bind `0.0.0.0`;
- port `8765`;
- database paths imported from the shared runtime-path owner;
- live collection when fixture directory is absent.

Do not copy default path strings into this module.

Print:

    Serving ERD reports on http://<bind>:<port>/

Run `ThreadingHTTPServer` until Ctrl-C, then close it cleanly.

## Request handler construction

Use a handler factory or a dedicated configuration object captured by a
handler subclass. Do not make tests race by mutating global class attributes
shared across simultaneously running servers.

Configuration contains:

    ReportSources
    client_path
    fixture_directory

Each live request creates fresh queue/cache connections through
`collect_report`. The server retains no SQLite connections across requests.

## Routes

### `GET /api/view`

Default inferred report. Query parameters:

| Parameter | Mapping |
|---|---|
| `selector` | full inferred spine/reference string; omitted means root |
| `tree` | boolean |
| `active_only` | boolean |
| `status` | repeatable lifecycle |
| `minimum_answer_count` / `maximum_answer_count` | integer |
| `budget` / `priority` | integer |
| `sort` / `limit` | collection controls |
| `claims` / `answers` | detail booleans |
| `epoch` / `since_seconds` / `sample_size` / `by` | hotspot/detail controls where valid |

### Explicit kinds

    GET /api/view/queue
    GET /api/view/workers
    GET /api/view/cache
    GET /api/view/hotspots

These accept the same compatible selector/filter parameters. A single worker
uses `worker=N` on the workers endpoint.

Do not expose positional reserved path names for word/branch reports. They
remain inferred through `selector` on `/api/view`, so the HTTP and CLI grammar
cannot drift.

### Static client

`GET /` and `GET /index.html` return `report_client.html` from the server
module's directory. Do not serve arbitrary filesystem paths.

Everything else returns 404.

## Parameter validation

Use `urllib.parse.urlsplit` and `parse_qs(..., keep_blank_values=True)`.

- Accept booleans only as `1`, `0`, `true`, or `false`.
- Reject duplicate scalar parameters.
- Permit repeated `status` only.
- Reject unknown parameters.
- Apply the same minimum/maximum bounds as terminal argparse.
- Reject URL/request targets longer than 8192 bytes.
- Pass the selector string to `parse_report_selector` unchanged after URL
  decoding.
- Convert validation failures to 400 JSON:

      {"error": {"kind": "invalid_request", "message": "..."}}

- Unknown semantic branch/reference becomes 404 JSON.
- Unexpected collection failure becomes 500 JSON without a traceback or
  local filesystem details beyond source errors already represented by the
  report contract.

## Responses

Every API success:

- status 200;
- `Content-Type: application/json; charset=utf-8`;
- `Cache-Control: no-store`;
- `X-Content-Type-Options: nosniff`;
- exact `Content-Length`;
- body `json.dumps(report, sort_keys=True).encode("utf-8")`.

Static HTML uses `text/html; charset=utf-8` and the same no-store/nosniff
headers.

Override `log_message` so two-second polling does not flood stdout. Unexpected
server-side exceptions may log one concise stderr line.

Reject mutation methods with 405 and `Allow: GET`. The service must never call
queue mutation methods.

## Fixture mode

`--fixture-directory` makes API routes return canned envelopes without
opening databases. Choose the fixture by request:

- no selector/root: `overview.json`;
- inferred word: `word.json`;
- inferred branch: `branch.json`;
- `tree=1`: `tree.json`;
- explicit kinds: matching filename.

Still parse and validate the request before selecting a fixture. Read and
`json.loads` each fixture at server startup, require its
`schema_version == SCHEMA_VERSION`, and serve the re-serialized object. A bad
fixture prevents startup rather than failing later during a request.

Fixtures collectively exercise:

- user and cooperative active branches;
- pending, finalizing, done, and unqueued lifecycle;
- CACHE and QUEUE as explored words;
- exact, loss, missing, and not-applicable cache states;
- bulk-eliminated and evaluated candidates;
- live, idle, finalizing, stale, and dead workers;
- exact/cut/loss finalization history;
- cut-reuse misses;
- active-only tree context;
- coordination workload-bucket hotspots.

## Placeholder client

`report_client.html` is a valid self-contained document that says the browser
client arrives in phase 5 and includes a link to `/api/view`. No external
resources.

## Tests

Start real ephemeral `ThreadingHTTPServer(("127.0.0.1", 0), handler)`
instances in daemon threads and use `urllib.request`.

Required cases:

1. Live root view over temporary fresh databases equals a direct
   `collect_report` call in contract shape.
2. Selector inference returns word and branch reports from the same endpoint.
3. Tree and active-only parameters reach the normalized echoed filters.
4. Every explicit report endpoint returns its kind.
5. CACHE and QUEUE selector requests return word fixtures, never cache/queue
   report fixtures.
6. Repeated status is accepted; repeated scalar and unknown parameters are
   rejected.
7. Invalid booleans, integers, selector, limit, sample, and overlong target
   return 400.
8. Unknown branch reference returns 404.
9. Fixture startup validates all files and fixture requests open no SQLite
   path.
10. `/` returns the placeholder; arbitrary paths return 404.
11. POST/PUT/DELETE return 405.
12. Headers and content lengths are correct.
13. Two servers with different configurations do not share paths or fixtures.
14. A partial source failure remains HTTP 200 with the report's source error.
15. Unexpected collector failure returns sanitized 500 JSON.

## Acceptance checklist

- [ ] All terminal report requests have an HTTP representation.
- [ ] CLI and HTTP both construct the same `ReportRequest` semantics.
- [ ] The server contains no report SQL or formatting.
- [ ] Fixture mode is database-free and covers every report shape.
- [ ] The service is strictly read-only.
- [ ] Runtime defaults have one owner.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
