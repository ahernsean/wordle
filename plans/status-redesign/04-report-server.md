# Phase 4 — HTTP report service

Read phases 00–03d and `AGENTS.md` first. Requires phase 3d merged.

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
- `tests/fixtures/reports/queue-tree.json` — new
- `tests/fixtures/reports/workers.json` — new
- `tests/fixtures/reports/workers-tree.json` — new
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

- bind `127.0.0.1`;
- port `8765`;
- database paths imported from the shared runtime-path owner;
- live collection when fixture directory is absent.

Do not copy default path strings into this module.

Print:

    Serving ERD reports on http://<bind>:<port>/

Run `ThreadingHTTPServer` until Ctrl-C, then close it cleanly.

### Deployment boundary

The safe default is loopback. For private tailnet access on the Rocky host,
keep the report server on `127.0.0.1:8765` and front it with Tailscale Serve.
The current Tailscale CLI form for the short MagicDNS URL
`http://rocky/` is:

    tailscale serve --http=80 localhost:8765

Tailscale configuration is an operator action, not something this repository
installs or mutates. Confirm external CLI syntax with `tailscale serve --help`
and the Tailscale Serve documentation at deployment. A direct LAN deployment
must opt in with
`--bind 0.0.0.0` and use the explicit server port; it has no authentication or
TLS. Do not bind the Python process directly to privileged port 80.

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
| `branch_status` | comma-separated `active`, `pending`, `done`, `unqueued`, or `all` |
| `branch_phase` | comma-separated `queued`, `evaluating`, `finalizing`, `complete`, or `all` |
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

The root overview defaults to `branch_status=active`. An explicit `all`
disables that axis's filter. Other report kinds default to all statuses.

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
- Parse each branch filter from one comma-separated scalar value; reject
  duplicates within the value.
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
opening databases. Fully parse, validate, and infer the request before
choosing a fixture. Invalid combinations fail exactly as live requests do.

Fixture selection is ordered:

1. On `/api/view`, no selector/root uses `overview.json`, inferred word uses
   `word.json`, and inferred branch uses `branch.json`; valid `tree=1` takes
   precedence and uses `tree.json`.
2. Explicit non-tree kinds use their matching filename.
3. `queue?tree=1` uses `queue-tree.json`, and `workers?tree=1` uses
   `workers-tree.json`.
4. `cache?tree=1` and `hotspots?tree=1` are invalid and return 400 before
   fixture lookup.

Thus positional selectors `CACHE` and `QUEUE` are inferred as words before
fixture selection and use `word.json`, never explicit-kind fixtures.

Read and `json.loads` each fixture at server startup, require its
`schema_version == SCHEMA_VERSION`, and serve the re-serialized object. A bad
fixture prevents startup rather than failing later during a request.

Fixtures collectively exercise:

- user and cooperative active branches;
- valid branch-status and branch-phase combinations;
- CACHE and QUEUE as explored words;
- exact, loss, missing, and not-applicable cache states;
- bulk-eliminated and evaluated candidates;
- live, idle, finalizing, stale, and dead workers;
- exact/cut/loss finalization history;
- cut-reuse misses;
- filtered tree context;
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
3. Tree and comma-separated branch filters reach normalized request state.
4. Every explicit report endpoint returns its kind.
5. CACHE and QUEUE selector requests return word fixtures, never cache/queue
   report fixtures.
6. Inferred tree, queue tree, and workers tree use distinct fixtures;
   cache/hotspot tree requests return 400 before fixture lookup.
7. Comma-separated status and phase values are accepted; repeated scalar and unknown parameters are
   rejected.
8. Invalid booleans, integers, selector, limit, sample, and overlong target
   return 400.
9. Unknown branch reference returns 404.
10. Fixture startup validates all files and fixture requests open no SQLite
   path.
11. `/` returns the placeholder; arbitrary paths return 404.
12. POST/PUT/DELETE return 405.
13. Headers and content lengths are correct.
14. Two servers with different configurations do not share paths or fixtures.
15. A partial source failure remains HTTP 200 with the report's source error.
16. Unexpected collector failure returns sanitized 500 JSON.

## Acceptance checklist

- [ ] All terminal report requests have an HTTP representation.
- [ ] CLI and HTTP both construct the same `ReportRequest` semantics.
- [ ] The server contains no report SQL or formatting.
- [ ] Fixture mode is database-free and covers every report shape.
- [ ] The service is strictly read-only.
- [ ] Runtime defaults have one owner.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
