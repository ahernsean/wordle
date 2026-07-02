# Phase 3 — Browser client: `status_client.html`

Read `00-overview.md` first. Requires phase 2 merged. Develop against the
fixture server: `python status_server.py --fixture status_fixture.json`, then
open `http://<host>:8765/` in a browser (phone or desktop).

Automated browser tests use **Playwright with headless Chromium** — a
development-only dependency; the shipped server and client stay
dependency-free. On a normal Linux machine install it with
`pip install playwright && playwright install chromium`. In a Claude Code
remote/web session Chromium is already provided via
`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers` — do NOT run
`playwright install` there.

## Goal

Replace the placeholder `status_client.html` with a single self-contained HTML
file (inline CSS and vanilla JavaScript, no external resources of any kind)
that polls `/api/status` and renders the snapshot with:

1. **Semantic change highlighting** — changes detected by comparing fields of
   consecutive snapshots keyed by stable identity, with per-field rules —
   replacing the terminal's character-diff-in-red.
2. **Width-adaptive layout** — content adapts to the viewport (phone ≈ 390 px
   through desktop) via CSS; headers and data cells share one rendering path.

## Non-goals

- No canvas, no SVG charts, no branch tree (phase 4).
- No server changes.
- No framework, no build step, no CDN fetch — the file must render on a LAN
  with no internet access.

## Files touched

- `status_client.html` — replaced
- `test_status_client.py` — new (Playwright browser tests, skip-guarded)

## Data contract

The snapshot schema is normative in `plans/status-redesign/01-status-model.md`.
The model sends raw data only; this client computes every derived value:

| Derived value | Formula (fields from the snapshot) |
|---|---|
| worker heartbeat age | `snapshot.generated_at - worker.updated_at` — NEVER use the browser clock (`Date.now()`) against snapshot timestamps; clocks may disagree and fixtures are frozen |
| branch % done | `100 * done_candidates / n_candidates` (0 when `n_candidates` is 0) |
| branch ETA seconds | with `elapsed = generated_at - created_at`: `(n_candidates - done_candidates) / (done_candidates / elapsed)`, only when `worker_count > 0`, `0 < done_candidates < n_candidates`, and `elapsed > 0`; otherwise show `-` |
| cache hit rate | `100 * hits / (hits + misses)` from `worker_totals`, blank when the denominator is 0 |
| ERD-cutoff / depth-pruned rates | `n_cutoff / (n_ok + n_cutoff + n_pruned)` and `n_pruned / (...)` from `worker_totals`, blank when the denominator is 0 |
| hang suspicion (`~?`) | `age <= 10 && node_rate === 0 && cur_nodes > 0` (mirrors the terminal) |
| worker state label | `branch_key_hex === null` → "idle"; `branch_key_hex` not among `branches[].branch_key_hex` → "finalizing"; else on-branch |
| node rate display | `Math.round(node_rate / 1000) + 'k/s'` when non-zero |
| duration display | `MMmSSs` under an hour, `HhMMm` above (mirrors `_fmt_duration`) |

## Page structure

Dark theme, system monospace font (`ui-monospace, SFMono-Regular, Menlo,
Consolas, monospace`). Colors: background `#121213`, text `#d7dadc`,
dim text `#818384`, green `#538d4e`, yellow `#b59f3b`, tile-gray `#3a3a3c`,
alert red `#cc4444`, amber `#d0a215`. (Greens/yellows/grays are the Wordle
tile palette; keep them exact.)

Top to bottom:

1. **Header strip** — one flex row that wraps: title `ERD_ALL Precache`,
   the snapshot time (`generated_at` rendered as local HH:MM:SS), cache totals
   (`total_erd_branches` with thousands separators, `+recent_5m_branches/5m`),
   hit rate, cutoff/pruned rates, and the queue counts
   (`done / user / coop / pending` from `counts`, where `user` is
   `counts.in_progress`). If `queue.ok` or `cache.ok` is false, show its
   `error` string here in alert red.
2. **Connection state** — a "updated Ns ago" chip in the header, driven by a
   1-second local ticker counting from the last *successful* fetch (this is
   UI liveness, not data age, so the local clock is correct here). After two
   consecutive failed or missed polls, the chip turns alert red and reads
   `disconnected — last data Ns ago`. Data keeps rendering; never blank the
   page on failure.
3. **Branch cards** — one card per element of `branches` that has
   `worker_count > 0` OR is expanded (see Interactions); a card contains:
   - header row: `#branch_id`, `n_words`w, `d{guess_depth}`, % done,
     best guess (uppercase; append `*` when `best_guess_is_answer`),
     `best_erd` to 3 decimals, worker count, ETA, and a `coop` badge when
     `is_cooperative`
   - **spine row**: each spine entry rendered as the uppercase guess followed
     by its pattern as five **tiles** — inline-block squares (~1em) colored
     green/yellow/tile-gray for `g`/`y`/`-`, with the pattern string also in
     a `title` attribute. Entries joined by `▸`. When
     `guess_depth > spine.length`, append `▸ ?×N` where
     `N = guess_depth - spine.length` (unrecorded spine).
   - **worker rows table**: one row per worker whose `branch_key_hex` matches:
     `W{worker_number}`, `claim_idx`, current candidate (uppercase + `*` when
     `cur_candidate_is_answer`), `d{cur_max_depth}`, node rate, descent sizes
     (the `size` values of `descent` joined by `→`), age in seconds with
     staleness/hang badges per the rules below.
4. **Idle / finalizing / dead workers** — a single compact section listing
   workers not shown in any card, with their state label and age.
5. **Empty state** — when no branch has workers: the line
   `no branches being worked`, still inside the normal layout.

## Client architecture (normative — this is what makes the page testable)

Split the script into two layers:

1. **`applySnapshot(snapshot, previousSnapshot)`** — does ALL rendering and
   diffing: full re-render of the content container's DOM from the two
   snapshots (template strings + `innerHTML` is fine), computing every change
   class synchronously before insertion. It must not fetch, read clocks for
   data ages, or depend on any state other than its arguments and the
   expansion set. Assign it to `window.applySnapshot` so tests can call it
   directly with fixture data via Playwright's `page.evaluate`.
2. **The poll loop** — fetches `/api/status`, tracks connection state,
   remembers the previous snapshot, and calls `applySnapshot`. Nothing else.

The poll interval is read from the URL so tests can speed it up or park it:

```javascript
const POLL_INTERVAL_MILLISECONDS =
    Number(new URLSearchParams(location.search).get('poll')) || 2000;
```

All cell values are produced by a single helper `cell(value, changeClass)` so
headers and data flow through the same path and change classes cannot drift
from values. Give the elements tests must find stable ids or data attributes:
each branch card gets `data-branch="<branch_key_hex>"`, each worker row
`data-worker="<worker_id>"`, and the connection chip `id="connection"`.

## Change highlighting rules (normative)

Diff each snapshot against the previous one **by identity, then by field**:
branches matched on `branch_key_hex`, workers on `worker_id`. Position on
screen plays no role. Rules:

| Rule | Trigger | Effect (CSS class on the cell/row) |
|---|---|---|
| value change | field value differs from previous snapshot for the same identity | `flash-changed` — alert-red text, fading to normal over 1.5 s |
| improvement | `best_erd` (branch or worker) decreased, or a branch's `done_candidates` increased | `flash-improved` — green, fading over 1.5 s, instead of `flash-changed` |
| appearance | identity present now, absent in previous snapshot | whole card/row gets `flash-added` (brief green left-border pulse) |
| stale heartbeat | age > 5 s | age cell gets persistent class `stale-warn` (amber) |
| dead heartbeat | age > `worker_liveness_seconds` | `stale-dead` (alert red) and append ` !!` |
| hang suspicion | formula above | append badge `~?` in amber |

Implementation: compute classes while rendering (comparing against
`previousSnapshot`), then after inserting the DOM, a
`setTimeout(..., 1500)` removes all `flash-*` classes; CSS
`transition: color 1.5s, border-color 1.5s` produces the fade. Fields that
merely tick with time (`age_seconds`, ETA) are exempt from the value-change
rule — only the two staleness rules apply to them.

Disappearance needs no effect: identities absent from the new snapshot simply
aren't rendered (a finalized branch's workers reappear under
idle/finalizing).

## Width adaptation (normative)

CSS only — no JavaScript measurement. Cards are `display: grid` items in a
container with `grid-template-columns: repeat(auto-fill, minmax(360px, 1fr))`,
so a phone gets one column and a desktop several. Within tables, columns are
tiered with classes on both `th` and `td` (same rendering path):

| Tier | Columns | Visibility |
|---|---|---|
| (always) | branch id, % done, best guess, best ERD, worker count; worker number, current candidate, age | all widths |
| `tier-2` | n_words, guess_depth, ETA, claim_idx, node rate | hidden below 480 px (`@media (max-width: 479px)`) |
| `tier-3` | descent sizes, claims_done, coop badge text (badge collapses to a dot) | hidden below 380 px |

Spine rows may wrap (they are flex rows of tiles); nothing on the page may
force horizontal scrolling of the whole body. Tap targets (cards, worker rows)
must be at least 32 px tall on narrow screens.

## Interactions

- **Tap/click a branch card header** toggles the card expanded: the spine
  shown one guess per line (`d1`, `d2`, …, matching the terminal's branch
  detail), a `d{guess_depth+1} sweep done/total best …` line, and per-worker
  descent detail (each `descent` entry as `d{guess_depth} GUESS tiles size w`,
  entries with null `guess` as `d? size w`). Expanded state is kept in a
  JavaScript `Set` of `branch_key_hex` so it survives re-renders; an expanded
  branch stays visible even after its workers leave (mirrors the terminal's
  hotkey-pinning behavior).
- **Tap a worker row** toggles the same descent detail inline for just that
  worker.
- Poll every `POLL_INTERVAL_MILLISECONDS` (2000 default; see Client
  architecture for the `?poll=` override). When `document.hidden`, skip
  polls; poll immediately on `visibilitychange` back to visible.

## Automated browser tests: `test_status_client.py`

Standard `unittest`, guarded so the suite stays green where Playwright is not
installed:

```python
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, 'playwright not installed')
class StatusClientTest(unittest.TestCase):
    ...
```

Harness, once per test class (`setUpClass` / `tearDownClass`):

- Start the real server exactly as `test_status_server.py` does
  (`ThreadingHTTPServer(('127.0.0.1', 0), StatusRequestHandler)` in a daemon
  thread) with `StatusRequestHandler.fixture_path = 'status_fixture.json'`.
- `playwright = sync_playwright().start()`;
  `browser = playwright.chromium.launch()` (headless is the default);
  a fresh `page = browser.new_page()` per test.
- Load with the poll loop parked so tests control rendering:
  `page.goto(url + '?poll=3600000')`, then wait for the first render
  (`page.wait_for_selector('[data-branch]')`).
- In Python, load `status_fixture.json` once; tests that need "snapshot B"
  mutate a `copy.deepcopy` of it and inject both with
  `page.evaluate('([a, b]) => applySnapshot(b, a)', [snapshot_a, snapshot_b])`.
- Assert flash classes immediately after `evaluate` returns — `applySnapshot`
  applies them synchronously and the 1.5 s removal timer won't have fired.

Required cases:

1. **Fixture render**: all four fixture workers appear (two rows under the
   user-queued branch card, one idle, one dead); the cooperative branch card
   shows the `coop` badge and the `▸ ?×1` unrecorded-spine marker.
2. **Tile colors**: a tile for `g` has computed background `rgb(83, 141, 78)`
   (`#538d4e`), `y` → `rgb(181, 159, 59)`, `-` → `rgb(58, 58, 60)`
   (via `locator.evaluate('el => getComputedStyle(el).backgroundColor')`).
3. **Responsive tiers**: at viewport width 390 the first `.tier-2` cell
   `is_hidden()`; at 375 `.tier-3` is also hidden; at 800 both are visible.
   At every width tested:
   `page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')`
   is true (no horizontal body scroll).
4. **Value change flashes red**: mutate a worker's `cur_candidate`, inject,
   assert that worker row's candidate cell has class `flash-changed` and no
   other row gained a flash class.
5. **Improvement flashes green**: lower a branch's `best_erd` and raise its
   `done_candidates`; both cells get `flash-improved`, not `flash-changed`.
6. **Appearance**: add a worker to snapshot B; its row has `flash-added`.
7. **Time-tick exemption**: change only `generated_at` (+2) so every age
   grows; assert no `flash-*` class anywhere.
8. **Staleness**: the fixture's dead worker's age cell has `stale-dead` and
   text ending `!!`; mutate a live worker's `updated_at` to make its age 7 s
   → `stale-warn`.
9. **Expansion persistence**: click a branch card header, assert the expanded
   detail (sweep line) is visible, inject a new snapshot pair, assert it is
   still expanded.
10. **Disconnect indicator**: open a second page with `?poll=100`, let it
    render, then `page.route('**/api/status', lambda route: route.abort())`;
    within a few polls `#connection` gains the disconnected class and the
    branch cards are still present. Unroute; it recovers.
11. **Screenshot artifacts** (review aid, not an assertion): write
    `page.screenshot(path=...)` at widths 390 and 1200 into a temp/scratch
    directory and print the paths, so a reviewer can eyeball the layout.

## Manual checklist (the part automation can't cover)

- [ ] On the actual iPhone browser against the fixture server: tiles render,
      tap targets are comfortable, no horizontal scroll, pinch behavior sane.
- [ ] Live end-to-end: `python status_server.py` against the real databases
      with the swarm running; branch/worker rows agree with
      `python erd_search.py status`.

## Acceptance checklist

- [ ] `status_client.html` is fully self-contained (grep it for `http://`,
      `https://`, `//` URLs — none may appear in `src`/`href` attributes).
- [ ] `python -m unittest test_status_client` passes with Playwright
      installed, and is cleanly SKIPPED (not failed) without it.
- [ ] Full test suite passes.
- [ ] Both manual checklist items pass.
- [ ] No file outside the "Files touched" list is modified.
