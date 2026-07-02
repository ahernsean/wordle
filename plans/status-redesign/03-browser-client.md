# Phase 3 — Browser client: `status_client.html`

Read `00-overview.md` first. Requires phase 2 merged. Develop against the
fixture server: `python status_server.py --fixture status_fixture.json`, then
open `http://<host>:8765/` in a browser (phone or desktop).

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

Rendering approach: full re-render on every poll. One function
`render(snapshot, previousSnapshot)` rebuilds the content container's DOM
(template strings + `innerHTML` is fine). All cell values are produced by a
single helper `cell(value, changeClass)` so headers and data flow through the
same path and change classes cannot drift from values.

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
- Poll every 2000 ms (`POLL_INTERVAL_MILLISECONDS = 2000` at the top of the
  script). When `document.hidden`, skip polls; poll immediately on
  `visibilitychange` back to visible.

## Manual test checklist (run against the fixture server)

Since the client is a single static file, testing is manual, against
`status_server.py --fixture status_fixture.json`:

- [ ] All four fixture workers render: two on the user-queued branch card, one
      idle, one dead (red `!!`, age 120 s).
- [ ] The cooperative branch shows the `coop` badge and the `▸ ?×1`
      unrecorded-spine marker.
- [ ] Pattern tiles show correct colors for a known pattern string.
- [ ] Narrow window (< 400 px): tier-2/tier-3 columns disappear, no horizontal
      body scroll; wide window: multiple card columns.
- [ ] Edit a value in `status_fixture.json` (e.g. lower a `best_erd`, change a
      `cur_candidate`) while the server runs: within one poll the cell flashes
      green (improvement) / red (change) and fades.
- [ ] Kill the server: within ~2 polls the connection chip turns red and reads
      `disconnected`; data stays on screen. Restart: chip recovers.
- [ ] Expanding a card and waiting through several polls keeps it expanded.
- [ ] Live end-to-end: run `python status_server.py` against the real
      databases with the swarm running; verify branch/worker rows match
      `python erd_search.py status`.

Also run the full unit test suite (it must stay green; this phase should not
affect it).

## Acceptance checklist

- [ ] `status_client.html` is fully self-contained (grep it for `http://`,
      `https://`, `//` URLs — none may appear in `src`/`href` attributes).
- [ ] Every item in the manual test checklist passes.
- [ ] Full unit test suite passes.
- [ ] No file outside the "Files touched" list is modified.
