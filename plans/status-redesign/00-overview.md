# Status display redesign — overview and index

This directory holds the implementation plans for replacing the terminal swarm
status display with a client/server architecture: a status **model** (structured
data), a small HTTP **server**, and a **browser client**. Each numbered plan is
one self-contained phase, designed to be implemented by a fresh agent session
with no other context.

## How to use these plans (instructions for implementer agents)

1. **Read `CLAUDE.md` at the repo root first.** Its naming rules, vocabulary,
   and comment style are binding. The glossary below only adds terms CLAUDE.md
   does not define.
2. **Implement exactly one phase per session/PR**, in numeric order. Do not
   start a phase until the previous phase is merged.
3. **Touch only the files listed in the phase's "Files touched" section.**
4. **Follow the plan literally.** Where the plan gives a name, signature,
   JSON key, or constant, use it verbatim. If the plan contradicts what you
   find in the code (a function is missing, a signature differs), **stop and
   report the discrepancy** — do not improvise a workaround.
5. **Run the full test suite before committing**:
   `python -m unittest discover -s tests -t . -p 'test_*.py'`
   All tests must pass, including the new ones the phase requires.
6. A phase is done when every item in its "Acceptance checklist" is true.

## The phases

| Plan | Deliverable | Depends on |
|---|---|---|
| `01-status-model.md` | `status_model.py`: `collect_status()` returns the full status as a JSON-serializable dict | — |
| `02-status-server.md` | `status_server.py`: HTTP server for `/api/status` + static client, with a fixture mode; `status_fixture.json` | 01 |
| `03-browser-client.md` | `status_client.html`: polling browser UI with semantic change highlighting and width-adaptive layout | 02 |
| `04-visual-modalities.md` | Menu of independent visual upgrades: sweep bar (A), completion ring (B), worker chips (C), spine tree (D), candidate grid (E) | 03 |
| `05-landscape-view.md` | PLACEHOLDER (not implementable): pinch-zoom semantic-zoom map of the explored search landscape | 03, 04 item D, and a design conversation |

## Architecture and rationale

Today, `erd_search.py _print_status` interleaves data gathering and text
rendering, and `_redraw_status` diffs the rendered **characters** between
refreshes to color changes red. Two consequences:

- Change detection cannot be semantic (e.g. "highlight a heartbeat only when
  it is more than 5 s old"), and structural shifts (a worker moving between
  branches) cascade into spurious all-red regions.
- Layout cannot adapt to terminal width, because column widths and
  abbreviations are baked into f-strings at assembly time.

The fix for both is the same separation the client/server split needs anyway:

```
erd_queue.sqlite3 ─┐
                   ├─ status_model.collect_status() ──► snapshot dict
wordle_cache.…  ───┘                                       │
                                    ┌──────────────────────┤
                                    ▼                      ▼
                        status_server.py /api/status   (terminal display
                                    │                   keeps its own path,
                                    ▼                   unchanged)
                        status_client.html  (browser, polls JSON,
                        diffs snapshots by stable identity)
```

Design principles the plans enforce:

- **The model carries raw data, never presentation.** No percentages, no
  ETAs, no truncation, no abbreviations in the snapshot. Every derived value
  (hit rate, % done, ETA, staleness) is computed by the renderer from raw
  fields. This is what makes one model serve terminal, JSON, and browser.
- **Identity is explicit.** Branches are keyed by `branch_key_hex`, workers by
  `worker_id`. Clients diff consecutive snapshots by these keys, so a row
  moving on screen is never itself a "change".
- **The terminal display is untouched.** `_print_status` / `_redraw_status`
  keep working exactly as they do now; the browser client is additive. (The
  terminal path may be simplified later, but that is not part of these plans.)
- **Current terminal interaction semantics are preserved.** Branch order is
  sticky across refreshes, an expanded branch remains inspectable while it is
  finalizing, worker depth labels are absolute guess depths, and sweep detail
  shows completion density rather than pretending completed indices form a
  contiguous prefix. The browser may implement these differently, but must
  not regress the information those behaviors convey.
- **No new dependencies in shipped code.** Python stdlib only on the server; a
  single self-contained HTML file with vanilla JavaScript on the client (no
  CDN, no build step — it must work on a LAN with no internet). Tests are the
  one exception: browser behavior is verified with Playwright + headless
  Chromium as a development-only dependency, skip-guarded so the suite still
  passes where it isn't installed (see plan 03).

Remote access (Tailscale vs. tunnel) is deliberately **not** part of these
plans. The server binds to the LAN; how it becomes reachable from elsewhere is
an infrastructure decision independent of the code.

## Glossary (terms beyond CLAUDE.md's anchored vocabulary)

CLAUDE.md defines **guess**, **candidate**, **branch**, **guess_depth**,
**budget**, **ERD**, **max_remaining_depth**. These plans additionally use:

| Term | Meaning |
|---|---|
| **snapshot** | The complete status of the swarm at one instant, as one JSON-serializable dict. Produced by `status_model.collect_status()`. Schema in plan 01. |
| **spine** | The guesses played from the root to a branch, stored in `active_branches.spine` as space-joined `"GUESS pattern"` token pairs, e.g. `"SALET -g-g- CRANE bb-y-"`. |
| **descent** | A worker's live recursion path *below* its claimed branch, from the heartbeat's `cur_path` column ("rich spine" format, parsed by `_parse_spine` / `parse_rich_spine`). |
| **pattern string** | A response pattern as 5 characters of `g` (green), `y` (yellow), `-` (gray), e.g. `-g-g-`. Integer-coded patterns are converted with `wordle_ui.fmt_pattern`. |
| **branch_id** | Stable 4-hex-character label derived by hashing `branch_key`; same key → same id across refreshes and processes. |
| **branch_key_hex** | Full hex encoding of the `branch_key` bytes; the durable identity used for diffing between snapshots. |
| **heartbeat** | A worker's row in `worker_heartbeat` (queue DB), rewritten every ~2 s while the worker runs. |
| **live worker** | A worker whose heartbeat age ≤ `WORKER_LIVENESS_SECONDS` (30). Older heartbeats belong to dead workers. |
| **stale** | Renderer term for a heartbeat old enough to warn about but not yet dead (> 5 s). |
| **cooperative branch** | A sub-branch spawned by workers (priority ≥ 1,000,000) rather than queued by the user. Not counted in `pending_branches`. |
| **sweep** | A branch's pass over its candidate list: each candidate is claimed by index (`claim_idx`) and marked done when fully evaluated. |
| **idle worker** | A live worker whose heartbeat has `current_branch_key = NULL` (between claims). |
| **detached worker** | A worker whose `current_branch_key` refers to a branch no longer in the open-branches list (the branch is finalizing). |
| **answer flag** | A boolean marking a word that is a member of the NYT answer list (`NYT_wordlist.txt`); terminal shows it as a `*` suffix. |
