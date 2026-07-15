# Phase 3d — Terminal navigation and legacy cutover

Read phases 00–03c and `AGENTS.md` first. Requires phase 3c merged.

## Goal

Finish the terminal reporting transition:

- add identity-based in-session navigation to watched text reports;
- remove legacy read-only status and queue inspection surfaces;
- update operator documentation atomically with command removal.

After this phase, all terminal inspection flows through `erd_search.py view`.

## Files touched

- `report_terminal.py`
- `erd_search.py`
- `SWARM.md`
- `AGENTS.md`
- `tests/test_report_terminal.py`
- `tests/test_report_objects.py`
- `tests/test_queue_visibility.py`
- `tests/test_status_sections.py` — delete after equivalent report-terminal
  coverage exists

No report-model, query, cache, engine, or worker behavior changes belong in
this phase. If the cutover exposes a missing model capability, stop and fix it
in the owning earlier phase rather than adding terminal-only data assembly.

## Canonical terminal surface

The final read-only surface is:

    python erd_search.py view [SPINE ...] [REPORT OPTIONS]

It combines the options specified by phases 2 and 3a–3c:

- text, JSON, JSON Lines, color, source paths, and optional watch;
- inferred word and branch exploration plus optional claims/answers;
- queue, workers, single-worker, cache, and hotspot report flags;
- tree layout where compatible;
- lifecycle, answer-count, budget, priority, sort, and limit filters;
- bounded hotspot epoch, window, and sample controls.

The parser keeps all compatibility and conflict rules specified in those
phases. This phase changes navigation and removes obsolete entry points; it
does not reinterpret report requests.

## TTY navigation

Watched text adds:

- branch hotkeys mapped to full `branch_key_hex`;
- numeric worker selection;
- backspace/Escape returns to the previous report request;
- space refreshes;
- `q` quits.

Selections are immutable report requests pushed onto a small navigation
stack. Do not encode navigation by calling legacy command handlers. A branch
transitioning to finalizing remains pinned by identity until dismissed.

Non-TTY text and JSON Lines remain noninteractive. One-shot invocations do
not change terminal mode.

The navigation footer follows the phase 2 width contract: targets wrap as
complete tokens, no hotkey/identity token is clipped, and lower-priority help
text is omitted before a target is truncated. A terminal resize recomputes
all report tables and navigation wrapping on the next refresh.

## Legacy removal

Remove:

- `status` parser and handler;
- `cache-status`;
- read-only `queue` dashboard, `ls`, `tree`, `show`, `summary`, `top`, and
  `coverage` handlers/parsers;
- terminal-only data assembly and character-diff helpers superseded by
  `report_model` and `report_terminal`;
- obsolete legacy tests after their semantics are covered by report tests.

Retain:

- `start`, `stop`, `restart`, and `run`;
- `queue add/remove/clear/priority/reset-stale`;
- queue branch-reference behavior only through the semantic selector path.

Update `AGENTS.md` and `SWARM.md` in the same commit so no documented command
is left broken. Queue documentation describes mutations only; reporting
examples use `view`.

## Tests

Required coverage:

1. TTY branch and worker navigation builds report requests rather than calling
   legacy handlers.
2. Back restores the prior selector, kind, filters, and tree state.
3. A selected branch remains pinned through finalization by full identity.
4. Non-TTY and structured output remain noninteractive and contain no cursor
   control sequences.
5. Removed commands fail argparse while lifecycle and queue mutation commands
   still work.
6. Every `SWARM.md` command example parses.
7. Navigation targets and all final report kinds fit widths 50, 55, 59, 60,
   79, 80, and 120 before the legacy renderer is removed.
8. Every semantic assertion formerly in `tests/test_status_sections.py` has an
   equivalent report-model or report-terminal test before that file is
   deleted.

## Acceptance checklist

- [ ] All terminal inspection uses `erd_search.py view`.
- [ ] Navigation is report-request state keyed by full identities.
- [ ] The final terminal surface is adaptive and complete at 50 columns.
- [ ] Queue commands are mutation-only.
- [ ] `status` and `cache-status` are removed.
- [ ] `AGENTS.md`, `SWARM.md`, parser help, and tests name the same surface.
- [ ] No report data assembly is introduced in terminal code.
- [ ] Full test suite passes.
- [ ] No file outside the phase list is modified.
