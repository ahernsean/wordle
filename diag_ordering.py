"""diag_ordering.py — does best-first candidate ordering speed up budget-5 solves?

Order-only change is already proven correct (239 unit tests).  This measures
PERFORMANCE on real SALET branches, ordering OFF vs ON, cold each time.

Per (branch, config) we record:
  - wall   : the headline speedup
  - evals  : heartbeat fires (evaluate_candidate + _solve_subset visits) = work proxy
             -> confirms the *mechanism* (ordering should cut explored subtrees)
  - erd    : must be identical OFF vs ON (correctness at real scale)

Guardrails: per-solve deadline (engine-internal, 90s), smallest branch first,
if an OFF solve aborts (deadline) we stop escalating; outer `timeout` backstops.
Temp cache (warm decomps only), erd_queue read-only, 238-word branch excluded.
"""
import os
import sqlite3
import sys
import tempfile
import time

import wordle_engine
from cache_sqlite import ScoreCache
from wordle_engine import load_word_list, ResponseCache, min_expected_guesses, ERD_ALL

SIZES = [int(x) for x in sys.argv[1:]] or [20, 30, 45, 60]
BUDGET = 5
DEADLINE_S = 90.0
ORDER_ON = 8
ORDER_OFF = 10 ** 9   # gate so high ordering never triggers

from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_CANDIDATE_LIST_PATH,
    DEFAULT_QUEUE_PATH,
)

answers = load_word_list(DEFAULT_ANSWER_LIST_PATH)
vocab = load_word_list(DEFAULT_CANDIDATE_LIST_PATH)

# Read branch keys (read-only), smallest-first, excluding the 238-word branch.
q = sqlite3.connect(DEFAULT_QUEUE_PATH, timeout=10); q.row_factory = sqlite3.Row
branches = []
for sz in SIZES:
    row = q.execute("""SELECT branch_key, n_words FROM pending_branches
                       WHERE source_word='salet' AND n_words>=? AND n_words<>238
                       ORDER BY n_words ASC LIMIT 1""", (sz,)).fetchone()
    if row and row['branch_key'] not in [b[0] for b in branches]:
        branches.append((row['branch_key'], row['n_words']))
q.close()
print(f'branches: {[n for _, n in branches]}  budget={BUDGET} deadline={DEADLINE_S}s',
      flush=True)

# Temp cache with warm decompositions copied from production.
cache_tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False); cache_tmp.close()
ScoreCache(cache_tmp.name, answers).close()
c = sqlite3.connect(cache_tmp.name)
c.execute(f"ATTACH '{DEFAULT_CACHE_PATH}' AS prod")
c.execute("INSERT OR IGNORE INTO response_decomposition SELECT * FROM prod.response_decomposition")
c.commit(); c.execute("DETACH prod"); c.close()

SC = ScoreCache(cache_tmp.name, answers, checkpoint_on_close=False)
RC = ResponseCache(answers, SC)


def clear_results():
    # Clear BOTH layers: the in-memory mirror AND the SQLite rows, via SC's own
    # (autocommit) connection — else the 2nd config hits results the 1st wrote.
    SC._mem_cache.clear()
    SC._conn.execute("DELETE FROM branch_best_by_policy")


def solve(words, gate):
    wordle_engine.ORDER_MIN_N = gate
    clear_results()
    n = [0]
    def hb():
        n[0] += 1
    t0 = time.time()
    erd = min_expected_guesses(words, RC, SC, guesses=vocab, budget=BUDGET,
                               deadline=t0 + DEADLINE_S, heartbeat=hb)
    return time.time() - t0, erd, n[0]


print('\n size |     OFF (wall / evals / erd)      |      ON (wall / evals / erd)      | speedup', flush=True)
for key, n_words in branches:
    words = sorted(key[i:i+5].decode() for i in range(0, len(key), 5))
    # warm all decomps for THIS subset so neither config pays cold scoring
    for g in vocab:
        RC.group_counts(g, words)
    w_off, e_off, ev_off = solve(words, ORDER_OFF)
    w_on, e_on, ev_on = solve(words, ORDER_ON)
    erd_off = 'abort' if e_off is None else f'{e_off:.5f}'
    erd_on = 'abort' if e_on is None else f'{e_on:.5f}'
    match = 'OK' if (e_off is None) == (e_on is None) and (
        e_off is None or abs(e_off - e_on) < 1e-9) else 'MISMATCH!!'
    sp = (w_off / w_on) if w_on > 0 else 0
    print(f' {n_words:4d} | {w_off:7.1f}s {ev_off:9,d} {erd_off:>9} '
          f'| {w_on:7.1f}s {ev_on:9,d} {erd_on:>9} | {sp:.2f}x  {match}',
          flush=True)
    if e_off is None:   # OFF couldn't finish this size -> don't escalate
        print('  OFF hit deadline; stopping escalation.', flush=True)
        break

SC.close()
for ext in ('', '-wal', '-shm'):
    try: os.unlink(cache_tmp.name + ext)
    except OSError: pass
