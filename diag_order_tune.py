"""diag_order_tune.py — robustly re-measure the ORDER_MIN_N threshold.

Gates {4,6,8} across several branch sizes incl. larger ones, REPEATED cold runs
(min-of-N, since the min is the least system-noise-contended estimate of true
compute time). Wall is the objective; erd must be constant; evals secondary.

Guardrails: per-solve deadline (engine-internal), cold cache each run,
erd_queue read-only, 238-word branch excluded.
"""
import os
import sqlite3
import tempfile
import time

import wordle_engine
from cache_sqlite import ScoreCache
from wordle_engine import load_word_list, ResponseCache, min_expected_guesses

BRANCH_SIZES = [48, 80, 120]
GATES = [4, 6, 8]
REPEATS = 3
BUDGET = 5
DEADLINE_S = 150.0

from runtime_paths import DEFAULT_ANSWER_LIST_PATH, DEFAULT_CANDIDATE_LIST_PATH

answers = load_word_list(DEFAULT_ANSWER_LIST_PATH)
vocab = load_word_list(DEFAULT_CANDIDATE_LIST_PATH)

q = sqlite3.connect('erd_queue.sqlite3', timeout=10); q.row_factory = sqlite3.Row
branches = []
for sz in BRANCH_SIZES:
    row = q.execute("""SELECT branch_key, n_words FROM pending_branches
                       WHERE source_word='salet' AND n_words>=? AND n_words<>238
                       ORDER BY n_words ASC LIMIT 1""", (sz,)).fetchone()
    if row:
        branches.append((row['branch_key'], row['n_words']))
q.close()

cache_tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False); cache_tmp.close()
ScoreCache(cache_tmp.name, answers).close()
c = sqlite3.connect(cache_tmp.name)
c.execute("ATTACH 'wordle_cache.sqlite3' AS prod")
c.execute("INSERT OR IGNORE INTO response_decomposition SELECT * FROM prod.response_decomposition")
c.commit(); c.execute("DETACH prod"); c.close()

SC = ScoreCache(cache_tmp.name, answers, checkpoint_on_close=False)
RC = ResponseCache(answers, SC)


def solve(words, gate):
    wordle_engine.ORDER_MIN_N = gate
    SC._mem_cache.clear()
    SC._conn.execute("DELETE FROM branch_best_by_policy")
    t0 = time.time()
    erd = min_expected_guesses(words, RC, SC, guesses=vocab, budget=BUDGET,
                               deadline=t0 + DEADLINE_S)
    return time.time() - t0, erd


for key, n_words in branches:
    words = sorted(key[i:i+5].decode() for i in range(0, len(key), 5))
    for g in vocab:
        RC.group_counts(g, words)
    print(f'\n=== branch n={n_words} (budget={BUDGET}, min of {REPEATS}) ===', flush=True)
    print(' gate |  runs (s)                 |  min   | erd', flush=True)
    mins = {}
    for gate in GATES:
        runs = []
        erd = None
        for _ in range(REPEATS):
            w, e = solve(words, gate)
            runs.append(w); erd = e
        m = min(runs)
        mins[gate] = m
        erd_s = 'abort' if erd is None else f'{erd:.5f}'
        runs_s = ' '.join(f'{r:6.2f}' for r in runs)
        print(f' {gate:>4} | {runs_s} | {m:6.2f} | {erd_s}', flush=True)
    base = mins.get(8)
    if base:
        rel = {g: f'{base/mins[g]:.2f}x' for g in GATES}
        best = min(mins, key=mins.get)
        print(f'  vs gate8: {rel}   -> min at gate={best}', flush=True)

SC.close()
for ext in ('', '-wal', '-shm'):
    try: os.unlink(cache_tmp.name + ext)
    except OSError: pass
