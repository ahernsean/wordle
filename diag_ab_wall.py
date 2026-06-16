"""diag_ab_wall.py — wall measurement: alpha-beta (new) vs pre-change (old).

Cold runs, min-of-N, on the same salet branches the ORDER_MIN_N sweep used
(n~48/81/146), FULL vocab, budget=5.  Reports min wall + heartbeat evals per
engine and the new/old ratios.  ERD must be identical across engines (printed
so any divergence is obvious).

Both engines share one ScoreCache (cleared cold before every solve) and one
ResponseCache (decomp memo is engine-agnostic).  response_decomposition is
pre-seeded from prod so group splits are cheap and runs are comparable.
"""
import importlib.util
import os
import sqlite3
import tempfile
import time

import wordle_engine as new_eng
from cache_sqlite import ScoreCache


def _load_old():
    spec = importlib.util.spec_from_file_location(
        'wordle_engine_old', '/tmp/wordle_engine_old.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old_eng = _load_old()

BRANCH_SIZES = [48, 80, 120]
BUDGET = 5
REPEATS = 2
DEADLINE_S = 200.0

answers = new_eng.load_word_list('NYT_wordlist.txt')
vocab = new_eng.load_word_list('wordle.txt')

q = sqlite3.connect('erd_queue.sqlite3', timeout=10); q.row_factory = sqlite3.Row
branches = []
for sz in BRANCH_SIZES:
    row = q.execute("""SELECT branch_key, n_words FROM pending_subgroups
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
RC = new_eng.ResponseCache(answers, SC)


def solve(eng, words):
    SC._mem_cache.clear()
    SC._conn.execute("DELETE FROM branch_best_by_policy")
    cnt = [0]
    t0 = time.time()
    erd = eng.min_expected_guesses(
        words, RC, SC, guesses=vocab, budget=BUDGET, deadline=t0 + DEADLINE_S,
        heartbeat=lambda: cnt.__setitem__(0, cnt[0] + 1))
    return time.time() - t0, erd, cnt[0]


print(f'budget={BUDGET}  min of {REPEATS}  vocab={len(vocab)}', flush=True)
for key, n_words in branches:
    words = sorted(key[i:i + 5].decode() for i in range(0, len(key), 5))
    for g in vocab:                       # warm decomp memo for top node
        RC.group_counts(g, words)
    print(f'\n=== branch n={n_words} ===', flush=True)
    stats = {}
    for tag, eng in (('old', old_eng), ('new', new_eng)):
        runs = []
        erd = evals = None
        for _ in range(REPEATS):
            w, e, ev = solve(eng, words)
            runs.append(w); erd = e; evals = ev
        m = min(runs)
        stats[tag] = (m, erd, evals)
        erd_s = 'abort' if erd is None else f'{erd:.5f}'
        print(f'  {tag}: wall {m:7.2f}s  erd {erd_s}  evals {evals}', flush=True)
    (wo, eo, vo) = stats['old']; (wn, en, vn) = stats['new']
    erd_ok = 'ERD MATCH' if (eo == en or (eo and en and abs(eo - en) < 1e-9)) else '*** ERD DIFF ***'
    print(f'  -> wall {wo/max(1e-9,wn):.2f}x  evals {vo/max(1,vn):.2f}x  {erd_ok}', flush=True)

SC.close()
for ext in ('', '-wal', '-shm'):
    try: os.unlink(cache_tmp.name + ext)
    except OSError: pass
print('\ndone', flush=True)
