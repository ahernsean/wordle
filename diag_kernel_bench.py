"""diag_kernel_bench.py -- §4 kernel bench: matrix-on vs matrix-off.

For each branch size, solves the same branch with and without a shared
PatternMatrix, checks the two ERDs are exactly equal (exiting nonzero on any
mismatch instead of just flagging it in the log -- a tolerance here would be
the same design bug Part II rule 1 forbids everywhere else), and reports
wall time, node count (heartbeat ticks), and the fraction of wall time spent
inside ResponseCache.group_words -- the per-node share §5 would vectorize next.

Self-contained: builds its own temp ScoreCache and PatternMatrix from the
real word lists rather than touching any production database, so it is safe
to run standalone. Not part of the unittest suite (same family as the other
diag_*.py scripts) -- run directly with `python3 diag_kernel_bench.py`.

DIAG_BRANCH_SIZES / DIAG_BUDGET / DIAG_DEADLINE_S / DIAG_SEED env vars
override the defaults below, for a quick sanity check on small branches
without editing the file. DIAG_VOCAB_SAMPLE further shrinks the guess
vocabulary (0 = full wordle.txt) so the matrix build itself stays fast when
no pre-warmed decomposition cache is available -- the real bench run on the
box should leave it at 0 against a warm production decomposition table.
"""
import os
import random
import sys
import tempfile
import time

import pattern_matrix
import wordle_engine as eng
from cache_sqlite import ScoreCache

BRANCH_SIZES = [int(s) for s in
                os.environ.get('DIAG_BRANCH_SIZES', '8,30,81,146,500').split(',')]
BUDGET = int(os.environ.get('DIAG_BUDGET', '5'))
DEADLINE_S = float(os.environ.get('DIAG_DEADLINE_S', '60'))
SEED = int(os.environ.get('DIAG_SEED', '0'))
VOCAB_SAMPLE = int(os.environ.get('DIAG_VOCAB_SAMPLE', '0'))

answers = eng.load_word_list('NYT_wordlist.txt')
vocab = eng.load_word_list('wordle.txt')
rng = random.Random(SEED)
if VOCAB_SAMPLE:
    answers = sorted(rng.sample(answers, min(VOCAB_SAMPLE, len(answers))))
    vocab = sorted(set(rng.sample(vocab, min(VOCAB_SAMPLE, len(vocab)))) | set(answers))

pm = pattern_matrix.PatternMatrix.build(vocab, answers)


def _fresh_cache():
    tmp = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
    tmp.close()
    return tmp.name, ScoreCache(tmp.name, answers, checkpoint_on_close=False)


def _timed_group_words(rcache):
    """Wraps rcache.group_words to accumulate elapsed seconds in acc[0]."""
    acc = [0.0]
    original = rcache.group_words

    def wrapped(guess, subset):
        t0 = time.perf_counter()
        result = original(guess, subset)
        acc[0] += time.perf_counter() - t0
        return result

    rcache.group_words = wrapped
    return acc


def solve(branch_words, matrix):
    path, sc = _fresh_cache()
    rcache = eng.ResponseCache(answers, sc)
    group_words_seconds = _timed_group_words(rcache)
    nodes = [0]
    t0 = time.time()
    erd = eng.min_expected_guesses(
        branch_words, rcache, sc, guesses=vocab, budget=BUDGET,
        deadline=t0 + DEADLINE_S, pattern_matrix=matrix,
        heartbeat=lambda: nodes.__setitem__(0, nodes[0] + 1))
    wall = time.time() - t0
    sc.close()
    for ext in ('', '-wal', '-shm'):
        try:
            os.unlink(path + ext)
        except OSError:
            pass
    return wall, erd, nodes[0], group_words_seconds[0]


print(f'budget={BUDGET}  deadline={DEADLINE_S}s  vocab={len(vocab)}  '
      f'answers={len(answers)}  sizes={BRANCH_SIZES}', flush=True)
mismatches = []
for size in BRANCH_SIZES:
    if size > len(answers):
        print(f'\n=== branch n={size} skipped (larger than answer list) ===', flush=True)
        continue
    branch_words = sorted(rng.sample(answers, size))
    print(f'\n=== branch n={size} ===', flush=True)
    results = {}
    for tag, matrix in (('off', None), ('on', pm)):
        wall, erd, nodes, group_words_seconds = solve(branch_words, matrix)
        share = (group_words_seconds / wall * 100) if wall > 0 else 0.0
        erd_s = 'abort' if erd is None else f'{erd:.5f}'
        print(f'  {tag:>3}: wall {wall:7.2f}s  erd {erd_s}  nodes {nodes:7d}  '
              f'group_words {share:5.1f}%', flush=True)
        results[tag] = erd
    # Exact equality, per Part II rule 1: the matrix-on and matrix-off
    # paths must produce bit-identical results, never "close enough" —
    # a tolerance here would hide the exact bug this bench exists to
    # catch. `==` also covers the both-aborted (None) case correctly.
    expected, actual = results['off'], results['on']
    if expected == actual:
        print('  -> ERD MATCH', flush=True)
    else:
        print(f'  -> *** ERD DIFF *** off={expected!r} on={actual!r}', flush=True)
        mismatches.append((size, expected, actual))

print('\ndone', flush=True)
if mismatches:
    print(f'\n{len(mismatches)} branch(es) diverged between matrix-on and '
          f'matrix-off: {mismatches}', flush=True)
    sys.exit(1)
