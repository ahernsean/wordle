"""diag_ab_equiv.py — correctness check: alpha-beta engine must return the
EXACT same ERD as the pre-change engine on every input.

Imports the committed engine (no ceilings) and the current engine (alpha-beta)
side by side, runs min_expected_guesses on a spread of real word sets at several
budgets, and asserts the costs match to 1e-9.  score_cache=None on both so each
recomputes from scratch (no shared SQLite state).
"""
import importlib.util

import wordle_engine as new_eng


def _load_old():
    spec = importlib.util.spec_from_file_location(
        'wordle_engine_old', '/tmp/wordle_engine_old.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old_eng = _load_old()

answers = new_eng.load_word_list('NYT_wordlist.txt')
vocab = new_eng.load_word_list('wordle.txt')
# Bounded guess vocabulary (same for both engines) to keep the check fast.
guess_vocab = sorted(set(vocab[:250]) | set(answers[:60]))

CASES = [
    (8, None), (12, None), (16, None),
    (12, 6), (20, 6), (30, 5), (40, 5),
]

print(f'guess_vocab={len(guess_vocab)} words')
all_ok = True
for n, budget in CASES:
    words = answers[:n]
    new_rc = new_eng.ResponseCache(answers)
    old_rc = old_eng.ResponseCache(answers)
    c_new = new_eng.min_expected_guesses(
        words, new_rc, None, guesses=guess_vocab, budget=budget)
    c_old = old_eng.min_expected_guesses(
        words, old_rc, None, guesses=guess_vocab, budget=budget)
    if c_new is None or c_old is None:
        match = (c_new is None and c_old is None)
    else:
        match = abs(c_new - c_old) < 1e-9
    all_ok = all_ok and match
    tag = 'OK ' if match else 'MISMATCH'
    print(f'  n={n:>3} budget={str(budget):>4}: new={c_new}  old={c_old}  {tag}')

print('\nALL MATCH' if all_ok else '\n*** MISMATCH DETECTED ***')
