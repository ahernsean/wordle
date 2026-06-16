"""backfill_max_depth.py — one-time migration for depth-limited ERD.

The legacy ERD_ALL cache holds correct *unlimited-optimal* (best_word, ERD)
rows but no max_depth, so budget-aware reads reject them (depth unknown) and
recompute from scratch — intractable.  But a legacy entry's worst-case line
length can be recovered cheaply by walking its already-cached best_word
subtree (1 + max of children's depths), with NO re-search.

Legacy rows were computed with no depth cap, so the cap never excluded any
candidate: they are untainted by definition.  We therefore set solve_budget
= NULL (reusable at any budget >= max_depth).  Entries whose subtree is not
fully cached can't be walked; we leave them NULL (recomputed on demand).

Reports the max_depth distribution, notably how many entries have
max_depth > 5 — positions whose unlimited-optimal strategy can't win within a
post-opener budget of 5 (a losing tail the cap will force a safer line for).
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from cache_sqlite import ScoreCache
from wordle_engine import load_word_list, ResponseCache, ERD_ALL

ANSWER_FILE = 'NYT_wordlist.txt'
CACHE = 'wordle_cache.sqlite3'
BATCH = 20000

INCOMPLETE = -1   # memo sentinel: subtree not fully cached, can't backfill


def main():
    answers = load_word_list(ANSWER_FILE)
    sc = ScoreCache(CACHE, answers, checkpoint_on_close=False)
    rc = ResponseCache(answers, sc)
    uid = sc.universe_id

    def subset_words(key):
        return [key[i:i + 5].decode() for i in range(0, len(key), 5)]

    memo = {}   # subset_key bytes -> max_depth int (or INCOMPLETE)

    def max_depth(key):
        cached = memo.get(key)
        if cached is not None:
            return cached
        words = subset_words(key)
        n = len(words)
        if n == 1:
            memo[key] = 1
            return 1
        ent = sc.read_with_depth(key, ERD_ALL)
        if ent is None:
            memo[key] = INCOMPLETE
            return INCOMPLETE
        best_word, _score, md, _sb = ent
        if md is not None:           # already known (pre-set or prior run)
            memo[key] = md
            return md
        groups = rc.group_words(best_word, words)
        result = 1
        for sub in groups.values():
            k = len(sub)
            if k == 1 and sub[0] == best_word:
                continue             # self: solved by best_word, line length 1
            if k == 1:
                result = max(result, 2)   # best_word wrong -> play the one word
                continue
            sub_md = max_depth(ScoreCache.encode_subset(sub))
            if sub_md == INCOMPLETE:
                memo[key] = INCOMPLETE
                return INCOMPLETE
            result = max(result, 1 + sub_md)
        memo[key] = result
        return result

    total = sc._conn.execute(
        "SELECT COUNT(*) FROM subgroup_best_by_policy "
        "WHERE policy=? AND universe_id=? AND max_depth IS NULL",
        (ERD_ALL, uid)).fetchone()[0]
    print(f'legacy entries to backfill: {total:,}', flush=True)

    # Snapshot the keys up front (we mutate max_depth as we go).
    keys = [r[0] for r in sc._conn.execute(
        "SELECT subset_key FROM subgroup_best_by_policy "
        "WHERE policy=? AND universe_id=? AND max_depth IS NULL",
        (ERD_ALL, uid))]
    print(f'snapshotted {len(keys):,} keys', flush=True)

    hist = Counter()
    incomplete = 0
    pending = []   # (max_depth, subset_key)
    t0 = time.time()

    def flush():
        if not pending:
            return
        sc._conn.execute('BEGIN')
        # Guard with max_depth IS NULL: if a worker has, since the snapshot,
        # rewritten this entry with a real depth (and possibly a tainted
        # solve_budget), this UPDATE must NOT clobber its solve_budget back to
        # NULL — that would mislabel a budget-specific entry as reusable at any
        # budget and yield wrong ERD answers.  The guard makes the backfill a
        # no-op on any row a worker has already filled in, so it is safe to run
        # concurrently and is naturally idempotent.
        sc._conn.executemany(
            "UPDATE subgroup_best_by_policy SET max_depth=?, solve_budget=NULL "
            "WHERE subset_key=? AND policy=? AND universe_id=? "
            "AND max_depth IS NULL",
            [(md, k, ERD_ALL, uid) for md, k in pending])
        sc._conn.execute('COMMIT')
        pending.clear()

    for i, key in enumerate(keys, 1):
        md = max_depth(key)
        if md == INCOMPLETE:
            incomplete += 1
        else:
            hist[md] += 1
            pending.append((md, key))
            if len(pending) >= BATCH:
                flush()
        if i % 100000 == 0:
            rate = i / max(1e-6, time.time() - t0)
            print(f'  {i:,}/{len(keys):,}  ({rate:,.0f}/s)  '
                  f'incomplete={incomplete:,}', flush=True)
    flush()
    sc.checkpoint()
    sc.close()

    print('\n=== max_depth distribution ===', flush=True)
    for d in sorted(hist):
        print(f'  depth {d:>2}: {hist[d]:,}')
    over5 = sum(c for d, c in hist.items() if d > 5)
    print(f'\nbackfilled: {sum(hist.values()):,}')
    print(f'incomplete (left NULL, recompute on demand): {incomplete:,}')
    print(f'max_depth > 5 (unlimited optimum exceeds post-opener budget): {over5:,}')


if __name__ == '__main__':
    sys.exit(main())
