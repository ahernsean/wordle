#!/usr/bin/env python3.13
"""verify_branch_depths.py — audit branch_best_by_policy's max_depth column.

A branch row's `max_depth` is not an independent measurement: it is fully
determined by the row's own `best_guess` and the `max_depth` of each response
group that guess produces.  Any disagreement between the stored value and that
fold is therefore a genuine inconsistency, not a difference of opinion between
two searches.

A branch holds up to two kinds of exact result — the unrestricted optimum in
`branch_best_by_policy`, and one per budget in
`branch_best_by_policy_and_budget` — and the fold has to read the one its
parent actually used.  An unrestricted parent's subtrees were solved
unrestricted; a parent solved under budget b spent one guess reaching each
child, so it read them at b-1: the unrestricted child when its own worst case
fits there, and otherwise the child solved at exactly b-1.

Both facts once shared one row, and the last write won.  Nothing records which
of a child's values a parent folded, so a parent left holding the other one
cannot be found by query — only by redoing the fold.  That is what this audit
does, and it stays worth doing after the split: it is what says whether a cache
carries damage from before it.

Two directions of disagreement, with very different consequences:

  stored below the fold — unsound.  `_cache_reuse` gates an untainted entry on
    `max_depth <= budget`, so an understated depth offers a strategy at a
    budget that strategy cannot actually meet.
  stored above the fold — conservative.  Reuse is refused where it was
    available; results stay correct.

The pass runs bottom-up (ascending branch size), so a child corrected in this
run is what its parents are folded against.  A naive fold that reads only
stored values undercounts: an understated child understates its ancestors, and
reading that same child back agrees with them.

`--repair` writes each folded value back, and rewrites `max_depth` only.  A
stale `best_score` is reported but never rewritten: a wrong ERD may mean
`best_guess` is no longer the argmin, which only a re-search
(verify_erd_cache.py) can settle.

The two directions are not equally safe to repair, so they are not repaired
alike.  Raising a depth only ever withdraws reuse, and is always applied.
Lowering one widens the budget range the row is offered at, which is a claim
about a strategy — so it is applied only when the row's `best_score` agrees
with its own fold, and withheld otherwise rather than extending the reach of a
score this pass has just contradicted.

A repaired row needs nothing invalidated above it.  A candidate's own ERD is
folded from its response groups' rows on every read, so the next report sees
the repaired depth.

An audit-only run opens the cache read-only, so it can be run against a live
one.  Stop the swarm before running with --repair.

Usage:
    python3.13 verify_branch_depths.py
    python3.13 verify_branch_depths.py --list 20
    python3.13 verify_branch_depths.py --repair
    python3.13 verify_branch_depths.py --json

Exits 1 when an audit-only run finds rows whose stored depth is below the
fold, so a scheduled run reports the unsound ones without being read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

from cache_sqlite import ScoreCache, branch_reference
from erd_queue import decode_subset
from wordle_engine import ERD_ALL, VALID_ERD_POLICIES, ResponseCache, load_word_list

from runtime_paths import (
    DEFAULT_ANSWER_LIST_PATH,
    DEFAULT_CACHE_PATH,
    ensure_runtime_dir,
)

# A singleton branch is never stored: _solve_subset answers n == 1 directly
# with a max_remaining_depth of 1, before it ever reads the cache.
SINGLETON_MAX_REMAINING_DEPTH = 1

# The ERD fold sums its groups in evaluate_candidate's own order, so a row the
# solver wrote reproduces bit-exactly.  The tolerance only absorbs rows written
# by some other path.
SCORE_TOLERANCE = 1e-9


class DepthFold:
    """One row's max_depth and ERD as its own stored strategy determines them.

    `depth` and `erd` are None when `missing` or `degenerate` is set — the fold
    could not be completed, so the stored values are neither confirmed nor
    contradicted.
    """

    __slots__ = ('depth', 'erd', 'missing', 'degenerate')

    def __init__(self, depth=None, erd=None, missing=(), degenerate=False):
        self.depth = depth
        self.erd = erd
        self.missing = tuple(missing)
        self.degenerate = degenerate

    @property
    def complete(self):
        return self.depth is not None


def fold_branch(branch_words, best_guess, response_cache, child_lookup):
    """Fold one branch's max_depth and ERD from best_guess's response groups.

    Mirrors evaluate_candidate's recurrence: the guess itself is one guess; the
    group holding only the guess is finished by playing it; every other group
    costs one guess more than its own subtree.

    child_lookup(branch_key) returns (max_depth, erd) for a stored group, or
    None when no row holds it.  It carries the scope: a fold is only sound
    against the results its own search would have read, so the caller supplies
    a lookup already bound to the parent's budget.  Groups of one word are
    answered here rather than looked up, matching _solve_subset's own n == 1
    base case.
    """
    n = len(branch_words)
    groups = response_cache.group_words(best_guess, branch_words)
    depth = 1
    erd = 1.0
    missing = []
    # Largest group first, as evaluate_candidate accumulates them: floating
    # point addition is not associative, so the ERD fold only reproduces a
    # stored best_score exactly when it adds the same terms in the same order.
    for group in sorted(groups.values(), key=len, reverse=True):
        k = len(group)
        if k >= n:
            # The guess separates nothing: this row's own branch back again.
            return DepthFold(degenerate=True)
        if k == 1:
            if group[0] == best_guess:
                continue
            child_depth, child_erd = SINGLETON_MAX_REMAINING_DEPTH, 1.0
        else:
            child = child_lookup(ScoreCache.encode_subset(group))
            if child is None or child[0] is None:
                missing.append(ScoreCache.encode_subset(group))
                continue
            child_depth, child_erd = child
        depth = max(depth, 1 + child_depth)
        erd += (k / n) * child_erd
    if missing:
        return DepthFold(missing=missing)
    return DepthFold(depth=depth, erd=erd)


class DepthAudit:
    """Bottom-up fold of every stored branch row, with optional repair.

    Rows must arrive in ascending branch size so each row's response groups —
    always strictly smaller than the row itself — are already folded, in both
    scopes.
    """

    def __init__(self, score_cache, policy, response_cache, repair=False):
        self._cache = score_cache
        self._policy = policy
        self._responses = response_cache
        self._repair = repair
        # (branch_key, solve_budget) -> (max_depth, best_score) as the audit
        # now believes them: the folded depth where the fold completed, the
        # stored depth otherwise.  solve_budget None is the unrestricted result.
        self._known = {}
        self.checked = 0
        self.legacy = 0
        self.incomplete = 0
        self.unresolved_groups = 0
        self.degenerate = 0
        self.depth_too_low = 0
        self.depth_too_high = 0
        self.score_stale = 0
        self.repaired = 0
        self.repair_withheld = 0
        self.depth_deltas = Counter()
        self.tainted_split = Counter()
        self.mismatch_sizes = Counter()
        self.findings = []

    def _child_lookup(self, parent_budget):
        """Group lookup as a parent at `parent_budget` would have read it.

        An unrestricted parent read unrestricted children.  A parent capped at
        b reached each child having spent one guess, so it read them at b-1 —
        preferring the unrestricted child whose own worst case fits there, and
        falling back to the one solved at exactly b-1.  Mirrors
        ScoreCache.read_for_budget one level down.
        """
        if parent_budget is None:
            return lambda branch_key: self._known.get((branch_key, None))
        child_budget = parent_budget - 1

        def lookup(branch_key):
            canonical = self._known.get((branch_key, None))
            if (canonical is not None and canonical[0] is not None
                    and canonical[0] <= child_budget):
                return canonical
            return self._known.get((branch_key, child_budget))

        return lookup

    def run(self, rows, list_limit=0, progress=None):
        for row in rows:
            self._audit_row(row, list_limit)
            if progress is not None:
                progress(self.checked)
        return self

    def _audit_row(self, row, list_limit):
        branch_key = bytes(row['branch_key'])
        stored_depth = row['max_depth']
        stored_score = row['best_score']
        best_guess = row['best_guess']
        solve_budget = row['solve_budget']
        self.checked += 1
        scope = solve_budget
        self._known[(branch_key, scope)] = (stored_depth, stored_score)

        if stored_depth is None:
            # A legacy row records no depth at all; the budget-aware reader
            # already rejects it, so there is nothing to contradict.
            self.legacy += 1
            return

        branch_words = decode_subset(branch_key)
        fold = fold_branch(branch_words, best_guess, self._responses,
                           self._child_lookup(scope))
        if fold.degenerate:
            self.degenerate += 1
            return
        if not fold.complete:
            self.incomplete += 1
            self.unresolved_groups += len(fold.missing)
            return

        self._known[(branch_key, scope)] = (fold.depth, stored_score)
        score_agrees = abs(fold.erd - stored_score) <= SCORE_TOLERANCE
        if not score_agrees:
            self.score_stale += 1
        if fold.depth == stored_depth:
            return

        if fold.depth > stored_depth:
            # Raising a depth only withdraws reuse; safe whatever the score is.
            self.depth_too_low += 1
            safe_to_repair = True
        else:
            # Lowering one offers the row at budgets that previously rejected
            # it.  That is a claim about the strategy, so it needs a score this
            # pass has confirmed rather than one it has just contradicted.
            self.depth_too_high += 1
            safe_to_repair = score_agrees
        self.depth_deltas[(stored_depth, fold.depth)] += 1
        self.tainted_split['tainted' if solve_budget is not None else 'untainted'] += 1
        self.mismatch_sizes[len(branch_words)] += 1
        if len(self.findings) < list_limit:
            self.findings.append({
                'branch_reference': branch_reference(branch_key),
                'branch_size': len(branch_words),
                'best_guess': best_guess,
                'stored_max_depth': stored_depth,
                'folded_max_depth': fold.depth,
                'solve_budget': solve_budget,
            })
        if not self._repair:
            return
        if not safe_to_repair:
            self.repair_withheld += 1
            return
        if self._cache.repair_max_depth(branch_key, self._policy, fold.depth,
                                        solve_budget=scope):
            self.repaired += 1

    def summary(self):
        return {
            'checked': self.checked,
            'legacy': self.legacy,
            'incomplete': self.incomplete,
            'unresolved_groups': self.unresolved_groups,
            'degenerate': self.degenerate,
            'depth_too_low': self.depth_too_low,
            'depth_too_high': self.depth_too_high,
            'score_stale': self.score_stale,
            'repaired': self.repaired,
            'repair_withheld': self.repair_withheld,
            'depth_deltas': {f'{was} -> {now}': count
                             for (was, now), count in sorted(self.depth_deltas.items())},
            'tainted_split': dict(sorted(self.tainted_split.items())),
            'mismatch_sizes': dict(sorted(self.mismatch_sizes.items())),
            'findings': self.findings,
        }


class _ReadOnlyDecompositions:
    """A ResponseCache backing store that reads the cache but never adds to it.

    ResponseCache persists a guess's pattern blob on first use, which would
    make an audit a writer.  Reads still go to the cache, so a run over a
    warm one pays nothing to recompute.
    """

    def __init__(self, score_cache):
        self._cache = score_cache

    def read_decomposition(self, guess):
        return self._cache.read_decomposition(guess)

    def write_decomposition(self, guess, blob):
        pass


def iter_rows(score_cache, policy):
    """Every stored branch result for one policy, smallest branch first.

    Spans both branch tables, so a branch with an unrestricted result and two
    budget-specific ones yields three rows — each is a separate fact with its
    own fold.  Yielded one branch size at a time, each wave read to completion
    before it is handed out: --repair updates the same tables the rows come
    from, and a cursor still open over one would be reading a moving target.
    """
    conn = score_cache._conn
    answer_list_id = score_cache.answer_list_id
    sizes = [row[0] for row in conn.execute("""
        SELECT DISTINCT length(branch_key) / 5 AS branch_size
        FROM branch_best_by_policy
        WHERE policy = ? AND answer_list_id = ?
        UNION
        SELECT DISTINCT length(branch_key) / 5 AS branch_size
        FROM branch_best_by_policy_and_budget
        WHERE policy = ? AND answer_list_id = ?
        ORDER BY branch_size
    """, (policy, answer_list_id, policy, answer_list_id)).fetchall()]
    for size in sizes:
        # Both scopes of one branch size together: a parent is strictly larger
        # than every group it folds, so anything a fold needs is already done.
        yield from conn.execute("""
            SELECT branch_key, best_guess, best_score, max_depth, solve_budget
            FROM branch_best_by_policy
            WHERE policy = ? AND answer_list_id = ? AND length(branch_key) / 5 = ?
            UNION ALL
            SELECT branch_key, best_guess, best_score, max_depth, solve_budget
            FROM branch_best_by_policy_and_budget
            WHERE policy = ? AND answer_list_id = ? AND length(branch_key) / 5 = ?
        """, (policy, answer_list_id, size,
              policy, answer_list_id, size)).fetchall()


def _size_span(sizes):
    """Render a mismatch-size histogram as the branch-size span holding it."""
    if not sizes:
        return 'none'
    return f'n={min(sizes):,}-{max(sizes):,}'


def render_report(summary, elapsed, repair):
    lines = [
        f"checked {summary['checked']:,}   "
        f"incomplete {summary['incomplete']:,} "
        f"({summary['unresolved_groups']:,} groups unresolved)   "
        f"legacy {summary['legacy']:,}   "
        f"({elapsed:.0f}s)",
    ]
    if summary['degenerate']:
        lines.append(f"  best_guess separates nothing:   {summary['degenerate']:,}")
    lines.append(
        f"  stored max_depth TOO LOW  (unsound reuse): {summary['depth_too_low']:,}")
    lines.append(
        f"  stored max_depth too high (conservative):  {summary['depth_too_high']:,}")
    if summary['depth_deltas']:
        deltas = ',  '.join(f'{k}: {v:,}' for k, v in summary['depth_deltas'].items())
        lines.append(f'  deltas: {{{deltas}}}')
        split = ',  '.join(f'{k}: {v:,}' for k, v in summary['tainted_split'].items())
        lines.append(f'  tainted split: {{{split}}}')
        lines.append(f"  sizes: {_size_span(summary['mismatch_sizes'])}")
    lines.append(
        f"  stored best_score disagrees with its own fold: {summary['score_stale']:,}")
    if repair:
        lines.append(f"  max_depth rows repaired: {summary['repaired']:,}")
        lines.append(
            f"  repairs withheld (would widen reuse for a stale score): "
            f"{summary['repair_withheld']:,}")
    for finding in summary['findings']:
        lines.append(
            f"    {finding['branch_reference']}  n={finding['branch_size']:,}  "
            f"{finding['best_guess']}  stored {finding['stored_max_depth']} "
            f"-> folded {finding['folded_max_depth']}  "
            f"solve_budget={finding['solve_budget']}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Audit branch_best_by_policy max_depth against the fold '
                    'its own best_guess and response groups determine.')
    parser.add_argument('--cache', default=DEFAULT_CACHE_PATH, metavar='PATH',
                        help='Cache database to audit (default: %(default)s)')
    parser.add_argument('--answers', default=DEFAULT_ANSWER_LIST_PATH, metavar='PATH',
                        help='Answer word list (default: %(default)s)')
    parser.add_argument('--policy', default=ERD_ALL, choices=sorted(VALID_ERD_POLICIES),
                        help='Search policy to audit (default: %(default)s)')
    parser.add_argument('--repair', action='store_true',
                        help='Write each folded max_depth back.  Stop the '
                             'swarm first: a worker holding a branch open '
                             'will overwrite the repair with its own value.')
    parser.add_argument('--list', type=int, default=0, metavar='N',
                        help='Also show the first N disagreeing rows')
    parser.add_argument('--json', action='store_true',
                        help='Emit the summary as JSON instead of text')
    args = parser.parse_args(argv)
    ensure_runtime_dir()

    if not os.path.exists(args.cache):
        parser.error(f'no cache at {os.path.abspath(args.cache)}')

    answer_words = load_word_list(args.answers)
    score_cache = ScoreCache(args.cache, answer_words, checkpoint_on_close=False,
                             read_only=not args.repair)
    # The fold reads response_decomposition rows the solver already wrote
    # rather than recomputing every guess's patterns.  It never writes one
    # back: a guess the cache has not decomposed stays decomposed in memory
    # for this run, so an audit adds nothing to the file it is auditing.
    responses = ResponseCache(score_cache.answer_words,
                              score_cache=_ReadOnlyDecompositions(score_cache))
    if not args.json:
        print(f'Cache  : {os.path.abspath(args.cache)}')
        print(f'Policy : {args.policy}')
        print(f'Mode   : {"repair" if args.repair else "audit only (read-only)"}')
        print(flush=True)

    started = time.time()
    audit = DepthAudit(score_cache, args.policy, responses, repair=args.repair)
    try:
        audit.run(iter_rows(score_cache, args.policy), list_limit=args.list)
    finally:
        score_cache.close()
    summary = audit.summary()

    if args.json:
        summary['elapsed_seconds'] = round(time.time() - started, 3)
        print(json.dumps(summary, indent=2))
    else:
        print(render_report(summary, time.time() - started, args.repair))
    return 1 if summary['depth_too_low'] and not args.repair else 0


if __name__ == '__main__':
    sys.exit(main())
