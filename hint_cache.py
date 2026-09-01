"""Read-only historical cache consulted for candidate ordering only.

A clean ERD rebuild runs against two databases with different trust levels: an
empty writable live cache holding only results the current solver recomputed,
and a quarantined historical cache whose rows were produced by earlier solver
versions and are therefore descriptive history, not certificates.

This module is the whole of the second one's interface.  HintCache answers
exactly one question — "which word did the historical search pick for this
branch?" — and its queries select `best_guess` alone.  A stored ERD, worst
case, solve budget, or loss row cannot leave this module, so no caller can fold
an unverified value into a new parent or use one as an alpha-beta ceiling.  The
hinted word is re-evaluated in full against the live cache before it can become
an incumbent; a wrong hint costs evaluation order and nothing else.

The artifact is a checkpointed snapshot, opened `mode=ro&immutable=1`.  Plain
`mode=ro` rejects SQL writes but still touches the WAL shared-memory sidecar,
so it makes the file read-only without making it observationally immutable.
`immutable=1` closes that gap, at the price of ignoring an uncheckpointed WAL —
which is why open_hint_cache refuses a nonempty WAL or a hot rollback journal
outright rather than silently reading a stale snapshot of the data.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from cache_sqlite import LRUDict, answer_list_id, present_pre_split_cache_by_scope

logger = logging.getLogger("wordle")

# Hints are read once per (branch, policy, budget) and then re-read every time
# a sibling search revisits the same branch, so a session memo removes most of
# the SQLite traffic.  Kept modest because the keys are branch keys — five
# bytes per answer word, so a near-root branch's key runs to several
# kilobytes — and a worker process lives for hours across millions of
# branches.  A miss costs one indexed read, which is what the lookup would
# have been anyway.
HINT_MEMO_ENTRIES = 50_000

# Distinguishes "memoized as no hint" from "not memoized yet"; a plain None
# default would re-query SQLite on every repeat visit to a branch the artifact
# does not cover, which is the common case on a partial artifact.
_MISSING = object()


class HintCacheError(Exception):
    """The configured hint artifact cannot be used as one.

    Raised at startup only.  Every condition it reports — a missing path, a
    path that resolves onto the live cache, an uncheckpointed WAL, an
    unreadable answer-list namespace — means the operator asked for something
    the run cannot honestly provide, so the run refuses to start rather than
    proceeding with hints silently disabled.
    """


def _sidecar_is_nonempty(path: Path, suffix: str) -> bool:
    sidecar = path.with_name(path.name + suffix)
    try:
        return sidecar.stat().st_size > 0
    except OSError:
        return False


def open_hint_cache(hint_path, answer_words, live_cache_path):
    """Open `hint_path` as a hint source for a run whose live cache is
    `live_cache_path`, or raise HintCacheError.

    Returns None when hint_path is None — running without hints is the
    default, not a failure.
    """
    if hint_path is None:
        return None
    return HintCache(hint_path, answer_words, live_cache_path=live_cache_path)


class HintCache:
    """Historical candidate choices, usable only to reorder a candidate list.

    Every public result is a word.  Nothing this class returns is admissible
    as an exact result, a bound, or a feasibility fact.
    """

    def __init__(self, db_path, answer_words, live_cache_path=None,
                 memo_entries=HINT_MEMO_ENTRIES):
        self.db_path = Path(db_path)
        self.answer_list_id = answer_list_id(list(answer_words))
        if live_cache_path is not None:
            self._refuse_live_cache(Path(live_cache_path))
        self._refuse_unless_immutable_snapshot()
        self._conn = self._open_immutable()
        self._memo = LRUDict(max_size=memo_entries)
        # Hint accounting, all per-process and all ordinary metrics.
        #
        # lookups/hits/accepted/rejected count LOOKUPS, from both hint sites.
        # A cooperative branch is looked up once by each worker that computes
        # its packing order, so these are not branch counts; what they measure
        # is coverage (hits/lookups: did the artifact name a word) and legality
        # (accepted/hits: was that word in the pool), and both ratios are
        # internally consistent because accepted + rejected == hits.
        #
        # The inline pair is a different, smaller population and is kept apart
        # for that reason: one worker owns a whole inline _solve_subset frame
        # from placement to winner, so inline_wins / inline_placements is a
        # like-for-like rate.  There is deliberately no swarm-branch winner
        # counter — a cooperative branch's winner is decided once, at finalize,
        # by whichever worker wins it, and is recorded per branch in
        # branch_finalize_log.hint_was_winner rather than added to a
        # per-process counter several workers would each contribute to.
        self.lookups = 0
        self.hits = 0
        self.accepted = 0
        self.rejected = 0
        self.inline_placements = 0
        self.inline_wins = 0
        self.namespace_branch_count = self._inspect_answer_list_namespace()

    def _refuse_live_cache(self, live_path):
        """Refuse a hint path that names the run's own writable cache.

        Checked before the file is opened: the whole separation rests on the
        two databases being different files, and a shell that expanded the
        same path into both flags would otherwise produce a run that reports
        hint hits for rows it wrote itself.
        """
        if self.db_path.resolve() == live_path.resolve():
            raise HintCacheError(
                f"hint cache {self.db_path} resolves to the live cache "
                f"{live_path}; the hint artifact must be a separate file")
        try:
            same = (self.db_path.exists() and live_path.exists()
                    and os.path.samefile(self.db_path, live_path))
        except OSError:  # pragma: no cover — stat raced with a delete
            same = False
        if same:
            raise HintCacheError(
                f"hint cache {self.db_path} is the same file as the live cache "
                f"{live_path}; the hint artifact must be a separate file")

    def _refuse_unless_immutable_snapshot(self):
        """Refuse anything but a present, fully checkpointed database file.

        immutable=1 tells SQLite the file cannot change, so it neither reads
        nor creates the -wal/-shm sidecars.  That is what keeps a worker run
        from leaving a fingerprint on the quarantined artifact, and it is also
        why uncheckpointed WAL content would be read straight past: the open
        must therefore reject a nonempty WAL rather than quietly serve an
        older snapshot of the same database.
        """
        if not self.db_path.exists():
            raise HintCacheError(f"hint cache {self.db_path} does not exist")
        if _sidecar_is_nonempty(self.db_path, "-wal"):
            raise HintCacheError(
                f"hint cache {self.db_path} has a nonempty write-ahead log; "
                f"checkpoint it into the database file before using it as a "
                f"hint artifact (an immutable open would ignore the log)")
        if _sidecar_is_nonempty(self.db_path, "-journal"):
            raise HintCacheError(
                f"hint cache {self.db_path} has a hot rollback journal; "
                f"recover it with a normal open before using it as a hint "
                f"artifact")

    def _open_immutable(self):
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro&immutable=1", uri=True,
                isolation_level=None)
        except sqlite3.Error as error:
            raise HintCacheError(
                f"hint cache {self.db_path} could not be opened read-only: "
                f"{error}") from error
        conn.row_factory = sqlite3.Row
        try:
            has_branch_results = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ('branch_best_by_policy',)).fetchone()
            if has_branch_results:
                present_pre_split_cache_by_scope(conn)
        except sqlite3.Error as error:
            conn.close()
            raise HintCacheError(
                f"hint cache {self.db_path} does not hold branch results: "
                f"{error}") from error
        if not has_branch_results:
            conn.close()
            raise HintCacheError(
                f"hint cache {self.db_path} does not hold branch results: "
                f"no branch_best_by_policy table")
        return conn

    def _inspect_answer_list_namespace(self):
        """Branch rows this artifact holds for the run's answer list.

        Every hint row is keyed by answer_list_id, so an artifact built for a
        different word list answers nothing.  Zero rows is a legitimate (if
        useless) state and starts normally; a namespace that cannot be read at
        all is a misconfigured artifact and refuses the run.
        """
        try:
            row = self._conn.execute("""
                SELECT (SELECT COUNT(*) FROM branch_best_by_policy
                        WHERE answer_list_id = ?)
                     + (SELECT COUNT(*) FROM branch_best_by_policy_and_budget
                        WHERE answer_list_id = ?) AS branch_rows
            """, (self.answer_list_id, self.answer_list_id)).fetchone()
        except sqlite3.Error as error:
            self.close()
            raise HintCacheError(
                f"hint cache {self.db_path} answer-list namespace could not be "
                f"inspected: {error}") from error
        return row["branch_rows"]

    def hint_candidate(self, branch_key, policy, budget, count_lookup=True):
        """The word the historical search chose for this branch, or None.

        Ordering information only.  The caller may put this word first in its
        candidate list; it may not treat its presence as evidence about the
        branch's cost, worst case, or solvability.

        A budget-specific historical row is preferred over the unrestricted
        one, which is the opposite of ScoreCache.read_for_budget's precedence
        and deliberately so: read_for_budget picks the row a search may
        *reuse*, where the unrestricted optimum wins because it is valid at
        more budgets, while this picks the row whose search conditions most
        closely match the caller's, because a closer match is the better guess
        at a strong candidate.  Nothing here affects correctness, so the two
        precedences need not agree.

        count_lookup=False reads the same answer without counting it.  A
        reporting caller — "which word did the artifact name for the branch
        this worker just finalized?" — is not a question the search asked, and
        counting it would inflate the lookup rate the hint payoff is measured
        against.
        """
        memo_key = (branch_key, policy, budget)
        cached = self._memo.get(memo_key, _MISSING)
        if cached is _MISSING:
            cached = self._read_hint(branch_key, policy, budget)
            self._memo[memo_key] = cached
        if count_lookup:
            self.lookups += 1
            if cached is not None:
                self.hits += 1
        return cached

    def _read_hint(self, branch_key, policy, budget):
        if budget is not None:
            row = self._conn.execute("""
                SELECT best_guess FROM branch_best_by_policy_and_budget
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
                  AND solve_budget = ?
            """, (branch_key, policy, self.answer_list_id, budget)).fetchone()
            if row is not None and row["best_guess"] is not None:
                return row["best_guess"]
        row = self._conn.execute("""
            SELECT best_guess FROM branch_best_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id)).fetchone()
        if row is not None and row["best_guess"] is not None:
            return row["best_guess"]
        return None

    def note_accepted_for_branch_order(self):
        """A hinted word led a swarm branch's packing order.

        Counted as an acceptance only.  Whether it went on to win the branch
        is settled at finalize, once, by one worker, and belongs on that
        branch's finalize row — not in a per-process counter every worker on
        the branch would contribute to independently.
        """
        self.accepted += 1

    def note_accepted_in_frame(self):
        """A hinted word led an inline solver frame's candidate list.

        Counted twice on purpose: once as an acceptance alongside the swarm's,
        and once in the inline population whose wins this process also sees.
        """
        self.accepted += 1
        self.inline_placements += 1

    def note_rejected(self):
        """A hinted word was absent from the candidate pool and ignored."""
        self.rejected += 1

    def note_inline_win(self):
        """An inline placement won its frame on recomputed evidence."""
        self.inline_wins += 1

    def stats(self):
        """Hint accounting for this process, for logs and heartbeats."""
        return {
            "hint_lookups": self.lookups,
            "hint_hits": self.hits,
            "hint_accepted": self.accepted,
            "hint_rejected": self.rejected,
            "hint_inline_placements": self.inline_placements,
            "hint_inline_wins": self.inline_wins,
        }

    def close(self):
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except sqlite3.Error:  # pragma: no cover — interpreter teardown
            pass
