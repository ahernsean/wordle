"""SQLite-backed cache for Wordle scores and lookahead results."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("wordle")


def _is_disk_io_error(exc):
    """True if exc is the transient 'disk I/O error' OperationalError that
    iCloud File Provider Storage raises when a sync pass holds the lock on
    the cache file or its WAL — see ScoreCache.checkpoint.
    """
    return "disk I/O error" in str(exc)


class ScoreCache:
    """Persists per-word scores and subgroup lookahead results.

    Tables:
      word_scores            — per-word scoring method results (level 1)
      subgroup_best_by_policy — the word a search policy judged best for a
                                subgroup, and the score that earned it that
                                judgment (levels 2+); the "by_policy" in the
                                table name carries the scoping that lets the
                                best_word/best_score columns stay short —
                                "best" is only ever read alongside the policy
                                that decided it
      universe               — fingerprint of the answer word set

    All entries are keyed by universe_id so a different answer list
    produces a clean namespace without needing a new file.
    """

    def __init__(self, db_path, answer_words, timeout=30.0,
                 checkpoint_on_close=True):
        self.db_path = Path(db_path)
        self.answer_words = list(answer_words)
        self.checkpoint_on_close = checkpoint_on_close
        self._conn = sqlite3.connect(
            self.db_path, timeout=timeout, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self.universe_id = self._ensure_universe()
        self.read_hits = 0
        self.read_misses = 0
        self.write_count = 0
        # In-memory mirror of subgroup_best_by_policy rows seen this session.
        # Subgroup results are write-once/exact, so a hit here is as good as
        # a SQLite hit but ~1000x cheaper — recursive ERD search re-reads the
        # same small subgroups millions of times across sibling branches.
        self._mem_cache = {}

    def _is_migration_done(self, name):
        """Return True if migration `name` has been recorded as complete."""
        return self._conn.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?", (name,)
        ).fetchone() is not None

    def _mark_migration_done(self, name):
        """Record migration `name` as complete so it is skipped on future opens."""
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, completed_at) VALUES (?, ?)",
            (name, int(time.time()))
        )

    def _ensure_schema(self):
        # Must be first: every migration guard below reads from this table.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name         TEXT PRIMARY KEY,
                completed_at INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                universe_id  TEXT PRIMARY KEY,
                answer_hash  TEXT NOT NULL,
                answer_count INTEGER NOT NULL,
                created_at   INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS response_decomposition (
                guess        TEXT    NOT NULL,
                universe_id  TEXT    NOT NULL,
                patterns     BLOB    NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (guess, universe_id)
            )
        """)
        # The subgroup-result table and its columns have been renamed twice:
        #   lookahead_result(best_word, best_entropy)
        #     -> subgroup_pick(picked_word, picked_score): "best" wrongly
        #        implied one policy's choice is objectively superior to
        #        another's, and "entropy" is flatly wrong for ERD-policy
        #        rows, which store an expected-remaining-guesses *cost*
        #        (lower is better — the opposite sense of entropy).
        #   subgroup_pick(picked_word, picked_score)
        #     -> subgroup_best_by_policy(best_word, best_score): "picked"
        #        solved the superiority problem but read awkwardly on its
        #        own ("the picked word for a subgroup" — picked by what?).
        #        Moving the missing context ("by policy") into the *table*
        #        name lets the column names stay short and natural —
        #        "best_word"/"best_score" — since any reader of the columns
        #        already has the table's name, and therefore the policy
        #        column, in view.
        # Migrate in place from either prior state so existing rows survive.
        tables = {row["name"] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "lookahead_result" in tables and "subgroup_pick" not in tables \
                and "subgroup_best_by_policy" not in tables:
            self._conn.execute(
                "ALTER TABLE lookahead_result RENAME TO subgroup_best_by_policy")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN best_entropy TO best_score")
            self._conn.execute("DROP INDEX IF EXISTS idx_lookahead")
        tables = {row["name"] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "subgroup_pick" in tables and "subgroup_best_by_policy" not in tables:
            self._conn.execute(
                "ALTER TABLE subgroup_pick RENAME TO subgroup_best_by_policy")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN picked_word TO best_word")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN picked_score TO best_score")
            self._conn.execute("DROP INDEX IF EXISTS idx_subgroup_pick")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS subgroup_best_by_policy (
                subset_key   BLOB NOT NULL,
                policy       TEXT NOT NULL,
                universe_id  TEXT NOT NULL,
                best_word    TEXT NOT NULL,
                best_score   REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (subset_key, policy, universe_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subgroup_best_by_policy
            ON subgroup_best_by_policy(universe_id, policy)
        """)
        # Covers MAX(updated_at) WHERE universe_id = ? — used by last_write_ts()
        # on every startup. Without this index, that query scans all 3M+ rows.
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subgroup_updated
            ON subgroup_best_by_policy(universe_id, updated_at)
        """)
        # 'subset_blob' was renamed to 'subset_key' — same encoding, cleaner
        # name that matches every other reference in this file.  Databases
        # migrated from lookahead_result or subgroup_pick may still carry the
        # old column name.
        cols = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(subgroup_best_by_policy)")}
        if cols and "subset_key" not in cols and "subset_blob" in cols:  # pragma: migration
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy "
                "RENAME COLUMN subset_blob TO subset_key")
        # max_depth: worst-case line length of best_word's strategy.  ERD is
        # now depth-limited ("expected remaining depth AND a guaranteed win
        # within budget"), so a cached entry is only reusable at a remaining
        # budget B when max_depth <= B.  Existing rows predate this and get
        # NULL — read as "depth unknown", hence never budget-safe, so they're
        # recomputed under the cap rather than trusted.  Nullable so the
        # column adds cleanly to a multi-GB file (metadata-only ALTER).
        cols = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(subgroup_best_by_policy)")}
        if cols and "max_depth" not in cols:
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy ADD COLUMN max_depth INTEGER")
        # solve_budget encodes the reuse range of a depth-limited entry:
        #   NULL  -> untainted: the cap never excluded any candidate anywhere,
        #            so the value IS the unconstrained optimum.  Reusable at
        #            any remaining budget >= max_depth.
        #   = b   -> tainted: a sibling candidate was killed by the cap, so
        #            this winner is only optimal *at budget b* (one more guess
        #            could revive the killed sibling).  Reusable only when the
        #            remaining budget == b.
        # Legacy rows are NULL but also have NULL max_depth, so the budget-aware
        # reader rejects them (unknown depth) and recomputes.
        cols = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(subgroup_best_by_policy)")}
        if cols and "solve_budget" not in cols:
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy ADD COLUMN solve_budget INTEGER")
        # ERD policy names were renamed so both axes of the (guess-universe x
        # compliance-filter) selection are spelled out in the namespace
        # itself — 'erd_all' named only the universe, 'erd_answers' folded
        # both axes into one word, and 'erd_constrained' named neither
        # explicitly. The new names are uniform: erd_<universe>_<compliance>.
        #   erd_all     -> erd_words_unfiltered   (all words,   no clue filter)
        #   erd_answers -> erd_answers_compliant  (answer list, clue-compliant)
        # 'erd_constrained' has no persisted rows: hard-mode ERD is
        # path-dependent and lives only in a transient MemoryScoreCache.
        if not self._is_migration_done('rename_erd_policies'):  # pragma: migration
            for old, new in (('erd_all', 'erd_words_unfiltered'),
                             ('erd_answers', 'erd_answers_compliant')):
                exists = self._conn.execute(
                    "SELECT 1 FROM subgroup_best_by_policy WHERE policy = ? LIMIT 1",
                    (old,)
                ).fetchone()
                if exists is not None:
                    self._conn.execute(
                        "UPDATE subgroup_best_by_policy SET policy = ? WHERE policy = ?",
                        (new, old))
            self._mark_migration_done('rename_erd_policies')
        # word_scores used to be keyed only by (word, method, universe_id) —
        # i.e. scoped to the whole answer set, so it could only ever cache
        # the very first guess of a game. Replace it with a subset-scoped
        # table (mirroring subgroup_best_by_policy) so any remaining-word position
        # that recurs gets its scores cached, not just the opening one.
        old_cols = {row["name"] for row in
                    self._conn.execute("PRAGMA table_info(word_scores)")}
        if old_cols and "subset_hash" not in old_cols:  # pragma: migration
            self._conn.execute("DROP TABLE word_scores")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS word_scores (
                subset_hash  TEXT    NOT NULL,
                word         TEXT    NOT NULL,
                method       TEXT    NOT NULL,
                score        REAL    NOT NULL,
                universe_id  TEXT    NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, universe_id, word)
            )
        """)
        # ScoringMethod.MINIMAX (named for the optimization strategy applied
        # to the metric) was renamed to MAX_GROUP_SIZE (named for the metric
        # itself, consistent with ENTROPY_GAIN/WEIGHTED_AVG/PROB_FINISH) —
        # carry forward any rows persisted under the old method key. Checked
        # via existence-first LIMIT 1 (see _purge_legacy_rows) so a table with
        # no such rows — the steady state once this has run once — costs only
        # a single indexed-or-not probe, not a full scan, on every connection.
        if not self._is_migration_done('rename_method_minimax'):
            stale_method = self._conn.execute(
                "SELECT 1 FROM word_scores WHERE method = 'minimax' LIMIT 1"
            ).fetchone()
            if stale_method is not None:
                self._conn.execute("""
                    UPDATE word_scores SET method = 'max_group_size'
                    WHERE method = 'minimax'
                """)
            self._mark_migration_done('rename_method_minimax')
        # All valid 5-letter words are ASCII, so a null byte identifies the
        # old null-separated subset-key encoding.
        self._purge_legacy_rows("instr(subset_key, char(0)) > 0", (),
                                migration_name='purge_null_sep_keys')
        # 'erd' was renamed to 'erd_answers' and then superseded by 'erd_all'.
        self._purge_legacy_rows("policy = ?", ('erd',),
                                migration_name='purge_policy_erd')
        # 'erd_hard' was renamed to 'erd_constrained'; constraint-compliant
        # mode is now always transient (MemoryScoreCache), so any persisted
        # rows under either name are useless regardless of age.
        self._purge_legacy_rows("policy = ?", ('erd_hard',),
                                migration_name='purge_policy_erd_hard')

    def _purge_legacy_rows(self, where, params, migration_name=None):
        """One-time cleanup of stale subgroup_best_by_policy rows.

        Once a legacy batch is gone it stays gone, so a full-table DELETE on
        every connection open (including each ERDSolver thread) would scan
        the whole table for nothing.  Check existence first — LIMIT 1 lets
        SQLite stop at the first match — and only DELETE when there's
        actually something to remove.

        migration_name: if given, skip the entire check on future opens once
        it has been recorded as done in schema_migrations.
        """
        if migration_name and self._is_migration_done(migration_name):
            return
        exists = self._conn.execute(
            f"SELECT 1 FROM subgroup_best_by_policy WHERE {where} LIMIT 1", params
        ).fetchone()
        if exists is not None:
            self._conn.execute(
                f"DELETE FROM subgroup_best_by_policy WHERE {where}", params)
        if migration_name:  # pragma: migration
            self._mark_migration_done(migration_name)

    def _ensure_universe(self):
        canonical = "\n".join(self.answer_words)
        universe_id = hashlib.sha256(canonical.encode()).hexdigest()
        now = int(time.time())
        self._conn.execute("""
            INSERT OR IGNORE INTO universe
                (universe_id, answer_hash, answer_count, created_at)
            VALUES (?, ?, ?, ?)
        """, (universe_id, universe_id, len(self.answer_words), now))
        return universe_id

    def close(self):
        if self.checkpoint_on_close:
            self.checkpoint()
        self._conn.close()

    def checkpoint(self):
        """Fold the WAL into the main database file (PRAGMA wal_checkpoint(TRUNCATE)).

        Leaves wordle_cache.sqlite3 self-contained with no -wal/-shm
        sidecars, so it's always safe to copy off-device - and the latest
        writes survive even if iOS suspends/kills the process without a
        clean close().

        This is an optimization, not a durability requirement: every write
        is already committed to the WAL, so a failed checkpoint loses
        nothing. On iOS the cache file lives under iCloud's File Provider
        Storage, where a sync pass can transiently hold the exclusive lock
        TRUNCATE needs - swallow that rather than letting it take down a
        background solver thread (or close()) over a no-op.
        """
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:
            logger.warning("wal_checkpoint(TRUNCATE) failed: %s", exc)

    # ------------------------------------------------------------------
    # Subgroup lookahead cache (levels 2+)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_subset(words):
        """Canonical key for a set of words: sorted, concatenated, no separator.

        All Wordle words are exactly 5 ASCII characters, so a key of length 5N
        encodes exactly N words recoverable by slicing at fixed 5-byte offsets.
        """
        return "".join(sorted(words)).encode("utf-8")

    def read(self, subset_key, policy):
        """Return (best_word, best_score) or None on cache miss.

        best_word is whichever word this policy's search judged best for
        this subgroup — judged by that policy's own metric, not some
        universal notion of "best". best_score is that metric's value for
        best_word, and its meaning is policy-dependent: an entropy in bits
        (higher is better) for lookahead policies ('full'/'hard'), or an
        expected-remaining-guesses cost (lower is better) for ERD policies
        ('erd_all'/'erd_answers'/'erd_constrained'). Callers that care about
        the number must already know which policy they asked for — the
        table name (subgroup_best_by_policy) and its policy column carry
        that scoping, so the columns themselves can stay "best_word"/
        "best_score" without re-litigating it.
        """
        cached = self._mem_cache.get((subset_key, policy))
        if cached is not None:
            self.read_hits += 1
            return cached[:2]
        full = self._read_full(subset_key, policy)
        if full is None:
            return None
        return full[:2]

    def read_with_depth(self, subset_key, policy):
        """Like read(), but returns (best_word, best_score, max_depth, solve_budget).

        max_depth is the worst-case line length of best_word's strategy (None
        for legacy rows).  solve_budget is the reuse-range marker (see schema):
        None = untainted, reusable at any budget >= max_depth; an int b =
        tainted, reusable only at remaining budget == b.  A budget-aware caller
        must apply that rule; a legacy row (max_depth None) is never reusable.
        """
        cached = self._mem_cache.get((subset_key, policy))
        if cached is not None:
            self.read_hits += 1
            return cached
        return self._read_full(subset_key, policy)

    def _read_full(self, subset_key, policy):
        row = self._conn.execute("""
            SELECT best_word, best_score, max_depth, solve_budget
            FROM subgroup_best_by_policy
            WHERE subset_key = ? AND policy = ? AND universe_id = ?
        """, (subset_key, policy, self.universe_id)).fetchone()
        if row is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        result = (row["best_word"], row["best_score"],
                  row["max_depth"], row["solve_budget"])
        self._mem_cache[(subset_key, policy)] = result
        return result

    def reset_read_counters(self):
        self.read_hits = 0
        self.read_misses = 0

    def write(self, subset_key, policy, best_word, best_score,
              max_depth=None, solve_budget=None):
        """Store the word a policy's search judged best for a subgroup, its
        score, and (for depth-limited ERD) the worst-case line length of that
        strategy plus its reuse-range marker.  max_depth=None marks a
        legacy/unbudgeted write; solve_budget None=untainted, int=tainted at
        that budget (see read_with_depth / schema).

        A transient 'disk I/O error' (e.g. iCloud File Provider Storage
        holding the cache file's lock during a sync pass — see checkpoint())
        is logged and swallowed rather than propagated: this runs at every
        level of a min_expected_guesses recursion, so letting it raise would
        unwind the entire call stack and abort the background solver thread,
        discarding every result computed this run — not just this one.
        best_word/best_score are still recorded in _mem_cache so this run's
        recursion keeps the memoization benefit even when the on-disk write
        fails; the row is simply recomputed on a later run.
        """
        now = int(time.time())
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO subgroup_best_by_policy
                    (subset_key, policy, universe_id,
                     best_word, best_score, updated_at, max_depth, solve_budget)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (subset_key, policy, self.universe_id,
                  best_word, best_score, now, max_depth, solve_budget))
            self.write_count += 1
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write(%s, %s, %.3f) failed: %s",
                            policy, best_word, best_score, exc)
        self._mem_cache[(subset_key, policy)] = (
            best_word, best_score, max_depth, solve_budget)

    def read_detail(self, subset_key, policy):
        """Like read(), but also returns the unix timestamp of the last write.

        Returns (best_word, best_score, updated_at) or None on a miss. Used
        by cache spot-checks (e.g. wordle.py's verify command) to show when
        a cached entry was written, alongside the per-prompt timestamps.
        """
        row = self._conn.execute("""
            SELECT best_word, best_score, updated_at
            FROM subgroup_best_by_policy
            WHERE subset_key = ? AND policy = ? AND universe_id = ?
        """, (subset_key, policy, self.universe_id)).fetchone()
        if row is None:
            return None
        return (row["best_word"], row["best_score"], row["updated_at"])

    def delete(self, subset_key, policy):
        """Remove a cached subgroup result so it gets recomputed.

        For invalidating an entry a spot-check has found to be inconsistent
        with its own cached subtree.
        """
        self._conn.execute("""
            DELETE FROM subgroup_best_by_policy
            WHERE subset_key = ? AND policy = ? AND universe_id = ?
        """, (subset_key, policy, self.universe_id))
        self._mem_cache.pop((subset_key, policy), None)

    # ------------------------------------------------------------------
    # Response decomposition cache (guess -> per-answer pattern bytes)
    # ------------------------------------------------------------------

    def read_decomposition(self, guess):
        """Return the cached pattern-byte blob for guess, or None on a miss.

        The blob holds one byte per answer word, in the same order as the
        answer list this cache was opened with — so the caller can zip it
        back against that list to recover the {answer: pattern} mapping.
        """
        row = self._conn.execute("""
            SELECT patterns FROM response_decomposition
            WHERE guess = ? AND universe_id = ?
        """, (guess, self.universe_id)).fetchone()
        if row is None:
            return None
        return row["patterns"]

    def write_decomposition(self, guess, patterns):
        """Store the pattern-byte blob (one byte per answer, canonical order).

        Swallows a transient 'disk I/O error' the same way write() does —
        see its docstring. ResponseCache._ensure caches the blob in memory
        regardless, so this run proceeds unaffected; only the on-disk copy
        is missing until a later run repersists it.
        """
        now = int(time.time())
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO response_decomposition
                    (guess, universe_id, patterns, updated_at)
                VALUES (?, ?, ?, ?)
            """, (guess, self.universe_id, patterns, now))
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_decomposition(%s) failed: %s", guess, exc)

    # ------------------------------------------------------------------
    # Word score cache (level 1, all ScoringMethods)
    # ------------------------------------------------------------------

    @staticmethod
    def _subset_hash(subset_key):
        """Compact, fixed-size key for a (potentially large) subset blob."""
        return hashlib.sha256(subset_key).hexdigest()

    def has_scores(self, subset_key, method):
        """Return True if any scores are cached for this subset/method/universe."""
        subset_hash = self._subset_hash(subset_key)
        return self._conn.execute("""
            SELECT 1 FROM word_scores
            WHERE subset_hash = ? AND method = ? AND universe_id = ?
            LIMIT 1
        """, (subset_hash, method, self.universe_id)).fetchone() is not None

    def read_scores(self, subset_key, method):
        """Return list of (word, score) for this subset/method/universe, or None if empty."""
        subset_hash = self._subset_hash(subset_key)
        rows = self._conn.execute("""
            SELECT word, score FROM word_scores
            WHERE subset_hash = ? AND method = ? AND universe_id = ?
        """, (subset_hash, method, self.universe_id)).fetchall()
        if not rows:
            return None
        return [(r["word"], r["score"]) for r in rows]

    def write_scores(self, subset_key, scores, method):
        """Store list of (word, score) tuples for this subset/method/universe.

        Swallows a transient 'disk I/O error' the same way write() does —
        see its docstring.
        """
        subset_hash = self._subset_hash(subset_key)
        now = int(time.time())
        try:
            self._conn.execute("BEGIN")
            self._conn.executemany("""
                INSERT OR REPLACE INTO word_scores
                    (subset_hash, word, method, score, universe_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(subset_hash, w, method, s, self.universe_id, now)
                  for w, s in scores])
            self._conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_scores(..., %s) failed: %s", method, exc)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def last_write_ts(self):
        """Return the unix timestamp of the most recent ERD write, or None."""
        row = self._conn.execute("""
            SELECT MAX(updated_at) AS m
            FROM subgroup_best_by_policy WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        return row["m"] if row else None

    def stats(self):
        """Return (subgroup_best_rows, word_score_rows, decomposition_rows, last_updated_ts)."""
        sp = self._conn.execute("""
            SELECT COUNT(*) AS c, MAX(updated_at) AS m
            FROM subgroup_best_by_policy WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        ws = self._conn.execute("""
            SELECT COUNT(*) AS c FROM word_scores WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        rd = self._conn.execute("""
            SELECT COUNT(*) AS c FROM response_decomposition WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        return sp["c"] or 0, ws["c"] or 0, rd["c"] or 0, sp["m"]


class MemoryScoreCache:
    """Transient in-memory ERD cache for path-dependent computations (hard mode).

    Implements the same read/write/encode_subset interface as ScoreCache so it
    can be passed directly to min_expected_guesses.  Results are never persisted.

    Hard-mode ERD results are valid only for the exact eligible-guess
    vocabulary (the word set surviving every accumulated Restriction) that
    produced them — not merely for a particular current_words snapshot, which
    can coincide across genuinely different guess histories (e.g. via undo).
    Entries are therefore namespaced by a fingerprint of that vocabulary
    (see fingerprint_vocabulary / set_scope): switching scope makes entries
    from other vocabularies invisible (no false hits) while leaving them
    intact, so a recurring vocabulary becomes reusable again for free —
    no explicit eviction needed.
    """

    def __init__(self):
        # (scope, subset_key, policy) -> (best_word, best_score, max_depth, solve_budget)
        self._data = {}
        self._scope = None
        self.read_hits = 0
        self.read_misses = 0
        self.write_count = 0

    @staticmethod
    def fingerprint_vocabulary(words):
        """Order-independent digest identifying an eligible-guess word set."""
        canonical = "\n".join(sorted(words))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def set_scope(self, fingerprint):
        """Switch the active vocabulary scope for subsequent read/write calls."""
        self._scope = fingerprint

    def read(self, subset_key, policy):
        result = self._data.get((self._scope, subset_key, policy))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result[:2]

    def read_with_depth(self, subset_key, policy):
        result = self._data.get((self._scope, subset_key, policy))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result

    def reset_read_counters(self):
        self.read_hits = 0
        self.read_misses = 0

    def write(self, subset_key, policy, best_word, best_score,
              max_depth=None, solve_budget=None):
        self._data[(self._scope, subset_key, policy)] = (
            best_word, best_score, max_depth, solve_budget)
        self.write_count += 1

    def close(self):
        pass
