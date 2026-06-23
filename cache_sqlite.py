"""SQLite-backed cache for Wordle scores and lookahead results."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("wordle")


class _LRUDict:
    """Fixed-capacity LRU cache backed by an OrderedDict.

    Evicts the least-recently-used entry when the capacity is reached.
    All operations are O(1).  When max_size is None the cache is unbounded
    (identical behaviour to a plain dict, but with the move-to-end overhead
    on every access — callers that want truly unbounded should pass None to
    opt out of the overhead).
    """

    def __init__(self, max_size=None):
        self._max = max_size
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if self._max is not None and len(self._data) > self._max:
            self._data.popitem(last=False)

    def __getitem__(self, key):
        self._data.move_to_end(key)
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def pop(self, key, *args):
        return self._data.pop(key, *args)

    def __len__(self):
        return len(self._data)


def _available_ram_bytes() -> int:
    """Return MemAvailable from /proc/meminfo, or 0 on any read error."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def mem_cache_limit(n_workers: int, ram_fraction: float = 0.4,
                    bytes_per_entry: int = 250) -> int:
    """Compute a per-worker _mem_cache entry cap from available RAM.

    Divides (ram_fraction * available_ram) evenly across n_workers.  Falls
    back to 500,000 entries if available RAM cannot be determined.
    bytes_per_entry is an estimate of the Python memory cost per cache entry
    (branch_key bytes blob + tuple + dict-node overhead).
    """
    available = _available_ram_bytes()
    if available <= 0 or n_workers <= 0:
        return 500_000
    return max(10_000, int(available * ram_fraction / n_workers / bytes_per_entry))


def _is_disk_io_error(exc):
    """True if exc is the transient 'disk I/O error' OperationalError that
    iCloud File Provider Storage raises when a sync pass holds the lock on
    the cache file or its WAL — see ScoreCache.checkpoint.
    """
    return "disk I/O error" in str(exc)


class ScoreCache:
    """Persists per-word scores and branch lookahead results.

    Tables:
      candidate_scores    — per-word scoring method results (level 1)
      branch_best_by_policy — the word a search policy judged best for a
                              branch, and the score that earned it that
                              judgment (levels 2+); the "by_policy" in the
                              table name carries the scoping that lets the
                              best_guess/best_score columns stay short —
                              "best" is only ever read alongside the policy
                              that decided it
      answer_list         — fingerprint of the answer word set

    All entries are keyed by answer_list_id so a different answer list
    produces a clean namespace without needing a new file.
    """

    def __init__(self, db_path, answer_words, timeout=30.0,
                 checkpoint_on_close=True, max_mem_entries=None):
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
        self.answer_list_id = self._ensure_answer_list()
        self.read_hits = 0
        self.read_misses = 0
        self.write_count = 0
        # In-memory mirror of branch_best_by_policy rows seen this session.
        # Branch results are write-once/exact, so a hit here is as good as
        # a SQLite hit but ~1000x cheaper — recursive ERD search re-reads the
        # same small branches millions of times across sibling branches.
        # max_mem_entries caps the cache size with LRU eviction so long-lived
        # worker processes do not consume unbounded memory.  None = unbounded.
        self._mem_cache = _LRUDict(max_size=max_mem_entries)
        # Session mirror of proven losses (positive hits only): (branch_key,
        # policy) -> largest budget at which the branch is proven a loss.  A
        # worker re-encounters the same inseparable residue under thousands of
        # candidates within one branch sweep; this turns each repeat into an
        # O(1) hit instead of a fresh exhaustive disproof.
        self._loss_mem_cache = _LRUDict(max_size=max_mem_entries)

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
            CREATE TABLE IF NOT EXISTS answer_list (
                answer_list_id TEXT PRIMARY KEY,
                answer_hash    TEXT NOT NULL,
                answer_count   INTEGER NOT NULL,
                created_at     INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS response_decomposition (
                guess          TEXT    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                patterns       BLOB    NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (guess, answer_list_id)
            )
        """)
        # Old databases may have either of two predecessor table structures:
        #   lookahead_result(subset_key, policy, universe_id,
        #                    best_word, best_entropy, updated_at)
        #   subgroup_pick(subset_key, policy, universe_id,
        #                 picked_word, picked_score, updated_at)
        # Both are intermediate schemas on the way to branch_best_by_policy.
        # Upgrade them in place so their rows survive as valid cache entries.
        tables = {row["name"] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "lookahead_result" in tables and "subgroup_pick" not in tables \
                and "subgroup_best_by_policy" not in tables \
                and "branch_best_by_policy" not in tables:
            self._conn.execute(
                "ALTER TABLE lookahead_result RENAME TO subgroup_best_by_policy")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN best_entropy TO best_score")
            self._conn.execute("DROP INDEX IF EXISTS idx_lookahead")
        tables = {row["name"] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "subgroup_pick" in tables and "subgroup_best_by_policy" not in tables \
                and "branch_best_by_policy" not in tables:
            self._conn.execute(
                "ALTER TABLE subgroup_pick RENAME TO subgroup_best_by_policy")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN picked_word TO best_word")
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy RENAME COLUMN picked_score TO best_score")
            self._conn.execute("DROP INDEX IF EXISTS idx_subgroup_pick")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS branch_best_by_policy (
                branch_key   BLOB NOT NULL,
                policy       TEXT NOT NULL,
                answer_list_id TEXT NOT NULL,
                best_guess   TEXT NOT NULL,
                best_score   REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (branch_key, policy, answer_list_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_best_by_policy
            ON branch_best_by_policy(answer_list_id, policy)
        """)
        # Covers MAX(updated_at) WHERE answer_list_id = ? — used by last_write_ts()
        # on every startup. Without this index, that query scans all 3M+ rows.
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_updated
            ON branch_best_by_policy(answer_list_id, updated_at)
        """)
        # Proven depth-limited losses: a branch with no winning strategy within
        # loss_budget guesses.  Distinct from branch_best_by_policy, whose
        # best_guess is NOT NULL — a loss has no best guess to record.  A loss
        # within b guesses is also a loss within any q <= b (fewer guesses can
        # only be harder), so a row is reusable for every query budget <=
        # loss_budget; loss_budget holds the largest budget at which the loss is
        # proven.  Lets the recurring inseparable residues of a hard branch be
        # proven once instead of re-swept under every candidate that produces them.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS branch_loss_by_policy (
                branch_key     BLOB    NOT NULL,
                policy         TEXT    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                loss_budget    INTEGER NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (branch_key, policy, answer_list_id)
            )
        """)
        # 'subset_blob' was renamed to 'subset_key' — same encoding, cleaner
        # name. Databases migrated from lookahead_result or subgroup_pick may
        # still carry the old column name (in subgroup_best_by_policy before
        # the rename_subgroup_to_branch migration below).
        cols = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(subgroup_best_by_policy)")}
        if cols and "subset_key" not in cols and "subset_blob" in cols:  # pragma: migration
            self._conn.execute(
                "ALTER TABLE subgroup_best_by_policy "
                "RENAME COLUMN subset_blob TO subset_key")
        # max_depth: worst-case line length of best_guess's strategy.  ERD is
        # now depth-limited ("expected remaining depth AND a guaranteed win
        # within budget"), so a cached entry is only reusable at a remaining
        # budget B when max_depth <= B.  Existing rows predate this and get
        # NULL — read as "depth unknown", hence never budget-safe, so they're
        # recomputed under the cap rather than trusted.  Nullable so the
        # column adds cleanly to a multi-GB file (metadata-only ALTER).
        for tbl in ('subgroup_best_by_policy', 'branch_best_by_policy'):
            cols = {row["name"] for row in
                    self._conn.execute(f"PRAGMA table_info({tbl})")}
            if cols and "max_depth" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {tbl} ADD COLUMN max_depth INTEGER")
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
        for tbl in ('subgroup_best_by_policy', 'branch_best_by_policy'):
            cols = {row["name"] for row in
                    self._conn.execute(f"PRAGMA table_info({tbl})")}
            if cols and "solve_budget" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {tbl} ADD COLUMN solve_budget INTEGER")
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
            for tbl in ('subgroup_best_by_policy', 'branch_best_by_policy'):
                t_exists = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,)).fetchone()
                if t_exists is None:
                    continue
                for old, new in (('erd_all', 'erd_words_unfiltered'),
                                 ('erd_answers', 'erd_answers_compliant')):
                    exists = self._conn.execute(
                        f"SELECT 1 FROM {tbl} WHERE policy = ? LIMIT 1",
                        (old,)
                    ).fetchone()
                    if exists is not None:
                        self._conn.execute(
                            f"UPDATE {tbl} SET policy = ? WHERE policy = ?",
                            (new, old))
            self._mark_migration_done('rename_erd_policies')
        # word_scores used to be keyed only by (word, method, universe_id) —
        # i.e. scoped to the whole answer set, so it could only ever cache
        # the very first guess of a game. Replace it with a subset-scoped
        # table (mirroring branch_best_by_policy) so any remaining-word position
        # that recurs gets its scores cached, not just the opening one.
        old_cols = {row["name"] for row in
                    self._conn.execute("PRAGMA table_info(word_scores)")}
        if old_cols and "subset_hash" not in old_cols:  # pragma: migration
            self._conn.execute("DROP TABLE word_scores")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_scores (
                subset_hash    TEXT    NOT NULL,
                word           TEXT    NOT NULL,
                method         TEXT    NOT NULL,
                score          REAL    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, method, answer_list_id, word)
            )
        """)
        # Legacy rows may carry the method key 'minimax' — an earlier name for
        # the MAX_GROUP_SIZE scoring method that named the search strategy
        # rather than the metric, making rows uninterpretable without external
        # context.  Rewrite them to 'max_group_size' so the database is
        # self-describing. Checked
        # via existence-first LIMIT 1 (see _purge_legacy_rows) so a table with
        # no such rows — the steady state once this has run once — costs only
        # a single indexed-or-not probe, not a full scan, on every connection.
        if not self._is_migration_done('rename_method_minimax'):
            for tbl in ('word_scores', 'candidate_scores'):
                t_exists = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,)).fetchone()
                if t_exists is None:
                    continue
                stale_method = self._conn.execute(
                    f"SELECT 1 FROM {tbl} WHERE method = 'minimax' LIMIT 1"
                ).fetchone()
                if stale_method is not None:
                    self._conn.execute(
                        f"UPDATE {tbl} SET method = 'max_group_size'"
                        " WHERE method = 'minimax'")
            self._mark_migration_done('rename_method_minimax')
        # Rename subgroup_best_by_policy -> branch_best_by_policy, and columns:
        #   subset_key -> branch_key
        #   best_word  -> best_guess
        #   universe_id -> answer_list_id
        if not self._is_migration_done('rename_subgroup_to_branch'):  # pragma: migration
            tables = {row["name"] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "subgroup_best_by_policy" in tables:
                # The CREATE TABLE IF NOT EXISTS above may have created an empty
                # branch_best_by_policy shell; drop it before the rename so we
                # don't get a "table already exists" conflict.
                self._conn.execute("DROP TABLE IF EXISTS branch_best_by_policy")
                self._conn.execute("DROP INDEX IF EXISTS idx_branch_best_by_policy")
                self._conn.execute("DROP INDEX IF EXISTS idx_branch_updated")
                self._conn.execute(
                    "ALTER TABLE subgroup_best_by_policy RENAME TO branch_best_by_policy")
            cols = {row["name"] for row in
                    self._conn.execute("PRAGMA table_info(branch_best_by_policy)")}
            if cols and "subset_key" in cols:
                self._conn.execute(
                    "ALTER TABLE branch_best_by_policy "
                    "RENAME COLUMN subset_key TO branch_key")
            if cols and "best_word" in cols:
                self._conn.execute(
                    "ALTER TABLE branch_best_by_policy "
                    "RENAME COLUMN best_word TO best_guess")
            self._mark_migration_done('rename_subgroup_to_branch')
        # Rename word_scores -> candidate_scores
        if not self._is_migration_done('rename_word_scores'):  # pragma: migration
            tables = {row["name"] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "word_scores" in tables:
                # Drop the empty candidate_scores shell created by CREATE TABLE
                # IF NOT EXISTS before renaming the old table into its place.
                self._conn.execute("DROP TABLE IF EXISTS candidate_scores")
                self._conn.execute("ALTER TABLE word_scores RENAME TO candidate_scores")
            self._mark_migration_done('rename_word_scores')
        # Rename universe -> answer_list and universe_id -> answer_list_id
        # in all tables that carry it.
        if not self._is_migration_done('rename_universe_to_answer_list'):  # pragma: migration
            tables = {row["name"] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "universe" in tables:
                # Drop the empty answer_list shell from CREATE TABLE IF NOT EXISTS.
                self._conn.execute("DROP TABLE IF EXISTS answer_list")
                self._conn.execute("ALTER TABLE universe RENAME TO answer_list")
            for tbl, old_col in [
                ('answer_list',            'universe_id'),
                ('response_decomposition', 'universe_id'),
                ('branch_best_by_policy',  'universe_id'),
                ('candidate_scores',       'universe_id'),
            ]:
                cols = {row["name"] for row in
                        self._conn.execute(f"PRAGMA table_info({tbl})")}
                if cols and old_col in cols:
                    self._conn.execute(
                        f"ALTER TABLE {tbl} RENAME COLUMN {old_col} TO answer_list_id")
            self._mark_migration_done('rename_universe_to_answer_list')
        # All valid 5-letter words are ASCII, so a null byte identifies the
        # old null-separated branch-key encoding.
        self._purge_legacy_rows("instr(branch_key, char(0)) > 0", (),
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
        """One-time cleanup of stale branch_best_by_policy rows.

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
            f"SELECT 1 FROM branch_best_by_policy WHERE {where} LIMIT 1", params
        ).fetchone()
        if exists is not None:
            self._conn.execute(
                f"DELETE FROM branch_best_by_policy WHERE {where}", params)
        if migration_name:  # pragma: migration
            self._mark_migration_done(migration_name)

    def _ensure_answer_list(self):
        canonical = "\n".join(self.answer_words)
        answer_list_id = hashlib.sha256(canonical.encode()).hexdigest()
        now = int(time.time())
        self._conn.execute("""
            INSERT OR IGNORE INTO answer_list
                (answer_list_id, answer_hash, answer_count, created_at)
            VALUES (?, ?, ?, ?)
        """, (answer_list_id, answer_list_id, len(self.answer_words), now))
        return answer_list_id

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
    # Branch lookahead cache (levels 2+)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_subset(words):
        """Canonical key for a set of words: sorted, concatenated, no separator.

        All Wordle words are exactly 5 ASCII characters, so a key of length 5N
        encodes exactly N words recoverable by slicing at fixed 5-byte offsets.
        """
        return "".join(sorted(words)).encode("utf-8")

    def read(self, branch_key, policy):
        """Return (best_guess, best_score) or None on cache miss.

        best_guess is whichever word this policy's search judged best for
        this branch — judged by that policy's own metric, not some
        universal notion of "best". best_score is that metric's value for
        best_guess, and its meaning is policy-dependent: an entropy in bits
        (higher is better) for lookahead policies ('full'/'hard'), or an
        expected-remaining-guesses cost (lower is better) for ERD policies
        ('erd_words_unfiltered'/'erd_answers_compliant'). Callers that care
        about the number must already know which policy they asked for — the
        table name (branch_best_by_policy) and its policy column carry that
        scoping, so the columns themselves can stay "best_guess"/"best_score"
        without re-litigating it.
        """
        cached = self._mem_cache.get((branch_key, policy))
        if cached is not None:
            self.read_hits += 1
            return cached[:2]
        full = self._read_full(branch_key, policy)
        if full is None:
            return None
        return full[:2]

    def read_with_depth(self, branch_key, policy):
        """Like read(), but returns (best_guess, best_score, max_depth, solve_budget).

        max_depth is the worst-case line length of best_guess's strategy (None
        for legacy rows).  solve_budget is the reuse-range marker (see schema):
        None = untainted, reusable at any budget >= max_depth; an int b =
        tainted, reusable only at remaining budget == b.  A budget-aware caller
        must apply that rule; a legacy row (max_depth None) is never reusable.
        """
        cached = self._mem_cache.get((branch_key, policy))
        if cached is not None:
            self.read_hits += 1
            return cached
        return self._read_full(branch_key, policy)

    def _read_full(self, branch_key, policy):
        row = self._conn.execute("""
            SELECT best_guess, best_score, max_depth, solve_budget
            FROM branch_best_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id)).fetchone()
        if row is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        result = (row["best_guess"], row["best_score"],
                  row["max_depth"], row["solve_budget"])
        self._mem_cache[(branch_key, policy)] = result
        return result

    def reset_read_counters(self):
        self.read_hits = 0
        self.read_misses = 0

    def write(self, branch_key, policy, best_guess, best_score,
              max_depth=None, solve_budget=None):
        """Store the word a policy's search judged best for a branch, its
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
        best_guess/best_score are still recorded in _mem_cache so this run's
        recursion keeps the memoization benefit even when the on-disk write
        fails; the row is simply recomputed on a later run.
        """
        now = int(time.time())
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO branch_best_by_policy
                    (branch_key, policy, answer_list_id,
                     best_guess, best_score, updated_at, max_depth, solve_budget)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (branch_key, policy, self.answer_list_id,
                  best_guess, best_score, now, max_depth, solve_budget))
            self.write_count += 1
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write(%s, %s, %.3f) failed: %s",
                            policy, best_guess, best_score, exc)
        self._mem_cache[(branch_key, policy)] = (
            best_guess, best_score, max_depth, solve_budget)

    def read_loss(self, branch_key, policy):
        """Largest budget at which `branch_key` is proven a loss, or None.

        A return of b means "no winning strategy within b guesses"; the caller
        treats any query budget q <= b as a loss.  Positive hits are mirrored in
        a session cache; misses fall through to SQLite (so a loss another worker
        just proved is seen) — a stale miss only costs a recompute, never a wrong
        answer.
        """
        cached = self._loss_mem_cache.get((branch_key, policy))
        if cached is not None:
            return cached or None        # 0 = "no loss known" sentinel
        row = self._conn.execute("""
            SELECT loss_budget
            FROM branch_loss_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id)).fetchone()
        # Cache the miss (sentinel 0) too, so a no-loss branch revisited millions
        # of times across sibling searches is not re-queried.  A later loss by a
        # peer is missed until eviction — sound, since that only forgoes reuse.
        value = row["loss_budget"] if row is not None else 0
        self._loss_mem_cache[(branch_key, policy)] = value
        return value or None

    def write_loss(self, branch_key, policy, budget):
        """Record `branch_key` as proven unsolvable within `budget` guesses,
        keeping the largest budget seen (the widest reuse range).  Disk I/O
        errors are logged and swallowed like write() — the session mirror still
        carries the verdict for the rest of this run.
        """
        prior = self._loss_mem_cache.get((branch_key, policy))
        if prior is not None and prior >= budget:
            self._loss_mem_cache[(branch_key, policy)] = prior  # refresh LRU
            return
        now = int(time.time())
        try:
            self._conn.execute("""
                INSERT INTO branch_loss_by_policy
                    (branch_key, policy, answer_list_id, loss_budget, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (branch_key, policy, answer_list_id)
                DO UPDATE SET loss_budget = MAX(loss_budget, excluded.loss_budget),
                              updated_at  = excluded.updated_at
            """, (branch_key, policy, self.answer_list_id, budget, now))
            self.write_count += 1
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_loss(%s, budget=%d) failed: %s",
                           policy, budget, exc)
        self._loss_mem_cache[(branch_key, policy)] = budget

    def read_detail(self, branch_key, policy):
        """Like read(), but also returns the unix timestamp of the last write.

        Returns (best_guess, best_score, updated_at) or None on a miss. Used
        by cache spot-checks (e.g. wordle.py's verify command) to show when
        a cached entry was written, alongside the per-prompt timestamps.
        """
        row = self._conn.execute("""
            SELECT best_guess, best_score, updated_at
            FROM branch_best_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id)).fetchone()
        if row is None:
            return None
        return (row["best_guess"], row["best_score"], row["updated_at"])

    def delete(self, branch_key, policy):
        """Remove a cached branch result so it gets recomputed.

        For invalidating an entry a spot-check has found to be inconsistent
        with its own cached subtree.
        """
        self._conn.execute("""
            DELETE FROM branch_best_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id))
        self._mem_cache.pop((branch_key, policy), None)

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
            WHERE guess = ? AND answer_list_id = ?
        """, (guess, self.answer_list_id)).fetchone()
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
                    (guess, answer_list_id, patterns, updated_at)
                VALUES (?, ?, ?, ?)
            """, (guess, self.answer_list_id, patterns, now))
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_decomposition(%s) failed: %s", guess, exc)

    # ------------------------------------------------------------------
    # Candidate score cache (level 1, all ScoringMethods)
    # ------------------------------------------------------------------

    @staticmethod
    def _subset_hash(branch_key):
        """Compact, fixed-size key for a (potentially large) branch blob."""
        return hashlib.sha256(branch_key).hexdigest()

    def has_scores(self, branch_key, method):
        """Return True if any scores are cached for this branch/method/universe."""
        subset_hash = self._subset_hash(branch_key)
        return self._conn.execute("""
            SELECT 1 FROM candidate_scores
            WHERE subset_hash = ? AND method = ? AND answer_list_id = ?
            LIMIT 1
        """, (subset_hash, method, self.answer_list_id)).fetchone() is not None

    def read_scores(self, branch_key, method):
        """Return list of (word, score) for this branch/method/universe, or None if empty."""
        subset_hash = self._subset_hash(branch_key)
        rows = self._conn.execute("""
            SELECT word, score FROM candidate_scores
            WHERE subset_hash = ? AND method = ? AND answer_list_id = ?
        """, (subset_hash, method, self.answer_list_id)).fetchall()
        if not rows:
            return None
        return [(r["word"], r["score"]) for r in rows]

    def write_scores(self, branch_key, scores, method):
        """Store list of (word, score) tuples for this branch/method/universe.

        Swallows a transient 'disk I/O error' the same way write() does —
        see its docstring.
        """
        subset_hash = self._subset_hash(branch_key)
        now = int(time.time())
        try:
            self._conn.execute("BEGIN")
            self._conn.executemany("""
                INSERT OR REPLACE INTO candidate_scores
                    (subset_hash, word, method, score, answer_list_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(subset_hash, w, method, s, self.answer_list_id, now)
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
            FROM branch_best_by_policy WHERE answer_list_id = ?
        """, (self.answer_list_id,)).fetchone()
        return row["m"] if row else None

    def stats(self):
        """Return (branch_best_rows, candidate_score_rows, decomposition_rows, last_updated_ts)."""
        sp = self._conn.execute("""
            SELECT COUNT(*) AS c, MAX(updated_at) AS m
            FROM branch_best_by_policy WHERE answer_list_id = ?
        """, (self.answer_list_id,)).fetchone()
        ws = self._conn.execute("""
            SELECT COUNT(*) AS c FROM candidate_scores WHERE answer_list_id = ?
        """, (self.answer_list_id,)).fetchone()
        rd = self._conn.execute("""
            SELECT COUNT(*) AS c FROM response_decomposition WHERE answer_list_id = ?
        """, (self.answer_list_id,)).fetchone()
        return sp["c"] or 0, ws["c"] or 0, rd["c"] or 0, sp["m"]


class MemoryScoreCache:
    """Transient in-memory ERD cache for path-dependent computations (hard mode).

    Implements the same read/write/encode_subset interface as ScoreCache so it
    can be passed directly to min_expected_guesses.  Results are never persisted.

    Hard-mode ERD results are valid only for the exact eligible-guess
    vocabulary (the word set surviving every accumulated Restriction) that
    produced them — not merely for a particular branch_words snapshot, which
    can coincide across genuinely different guess histories (e.g. via undo).
    Entries are therefore namespaced by a fingerprint of that vocabulary
    (see fingerprint_vocabulary / set_scope): switching scope makes entries
    from other vocabularies invisible (no false hits) while leaving them
    intact, so a recurring vocabulary becomes reusable again for free —
    no explicit eviction needed.
    """

    def __init__(self):
        # (scope, branch_key, policy) -> (best_guess, best_score, max_depth, solve_budget)
        self._data = {}
        # (scope, branch_key, policy) -> largest budget proven a loss
        self._losses = {}
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

    def read(self, branch_key, policy):
        result = self._data.get((self._scope, branch_key, policy))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result[:2]

    def read_with_depth(self, branch_key, policy):
        result = self._data.get((self._scope, branch_key, policy))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result

    def reset_read_counters(self):
        self.read_hits = 0
        self.read_misses = 0

    def write(self, branch_key, policy, best_guess, best_score,
              max_depth=None, solve_budget=None):
        self._data[(self._scope, branch_key, policy)] = (
            best_guess, best_score, max_depth, solve_budget)
        self.write_count += 1

    def read_loss(self, branch_key, policy):
        return self._losses.get((self._scope, branch_key, policy))

    def write_loss(self, branch_key, policy, budget):
        key = (self._scope, branch_key, policy)
        prior = self._losses.get(key)
        if prior is None or budget > prior:
            self._losses[key] = budget
        self.write_count += 1

    def close(self):
        pass
