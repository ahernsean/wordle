"""SQLite-backed cache for Wordle scores and lookahead results."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("wordle")


def branch_reference(branch_key: bytes) -> str:
    """Return the stable handle for one encoded branch answer set."""
    return hashlib.sha1(bytes(branch_key)).hexdigest()[:12]


# Two searches reaching the same optimum sum the same terms in the same order,
# so they agree exactly; the tolerance only absorbs a value that reached the
# cache by some other route.
EXACT_SCORE_TOLERANCE = 1e-9


def exact_results_agree(stored_score, stored_max_depth,
                        incoming_score, incoming_max_depth) -> bool:
    """Whether two exact results for one scope are the same certificate.

    Equal cost is not enough.  max_depth is ancestor-visible — a parent folds
    a child's worst case into its own — so two equal-cost strategies with
    different worst cases are different certificates, and a parent folded from
    one does not describe a subtree the other supports.

    import_cache expresses this same rule in SQL, over whole tables at once;
    test_the_sql_equivalence_rule_matches_the_python_one keeps the two in step.
    """
    if abs(stored_score - incoming_score) > EXACT_SCORE_TOLERANCE:
        return False
    return stored_max_depth == incoming_max_depth


def _branch_facts_by_key(rows):
    """Group exact branch rows into (unrestricted_row, {solve_budget: row}).

    Both branch tables carry the same columns, so one pass over their union
    separates a branch's unrestricted result from its budget-specific ones
    without the caller having to know which table a row came from.
    """
    facts = {}
    for row in rows:
        key = bytes(row["branch_key"])
        canonical, by_budget = facts.get(key, (None, None))
        if by_budget is None:
            by_budget = {}
        if row["solve_budget"] is None:
            canonical = row
        else:
            by_budget[row["solve_budget"]] = row
        facts[key] = (canonical, by_budget)
    return facts


class CacheWriteConflict(Exception):
    """Two exact results disagree for one branch at one budget scope.

    Within a scope the optimum is a single number, so a second exact write
    naming a different one means the two searches cannot both be right.
    Recording either would invalidate whichever ancestors folded the other,
    which is the failure this schema exists to prevent — so the write is
    refused instead.
    """


def answer_list_id(answer_words) -> str:
    """Return the namespace key for one answer word list."""
    return hashlib.sha256("\n".join(answer_words).encode()).hexdigest()


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

    def pop_matching(self, predicate):
        """Drop every entry whose key satisfies predicate."""
        for key in [key for key in self._data if predicate(key)]:
            del self._data[key]

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
      candidate_erd_by_policy — a candidate's own ERD at a branch, folded
                              from its response groups' branch_best_by_policy
                              rows once every one of them is exact; distinct
                              from candidate_scores, whose methods are cheap
                              pre-solve heuristics rather than exact results
      answer_list         — fingerprint of the answer word set

    All entries are keyed by answer_list_id so a different answer list
    produces a clean namespace without needing a new file.
    """

    def __init__(self, db_path, answer_words, timeout=30.0,
                 checkpoint_on_close=True, max_mem_entries=None,
                 read_only=False):
        self.db_path = Path(db_path)
        self.answer_words = list(answer_words)
        self.read_only = read_only
        # A read-only cache never checkpoints: TRUNCATE is itself a write.
        self.checkpoint_on_close = checkpoint_on_close and not read_only
        if read_only:
            # An inspection pass over a live cache must leave no trace, not
            # even the schema migration and answer-list row an ordinary open
            # writes.  SQLite enforces that for us: mode=ro rejects every
            # write, so a caller that reaches for one gets an error instead of
            # a silently swallowed no-op.  It also refuses to create the file,
            # which is what keeps a mistyped path from reading as an empty
            # database.
            self._conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True,
                timeout=timeout, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
            self._supply_missing_budget_table()
            self.answer_list_id = answer_list_id(self.answer_words)
        else:
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
        # Exact results re-derived and found already stored.
        self.redundant_write_count = 0
        # ...of which the stored worst case differed from the one just
        # computed, so the caller adopted the stored certificate.
        self.adopted_depth_count = 0
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

    def __del__(self):
        conn = getattr(self, '_conn', None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.ProgrammingError:
                # conn was created on a different thread than the one
                # finalizing it; SQLite connections are thread-affine, so
                # closing here is impossible.
                pass

    def _supply_missing_budget_table(self):
        """Stand in for branch_best_by_policy_and_budget on a pre-split cache.

        A cache written before the split has no such table — the migration
        that creates it runs in _ensure_schema, which a read-only open
        deliberately skips.  Every reader that spans both scopes would then
        fail on a database it is meant to be able to inspect, and inspecting
        an un-migrated cache is the whole point of opening one read-only.

        A TEMP view supplies it.  Temp objects live outside the file, so this
        writes nothing, and SQLite resolves an unqualified name against temp
        before main, so the readers need no special case.  Empty is the honest
        answer: a pre-split cache keeps its budget-specific results in the
        canonical table with solve_budget set, and each row still names the
        scope it belongs to.
        """
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ('branch_best_by_policy_and_budget',)).fetchone()
        if exists:
            return
        self._conn.execute("""
            CREATE TEMP VIEW branch_best_by_policy_and_budget AS
            SELECT branch_key, branch_reference, policy, answer_list_id,
                   solve_budget, best_guess, best_score, updated_at, max_depth
            FROM branch_best_by_policy WHERE 0
        """)

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
                branch_reference TEXT,
                policy       TEXT NOT NULL,
                answer_list_id TEXT NOT NULL,
                best_guess   TEXT NOT NULL,
                best_score   REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (branch_key, policy, answer_list_id)
            )
        """)
        # A branch has two kinds of exact result, and they are different facts:
        # the unrestricted optimum, and the optimum among strategies feasible
        # at one remaining-depth budget.  Both can be right and differ.  They
        # live in separate tables because one row cannot hold both: a shared
        # key makes either write destroy the other, after ancestors may already
        # have folded the value it displaced, and nothing records which one
        # they folded.
        #
        # branch_best_by_policy holds only the unrestricted optima, so its
        # solve_budget column is NULL on every row it now accepts.  The column
        # stays because dropping it would rebuild a multi-GB table to no end,
        # and legacy rows are read through it.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS branch_best_by_policy_and_budget (
                branch_key   BLOB NOT NULL,
                branch_reference TEXT,
                policy       TEXT NOT NULL,
                answer_list_id TEXT NOT NULL,
                solve_budget INTEGER NOT NULL,
                best_guess   TEXT NOT NULL,
                best_score   REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                max_depth    INTEGER,
                PRIMARY KEY (branch_key, policy, answer_list_id, solve_budget)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_best_by_policy_and_budget
            ON branch_best_by_policy_and_budget(answer_list_id, policy)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_budget_updated
            ON branch_best_by_policy_and_budget(answer_list_id, updated_at)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_budget_reference
            ON branch_best_by_policy_and_budget(branch_reference)
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
                branch_reference TEXT,
                policy         TEXT    NOT NULL,
                answer_list_id TEXT    NOT NULL,
                loss_budget    INTEGER NOT NULL,
                updated_at     INTEGER NOT NULL,
                PRIMARY KEY (branch_key, policy, answer_list_id)
            )
        """)
        # A candidate's own ERD at a branch: the fold of every one of its
        # response groups, once every group is itself an exact
        # branch_best_by_policy row.  Keyed by subset_hash rather than the raw
        # branch_key blob (see candidate_scores._subset_hash) because one
        # branch — most often the root, the whole answer list — accumulates a
        # row per solved candidate, and the root's own key is the largest
        # blob in the schema.  response_group_count lets a reader detect a
        # stale row (a changed vocabulary reshapes a candidate's groups)
        # without a second query.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_erd_by_policy (
                subset_hash          TEXT    NOT NULL,
                candidate_word       TEXT    NOT NULL,
                policy               TEXT    NOT NULL,
                answer_list_id       TEXT    NOT NULL,
                erd                  REAL    NOT NULL,
                max_remaining_depth  INTEGER NOT NULL,
                response_group_count INTEGER NOT NULL,
                updated_at           INTEGER NOT NULL,
                PRIMARY KEY (subset_hash, candidate_word, policy, answer_list_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_erd_by_policy
            ON candidate_erd_by_policy(answer_list_id, policy)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_source_summaries (
                source_word TEXT NOT NULL, policy TEXT NOT NULL,
                answer_list_id TEXT NOT NULL, completed_at INTEGER NOT NULL,
                elapsed_millis INTEGER, worker_millis INTEGER NOT NULL,
                telemetry_epochs TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (source_word, policy, answer_list_id)
            )
        """)
        summary_columns = {row["name"] for row in self._conn.execute(
            "PRAGMA table_info(completed_source_summaries)")}
        if "telemetry_epochs" not in summary_columns:
            self._conn.execute(
                "ALTER TABLE completed_source_summaries "
                "ADD COLUMN telemetry_epochs TEXT NOT NULL DEFAULT ''")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS root_response_group_summaries (
                source_word TEXT NOT NULL, response_pattern TEXT NOT NULL,
                policy TEXT NOT NULL, answer_list_id TEXT NOT NULL,
                branch_count INTEGER NOT NULL, search_node_count INTEGER NOT NULL,
                worker_millis INTEGER NOT NULL, first_created_at INTEGER,
                last_finalized_at INTEGER, telemetry_epochs TEXT NOT NULL,
                PRIMARY KEY (source_word, response_pattern, policy, answer_list_id)
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
        if not self._is_migration_done('add_branch_references'):
            for table_name in ('branch_best_by_policy', 'branch_loss_by_policy'):
                columns = {row["name"] for row in self._conn.execute(
                    f"PRAGMA table_info({table_name})")}
                if "branch_reference" not in columns:
                    self._conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN branch_reference TEXT")
                rows = self._conn.execute(
                    f"SELECT rowid, branch_key FROM {table_name} "
                    "WHERE branch_reference IS NULL"
                ).fetchall()
                self._conn.executemany(
                    f"UPDATE {table_name} SET branch_reference = ? WHERE rowid = ?",
                    [(branch_reference(row["branch_key"]), row["rowid"])
                     for row in rows],
                )
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_reference "
                    f"ON {table_name}(branch_reference)"
                )
            self._mark_migration_done('add_branch_references')
        # Budget-specific results used to share the canonical table's key, so
        # a branch's two facts overwrote one another.  Move them to the table
        # that can hold both.  candidate_erd_by_policy goes with them: every
        # row memoises a fold over branch rows under the old identity, and is
        # trusted on a matching response-group count alone, so none of them can
        # be relied on across the split.  Each is re-earned by one fold.
        if not self._is_migration_done('split_budget_specific_branch_results'):  # pragma: migration
            self._conn.execute("""
                INSERT OR IGNORE INTO branch_best_by_policy_and_budget
                    (branch_key, branch_reference, policy, answer_list_id,
                     solve_budget, best_guess, best_score, updated_at, max_depth)
                SELECT branch_key, branch_reference, policy, answer_list_id,
                       solve_budget, best_guess, best_score, updated_at, max_depth
                FROM branch_best_by_policy
                WHERE solve_budget IS NOT NULL
            """)
            self._conn.execute(
                "DELETE FROM branch_best_by_policy WHERE solve_budget IS NOT NULL")
            self._conn.execute("DELETE FROM candidate_erd_by_policy")
            self._mark_migration_done('split_budget_specific_branch_results')

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
        list_id = answer_list_id(self.answer_words)
        now = int(time.time())
        self._conn.execute("""
            INSERT OR IGNORE INTO answer_list
                (answer_list_id, answer_hash, answer_count, created_at)
            VALUES (?, ?, ?, ?)
        """, (list_id, list_id, len(self.answer_words), now))
        return list_id

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
        cached = self._mem_cache.get((branch_key, policy, None))
        if cached is not None:
            self.read_hits += 1
            return cached[:2]
        full = self._read_full(branch_key, policy)
        if full is None:
            return None
        return full[:2]

    def read_with_depth(self, branch_key, policy):
        """Like read(), but returns (best_guess, best_score, max_depth, solve_budget).

        Answers for the branch's *unrestricted* result only — the optimum over
        all strategies, reusable at any budget its own max_depth can meet.
        solve_budget is therefore None on every row this returns except a
        legacy one, and a legacy row (max_depth None) is never budget-reusable.
        A search under a cap wants read_for_budget, which consults the
        budget-specific results too.
        """
        cached = self._mem_cache.get((branch_key, policy, None))
        if cached is not None:
            self.read_hits += 1
            return cached
        return self._read_full(branch_key, policy)

    def _read_stored_row(self, branch_key, policy, solve_budget):
        """One scope's row straight from SQLite, with no session mirror.

        The mirror can hold a value this connection read before another wrote,
        so anything reconciling against what is *durably* stored — the write
        path's conflict check — has to come here rather than through the
        cached reads.
        """
        if solve_budget is None:
            row = self._conn.execute("""
                SELECT best_guess, best_score, max_depth, solve_budget
                FROM branch_best_by_policy
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
            """, (branch_key, policy, self.answer_list_id)).fetchone()
        else:
            row = self._conn.execute("""
                SELECT best_guess, best_score, max_depth, solve_budget
                FROM branch_best_by_policy_and_budget
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
                  AND solve_budget = ?
            """, (branch_key, policy, self.answer_list_id,
                  solve_budget)).fetchone()
        if row is None:
            return None
        return (row["best_guess"], row["best_score"],
                row["max_depth"], row["solve_budget"])

    def _read_full(self, branch_key, policy):
        result = self._read_stored_row(branch_key, policy, None)
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        self._mem_cache[(branch_key, policy, None)] = result
        return result

    def _read_full_at_budget(self, branch_key, policy, budget):
        """The exact result stored for `branch_key` at remaining budget `budget`.

        A budget-specific row is valid only at the budget it was solved under,
        so it is looked up by that budget rather than filtered after the fact.
        """
        cached = self._mem_cache.get((branch_key, policy, budget))
        if cached is not None:
            self.read_hits += 1
            return cached
        result = self._read_stored_row(branch_key, policy, budget)
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        self._mem_cache[(branch_key, policy, budget)] = result
        return result

    def read_for_budget(self, branch_key, policy, budget):
        """The entry a search at `budget` should reuse, or None.

        The unrestricted result wins whenever its strategy fits: it is
        globally optimal, so it is also optimal within any budget its own
        worst case can meet.  Only when it does not fit is the budget-specific
        result consulted, and only the one solved at exactly this budget — a
        row from another budget is optimal against a different set of feasible
        strategies and is not an exact hit here.  An unlimited search
        (budget None) reads the unrestricted table alone.

        Returns the same (best_guess, best_score, max_depth, solve_budget)
        tuple the plain reads return, so `wordle_engine._cache_reuse` remains
        the one place the reuse rule is stated.
        """
        canonical = self.read_with_depth(branch_key, policy)
        if budget is None:
            return canonical
        if canonical is not None:
            max_remaining_depth = canonical[2]
            if max_remaining_depth is not None and max_remaining_depth <= budget:
                return canonical
        return self._read_full_at_budget(branch_key, policy, budget)

    def reset_read_counters(self):
        self.read_hits = 0
        self.read_misses = 0

    def write(self, branch_key, policy, best_guess, best_score,
              max_depth=None, solve_budget=None):
        """Store the word a policy's search judged best for a branch, its
        score, and (for depth-limited ERD) the worst-case line length of that
        strategy plus its reuse-range marker.  max_depth=None marks a
        legacy/unbudgeted write.  solve_budget routes the result: None is the
        unrestricted optimum and goes to branch_best_by_policy; an int is the
        optimum under that cap and goes to branch_best_by_policy_and_budget.
        The two are separate facts and neither displaces the other.

        A result already stored for the same branch at the same scope is kept
        rather than replaced, and **returned**: the caller must adopt it before
        folding anything, because what a solver hands its parent has to be what
        the cache durably holds.  Equal-cost strategies can differ in
        max_depth, which is ancestor-visible, so a caller that kept its own
        worst case would fold a parent the stored child does not support —
        the inconsistent ancestry this schema exists to prevent, reached
        without any overwrite.

        Returns the durable (best_guess, best_score, max_depth, solve_budget).
        A second result that disagrees on the *cost* cannot be reconciled by
        adoption — both claim to be the optimum, so one of them is wrong — and
        raises CacheWriteConflict.

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
        entry = (best_guess, best_score, max_depth, solve_budget)
        if solve_budget is None:
            table = 'branch_best_by_policy'
            conflict_target = 'branch_key, policy, answer_list_id'
        else:
            table = 'branch_best_by_policy_and_budget'
            conflict_target = 'branch_key, policy, answer_list_id, solve_budget'
        try:
            # Creating the row IS the check.  A read followed by an insert
            # leaves a window two workers both pass through, and the second
            # insert would then displace a result an ancestor may already have
            # folded -- with neither writer noticing.  DO NOTHING makes the
            # uniqueness constraint decide it: exactly one writer creates the
            # row, and every other reconciles against what that one stored.
            before = self._conn.total_changes
            self._conn.execute(f"""
                INSERT INTO {table}
                    (branch_key, branch_reference, policy, answer_list_id,
                     best_guess, best_score, updated_at, max_depth, solve_budget)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT({conflict_target}) DO NOTHING
            """, (branch_key, branch_reference(branch_key), policy, self.answer_list_id,
                  best_guess, best_score, now, max_depth, solve_budget))
            if self._conn.total_changes > before:
                self.write_count += 1
            else:
                # Someone already holds this scope.  Read what they stored --
                # from SQLite, not the mirror, which may predate their write.
                stored = self._read_stored_row(branch_key, policy, solve_budget)
                if stored is None:
                    # Deleted between the insert and this read.  Nothing is
                    # stored to reconcile against and nothing was written, so
                    # do not mirror a value the cache does not hold; the
                    # caller's own result is the only one in play.
                    self._mem_cache.pop((branch_key, policy, solve_budget), None)
                    return entry
                if abs(stored[1] - best_score) > EXACT_SCORE_TOLERANCE:
                    logger.error(
                        "conflicting exact results for %s at policy=%s "
                        "budget=%s: stored %s/%.6f, incoming %s/%.6f",
                        branch_reference(branch_key), policy, solve_budget,
                        stored[0], stored[1], best_guess, best_score)
                    raise CacheWriteConflict(
                        f"{branch_reference(branch_key)} at policy={policy} "
                        f"budget={solve_budget}: stored {stored[1]!r}, "
                        f"incoming {best_score!r}")
                # The same optimum, reached again.  The stored row stands and
                # the caller adopts it: an ancestor may already have folded its
                # max_depth, and equal cost does not make two worst cases
                # interchangeable.
                self.redundant_write_count += 1
                if stored[2] != max_depth:
                    self.adopted_depth_count += 1
                self._mem_cache[(branch_key, policy, solve_budget)] = stored
                return stored
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write(%s, %s, %.3f) failed: %s",
                            policy, best_guess, best_score, exc)
        self._mem_cache[(branch_key, policy, solve_budget)] = entry
        return entry

    def read_loss(self, branch_key, policy, refresh=False):
        """Largest budget at which `branch_key` is proven a loss, or None.

        A return of b means "no winning strategy within b guesses"; the caller
        treats any query budget q <= b as a loss.  Positive hits are mirrored in
        a session cache.  Pass refresh=True while polling a cooperative branch
        so a peer's newly published loss replaces a cached miss.
        """
        cached = self._loss_mem_cache.get((branch_key, policy))
        if cached is not None and not refresh:
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
                    (branch_key, branch_reference, policy, answer_list_id,
                     loss_budget, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (branch_key, policy, answer_list_id)
                DO UPDATE SET loss_budget = MAX(loss_budget, excluded.loss_budget),
                              updated_at  = excluded.updated_at
            """, (branch_key, branch_reference(branch_key), policy,
                  self.answer_list_id, budget, now))
            self.write_count += 1
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_loss(%s, budget=%d) failed: %s",
                           policy, budget, exc)
        self._loss_mem_cache[(branch_key, policy)] = budget

    def delete_loss(self, branch_key, policy):
        """Remove a proven-loss row so the disproof gets re-established.

        For invalidating a loss a verification pass has found suspect.  Also
        drops the session mirror's entry (verdict or no-loss sentinel alike)
        so the next read_loss falls through to SQLite instead of resurrecting
        the deleted verdict from memory.
        """
        self._conn.execute("""
            DELETE FROM branch_loss_by_policy
            WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
        """, (branch_key, policy, self.answer_list_id))
        self._loss_mem_cache.pop((branch_key, policy), None)

    def branch_keys_for_reference_prefix(self, digest_prefix):
        """Return distinct branch keys whose durable handles match a prefix."""
        upper_bound = digest_prefix[:-1] + chr(ord(digest_prefix[-1]) + 1)
        rows = self._conn.execute(
            """SELECT branch_key FROM branch_best_by_policy
               WHERE branch_reference >= ? AND branch_reference < ?
               UNION
               SELECT branch_key FROM branch_best_by_policy_and_budget
               WHERE branch_reference >= ? AND branch_reference < ?
               UNION
               SELECT branch_key FROM branch_loss_by_policy
               WHERE branch_reference >= ? AND branch_reference < ?""",
            (digest_prefix, upper_bound, digest_prefix, upper_bound,
             digest_prefix, upper_bound),
        ).fetchall()
        return [bytes(row["branch_key"]) for row in rows]

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

    def repair_max_depth(self, branch_key, policy, max_depth, solve_budget=None):
        """Correct one branch row's max_depth without disturbing its strategy.

        max_depth is fully determined by best_guess and the max_depth of that
        guess's response groups, so a row whose stored value disagrees with
        that fold can be set to the folded value without re-searching: the
        strategy is unchanged, only the worst-case line length it was
        recorded with.  best_guess, best_score and solve_budget are left as
        they are.

        updated_at moves to now, because the row did change: export_cache's
        --since selects on it, so a repair that left the timestamp alone would
        be dropped from every incremental export.  A full export does not
        carry it either — import_cache keeps the target's row for a collision
        that isn't tainted->untainted — so each cache is repaired on its own
        machine rather than receiving another's repairs.

        solve_budget names which of the branch's results to correct: None for
        the unrestricted one, a budget for the result solved under that cap.

        Returns True when a row was updated.
        """
        if solve_budget is None:
            sql = """
                UPDATE branch_best_by_policy SET max_depth = ?, updated_at = ?
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
            """
            params = (max_depth, int(time.time()), branch_key, policy,
                      self.answer_list_id)
        else:
            sql = """
                UPDATE branch_best_by_policy_and_budget
                SET max_depth = ?, updated_at = ?
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
                  AND solve_budget = ?
            """
            params = (max_depth, int(time.time()), branch_key, policy,
                      self.answer_list_id, solve_budget)
        cursor = self._conn.execute(sql, params)
        self._mem_cache.pop((branch_key, policy, solve_budget), None)
        return cursor.rowcount > 0

    def delete_candidate_erds_for_policy(self, policy):
        """Drop every folded candidate ERD for one policy, returning the count.

        Each of those rows memoises a fold over branch_best_by_policy rows and
        is trusted on a matching response-group count alone, so a change to any
        branch row it folded leaves it stating a stale ERD and max_remaining_depth
        that no read re-checks.  A repair pass changes branch rows without
        changing any group count, and the memo is keyed by a hash of the
        parent's word set, so there is no way to ask which folds touched a
        given branch.  Dropping the policy's folds is therefore the narrowest
        invalidation the schema supports; each is re-earned by one fold.
        """
        cursor = self._conn.execute("""
            DELETE FROM candidate_erd_by_policy
            WHERE policy = ? AND answer_list_id = ?
        """, (policy, self.answer_list_id))
        return cursor.rowcount

    def delete(self, branch_key, policy):
        """Remove every exact result for a branch so it gets recomputed.

        For invalidating an entry a spot-check has found to be inconsistent
        with its own cached subtree.  Both scopes go: the branch is being
        recomputed, not one of its results.

        The session mirror is cleared by matching the branch rather than by
        the scopes some prior query listed.  A budget-specific result written
        between such a query and the delete is removed from SQLite by the
        delete's own WHERE clause but would keep being served from memory.
        """
        for table in ('branch_best_by_policy', 'branch_best_by_policy_and_budget'):
            self._conn.execute(f"""
                DELETE FROM {table}
                WHERE branch_key = ? AND policy = ? AND answer_list_id = ?
            """, (branch_key, policy, self.answer_list_id))
        self._mem_cache.pop_matching(
            lambda key: key[0] == branch_key and key[1] == policy)

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

    # ------------------------------------------------------------------
    # Candidate ERD cache (a candidate's own solved ERD at a branch)
    # ------------------------------------------------------------------

    def read_candidate_erd(self, branch_key, candidate_word, policy):
        """Return a candidate's stored ERD summary at a branch, or None.

        Only ever written once every one of the candidate's response groups
        is itself an exact branch_best_by_policy row (see write_candidate_erd),
        so a hit here needs no further reusability check beyond the caller
        confirming response_group_count still matches its own grouping.
        """
        subset_hash = self._subset_hash(branch_key)
        row = self._conn.execute("""
            SELECT erd, max_remaining_depth, response_group_count, updated_at
            FROM candidate_erd_by_policy
            WHERE subset_hash = ? AND candidate_word = ? AND policy = ?
              AND answer_list_id = ?
        """, (subset_hash, candidate_word, policy, self.answer_list_id)).fetchone()
        return dict(row) if row is not None else None

    def candidate_erd_map(self, policy):
        """Bulk-load every stored candidate ERD for a policy, keyed by
        (subset_hash, candidate_word).

        Mirrors report_branch_row_maps: folding a whole vocabulary one
        candidate at a time would otherwise cost one query per candidate.
        """
        return {
            (row["subset_hash"], row["candidate_word"]): dict(row)
            for row in self._conn.execute("""
                SELECT subset_hash, candidate_word, erd, max_remaining_depth,
                       response_group_count, updated_at
                FROM candidate_erd_by_policy
                WHERE policy = ? AND answer_list_id = ?
            """, (policy, self.answer_list_id))
        }

    def candidate_erd_from_map(self, branch_key, candidate_word, stored_map):
        """Look up a candidate's stored ERD in a map from candidate_erd_map."""
        return stored_map.get((self._subset_hash(branch_key), candidate_word))

    def delete_candidate_erd(self, branch_key, candidate_word, policy):
        """Drop a candidate's folded ERD at a branch.

        The stored fold is a memo whose premise is that every response group
        behind it is an exact branch_best_by_policy row (see
        write_candidate_erd), and a reader trusts it without re-checking those
        rows.  Deleting any of them breaks the premise, so whoever deletes them
        drops this row in the same breath; the next read folds afresh and
        persists again once the groups are solved.
        """
        self._conn.execute("""
            DELETE FROM candidate_erd_by_policy
            WHERE subset_hash = ? AND candidate_word = ? AND policy = ?
              AND answer_list_id = ?
        """, (self._subset_hash(branch_key), candidate_word, policy,
              self.answer_list_id))

    def write_candidate_erd(self, branch_key, candidate_word, policy, erd,
                             max_remaining_depth, response_group_count):
        """Persist a candidate's own solved ERD at a branch.

        Every response group behind it is already an exact, finalized
        branch_best_by_policy row by the time a caller has an `erd` to pass
        here, so the value is immutable going forward — write-once in the
        same sense those rows are. Disk I/O errors are logged and swallowed
        like write().
        """
        subset_hash = self._subset_hash(branch_key)
        now = int(time.time())
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO candidate_erd_by_policy
                    (subset_hash, candidate_word, policy, answer_list_id,
                     erd, max_remaining_depth, response_group_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (subset_hash, candidate_word, policy, self.answer_list_id,
                  erd, max_remaining_depth, response_group_count, now))
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_candidate_erd(%s, %s) failed: %s",
                           policy, candidate_word, exc)

    def completed_source_summary_map(self, policy):
        return {row["source_word"].lower(): dict(row) for row in self._conn.execute("""
            SELECT source_word, completed_at, elapsed_millis, worker_millis,
                   telemetry_epochs
            FROM completed_source_summaries
            WHERE policy = ? AND answer_list_id = ?
        """, (policy, self.answer_list_id))}

    def write_completed_source_summary(self, source_word, policy, completed_at,
                                       elapsed_millis, worker_millis,
                                       telemetry_epochs=()):
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO completed_source_summaries
                    (source_word, policy, answer_list_id, completed_at,
                     elapsed_millis, worker_millis, telemetry_epochs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_word.lower(), policy, self.answer_list_id, completed_at,
                  elapsed_millis, worker_millis,
                  ",".join(str(epoch) for epoch in sorted(set(telemetry_epochs)))))
        except sqlite3.OperationalError as exc:
            if not _is_disk_io_error(exc):
                raise
            logger.warning("write_completed_source_summary(%s, %s) failed: %s",
                           source_word, policy, exc)

    def add_root_response_group_summary(self, source_word, response_pattern,
                                        policy, nodes, worker_millis,
                                        created_at, finalized_at, epoch):
        row = self._conn.execute("""
            SELECT telemetry_epochs FROM root_response_group_summaries
            WHERE source_word = ? AND response_pattern = ? AND policy = ?
              AND answer_list_id = ?
        """, (source_word.lower(), response_pattern, policy,
              self.answer_list_id)).fetchone()
        epochs = {str(epoch)}
        if row is not None:
            epochs.update(filter(None, row["telemetry_epochs"].split(",")))
        self._conn.execute("""
            INSERT INTO root_response_group_summaries
                (source_word, response_pattern, policy, answer_list_id,
                 branch_count, search_node_count, worker_millis,
                 first_created_at, last_finalized_at, telemetry_epochs)
            VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(source_word, response_pattern, policy, answer_list_id)
            DO UPDATE SET branch_count = branch_count + 1,
                search_node_count = search_node_count + excluded.search_node_count,
                worker_millis = worker_millis + excluded.worker_millis,
                first_created_at = MIN(first_created_at, excluded.first_created_at),
                last_finalized_at = MAX(last_finalized_at, excluded.last_finalized_at),
                telemetry_epochs = excluded.telemetry_epochs
        """, (source_word.lower(), response_pattern, policy, self.answer_list_id,
              nodes, worker_millis or 0, created_at, finalized_at,
              ",".join(sorted(epochs, key=int))))

    def root_response_group_summary_map(self, source_word, policy):
        rows = self._conn.execute("""
            SELECT * FROM root_response_group_summaries
            WHERE source_word = ? AND policy = ? AND answer_list_id = ?
        """, (source_word.lower(), policy, self.answer_list_id)).fetchall()
        return {row["response_pattern"]: dict(row) for row in rows}

    def last_write_ts(self):
        """Return the unix timestamp of the most recent ERD write, or None."""
        row = self._conn.execute("""
            SELECT MAX(m) AS m FROM (
                SELECT MAX(updated_at) AS m FROM branch_best_by_policy
                WHERE answer_list_id = ?
                UNION ALL
                SELECT MAX(updated_at) AS m FROM branch_best_by_policy_and_budget
                WHERE answer_list_id = ?
            )
        """, (self.answer_list_id, self.answer_list_id)).fetchone()
        return row["m"] if row else None

    def erd_report_summary(self, policy: str, recent_since: int) -> dict:
        """Return bounded aggregate ERD counts for the current answer list."""
        exact = self._conn.execute("""
            SELECT COUNT(*) AS exact_branch_count,
                   COUNT(CASE WHEN updated_at >= ? THEN 1 END)
                       AS recent_exact_branch_count
            FROM branch_best_by_policy
            WHERE policy = ? AND answer_list_id = ?
        """, (recent_since, policy, self.answer_list_id)).fetchone()
        # Counted separately, never unioned: a branch holding results at three
        # budgets is one branch, and adding the rows up would report it as four.
        budgeted = self._conn.execute("""
            SELECT COUNT(*) AS budgeted_result_count,
                   COUNT(DISTINCT branch_key) AS budgeted_branch_count,
                   COUNT(CASE WHEN updated_at >= ? THEN 1 END)
                       AS recent_budgeted_result_count
            FROM branch_best_by_policy_and_budget
            WHERE policy = ? AND answer_list_id = ?
        """, (recent_since, policy, self.answer_list_id)).fetchone()
        loss = self._conn.execute("""
            SELECT COUNT(*) AS loss_branch_count
            FROM branch_loss_by_policy
            WHERE policy = ? AND answer_list_id = ?
        """, (policy, self.answer_list_id)).fetchone()
        return {
            "exact_branch_count": exact["exact_branch_count"],
            "recent_exact_branch_count": exact["recent_exact_branch_count"],
            "budgeted_result_count": budgeted["budgeted_result_count"],
            "budgeted_branch_count": budgeted["budgeted_branch_count"],
            "recent_budgeted_result_count":
                budgeted["recent_budgeted_result_count"],
            "loss_branch_count": loss["loss_branch_count"],
        }

    @staticmethod
    def _exact_row_for_budget(facts, budget):
        """The row a search at `budget` would reuse, from one branch's facts.

        Mirrors read_for_budget: the unrestricted result wins whenever its own
        worst case fits, and only then does the budget-specific one apply.
        `facts` is (unrestricted_row, {solve_budget: row}); either half may be
        empty.
        """
        if facts is None:
            return None
        canonical, by_budget = facts
        if budget is None:
            return canonical
        if canonical is not None:
            max_remaining_depth = canonical["max_depth"]
            if max_remaining_depth is not None and max_remaining_depth <= budget:
                return canonical
        return by_budget.get(budget)

    @staticmethod
    def _report_cache_state_from_rows(branch_key, exact_row, loss_row, budget):
        answer_count = len(branch_key) // 5
        if answer_count < 2:
            return {
                "cache_state": "not_applicable",
                "best_guess": None,
                "best_erd": None,
                "max_remaining_depth": None,
                "solve_budget": None,
                "tainted": False,
                "loss_budget": None,
                "updated_at": None,
            }
        if exact_row is not None:
            max_remaining_depth = exact_row["max_depth"]
            solve_budget = exact_row["solve_budget"]
            reusable = False
            if budget is None:
                reusable = (
                    solve_budget is None and max_remaining_depth is not None
                )
            elif max_remaining_depth is not None:
                reusable = (
                    max_remaining_depth <= budget
                    if solve_budget is None else solve_budget == budget
                )
            if reusable:
                return {
                    "cache_state": "exact",
                    "best_guess": exact_row["best_guess"],
                    "best_erd": exact_row["best_score"],
                    "max_remaining_depth": max_remaining_depth,
                    "solve_budget": solve_budget,
                    "tainted": solve_budget is not None,
                    "loss_budget": None,
                    "updated_at": exact_row["updated_at"],
                }
        if (
            loss_row is not None
            and budget is not None
            and budget <= loss_row["loss_budget"]
        ):
            return {
                "cache_state": "loss",
                "best_guess": None,
                "best_erd": None,
                "max_remaining_depth": None,
                "solve_budget": None,
                "tainted": False,
                "loss_budget": loss_row["loss_budget"],
                "updated_at": loss_row["updated_at"],
            }
        return {
            "cache_state": "missing",
            "best_guess": None,
            "best_erd": None,
            "max_remaining_depth": None,
            "solve_budget": None,
            "tainted": False,
            "loss_budget": None,
            "updated_at": None,
        }

    @staticmethod
    def report_branch_state_without_rows(branch_key, budget=None):
        """Return report state when no exact or loss cache rows are available."""
        return ScoreCache._report_cache_state_from_rows(
            bytes(branch_key), None, None, budget
        )

    def report_branch_state(self, branch_key, policy, budget=None) -> dict:
        """Return the reusable cache state for one branch and budget."""
        return self.report_branch_states([branch_key], policy, budget)[bytes(branch_key)]

    def report_branch_states(self, branch_keys, policy, budget=None) -> dict:
        """Return reusable cache states for a bounded set of branch keys."""
        if not branch_keys:
            return {}
        keys = [bytes(branch_key) for branch_key in branch_keys]
        placeholders = ",".join("?" for _ in keys)
        exact_rows = self._conn.execute(
            f"""SELECT branch_key, best_guess, best_score, updated_at,
                       max_depth, solve_budget
                FROM branch_best_by_policy
                WHERE policy = ? AND answer_list_id = ?
                  AND branch_key IN ({placeholders})
                UNION ALL
                SELECT branch_key, best_guess, best_score, updated_at,
                       max_depth, solve_budget
                FROM branch_best_by_policy_and_budget
                WHERE policy = ? AND answer_list_id = ?
                  AND branch_key IN ({placeholders})""",
            [policy, self.answer_list_id, *keys,
             policy, self.answer_list_id, *keys],
        ).fetchall()
        loss_rows = self._conn.execute(
            f"""SELECT branch_key, loss_budget, updated_at
                FROM branch_loss_by_policy
                WHERE policy = ? AND answer_list_id = ?
                  AND branch_key IN ({placeholders})""",
            [policy, self.answer_list_id, *keys],
        ).fetchall()
        exact_by_key = _branch_facts_by_key(exact_rows)
        loss_by_key = {bytes(row["branch_key"]): row for row in loss_rows}
        return {
            key: self._report_cache_state_from_rows(
                key, self._exact_row_for_budget(exact_by_key.get(key), budget),
                loss_by_key.get(key), budget
            )
            for key in keys
        }

    def report_branch_row_maps(self, policy):
        """Bulk-load every exact and loss row for a policy, keyed by branch_key.

        Folding a whole candidate vocabulary at once would otherwise need one
        `IN (...)` query per candidate; loading the full maps once and looking
        up in memory keeps the leaderboard a single pass over the cache.  The
        rows carry the same columns `_report_cache_state_from_rows` reads, so
        the caller reuses that reusability gate unchanged.
        """
        exact_by_key = _branch_facts_by_key(self._conn.execute(
            """SELECT branch_key, best_guess, best_score, updated_at,
                      max_depth, solve_budget
               FROM branch_best_by_policy
               WHERE policy = ? AND answer_list_id = ?
               UNION ALL
               SELECT branch_key, best_guess, best_score, updated_at,
                      max_depth, solve_budget
               FROM branch_best_by_policy_and_budget
               WHERE policy = ? AND answer_list_id = ?""",
            (policy, self.answer_list_id, policy, self.answer_list_id),
        ))
        loss_by_key = {
            bytes(row["branch_key"]): row
            for row in self._conn.execute(
                """SELECT branch_key, loss_budget, updated_at
                   FROM branch_loss_by_policy
                   WHERE policy = ? AND answer_list_id = ?""",
                (policy, self.answer_list_id),
            )
        }
        return exact_by_key, loss_by_key

    def report_branch_states_from_maps(
        self, branch_keys, exact_by_key, loss_by_key, budget=None
    ) -> dict:
        """Reusable cache states for keys, from pre-loaded row maps.

        `report_branch_row_maps` loads every exact/loss row for a policy once;
        this applies the same reusability gate `report_branch_states` uses
        without a per-key query, so a whole candidate vocabulary folds in one
        pass.  The maps must carry the columns the gate reads, which is exactly
        what `report_branch_row_maps` returns.
        """
        return {
            bytes(branch_key): self._report_cache_state_from_rows(
                bytes(branch_key),
                self._exact_row_for_budget(
                    exact_by_key.get(bytes(branch_key)), budget),
                loss_by_key.get(bytes(branch_key)),
                budget,
            )
            for branch_key in branch_keys
        }

    def report_recent_rows(self, policy, since, limit) -> list[dict]:
        """Return bounded recently updated exact branch rows."""
        rows = self._conn.execute("""
            SELECT branch_key, best_guess, best_score, updated_at,
                   max_depth, solve_budget
            FROM branch_best_by_policy
            WHERE policy = ? AND answer_list_id = ? AND updated_at >= ?
            UNION ALL
            SELECT branch_key, best_guess, best_score, updated_at,
                   max_depth, solve_budget
            FROM branch_best_by_policy_and_budget
            WHERE policy = ? AND answer_list_id = ? AND updated_at >= ?
            ORDER BY updated_at DESC, branch_key
            LIMIT ?
        """, (policy, self.answer_list_id, since,
              policy, self.answer_list_id, since, limit)).fetchall()
        return [
            {
                "branch_key": bytes(row["branch_key"]),
                "best_guess": row["best_guess"],
                "best_erd": row["best_score"],
                "updated_at": row["updated_at"],
                "max_remaining_depth": row["max_depth"],
                "solve_budget": row["solve_budget"],
                "tainted": row["solve_budget"] is not None,
            }
            for row in rows
        ]

    def report_cache_distributions(self, policy) -> dict:
        """Return exact/loss counts grouped by their cache reuse axes."""
        # Rows, not branches: a branch with results at three budgets
        # contributes one unrestricted row and three budget-specific ones, and
        # the by_solve_budget axis is what tells them apart.
        exact_rows = self._conn.execute("""
            SELECT max_depth, solve_budget, COUNT(*) AS branch_count
            FROM branch_best_by_policy
            WHERE policy = ? AND answer_list_id = ?
            GROUP BY max_depth, solve_budget
            UNION ALL
            SELECT max_depth, solve_budget, COUNT(*) AS branch_count
            FROM branch_best_by_policy_and_budget
            WHERE policy = ? AND answer_list_id = ?
            GROUP BY max_depth, solve_budget
        """, (policy, self.answer_list_id, policy, self.answer_list_id)).fetchall()
        loss_rows = self._conn.execute("""
            SELECT loss_budget, COUNT(*) AS branch_count
            FROM branch_loss_by_policy
            WHERE policy = ? AND answer_list_id = ?
            GROUP BY loss_budget
        """, (policy, self.answer_list_id)).fetchall()
        by_max_remaining_depth = {}
        by_solve_budget = {}
        by_taint = {"untainted": 0, "tainted": 0}
        for row in exact_rows:
            max_key = "unknown" if row["max_depth"] is None else str(row["max_depth"])
            budget_key = (
                "unbounded" if row["solve_budget"] is None
                else str(row["solve_budget"])
            )
            count = row["branch_count"]
            by_max_remaining_depth[max_key] = (
                by_max_remaining_depth.get(max_key, 0) + count
            )
            by_solve_budget[budget_key] = by_solve_budget.get(budget_key, 0) + count
            taint_key = "tainted" if row["solve_budget"] is not None else "untainted"
            by_taint[taint_key] += count
        by_loss_budget = {
            str(row["loss_budget"]): row["branch_count"] for row in loss_rows
        }
        return {
            "state_branch_counts": {
                "exact_branch_count": sum(row["branch_count"] for row in exact_rows),
                "loss_branch_count": sum(row["branch_count"] for row in loss_rows),
            },
            "exact_branch_count_by_max_remaining_depth": by_max_remaining_depth,
            "exact_branch_count_by_solve_budget": by_solve_budget,
            "exact_branch_count_by_taint": by_taint,
            "loss_branch_count_by_loss_budget": by_loss_budget,
        }

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
        # (scope, branch_key, policy, solve_budget) ->
        #     (best_guess, best_score, max_depth, solve_budget)
        # solve_budget is part of the key for the same reason it is in SQLite:
        # a branch's unrestricted optimum and its optimum under a cap are
        # different facts, and one entry cannot hold both.
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
        result = self._data.get((self._scope, branch_key, policy, None))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result[:2]

    def read_with_depth(self, branch_key, policy):
        result = self._data.get((self._scope, branch_key, policy, None))
        if result is None:
            self.read_misses += 1
            return None
        self.read_hits += 1
        return result

    def read_for_budget(self, branch_key, policy, budget):
        """The entry a search at `budget` should reuse — see ScoreCache."""
        canonical = self.read_with_depth(branch_key, policy)
        if budget is None:
            return canonical
        if canonical is not None:
            max_remaining_depth = canonical[2]
            if max_remaining_depth is not None and max_remaining_depth <= budget:
                return canonical
        result = self._data.get((self._scope, branch_key, policy, budget))
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
        """Store an exact result, or adopt the one already held.

        Same invariant as ScoreCache.write, for the same reason: max_depth is
        ancestor-visible, so a result already stored for this scope stands and
        is returned for the caller to adopt rather than being replaced.  A
        disagreement on cost raises CacheWriteConflict.
        """
        key = (self._scope, branch_key, policy, solve_budget)
        stored = self._data.get(key)
        if stored is not None:
            if abs(stored[1] - best_score) > EXACT_SCORE_TOLERANCE:
                raise CacheWriteConflict(
                    f"{branch_reference(branch_key)} at policy={policy} "
                    f"budget={solve_budget}: stored {stored[1]!r}, "
                    f"incoming {best_score!r}")
            return stored
        entry = (best_guess, best_score, max_depth, solve_budget)
        self._data[key] = entry
        self.write_count += 1
        return entry

    def read_loss(self, branch_key, policy, refresh=False):
        return self._losses.get((self._scope, branch_key, policy))

    def write_loss(self, branch_key, policy, budget):
        key = (self._scope, branch_key, policy)
        prior = self._losses.get(key)
        if prior is None or budget > prior:
            self._losses[key] = budget
        self.write_count += 1

    def close(self):
        pass
