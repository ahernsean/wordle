"""SQLite-backed cache for Wordle scores and lookahead results."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


class ScoreCache:
    """Persists per-word scores and subgroup lookahead results.

    Tables:
      word_scores     — per-word scoring method results (level 1)
      lookahead_result — best step-2 word per subgroup (levels 2+)
      universe        — fingerprint of the answer word set

    All entries are keyed by universe_id so a different answer list
    produces a clean namespace without needing a new file.
    """

    def __init__(self, db_path, answer_words, timeout=30.0):
        self.db_path = Path(db_path)
        self.answer_words = list(answer_words)
        self._conn = sqlite3.connect(
            self.db_path, timeout=timeout, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self.universe_id = self._ensure_universe()

    def _ensure_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                universe_id  TEXT PRIMARY KEY,
                answer_hash  TEXT NOT NULL,
                answer_count INTEGER NOT NULL,
                created_at   INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lookahead_result (
                subset_key   BLOB NOT NULL,
                policy       TEXT NOT NULL,
                universe_id  TEXT NOT NULL,
                best_word    TEXT NOT NULL,
                best_entropy REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (subset_key, policy, universe_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lookahead
            ON lookahead_result(universe_id, policy)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS word_scores (
                word         TEXT    NOT NULL,
                method       TEXT    NOT NULL,
                score        REAL    NOT NULL,
                universe_id  TEXT    NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (word, method, universe_id)
            )
        """)
        # All valid 5-letter words are ASCII, so a null byte identifies the
        # old null-separated subset-key encoding.
        self._purge_legacy_rows("instr(subset_key, char(0)) > 0", ())
        # 'erd' was renamed to 'erd_answers' and then superseded by 'erd_all'.
        self._purge_legacy_rows("policy = ?", ('erd',))
        # 'erd_hard' was renamed to 'erd_constrained'; constraint-compliant
        # mode is now always transient (MemoryScoreCache), so any persisted
        # rows under either name are useless regardless of age.
        self._purge_legacy_rows("policy = ?", ('erd_hard',))

    def _purge_legacy_rows(self, where, params):
        """One-time cleanup of stale lookahead_result rows.

        Once a legacy batch is gone it stays gone, so a full-table DELETE on
        every connection open (including each ERDWarmer thread) would scan
        the whole table for nothing.  Check existence first — LIMIT 1 lets
        SQLite stop at the first match — and only DELETE when there's
        actually something to remove.
        """
        exists = self._conn.execute(
            f"SELECT 1 FROM lookahead_result WHERE {where} LIMIT 1", params
        ).fetchone()
        if exists is not None:
            self._conn.execute(f"DELETE FROM lookahead_result WHERE {where}", params)

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
        self._conn.close()

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
        """Return (best_word, best_entropy) or None on cache miss."""
        row = self._conn.execute("""
            SELECT best_word, best_entropy
            FROM lookahead_result
            WHERE subset_key = ? AND policy = ? AND universe_id = ?
        """, (subset_key, policy, self.universe_id)).fetchone()
        if row is None:
            return None
        return row["best_word"], row["best_entropy"]

    def write(self, subset_key, policy, best_word, best_entropy):
        """Store a completed subgroup result."""
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO lookahead_result
                (subset_key, policy, universe_id,
                 best_word, best_entropy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (subset_key, policy, self.universe_id,
              best_word, best_entropy, now))

    # ------------------------------------------------------------------
    # Word score cache (level 1, all ScoringMethods)
    # ------------------------------------------------------------------

    def read_scores(self, method):
        """Return list of (word, score) for this method/universe, or None if empty."""
        rows = self._conn.execute("""
            SELECT word, score FROM word_scores
            WHERE method = ? AND universe_id = ?
        """, (method, self.universe_id)).fetchall()
        if not rows:
            return None
        return [(r["word"], r["score"]) for r in rows]

    def write_scores(self, scores, method):
        """Store list of (word, score) tuples for this method/universe."""
        now = int(time.time())
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany("""
                INSERT OR REPLACE INTO word_scores
                    (word, method, score, universe_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, [(w, method, s, self.universe_id, now) for w, s in scores])
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def stats(self):
        """Return (lookahead_rows, word_score_rows, last_updated_ts)."""
        la = self._conn.execute("""
            SELECT COUNT(*) AS c, MAX(updated_at) AS m
            FROM lookahead_result WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        ws = self._conn.execute("""
            SELECT COUNT(*) AS c FROM word_scores WHERE universe_id = ?
        """, (self.universe_id,)).fetchone()
        return la["c"] or 0, ws["c"] or 0, la["m"]


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
        self._data = {}  # (scope, subset_key_bytes, policy) -> (best_word, best_score)
        self._scope = None

    @staticmethod
    def fingerprint_vocabulary(words):
        """Order-independent digest identifying an eligible-guess word set."""
        canonical = "\n".join(sorted(words))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def set_scope(self, fingerprint):
        """Switch the active vocabulary scope for subsequent read/write calls."""
        self._scope = fingerprint

    def read(self, subset_key, policy):
        return self._data.get((self._scope, subset_key, policy))

    def write(self, subset_key, policy, best_word, best_entropy):
        self._data[(self._scope, subset_key, policy)] = (best_word, best_entropy)

    def close(self):
        pass
