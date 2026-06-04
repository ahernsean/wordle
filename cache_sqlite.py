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
                subset_blob  BLOB NOT NULL,
                policy       TEXT NOT NULL,
                universe_id  TEXT NOT NULL,
                best_word    TEXT NOT NULL,
                best_entropy REAL NOT NULL,
                updated_at   INTEGER NOT NULL,
                PRIMARY KEY (subset_blob, policy, universe_id)
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
        """Encode a word list as a deterministic blob key."""
        return "\0".join(sorted(words)).encode("utf-8")

    def read(self, subset_blob, policy):
        """Return (best_word, best_entropy) or None on cache miss."""
        row = self._conn.execute("""
            SELECT best_word, best_entropy
            FROM lookahead_result
            WHERE subset_blob = ? AND policy = ? AND universe_id = ?
        """, (subset_blob, policy, self.universe_id)).fetchone()
        if row is None:
            return None
        return row["best_word"], row["best_entropy"]

    def write(self, subset_blob, policy, best_word, best_entropy):
        """Store a completed subgroup result."""
        now = int(time.time())
        self._conn.execute("""
            INSERT OR REPLACE INTO lookahead_result
                (subset_blob, policy, universe_id,
                 best_word, best_entropy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (subset_blob, policy, self.universe_id,
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
