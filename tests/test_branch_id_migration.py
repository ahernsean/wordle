"""Migration test for branch-key normalization: a pre-normalization queue
database (candidate_claims / candidate_republish / cut_results keyed by the fat
branch_key BLOB) must upgrade in place to the branches registry + branch_id
foreign key, preserving every row.
"""
import os
import sqlite3
import tempfile
import unittest

from erd_queue import ERDQueue, encode_subset

# The pre-normalization DDL, verbatim, so the test builds exactly the schema
# older code left behind.
_OLD_SCHEMA = """
CREATE TABLE candidate_claims (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    claimed_by TEXT,
    claimed_at INTEGER,
    done       INTEGER NOT NULL DEFAULT 0,
    done_at    INTEGER,
    bundle_id  TEXT,
    PRIMARY KEY (branch_key, idx)
);
CREATE TABLE candidate_republish (
    branch_key BLOB    NOT NULL,
    idx        INTEGER NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (branch_key, idx)
);
CREATE TABLE cut_results (
    branch_key BLOB    PRIMARY KEY,
    budget     INTEGER NOT NULL,
    bound      REAL    NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE active_branches (
    branch_key     BLOB    PRIMARY KEY,
    n_words        INTEGER NOT NULL,
    n_candidates   INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    best_erd       REAL,
    best_guess     TEXT,
    status         TEXT    NOT NULL DEFAULT 'open',
    created_at     INTEGER,
    finalized_at   INTEGER,
    budget         INTEGER,
    best_max_depth INTEGER,
    tainted        INTEGER NOT NULL DEFAULT 0,
    nodes_spent    INTEGER NOT NULL DEFAULT 0,
    spine          TEXT,
    ceiling        REAL,
    cut_occurred   INTEGER NOT NULL DEFAULT 0,
    bulk_done_candidates INTEGER NOT NULL DEFAULT 0,
    bulk_done_bound REAL,
    pack_cursor    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE pending_branches (
    branch_key     BLOB    NOT NULL,
    n_words        INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    source_word    TEXT,
    source_pattern INTEGER,
    status         TEXT    NOT NULL DEFAULT 'pending',
    claimed_by     TEXT,
    claimed_at     INTEGER,
    completed_at   INTEGER,
    PRIMARY KEY (branch_key)
);
"""


class TestBranchIdMigration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "q.sqlite3")
        self.key_a = encode_subset(["crane", "slate", "trace"])
        self.key_b = encode_subset(["abbey", "abide", "abode", "above"])
        con = sqlite3.connect(self.path)
        con.executescript(_OLD_SCHEMA)
        con.executemany(
            "INSERT INTO candidate_claims "
            "(branch_key, idx, claimed_by, done) VALUES (?, ?, 'w', ?)",
            [(self.key_a, 0, 1), (self.key_a, 1, 0), (self.key_b, 0, 1)])
        con.execute(
            "INSERT INTO candidate_republish (branch_key, idx, count) "
            "VALUES (?, 1, 3)", (self.key_a,))
        con.execute(
            "INSERT INTO cut_results (branch_key, budget, bound, created_at) "
            "VALUES (?, 3, 2.5, 0)", (self.key_b,))
        con.execute(
            "INSERT INTO active_branches (branch_key, n_words, n_candidates, "
            "best_erd, best_guess, status) VALUES (?, 3, 40, 1.5, 'crane', 'open')",
            (self.key_a,))
        con.execute(
            "INSERT INTO pending_branches (branch_key, n_words, priority, "
            "source_word) VALUES (?, 4, 1, 'salet')", (self.key_b,))
        con.commit()
        con.close()

    def test_old_schema_migrates_preserving_every_row(self):
        q = ERDQueue(self.path)   # opening runs _migrate()
        self.addCleanup(q.close)

        # Both keys are registered with distinct ids, stored once each.
        id_a = q._intern_branch(self.key_a)
        id_b = q._intern_branch(self.key_b)
        self.assertIsNotNone(id_a)
        self.assertIsNotNone(id_b)
        self.assertNotEqual(id_a, id_b)
        self.assertEqual(
            q._conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0], 2)

        # candidate_claims rows survive with their done flags intact.
        self.assertEqual(q.branch_done_candidates(self.key_a), 1)
        self.assertEqual(q.branch_done_candidates(self.key_b), 1)
        self.assertEqual(
            q._conn.execute(
                "SELECT COUNT(*) FROM candidate_claims WHERE branch_id = ?",
                (id_a,)).fetchone()[0], 2)

        # candidate_republish and cut_results survive, keyed by branch_id.
        self.assertEqual(
            q._conn.execute(
                "SELECT count FROM candidate_republish "
                "WHERE branch_id = ? AND idx = 1", (id_a,)).fetchone()[0], 3)
        self.assertEqual(q.read_cut_result(self.key_b), (2.5, 3))

        # active_branches / pending_branches survive, keyed by branch_id, with
        # branch_key reachable only through the registry.
        active = q.get_active_branch(self.key_a)
        self.assertIsNotNone(active)
        self.assertEqual(active["best_guess"], "crane")
        self.assertEqual(bytes(active["branch_key"]), self.key_a)
        self.assertTrue(q.has_pending_row(self.key_b))

        # The fat blob column is gone from every normalized operational table.
        for table in ("candidate_claims", "candidate_republish", "cut_results",
                      "active_branches", "pending_branches"):
            cols = {r[1] for r in
                    q._conn.execute(f"PRAGMA table_info({table})")}
            self.assertIn("branch_id", cols)
            self.assertNotIn("branch_key", cols)
        hb_cols = {r[1] for r in
                   q._conn.execute("PRAGMA table_info(worker_heartbeat)")}
        self.assertIn("current_branch_id", hb_cols)
        self.assertNotIn("current_branch_key", hb_cols)

    def test_reopen_after_migration_is_idempotent(self):
        ERDQueue(self.path).close()   # first open migrates
        q = ERDQueue(self.path)       # second open must be a clean no-op
        self.addCleanup(q.close)
        self.assertEqual(q.branch_done_candidates(self.key_a), 1)


if __name__ == "__main__":
    unittest.main()
