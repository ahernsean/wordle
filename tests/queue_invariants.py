"""Test harness helpers for enforcing queue opener-work invariants."""


class OpenerWorkInvariantCheckMixin:
    def mark_done(self, branch_key):
        """Finish a branch, then do what a worker does for the openers it owes.

        Resolving an opener's last branch leaves its ERD reduction owed, and
        the opener is not done until that is stored.  Tests of queue
        lifecycles that are not about the reduction stand in for the worker by
        marking the ready openers done straight away.
        """
        ready = super().mark_done(branch_key)
        self.mark_openers_complete(ready or ())
        return ready

    def close(self):
        if getattr(self, "_opener_work_invariants_checked", False):
            return
        violations = self.check_opener_work_invariants()
        super().close()
        self._opener_work_invariants_checked = True
        if violations:
            raise AssertionError("\n".join(violations))


def add_second_request(queue, rows):
    """Queue (branch_key, n_words, priority, opener, opener_pattern) rows
    under a new opener-work request even though their opener already has an
    unfinished one.

    add_pending_many no longer does this, but a queue written before it
    attached to an opener's existing request can hold two unfinished requests
    for one opener.  The code that reads requests still has to handle that
    shape, so its tests build it directly.  Rows are expected to share one
    opener and priority.
    """
    _branch_key, _n_words, priority, opener, _pattern = rows[0]
    connection = queue._conn
    connection.execute("BEGIN IMMEDIATE")
    try:
        opener_work_id = connection.execute("""
            INSERT INTO opener_work
                (opener, requested_priority, requested_at, state)
            VALUES (?, ?, strftime('%s', 'now'), 'queued')
        """, (opener, priority)).lastrowid
        for branch_key, n_words, row_priority, _opener, pattern in rows:
            branch_id = queue._intern_branch(branch_key, create=True)
            connection.execute("""
                INSERT INTO pending_branches
                    (branch_id, n_words, priority, opener, opener_pattern,
                     status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(branch_id) DO UPDATE SET
                    priority = MAX(priority, excluded.priority)
            """, (branch_id, n_words, row_priority, opener, pattern))
            connection.execute("""
                INSERT INTO branch_opener_work
                    (branch_id, opener_work_id, parent_branch_id,
                     opener_pattern)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(branch_id, opener_work_id) DO NOTHING
            """, (branch_id, opener_work_id, pattern))
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return opener_work_id
