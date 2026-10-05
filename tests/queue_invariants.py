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
