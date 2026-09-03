"""Test harness helpers for enforcing queue opener-work invariants."""


class OpenerWorkInvariantCheckMixin:
    def close(self):
        if getattr(self, "_opener_work_invariants_checked", False):
            return
        violations = self.check_opener_work_invariants()
        super().close()
        self._opener_work_invariants_checked = True
        if violations:
            raise AssertionError("\n".join(violations))
