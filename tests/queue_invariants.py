"""Test harness helpers for enforcing queue source-work invariants."""


class SourceWorkInvariantCheckMixin:
    def close(self):
        if getattr(self, "_source_work_invariants_checked", False):
            return
        violations = self.check_source_work_invariants()
        super().close()
        self._source_work_invariants_checked = True
        if violations:
            raise AssertionError("\n".join(violations))
