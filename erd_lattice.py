"""Exact-lattice comparisons for expected remaining depth values."""

# Worst on-lattice residual measured across the full production corpus was
# 3.6e-7 (issue #186); this keeps a ~3x margin while still routing any
# genuinely off-lattice value to erd_ge's raw-float fallback below.
ERD_LATTICE_NOISE_MARGIN = 1e-6


def erd_numerator(value, n_answers, *, noise_margin=ERD_LATTICE_NOISE_MARGIN):
    """Return the integer numerator of value, if value is exactly
    k/n_answers to within ``noise_margin`` in scaled numerator units; None if
    it lies off that lattice.

    A branch's ERD at any budget is (sum of integer line lengths) / n_answers,
    so every achievable ERD is exactly k/n_answers: a one-dimensional lattice
    of points spaced 1/n_answers apart.  The unpadded alpha-beta threshold
    lands on that same lattice.  Carried as float64 the value
    accumulates ~1e-9 of representation noise; multiplying by n_answers and
    rounding recovers the exact k.  Distinct lattice points are >= 1 apart
    after that scaling, so the noise margin can only ever collapse two
    encodings of the same value, never merge two different ERDs."""
    numerator = value * n_answers
    rounded = round(numerator)
    if abs(numerator - rounded) < noise_margin:
        return rounded
    return None


def erd_ge(left_value, right_value, n_answers):
    """True iff ERD left_value >= right_value for two values of an
    n_answers-word branch.

    When both reconstruct to an exact k/n_answers numerator, compares those
    integers, so float64 noise on the same rational cannot force a spurious
    cut re-solve.  Falls back to the raw float compare when either value is
    off that lattice: this comparison only ever gates reuse of already-proven
    work, so degrading to the pre-fix behavior is always safe, never wrong,
    and never fatal — unlike raising, which would turn an off-lattice operand
    into a worker crash instead of a merely-wasted re-solve."""
    left_numerator = erd_numerator(left_value, n_answers)
    right_numerator = erd_numerator(right_value, n_answers)
    if left_numerator is not None and right_numerator is not None:
        return left_numerator >= right_numerator
    return left_value >= right_value
