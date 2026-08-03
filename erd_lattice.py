"""Exact-lattice comparisons for expected remaining depth values."""

# Worst on-lattice residual measured across the full production corpus was
# 3.6e-7; this keeps a ~3x margin while still routing genuinely off-lattice
# values to the raw-float fallback below.
ERD_LATTICE_NOISE_MARGIN = 1e-6


def erd_numerator(value, n_answers, *, noise_margin=ERD_LATTICE_NOISE_MARGIN):
    """Return the integer numerator of an on-lattice ERD value, or None.

    A branch's ERD is an integer sum of line lengths divided by its answer
    count. Float representation noise can therefore be removed by comparing
    the reconstructed integer numerators.
    """
    numerator = value * n_answers
    rounded = round(numerator)
    if abs(numerator - rounded) < noise_margin:
        return rounded
    return None


def erd_ge(left_value, right_value, n_answers):
    """Return whether one ERD is at least another for the same branch size."""
    left_numerator = erd_numerator(left_value, n_answers)
    right_numerator = erd_numerator(right_value, n_answers)
    if left_numerator is not None and right_numerator is not None:
        return left_numerator >= right_numerator
    return left_value >= right_value
