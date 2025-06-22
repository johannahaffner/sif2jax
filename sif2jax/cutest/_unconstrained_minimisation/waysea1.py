import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class WAYSEA1(AbstractUnconstrainedMinimisation):
    """WAYSEA1 problem - SCIPY global optimization benchmark example WayburnSeader01.

    Fit: y = (X_1^6 + x_2^4, 2x_1 + x_2) + e

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    def objective(self, y, args):
        """Compute the sum of squares objective."""
        del args
        x1, x2 = y

        # F1 = x1^6 + x2^4 - 17
        f1 = x1**6 + x2**4 - 17.0

        # F2 = 2*x1 + x2 - 4
        f2 = 2.0 * x1 + x2 - 4.0

        # Sum of squares
        return f1**2 + f2**2

    def y0(self):
        """Initial guess."""
        return jnp.array([1.0, 5.0])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
