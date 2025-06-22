import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class WAYSEA2(AbstractUnconstrainedMinimisation):
    """WAYSEA2 problem - SCIPY global optimization benchmark example WayburnSeader02.

    Fit: y = (-4 x_1^2 -4 x_2^2 + 2.5 x_1 + 13 x_2 - 9.340125, x_2 - 1) + e

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

        # F1 = -4*x1^2 - 4*x2^2 + 2.5*x1 + 13*x2 - 9.340125
        f1 = -4.0 * x1**2 - 4.0 * x2**2 + 2.5 * x1 + 13.0 * x2 - 9.340125

        # F2 = x2 - 1
        f2 = x2 - 1.0

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
        return jnp.array([0.424861025, 1.0])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
