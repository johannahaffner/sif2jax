"""SCIPY global optimization benchmark example ElAttarVidyasagarDutta.

Box-constrained version.

Fit: (x_1^2 + x_2 − 10, x_1 + x_2^2 − 7, x_1^2 + x_2^3 − 1) + e = 0

Source: Problem from the SCIPY benchmark set
https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

SIF input: Nick Gould

Classification: SBR2-MN-2-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class ELATVIDUB(AbstractBoundedMinimisation):
    """ElAttarVidyasagarDutta benchmark problem (bounded version)."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Starting point."""
        return jnp.array([1.0, 5.0])

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Variable bounds: -100 ≤ x ≤ 100."""
        lower = jnp.array([-100.0, -100.0])
        upper = jnp.array([100.0, 100.0])
        return lower, upper

    def objective(self, y, args):
        """Least squares objective: sum of squared residuals."""
        del args
        x1, x2 = y

        # Three residual functions
        f1 = x1**2 + x2 - 10.0  # x1² + x2 - 10
        f2 = x1 + x2**2 - 7.0  # x1 + x2² - 7
        f3 = x1**2 + x2**3 - 1.0  # x1² + x2³ - 1

        # Least squares objective
        return f1**2 + f2**2 + f3**2

    def num_variables(self):
        return 2

    @property
    def expected_result(self):
        """Expected optimal solution (not provided)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Lower bound is 0.0 for least squares problems
        return jnp.array(0.0)
