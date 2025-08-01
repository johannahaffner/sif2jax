"""Problem 14 (chained and modified HS53) from Luksan.

This is a least squares problem from the paper:
L. Luksan
"Hybrid methods in large sparse nonlinear least squares"
J. Optimization Theory & Applications 89(3) 575-595 (1996)

SIF input: Nick Gould, June 2017.

least-squares version

classification SUR2-AN-V-0
"""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN14LS(AbstractUnconstrainedMinimisation):
    """Problem 14 (chained and modified HS53) from Luksan - least squares version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 32  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 3*S + 2."""
        return 3 * self.s + 2

    @property
    def y0(self) -> Array:
        """Initial guess: x(i) = -1.0 for all i."""
        return -jnp.ones(self.n, dtype=jnp.float64)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y: Array, args) -> Array:
        """Compute the least squares objective function.

        The objective is the sum of squares of M = 7*S residuals.
        Each block of S generates 7 residuals:
        - E(k): -10*x(i+1) + 10*x(i)^2
        - E(k+1): x(i+1) + x(i+2) - 2.0
        - E(k+2): x(i+3) - 1.0
        - E(k+3): x(i+4) - 1.0
        - E(k+4): x(i) + 3*x(i+1)
        - E(k+5): x(i+2) + x(i+3) - 2*x(i+4)
        - E(k+6): -10*x(i+4) + 10*x(i+1)^2
        """
        del args  # Not used

        x = y
        s = self.s

        # Initialize residual vector
        residuals = []

        # Loop over S blocks
        i = 0  # Variable index (0-based)

        for j in range(s):
            # Each block has 7 equations
            # Variables involved: x[i], x[i+1], x[i+2], x[i+3], x[i+4]

            # E(k): -10*x(i+1) + 10*x(i)^2
            res1 = -10.0 * x[i + 1] + 10.0 * x[i] * x[i]
            residuals.append(res1)

            # E(k+1): x(i+1) + x(i+2) - 2.0
            res2 = x[i + 1] + x[i + 2] - 2.0
            residuals.append(res2)

            # E(k+2): x(i+3) - 1.0
            res3 = x[i + 3] - 1.0
            residuals.append(res3)

            # E(k+3): x(i+4) - 1.0
            res4 = x[i + 4] - 1.0
            residuals.append(res4)

            # E(k+4): x(i) + 3*x(i+1)
            res5 = x[i] + 3.0 * x[i + 1]
            residuals.append(res5)

            # E(k+5): x(i+2) + x(i+3) - 2*x(i+4)
            res6 = x[i + 2] + x[i + 3] - 2.0 * x[i + 4]
            residuals.append(res6)

            # E(k+6): -10*x(i+4) + 10*x(i+1)^2
            res7 = -10.0 * x[i + 4] + 10.0 * x[i + 1] * x[i + 1]
            residuals.append(res7)

            # Update indices
            i += 3

        # Sum of squares (L2 group type in SIF)
        residuals_array = jnp.array(residuals)
        return jnp.sum(residuals_array**2)

    @property
    def expected_result(self) -> Array | None:
        """Expected optimal solution."""
        # No specific expected result provided in SIF
        return None

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected objective value."""
        # For least squares, the expected value is 0
        return jnp.array(0.0)
