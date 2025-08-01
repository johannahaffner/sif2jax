"""Problem 13 (chained and modified HS48) from Luksan.

This is a system of nonlinear equations from the paper:
L. Luksan
"Hybrid methods in large sparse nonlinear least squares"
J. Optimization Theory & Applications 89(3) 575-595 (1996)

SIF input: Nick Gould, June 2017.

classification NOR2-AN-V-V
"""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN13(AbstractNonlinearEquations):
    """Problem 13 (chained and modified HS48) from Luksan."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 32  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 3*S + 2."""
        return 3 * self.s + 2

    @property
    def m(self) -> int:
        """Number of equations: 7*S."""
        return 7 * self.s

    @property
    def y0(self) -> Array:
        """Initial guess: x(i) = -1.0 for all i."""
        return -jnp.ones(self.n, dtype=jnp.float64)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        del args  # Not used

        x = y
        s = self.s
        m = self.m

        # Initialize residual vector
        residuals = jnp.zeros(m, dtype=jnp.float64)

        # Loop over S blocks
        i = 0  # Variable index (0-based)
        k = 0  # Equation index (0-based)

        for j in range(s):
            # Each block has 7 equations
            # Variables involved: x[i], x[i+1], x[i+2], x[i+3], x[i+4]

            # E(k): -10*x(i+1) + 10*x(i)^2
            residuals = residuals.at[k].set(-10.0 * x[i + 1] + 10.0 * x[i] * x[i])

            # E(k+1): -10*x(i+2) + 10*x(i+1)^2
            residuals = residuals.at[k + 1].set(
                -10.0 * x[i + 2] + 10.0 * x[i + 1] * x[i + 1]
            )

            # E(k+2): (x(i+2) - x(i+3))^2
            diff = x[i + 2] - x[i + 3]
            residuals = residuals.at[k + 2].set(diff * diff)

            # E(k+3): (x(i+3) - x(i+4))^2
            diff = x[i + 3] - x[i + 4]
            residuals = residuals.at[k + 3].set(diff * diff)

            # E(k+4): x(i) + x(i+2) + x(i+1)^2 - 30
            residuals = residuals.at[k + 4].set(
                x[i] + x[i + 2] + x[i + 1] * x[i + 1] - 30.0
            )

            # E(k+5): x(i+1) + x(i+3) - x(i+2)^2 - 10
            residuals = residuals.at[k + 5].set(
                x[i + 1] + x[i + 3] - x[i + 2] * x[i + 2] - 10.0
            )

            # E(k+6): x(i) * x(i+4) - 10
            residuals = residuals.at[k + 6].set(x[i] * x[i + 4] - 10.0)

            # Update indices
            i += 3
            k += 7

        return residuals

    @property
    def expected_result(self) -> Array | None:
        """Expected optimal solution."""
        # No specific expected result provided in SIF
        return None

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected objective value (sum of squares)."""
        # For nonlinear equations, the expected value is 0
        return jnp.array(0.0)

    def constraint(self, y: Array):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
