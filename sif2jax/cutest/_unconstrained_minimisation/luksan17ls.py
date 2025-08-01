"""Problem 17 (sparse trigonometric) from Luksan.

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


class LUKSAN17LS(AbstractUnconstrainedMinimisation):
    """Problem 17 (sparse trigonometric) from Luksan - least squares version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 49  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 2*S + 2."""
        return 2 * self.s + 2

    @property
    def y0(self) -> Array:
        """Initial guess: pattern (-0.8, 1.2, -1.2, 0.8) repeated."""
        x = jnp.zeros(self.n, dtype=jnp.float64)
        pattern = jnp.array([-0.8, 1.2, -1.2, 0.8], dtype=jnp.float64)
        for i in range(self.n):
            x = x.at[i].set(pattern[i % 4])
        return x

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y: Array, args) -> Array:
        """Compute the least squares objective function.

        The objective is the sum of squares of M = 4*S residuals.
        Each block contributes 4 residuals corresponding to data values Y1-Y4.
        Each residual is the sum over q=1,2,3,4 of trigonometric terms minus the data.
        """
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([30.6, 72.2, 124.4, 187.4], dtype=jnp.float64)

        # Initialize residual vector
        residuals = []

        # Loop over S blocks
        i = 0  # Variable index (0-based)

        for j in range(s):
            # Each block contributes 4 equations
            for l in range(1, 5):  # l = 1, 2, 3, 4
                # For each equation, sum over q = 1, 2, 3, 4
                eq_sum = 0.0

                for q in range(1, 5):  # q = 1, 2, 3, 4
                    # Variable index for x(i+q) in SIF (1-based)
                    # In 0-based: x[i+q-1]
                    var_idx = i + q - 1

                    # For sine term: a = -l * q^2
                    a_sin = -float(l) * float(q * q)
                    sin_term = a_sin * jnp.sin(x[var_idx])

                    # For cosine term: a = l^2 * q
                    a_cos = float(l * l) * float(q)
                    cos_term = a_cos * jnp.cos(x[var_idx])

                    eq_sum += sin_term + cos_term

                # Residual: sum - Y[l-1]
                residual = eq_sum - Y[l - 1]
                residuals.append(residual)

            # Move to next block of variables (stride 2)
            i += 2

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
