"""Problem 15 (sparse signomial) from Luksan.

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


class LUKSAN15LS(AbstractUnconstrainedMinimisation):
    """Problem 15 (sparse signomial) from Luksan - least squares version."""

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
        Each residual is the sum over p=1,2,3 of signomial terms minus the data value.
        """
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([35.8, 11.2, 6.2, 4.4], dtype=jnp.float64)

        # Initialize residual vector
        residuals = []

        # Loop over S blocks
        i = 0  # Variable index (0-based)

        for j in range(s):
            # Each block contributes 4 equations
            # Variables involved: x[i], x[i+1], x[i+2], x[i+3]

            for l in range(1, 5):  # l = 1, 2, 3, 4
                # For each equation, sum over p = 1, 2, 3
                eq_sum = 0.0

                for p in range(1, 4):  # p = 1, 2, 3
                    # P2OL = p^2 / l
                    # PLI = 1 / (p * l)
                    p2ol = float(p * p) / float(l)
                    pli = 1.0 / (float(p) * float(l))

                    # Compute signomial term: x1 * x2^2 * x3^3 * x4^4
                    x1 = x[i]
                    x2 = x[i + 1]
                    x3 = x[i + 2]
                    x4 = x[i + 3]

                    signom_val = x1 * (x2**2) * (x3**3) * (x4**4)

                    # Apply sign based on whether signom_val is positive
                    sign_p = jnp.where(signom_val > 0, 1.0, -1.0)
                    p_val = signom_val * sign_p  # This gives |signom_val|

                    # F = p2ol * p^pli
                    f_val = p2ol * (p_val**pli)

                    eq_sum += f_val

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
