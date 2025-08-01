"""Problem 15 (sparse signomial) from Luksan.

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


class LUKSAN15(AbstractNonlinearEquations):
    """Problem 15 (sparse signomial) from Luksan."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 49  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 2*S + 2."""
        return 2 * self.s + 2

    @property
    def m(self) -> int:
        """Number of equations: 4*S."""
        return 4 * self.s

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

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        del args  # Not used

        x = y
        s = self.s
        m = self.m

        # Data values
        Y = jnp.array([35.8, 11.2, 6.2, 4.4], dtype=jnp.float64)

        # Initialize residual vector
        residuals = jnp.zeros(m, dtype=jnp.float64)

        # Loop over S blocks
        k = 0  # Equation index (0-based)
        i = 0  # Variable index (0-based)

        for j in range(s):
            # Each block contributes 4 equations
            # Variables involved: x[i+1], x[i+2], x[i+3], x[i+4] (1-based in SIF)
            # In 0-based: x[i], x[i+1], x[i+2], x[i+3]

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

                # Set residual: sum - Y[l-1]
                residuals = residuals.at[k].set(eq_sum - Y[l - 1])
                k += 1

            # Move to next block of variables (stride 2)
            i += 2

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
