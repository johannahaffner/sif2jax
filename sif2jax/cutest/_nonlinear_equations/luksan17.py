"""Problem 17 (sparse trigonometric) from Luksan.

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


class LUKSAN17(AbstractNonlinearEquations):
    """Problem 17 (sparse trigonometric) from Luksan."""

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
        Y = jnp.array([30.6, 72.2, 124.4, 187.4], dtype=jnp.float64)

        # Initialize residual vector
        residuals = jnp.zeros(m, dtype=jnp.float64)

        # Loop over S blocks
        k = 0  # Equation index (0-based)
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
