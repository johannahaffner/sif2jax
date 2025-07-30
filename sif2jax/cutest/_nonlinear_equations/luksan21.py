"""Problem 21 (modified discrete boundary value) from Luksan.

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


class LUKSAN21(AbstractNonlinearEquations):
    """Problem 21 (modified discrete boundary value) from Luksan."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 100  # Number of variables (default from SIF)

    @property
    def m(self) -> int:
        """Number of equations: M = N."""
        return self.n

    @property
    def y0(self) -> Array:
        """Initial guess: x(i) = i*h*(i*h - 1) where h = 1/(n+1)."""
        h = 1.0 / (self.n + 1)
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        ih = i_vals * h
        return ih * (ih - 1.0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        del args  # Not used

        x = y
        n = self.n
        m = self.m

        # Parameters
        h = 1.0 / (n + 1)
        h2 = h * h
        h2_half = 0.5 * h2

        # Initialize residual vector
        residuals = jnp.zeros(m, dtype=jnp.float64)

        # Equation 1: 2*x(1) - x(2) + h^2/2 * (x(1) + h + 1)^3 + 1
        hi = h  # h * i where i = 1
        xhip1 = x[0] + hi + 1.0
        residuals = residuals.at[0].set(2.0 * x[0] - x[1] + h2_half * (xhip1**3) + 1.0)

        # Equations 2 to M-1: 2*x(i) - x(i-1) - x(i+1) + h^2/2 * (x(i) + h*i + 1)^3 + 1
        for i in range(2, m):
            idx = i - 1  # 0-based index
            hi = h * i
            xhip1 = x[idx] + hi + 1.0
            residuals = residuals.at[idx].set(
                2.0 * x[idx] - x[idx - 1] - x[idx + 1] + h2_half * (xhip1**3) + 1.0
            )

        # Equation M: 2*x(n) - x(n-1) + h^2/2 * (x(n) + h*n + 1)^3 + 1
        hi = h * n
        xhip1 = x[n - 1] + hi + 1.0
        residuals = residuals.at[m - 1].set(
            2.0 * x[n - 1] - x[n - 2] + h2_half * (xhip1**3) + 1.0
        )

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
