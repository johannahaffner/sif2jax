"""Problem 21 (modified discrete boundary value) from Luksan.

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


class LUKSAN21LS(AbstractUnconstrainedMinimisation):
    """Problem 21 (modified discrete boundary value) from Luksan - LS version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 100  # Number of variables (default from SIF)

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

    def objective(self, y: Array, args) -> Array:
        """Compute the least squares objective function.

        The objective is the sum of squares of M = N residuals.
        The residuals come from a discretized boundary value problem:
        - E(1): 2*x(1) - x(2) + h^2/2 * (x(1) + h + 1)^3 + 1
        - E(i): 2*x(i) - x(i-1) - x(i+1) + h^2/2 * (x(i) + h*i + 1)^3 + 1  (i=2,...,N-1)
        - E(N): 2*x(N) - x(N-1) + h^2/2 * (x(N) + h*N + 1)^3 + 1
        """
        del args  # Not used

        x = y
        n = self.n
        m = n  # Number of residuals equals number of variables

        # Parameters
        h = 1.0 / (n + 1)
        h2 = h * h
        h2_half = 0.5 * h2

        # Initialize residual vector
        residuals = []

        # Equation 1: 2*x(1) - x(2) + h^2/2 * (x(1) + h + 1)^3 + 1
        hi = h  # h * i where i = 1
        xhip1 = x[0] + hi + 1.0
        res1 = 2.0 * x[0] - x[1] + h2_half * (xhip1**3) + 1.0
        residuals.append(res1)

        # Equations 2 to M-1: 2*x(i) - x(i-1) - x(i+1) + h^2/2 * (x(i) + h*i + 1)^3 + 1
        for i in range(2, m):
            idx = i - 1  # 0-based index
            hi = h * i
            xhip1 = x[idx] + hi + 1.0
            res = 2.0 * x[idx] - x[idx - 1] - x[idx + 1] + h2_half * (xhip1**3) + 1.0
            residuals.append(res)

        # Equation M: 2*x(n) - x(n-1) + h^2/2 * (x(n) + h*n + 1)^3 + 1
        hi = h * n
        xhip1 = x[n - 1] + hi + 1.0
        res_n = 2.0 * x[n - 1] - x[n - 2] + h2_half * (xhip1**3) + 1.0
        residuals.append(res_n)

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
