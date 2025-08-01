import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN21LS(AbstractUnconstrainedMinimisation):
    """Problem 21 (modified discrete boundary value) from Luksan.

    This is a least squares problem from the paper:
    L. Luksan
    "Hybrid methods in large sparse nonlinear least squares"
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    least-squares version

    classification SUR2-AN-V-0
    """

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

        # Parameters
        h = 1.0 / (n + 1)
        h2 = h * h
        h2_half = 0.5 * h2

        # Create index arrays for vectorized computation
        i_vals = jnp.arange(
            1, n + 1, dtype=x.dtype
        )  # 1 to n (1-indexed as in original)
        hi_vals = h * i_vals  # h * i for all i

        # Compute cubic terms for all equations: (x(i) + h*i + 1)^3
        xhip1_vals = x + hi_vals + 1.0  # x(i) + h*i + 1 for all i
        cubic_terms = h2_half * (xhip1_vals**3)  # h^2/2 * (x(i) + h*i + 1)^3

        # Handle the three cases using vectorized operations
        # Case 1: First equation (i=1): 2*x(1) - x(2) + cubic_term + 1
        # Case 2: Middle eqs (2 <= i <= n-1): 2*x(i) - x(i-1) - x(i+1) + cubic + 1
        # Case 3: Last equation (i=n): 2*x(n) - x(n-1) + cubic_term + 1

        # Initialize residuals with 2*x(i) + cubic_term + 1 for all i
        residuals = 2.0 * x + cubic_terms + 1.0

        # Subtract x(i+1) from equations 1 to n-1
        residuals = residuals.at[:-1].add(-x[1:])

        # Subtract x(i-1) from equations 2 to n
        residuals = residuals.at[1:].add(-x[:-1])

        # Sum of squares (L2 group type in SIF)
        return jnp.sum(residuals**2)

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
