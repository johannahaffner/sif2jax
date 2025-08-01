import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN21(AbstractNonlinearEquations):
    """Problem 21 (modified discrete boundary value) from Luksan.

    This is a system of nonlinear equations from the paper:
    L. Luksan
    "Hybrid methods in large sparse nonlinear least squares"
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    classification NOR2-AN-V-V
    """

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

        # Initialize all with 2*x(i) + cubic_term + 1
        residuals = 2.0 * x + cubic_terms + 1.0

        # Subtract x(i-1) for equations 2 to n (indices 1 to n-1)
        residuals = residuals.at[1:].add(-x[:-1])

        # Subtract x(i+1) for equations 1 to n-1 (indices 0 to n-2)
        residuals = residuals.at[:-1].add(-x[1:])

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
