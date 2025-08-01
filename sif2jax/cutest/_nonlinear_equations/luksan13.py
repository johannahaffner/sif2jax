import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN13(AbstractNonlinearEquations):
    """Problem 13 (chained and modified HS48) from Luksan.

    This is a system of nonlinear equations from the paper:
    L. Luksan
    "Hybrid methods in large sparse nonlinear least squares"
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    classification NOR2-AN-V-V
    """

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

        # Vectorized computation
        # Create indices for all blocks at once
        i_indices = jnp.arange(s) * 3  # [0, 3, 6, ..., 3*(s-1)]

        # Extract all variables needed for vectorized operations
        x0 = x[i_indices]  # x[i] for each block
        x1 = x[i_indices + 1]  # x[i+1] for each block
        x2 = x[i_indices + 2]  # x[i+2] for each block
        x3 = x[i_indices + 3]  # x[i+3] for each block
        x4 = x[i_indices + 4]  # x[i+4] for each block

        # Compute all 7 types of residuals for all blocks at once
        # E(k): -10*x(i+1) + 10*x(i)^2
        res1 = -10.0 * x1 + 10.0 * x0 * x0

        # E(k+1): -10*x(i+2) + 10*x(i+1)^2
        res2 = -10.0 * x2 + 10.0 * x1 * x1

        # E(k+2): (x(i+2) - x(i+3))^2
        res3 = (x2 - x3) ** 2

        # E(k+3): (x(i+3) - x(i+4))^2
        res4 = (x3 - x4) ** 2

        # E(k+4): x(i) + x(i+2) + x(i+1)^2 - 30
        res5 = x0 + x2 + x1 * x1 - 30.0

        # E(k+5): x(i+1) + x(i+3) - x(i+2)^2 - 10
        res6 = x1 + x3 - x2 * x2 - 10.0

        # E(k+6): x(i) * x(i+4) - 10
        res7 = x0 * x4 - 10.0

        # Stack all residuals in the correct order
        # Each block contributes 7 residuals in sequence
        residuals = jnp.stack(
            [res1, res2, res3, res4, res5, res6, res7], axis=1
        ).flatten()

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
