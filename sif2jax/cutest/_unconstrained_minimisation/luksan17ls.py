import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN17LS(AbstractUnconstrainedMinimisation):
    """Problem 17 (sparse trigonometric) from Luksan.

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
    s: int = 49  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 2*S + 2."""
        return 2 * self.s + 2

    @property
    def y0(self) -> Array:
        """Initial guess: pattern (-0.8, 1.2, -1.2, 0.8) repeated."""
        pattern = jnp.array([-0.8, 1.2, -1.2, 0.8], dtype=jnp.float64)
        # Repeat pattern to cover all n variables
        full_pattern = jnp.tile(pattern, (self.n + 3) // 4)[: self.n]
        return full_pattern

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

        s = self.s
        Y = jnp.array([30.6, 72.2, 124.4, 187.4], dtype=y.dtype)

        # Stride-2 slices for 4 variables per block
        x1, x2 = y[: 2 * s : 2], y[1 : 2 * s + 1 : 2]
        x3, x4 = y[2 : 2 * s + 2 : 2], y[3 : 2 * s + 3 : 2]

        s1, c1 = jnp.sin(x1), jnp.cos(x1)
        s2, c2 = jnp.sin(x2), jnp.cos(x2)
        s3, c3 = jnp.sin(x3), jnp.cos(x3)
        s4, c4 = jnp.sin(x4), jnp.cos(x4)

        # Factor: sum_q(q^2 * sin(xq)) and sum_q(q * cos(xq))
        sq2s = s1 + 4.0 * s2 + 9.0 * s3 + 16.0 * s4
        sqc = c1 + 2.0 * c2 + 3.0 * c3 + 4.0 * c4

        # Residuals per l: -l*sq2s + l^2*sqc - Y[l]
        eq1 = -sq2s + sqc - Y[0]
        eq2 = -2.0 * sq2s + 4.0 * sqc - Y[1]
        eq3 = -3.0 * sq2s + 9.0 * sqc - Y[2]
        eq4 = -4.0 * sq2s + 16.0 * sqc - Y[3]

        residuals = jnp.stack([eq1, eq2, eq3, eq4], axis=1).flatten()
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
