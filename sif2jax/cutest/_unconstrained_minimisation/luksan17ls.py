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

        # Stack 4 variables per block into (4, s), compute sin/cos once
        x_q = jnp.stack(
            [
                y[: 2 * s : 2],
                y[1 : 2 * s + 1 : 2],
                y[2 : 2 * s + 2 : 2],
                y[3 : 2 * s + 3 : 2],
            ]
        )  # (4, s)
        sin_q = jnp.sin(x_q)  # (4, s)
        cos_q = jnp.cos(x_q)  # (4, s)

        # Factor: a_sin[l,q] = -l*q^2, a_cos[l,q] = l^2*q
        # sum_q(a_sin*sin + a_cos*cos) = -l * sum_q(q^2*sin) + l^2 * sum_q(q*cos)
        q_sq = jnp.array([1.0, 4.0, 9.0, 16.0], dtype=y.dtype)  # q^2
        q_vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=y.dtype)  # q
        sq2s = q_sq @ sin_q  # sum_q(q^2 * sin), shape (s,)
        sqc = q_vec @ cos_q  # sum_q(q * cos), shape (s,)

        # eq[l] = -l * sq2s + l^2 * sqc - Y[l]
        l_vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=y.dtype)
        eq = -l_vec[:, None] * sq2s[None, :] + (l_vec**2)[:, None] * sqc[None, :]

        residuals = (eq - Y[:, None]).T.flatten()
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
