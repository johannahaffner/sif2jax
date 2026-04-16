import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN15LS(AbstractUnconstrainedMinimisation):
    """Problem 15 (sparse signomial) from Luksan.

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
        Each residual is the sum over p=1,2,3 of signomial terms minus the data value.
        """
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([35.8, 11.2, 6.2, 4.4], dtype=x.dtype)

        # Stride-2 slices for the 4 variables per block
        x1 = x[: 2 * s : 2]
        x2 = x[1 : 2 * s + 1 : 2]
        x3 = x[2 : 2 * s + 2 : 2]
        x4 = x[3 : 2 * s + 3 : 2]

        # Signomial: x1 * x2^2 * x3^3 * x4^4, shape (s,)
        signom = x1 * (x2**2) * (x3**3) * (x4**4)

        # |signom| for fractional powers
        abs_signom = signom * jnp.where(signom > 0, 1.0, -1.0)  # (s,)

        # p = [1,2,3], l = [1,2,3,4] — keep as small 1D vectors, broadcast
        p = jnp.array([1.0, 2.0, 3.0], dtype=x.dtype)  # (3,)
        l = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=x.dtype)  # (4,)

        # p^2/l: (3,4) via outer ops, then broadcast over s
        p2ol = (p[:, None] ** 2) / l[None, :]  # (3, 4)
        pli = 1.0 / (p[:, None] * l[None, :])  # (3, 4)

        # F[p,l,j] = p2ol[p,l] * |signom[j]|^pli[p,l]
        # Broadcast: (3,4,1) * (1,1,s)^(3,4,1) -> (3,4,s)
        F = p2ol[:, :, None] * (abs_signom[None, None, :] ** pli[:, :, None])

        # Sum over p -> (4, s), subtract Y -> residuals
        residuals = (jnp.sum(F, axis=0) - Y[:, None]).T.flatten()

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
