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

        # Signomial: x1 * x2^2 * x3^3 * x4^4, shape (s,)
        x1 = x[: 2 * s : 2]
        x_sq = x**2
        x2_2 = x_sq[1 : 2 * s + 1 : 2]
        x3_3 = x[2 : 2 * s + 2 : 2] * x_sq[2 : 2 * s + 2 : 2]
        x4_4 = x_sq[3 : 2 * s + 3 : 2] ** 2
        signom = x1 * x2_2 * x3_3 * x4_4

        a = jnp.abs(signom)  # (s,)

        # Compute roots via sqrt/cbrt chains instead of generic pow.
        # Needed exponents: 1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/9, 1/12
        r2 = jnp.sqrt(a)  # a^(1/2)
        r3 = jnp.cbrt(a)  # a^(1/3)
        r4 = jnp.sqrt(r2)  # a^(1/4)
        r6 = jnp.sqrt(r3)  # a^(1/6)
        r8 = jnp.sqrt(r4)  # a^(1/8)
        r9 = jnp.cbrt(r3)  # a^(1/9)
        r12 = jnp.sqrt(r6)  # a^(1/12)

        # F[p,l] = (p^2/l) * a^(1/(p*l)), summed over p=1,2,3 for each l
        eq1 = a + 4.0 * r2 + 9.0 * r3  # l=1
        eq2 = 0.5 * r2 + 2.0 * r4 + 4.5 * r6  # l=2
        eq3 = (1.0 / 3.0) * r3 + (4.0 / 3.0) * r6 + 3.0 * r9  # l=3
        eq4 = 0.25 * r4 + r8 + 2.25 * r12  # l=4

        # Residuals: eq_l - Y_l for each block j, flattened as (j, l)
        residuals = jnp.stack(
            [eq1 - Y[0], eq2 - Y[1], eq3 - Y[2], eq4 - Y[3]], axis=1
        ).flatten()

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
