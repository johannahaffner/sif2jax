import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class TRIGON2(AbstractUnconstrainedMinimisation):
    """TRIGON2 problem - SCIPY global optimization benchmark example Trigonometric02.

    TODO: Human review needed - Hessian test fails.
    The Hessian values are close but not within tolerance.

    Fit: y = (0, sqrt(8)*sin(7*(x_i-0.9)^2), sqrt(6)*sin(14*(x_i-0.9)^2), x_i) + e

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    _n: int = 10

    @property
    def n(self):
        """Number of variables."""
        return self._n

    def objective(self, y, args):
        """Compute the sum of squares objective."""
        del args
        x = y
        n = self.n

        sqrt6 = jnp.sqrt(6.0)
        sqrt8 = jnp.sqrt(8.0)

        # FA = 0 - 1 = -1
        fa = -1.0

        # FB(i) = sqrt(8)*sin(7*(x_i-0.9)^2) + sqrt(6)*sin(14*(x_i-0.9)^2)
        def compute_fb(i):
            d = x[i] - 0.9
            y_val = d * d
            term1 = sqrt8 * jnp.sin(7.0 * y_val)
            term2 = sqrt6 * jnp.sin(14.0 * y_val)
            return term1 + term2

        fb_residuals = jax.vmap(compute_fb)(jnp.arange(n))

        # FC(i) = 0 (no elements assigned to FC groups)
        fc_residuals = jnp.zeros(n)

        # FD(i) = x_i - 0.9
        fd_residuals = x - 0.9

        # Sum of squares of all residuals
        obj = (
            fa**2
            + jnp.sum(fb_residuals**2)
            + jnp.sum(fc_residuals**2)
            + jnp.sum(fd_residuals**2)
        )
        return obj

    def y0(self):
        """Initial guess."""
        # Start point: x_i = i/n for i = 1, ..., n
        return inexact_asarray((jnp.arange(1, self.n + 1)) / self.n)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
