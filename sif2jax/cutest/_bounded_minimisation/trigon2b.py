import jax
import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class TRIGON2B(AbstractBoundedMinimisation):
    """TRIGON2B problem - SCIPY global optimization benchmark example Trigonometric02.

    Fit: y = (0, sqrt(8)*sin(7*(x_i-0.9)^2), sqrt(6)*sin(14*(x_i-0.9)^2), x_i) + e

    Version with box-constrained feasible region: -500 <= x_i <= 500
    (Note: SIF has two bound sets, but appears to use second: 0 <= x_i <= 1)

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, July 2021

    Classification: SBR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    _n: int = 10

    def __init__(self, n: int = 10):
        self._n = n

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

        # Sum of squares
        total = (
            fa**2
            + jnp.sum(fb_residuals**2)
            + jnp.sum(fc_residuals**2)
            + jnp.sum(fd_residuals**2)
        )
        return total

    @property
    def bounds(self):
        """Bounds: 0 <= x_i <= 1 for all i (using BOUNDS2 from SIF)."""
        n = self.n
        lower = jnp.zeros(n, dtype=jnp.float64)
        upper = jnp.ones(n, dtype=jnp.float64)
        return lower, upper

    @property
    def y0(self):
        """Initial guess: x_i = i/n for i = 1, ..., n."""
        n = self.n
        indices = jnp.arange(1, n + 1, dtype=jnp.float64)
        return indices / n

    @property
    def args(self):
        """No additional arguments required."""
        return None

    @property
    def expected_result(self):
        """Expected result - solution with all variables at optimal values."""
        # The optimal solution is not explicitly given in the SIF file
        return jnp.full(self.n, 0.9, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the optimal solution."""
        # Should be close to 1.0 (FA residual squared) when x_i = 0.9
        return jnp.array(1.0)
