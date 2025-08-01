import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class TRIGON1B(AbstractBoundedMinimisation):
    """TRIGON1B problem - SCIPY global optimization benchmark example Trigonometric01.

    Fit: y = sum_{j=1}^{n} cos(x_j) + i (cos(x_i) + sin(x_i)) + e

    Version with box-constrained feasible region: 0 <= x_i <= π

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

        # Compute residuals F(i) for i = 1, ..., n
        def compute_residual(i):
            # F(i) = sum cos(x_j) + (i+1) * (cos(x_i) + sin(x_i)) - (n + i + 1)
            cos_sum = jnp.sum(jnp.cos(x))
            i_float = inexact_asarray(i)
            individual_term = (i_float + 1) * (jnp.cos(x[i]) + jnp.sin(x[i]))
            target = float(n) + i_float + 1
            return cos_sum + individual_term - target

        residuals = jax.vmap(compute_residual)(jnp.arange(n))
        return jnp.sum(residuals**2)

    @property
    def bounds(self):
        """Bounds: 0 <= x_i <= π for all i."""
        n = self.n
        lower = jnp.zeros(n, dtype=jnp.float64)
        upper = jnp.full(n, jnp.pi, dtype=jnp.float64)
        return lower, upper

    @property
    def y0(self):
        """Initial guess: all variables set to 0.1."""
        return jnp.full(self.n, 0.1, dtype=jnp.float64)

    @property
    def args(self):
        """No additional arguments required."""
        return None

    @property
    def expected_result(self):
        """Expected result - solution with all variables at optimal values."""
        # The optimal solution is not explicitly given in the SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the optimal solution."""
        # The optimal objective value is 0.0 according to the SIF comments
        return jnp.array(0.0)
