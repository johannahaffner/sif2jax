import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class TRIGON1(AbstractUnconstrainedMinimisation):
    """TRIGON1 problem - SCIPY global optimization benchmark example Trigonometric01.

    Fit: y = sum_{j=1}^{n} cos(x_j) + i (cos(x_i) + sin(x_i)) + e

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

    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.full(self.n, 0.1))

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
