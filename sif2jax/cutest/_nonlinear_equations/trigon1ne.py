import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class TRIGON1NE(AbstractNonlinearEquations):
    """TRIGON1NE problem - SCIPY global optimization benchmark example Trigonometric01.

    Fit: y = sum_{j=1}^{n} cos(x_j) + i (cos(x_i) + sin(x_i)) + e

    Nonlinear-equation formulation of TRIGON1.SIF

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: NOR2-MN-V-V
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

    @property
    def m(self):
        """Number of equations (same as number of variables)."""
        return self._n

    def constraint(self, y):
        """Compute the nonlinear equations."""
        x = y
        n = self.n

        # Compute equations F(i) for i = 1, ..., n
        def compute_equation(i):
            # F(i) = sum cos(x_j) + (i+1) * (cos(x_i) + sin(x_i)) - (n + i + 1)
            cos_sum = jnp.sum(jnp.cos(x))
            i_float = inexact_asarray(i)
            individual_term = (i_float + 1) * (jnp.cos(x[i]) + jnp.sin(x[i]))
            target = float(n) + i_float + 1
            return cos_sum + individual_term - target

        equations = jax.vmap(compute_equation)(jnp.arange(n))
        return equations, None  # No inequality constraints

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

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
        # For nonlinear equations, this is typically 0
        return jnp.array(0.0)
