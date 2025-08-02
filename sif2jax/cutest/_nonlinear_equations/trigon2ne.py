import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class TRIGON2NE(AbstractNonlinearEquations):
    """TRIGON2NE problem - SCIPY global optimization benchmark example Trigonometric02.

    Fit: y = (0, sqrt(8)*sin(7*(x_i-0.9)^2), sqrt(6)*sin(14*(x_i-0.9)^2), x_i) + e

    Nonlinear-equation formulation of TRIGON2.SIF

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
        """Number of equations: 3*n + 1."""
        return 3 * self._n + 1

    def constraint(self, y):
        """Compute the nonlinear equations."""
        x = y
        n = self.n

        sqrt6 = jnp.sqrt(6.0)
        sqrt8 = jnp.sqrt(8.0)

        # Vectorized computation
        d = x - 0.9
        y_val = d * d
        fb_vals = sqrt8 * jnp.sin(7.0 * y_val) + sqrt6 * jnp.sin(14.0 * y_val)
        fc_vals = jnp.zeros(n)  # FC(i) = 0
        fd_vals = d  # FD(i) = x_i - 0.9

        # Interleave FB, FC, FD values: FB(0), FC(0), FD(0), FB(1), FC(1), FD(1), ...
        interleaved = jnp.stack([fb_vals, fc_vals, fd_vals], axis=1).flatten()

        # Prepend FA = -1
        equations = jnp.concatenate([jnp.array([-1.0]), interleaved])

        return equations, None  # No inequality constraints

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

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
        # The optimal solution would have x_i = 0.9 to make FD(i) = 0
        return jnp.full(self.n, 0.9, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the optimal solution."""
        # For nonlinear equations, this is typically 0
        return jnp.array(0.0)
