import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class EGGCRATEB(AbstractBoundedMinimisation):
    """SCIPY global optimization benchmark example EGGCRATE.

    Fit: (x, y, 5*sin(x), 5*sin(y)) + e = 0

    Version with box-constrained feasible region.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, July 2021

    Classification: SBR2-MN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        del args
        x, y_var = y

        # From SIF structure: sum of squares of (x, y, 5*sin(x), 5*sin(y))
        # F1: x^2
        # F2: y^2
        # F3: (5*sin(x))^2
        # F4: (5*sin(y))^2

        term1 = x**2
        term2 = y_var**2
        term3 = (5.0 * jnp.sin(x)) ** 2
        term4 = (5.0 * jnp.sin(y_var)) ** 2

        return term1 + term2 + term3 + term4

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.array([1.0, 2.0]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        # Both x and y: -5.0 <= var <= 5.0
        lower_bounds = jnp.array([-5.0, -5.0])
        upper_bounds = jnp.array([5.0, 5.0])
        return lower_bounds, upper_bounds

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Global minimum is at (0, 0)
        return jnp.array([0.0, 0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Minimum value is 0.0
        return jnp.array(0.0)
