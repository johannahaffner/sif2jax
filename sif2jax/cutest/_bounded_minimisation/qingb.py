import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class QINGB(AbstractBoundedMinimisation):
    """SCIPY global optimization benchmark example Qing with box constraints.

    This is a sum-of-squares problem with the form:
    f(x) = sum((x_i^2 - i)^2) for i=1 to n

    Box constraints: -500 <= x_i <= 500 for all i

    The global minimum is at x_i = sqrt(i) with f* = 0.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/
    benchmarks/go_benchmark_functions

    Version with box-constrained feasible region.

    SIF input: Nick Gould, Jan 2020

    classification: SBR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5  # Default number of variables (SIF file default)

    @property
    def y0(self):
        """Starting point from SIF file - all variables initialized to 1.0."""
        return jnp.ones(self.n)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        The objective is sum((x_i^2 - i)^2) for i=1 to n.
        This is vectorized for efficiency, especially important for large n.
        """
        # Create index array [1, 2, ..., n]
        indices = jnp.arange(1, self.n + 1, dtype=y.dtype)

        # Compute (x_i^2 - i)^2 for all i
        residuals = y * y - indices

        # Sum of squares
        return jnp.sum(residuals * residuals)

    @property
    def bounds(self):
        """Variable bounds: -500 <= x_i <= 500 for all i."""
        lower = jnp.full(self.n, -500.0)
        upper = jnp.full(self.n, 500.0)
        return lower, upper

    @property
    def expected_result(self):
        """The optimal solution is x_i = sqrt(i)."""
        indices = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        return jnp.sqrt(indices)

    @property
    def expected_objective_value(self):
        """Global minimum value is 0.0."""
        return jnp.array(0.0)
