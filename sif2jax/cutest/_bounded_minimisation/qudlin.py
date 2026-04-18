import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


# TODO(claude): move to folder with bounded minimisation problems. Create new subclass
# AbstractBoundedQuadraticProblem in sif2jax/_problem.py. Then test, commit and push.
class QUDLIN(AbstractBoundedMinimisation):
    """A simple quadratic programming problem.

    The objective consists of linear terms and product terms x_i * x_{i+1}.

    SIF input: unknown.
    minor correction by Ph. Shott, Jan 1995.

    classification QBR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Number of variables
    m: int = 2500  # Number of product terms

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        Linear terms: -10 * i * x_i for i=1 to n
        Quadratic terms: x_i * x_{i+1} for i=1 to m
        """
        del args
        coeffs = jnp.arange(-10, -10 * self.n - 1, -10, dtype=y.dtype)
        m = min(self.m, self.n - 1)
        return jnp.dot(coeffs, y) + jnp.sum(y[:m] * y[1 : m + 1])

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 10."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, 10.0)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 (from SIF file)."""
        return jnp.array(0.0)
