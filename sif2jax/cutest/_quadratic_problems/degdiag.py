import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class DEGDIAG(AbstractBoundedMinimisation):
    """A degenerate bound constrained convex quadratic program with a diagonal Hessian.

    SIF input: Nick Gould, August 2011

    classification QBR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n_minus_1: int = 100000  # The number of variables - 1

    @property
    def n(self):
        """Number of variables."""
        return self.n_minus_1 + 1

    @property
    def y0(self):
        """Initial guess - all 2.0."""
        return jnp.full(self.n, 2.0)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: (1/2) x^T x.

        Simple diagonal quadratic with no linear term.
        """
        del args
        return 0.5 * jnp.sum(y**2)

    @property
    def bounds(self):
        """Variable bounds: x_i >= i/(n+1)."""
        i_vals = jnp.arange(self.n)
        i_vals = jnp.asarray(i_vals, dtype=self.y0.dtype)
        lower = i_vals / (self.n_minus_1 + 1.0)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result: x_i = i/(n+1)."""
        i_vals = jnp.arange(self.n)
        i_vals = jnp.asarray(i_vals, dtype=self.y0.dtype)
        return i_vals / (self.n_minus_1 + 1.0)

    @property
    def expected_objective_value(self):
        """Expected objective value at solution."""
        x_opt = self.expected_result
        return 0.5 * jnp.sum(x_opt**2)
