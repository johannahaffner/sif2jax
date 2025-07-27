import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class EXPQUAD(AbstractBoundedMinimisation):
    """A problem with mixed exponential and quadratic terms.

    SIF input: Ph. Toint, 1992.
               minor correction by Ph. Shott, Jan 1995.

    classification OBR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimensions
    n: int = eqx.field(default=1200, init=False)
    m: int = eqx.field(default=100, init=False)

    def objective(self, y, args):
        """Compute the objective function (vectorized)."""
        n = self.n
        m = self.m

        # Linear terms in objective: -10*i*x[i] for i=1 to n
        i_vals = jnp.arange(1, n + 1, dtype=jnp.float64)
        linear_obj = jnp.sum(-10.0 * i_vals * y)

        # Element contributions
        # First m elements: EXP type with parameters
        # From AMPL: exp(0.1*i*m*x[i]*x[i+1]) but SIF has i/m as parameter
        # So it's exp(0.1*i/m*x[i]*x[i+1])
        i_vals_exp = jnp.arange(1, m + 1, dtype=jnp.float64)
        p_vals = i_vals_exp / m
        x_vals = y[:m]
        y_vals = y[1 : m + 1]
        exp_terms = jnp.exp(0.1 * p_vals * x_vals * y_vals)

        # Elements m+1 to n-1: QUAD type
        # These use x[i] and x[n-1] (last variable)
        x_last = y[n - 1]
        x_quad = y[m : n - 1]
        quad_terms = 4.0 * x_quad * x_quad + 2.0 * x_last * x_last + x_quad * x_last

        return linear_obj + jnp.sum(exp_terms) + jnp.sum(quad_terms)

    @property
    def y0(self):
        """Starting point."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        n = self.n
        m = self.m

        # First m variables are bounded in [0, 10]
        # Rest are unbounded (-inf, inf)
        lower = jnp.full(n, -jnp.inf)
        lower = lower.at[:m].set(0.0)
        upper = jnp.full(n, jnp.inf)
        upper = upper.at[:m].set(10.0)

        return (lower, upper)

    @property
    def expected_result(self):
        """Expected solution (not provided)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value."""
        # Lower bound is 0.0
        return None
