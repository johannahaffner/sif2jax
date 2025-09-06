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
        """Compute the objective function (vectorized and optimized)."""
        n = self.n
        m = self.m

        # Linear terms in objective: -10*i*x[i] for i=1 to n
        # Pre-compute coefficients and use dot product
        linear_coeffs = -10.0 * jnp.arange(1, n + 1, dtype=y.dtype)
        linear_obj = jnp.dot(linear_coeffs, y)

        # Exponential elements: exp(0.1 * (i/m) * x[i] * x[i+1]) for i=1 to m
        # Vectorize all operations to avoid loops
        p_vals = jnp.arange(1, m + 1, dtype=y.dtype) / m
        products = y[:m] * y[1 : m + 1]  # x[i] * x[i+1]
        exp_terms = jnp.sum(jnp.exp(0.1 * p_vals * products))

        # Quadratic elements for variables m+1 to n-1
        # These use x[i] and x[n] (last variable)
        if m < n - 1:  # Only if there are quad terms
            x_last = y[n - 1]
            x_quad = y[m : n - 1]  # Variables from m+1 to n-1
            quad_obj = jnp.sum(4.0 * x_quad * x_quad + x_quad * x_last)
            quad_obj += (
                (n - 1 - m) * 2.0 * x_last * x_last
            )  # x_last appears in each quad term
        else:
            quad_obj = 0.0

        return linear_obj + exp_terms + quad_obj

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
