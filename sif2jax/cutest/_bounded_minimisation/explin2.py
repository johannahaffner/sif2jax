"""A problem involving exponential and linear terms with bounds.

This is a variant of EXPLIN with variable bounds. The problem combines
linear terms with exponential coupling terms between consecutive variables.

Source: Ph. Toint, 1992.

SIF input: Ph. Toint, 1992.

Classification: OBR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class EXPLIN2(AbstractBoundedMinimisation):
    """A bounded problem involving exponential and linear terms."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 1200  # Number of variables
    M: int = 100  # Number of exponential terms

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def y0(self):
        """Initial guess (default starting point)."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables (0 <= x <= 10)."""
        lw = jnp.zeros(self.n)
        up = 10.0 * jnp.ones(self.n)
        return lw, up

    def objective(self, y, args):
        """Compute the objective function.

        The objective is:
        sum_{i=1}^N (-10.0 * i) * x_i + sum_{i=1}^M exp(0.1 * (i/M) * x_i * x_{i+1})
        """
        del args  # Not used

        # Linear part: sum_{i=1}^N (-10.0 * i) * x_i
        # Vectorized computation: coefficients are [-10, -20, -30, ..., -10*N]
        coefficients = -10.0 * jnp.arange(1, self.N + 1, dtype=y.dtype)
        linear_part = jnp.dot(coefficients, y)

        # Exponential part: sum_{i=1}^M exp(0.1 * (i/M) * x_i * x_{i+1})
        # Vectorized computation over all M terms
        i_vals = jnp.arange(1, self.M + 1, dtype=y.dtype)
        p_vals = i_vals / self.M
        exp_terms = jnp.exp(0.1 * p_vals * y[: self.M] * y[1 : self.M + 1])
        exp_part = jnp.sum(exp_terms)

        return linear_part + exp_part

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
