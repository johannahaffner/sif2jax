"""A problem involving exponential and linear terms.

Source: Ph. Toint, 1992.
        minor correction by Ph. Shott, Jan 1995.

SIF input: Ph. Toint, 1992.

Classification: OBR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class EXPLIN(AbstractBoundedMinimisation):
    """A problem involving exponential and linear terms."""

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
        """Bounds on the variables."""
        # From SIF: XU EXPLIN    'DEFAULT'  10.0 - upper bounds of 10.0
        # Pycutest uses 0.0 as default lower bounds for this problem
        lw = jnp.zeros(self.n)
        up = 10.0 * jnp.ones(self.n)
        return lw, up

    def objective(self, y, args):
        """Compute the objective function.

        The objective is:
        sum_{i=1}^N (-10.0 * i) * x_i + sum_{i=1}^M exp(0.1 * x_i * x_{i+1})
        """
        del args  # Not used

        # Linear part: sum_{i=1}^N (-10.0 * i) * x_i
        # Vectorized computation: coefficients are [-10, -20, -30, ..., -10*N]
        coefficients = -10.0 * jnp.arange(1, self.N + 1, dtype=y.dtype)
        linear_part = jnp.dot(coefficients, y)

        # Exponential part: sum_{i=1}^M exp(0.1 * x_i * x_{i+1})
        # Vectorized computation using consecutive pairs
        exp_part = jnp.sum(jnp.exp(0.1 * y[: self.M] * y[1 : self.M + 1]))

        return linear_part + exp_part

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
