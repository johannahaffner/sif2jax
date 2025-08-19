"""RAT42LS - NIST least squares problem.

NIST Data fitting problem RAT42.

Fit: y = b1 / (1+exp[b2-b3*x]) + e

Source: Problem from the NIST nonlinear regression test set
  http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

Reference: Ratkowsky, D.A. (1983).
  Nonlinear Regression Modeling.
  New York, NY: Marcel Dekker, pp. 61 and 88.

SIF input: Nick Gould and Tyrone Rees, Oct 2015

Classification: SUR2-MN-3-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class RAT42LS(AbstractUnconstrainedMinimisation):
    """RAT42LS - NIST least squares problem."""

    _N: int = 3  # Number of variables
    _M: int = 9  # Number of data points
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points (from SIF file)
    x_data = jnp.array([9.0, 14.0, 21.0, 28.0, 42.0, 57.0, 63.0, 70.0, 79.0])
    y_data = jnp.array([8.93, 10.80, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08])

    @property
    def n(self):
        """Number of variables."""
        return self._N

    @property
    def y0(self):
        """Initial guess."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([100.0, 1.0, 0.1])
        else:
            # START2
            return jnp.array([75.0, 2.5, 0.07])

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the least squares objective.

        The objective is sum_i (b1 / (1 + exp(b2 - b3*x_i)) - y_i)^2
        """
        del args
        b1, b2, b3 = y

        # Compute the model predictions for each data point
        exp_term = jnp.exp(b2 - b3 * self.x_data)
        predictions = b1 / (1.0 + exp_term)

        # Compute squared residuals
        residuals = predictions - self.y_data

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value - not provided in SIF."""
        return None
