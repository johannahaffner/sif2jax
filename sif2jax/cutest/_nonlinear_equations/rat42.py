"""RAT42 - NIST nonlinear equations problem.

NIST Data fitting problem RAT42 given as an inconsistent set of
nonlinear equations.

Fit: y = b1 / (1+exp[b2-b3*x]) + e

Source: Problem from the NIST nonlinear regression test set
  http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

Reference: Ratkowsky, D.A. (1983).
  Nonlinear Regression Modeling.
  New York, NY: Marcel Dekker, pp. 61 and 88.

SIF input: Nick Gould and Tyrone Rees, Oct 2015

Classification: NOR2-MN-3-9
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class RAT42(AbstractNonlinearEquations):
    """RAT42 - NIST nonlinear equations problem."""

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
    def m(self):
        """Number of equations."""
        return self._M

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
    def bounds(self):
        """No explicit bounds."""
        return None

    @property
    def args(self):
        """No additional arguments."""
        return None

    def constraint(self, y):
        """Compute the nonlinear equations.

        The equations are F(i) = b1 / (1 + exp(b2 - b3*x_i)) - y_i = 0
        """
        b1, b2, b3 = y

        # Compute the model predictions for each data point
        exp_term = jnp.exp(b2 - b3 * self.x_data)
        predictions = b1 / (1.0 + exp_term)

        # Return residuals (predictions - data)
        residuals = predictions - self.y_data

        # For nonlinear equations, we return (equality, inequality) constraints
        # All constraints are equalities here
        return residuals, None

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value."""
        return jnp.array(0.0)  # Nonlinear equations should have residuals = 0
