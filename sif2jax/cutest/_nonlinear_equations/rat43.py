"""RAT43 - NIST nonlinear equations problem.

NIST Data fitting problem RAT43 given as an inconsistent set of
nonlinear equations.

Fit: y = b1 / ((1+exp[b2-b3*x])**(1/b4)) + e

Source: Problem from the NIST nonlinear regression test set
  http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

Reference: Ratkowsky, D.A. (1983).
  Nonlinear Regression Modeling.
  New York, NY: Marcel Dekker, pp. 62 and 88.

SIF input: Nick Gould and Tyrone Rees, Oct 2015

Classification: NOR2-MN-4-15
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class RAT43(AbstractNonlinearEquations):
    """RAT43 - NIST nonlinear equations problem."""

    _N: int = 4  # Number of variables
    _M: int = 15  # Number of data points
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points (from SIF file)
    x_data = jnp.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ]
    )
    y_data = jnp.array(
        [
            16.08,
            33.83,
            65.80,
            97.20,
            191.55,
            326.20,
            386.87,
            520.53,
            590.03,
            651.92,
            724.93,
            699.56,
            689.96,
            637.56,
            717.41,
        ]
    )

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
            return jnp.array([100.0, 10.0, 1.0, 1.0])
        else:
            # START2
            return jnp.array([700.0, 5.0, 0.75, 1.3])

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

        The equations are F(i) = b1 / ((1 + exp(b2 - b3*x_i))**(1/b4)) - y_i = 0
        """
        b1, b2, b3, b4 = y

        # Compute the model predictions for each data point
        exp_term = jnp.exp(b2 - b3 * self.x_data)
        denominator = (1.0 + exp_term) ** (1.0 / b4)
        predictions = b1 / denominator

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
