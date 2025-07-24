"""NIST Data fitting problem MGH10.

Fit model: y = b1 * exp[b2/(x+b3)] + e

Source: Meyer, R. R. (1970). Theoretical and computational aspects of
nonlinear regression. In Nonlinear Programming (J. Rosen, O. Mangasarian,
and K. Ritter, eds.), pp. 465-486. Academic Press, New York.

This is a nonlinear regression problem from the NIST nonlinear regression test set.

SIF input: Nick Gould, Oct 1992.

Classification: SUR2-MN-3-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MGH10LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem MGH10."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points (x, y)
    data_x = jnp.array(
        [
            50.0,
            55.0,
            60.0,
            65.0,
            70.0,
            75.0,
            80.0,
            85.0,
            90.0,
            95.0,
            100.0,
            105.0,
            110.0,
            115.0,
            120.0,
            125.0,
        ]
    )
    data_y = jnp.array(
        [
            34780.0,
            28610.0,
            23650.0,
            19630.0,
            16370.0,
            13720.0,
            11540.0,
            9744.0,
            8261.0,
            7030.0,
            6005.0,
            5147.0,
            4427.0,
            3820.0,
            3307.0,
            2872.0,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 3

    @property
    def y0(self):
        """Initial guess."""
        # Starting values to match pycutest
        return jnp.array([2.0, 400000.0, 25000.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the sum of squares objective function."""
        del args  # Not used

        b1, b2, b3 = y[0], y[1], y[2]

        # Model: y = b1 * exp[b2/(x+b3)]
        model_values = b1 * jnp.exp(b2 / (self.data_x + b3))

        # Sum of squared residuals
        residuals = self.data_y - model_values
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """NIST certified parameter values."""
        # Certified values from NIST
        return jnp.array([0.00560963647, 6181.3463463, 345.22363462])

    @property
    def expected_objective_value(self):
        """NIST certified objective value."""
        # Standard deviation of residuals: 87.9458 with 13 degrees of freedom
        return jnp.array(87.9458**2 * 13)  # Sum of squared residuals
