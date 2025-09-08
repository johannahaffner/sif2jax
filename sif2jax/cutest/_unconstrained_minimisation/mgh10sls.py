import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MGH10SLS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem MGH10 (scaled least squares version).

    Fit model: y = 0.01 * b1 * exp[ 1000 * b2 / (x + 100 * b3 ) ] + e

    Source: Meyer, R. R. (1970). Theoretical and computational aspects of
    nonlinear regression. In Nonlinear Programming (J. Rosen, O. Mangasarian,
    and K. Ritter, eds.), pp. 465-486. Academic Press, New York.

    This is a nonlinear regression problem from the NIST nonlinear regression test set.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015
    Least-squares version of MGH10S.SIF, Nick Gould, Jan 2020

    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

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
        return jnp.array(3)

    @property
    def y0(self):
        """Initial guess."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([200.0, 400.0, 250.0])
        else:
            # START2
            return jnp.array([2.0, 4.0, 2.5])

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the sum of squares objective function."""
        del args  # Not used

        b1, b2, b3 = y[0], y[1], y[2]

        # Model: y = 0.01 * b1 * exp[ 1000 * b2 / (x + 100 * b3 ) ]
        model_values = 0.01 * b1 * jnp.exp(1000.0 * b2 / (self.data_x + 100.0 * b3))

        # Sum of squared residuals
        residuals = self.data_y - model_values
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # The SIF file doesn't provide a solution
        return None
