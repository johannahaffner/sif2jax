import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MGH17LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem MGH17 (least squares version).

    Fit model: y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5] + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Osborne, M. R. (1972).
      Some aspects of nonlinear least squares calculations.
      In Numerical Methods for Nonlinear Optimization, Lootsma (Ed).
      New York, NY:  Academic Press, pp. 171-189.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-5-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points (x, y)
    data_x = jnp.array(
        [
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            160.0,
            170.0,
            180.0,
            190.0,
            200.0,
            210.0,
            220.0,
            230.0,
            240.0,
            250.0,
            260.0,
            270.0,
            280.0,
            290.0,
            300.0,
            310.0,
            320.0,
        ]
    )
    data_y = jnp.array(
        [
            0.844,
            0.908,
            0.932,
            0.936,
            0.925,
            0.908,
            0.881,
            0.850,
            0.818,
            0.784,
            0.751,
            0.718,
            0.685,
            0.658,
            0.628,
            0.603,
            0.580,
            0.558,
            0.538,
            0.522,
            0.506,
            0.490,
            0.478,
            0.467,
            0.457,
            0.448,
            0.438,
            0.431,
            0.424,
            0.420,
            0.414,
            0.411,
            0.406,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(5)

    @property
    def y0(self):
        """Initial guess."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([50.0, 150.0, -100.0, 1.0, 2.0])
        else:
            # START2
            return jnp.array([0.5, 1.5, -1.0, 0.01, 0.02])

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the sum of squares objective function."""
        del args  # Not used

        b1, b2, b3, b4, b5 = y[0], y[1], y[2], y[3], y[4]

        # Model: y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5]
        model_values = (
            b1 + b2 * jnp.exp(-self.data_x * b4) + b3 * jnp.exp(-self.data_x * b5)
        )

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
