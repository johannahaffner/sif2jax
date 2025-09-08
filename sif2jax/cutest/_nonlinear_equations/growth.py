import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class GROWTH(AbstractNonlinearEquations):
    """
    GROWTH - GROWTH problem in 3 variables.

    Fit the observed growth g(n) from Gaussian Elimination
    with complete pivoting to a function of the form

         U1 * n ** ( U2 + LOG(n) * U3 )

    SIF input: Nick Gould, Nov, 1991.

    Classification: NOR2-AN-3-12
    """

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(3)

    def num_residuals(self):
        """Number of residual equations."""
        return jnp.array(12)

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # From SIF: U1 = 100.0, U2 = 0.0 (default), U3 = 0.0 (default)
        return jnp.array([100.0, 0.0, 0.0])

    @property
    def bounds(self):
        """Variable bounds - all variables are free."""
        return None

    def residual(self, y, args):
        """
        Residual functions: fit model to observed data.

        Model: f(n) = U1 * n^(U2 + log(n) * U3)
        Data points: (n, g(n)) for n = 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25

        Residuals: f(n) - g(n) = 0 for each data point
        """
        del args
        u1, u2, u3 = y

        # Data points from SIF file
        n_values = jnp.array(
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 25.0]
        )
        g_values = jnp.array(
            [
                8.0,
                8.4305,
                9.5294,
                10.4627,
                12.0,
                13.0205,
                14.5949,
                16.1078,
                18.0596,
                20.4569,
                24.25,
                32.9863,
            ]
        )

        # Model function: U1 * n^(U2 + log(n) * U3)
        log_n = jnp.log(n_values)
        power = u2 + log_n * u3
        model_values = u1 * (n_values**power)

        # Residuals: model - observed = 0
        residuals = model_values - g_values

        return residuals

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(0.0)

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
