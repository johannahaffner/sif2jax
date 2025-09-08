import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER6ENE(AbstractNonlinearEquations):
    """A nonlinear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                A10 X**10 + L * EXP( -K X**2 )

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-8-13
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 13

    # X values (radians) - indices 12-24
    X_data = jnp.array(
        [
            0.000000,
            1.570796,
            1.396263,
            1.221730,
            1.047198,
            0.872665,
            0.785398,
            0.732789,
            0.698132,
            0.610865,
            0.523599,
            0.349066,
            0.174533,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            10.678659,
            75.414511,
            41.513459,
            20.104735,
            7.432436,
            1.298082,
            0.171300,
            0.000000,
            0.068203,
            0.774499,
            2.070002,
            5.574556,
            9.026378,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 8  # A0, A2, A4, A6, A8, A10, K, L

    @property
    def M_residuals(self):
        """Number of residual functions."""
        return jnp.array(13)

    def residual(self, y, args):
        """Compute the residual functions."""
        del args

        # Extract variables
        A0, A2, A4, A6, A8, A10, K, L = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4
        X_8 = X_sqr * X_6
        X_10 = X_sqr * X_8

        # Model predictions: polynomial + exponential term
        predictions = (
            A0
            + A2 * X_sqr
            + A4 * X_4
            + A6 * X_6
            + A8 * X_8
            + A10 * X_10
            + L * jnp.exp(-K * X_sqr)
        )

        # Residuals (prediction - observation)
        residuals = predictions - self.Y_data

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(8)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(8)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # All polynomial coefficients are free
        # K should be non-negative for exponential decay
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                0.0,
                -jnp.inf,
            ]
        )
        upper = jnp.full(8, jnp.inf)
        return lower, upper
