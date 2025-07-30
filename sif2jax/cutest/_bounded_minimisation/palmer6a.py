import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER6A(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
                + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification SBR2-RN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 24

    # X values (radians) - starting from index 12
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
        return 6  # A0, A2, A4, A6, B, C

    def objective(self, y, args):
        """Compute the objective function (least squares)."""
        del args

        # Extract variables
        A0, A2, A4, A6, B, C = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4

        # Model predictions
        predictions = A0 + A2 * X_sqr + A4 * X_4 + A6 * X_6 + B / (C + X_sqr)

        # Residuals
        residuals = predictions - self.Y_data

        # Sum of squares
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(6)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(6)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(4.0606141e-02)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A0-A6 are free, B and C have lower bounds
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.full(6, jnp.inf)
        return lower, upper
