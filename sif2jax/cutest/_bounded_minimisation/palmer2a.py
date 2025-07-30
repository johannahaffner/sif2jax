import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER2A(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=O TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
                + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification SBR2-RN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 23

    # X values (radians)
    X_data = jnp.array(
        [
            -1.745329,
            -1.570796,
            -1.396263,
            -1.221730,
            -1.047198,
            -0.937187,
            -0.872665,
            -0.698132,
            -0.523599,
            -0.349066,
            -0.174533,
            0.0,
            0.174533,
            0.349066,
            0.523599,
            0.698132,
            0.872665,
            0.937187,
            1.047198,
            1.221730,
            1.396263,
            1.570796,
            1.745329,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            72.676767,
            40.149455,
            18.8548,
            6.4762,
            0.8596,
            0.00000,
            0.2730,
            3.2043,
            8.1080,
            13.4291,
            17.7149,
            19.4529,
            17.7149,
            13.4291,
            8.1080,
            3.2053,
            0.2730,
            0.00000,
            0.8596,
            6.4762,
            18.8548,
            40.149455,
            72.676767,
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
        return jnp.array(1.7437656e-02)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A0-A6 are free, B and C have lower bounds
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.full(6, jnp.inf)
        return lower, upper
