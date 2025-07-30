import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER2E(AbstractBoundedMinimisation):
    """A nonlinear least squares problem arising from chemical kinetics.

    model: H-N=C=O TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                A10 X**10 + L * EXP( -K X**2 )

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification SBR2-RN-8-0
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
        return 8  # A0, A2, A4, A6, A8, A10, K, L

    def objective(self, y, args):
        """Compute the objective function (least squares)."""
        del args

        # Extract variables
        A0, A2, A4, A6, A8, A10, K, L = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4
        X_8 = X_sqr * X_6
        X_10 = X_sqr * X_8

        # Model predictions
        predictions = (
            A0
            + A2 * X_sqr
            + A4 * X_4
            + A6 * X_6
            + A8 * X_8
            + A10 * X_10
            + L * jnp.exp(-K * X_sqr)
        )

        # Residuals
        residuals = predictions - self.Y_data

        # Sum of squares
        return jnp.sum(residuals**2)

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
        """Expected optimal objective value."""
        return jnp.array(7.9851969e-05)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # All variables are free (FR) in the SIF file
        # K should be non-negative for the exponential to decay
        # From pycutest behavior, K has lower bound 0
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf]
        )
        upper = jnp.full(8, jnp.inf)
        return lower, upper
