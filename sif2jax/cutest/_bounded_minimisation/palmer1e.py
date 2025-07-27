import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER1E(AbstractBoundedMinimisation):
    """A nonlinear least squares problem arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
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
    M: int = 35

    # X values (radians)
    X_data = jnp.array(
        [
            -1.788963,
            -1.745329,
            -1.658063,
            -1.570796,
            -1.483530,
            -1.396263,
            -1.308997,
            -1.218612,
            -1.134464,
            -1.047198,
            -0.872665,
            -0.698132,
            -0.523599,
            -0.349066,
            -0.174533,
            0.0000000,
            1.788963,
            1.745329,
            1.658063,
            1.570796,
            1.483530,
            1.396263,
            1.308997,
            1.218612,
            1.134464,
            1.047198,
            0.872665,
            0.698132,
            0.523599,
            0.349066,
            0.174533,
            -1.8762289,
            -1.8325957,
            1.8762289,
            1.8325957,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            78.596218,
            65.77963,
            43.96947,
            27.038816,
            14.6126,
            6.2614,
            1.538330,
            0.000000,
            1.188045,
            4.6841,
            16.9321,
            33.6988,
            52.3664,
            70.1630,
            83.4221,
            88.3995,
            78.596218,
            65.77963,
            43.96947,
            27.038816,
            14.6126,
            6.2614,
            1.538330,
            0.000000,
            1.188045,
            4.6841,
            16.9321,
            33.6988,
            52.3664,
            70.1630,
            83.4221,
            108.18086,
            92.733676,
            108.18086,
            92.733676,
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
        return jnp.array(8.352321e-04)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # All variables are free (FR) in the SIF file, but K should be non-negative
        # for the exponential to decay (based on pycutest behavior)
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf]
        )
        upper = jnp.full(8, jnp.inf)
        return lower, upper
