import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER1B(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
    fitting Y to A2 X**2 + A4 X**4
                + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification SBR2-RN-4-0
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
        return 4  # A2, A4, B, C

    def objective(self, y, args):
        """Compute the objective function (least squares)."""
        del args

        # Extract variables
        A2, A4, B, C = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr

        # Model predictions
        predictions = A2 * X_sqr + A4 * X_4 + B / (C + X_sqr)

        # Residuals
        residuals = predictions - self.Y_data

        # Sum of squares
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(4)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(4)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(3.44734948)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A2, A4 are free, B and C have lower bounds
        lower = jnp.array([-jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.full(4, jnp.inf)
        return lower, upper
