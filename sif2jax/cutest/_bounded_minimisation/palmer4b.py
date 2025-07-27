import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER4B(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
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
    M: int = 23

    # X values (radians)
    X_data = jnp.array(
        [
            -1.658063,
            -1.570796,
            -1.396263,
            -1.221730,
            -1.047198,
            -0.872665,
            -0.741119,
            -0.698132,
            -0.523599,
            -0.349066,
            -0.174533,
            0.0,
            0.174533,
            0.349066,
            0.523599,
            0.698132,
            0.741119,
            0.872665,
            1.047198,
            1.221730,
            1.396263,
            1.570796,
            1.658063,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            67.27625,
            52.8537,
            30.2718,
            14.9888,
            5.5675,
            0.92603,
            0.0,
            0.085108,
            1.867422,
            5.014768,
            8.263520,
            9.8046208,
            8.263520,
            5.014768,
            1.867422,
            0.085108,
            0.0,
            0.92603,
            5.5675,
            14.9888,
            30.2718,
            52.8537,
            67.27625,
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
        return jnp.array(3.10595086)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A2, A4 are free, B and C have lower bounds
        lower = jnp.array([-jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.full(4, jnp.inf)
        return lower, upper
