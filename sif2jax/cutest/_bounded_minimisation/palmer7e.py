import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER7E(AbstractBoundedMinimisation):
    """A nonlinear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                A10 X**10 + L * EXP( -K X**2 )

    Source:
    M.  Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification SBR2-RN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 24

    # X values (radians) - starting from index 12
    X_data = jnp.array(
        [
            0.000000,
            0.139626,
            0.261799,
            0.436332,
            0.565245,
            0.512942,
            0.610865,
            0.785398,
            0.959931,
            1.134464,
            1.308997,
            1.483530,
            1.658063,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            4.419446,
            3.564931,
            2.139067,
            0.404686,
            0.000000,
            0.035152,
            0.146813,
            2.718058,
            9.474417,
            26.132221,
            41.451561,
            72.283164,
            117.630959,
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
        return jnp.array(1.48003482e-04)

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
