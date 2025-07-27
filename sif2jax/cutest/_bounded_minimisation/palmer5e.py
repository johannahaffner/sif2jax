import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER5E(AbstractBoundedMinimisation):
    """A nonlinear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
                A10 T_10 + A12 T_12 + A14 T_14
                + L * EXP( -K X**2 )
    where T_i is the i-th (shifted) Chebyshev polynomial

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification SBR2-RN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points (indices 12-23 in original data)
    M: int = 12

    # X values (radians) - starting from index 12
    X_data = jnp.array(
        [
            0.000000,
            1.570796,
            1.396263,
            1.308997,
            1.221730,
            1.125835,
            1.047198,
            0.872665,
            0.698132,
            0.523599,
            0.349066,
            0.174533,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            4.419446,
            95.021015,
            71.6363,
            55.3397,
            41.4459,
            30.131611,
            22.2144,
            10.4570,
            3.8254,
            0.92426,
            0.06596,
            0.00000,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 10  # A0, A2, A4, A6, A8, A10, A12, A14, K, L

    def objective(self, y, args):
        """Compute the objective function (least squares).

        TODO: This implementation uses simple polynomials instead of Chebyshev.
        Needs proper Chebyshev polynomial calculation.
        """
        del args

        # Extract variables
        A0, A2, A4, A6, A8, A10, A12, A14, K, L = y

        # Precompute powers of X (placeholder - should use Chebyshev)
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4
        X_8 = X_sqr * X_6
        X_10 = X_sqr * X_8
        X_12 = X_sqr * X_10
        X_14 = X_sqr * X_12

        # Model predictions (placeholder - should use Chebyshev polynomials)
        predictions = (
            A0
            + A2 * X_sqr
            + A4 * X_4
            + A6 * X_6
            + A8 * X_8
            + A10 * X_10
            + A12 * X_12
            + A14 * X_14
            + L * jnp.exp(-K * X_sqr)
        )

        # Residuals
        residuals = predictions - self.Y_data

        # Sum of squares
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(10)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(10)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(2.53004766e-01)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # All variables are free, but K should be non-negative for exponential decay
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
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
        upper = jnp.full(10, jnp.inf)
        return lower, upper
