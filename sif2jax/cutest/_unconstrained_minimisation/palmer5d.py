import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER5D(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification QUR2-RN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 23

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
            83.57418,
            81.007654,
            18.983286,
            8.051067,
            2.044762,
            0.000000,
            1.170451,
            10.479881,
            25.785001,
            44.126844,
            62.822177,
            77.719674,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 4  # A0, A2, A4, A6

    def objective(self, y, args):
        """Compute the objective function (least squares)."""
        del args

        # Extract variables
        A0, A2, A4, A6 = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4

        # Model predictions
        predictions = A0 + A2 * X_sqr + A4 * X_4 + A6 * X_6

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
        return jnp.array(5.0310687e-02)
