import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER5A(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    TODO: Human review needed - Chebyshev polynomial calculation incorrect

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
                A10 T_10 + B / ( C + X**2 ), B, C nonnegative.
    where T_i is the i-th (shifted) Chebyshev polynomial

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification SBR2-RN-8-0
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
        return 8  # A0, A2, A4, A6, A8, A10, B, C

    def objective(self, y, args):
        """Compute the objective function (least squares)."""
        del args

        # Extract variables
        A0, A2, A4, A6, A8, A10, B, C = y

        # Constants from SIF
        A = -1.0
        B_const = 1.570796  # X13
        DIFF = 2.0

        # Initialize sum of squares
        obj = 0.0

        for i in range(12):  # Data points from 12 to 23 in SIF
            X = self.X_data[i]
            Y_true = self.Y_data[i]

            # Compute shifted Chebyshev polynomials
            # Y = (2*X - A - B) / DIFF = (2*X - (-1) - 1.570796) / 2
            Y = (2.0 * X - A - B_const) / DIFF

            # T0 = 1.0
            T0 = 1.0
            # T1 = Y
            T1 = Y

            # Recurrence relation: T(j) = 2*Y*T(j-1) - T(j-2)
            # T2 = 2*Y*T1 - T0
            T2 = 2.0 * Y * T1 - T0
            # T3 = 2*Y*T2 - T1
            T3 = 2.0 * Y * T2 - T1
            # T4 = 2*Y*T3 - T2
            T4 = 2.0 * Y * T3 - T2
            # T5 = 2*Y*T4 - T3
            T5 = 2.0 * Y * T4 - T3
            # T6 = 2*Y*T5 - T4
            T6 = 2.0 * Y * T5 - T4
            # T7 = 2*Y*T6 - T5
            T7 = 2.0 * Y * T6 - T5
            # T8 = 2*Y*T7 - T6
            T8 = 2.0 * Y * T7 - T6
            # T9 = 2*Y*T8 - T7
            T9 = 2.0 * Y * T8 - T7
            # T10 = 2*Y*T9 - T8
            T10 = 2.0 * Y * T9 - T8

            # Model prediction
            X_sqr = X * X
            prediction = (
                A0 * T0
                + A2 * T2
                + A4 * T4
                + A6 * T6
                + A8 * T8
                + A10 * T10
                + B / (C + X_sqr)
            )

            # Residual
            residual = prediction - Y_true

            # Add to sum of squares
            obj += residual * residual

        return jnp.array(obj)

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
        return jnp.array(4.0606141e-02)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A0-A10 are free, B and C have lower bounds (B=0, C=0.00001 from pycutest)
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, 0.00001]
        )
        upper = jnp.full(8, jnp.inf)
        return lower, upper
