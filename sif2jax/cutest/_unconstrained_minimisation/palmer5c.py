import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER5C(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
                 A10 T_10
    where T_i is the i-th (shifted) Chebyshev polynomial

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.

    classification: QUR2-RN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 6  # 6 coefficients
    m: int = 12  # 12 data points (X12 through X23)

    @property
    def y0(self):
        # All coefficients start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians) - only X12 through X23 are used
        x_data = jnp.array(
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

        # Y data values (KJmol-1)
        y_data = jnp.array(
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

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals using Chebyshev polynomials."""
        a0, a2, a4, a6, a8, a10 = y
        x_data, y_data = args

        # Transform X to Y = 2*X/(B-A) - (A+B)/(B-A) where A = -1.570796, B = 1.570796
        # This maps the interval [A, B] to [-1, 1] for Chebyshev polynomials
        a_val = -1.570796
        b_val = 1.570796
        diff = b_val - a_val
        y_transformed = 2.0 * x_data / diff - (a_val + b_val) / diff

        # Compute Chebyshev polynomials recursively
        # T0 = 1, T1 = y, T_n = 2*y*T_{n-1} - T_{n-2}
        t0 = jnp.ones_like(y_transformed)
        t1 = y_transformed

        # T2 = 2*y*T1 - T0
        t2 = 2.0 * y_transformed * t1 - t0

        # T4 = 2*y*T3 - T2, but we need T3 first
        t3 = 2.0 * y_transformed * t2 - t1
        t4 = 2.0 * y_transformed * t3 - t2

        # T6 = 2*y*T5 - T4, but we need T5 first
        t5 = 2.0 * y_transformed * t4 - t3
        t6 = 2.0 * y_transformed * t5 - t4

        # T8 = 2*y*T7 - T6, but we need T7 first
        t7 = 2.0 * y_transformed * t6 - t5
        t8 = 2.0 * y_transformed * t7 - t6

        # T10 = 2*y*T9 - T8, but we need T9 first
        t9 = 2.0 * y_transformed * t8 - t7
        t10 = 2.0 * y_transformed * t9 - t8

        # Model prediction using even Chebyshev polynomials only
        predicted = a0 * t0 + a2 * t2 + a4 * t4 + a6 * t6 + a8 * t8 + a10 * t10

        # Compute sum of squared residuals
        residuals = predicted - y_data
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(5.0310687e-02)
