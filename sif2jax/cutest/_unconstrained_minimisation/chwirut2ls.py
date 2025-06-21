import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHWIRUT2LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem CHWIRUT2.

    The objective is to minimize a least-squares function with the model:
    y = exp[-b1*x]/(b2+b3*x) + e

    where b1, b2, b3 are the parameters to be determined and e represents
    the error term.

    Source: Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Chwirut, D., NIST (197?).
      Ultrasonic Reference Block Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # Number of variables
    m: int = 54  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2, b3 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                0.5,
                1.0,
                1.75,
                3.75,
                5.75,
                0.875,
                2.25,
                3.25,
                5.25,
                0.75,
                1.75,
                2.75,
                4.75,
                0.625,
                1.25,
                2.25,
                4.25,
                0.5,
                3.0,
                0.75,
                3.0,
                1.5,
                6.0,
                3.0,
                6.0,
                1.5,
                3.0,
                0.5,
                2.0,
                4.0,
                0.75,
                2.0,
                5.0,
                0.75,
                2.25,
                3.75,
                5.75,
                3.0,
                0.75,
                2.5,
                4.0,
                0.75,
                2.5,
                4.0,
                0.75,
                2.5,
                4.0,
                0.5,
                6.0,
                3.0,
                0.5,
                2.75,
                0.5,
                1.75,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                92.9,
                57.1,
                31.05,
                11.5875,
                8.025,
                63.6,
                21.4,
                14.25,
                8.475,
                63.8,
                26.8,
                16.4625,
                7.125,
                67.3,
                41.0,
                21.15,
                8.175,
                81.5,
                13.12,
                59.9,
                14.62,
                32.9,
                5.44,
                12.56,
                5.44,
                32.0,
                13.95,
                75.8,
                20.0,
                10.42,
                59.5,
                21.67,
                8.55,
                62.0,
                20.2,
                7.76,
                3.75,
                11.81,
                54.7,
                23.7,
                11.55,
                61.3,
                17.7,
                8.74,
                59.2,
                16.3,
                8.62,
                81.0,
                4.87,
                14.62,
                81.7,
                17.17,
                81.3,
                28.9,
            ]
        )

        # Define function to compute model value for a single x
        def compute_model(x):
            # Model: y = exp[-b1*x]/(b2+b3*x)
            numerator = jnp.exp(-b1 * x)
            denominator = b2 + b3 * x
            return numerator / denominator

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute residuals and return sum of squares
        residuals = y_pred - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([0.1, 0.01, 0.02])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not directly provided in the SIF file
        # Values from NIST: 0.167, 0.0052, 0.011
        return jnp.array([0.167, 0.0052, 0.011])

    def expected_objective_value(self):
        # Certified value from NIST: 513.05
        return jnp.array(513.05)
