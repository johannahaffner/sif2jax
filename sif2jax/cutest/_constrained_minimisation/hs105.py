import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class HS105(AbstractConstrainedMinimisation):
    """Problem 105 from the Hock-Schittkowski test collection.

    An 8-variable maximum-likelihood estimation problem with 235 data points.

    f(x) = -∑ln((aᵢ(x) + bᵢ(x) + cᵢ(x))/√2π)
           i=1 to 235

    where:
    aᵢ(x) = x₁/x₆ exp(-(y₁ - x₃)²/(2x₆²))
    bᵢ(x) = x₂/x₇ exp(-(y₁ - x₄)²/(2x₇²))
    cᵢ(x) = (1 - x₂ - x₁)/x₈ exp(-(y₁ - x₅)²/(2x₈²))

    and y₁ values are from Appendix A

    Subject to:
        1 - x₁ - x₂ ≥ 0
        Variable bounds as specified in the problem

    Source: problem 105 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13]

    Classification: GLR-P1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # All 235 data points from the AMPL file
        y_data = inexact_asarray(
            jnp.array(
                [
                    # y[1] = 95
                    95,
                    # y[2] = 105
                    105,
                    # y[3..6] = 110
                    110,
                    110,
                    110,
                    110,
                    # y[7..10] = 115
                    115,
                    115,
                    115,
                    115,
                    # y[11..25] = 120
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    # y[26..40] = 125
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    125,
                    # y[41..55] = 130
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    130,
                    # y[56..68] = 135
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    135,
                    # y[69..89] = 140
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    140,
                    # y[90..101] = 145
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    145,
                    # y[102..118] = 150
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    150,
                    # y[119..122] = 155
                    155,
                    155,
                    155,
                    155,
                    # y[123..142] = 160
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    160,
                    # y[143..150] = 165
                    165,
                    165,
                    165,
                    165,
                    165,
                    165,
                    165,
                    165,
                    # y[151..167] = 170
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    170,
                    # y[168..175] = 175
                    175,
                    175,
                    175,
                    175,
                    175,
                    175,
                    175,
                    175,
                    # y[176..181] = 180
                    180,
                    180,
                    180,
                    180,
                    180,
                    180,
                    # y[182..187] = 185
                    185,
                    185,
                    185,
                    185,
                    185,
                    185,
                    # y[188..194] = 190
                    190,
                    190,
                    190,
                    190,
                    190,
                    190,
                    190,
                    # y[195..198] = 195
                    195,
                    195,
                    195,
                    195,
                    # y[199..201] = 200
                    200,
                    200,
                    200,
                    # y[202..204] = 205
                    205,
                    205,
                    205,
                    # y[205..212] = 210
                    210,
                    210,
                    210,
                    210,
                    210,
                    210,
                    210,
                    210,
                    # y[213] = 215
                    215,
                    # y[214..219] = 220
                    220,
                    220,
                    220,
                    220,
                    220,
                    220,
                    # y[220..224] = 230
                    230,
                    230,
                    230,
                    230,
                    230,
                    # y[225] = 235
                    235,
                    # y[226..232] = 240
                    240,
                    240,
                    240,
                    240,
                    240,
                    240,
                    240,
                    # y[233] = 245
                    245,
                    # y[234..235] = 250
                    250,
                    250,
                ]
            )
        )

        # Validate we have exactly 235 points
        assert len(y_data) == 235, f"Expected 235 data points, got {len(y_data)}"

        total_log_likelihood = 0.0
        sqrt_2pi = jnp.sqrt(2 * jnp.pi)

        for yi in y_data:
            # Three Gaussian components a[i], b[i], c[i]
            a_i = x1 / x6 * jnp.exp(-((yi - x3) ** 2) / (2 * x6**2))
            b_i = x2 / x7 * jnp.exp(-((yi - x4) ** 2) / (2 * x7**2))
            c_i = (1 - x2 - x1) / x8 * jnp.exp(-((yi - x5) ** 2) / (2 * x8**2))

            # Sum of the three components
            mixture = a_i + b_i + c_i

            # Add to log-likelihood (note: we return negative for minimization)
            total_log_likelihood += jnp.log(mixture / sqrt_2pi)

        return jnp.array(-total_log_likelihood)

    def y0(self):
        return jnp.array(
            [0.1, 0.2, 100.0, 125.0, 175.0, 11.2, 13.2, 15.8]
        )  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.4128928,
                0.4033526,
                131.2613,
                164.3135,
                217.4222,
                12.28018,
                15.77170,
                20.74682,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(1138.416240)

    def bounds(self):
        # Bounds from the AMPL formulation
        lower = jnp.array([0.001, 0.001, 100.0, 130.0, 170.0, 5.0, 5.0, 5.0])
        upper = jnp.array([0.499, 0.449, 180.0, 210.0, 240.0, 25.0, 25.0, 25.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # Single inequality constraint: 1 - x₁ - x₂ ≥ 0
        ineq1 = 1 - x1 - x2

        inequality_constraints = jnp.array([ineq1])
        return None, inequality_constraints
