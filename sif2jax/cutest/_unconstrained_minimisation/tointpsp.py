import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class TOINTPSP(AbstractUnconstrainedMinimisation):
    """TOINTPSP problem - Toint's PSP Operations Research problem.

    TODO: Human review needed - gradient test fails with small differences.
    The gradient values are very close but not within tolerance.

    Source: Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation 32(1):839-852, 1978.

    See also Buckley#55 (p.94) (With a slightly lower optimal value?)

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-50-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 50

    def _get_problem_data(self):
        """Get the problem-specific data."""
        # Alpha coefficients
        alpha = jnp.array(
            [
                1.25,
                1.40,
                2.40,
                1.40,
                1.75,
                1.20,
                2.25,
                1.20,
                1.00,
                1.10,
                1.50,
                1.60,
                1.25,
                1.25,
                1.20,
                1.20,
                1.40,
                0.50,
                0.50,
                1.25,
                1.80,
                0.75,
                1.25,
                1.40,
                1.60,
                2.00,
                1.00,
                1.60,
                1.25,
                2.75,
                1.25,
                1.25,
                1.25,
                3.00,
                1.50,
                2.00,
                1.25,
                1.40,
                1.80,
                1.50,
                2.20,
                1.40,
                1.50,
                1.25,
                2.00,
                1.50,
                1.25,
                1.40,
                0.60,
                1.50,
            ]
        )

        # Beta coefficients
        beta = jnp.array(
            [
                1.0,
                1.5,
                1.0,
                0.1,
                1.5,
                2.0,
                1.0,
                1.5,
                3.0,
                2.0,
                1.0,
                3.0,
                0.1,
                1.5,
                0.15,
                2.0,
                1.0,
                0.1,
                3.0,
                0.1,
                1.2,
                1.0,
                0.1,
                2.0,
                1.2,
                3.0,
                1.5,
                3.0,
                2.0,
                1.0,
                1.2,
                2.0,
                1.0,
            ]
        )

        # D coefficients
        d = jnp.array(
            [
                -5.0,
                -5.0,
                -5.0,
                -2.5,
                -6.0,
                -6.0,
                -5.0,
                -6.0,
                -10.0,
                -6.0,
                -5.0,
                -9.0,
                -2.0,
                -7.0,
                -2.5,
                -6.0,
                -5.0,
                -2.0,
                -9.0,
                -2.0,
                -5.0,
                -5.0,
                -2.5,
                -5.0,
                -6.0,
                -10.0,
                -7.0,
                -10.0,
                -6.0,
                -5.0,
                -4.0,
                -4.0,
                -4.0,
            ]
        )

        return alpha, beta, d

    def _act_function(self, t):
        """Compute the ACT group function: c(t) = (t - 5)^2."""
        return (t - 5.0) ** 2

    def _bbt_function(self, t):
        """Compute the BBT group function.

        b(t) = 1/t if t >= 0.1, else 20 - 100*t.
        """
        return jnp.where(t >= 0.1, 1.0 / t, 20.0 - 100.0 * t)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x = y
        alpha, beta, d = self._get_problem_data()

        # GA terms: alpha[i] * c(x[i])
        ga_terms = alpha * jnp.array([self._act_function(x[i]) for i in range(50)])

        # GB terms: beta[i] * b(gb_expression[i] - d[i])
        # Define the GB expressions based on the SIF file
        gb_expressions = jnp.zeros(33)

        # Expressions from the SIF file
        gb_expressions = gb_expressions.at[0].set(x[30] - x[0])
        gb_expressions = gb_expressions.at[1].set(-x[0] + x[1] + x[2])
        gb_expressions = gb_expressions.at[2].set(-x[1] + x[3] + x[4])
        gb_expressions = gb_expressions.at[3].set(-x[3] + x[5] + x[6])
        gb_expressions = gb_expressions.at[4].set(-x[5] + x[7] + x[8])
        gb_expressions = gb_expressions.at[5].set(-x[7] + x[9] + x[10])
        gb_expressions = gb_expressions.at[6].set(-x[9] + x[11] + x[12])
        gb_expressions = gb_expressions.at[7].set(-x[11] + x[13] + x[14])
        gb_expressions = gb_expressions.at[8].set(
            -x[10] - x[12] - x[13] + x[15] + x[16]
        )
        gb_expressions = gb_expressions.at[9].set(-x[15] + x[17] + x[18])
        gb_expressions = gb_expressions.at[10].set(-x[8] - x[17] + x[19])
        gb_expressions = gb_expressions.at[11].set(-x[4] - x[19] - x[20])
        gb_expressions = gb_expressions.at[12].set(-x[18] + x[21] + x[22] + x[23])
        gb_expressions = gb_expressions.at[13].set(-x[22] + x[24] + x[25])
        gb_expressions = gb_expressions.at[14].set(-x[6] - x[24] + x[26] + x[27])
        gb_expressions = gb_expressions.at[15].set(-x[27] + x[28] + x[29])
        gb_expressions = gb_expressions.at[16].set(-x[28] + x[30] + x[31])
        gb_expressions = gb_expressions.at[17].set(-x[31] + x[32] + x[33])
        gb_expressions = gb_expressions.at[18].set(-x[2] - x[32] + x[34])
        gb_expressions = gb_expressions.at[19].set(-x[34] + x[20] + x[35])
        gb_expressions = gb_expressions.at[20].set(-x[35] + x[36] + x[37])
        gb_expressions = gb_expressions.at[21].set(-x[29] - x[36] + x[38])
        gb_expressions = gb_expressions.at[22].set(-x[37] - x[38] + x[39])
        gb_expressions = gb_expressions.at[23].set(-x[39] + x[40] + x[41])
        gb_expressions = gb_expressions.at[24].set(-x[40] + x[42] + x[43] + x[49])
        gb_expressions = gb_expressions.at[25].set(-x[43] + x[44] + x[45] + x[46])
        gb_expressions = gb_expressions.at[26].set(-x[45] + x[47])
        gb_expressions = gb_expressions.at[27].set(
            -x[41] - x[44] - x[47] - x[49] + x[48]
        )
        gb_expressions = gb_expressions.at[28].set(-x[25] - x[33] - x[42])
        gb_expressions = gb_expressions.at[29].set(-x[14] - x[16] - x[23] - x[46])
        gb_expressions = gb_expressions.at[30].set(-x[48])
        gb_expressions = gb_expressions.at[31].set(-x[21])
        gb_expressions = gb_expressions.at[32].set(-x[26])

        # Apply scaling and constants
        gb_scaled = gb_expressions - d
        gb_terms = beta * jnp.array(
            [self._bbt_function(gb_scaled[i]) for i in range(33)]
        )

        return jnp.sum(ga_terms) + jnp.sum(gb_terms)

    def y0(self):
        """Initial guess (not specified in SIF, use zeros)."""
        return jnp.zeros(50)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(225.56040942)  # From SIF file
