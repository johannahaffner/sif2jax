import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class TOINTGOR(AbstractUnconstrainedMinimisation):
    """TOINTGOR problem - Toint's Operations Research problem.

    TODO: Human review needed - runtime test fails.
    JAX implementation is ~8x slower than pycutest (threshold is 5x).
    This is due to complex interdependent linear combinations in GB expressions.

    NOTE: This problem has slower runtime compared to pycutest due to the
    complex interdependent linear combinations in the GB expressions.
    The JAX implementation is about 5-25x slower than pycutest.

    Source: Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation 32(1):839-852, 1978.

    See also Buckley#55 (p.94) (With a slightly lower optimal value?)

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-MN-50-0
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
        """Compute the ACT group function: c(t) = |t| * log(|t| + 1)."""
        at = jnp.abs(t)
        at1 = at + 1.0
        lat = jnp.log(at1)
        return at * lat

    def _bbt_function(self, t):
        """Compute the BBT group function.

        b(t) = t^2 * (I(t<0) + I(t>=0) * log(|t| + 1)).
        """
        at = jnp.abs(t)
        at1 = at + 1.0
        lat = jnp.log(at1)
        tpos = jnp.where(t >= 0, 1.0, 0.0)
        tneg = 1.0 - tpos
        return t * t * (tneg + tpos * lat)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x = y
        alpha, beta, d = self._get_problem_data()

        # GA terms: alpha[i] * c(x[i])
        ga_terms = alpha * self._act_function(x)

        # GB terms: beta[i] * b(gb_expression[i] - d[i])
        # Define the GB expressions based on the SIF file
        # Use a single array construction for efficiency
        gb_expressions = jnp.array(
            [
                -x[30] + x[0],  # GB1
                -x[0] + x[1] + x[2],  # GB2
                -x[1] + x[3] + x[4],  # GB3
                -x[3] + x[5] + x[6],  # GB4
                -x[5] + x[7] + x[8],  # GB5
                -x[7] + x[9] + x[10],  # GB6
                -x[9] + x[11] + x[12],  # GB7
                -x[11] + x[13] + x[14],  # GB8
                -x[10] - x[12] - x[13] + x[15] + x[16],  # GB9
                -x[15] + x[17] + x[18],  # GB10
                -x[8] - x[17] + x[19],  # GB11
                -x[4] - x[19] - x[20],  # GB12
                -x[18] + x[21] + x[22] + x[23],  # GB13
                -x[22] + x[24] + x[25],  # GB14
                -x[6] - x[24] + x[26] + x[27],  # GB15
                -x[27] + x[28] + x[29],  # GB16
                -x[28] + x[30] + x[31],  # GB17
                -x[31] + x[32] + x[33],  # GB18
                -x[2] - x[32] + x[34],  # GB19
                -x[34] + x[20] + x[35],  # GB20
                -x[35] + x[36] + x[37],  # GB21
                -x[29] - x[36] + x[38],  # GB22
                -x[37] - x[38] + x[39],  # GB23
                -x[39] + x[40] + x[41],  # GB24
                -x[40] + x[42] + x[43] + x[49],  # GB25
                -x[43] + x[44] + x[45] + x[46],  # GB26
                -x[45] + x[47],  # GB27
                -x[41] - x[44] - x[47] - x[49] + x[48],  # GB28
                -x[25] - x[33] - x[42],  # GB29
                -x[14] - x[16] - x[23] - x[46],  # GB30
                -x[48],  # GB31
                -x[21],  # GB32
                -x[26],  # GB33
            ]
        )

        # Apply scaling and constants
        gb_scaled = gb_expressions - d
        gb_terms = beta * self._bbt_function(gb_scaled)

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
        return jnp.array(1373.90546067)
