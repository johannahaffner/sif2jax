import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class TOINTGOR(AbstractUnconstrainedMinimisation):
    """TOINTGOR problem - Toint's Operations Research problem.

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
        alpha, beta, d = self._get_problem_data()

        # GA terms: sum_i alpha[i] * ACT(x[i])
        ga_terms = alpha * self._act_function(y)

        # GB terms: sum_j beta[j] * BBT(gb_expr[j] - d[j])
        # Each GB expression is a sparse linear combination of x variables
        # with coefficients +1 or -1. We vectorise using index arrays into
        # a zero-padded x, with index 50 as sentinel (maps to 0).
        S = 50  # sentinel index (x_pad[50] = 0)
        x_pad = jnp.concatenate([y, jnp.zeros(1, dtype=y.dtype)])

        # fmt: off
        # Negative contribution indices (up to 4 per GB expression)
        n1 = jnp.array([30, 0, 1, 3, 5, 7, 9,11,10,15, 8, 4,18,22, 6,27,28,31, 2,34,35,29,37,39,40,43,45,41,25,14,48,21,26])  # noqa: E501
        n2 = jnp.array([ S, S, S, S, S, S, S, S,12, S,17,19, S, S,24, S, S, S,32, S, S,36,38, S, S, S, S,44,33,16, S, S, S])  # noqa: E501
        n3 = jnp.array([ S, S, S, S, S, S, S, S,13, S, S,20, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,47,42,23, S, S, S])  # noqa: E501
        n4 = jnp.array([ S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,49, S,46, S, S, S])  # noqa: E501
        # Positive contribution indices (up to 3 per GB expression)
        p1 = jnp.array([ 0, 1, 3, 5, 7, 9,11,13,15,17,19, S,21,24,26,28,30,32,34,20,36,38,39,40,42,44,47,48, S, S, S, S, S])  # noqa: E501
        p2 = jnp.array([ S, 2, 4, 6, 8,10,12,14,16,18, S, S,22,25,27,29,31,33, S,35,37, S, S,41,43,45, S, S, S, S, S, S, S])  # noqa: E501
        p3 = jnp.array([ S, S, S, S, S, S, S, S, S, S, S, S,23, S, S, S, S, S, S, S, S, S, S, S,49,46, S, S, S, S, S, S, S])  # noqa: E501
        # fmt: on

        gb_expr = (
            -x_pad[n1]
            - x_pad[n2]
            - x_pad[n3]
            - x_pad[n4]
            + x_pad[p1]
            + x_pad[p2]
            + x_pad[p3]
        )
        gb_terms = beta * self._bbt_function(gb_expr - d)

        return jnp.sum(ga_terms) + jnp.sum(gb_terms)

    @property
    def y0(self):
        """Initial guess (not specified in SIF, use zeros)."""
        return jnp.zeros(50)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(1373.90546067)
