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
        # with coefficients +1 or -1 (max 5 terms per expression).
        # We use a (33,5) index array into zero-padded x and a matching
        # (33,5) coefficient array of {-1, 0, +1}.
        S = 50  # sentinel index (x_pad[50] = 0, coeff = 0)
        x_pad = jnp.concatenate([y, jnp.zeros(1, dtype=y.dtype)])

        # fmt: off
        # indices[j, k] = which x variable participates in GB expression j
        idx = jnp.array([
            [30, 0, S, S, S],  # GB1:  -x30 +x0
            [ 0, 1, 2, S, S],  # GB2:  -x0  +x1 +x2
            [ 1, 3, 4, S, S],  # GB3:  -x1  +x3 +x4
            [ 3, 5, 6, S, S],  # GB4:  -x3  +x5 +x6
            [ 5, 7, 8, S, S],  # GB5:  -x5  +x7 +x8
            [ 7, 9,10, S, S],  # GB6:  -x7  +x9 +x10
            [ 9,11,12, S, S],  # GB7:  -x9  +x11+x12
            [11,13,14, S, S],  # GB8:  -x11 +x13+x14
            [10,12,13,15,16],  # GB9:  -x10-x12-x13+x15+x16
            [15,17,18, S, S],  # GB10: -x15 +x17+x18
            [ 8,17,19, S, S],  # GB11: -x8 -x17+x19
            [ 4,19,20, S, S],  # GB12: -x4 -x19-x20
            [18,21,22,23, S],  # GB13: -x18 +x21+x22+x23
            [22,24,25, S, S],  # GB14: -x22 +x24+x25
            [ 6,24,26,27, S],  # GB15: -x6 -x24+x26+x27
            [27,28,29, S, S],  # GB16: -x27 +x28+x29
            [28,30,31, S, S],  # GB17: -x28 +x30+x31
            [31,32,33, S, S],  # GB18: -x31 +x32+x33
            [ 2,32,34, S, S],  # GB19: -x2 -x32+x34
            [34,20,35, S, S],  # GB20: -x34 +x20+x35
            [35,36,37, S, S],  # GB21: -x35 +x36+x37
            [29,36,38, S, S],  # GB22: -x29-x36+x38
            [37,38,39, S, S],  # GB23: -x37-x38+x39
            [39,40,41, S, S],  # GB24: -x39 +x40+x41
            [40,42,43,49, S],  # GB25: -x40 +x42+x43+x49
            [43,44,45,46, S],  # GB26: -x43 +x44+x45+x46
            [45,47, S, S, S],  # GB27: -x45 +x47
            [41,44,47,49,48],  # GB28: -x41-x44-x47-x49+x48
            [25,33,42, S, S],  # GB29: -x25-x33-x42
            [14,16,23,46, S],  # GB30: -x14-x16-x23-x46
            [48, S, S, S, S],  # GB31: -x48
            [21, S, S, S, S],  # GB32: -x21
            [26, S, S, S, S],  # GB33: -x26
        ])
        # coefficients: sign of each term (+1, -1, or 0 for sentinel)
        coeff = jnp.array([
            [-1, 1, 0, 0, 0],  # GB1
            [-1, 1, 1, 0, 0],  # GB2
            [-1, 1, 1, 0, 0],  # GB3
            [-1, 1, 1, 0, 0],  # GB4
            [-1, 1, 1, 0, 0],  # GB5
            [-1, 1, 1, 0, 0],  # GB6
            [-1, 1, 1, 0, 0],  # GB7
            [-1, 1, 1, 0, 0],  # GB8
            [-1,-1,-1, 1, 1],  # GB9
            [-1, 1, 1, 0, 0],  # GB10
            [-1,-1, 1, 0, 0],  # GB11
            [-1,-1,-1, 0, 0],  # GB12
            [-1, 1, 1, 1, 0],  # GB13
            [-1, 1, 1, 0, 0],  # GB14
            [-1,-1, 1, 1, 0],  # GB15
            [-1, 1, 1, 0, 0],  # GB16
            [-1, 1, 1, 0, 0],  # GB17
            [-1, 1, 1, 0, 0],  # GB18
            [-1,-1, 1, 0, 0],  # GB19
            [-1, 1, 1, 0, 0],  # GB20
            [-1, 1, 1, 0, 0],  # GB21
            [-1,-1, 1, 0, 0],  # GB22
            [-1,-1, 1, 0, 0],  # GB23
            [-1, 1, 1, 0, 0],  # GB24
            [-1, 1, 1, 1, 0],  # GB25
            [-1, 1, 1, 1, 0],  # GB26
            [-1, 1, 0, 0, 0],  # GB27
            [-1,-1,-1,-1, 1],  # GB28
            [-1,-1,-1, 0, 0],  # GB29
            [-1,-1,-1,-1, 0],  # GB30
            [-1, 0, 0, 0, 0],  # GB31
            [-1, 0, 0, 0, 0],  # GB32
            [-1, 0, 0, 0, 0],  # GB33
        ], dtype=y.dtype)
        # fmt: on

        # Single batched gather + coefficient multiply + row sum
        gb_expr = jnp.sum(coeff * x_pad[idx], axis=1)
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
