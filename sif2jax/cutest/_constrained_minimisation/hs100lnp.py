import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS100LNP(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 100 modified by Todd Plantenga.

    Source: problem 100 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.
    This problem has been modified 20 Oct 92 by Todd Plantenga as follows.
    The nonlinear inequality constraints are removed (if inactive
    at the solution) or changed to equalities (if active).

    SIF input: Ph. Toint, April 1991 and T. Plantenga, October 1992.

    classification OOR2-AN-7-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(7)

    @property
    def y0(self):
        """Initial guess."""
        # From START POINT
        return jnp.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args
        x1, x2, x3, x4, x5, x6, x7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        # From GROUPS and GROUP USES:
        # O1: (X1 - 10)^2 with L2 group type
        o1 = (x1 - 10.0) ** 2

        # O2: 0.2 * (X2 - 12)^2 with L2 group type
        o2 = 0.2 * (x2 - 12.0) ** 2

        # O4: 0.33333333 * (X4 - 11)^2 with L2 group type
        o4 = (1.0 / 3.0) * (x4 - 11.0) ** 2

        # O5: -10*X6 - 8*X7 + 10*X5^6 + ELT(X6, X7) + X3^4
        # ELT(X6, X7) = 7*X6^2 + X7^4 - 4*X6*X7
        elt = 7.0 * x6**2 + x7**4 - 4.0 * x6 * x7
        o5 = -10.0 * x6 - 8.0 * x7 + 10.0 * x5**6 + elt + x3**4

        return o1 + o2 + o4 + o5

    @property
    def bounds(self):
        """Variable bounds."""
        # From BOUNDS: FR (free/unbounded) - return None for unbounded problems
        return None

    def constraint(self, y):
        """Constraint functions."""
        x1, x2, x3, x4, x5, x6, x7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        # From GROUPS, CONSTANTS, and GROUP USES:

        # C1: -X3 - 5*X5 - 127 + (-2)*X1^2 + (-3)*X2^4 + (-4)*X4^2 = 0
        c1 = -x3 - 5.0 * x5 - 127.0 - 2.0 * x1**2 - 3.0 * x2**4 - 4.0 * x4**2

        # C4: -5*X6 + 11*X7 + (-4)*X1^2 + (-1)*X2^2 + 3*X1*X2 + (-2)*X3^2 = 0
        c4 = -5.0 * x6 + 11.0 * x7 - 4.0 * x1**2 - x2**2 + 3.0 * x1 * x2 - 2.0 * x3**2

        # Both constraints are equality constraints (type E in SIF)
        equalities = jnp.array([c1, c4])

        return equalities, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
