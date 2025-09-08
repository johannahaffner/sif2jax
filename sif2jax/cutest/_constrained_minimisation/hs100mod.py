import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS100MOD(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 100 modified by Ph. Toint.

    Source: a modification by Ph. Toint of problem 100 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Ph. Toint, April 1991.

    classification OOR2-AN-7-4
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
        # For L2 groups, GVAR = group_linear_combination - constant, then square
        # O1: GVAR = X1 - 10, then square (scale 1.0)
        o1 = (x1 - 10.0) ** 2

        # O2: GVAR = X2 - 12, then square (scale 0.2)
        o2 = 0.2 * ((x2 - 12.0) ** 2)

        # O4: GVAR = X4 - 11, then square (scale 1/3)
        o4 = (1.0 / 3.0) * ((x4 - 11.0) ** 2)

        # O5 group: -10*X6 - 8*X7 (linear) + element contributions (no L2 type)
        # Element contributions from GROUP USES:
        # X5P6 with coefficient 10.0: 10*X5^6
        # EL with coefficient 1.0: ELT(X6, X7) = 7*X6^2 + X7^4 - 4*X6*X7
        # X3P4 with coefficient 1.0: X3^4
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
        # For G-type constraints: group_value - constant >= 0

        # C1: group = -X3 - 5*X5 + elements, constant = -127
        c1_group = -x3 - 5.0 * x5 - 2.0 * x1**2 - 3.0 * x2**4 - 4.0 * x4**2
        c1 = c1_group - (-127.0)

        # C2: group = -7*X1 - 3*X2 - X4 + X5 + elements, constant = -282
        c2_group = -7.0 * x1 - 3.0 * x2 - x4 + x5 - 10.0 * x3**2
        c2 = c2_group - (-282.0)

        # C3: group = -23*X1 + 8*X7 + elements, constant = -196
        c3_group = -23.0 * x1 + 8.0 * x7 - x2**2 - 6.0 * x6**2
        c3 = c3_group - (-196.0)

        # C4: group = 11*X7 + 587*X4 + 391*X5 + 2193*X6 + elements, constant = 0
        c4 = (
            11.0 * x7
            + 587.0 * x4
            + 391.0 * x5
            + 2193.0 * x6
            - 4.0 * x1**2
            - x2**2
            + 3.0 * x1 * x2
            - 2.0 * x3**2
        )

        # All constraints are inequality constraints (type G in SIF)
        inequalities = jnp.array([c1, c2, c3, c4])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From OBJECT BOUND section
        return jnp.array(678.679637889)
