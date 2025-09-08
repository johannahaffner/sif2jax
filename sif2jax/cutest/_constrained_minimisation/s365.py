import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class S365(AbstractConstrainedMinimisation):
    """S365 problem.

    A problem by Walukiewicz.

    Problem 365 in K. Schittkowski,
    "More Test Problems for Nonlinear Programming Codes",
    Springer Verlag, Berlin, 1987.

    SIF input: Ph. Toint, March 1991.

    Classification: QOR2-AY-7-5

    This is a constrained optimization problem with 7 variables and 5 constraints.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 7

    @property
    def m(self):
        """Number of constraints."""
        return 5

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x1, x3 = y[0], y[2]

        # From SIF: OBJ group uses X1X3 element (2PR type: X * Y)
        return x1 * x3

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2, x3, x4, x5, x6, x7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        # Element calculations (vectorized where beneficial)
        # ISQ elements: (X - Y)^2
        e2 = (x4 - x6) * (x4 - x6)
        e3 = (x5 - x7) * (x5 - x7)

        # 2PR elements: X * Y (simple products)
        x1x3 = x1 * x3
        x3x4 = x3 * x4
        x2x5 = x2 * x5
        x3x6 = x3 * x6
        x2x7 = x2 * x7
        x1x5 = x1 * x5
        x1x7 = x1 * x7

        # Complex elements
        ep = jnp.sqrt(x2 * x2) + x3 * x3  # PX: sqrt(x2^2) + x3^2
        eq = jnp.sqrt(x3 * x3) + (x2 - x1) * (x2 - x1)  # QX: sqrt(x3^2) + (x2-x1)^2

        # Constraints computation
        c1 = e2 + e3 - 4.0
        c2 = x3x4 - x2x5 - ep
        c3 = x3x6 - x2x7 - ep
        c4 = x1x3 + x2x5 - x1x5 - x3x4 - eq
        c5 = x1x3 + x2x7 - x1x7 - x3x6 - eq

        ineq_constraints = inexact_asarray(jnp.array([c1, c2, c3, c4, c5]))
        return None, ineq_constraints

    def equality_constraints(self):
        """All constraints are inequalities."""
        return jnp.zeros(self.m, dtype=bool)

    @property
    def bounds(self):
        """Variable bounds."""
        # From SIF:
        # FR S365 'DEFAULT' means all variables are free by default
        # XL S365 X1 0.0, XL S365 X3 0.0, XL S365 X5 1.0, XL S365 X7 1.0
        # Only lower bounds are specified, upper bounds are free
        lower = inexact_asarray(
            jnp.array([0.0, -jnp.inf, 0.0, -jnp.inf, 1.0, -jnp.inf, 1.0])
        )
        upper = inexact_asarray(
            jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        )
        return lower, upper

    @property
    def y0(self):
        """Initial guess from SIF file."""
        return inexact_asarray(jnp.array([3.0, 0.0, 2.0, -1.5, 1.5, 5.0, 0.0]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # Two different solutions mentioned: 23.3137 (Schittkowski) and 0.0
        # Taking the lower bound as the expected optimal value
        return inexact_asarray(jnp.array(0.0))
