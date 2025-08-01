import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLAK6(AbstractConstrainedMinimisation):
    """
    POLAK6 problem.

    A nonlinear minmax problem in four variables. This is a variation
    on problem ROSENMMX.

    Source:
    E. Polak, D.H. Mayne and J.E. Higgins,
    "Superlinearly convergent algorithm for min-max problems"
    JOTA 69, pp. 407-439, 1991.

    SIF input: Ph. Toint, Nov 1993.

    classification  LOR2-AN-5-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, u = y
        # Linear objective: minimize u
        return u

    @property
    def y0(self):
        # Starting point (not specified in SIF, using default)
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution at (0, 1, 2, -1)
        return jnp.array([0.0, 1.0, 2.0, -1.0, -44.0])

    @property
    def expected_objective_value(self):
        # Solution value provided as -44.0
        return jnp.array(-44.0)

    def num_variables(self):
        return 5

    @property
    def bounds(self):
        # All variables are free (unbounded)
        return None

    def constraint(self, y):
        x1, x2, x3, x4, u = y

        # Element functions
        # E1 (EL42): (x1 - (x4+1)^4)^2
        b1 = x4 + 1.0
        e1 = (x1 - b1**4) ** 2

        # E2 (EL442): (x2 - (x1 - (x4+1)^4)^4)^2
        c = x1 - b1**4
        e2 = (x2 - c**4) ** 2

        # E3 (EL4): (x4+1)^4
        e3 = b1**4

        # E4 (EL44): (x1 - (x4+1)^4)^4
        a = x1 - b1**4
        e4 = a**4

        # X3SQ: x3^2
        x3_sq = x3 * x3

        # X4SQ: x4^2
        x4_sq = x4 * x4

        # Inequality constraints (L-type groups):
        # From SIF: constants are on the right-hand side, so:
        # F1: -u - 5*x1 - 5*x2 - 21*x3 + 7*x4 + e1 + e2
        #     + 2*x3^2 + x4^2 + 5*e3 + 5*e4 <= 0
        # F2: -u + 5*x1 - 15*x2 - 11*x3 - 3*x4 + 11*e1 + 11*e2
        #     + 12*x3^2 + 11*x4^2 - 5*e3 + 15*e4 <= 80
        # F3: -u - 15*x1 - 5*x2 - 21*x3 - 3*x4 + 11*e1 + 21*e2
        #     + 12*x3^2 + 21*x4^2 + 15*e3 + 5*e4 <= 100
        # F4: -u + 15*x1 - 15*x2 - 21*x3 - 3*x4 + 11*e1 + 11*e2
        #     + 12*x3^2 + x4^2 - 15*e3 + 15*e4 <= 50

        # pycutest returns the raw constraint values (left side minus right side)
        # For L-type constraints c(x) <= b, pycutest returns c(x) - b
        ineq_constraint = jnp.array(
            [
                -u
                - 5.0 * x1
                - 5.0 * x2
                - 21.0 * x3
                + 7.0 * x4
                + e1
                + e2
                + 2.0 * x3_sq
                + x4_sq
                + 5.0 * e3
                + 5.0 * e4
                - 0.0,
                -u
                + 5.0 * x1
                - 15.0 * x2
                - 11.0 * x3
                - 3.0 * x4
                + 11.0 * e1
                + 11.0 * e2
                + 12.0 * x3_sq
                + 11.0 * x4_sq
                - 5.0 * e3
                + 15.0 * e4
                - 80.0,
                -u
                - 15.0 * x1
                - 5.0 * x2
                - 21.0 * x3
                - 3.0 * x4
                + 11.0 * e1
                + 21.0 * e2
                + 12.0 * x3_sq
                + 21.0 * x4_sq
                + 15.0 * e3
                + 5.0 * e4
                - 100.0,
                -u
                + 15.0 * x1
                - 15.0 * x2
                - 21.0 * x3
                - 3.0 * x4
                + 11.0 * e1
                + 11.0 * e2
                + 12.0 * x3_sq
                + x4_sq
                - 15.0 * e3
                + 15.0 * e4
                - 50.0,
            ]
        )

        # Equality constraints: none
        eq_constraint = None

        return eq_constraint, ineq_constraint
