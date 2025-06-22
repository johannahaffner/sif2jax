import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CANTILVR(AbstractConstrainedMinimisation):
    """
    CANTILVR problem.

    Computation of a minimum weight cantilever consisting of 5 sections of
    square shaped tube of given thickness.

    Source:
    an example in a talk by W.K. Zhang and C. Fleury, LLN, 1994.

    SIF input: Ph. Toint, November 1994
               correction by S. Gratton & Ph. Toint, May 2024

    classification LOR2-MN-5-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        # Linear objective: minimize weight
        return 0.0624 * (x1 + x2 + x3 + x4 + x5)

    def y0(self):
        # Starting point
        return jnp.ones(5)

    def args(self):
        return None

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(1.3399595938)

    def bounds(self):
        # Variable bounds: 0.000001 <= x <= inf
        lower = jnp.full(5, 0.000001)
        upper = jnp.full(5, jnp.inf)
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y

        # No equality constraints
        eq_constraint = None

        # Single inequality constraint:
        # 61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 <= 1
        # In pycutest format (L-type), this is returned as:
        # constraint_value - constant
        ineq_constraint = jnp.array(
            [
                61.0 / x1**3
                + 37.0 / x2**3
                + 19.0 / x3**3
                + 7.0 / x4**3
                + 1.0 / x5**3
                - 1.0
            ]
        )

        return eq_constraint, ineq_constraint
