import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ALSOTAME(AbstractConstrainedMinimisation):
    """The ALSOTAME function.

    Another simple constrained problem.

    Source: A.R. Conn, N. Gould and Ph.L. Toint,
    "The LANCELOT User's Manual",
    Dept of Maths, FUNDP, 1991.

    SIF input: Ph. Toint, Jan 1991.

    Classification: OOR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x, y_var = y
        # Objective: exp(x - 2*y)
        return jnp.exp(x - 2 * y_var)

    def y0(self):
        # Starting point not given in SIF, using zeros
        return jnp.zeros(2)

    def args(self):
        return None

    def constraint(self, y):
        x, y_var = y
        # Equality constraint: sin(-x + y - 1) = 0
        eq_constraint = jnp.array([jnp.sin(-x + y_var - 1)])
        # No inequality constraints
        return eq_constraint, None

    def bounds(self):
        # x: -2.0 <= x <= 2.0
        # y: -1.5 <= y <= 1.5
        lower = jnp.array([-2.0, -1.5])
        upper = jnp.array([2.0, 1.5])
        return (lower, upper)

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # No expected objective value given in SIF file
        return None
