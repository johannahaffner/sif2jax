import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class FLT(AbstractConstrainedMinimisation):
    """FLT - A troublesome problem for filter methods.

    A simple 2-variable, 2-constraint optimization problem designed to test
    filter methods. The problem demonstrates challenging behavior for
    sequential quadratic programming (SQP) filter methods.

    Variables: X1, X2
    Objective: minimize (1 - X2)^2
    Constraints:
        CON1: X1^2 = 0
        CON2: X1^3 = 0

    Source: R. Fletcher, S. Leyffer and Ph. L. Toint,
    "On the global convergence of a filter-SQP method",
    SIAM J. Optimization 13 (2002):44-59.

    SIF input: Nick Gould, May 2008.

    Classification: QOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([1.0, 0.0])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Objective: minimize (1.0 - X2)^2
        # From SIF: N OBJ X2 1.0 + X FLT OBJ 1.0 + T OBJ L2
        # The L2 group type squares the group variable: GVAR^2
        # GVAR = constant + linear_terms = 1.0 - X2
        return (1.0 - x2) ** 2

    def constraint(self, y):
        x1, x2 = y

        # CON1: X1^2 = 0 (equality constraint)
        # CON2: X1^3 = 0 (equality constraint)
        eq_constraints = jnp.array(
            [
                x1**2,  # CON1
                x1**3,  # CON2
            ]
        )

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        # Both variables are free (XR in SIF)
        return None

    @property
    def expected_result(self):
        # The solution is X1=0 (from constraints X1^2=0 and X1^3=0)
        # and X2=1 (to minimize (1-X2)^2)
        return jnp.array([0.0, 1.0])

    @property
    def expected_objective_value(self):
        # At the solution X1=0, X2=1, objective = (1-1)^2 = 0
        return jnp.array(0.0)
