import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CHACONN2(AbstractConstrainedMinimisation):
    """
    CHACONN2 problem.

    A nonlinear minmax problem in two variables.

    Source:
    C. Charalambous and A.R. Conn,
    "An efficient method to solve the minmax problem directly",
    SINUM 15, pp. 162-187, 1978.

    SIF input: Ph. Toint, Nov 1993.

    classification LOR2-AY-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, u = y
        # Linear objective: minimize u
        return u

    def y0(self):
        # Starting point from SIF file
        return jnp.array([2.0, 2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(2.0)

    def bounds(self):
        # No variable bounds
        return None

    def constraint(self, y):
        x1, x2, u = y

        # No equality constraints
        eq_constraint = None

        # Three inequality constraints (L-type: <= 0)
        # F1: -u + (x1^4 + x2^2) <= 0
        # F2: -u + ((2-x1)^2 + (2-x2)^2) <= 0
        # F3: -u + 2.0*exp(x2-x1) <= 0
        # In pycutest format (raw values for L-type)
        ineq_constraint = jnp.array(
            [
                -u + (x1**4 + x2**2),  # F1 - different from CHACONN1
                -u + ((2.0 - x1) ** 2 + (2.0 - x2) ** 2),  # F2
                -u + 2.0 * jnp.exp(x2 - x1),  # F3
            ]
        )

        return eq_constraint, ineq_constraint
