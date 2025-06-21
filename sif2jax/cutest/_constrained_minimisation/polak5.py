import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLAK5(AbstractConstrainedMinimisation):
    """
    POLAK5 problem.

    A nonlinear minmax problem in two variables.

    Source:
    E. Polak, D.H. Mayne and J.E. Higgins,
    "Superlinearly convergent algorithm for min-max problems"
    JOTA 69, pp. 407-439, 1991.

    SIF input: Ph. Toint, Nov 1993.

    classification  LOR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, u = y
        # Linear objective: minimize u
        return u

    def y0(self):
        # Starting point
        return jnp.array([0.1, 0.1, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution value provided as 50.0
        return jnp.array(50.0)

    def num_variables(self):
        return 3

    def bounds(self):
        # All variables are free (unbounded)
        return jnp.array([-jnp.inf, -jnp.inf, -jnp.inf]), jnp.array(
            [jnp.inf, jnp.inf, jnp.inf]
        )

    def constraint(self, y):
        x1, x2, u = y

        # Element functions
        # E1: (x1 - x2^4 - 1)^2 with S = -1
        # E2: (x1 - x2^4 + 1)^2 with S = 1
        e1 = (x1 - x2**4 - 1.0) ** 2
        e2 = (x1 - x2**4 + 1.0) ** 2

        # X1SQ: x1^2
        x1sq = x1**2

        # Inequality constraints:
        # From SIF: F1 and F2 are L-type with coefficient -1.0 on U
        # F1: -u + 3*x1^2 + 50*e1 <= 0
        # F2: -u + 3*x1^2 + 50*e2 <= 0
        # pycutest returns raw values
        ineq_constraint = jnp.array(
            [3.0 * x1sq + 50.0 * e1 - u, 3.0 * x1sq + 50.0 * e2 - u]
        )

        # Equality constraints: none
        eq_constraint = None

        return eq_constraint, ineq_constraint
