import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLAK4(AbstractConstrainedMinimisation):
    """
    POLAK4 problem.

    A nonlinear minmax problem in two variables.

    Source:
    E. Polak, D.H. Mayne and J.E. Higgins,
    "Superlinearly convergent algorithm for min-max problems"
    JOTA 69, pp. 407-439, 1991.

    SIF input: Ph. Toint, Nov 1993.

    classification  LQR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, u = y
        # Linear objective: minimize u
        return u

    @property
    def y0(self):
        # Starting point
        return jnp.array([0.9, 0.1, 0.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Solution value provided as 0.0
        return jnp.array(0.0)

    def num_variables(self):
        return jnp.array(3)

    @property
    def bounds(self):
        # All variables are free (unbounded)
        return None

    def constraint(self, y):
        x1, x2, u = y

        # Element functions (SSQ type)
        # X1SQ: (x1 - 0)^2 = x1^2
        # X2SQ: (x2 - 0)^2 = x2^2
        # X1M2SQ: (x1 - 2)^2
        x1_sq = x1 * x1
        x2_sq = x2 * x2
        x1m2_sq = (x1 - 2.0) ** 2

        # Inequality constraints (L-type groups):
        # F1: -u - x1 + 2*x1^2 + 2*x2^2 <= 1.0
        # F2: -u + 0.01*x1^2 + 0.01*x2^2 <= 0.01
        # F3: -u + 100000*(x1-2)^2 + x2^2 <= 100000.0

        # pycutest returns the raw constraint values (left side minus right side)
        # For L-type constraints c(x) <= b, pycutest returns c(x) - b
        ineq_constraint = jnp.array(
            [
                -u - x1 + 2.0 * x1_sq + 2.0 * x2_sq - 1.0,
                -u + 0.01 * x1_sq + 0.01 * x2_sq - 0.01,
                -u + 100000.0 * x1m2_sq + x2_sq - 100000.0,
            ]
        )

        # Equality constraints: none
        eq_constraint = None

        return eq_constraint, ineq_constraint
