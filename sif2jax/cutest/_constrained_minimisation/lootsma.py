import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LOOTSMA(AbstractConstrainedMinimisation):
    """
    LOOTSMA problem.

    An example by F. A. Lootsma in "Constrained optimization via penalty functions"
    Philips Res. Repts., Vol. 23, pp. 408-423, 1968.

    N.B. Many current methods fail to find a feasible point when
    started from the given starting values

    Source:
    a contribution to fullfill the LANCELOT academic licence agreement.

    SIF input: Li-zhi Liao, Dept. of Mathematics,
               Hong Kong Baptist College, May 1994.

    classification OQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return x1**3 - 6.0 * x1**2 + 11.0 * x1 + x3

    def y0(self):
        # Starting point
        return jnp.array([1.0, 1.0, -3.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Solution value not provided in SIF file
        return None

    def num_variables(self):
        return 3

    def bounds(self):
        # X3 has upper bound of 5.0
        # All variables have default lower bound of 0 when no bounds are specified
        return jnp.array([0.0, 0.0, 0.0]), jnp.array([jnp.inf, jnp.inf, 5.0])

    def constraint(self, y):
        x1, x2, x3 = y

        # Equality constraints: none
        eq_constraint = None

        # Inequality constraints (all >= 0 form):
        # C1: -x1^2 - x2^2 + x3^2 >= 0  (or x3^2 >= x1^2 + x2^2)
        # C2: x1^2 + x2^2 + x3^2 >= 4.0
        ineq_constraint = jnp.array(
            [-(x1**2) - x2**2 + x3**2, x1**2 + x2**2 + x3**2 - 4.0]
        )

        return eq_constraint, ineq_constraint
