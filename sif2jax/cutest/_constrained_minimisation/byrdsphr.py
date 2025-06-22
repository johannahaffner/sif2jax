import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BYRDSPHR(AbstractConstrainedMinimisation):
    """
    BYRDSPHR problem.

    Minimize a linear function in the intersection of two spheres.

    Source:
    R. Byrd,
    Private communication, Chicago, 1992.

    SIF input: Ph. Toint, November 1992.

    classification LQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        # Linear objective: minimize -x1 - x2 - x3
        return -x1 - x2 - x3

    def y0(self):
        # Starting point
        return jnp.array([5.0, 0.0001, -0.0001])

    def args(self):
        return None

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(-4.68330049)

    def bounds(self):
        # No variable bounds
        return None

    def constraint(self, y):
        x1, x2, x3 = y

        # Two equality constraints (sphere equations)
        # SPH1: x1^2 + x2^2 + x3^2 = 9
        # SPH2: (x1-1)^2 + x2^2 + x3^2 = 9
        eq_constraint = jnp.array(
            [
                x1**2 + x2**2 + x3**2 - 9.0,  # SPH1
                (x1 - 1.0) ** 2 + x2**2 + x3**2 - 9.0,  # SPH2
            ]
        )

        # No inequality constraints
        ineq_constraint = None

        return eq_constraint, ineq_constraint
