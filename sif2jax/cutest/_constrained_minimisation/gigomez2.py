"""
A nonlinear minmax problem

Source:
C. Gigola and S. Gomez,
"A Regularization Method for Solving the Finite Convex Min-Max Problem",
SINUM 27(6), pp. 1621-1634, 1990.

SIF input: Ph. Toint, August 1993.

classification LOR2-AY-3-3
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GIGOMEZ2(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "GIGOMEZ2"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # 3 variables: X1, X2, Z
    n_inequality_constraints: int = 3  # 3 inequality constraints

    @property
    def y0(self):
        return jnp.array([2.0, 2.0, 2.0])  # X1, X2, Z

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        x1, x2, z = y
        return jnp.array(z)  # Minimize Z

    def constraint(self, y):
        x1, x2, z = y

        # All constraints are inequality constraints: g(y) >= 0
        # C1: z - x1**2 - x2**4 >= 0
        c1 = z - x1**2 - x2**4

        # C2: z - (2-x1)**2 - (2-x2)**2 >= 0
        c2 = z - (2.0 - x1) ** 2 - (2.0 - x2) ** 2

        # C3: z - 2*exp(x2-x1) >= 0
        c3 = z - 2.0 * jnp.exp(x2 - x1)

        # Return (None, inequality_constraints) since we have no equality constraints
        return None, jnp.array([c1, c2, c3])

    @property
    def bounds(self):
        # No bounds specified in the SIF file (FR = free)
        return None

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 80),
        # the optimal objective value is 1.95222
        return jnp.array(1.95222)
