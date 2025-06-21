import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT7(AbstractConstrainedMinimisation):
    """BT7 - Boggs-Tolle test problem 7.

    n = 5, m = 3.
    f(x) = 100(x₂ - x₁²)² + (1 - x₁)².
    g₁(x) = x₁x₂ - 1 - x₅.
    g₂(x) = x₂² + x₁ - x₃².
    g₃(x) = x₁ + x₅² - ½.

    Start 1: x₁ = -2, x₂ = 1, x₃ = 1, x₄ = 1, x₅ = 1.
    Start 2: x₁ = -20, x₂ = 10, x₃ = 1, x₄ = 1, x₅ = 1.
    Solution: x* = (-0.79212, -1.2624, 0.0, -0.89532, 1.1367).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: OQR2-AN-5-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([-2.0, 1.0, 1.0, 1.0, 1.0])
        elif self.y0_id == 1:
            return jnp.array([-20.0, 10.0, 1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-0.79212, -1.2624, 0.0, -0.89532, 1.1367])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 * x2 - 1 - x5
        g2 = x2**2 + x1 - x3**2
        g3 = x1 + x5**2 - 0.5
        equality_constraints = jnp.array([g1, g2, g3])
        return equality_constraints, None
