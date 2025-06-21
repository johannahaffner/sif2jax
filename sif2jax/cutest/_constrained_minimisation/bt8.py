import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT8(AbstractConstrainedMinimisation):
    """BT8 - Boggs-Tolle test problem 8.

    n = 5, m = 2.
    f(x) = x₁² + x₂² + x₃².
    g₁(x) = x₁ - x₄² + x₂² - 1.
    g₂(x) = x₁² + x₂² - x₅² - 1.

    Start 1: x₁ = 1, x₂ = 1, x₃ = 1, x₄ = 0, x₅ = 0.
    Start 2: x₁ = 7, x₂ = 7, x₃ = 7, x₄ = 0, x₅ = 0.
    Solution: x* = (1.0, 0.0, 0.0, 0.0, 0.0).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: QQR2-AN-5-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return x1**2 + x2**2 + x3**2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        elif self.y0_id == 1:
            return jnp.array([7.0, 7.0, 7.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 - x4**2 + x2**2 - 1
        g2 = x1**2 + x2**2 - x5**2 - 1
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
