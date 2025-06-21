import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT12(AbstractConstrainedMinimisation):
    """BT12 - Boggs-Tolle test problem 12.

    n = 5, m = 3.
    f(x) = 0.01x₁² + x₂².
    g₁(x) = x₁ + x₂ - x₃² - 25.
    g₂(x) = x₁² + x₂² - x₄² - 25.
    g₃(x) = x₁ - x₅² - 2.

    Start 1: xᵢ = 2, i = 1, ..., 5.
    Start 2: xᵢ = 1, i = 1, ..., 5.
    Start 3: xᵢ = 3, i = 1, ..., 5.
    Solution: x* = (15.811, 1.5811, 0, 15.083, 3.7164).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: QQR2-AN-5-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 3  # Default to solution from paper as per SIF
    provided_y0s: frozenset = frozenset({0, 1, 2, 3})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return 0.01 * x1**2 + x2**2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        elif self.y0_id == 2:
            return jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
        elif self.y0_id == 3:
            return jnp.array([15.811, 1.5811, 0.0, 15.083, 3.7164])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([15.811, 1.5811, 0.0, 15.083, 3.7164])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 + x2 - x3**2 - 25
        g2 = x1**2 + x2**2 - x4**2 - 25
        g3 = x1 - x5**2 - 2
        equality_constraints = jnp.array([g1, g2, g3])
        return equality_constraints, None
