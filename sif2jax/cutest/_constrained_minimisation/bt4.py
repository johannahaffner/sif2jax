import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT4(AbstractConstrainedMinimisation):
    """BT4 - Boggs-Tolle test problem 4.

    n = 3, m = 2.
    f(x) = x₁ - x₂ + x₂³.
    g₁(x) = x₁² + x₂² + x₃² - 25.
    g₂(x) = x₁ + x₂ + x₃ - 1.

    Start 1: x₁ = 3.1494, x₂ = 1.4523, x₃ = -3.6017.
    Start 2: x₁ = 3.122, x₂ = 1.489, x₃ = -3.611.
    Start 3: x₁ = -0.94562, x₂ = -2.35984, x₃ = 4.30546.
    Solution: x* = (4.0382, -2.9470, -0.09115).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: QQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 3  # Default to solution from paper as per SIF
    provided_y0s: frozenset = frozenset({0, 1, 2, 3})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return x1 - x2 + x2**3

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([3.1494, 1.4523, -3.6017])
        elif self.y0_id == 1:
            return jnp.array([3.122, 1.489, -3.611])
        elif self.y0_id == 2:
            return jnp.array([-0.94562, -2.35984, 4.30546])
        elif self.y0_id == 3:
            return jnp.array([4.0382, -2.9470, -0.09115])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([4.0382, -2.9470, -0.09115])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraints
        g1 = x1**2 + x2**2 + x3**2 - 25
        g2 = x1 + x2 + x3 - 1
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
