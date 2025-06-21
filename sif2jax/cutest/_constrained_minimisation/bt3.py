import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT3(AbstractConstrainedMinimisation):
    """BT3 - Boggs-Tolle test problem 3.

    n = 5, m = 3.
    f(x) = (x₁ - x₂)² + (x₂ + x₃ - 2)² + (x₄ - 1)² + (x₅ - 1)².
    g₁(x) = x₁ + 3x₂.
    g₂(x) = x₃ + x₄ - 2x₅.
    g₃(x) = x₂ - x₅.

    Start 1: xᵢ = 2, i = 1, ..., 5.
    Start 2: xᵢ = 20, i = 1, ..., 5.
    Start 3: xᵢ = 200, i = 1, ..., 5.
    Solution: x* = (-0.76744, 0.25581, 0.62791, -0.11628, 0.25581).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SLR2-AY-5-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 1  # Default to second starting point (20.0) as per SIF
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return (x1 - x2) ** 2 + (x2 + x3 - 2) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([20.0, 20.0, 20.0, 20.0, 20.0])
        elif self.y0_id == 2:
            return jnp.array([200.0, 200.0, 200.0, 200.0, 200.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-0.76744, 0.25581, 0.62791, -0.11628, 0.25581])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 + 3 * x2
        g2 = x3 + x4 - 2 * x5
        g3 = x2 - x5
        equality_constraints = jnp.array([g1, g2, g3])
        return equality_constraints, None
