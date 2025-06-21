import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT11(AbstractConstrainedMinimisation):
    """BT11 - Boggs-Tolle test problem 11.

    n = 5, m = 3.
    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₂ - x₃)² + (x₃ - x₄)⁴ + (x₄ - x₅)⁴.
    g₁(x) = x₁ + x₂² + x₃³ + 2 - √18.
    g₂(x) = x₂ - x₃² + x₄ + 2 - √8.
    g₃(x) = x₁ - x₅ - 2.

    Start 1: xᵢ = 2, i = 1, ..., 5.
    Start 2: xᵢ = 10, i = 1, ..., 5.
    Start 3: xᵢ = 50, i = 1, ..., 5.
    Solution: x* = (1.1912, 1.3626, 1.4728, 1.6349, 1.6790).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: OOR2-AY-5-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x2 - x3) ** 2
            + (x3 - x4) ** 4
            + (x4 - x5) ** 4
        )

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([10.0, 10.0, 10.0, 10.0, 10.0])
        elif self.y0_id == 2:
            return jnp.array([50.0, 50.0, 50.0, 50.0, 50.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.1912, 1.3626, 1.4728, 1.6349, 1.6790])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 + x2**2 + x3**3 + 2 - jnp.sqrt(18)
        g2 = x2 - x3**2 + x4 + 2 - jnp.sqrt(8)
        g3 = x1 - x5 - 2
        equality_constraints = jnp.array([g1, g2, g3])
        return equality_constraints, None
