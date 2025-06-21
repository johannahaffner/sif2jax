import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT6(AbstractConstrainedMinimisation):
    """BT6 - Boggs-Tolle test problem 6.

    n = 5, m = 2.
    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₃ - 1)² + (x₄ - 1)⁴ + (x₅ - 1)⁶.
    g₁(x) = x₄x₁² + sin(x₄ - x₅) - 2√2.
    g₂(x) = x₂ + x₃⁴x₅² - 8 - √2.

    Start 1: xᵢ = 2, i = 1, ..., 5.
    Start 2: xᵢ = 8, i = 1, ..., 5.
    Solution: x* = (1.1662, 1.1821, 1.3803, 1.5060, 0.61092).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: OOR2-AY-5-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x3 - 1) ** 2
            + (x4 - 1) ** 4
            + (x5 - 1) ** 6
        )

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([8.0, 8.0, 8.0, 8.0, 8.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.1662, 1.1821, 1.3803, 1.5060, 0.61092])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x4 * x1**2 + jnp.sin(x4 - x5) - 2 * jnp.sqrt(2)
        g2 = x2 + x3**4 * x5**2 - 8 - jnp.sqrt(2)
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
