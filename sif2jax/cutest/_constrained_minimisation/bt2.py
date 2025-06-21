import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT2(AbstractConstrainedMinimisation):
    """BT2 - Boggs-Tolle test problem 2.

    n = 3, m = 1.
    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₂ - x₃)⁴.
    g(x) = x₁(1 + x₂²) + x₃⁴ - 4 - 3√2.

    Start 1: xᵢ = 1, i = 1, 2, 3.
    Start 2: xᵢ = 10, i = 1, 2, 3.
    Start 3: xᵢ = 100, i = 1, 2, 3.
    Solution: x* = (1.1049, 1.1967, 1.5353).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: QQR2-AY-3-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 1  # Default to second starting point as per AMPL
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([1.0, 1.0, 1.0])
        elif self.y0_id == 1:
            return jnp.array([10.0, 10.0, 10.0])
        elif self.y0_id == 2:
            return jnp.array([100.0, 100.0, 100.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.1049, 1.1967, 1.5353])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: x₁(1 + x₂²) + x₃⁴ = 8.2426407
        equality_constraint = x1 * (1 + x2**2) + x3**4 - 8.2426407
        return equality_constraint, None
