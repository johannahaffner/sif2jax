import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT1(AbstractConstrainedMinimisation):
    """BT1 - Boggs-Tolle test problem 1.

    n = 2, m = 1.
    f(x) = 100x₁² + 100x₂² - x₁ - 100.
    g(x) = x₁² + x₂² - 1.

    Start: x₁ = 0.08, x₂ = 0.06.
    Solution: x* = (1.0, 0.0).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: QQR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y
        return 100 * x1**2 + 100 * x2**2 - x1 - 100

    def y0(self):
        return jnp.array([0.08, 0.06])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: x₁² + x₂² - 1 = 0
        equality_constraint = x1**2 + x2**2 - 1
        return equality_constraint, None
