import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT13(AbstractConstrainedMinimisation):
    """BT13 - Boggs-Tolle test problem 13.

    n = 5, m = 1.
    f(x) = x₅.
    g₁(x) = x₁² + (x₁ - 2x₂)² + (x₂ - 3x₃)² + (x₃ - 4x₄)² - x₅².

    Start 1: xᵢ = i, i = 1, ..., 4, x₅ = 228.
    Solution: x* = (0, 0, 0, 0, 0).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: LQR2-AY-5-1
    Note: A lower bound of 0.0 on x5 has been added to make the problem bounded below.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return x5

    def y0(self):
        return jnp.array([1.0, 2.0, 3.0, 3.0, 228.0])  # Note: x4 = 3.0 in SIF

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        # x5 has lower bound of 0.0
        return (
            jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0]),
            jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]),
        )

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraint
        equality_constraint = (
            x1**2 + (x1 - 2 * x2) ** 2 + (x2 - 3 * x3) ** 2 + (x3 - 4 * x4) ** 2 - x5**2
        )
        return equality_constraint, None
