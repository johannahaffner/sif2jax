import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS14(AbstractConstrainedMinimisation):
    """Problem 14 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with both equality and inequality
    constraints.

    f(x) = (x₁ - 2)² + (x₂ - 1)²

    Subject to:
        -0.25x₁² - x₂² + 1 ≥ 0
        x₁ - 2x₂ + 1 = 0

    Starting point: x₀ = (2, 2) (not feasible)
    Solution: x* = (0.5(√7 - 1), 0.25(√7 + 1))
    Optimal value: f(x*) = 9 - 2.875√7

    Source: problem 14 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13], Himmelblau [29]

    Classification: QQR-T1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (x1 - 2) ** 2 + (x2 - 1) ** 2

    def y0(self):
        return jnp.array([2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        sqrt7 = jnp.sqrt(7)
        x1 = 0.5 * (sqrt7 - 1)
        x2 = 0.25 * (sqrt7 + 1)
        return jnp.array([x1, x2])

    def expected_objective_value(self):
        sqrt7 = jnp.sqrt(7)
        return jnp.array(9 - 2.875 * sqrt7)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: x₁ - 2x₂ + 1 = 0
        equality_constraint = x1 - 2 * x2 + 1
        # Inequality constraint: -0.25x₁² - x₂² + 1 ≥ 0
        inequality_constraint = -0.25 * x1**2 - x2**2 + 1
        return equality_constraint, inequality_constraint
