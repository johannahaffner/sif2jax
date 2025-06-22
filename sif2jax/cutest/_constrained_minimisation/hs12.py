import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS12(AbstractConstrainedMinimisation):
    """Problem 12 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem.

    f(x) = 0.5x₁² + x₂² - x₁x₂ - 7x₁ - 7x₂

    Subject to: 25 - 4x₁² - x₂² ≥ 0

    Starting point: x₀ = (0, 0) (feasible)
    Solution: x* = (2, 3)
    Optimal value: f(x*) = -30

    Source: problem 12 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Mine e.al. [46]

    Classification: QQR-T1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return 0.5 * x1**2 + x2**2 - x1 * x2 - 7 * x1 - 7 * x2

    def y0(self):
        return jnp.array([0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([2.0, 3.0])

    def expected_objective_value(self):
        return jnp.array(-30.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: 25 - 4x₁² - x₂² ≥ 0
        inequality_constraint = 25 - 4 * x1**2 - x2**2
        return None, inequality_constraint
