import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS11(AbstractConstrainedMinimisation):
    """Problem 11 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem.

    f(x) = (x₁ - 5)² + x₂² - 25

    Subject to: -x₁² + x₂ ≥ 0

    Starting point: x₀ = (4.9, 0.1) (not feasible)
    Solution: x* = ((a-1)/√6, (a²-2+a⁻²)/6) where a = 7.5/6 + √338.5
    Optimal value: f(x*) = -8.49846 4223

    Source: problem 11 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Biggs [10]

    Classification: QQR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (x1 - 5) ** 2 + x2**2 - 25

    def y0(self):
        return jnp.array([4.9, 0.1])

    def args(self):
        return None

    def expected_result(self):
        # a = 7.5/6 + √338.5
        a = 7.5 / 6 + jnp.sqrt(338.5)
        x1 = (a - 1) / jnp.sqrt(6)
        x2 = (a**2 - 2 + a ** (-2)) / 6
        return jnp.array([x1, x2])

    def expected_objective_value(self):
        return jnp.array(-8.498464223)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: -x₁² + x₂ ≥ 0, rewritten as x₂ - x₁² ≥ 0
        inequality_constraint = x2 - x1**2
        return None, inequality_constraint
