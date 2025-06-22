import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS23(AbstractConstrainedMinimisation):
    """Problem 23 from the Hock-Schittkowski test collection.

    A 2-variable quadratic function with five inequality constraints and bounds.

    f(x) = x₁² + x₂²

    Subject to: x₁ + x₂ - 1 ≥ 0
                x₁² + x₂² - 1 ≥ 0
                9x₁² + x₂² - 9 ≥ 0
                x₁² - x₂ ≥ 0
                x₂² - x₁ ≥ 0
                -50 ≤ x₁ ≤ 50
                -50 ≤ x₂ ≤ 50

    Source: problem 23 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QQR-T1-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return x1**2 + x2**2

    def y0(self):
        return jnp.array([3.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(2.0)

    def bounds(self):
        return (jnp.array([-50.0, -50.0]), jnp.array([50.0, 50.0]))

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints
        inequality_constraint = jnp.array(
            [
                x1 + x2 - 1,  # x₁ + x₂ - 1 ≥ 0
                x1**2 + x2**2 - 1,  # x₁² + x₂² - 1 ≥ 0
                9 * x1**2 + x2**2 - 9,  # 9x₁² + x₂² - 9 ≥ 0
                x1**2 - x2,  # x₁² - x₂ ≥ 0
                x2**2 - x1,  # x₂² - x₁ ≥ 0
            ]
        )
        return None, inequality_constraint
