import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS22(AbstractConstrainedMinimisation):
    """Problem 22 from the Hock-Schittkowski test collection.

    A 2-variable quadratic function with two inequality constraints.

    f(x) = (x₁ - 2)² + (x₂ - 1)²

    Subject to: -x₁ - x₂ + 2 ≥ 0
                -x₁² + x₂ ≥ 0

    Source: problem 22 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13], Himmelblau [29], Sheela [57]

    Classification: QQR-T1-6
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
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints: -x₁ - x₂ + 2 ≥ 0 and -x₁² + x₂ ≥ 0
        inequality_constraint = jnp.array(
            [
                -x1 - x2 + 2,  # -x₁ - x₂ + 2 ≥ 0
                -(x1**2) + x2,  # -x₁² + x₂ ≥ 0
            ]
        )
        return None, inequality_constraint
