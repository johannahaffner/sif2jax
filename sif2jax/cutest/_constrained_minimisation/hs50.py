import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS50(AbstractConstrainedMinimisation):
    """Problem 50 from the Hock-Schittkowski test collection.

    A 5-variable polynomial objective function with three linear equality constraints.

    f(x) = (x₁ - x₂)² + (x₂ - x₃)² + (x₃ - x₄)⁴ + (x₄ - x₅)²

    Subject to:
        x₁ + 2x₂ + 3x₃ - 6 = 0
        x₂ + 2x₃ + 3x₄ - 6 = 0
        x₃ + 2x₄ + 3x₅ - 6 = 0

    Source: problem 50 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Huang, Aggerwal [34]

    Classification: PLR-T1-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (x1 - x2) ** 2 + (x2 - x3) ** 2 + (x3 - x4) ** 4 + (x4 - x5) ** 2

    def y0(self):
        return jnp.array([35.0, -31.0, 11.0, 5.0, -5.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1 + 2 * x2 + 3 * x3 - 6
        eq2 = x2 + 2 * x3 + 3 * x4 - 6
        eq3 = x3 + 2 * x4 + 3 * x5 - 6
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
