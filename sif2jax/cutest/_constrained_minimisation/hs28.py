import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS28(AbstractConstrainedMinimisation):
    """Problem 28 from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one equality constraint.

    f(x) = (x₁ + x₂)² + (x₂ + x₃)²

    Subject to: x₁ + 2x₂ + 3x₃ - 1 = 0

    Source: problem 28 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Huang, Aggarwal [34]

    Classification: QLR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 + x2) ** 2 + (x2 + x3) ** 2

    def y0(self):
        return jnp.array([-4.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.5, -0.5, 0.5])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: x₁ + 2x₂ + 3x₃ - 1 = 0
        equality_constraint = x1 + 2.0 * x2 + 3.0 * x3 - 1.0
        return equality_constraint, None
