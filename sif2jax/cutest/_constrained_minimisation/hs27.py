import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS27(AbstractConstrainedMinimisation):
    """Problem 27 from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one equality constraint.

    f(x) = 0.01(x₁ - 1)² + (x₂ - x₁²)²

    Subject to: x₁ + x₃² + 1 = 0

    Source: problem 27 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Miele e.al. [44,45]

    Classification: PQR-T1-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 0.01 * (x1 - 1.0) ** 2 + (x2 - x1**2) ** 2

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-1.0, 1.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(0.04)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: x₁ + x₃² + 1 = 0
        equality_constraint = x1 + x3**2 + 1.0
        return equality_constraint, None
