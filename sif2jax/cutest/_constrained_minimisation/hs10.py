import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS10(AbstractConstrainedMinimisation):
    """Problem 10 from the Hock-Schittkowski test collection.

    A 2-variable linear function with one inequality constraint.

    f(x) = x₁ - x₂

    Subject to: -3x₁² + 2x₁x₂ - x₂² + 1 ≥ 0

    Source: problem 10 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Biggs [10]

    Classification: LQR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return x1 - x2

    def y0(self):
        return jnp.array([-10.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: -3x₁² + 2x₁x₂ - x₂² + 1 ≥ 0
        inequality_constraint = -3 * x1**2 + 2 * x1 * x2 - x2**2 + 1
        return None, inequality_constraint
