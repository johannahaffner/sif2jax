import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS30(AbstractConstrainedMinimisation):
    """Problem 30 from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one inequality constraint and bounds.

    f(x) = x₁² + x₂² + x₃²

    Subject to: x₁² + x₂² - 1 ≥ 0
                1 ≤ x₁ ≤ 10
                -10 ≤ x₂ ≤ 10
                -10 ≤ x₃ ≤ 10

    Source: problem 30 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QQR-T1-8
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return x1**2 + x2**2 + x3**2

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        return (jnp.array([1.0, -10.0, -10.0]), jnp.array([10.0, 10.0, 10.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint: x₁² + x₂² - 1 ≥ 0
        inequality_constraint = jnp.array([x1**2 + x2**2 - 1.0])
        return None, inequality_constraint
