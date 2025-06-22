import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS33(AbstractConstrainedMinimisation):
    """Problem 33 from the Hock-Schittkowski test collection.

    A 3-variable polynomial function with two inequality constraints and bounds.

    f(x) = (x₁ - 1)(x₁ - 2)(x₁ - 3) + x₃

    Subject to: x₃² - x₂² - x₁² ≥ 0
                x₁² + x₂² + x₃² - 4 ≥ 0
                0 ≤ x₁
                0 ≤ x₂
                0 ≤ x₃ ≤ 5

    Source: problem 33 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Beltrami [6], Hartmann [28]

    Classification: PQR-T1-8
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 - 1.0) * (x1 - 2.0) * (x1 - 3.0) + x3

    def y0(self):
        return jnp.array([0.0, 0.0, 3.0])

    def args(self):
        return None

    def expected_result(self):
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array([0.0, sqrt2, sqrt2])

    def expected_objective_value(self):
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array(sqrt2 - 6.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([jnp.inf, jnp.inf, 5.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraints
        inequality_constraint = jnp.array(
            [
                x3**2 - x2**2 - x1**2,  # x₃² - x₂² - x₁² ≥ 0
                x1**2 + x2**2 + x3**2 - 4.0,  # x₁² + x₂² + x₃² - 4 ≥ 0
            ]
        )
        return None, inequality_constraint
