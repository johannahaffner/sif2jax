import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS29(AbstractConstrainedMinimisation):
    """Problem 29 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear function with one inequality constraint.

    f(x) = -x₁x₂x₃

    Subject to: -x₁² - 2x₂² - 4x₃² + 48 ≥ 0

    Source: problem 29 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Biggs [10]

    Classification: PQR-T1-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return -x1 * x2 * x3

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array([4.0, 2.0 * sqrt2, 2.0])

    def expected_objective_value(self):
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array(-16.0 * sqrt2)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint: -x₁² - 2x₂² - 4x₃² + 48 ≥ 0
        inequality_constraint = -(x1**2) - 2.0 * x2**2 - 4.0 * x3**2 + 48.0
        return None, inequality_constraint
