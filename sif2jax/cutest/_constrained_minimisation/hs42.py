import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS42(AbstractConstrainedMinimisation):
    """Problem 42 from the Hock-Schittkowski test collection.

    A 4-variable quadratic objective function with two equality constraints.

    f(x) = (x₁ - 1)² + (x₂ - 2)² + (x₃ - 3)² + (x₄ - 4)²

    Subject to:
        x₁ - 2 = 0
        x₃² + x₄² - 2 = 0

    Source: problem 42 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Brusch [14]

    Classification: QQR-T1-10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return (x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2 + (x4 - 4) ** 2

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution: x* = (2, 2, 0.6√2, 0.8√2)
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array([2.0, 2.0, 0.6 * sqrt2, 0.8 * sqrt2])

    def expected_objective_value(self):
        # f(x*) = 28 - 10√2
        return 28.0 - 10.0 * jnp.sqrt(2.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Equality constraints
        eq1 = x1 - 2
        eq2 = x3**2 + x4**2 - 2
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
