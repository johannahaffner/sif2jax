import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS52(AbstractConstrainedMinimisation):
    """Problem 52 from the Hock-Schittkowski test collection.

    A 5-variable quadratic objective function with three linear equality constraints.

    f(x) = (4x₁ - x₂)² + (x₂ + x₃ - 2)² + (x₄ - 1)² + (x₅ - 1)²

    Subject to:
        x₁ + 3x₂ = 0
        x₃ + x₄ - 2x₅ = 0
        x₂ - x₅ = 0

    Source: problem 52 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Miele e.al. [44,45]

    Classification: QLR-T1-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (4 * x1 - x2) ** 2 + (x2 + x3 - 2) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution: x* = (-33, 11, 180, -158, 11)/349
        return jnp.array(
            [-33.0 / 349.0, 11.0 / 349.0, 180.0 / 349.0, -158.0 / 349.0, 11.0 / 349.0]
        )

    def expected_objective_value(self):
        return jnp.array(1859.0 / 349.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1 + 3 * x2
        eq2 = x3 + x4 - 2 * x5
        eq3 = x2 - x5
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
