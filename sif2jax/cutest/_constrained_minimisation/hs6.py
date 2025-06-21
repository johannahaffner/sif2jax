import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS6(AbstractConstrainedMinimisation):
    """Problem 6 from the Hock-Schittkowski test collection.

    A 2-variable quadratic function with an equality constraint.

    f(x) = (1 - x₁)²

    Subject to: 10(x₂ - x₁²) = 0

    Source: problem 6 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QQR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (1 - x1) ** 2

    def y0(self):
        return jnp.array([-1.2, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: 10(x₂ - x₁²) = 0
        equality_constraint = 10 * (x2 - x1**2)
        return equality_constraint, None
