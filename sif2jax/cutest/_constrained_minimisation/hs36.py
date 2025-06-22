import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS36(AbstractConstrainedMinimisation):
    """Problem 36 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear function with one inequality constraint and bounds.

    f(x) = -x₁x₂x₃

    Subject to: 72 - x₁ - 2x₂ - 2x₃ ≥ 0
                0 ≤ x₁ ≤ 20
                0 ≤ x₂ ≤ 11
                0 ≤ x₃ ≤ 42

    Source: problem 36 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Biggs [10]

    Classification: PLR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return -x1 * x2 * x3

    def y0(self):
        return jnp.array([10.0, 10.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([20.0, 11.0, 15.0])

    def expected_objective_value(self):
        return jnp.array(-3300.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([20.0, 11.0, 42.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint: 72 - x₁ - 2x₂ - 2x₃ ≥ 0
        inequality_constraint = jnp.array([72.0 - x1 - 2.0 * x2 - 2.0 * x3])
        return None, inequality_constraint
