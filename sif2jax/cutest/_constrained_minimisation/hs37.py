import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS37(AbstractConstrainedMinimisation):
    """Problem 37 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear function with two inequality constraints and bounds.

    f(x) = -x₁x₂x₃

    Subject to: 72 - x₁ - 2x₂ - 2x₃ ≥ 0
                x₁ + 2x₂ + 2x₃ ≥ 0
                0 ≤ xᵢ ≤ 42, i = 1,2,3

    Source: problem 37 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Box [12]

    Classification: PLR-T1-3
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
        return jnp.array([24.0, 12.0, 12.0])

    def expected_objective_value(self):
        return jnp.array(-3456.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([42.0, 42.0, 42.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraints
        inequality_constraint = jnp.array(
            [
                72.0 - x1 - 2.0 * x2 - 2.0 * x3,  # 72 - x₁ - 2x₂ - 2x₃ ≥ 0
                x1 + 2.0 * x2 + 2.0 * x3,  # x₁ + 2x₂ + 2x₃ ≥ 0
            ]
        )
        return None, inequality_constraint
