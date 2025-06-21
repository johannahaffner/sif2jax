import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS34(AbstractConstrainedMinimisation):
    """Problem 34 from the Hock-Schittkowski test collection.

    A 3-variable linear function with two inequality constraints and bounds.

    f(x) = -x₁

    Subject to: x₂ - exp(x₁) ≥ 0
                x₃ - exp(x₂) ≥ 0
                0 ≤ x₁ ≤ 100
                0 ≤ x₂ ≤ 100
                0 ≤ x₃ ≤ 10

    Source: problem 34 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Eckhardt [24]

    Classification: LGR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return -x1

    def y0(self):
        return jnp.array([0.0, 1.05, 2.9])

    def args(self):
        return None

    def expected_result(self):
        ln10 = jnp.log(10.0)
        return jnp.array([ln10, 10.0, 10.0])

    def expected_objective_value(self):
        ln10 = jnp.log(10.0)
        return jnp.array(-ln10)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([100.0, 100.0, 10.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraints
        inequality_constraint = jnp.array(
            [
                x2 - jnp.exp(x1),  # x₂ - exp(x₁) ≥ 0
                x3 - jnp.exp(x2),  # x₃ - exp(x₂) ≥ 0
            ]
        )
        return None, inequality_constraint
