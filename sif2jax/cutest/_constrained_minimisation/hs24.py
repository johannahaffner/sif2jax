import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS24(AbstractConstrainedMinimisation):
    """Problem 24 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear function with three inequality constraints and bounds.

    f(x) = (1/(27√3)) * ((x₁ - 3)² - 9) * x₂³

    Subject to: x₁/√3 - x₂ ≥ 0
                x₁ + √3*x₂ ≥ 0
                -x₁ - √3*x₂ + 6 ≥ 0
                0 ≤ x₁
                0 ≤ x₂

    Source: problem 24 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Box [12]

    Classification: PLR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        sqrt3 = jnp.sqrt(3.0)
        return (1.0 / (27.0 * sqrt3)) * ((x1 - 3.0) ** 2 - 9.0) * (x2**3)

    def y0(self):
        return jnp.array([1.0, 0.5])

    def args(self):
        return None

    def expected_result(self):
        sqrt3 = jnp.sqrt(3.0)
        return jnp.array([3.0, sqrt3])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))

    def constraint(self, y):
        x1, x2 = y
        sqrt3 = jnp.sqrt(3.0)
        # Inequality constraints
        inequality_constraint = jnp.array(
            [
                x1 / sqrt3 - x2,  # x₁/√3 - x₂ ≥ 0
                x1 + sqrt3 * x2,  # x₁ + √3*x₂ ≥ 0
                -x1 - sqrt3 * x2 + 6.0,  # -x₁ - √3*x₂ + 6 ≥ 0
            ]
        )
        return None, inequality_constraint
