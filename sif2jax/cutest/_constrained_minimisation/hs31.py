import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS31(AbstractConstrainedMinimisation):
    """Problem 31 from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one inequality constraint and bounds.

    f(x) = 9x₁² + x₂² + 9x₃²

    Subject to: x₁x₂ - 1 ≥ 0
                -10 ≤ x₁ ≤ 10
                1 ≤ x₂ ≤ 10
                -10 ≤ x₃ ≤ 1

    Source: problem 31 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QQR-T1-9
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 9.0 * x1**2 + x2**2 + 9.0 * x3**2

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        sqrt3 = jnp.sqrt(3.0)
        return jnp.array([1.0 / sqrt3, sqrt3, 0.0])

    def expected_objective_value(self):
        return jnp.array(6.0)

    def bounds(self):
        return (jnp.array([-10.0, 1.0, -10.0]), jnp.array([10.0, 10.0, 1.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint: x₁x₂ - 1 ≥ 0
        inequality_constraint = jnp.array([x1 * x2 - 1.0])
        return None, inequality_constraint
