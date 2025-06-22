import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS21(AbstractConstrainedMinimisation):
    """Problem 21 from the Hock-Schittkowski test collection.

    A 2-variable quadratic function with inequality constraint and bounds.

    f(x) = 0.01x₁² + x₂² - 100

    Subject to: 10x₁ - x₂ - 10 ≥ 0
                2 ≤ x₁ ≤ 50
                -50 ≤ x₂ ≤ 50

    Source: problem 21 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QLR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return 0.01 * x1**2 + x2**2 - 100

    def y0(self):
        return jnp.array([-1.0, -1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([2.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(-99.96)

    def bounds(self):
        return (jnp.array([2.0, -50.0]), jnp.array([50.0, 50.0]))

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: 10x₁ - x₂ - 10 ≥ 0
        inequality_constraint = jnp.array([10 * x1 - x2 - 10])
        return None, inequality_constraint
