import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS32(AbstractConstrainedMinimisation):
    """Problem 32 from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one equality and one inequality
    constraint and bounds.

    f(x) = (x₁ + 3x₂ + x₃)² + 4(x₁ - x₂)²

    Subject to: 6x₂ + 4x₃ - x₁³ - 3 ≥ 0
                1 - x₁ - x₂ - x₃ = 0
                0 ≤ xᵢ, i = 1,2,3

    Source: problem 32 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Evtushenko [25]

    Classification: QPR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 + 3.0 * x2 + x3) ** 2 + 4.0 * (x1 - x2) ** 2

    def y0(self):
        return jnp.array([0.1, 0.7, 0.2])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 0.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([jnp.inf, jnp.inf, jnp.inf]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: 1 - x₁ - x₂ - x₃ = 0
        equality_constraint = jnp.array([1.0 - x1 - x2 - x3])
        # Inequality constraint: 6x₂ + 4x₃ - x₁³ - 3 ≥ 0
        inequality_constraint = jnp.array([6.0 * x2 + 4.0 * x3 - x1**3 - 3.0])
        return equality_constraint, inequality_constraint
