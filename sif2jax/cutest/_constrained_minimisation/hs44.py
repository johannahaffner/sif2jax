import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS44(AbstractConstrainedMinimisation):
    """Problem 44 from the Hock-Schittkowski test collection.

    A 4-variable linear objective function with six inequality constraints
    and bounds on variables.

    f(x) = x₁ - x₂ - x₃ - x₁x₃ + x₁x₄ + x₂x₃ - x₂x₄

    Subject to:
        8 - x₁ - 2x₂ ≥ 0
        12 - 4x₁ - x₂ ≥ 0
        12 - 3x₁ - 4x₂ ≥ 0
        8 - 2x₃ - x₄ ≥ 0
        8 - x₃ - 2x₄ ≥ 0
        5 - x₃ - x₄ ≥ 0
        0 ≤ xᵢ, i=1,...,4

    Source: problem 44 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Konno [37]

    Classification: QLR-T1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return x1 - x2 - x3 - x1 * x3 + x1 * x4 + x2 * x3 - x2 * x4

    def y0(self):
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 3.0, 0.0, 4.0])

    def expected_objective_value(self):
        return jnp.array(-15.0)

    def bounds(self):
        lower = jnp.array([0.0, 0.0, 0.0, 0.0])
        upper = None
        return lower, upper

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = 8 - x1 - 2 * x2
        ineq2 = 12 - 4 * x1 - x2
        ineq3 = 12 - 3 * x1 - 4 * x2
        ineq4 = 8 - 2 * x3 - x4
        ineq5 = 8 - x3 - 2 * x4
        ineq6 = 5 - x3 - x4
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4, ineq5, ineq6])
        return None, inequality_constraints
