import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS13(AbstractConstrainedMinimisation):
    """Problem 13 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with bounds.

    f(x) = (x₁ - 2)² + x₂²

    Subject to:
        (1 - x₁)³ - x₂ ≥ 0
        0 ≤ x₁
        0 ≤ x₂

    Starting point: x₀ = (-2, -2) (not feasible)
    Solution: x* = (1, 0)
    Optimal value: f(x*) = 1

    Source: problem 13 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Kuhn, Tucker [38]

    Classification: QPR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (x1 - 2) ** 2 + x2**2

    def y0(self):
        return jnp.array([-2.0, -2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        lower = jnp.array([0.0, 0.0])
        upper = jnp.array([jnp.inf, jnp.inf])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: (1 - x₁)³ - x₂ ≥ 0
        inequality_constraint = (1 - x1) ** 3 - x2
        return None, inequality_constraint
