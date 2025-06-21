import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS15(AbstractConstrainedMinimisation):
    """Problem 15 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with multiple inequality constraints
    and bounds.

    f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

    Subject to:
        x₁x₂ - 1 ≥ 0
        x₁ + x₂² ≥ 0
        x₁ ≤ 0.5

    Starting point: x₀ = (-2, 1) (not feasible)
    Solution: x* = (0.5, 2)
    Optimal value: f(x*) = 306.5

    Source: problem 15 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: PQR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2

    def y0(self):
        return jnp.array([-2.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.5, 2.0])

    def expected_objective_value(self):
        return jnp.array(306.5)

    def bounds(self):
        lower = jnp.array([-jnp.inf, -jnp.inf])
        upper = jnp.array([0.5, jnp.inf])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints: x₁x₂ - 1 ≥ 0 and x₁ + x₂² ≥ 0
        inequality_constraints = jnp.array([x1 * x2 - 1, x1 + x2**2])
        return None, inequality_constraints
