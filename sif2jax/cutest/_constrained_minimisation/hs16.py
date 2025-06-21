import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS16(AbstractConstrainedMinimisation):
    """Problem 16 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with multiple inequality constraints
    and bounds.

    f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

    Subject to:
        x₁ + x₂² ≥ 0
        x₁² + x₂ ≥ 0
        -0.5 ≤ x₁ ≤ 0.5
        x₂ ≤ 1

    Starting point: x₀ = (-2, 1) (not feasible)
    Solution: x* = (0.5, 0.25)
    Optimal value: f(x*) = 0.25

    Source: problem 16 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: PQR-T1-2
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
        return jnp.array([0.5, 0.25])

    def expected_objective_value(self):
        return jnp.array(0.25)

    def bounds(self):
        lower = jnp.array([-0.5, -jnp.inf])
        upper = jnp.array([0.5, 1.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints: x₁ + x₂² ≥ 0 and x₁² + x₂ ≥ 0
        inequality_constraints = jnp.array([x1 + x2**2, x1**2 + x2])
        return None, inequality_constraints
