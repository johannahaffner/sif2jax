import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS20(AbstractConstrainedMinimisation):
    """Problem 20 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with multiple inequality constraints
    and bounds.

    f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

    Subject to:
        x₁ + x₂² ≥ 0
        x₁² + x₂ ≥ 0
        x₁² + x₂² - 1 ≥ 0
        -0.5 ≤ x₁ ≤ 0.5

    Starting point: x₀ = (-2, 1) (not feasible)
    Solution: x* = (0.5, 0.5√3)
    Optimal value: f(x*) = 81.5 - 25√3

    Source: problem 20 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: PQR-T1-5
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
        sqrt3 = jnp.sqrt(3)
        return jnp.array([0.5, 0.5 * sqrt3])

    def expected_objective_value(self):
        sqrt3 = jnp.sqrt(3)
        return jnp.array(81.5 - 25 * sqrt3)

    def bounds(self):
        lower = jnp.array([-0.5, -jnp.inf])
        upper = jnp.array([0.5, jnp.inf])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints
        inequality_constraints = jnp.array([x1 + x2**2, x1**2 + x2, x1**2 + x2**2 - 1])
        return None, inequality_constraints
