import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS19(AbstractConstrainedMinimisation):
    """Problem 19 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with multiple inequality constraints
    and bounds.

    f(x) = (x₁ - 10)³ + (x₂ - 20)³

    Subject to:
        (x₁ - 5)² + (x₂ - 5)² - 100 ≥ 0
        -(x₂ - 5)² - (x₁ - 6)² + 82.81 ≥ 0
        13 ≤ x₁ ≤ 100
        0 ≤ x₂ ≤ 100

    Starting point: x₀ = (20.1, 5.84) (not feasible)
    Solution: x* = (14.095, 0.84296079)
    Optimal value: f(x*) = -6961.81381

    Source: problem 19 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Gould [27]

    Classification: PQR-T1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (x1 - 10) ** 3 + (x2 - 20) ** 3

    def y0(self):
        return jnp.array([20.1, 5.84])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([14.095, 0.84296079])

    def expected_objective_value(self):
        return jnp.array(-6961.81381)

    def bounds(self):
        lower = jnp.array([13.0, 0.0])
        upper = jnp.array([100.0, 100.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints
        inequality_constraints = jnp.array(
            [
                (x1 - 5) ** 2 + (x2 - 5) ** 2 - 100,
                -((x2 - 5) ** 2) - (x1 - 6) ** 2 + 82.81,
            ]
        )
        return None, inequality_constraints
