import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS18(AbstractConstrainedMinimisation):
    """Problem 18 from the Hock-Schittkowski test collection.

    A 2-variable constrained optimization problem with multiple inequality constraints
    and bounds.

    f(x) = 0.01x₁² + x₂²

    Subject to:
        x₁x₂ - 25 ≥ 0
        x₁² + x₂² - 25 ≥ 0
        2 ≤ x₁ ≤ 50
        0 ≤ x₂ ≤ 50

    Starting point: x₀ = (2, 2) (not feasible)
    Solution: x* = (√250, √2.5)
    Optimal value: f(x*) = 5

    Source: problem 18 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: QQR-T1-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return 0.01 * x1**2 + x2**2

    def y0(self):
        return jnp.array([2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([jnp.sqrt(250), jnp.sqrt(2.5)])

    def expected_objective_value(self):
        return jnp.array(5.0)

    def bounds(self):
        lower = jnp.array([2.0, 0.0])
        upper = jnp.array([50.0, 50.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints: x₁x₂ - 25 ≥ 0 and x₁² + x₂² - 25 ≥ 0
        inequality_constraints = jnp.array([x1 * x2 - 25, x1**2 + x2**2 - 25])
        return None, inequality_constraints
