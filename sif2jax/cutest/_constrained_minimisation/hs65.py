import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS65(AbstractConstrainedMinimisation):
    """Problem 65 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear objective function with one inequality constraint and bounds.

    f(x) = (x₁ - x₂)² + (x₁ + x₂ - 10)²/9 + (x₃ - 5)²

    Subject to:
        48 - x₁² - x₂² - x₃² ≥ 0
        -4.5 ≤ xᵢ ≤ 4.5, i=1,2
        -5 ≤ x₃ ≤ 5

    Source: problem 65 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Murtagh, Sargent [47]

    Classification: QQR-P1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 - x2) ** 2 + (x1 + x2 - 10) ** 2 / 9 + (x3 - 5) ** 2

    def y0(self):
        return jnp.array([-5.0, 5.0, 0.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([3.65046182, 3.65046168, 4.62041705])

    def expected_objective_value(self):
        return jnp.array(0.9535288567)

    def bounds(self):
        return (jnp.array([-4.5, -4.5, -5.0]), jnp.array([4.5, 4.5, 5.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint (g(x) ≥ 0)
        ineq1 = 48 - x1**2 - x2**2 - x3**2
        inequality_constraints = jnp.array([ineq1])
        return None, inequality_constraints
