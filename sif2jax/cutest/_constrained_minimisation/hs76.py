import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS76(AbstractConstrainedMinimisation):
    """Problem 76 from the Hock-Schittkowski test collection.

    A 4-variable quadratic objective function with three inequality constraints and
    bounds.

    f(x) = x₁² + 0.5*x₂² + x₃² + 0.5*x₄² - x₁*x₃ + x₃*x₄ - x₁ - 3*x₂ + x₃ - x₄

    Subject to:
        x₁ + 2*x₂ + x₃ + x₄ ≤ 5
        3*x₁ + x₂ + 2*x₃ - x₄ ≤ 4
        x₂ + 4*x₃ ≥ 1.5
        0 ≤ xᵢ, i=1,...,4

    Source: problem 76 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Murtagh, Sargent [47]

    Classification: QLR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return (
            x1**2
            + 0.5 * x2**2
            + x3**2
            + 0.5 * x4**2
            - x1 * x3
            + x3 * x4
            - x1
            - 3 * x2
            + x3
            - x4
        )

    def y0(self):
        return jnp.array([0.5, 0.5, 0.5, 0.5])  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.2727273, 2.090909, -0.26e-10, 0.5454545])

    def expected_objective_value(self):
        return jnp.array(-4.681818181)

    def bounds(self):
        return (
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf]),
        )

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraints following PyCUTEst convention: LHS - RHS
        # C1: x1 + 2*x2 + x3 + x4 <= 5
        ineq1 = x1 + 2 * x2 + x3 + x4 - 5
        # C2: 3*x1 + x2 + 2*x3 - x4 <= 4
        ineq2 = 3 * x1 + x2 + 2 * x3 - x4 - 4
        # C3: x2 + 4*x3 >= 1.5
        ineq3 = x2 + 4 * x3 - 1.5
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3])
        return None, inequality_constraints
