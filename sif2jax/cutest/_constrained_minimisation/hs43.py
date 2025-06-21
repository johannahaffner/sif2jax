import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS43(AbstractConstrainedMinimisation):
    """Problem 43 from the Hock-Schittkowski test collection (Rosen-Suzuki).

    A 4-variable quadratic objective function with three nonlinear inequality
    constraints.

    f(x) = x₁² + x₂² + 2x₃² + x₄² - 5x₁ - 5x₂ - 21x₃ + 7x₄

    Subject to:
        8 - x₁² - x₂² - x₃² - x₄² - x₁ + x₂ - x₃ + x₄ ≥ 0
        10 - x₁² - 2x₂² - x₃² - 2x₄² + x₁ + x₄ ≥ 0
        5 - 2x₁² - x₂² - x₃² - 2x₁ + x₂ + x₄ ≥ 0

    Source: problem 43 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Charalambous [18], Gould [27], Sheela [57]

    Classification: QQR-T1-11
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return x1**2 + x2**2 + 2 * x3**2 + x4**2 - 5 * x1 - 5 * x2 - 21 * x3 + 7 * x4

    def y0(self):
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 1.0, 2.0, -1.0])

    def expected_objective_value(self):
        return jnp.array(-44.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = 8 - x1**2 - x2**2 - x3**2 - x4**2 - x1 + x2 - x3 + x4
        ineq2 = 10 - x1**2 - 2 * x2**2 - x3**2 - 2 * x4**2 + x1 + x4
        ineq3 = 5 - 2 * x1**2 - x2**2 - x3**2 - 2 * x1 + x2 + x4
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3])
        return None, inequality_constraints
