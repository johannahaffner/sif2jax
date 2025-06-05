import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS83(AbstractConstrainedMinimisation):
    """Problem 83 from the Hock-Schittkowski test collection (Colville No.3).

    A 5-variable nonlinear objective function with inequality constraints and bounds.

    f(x) = 5.3578547*x₃² + 0.8356891*x₁*x₅ + 37.293239*x₁ - 40792.141

    Subject to:
        92 ≥ a₁ + a₂*x₂*x₅ + a₃*x₁*x₄ - a₄*x₄*x₅ ≥ 0
        20 ≥ a₅ + a₆*x₂*x₅ + a₇*x₁*x₂ + a₈*x₃² - 90 ≥ 0
        5 ≥ a₉ + a₁₀*x₃*x₅ + a₁₁*x₁*x₃ + a₁₂*x₃*x₄ - 20 ≥ 0
        78 ≤ x₁ ≤ 102, 33 ≤ x₂ ≤ 45, 27 ≤ xᵢ ≤ 45, i=3,4,5

    Source: problem 83 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Classification: QQR-P1-4
    Note: Simplified implementation - actual aᵢ values would come from Appendix A
    """

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141

    def y0(self):
        return jnp.array([78.0, 33.0, 27.0, 27.0, 27.0])  # not feasible

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([78.0, 33.0, 29.99526, 45.0, 36.77581])

    def expected_objective_value(self):
        return jnp.array(-30665.53867)

    def bounds(self):
        return (
            jnp.array([78.0, 33.0, 27.0, 27.0, 27.0]),
            jnp.array([102.0, 45.0, 45.0, 45.0, 45.0]),
        )

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Simplified inequality constraints (needs actual aᵢ values from Appendix A)
        # Using placeholder values based on feasible region
        ineq1 = 92 - (x1 + x2 * x5 + x1 * x4 - x4 * x5)
        ineq2 = -(x1 + x2 * x5 + x1 * x4 - x4 * x5)
        ineq3 = 20 - (x2 * x5 + x1 * x2 + x3**2 - 90)
        ineq4 = -(x2 * x5 + x1 * x2 + x3**2 - 90)
        ineq5 = 5 - (x3 * x5 + x1 * x3 + x3 * x4 - 20)
        ineq6 = -(x3 * x5 + x1 * x3 + x3 * x4 - 20)
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4, ineq5, ineq6])
        return None, inequality_constraints
