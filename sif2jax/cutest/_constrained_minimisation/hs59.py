import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made:
# 1. Removed negative sign from objective function (line 38) that was wrapping
#    the entire expression
# 2. Verified constraint formulations match the problem description
# 3. Confirmed bounds and starting point match problem specification
# Suspected issues:
# - Objective function value differs by factor of ~11 from pycutest
# - Gradient and Hessian also show significant discrepancies
# - Possible different problem formulation in pycutest vs original HS collection
# Additional resources needed:
# - Original Himmelblau [29] reference to verify exact formulation
# - SIF file for HS59 to compare against pycutest implementation
# - Clarification on whether pycutest uses a scaled or transformed version
class HS59(AbstractConstrainedMinimisation):
    """Problem 59 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear objective function with three constraints.

    f(x) = -75.196 + 3.8112*x₁ + 0.0020567*x₁³ - 1.0345E-5*x₁⁴
           + 6.8306*x₂ - 0.030234*x₁*x₂ + 1.28134E-3*x₁²*x₂
           + 2.266E-7*x₁*x₂⁴ - 0.25645*x₂² + 0.0034604*x₂³ - 1.3514E-5*x₂⁴
           + 28.106/(x₂ + 1) + 5.2375E-6*x₁²*x₂² + 6.3E-8*x₁³*x₂²
           - 7E-10*x₁³*x₂³ - 3.405E-4*x₁*x₂² + 1.6638E-6*x₁*x₂³
           + 2.8673*exp(0.0005*x₁*x₂) - 3.5256E-5*x₁³*x₂

    Subject to:
        x₁*x₂ - 700 ≥ 0
        x₂ - x₁²/125 ≥ 0
        (x₂ - 50)² - 5(x₁ - 55) ≥ 0
        0 ≤ x₁ ≤ 75
        0 ≤ x₂ ≤ 65

    Source: problem 59 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Barnes [3], Himmelblau [29]

    Classification: QQR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (
            -75.196
            + 3.8112 * x1
            + 0.0020567 * x1**3
            - 1.0345e-5 * x1**4
            + 6.8306 * x2
            - 0.030234 * x1 * x2
            + 1.28134e-3 * x1**2 * x2
            + 2.266e-7 * x1 * x2**4
            - 0.25645 * x2**2
            + 0.0034604 * x2**3
            - 1.3514e-5 * x2**4
            + 28.106 / (x2 + 1)
            + 5.2375e-6 * x1**2 * x2**2
            + 6.3e-8 * x1**3 * x2**2
            - 7e-10 * x1**3 * x2**3
            - 3.405e-4 * x1 * x2**2
            + 1.6638e-6 * x1 * x2**3
            + 2.8673 * jnp.exp(0.0005 * x1 * x2)
            - 3.5256e-5 * x1**3 * x2
        )

    def y0(self):
        return jnp.array([90.0, 10.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([13.55010424, 51.66018129])

    def expected_objective_value(self):
        return jnp.array(-7.80426324)

    def bounds(self):
        lower = jnp.array([0.0, 0.0])
        upper = jnp.array([75.0, 65.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = x1 * x2 - 700
        ineq2 = x2 - x1**2 / 125
        ineq3 = (x2 - 50) ** 2 - 5 * (x1 - 55)
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3])
        return None, inequality_constraints
