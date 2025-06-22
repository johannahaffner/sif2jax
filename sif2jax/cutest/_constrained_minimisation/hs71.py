import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS71(AbstractConstrainedMinimisation):
    """Problem 71 from the Hock-Schittkowski test collection.

    A 4-variable nonlinear objective function with one equality constraint, one
    inequality constraint and bounds.

    f(x) = x₁*x₄*(x₁ + x₂ + x₃) + x₃

    Subject to:
        x₁*x₂*x₃*x₄ - 25 ≥ 0
        x₁² + x₂² + x₃² + x₄² - 40 = 0
        1 ≤ xᵢ ≤ 5, i=1,...,4

    Source: problem 71 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bartholomew-Biggs [4]

    Classification: PPR-P1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return x1 * x4 * (x1 + x2 + x3) + x3

    def y0(self):
        return jnp.array([1.0, 5.0, 5.0, 1.0])  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 4.742999, 3.821150, 1.379408])

    def expected_objective_value(self):
        return jnp.array(17.0140173)

    def bounds(self):
        return (jnp.array([1.0, 1.0, 1.0, 1.0]), jnp.array([5.0, 5.0, 5.0, 5.0]))

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraint (g(x) ≥ 0)
        ineq1 = x1 * x2 * x3 * x4 - 25
        # Equality constraint
        eq1 = x1**2 + x2**2 + x3**2 + x4**2 - 40

        equality_constraints = jnp.array([eq1])
        inequality_constraints = jnp.array([ineq1])
        return equality_constraints, inequality_constraints
