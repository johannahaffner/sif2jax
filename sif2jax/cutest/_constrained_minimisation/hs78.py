import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS78(AbstractConstrainedMinimisation):
    """Problem 78 from the Hock-Schittkowski test collection.

    A 5-variable nonlinear objective function with three equality constraints.

    f(x) = x₁*x₂*x₃*x₄*x₅

    Subject to:
        x₁² + x₂² + x₃² + x₄² + x₅² - 10 = 0
        x₂*x₃ - 5*x₄*x₅ = 0
        x₁³ + x₂³ + 1 = 0

    Source: problem 78 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Asaadi [1], Powell [51]

    Classification: PPR-P1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return x1 * x2 * x3 * x4 * x5

    def y0(self):
        return jnp.array(
            [-2.0, 1.5, 2.0, -1.0, -1.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-1.717142, 1.595708, 1.827248, -0.7636429, -0.7636435])

    def expected_objective_value(self):
        return jnp.array(-2.91970041)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10
        eq2 = x2 * x3 - 5 * x4 * x5
        eq3 = x1**3 + x2**3 + 1
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
