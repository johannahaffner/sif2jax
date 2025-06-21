import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS81(AbstractConstrainedMinimisation):
    """Problem 81 from the Hock-Schittkowski test collection.

    A 5-variable exponential objective function with three equality constraints and
    bounds.

    f(x) = exp(x₁*x₂*x₃*x₄*x₅) - 0.5*(x₁³ + x₂³ + 1)²

    Subject to:
        x₁² + x₂² + x₃² + x₄² + x₅² - 10 = 0
        x₂*x₃ - 5*x₄*x₅ = 0
        x₁³ + x₂³ + 1 = 0
        -2.3 ≤ xᵢ ≤ 2.3, i=1,2
        -3.2 ≤ xᵢ ≤ 3.2, i=3,4,5

    Source: problem 81 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Powell [52]

    Classification: GPR-P1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return jnp.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1) ** 2

    def y0(self):
        return jnp.array([-2.0, 2.0, 2.0, -1.0, -1.0])  # not feasible

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-1.717142, 1.159571, 1.827248, -0.7636474, -0.7636390])

    def expected_objective_value(self):
        return jnp.array(0.0539498478)

    def bounds(self):
        return (
            jnp.array([-2.3, -2.3, -3.2, -3.2, -3.2]),
            jnp.array([2.3, 2.3, 3.2, 3.2, 3.2]),
        )

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10
        eq2 = x2 * x3 - 5 * x4 * x5
        eq3 = x1**3 + x2**3 + 1
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
