import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS47(AbstractConstrainedMinimisation):
    """Problem 47 from the Hock-Schittkowski test collection.

    A 5-variable polynomial objective function with three equality constraints.

    f(x) = (x₁ - x₂)² + (x₂ - x₃)³ + (x₃ - x₄)⁴ + (x₄ - x₅)⁴

    Subject to:
        x₁ + x₂² + x₃³ - 3 = 0
        x₂ - x₃² + x₄ - 1 = 0
        x₁x₅ - 1 = 0

    Source: problem 47 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Huang, Aggerwal [34], Miele e.al. [43]

    Classification: PPR-T1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (x1 - x2) ** 2 + (x2 - x3) ** 3 + (x3 - x4) ** 4 + (x4 - x5) ** 4

    def y0(self):
        return jnp.array([2.0, jnp.sqrt(2.0), -1.0, 2.0 - jnp.sqrt(2.0), 0.5])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1 + x2**2 + x3**3 - 3
        eq2 = x2 - x3**2 + x4 - 1
        eq3 = x1 * x5 - 1
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
