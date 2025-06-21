import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS46(AbstractConstrainedMinimisation):
    """Problem 46 from the Hock-Schittkowski test collection.

    A 5-variable polynomial objective function with two equality constraints.

    f(x) = (x₁ - x₂)² + (x₃ - 1)² + (x₄ - 1)⁴ + (x₅ - 1)⁶

    Subject to:
        x₁²x₄ + sin(x₄ - x₅) - 1 = 0
        x₂ + x₃⁴x₄² - 2 = 0

    Source: problem 46 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Huang, Aggerwal [34], Miele e.al. [43]

    Classification: PGR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (x1 - x2) ** 2 + (x3 - 1) ** 2 + (x4 - 1) ** 4 + (x5 - 1) ** 6

    def y0(self):
        return jnp.array([jnp.sqrt(2.0) / 2.0, 1.75, 0.5, 2.0, 2.0])

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
        eq1 = x1**2 * x4 + jnp.sin(x4 - x5) - 1
        eq2 = x2 + x3**4 * x4**2 - 2
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
