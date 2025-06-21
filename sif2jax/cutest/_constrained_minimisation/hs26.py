import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS26(AbstractConstrainedMinimisation):
    """Problem 26 from the Hock-Schittkowski test collection.

    A 3-variable polynomial function with one equality constraint.

    f(x) = (x₁ - x₂)² + (x₂ - x₃)⁴

    Subject to: (1 + x₂²)x₁ + x₃⁴ - 3 = 0

    Source: problem 26 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Huang, Aggarwal [34], Miele e.al. [43]

    Classification: PPR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        return jnp.array([-2.6, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: (1 + x₂²)x₁ + x₃⁴ - 3 = 0
        equality_constraint = (1.0 + x2**2) * x1 + x3**4 - 3.0
        return equality_constraint, None
