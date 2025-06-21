import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS77(AbstractConstrainedMinimisation):
    """Problem 77 from the Hock-Schittkowski test collection.

    A 5-variable nonlinear objective function with two equality constraints.

    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₃ - 1)² + (x₄ - 1)⁴ + (x₅ - 1)⁶

    Subject to:
        x₁²*x₄ + sin(x₄ - x₅) - 2√2 = 0
        x₂ + x₃⁴*x₄² - 8 - √2 = 0

    Source: problem 77 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Miele e.al. [42,44,45]

    Classification: OOR2-AY-5-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x3 - 1) ** 2
            + (x4 - 1) ** 4
            + (x5 - 1) ** 6
        )

    def y0(self):
        return jnp.array(
            [2.0, 2.0, 2.0, 2.0, 2.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.166172, 1.182111, 1.380257, 1.506036, 0.6109203])

    def expected_objective_value(self):
        return jnp.array(0.24150513)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints from SIF file
        # CON1: x1²*x4 + sin(x4 - x5) = 2√2
        # CON2: x2 + x3⁴*x4² = √2 + 8
        eq1 = x1**2 * x4 + jnp.sin(x4 - x5) - 2 * jnp.sqrt(2)
        eq2 = x2 + x3**4 * x4**2 - 8 - jnp.sqrt(2)
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
