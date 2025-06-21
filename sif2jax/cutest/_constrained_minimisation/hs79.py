import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS79(AbstractConstrainedMinimisation):
    """Problem 79 from the Hock-Schittkowski test collection.

    A 5-variable nonlinear objective function with three equality constraints.

    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₂ - x₃)² + (x₃ - x₄)⁴ + (x₄ - x₅)⁴

    Subject to:
        x₁ + x₂² + x₃³ - 2 - 3√2 = 0
        x₂ - x₃² + x₄ + 2 - 2√2 = 0
        x₁*x₅ - 2 = 0

    Source: problem 79 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Miele e.al. [42,44,45]

    Classification: PPR-P1-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x2 - x3) ** 2
            + (x3 - x4) ** 4
            + (x4 - x5) ** 4
        )

    def y0(self):
        return jnp.array(
            [2.0, 2.0, 2.0, 2.0, 2.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.191127, 1.362603, 1.472818, 1.635017, 1.679081])

    def expected_objective_value(self):
        return jnp.array(0.0787768209)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        eq1 = x1 + x2**2 + x3**3 - 2 - 3 * jnp.sqrt(2)
        eq2 = x2 - x3**2 + x4 + 2 - 2 * jnp.sqrt(2)
        eq3 = x1 * x5 - 2
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
