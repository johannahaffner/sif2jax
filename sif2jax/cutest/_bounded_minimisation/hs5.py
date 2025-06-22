import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS5(AbstractBoundedMinimisation):
    """Problem 5 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear function with bounds on both variables.

    f(x) = sin(x₁ + x₂) + (x₁ - x₂)² - 1.5x₁ + 2.5x₂ + 1

    Subject to: -1.5 ≤ x₁ ≤ 4, -3 ≤ x₂ ≤ 3

    Source: problem 5 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: McCormick [41]

    Classification: GBR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return jnp.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1

    def y0(self):
        return jnp.array([0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # From PDF: x* = (-π/2 + 1/2, -π/2 - 1/2)
        return jnp.array([-jnp.pi / 2 + 0.5, -jnp.pi / 2 - 0.5])

    def expected_objective_value(self):
        # From PDF: f(x*) = -2√3 - π/2
        return jnp.array(-2 * jnp.sqrt(3) - jnp.pi / 2)

    def bounds(self):
        # -1.5 ≤ x1 ≤ 4, -3 ≤ x2 ≤ 3
        lower = jnp.array([-1.5, -3.0])
        upper = jnp.array([4.0, 3.0])
        return lower, upper
