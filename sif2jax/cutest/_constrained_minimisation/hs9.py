import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS9(AbstractConstrainedMinimisation):
    """Problem 9 from the Hock-Schittkowski test collection.

    A 2-variable trigonometric function with one equality constraint.

    f(x) = sin(πx₁/12) cos(πx₂/16)

    Subject to: 4x₁ - 3x₂ = 0

    Source: problem 9 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Miele e.al. [44]

    Classification: GLR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return jnp.sin(jnp.pi * x1 / 12) * jnp.cos(jnp.pi * x2 / 16)

    def y0(self):
        return jnp.array([0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # From PDF: x* = (12k - 3, 16k - 4), k=0, ±1, ±2, ...
        # Using k=-1 to get the solution that gives f(x*) = -0.5
        return jnp.array([-15.0, -20.0])

    def expected_objective_value(self):
        return jnp.array(-0.5)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: 4x₁ - 3x₂ = 0
        equality_constraint = 4 * x1 - 3 * x2
        return equality_constraint, None
