import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS7(AbstractConstrainedMinimisation):
    """Problem 7 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear function with an equality constraint.

    f(x) = ln(1 + x₁²) - x₂

    Subject to: (1 + x₁²)² + x₂² - 4 = 0

    Source: problem 7 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Miele e.al. [44,45]

    Classification: GPR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return jnp.log(1 + x1**2) - x2

    def y0(self):
        return jnp.array([2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, jnp.sqrt(3.0)])

    def expected_objective_value(self):
        return jnp.array(-jnp.sqrt(3.0))

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: (1 + x₁²)² + x₂² - 4 = 0
        equality_constraint = (1 + x1**2) ** 2 + x2**2 - 4
        return equality_constraint, None
