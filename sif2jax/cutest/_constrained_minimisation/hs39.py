import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS39(AbstractConstrainedMinimisation):
    """Problem 39 from the Hock-Schittkowski test collection.

    A 4-variable linear objective function with two nonlinear equality constraints.

    f(x) = -x₁

    Subject to:
        x₂ - x₁³ - x₃² = 0
        x₁² - x₂ - x₄² = 0

    Source: problem 39 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Miele e.al. [44,45]

    Classification: LPR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return -x1

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0, 0.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Equality constraints
        eq1 = x2 - x1**3 - x3**2
        eq2 = x1**2 - x2 - x4**2
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
