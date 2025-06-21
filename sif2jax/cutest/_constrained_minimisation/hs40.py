import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS40(AbstractConstrainedMinimisation):
    """Problem 40 from the Hock-Schittkowski test collection.

    A 4-variable polynomial objective function with three equality constraints.

    f(x) = -x₁x₂x₃x₄

    Subject to:
        x₁³ + x₂² - 1 = 0
        x₁²x₄ - x₃ = 0
        x₄² - x₂ = 0

    Source: problem 40 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Beltrami [6], Indusi [35]

    Classification: PPR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return -x1 * x2 * x3 * x4

    def y0(self):
        return jnp.array([0.8, 0.8, 0.8, 0.8])

    def args(self):
        return None

    def expected_result(self):
        # Solution: x* = (2^a, 2^{2b}, (-1)^{1}2^c, (-1)^{1}2^b)
        # where a = -1/3, b = -1/4, c = -11/12
        # This gives: (2^{-1/3}, 2^{-1/2}, -2^{-11/12}, -2^{-1/4})
        a = -1.0 / 3.0
        b = -1.0 / 4.0
        c = -11.0 / 12.0
        return jnp.array([2.0**a, 2.0 ** (2 * b), -(2.0**c), -(2.0**b)])

    def expected_objective_value(self):
        return jnp.array(-0.25)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Equality constraints
        eq1 = x1**3 + x2**2 - 1
        eq2 = x1**2 * x4 - x3
        eq3 = x4**2 - x2
        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
