import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS8(AbstractConstrainedMinimisation):
    """Problem 8 from the Hock-Schittkowski test collection.

    A 2-variable constant objective with two equality constraints.

    f(x) = -1

    Subject to: x₁² + x₂² - 25 = 0
                x₁x₂ - 9 = 0

    Source: problem 8 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: CQR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del y, args
        return jnp.array(-1.0)

    def y0(self):
        return jnp.array([2.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # From PDF: a = √((25 + √301)/2), b = √((25 - √301)/2)
        # Four solutions: (a, 9/a), (-a, -9/a), (b, 9/b), (-b, -9/b)
        # Using the first solution
        a = jnp.sqrt((25 + jnp.sqrt(301)) / 2)
        return jnp.array([a, 9.0 / a])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraints: x₁² + x₂² - 25 = 0, x₁x₂ - 9 = 0
        c1 = x1**2 + x2**2 - 25
        c2 = x1 * x2 - 9
        return (c1, c2), None
