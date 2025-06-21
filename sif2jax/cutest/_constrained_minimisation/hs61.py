import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS61(AbstractConstrainedMinimisation):
    """Problem 61 from the Hock-Schittkowski test collection.

    A 3-variable quadratic objective function with two equality constraints.

    f(x) = 4*x₁² + 2*x₂² + 2*x₃² - 33*x₁ + 16*x₂ - 24*x₃

    Subject to:
        3*x₁ - 2*x₂² - 7 = 0
        4*x₁ - x₃² - 11 = 0

    Source: problem 61 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Fletcher, Lill [26]

    Classification: QQR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 4 * x1**2 + 2 * x2**2 + 2 * x3**2 - 33 * x1 + 16 * x2 - 24 * x3

    def y0(self):
        return jnp.array([0.0, 0.0, 0.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([5.32677016, -2.11899864, 3.21046424])

    def expected_objective_value(self):
        return jnp.array(-143.6461422)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraints
        eq1 = 3 * x1 - 2 * x2**2 - 7
        eq2 = 4 * x1 - x3**2 - 11
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
