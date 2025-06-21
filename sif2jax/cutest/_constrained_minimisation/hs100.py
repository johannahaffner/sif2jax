import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS100(AbstractConstrainedMinimisation):
    """Problem 100 from the Hock-Schittkowski test collection.

    A 7-variable nonlinear objective function with four inequality constraints.

    f(x) = (x₁ - 10)² + 5*(x₂ - 12)² + x₄⁴ + 3*(x₄ - 11)²
           + 10*x₅⁶ + 7*x₆² + x₇⁴ - 4*x₆*x₇ - 10*x₆ - 8*x₇

    Subject to:
        127 - 2*x₁² - 3*x₂⁴ - x₃ - 4*x₄² - 5*x₅ ≥ 0
        282 - 7*x₁ - 3*x₂ - 10*x₃² - x₄ + x₅ ≥ 0
        196 - 23*x₁ - x₂² - 6*x₆² + 8*x₇ ≥ 0
        -4*x₁² - x₂² + 3*x₁*x₂ - 2*x₃² - 5*x₆ + 11*x₇ ≥ 0

    Source: problem 100 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Asaadi [1], Charalambous [18], Wong [59]

    Classification: PPR-P1-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7 = y
        return (
            (x1 - 10) ** 2
            + 5 * (x2 - 12) ** 2
            + x3**4
            + 3 * (x4 - 11) ** 2
            + 10 * x5**6
            + 7 * x6**2
            + x7**4
            - 4 * x6 * x7
            - 10 * x6
            - 8 * x7
        )

    def y0(self):
        return jnp.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0])  # feasible

    def args(self):
        return None

    def expected_result(self):
        return jnp.array(
            [2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]
        )

    def expected_objective_value(self):
        return jnp.array(680.6300573)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = 127 - 2 * x1**2 - 3 * x2**4 - x3 - 4 * x4**2 - 5 * x5
        ineq2 = 282 - 7 * x1 - 3 * x2 - 10 * x3**2 - x4 + x5
        ineq3 = 196 - 23 * x1 - x2**2 - 6 * x6**2 + 8 * x7
        ineq4 = -4 * x1**2 - x2**2 + 3 * x1 * x2 - 2 * x3**2 - 5 * x6 + 11 * x7
        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4])
        return None, inequality_constraints
