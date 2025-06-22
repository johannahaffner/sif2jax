import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS113(AbstractConstrainedMinimisation):
    """Problem 113 from the Hock-Schittkowski test collection.

    A 10-variable quadratic problem (Wong No.2) with inequality constraints.

    f(x) = x₁² + x₂² + x₁x₂ - 14x₁ - 16x₂ + (x₃ - 10)²
           + 4(x₄ - 5)² + (x₅ - 3)² + 2(x₆ - 1)² + 5x₇²
           + 7(x₈ - 11)² + 2(x₉ - 10)² + (x₁₀ - 7)² + 45

    Subject to:
        Eight inequality constraints
        No explicit bounds

    Source: problem 113 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Asaadi [1], Charalambous [18], Wong [59]

    Classification: QQR-P1-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y

        return (
            x1**2
            + x2**2
            + x1 * x2
            - 14 * x1
            - 16 * x2
            + (x3 - 10) ** 2
            + 4 * (x4 - 5) ** 2
            + (x5 - 3) ** 2
            + 2 * (x6 - 1) ** 2
            + 5 * x7**2
            + 7 * (x8 - 11) ** 2
            + 2 * (x9 - 10) ** 2
            + (x10 - 7) ** 2
            + 45
        )

    def y0(self):
        return jnp.array(
            [2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 7.0, 3.0, 6.0, 10.0]
        )  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                2.171996,
                2.363683,
                8.773926,
                5.095984,
                0.9906548,
                1.430574,
                1.321644,
                9.828726,
                8.280092,
                8.375927,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(24.3062091)

    def bounds(self):
        # No explicit bounds
        lower = jnp.array([-jnp.inf] * 10)
        upper = jnp.array([jnp.inf] * 10)
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y

        # Eight inequality constraints from the PDF
        ineq1 = 105 - 4 * x1 - 5 * x2 + 3 * x7 - 9 * x8

        ineq2 = -10 * x1 + 8 * x2 + 17 * x7 - 2 * x8

        ineq3 = 8 * x1 - 2 * x2 - 5 * x9 + 2 * x10 + 12

        ineq4 = -3 * (x1 - 2) ** 2 - 4 * (x2 - 3) ** 2 - 2 * x3**2 + 7 * x4 + 120

        ineq5 = -5 * x1**2 - 8 * x2 - (x3 - 6) ** 2 + 2 * x4 + 40

        ineq6 = -0.5 * (x1 - 8) ** 2 - 2 * (x2 - 4) ** 2 - 3 * x5**2 + x6 + 30

        ineq7 = -(x1**2) - 2 * (x2 - 2) ** 2 + 2 * x1 * x2 - 14 * x5 + 6 * x6

        ineq8 = 3 * x1 - 6 * x2 - 12 * (x9 - 8) ** 2 + 7 * x10

        inequality_constraints = jnp.array(
            [ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8]
        )
        return None, inequality_constraints
