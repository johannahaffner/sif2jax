import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS114(AbstractConstrainedMinimisation):
    """Problem 114 from the Hock-Schittkowski test collection.

    A 10-variable alkylation process optimization problem.

    f(x) = 5.04x₁ + 0.035x₂ + 10x₃ + 3.36x₅ - 0.063x₄x₇

    Subject to:
        Eleven constraints (8 inequalities + 3 equalities)
        Variable bounds from Appendix A

    Source: problem 114 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13]

    Classification: QQR-P1-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y
        return 5.04 * x1 + 0.035 * x2 + 10 * x3 + 3.36 * x5 - 0.063 * x4 * x7

    def y0(self):
        return jnp.array(
            [1745.0, 12000.0, 110.0, 3048.0, 1974.0, 89.2, 92.8, 8.0, 3.6, 145.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                1698.096,
                15818.73,
                54.10228,
                3031.226,
                2000.0,
                90.11537,
                95.0,
                10.49336,
                1.561636,
                153.53535,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-1768.80696)

    def bounds(self):
        # Bounds from Appendix A (simplified)
        lower = jnp.array(
            [
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
                0.00001,
            ]
        )
        upper = jnp.array(
            [2000.0, 16000.0, 120.0, 5000.0, 2000.0, 110.0, 100.0, 50.0, 20.0, 500.0]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y

        # Constants from the PDF
        a = 0.99
        b = 0.9

        # Define g functions as shown in the PDF
        g1 = 35.82 - 0.222 * x10 - b * x9
        g2 = -133 + 3 * x7 - a * x10
        g3 = -g1 + x9 * (1 / b - b)
        g4 = -g2 + (1 / a - a) * x10
        g5 = 1.12 * x1 + 0.13167 * x1 * x8 - 0.00667 * x1 * x8**2 - a * x4
        g6 = 57.425 + 1.098 * x8 - 0.038 * x8**2 + 0.325 * x6 - a * x7
        g7 = -g5 + (1 / a - a) * x4
        g8 = -g6 + (1 / a - a) * x7

        # Eight inequality constraints
        ineq1 = g1
        ineq2 = g2
        ineq3 = g3
        ineq4 = g4
        ineq5 = g5
        ineq6 = g6
        ineq7 = g7
        ineq8 = g8

        # Three equality constraints
        eq1 = 1.22 * x4 - x1 - x5
        eq2 = 98000 * x3 / (x4 * x9 + 1000 * x3) - x6
        eq3 = (x2 + x5) / x1 - x8

        inequality_constraints = jnp.array(
            [ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8]
        )
        equality_constraints = jnp.array([eq1, eq2, eq3])

        return equality_constraints, inequality_constraints
