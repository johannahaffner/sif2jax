import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS108(AbstractConstrainedMinimisation):
    """Problem 108 from the Hock-Schittkowski test collection.

    A 9-variable quadratic optimization problem with many inequality constraints.

    f(x) = -0.5(x₁x₄ - x₂x₃ + x₃x₉ - x₅x₉ + x₅x₈ - x₆x₇)

    Subject to:
        Thirteen inequality constraints involving quadratic terms
        One positivity constraint on x₉

    Source: problem 108 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Himmelblau [29], Pearson [49]

    Classification: QQR-P1-6
    """

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y
        return -0.5 * (x1 * x4 - x2 * x3 + x3 * x9 - x5 * x9 + x5 * x8 - x6 * x7)

    def y0(self):
        return jnp.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.8841292,
                0.4672425,
                0.03742076,
                0.9992996,
                0.8841292,
                0.4672424,
                0.03742076,
                0.9992996,
                2.6e-19,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-0.866025403)

    def bounds(self):
        # No explicit bounds except x₉ ≥ 0
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                0.0,
            ]
        )
        upper = jnp.array(
            [
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
            ]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y

        # Thirteen inequality constraints from the PDF
        ineq1 = 1 - x3**2 - x4**2

        ineq2 = 1 - x9**2

        ineq3 = 1 - x5**2 - x6**2

        ineq4 = 1 - x1**2 - (x2 - x9) ** 2

        ineq5 = 1 - (x1 - x5) ** 2 - (x2 - x6) ** 2

        ineq6 = 1 - (x1 - x7) ** 2 - (x2 - x8) ** 2

        ineq7 = 1 - (x3 - x5) ** 2 - (x4 - x6) ** 2

        ineq8 = 1 - (x3 - x7) ** 2 - (x4 - x8) ** 2

        ineq9 = 1 - x7**2 - (x8 - x9) ** 2

        ineq10 = x1 * x4 - x2 * x3

        ineq11 = x3 * x9

        ineq12 = -x5 * x9

        ineq13 = x5 * x8 - x6 * x7

        inequality_constraints = jnp.array(
            [
                ineq1,
                ineq2,
                ineq3,
                ineq4,
                ineq5,
                ineq6,
                ineq7,
                ineq8,
                ineq9,
                ineq10,
                ineq11,
                ineq12,
                ineq13,
            ]
        )
        return None, inequality_constraints
