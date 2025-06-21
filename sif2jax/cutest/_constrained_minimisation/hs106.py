import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS106(AbstractConstrainedMinimisation):
    """Problem 106 from the Hock-Schittkowski test collection.

    An 8-variable heat exchanger design problem with linear objective and constraints.

    f(x) = x₁ + x₂ + x₃

    Subject to:
        Six inequality constraints involving bilinear terms
        Variable bounds as specified

    Source: problem 106 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Avriel, Williams [2], Dembo [22]

    Classification: LQR-P1-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8 = y
        return x1 + x2 + x3

    def y0(self):
        return jnp.array(
            [5000.0, 5000.0, 5000.0, 200.0, 350.0, 150.0, 225.0, 425.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                579.3167,
                1359.943,
                5110.071,
                182.0174,
                295.5985,
                217.9799,
                286.4162,
                395.5979,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(7049.330923)

    def bounds(self):
        # Bounds from the PDF
        lower = jnp.array([100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        upper = jnp.array(
            [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # Six inequality constraints from the PDF
        ineq1 = 1 - 0.0025 * (x4 + x6)

        ineq2 = 1 - 0.0025 * (x5 + x7 - x4)

        ineq3 = 1 - 0.01 * (x8 - x5)

        ineq4 = x1 * x6 - 833.33252 * x4 - 100 * x1 + 83333.333

        ineq5 = x2 * x7 - 1250 * x5 - x2 * x4 + 1250 * x4

        ineq6 = x3 * x8 - 1250000 - x3 * x5 + 2500 * x5

        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4, ineq5, ineq6])
        return None, inequality_constraints
