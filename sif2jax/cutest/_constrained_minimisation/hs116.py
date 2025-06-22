import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS116(AbstractConstrainedMinimisation):
    """Problem 116 from the Hock-Schittkowski test collection.

    A 13-variable 3-stage membrane separation optimization problem.

    f(x) = x₁₁ + x₁₂ + x₁₃

    Subject to:
        Fourteen inequality constraints involving bilinear and quadratic terms
        Variable bounds from Appendix A

    Source: problem 116 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Dembo [21,22]

    Classification: LQR-P1-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        return y[10] + y[11] + y[12]  # x₁₁ + x₁₂ + x₁₃

    def y0(self):
        return jnp.array(
            [
                0.5,
                0.8,
                0.9,
                0.1,
                0.14,
                0.5,
                489.0,
                80.0,
                650.0,
                450.0,
                150.0,
                150.0,
                150.0,
            ]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.8037703,
                0.8999860,
                0.9709724,
                0.09999952,
                0.1908154,
                0.4605717,
                574.0803,
                74.08043,
                500.0162,
                1.0,
                20.23413,
                77.34755,
                0.00673039,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(97.588409)

    def bounds(self):
        # Bounds from Appendix A (simplified)
        lower = jnp.array(
            [0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        upper = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = y

        # Fourteen inequality constraints from the SIF file
        # Note: The objective bounds 50 ≤ f(x) ≤ 250 are handled as a
        # ranged constraint C4 in the SIF file, represented as a single
        # constraint in pycutest
        ineq1 = x3 - x2
        ineq2 = x2 - x1
        ineq3 = 1 - 0.002 * x7 + 0.002 * x8

        # Objective lower bound (C4 in SIF): f(x) ≥ 50
        f_val = self.objective(y, self.args())
        ineq4 = f_val - 50.0

        ineq5 = x13 - 1.262626 * x10 + 1.231059 * x3 * x10
        ineq6 = x5 - 0.03475 * x2 - 0.975 * x2 * x5 + 0.00975 * x2**2
        ineq7 = x6 - 0.03475 * x3 - 0.975 * x3 * x6 + 0.00975 * x3**2
        ineq8 = x5 * x7 - x1 * x8 - x4 * x7 + x4 * x8
        ineq9 = 1 - 0.002 * (x2 * x9 + x5 * x8 - x1 * x8 - x6 * x9) - x5 - x6
        ineq10 = x2 * x9 - x3 * x10 - x6 * x9 - 500 * x2 + 500 * x6 + x2 * x10
        ineq11 = x2 - 0.9 - 0.002 * (x2 * x10 - x3 * x10)
        ineq12 = x4 - 0.03475 * x1 - 0.975 * x1 * x4 + 0.00975 * x1**2
        ineq13 = x11 - 1.262626 * x8 + 1.231059 * x1 * x8
        ineq14 = x12 - 1.262626 * x9 + 1.231059 * x2 * x9

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
                ineq14,
            ]
        )
        return None, inequality_constraints
