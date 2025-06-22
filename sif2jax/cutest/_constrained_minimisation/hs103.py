import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS103(AbstractConstrainedMinimisation):
    """Problem 103 from the Hock-Schittkowski test collection.

    A 7-variable nonlinear objective function with six inequality constraints and
    bounds. This is part of problems 101-103 which share the same form but different
    parameter values.

    f(x) = 10x₁x₂⁻¹x₄²x₆⁻³x₇ᵃ + 15x₁⁻¹x₂⁻²x₃x₄x₅⁻¹x₇⁻·⁵
           + 20x₁⁻²x₂x₄⁻¹x₅⁻²x₆²x₇ + 25x₁²x₂²x₃⁻¹x₅·⁵x₆⁻²x₇

    where a = 0.5 for HS103

    Subject to:
        Four complex inequality constraints involving fractional powers
        100 ≤ f(x) ≤ 3000
        0.1 ≤ xᵢ ≤ 10, i=1,...,6
        0.01 ≤ x₇ ≤ 10

    Source: problem 103 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Beck, Ecker [5], Dembo [22]

    Classification: PPR-P1-(8,9,10)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7 = y
        a = 0.5  # Parameter for HS103

        term1 = 10 * x1 * x4**2 * x7**a / (x2 * x6**3)
        term2 = 15 * x3 * x4 / (x1 * x2**2 * x5 * x7**0.5)
        term3 = 20 * x2 * x6 / (x1**2 * x4 * x5**2)
        term4 = 25 * x1**2 * x2**2 * x5**0.5 * x7 / (x3 * x6**2)

        return term1 + term2 + term3 + term4

    def y0(self):
        return jnp.array(
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution is given in Appendix A according to the problem
        return None  # Would need Appendix A data

    def expected_objective_value(self):
        return jnp.array(543.667958)

    def bounds(self):
        lower = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01])
        upper = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7 = y

        # Four inequality constraints from the AMPL formulation
        # Note: pycutest uses the convention g(x) ≤ 0, so we negate our
        # g(x) ≥ 0 constraints
        ineq1 = -(
            1
            - 0.5 * x1**0.5 * x7 / (x3 * x6**2)
            - 0.7 * x1**3 * x2 * x6 * x7**0.5 / x3**2
            - 0.2 * x3 * x6 ** (2 / 3) * x7 ** (0.25) / (x2 * x4**0.5)
        )

        ineq2 = -(
            1
            - 1.3 * x2 * x6 / (x1**0.5 * x3 * x5)
            - 0.8 * x3 * x6**2 / (x4 * x5)
            - 3.1 * x2**0.5 * x6 ** (1 / 3) / (x1 * x4**2 * x5)
        )

        ineq3 = -(
            1
            - 2 * x1 * x5 * x7 ** (1 / 3) / (x3**1.5 * x6)
            - 0.1 * x2 * x5 / (x3**0.5 * x6 * x7**0.5)
            - x2 * x3**0.5 * x5 / x1
            - 0.65 * x3 * x5 * x7 / (x2**2 * x6)
        )

        ineq4 = -(
            1
            - 0.2 * x2 * x5**0.5 * x7 ** (1 / 3) / (x1**2 * x4)
            - 0.3 * x1**0.5 * x2**2 * x3 * x4 ** (1 / 3) * x7**0.25 / x5 ** (2 / 3)
            - 0.4 * x3 * x5 * x7**0.75 / (x1**3 * x2**2)
            - 0.5 * x4 * x7**0.5 / x3**2
        )

        # Objective bound constraint
        # Note: The problem has bounds 100 ≤ f(x) ≤ 3000
        # But pycutest from SIF file only includes the upper bound as a constraint
        f_val = self.objective(y, self.args())
        obj_constraint = f_val - 3000.0  # f(x) - 3000 ≤ 0

        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4, obj_constraint])
        return None, inequality_constraints
