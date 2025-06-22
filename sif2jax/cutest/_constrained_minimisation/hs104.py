import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS104(AbstractConstrainedMinimisation):
    """Problem 104 from the Hock-Schittkowski test collection.

    An 8-variable optimal reactor design problem with complex fractional power
    constraints.

    f(x) = .4x₁¹·⁶⁷x₇⁻·⁶⁷ + .4x₂¹·⁶⁷x₈⁻·⁶⁷ + 10 - x₁ - x₂

    Subject to:
        Four inequality constraints with fractional powers
        1 ≤ f(x) ≤ 4.2
        0.1 ≤ xᵢ ≤ 10, i=1,...,8

    Source: problem 104 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Dembo [22], Rijckaert [53]

    Classification: PPR-P1-11
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        term1 = 0.4 * x1 ** (0.67) * x7 ** (-0.67)
        term2 = 0.4 * x2 ** (0.67) * x8 ** (-0.67)
        term3 = 10 - x1 - x2

        return term1 + term2 + term3

    def y0(self):
        return jnp.array(
            [6.0, 3.0, 0.4, 0.2, 6.0, 6.0, 1.0, 0.5]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                6.465114,
                2.232709,
                0.6673975,
                0.5957564,
                5.932676,
                5.527235,
                1.013322,
                0.4006682,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(3.9511634396)

    def bounds(self):
        lower = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        upper = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # Four inequality constraints from the AMPL formulation
        # Note: pycutest uses the convention g(x) ≤ 0, so we negate our
        # g(x) ≥ 0 constraints
        ineq1 = -(1 - 0.0588 * x5 * x7 - 0.1 * x1)

        ineq2 = -(1 - 0.0588 * x6 * x8 - 0.1 * x1 - 0.1 * x2)

        ineq3 = -(1 - 4 * x3 / x5 - 2 / (x3**0.71 * x5) - 0.0588 * x7 / x3**1.3)

        ineq4 = -(1 - 4 * x4 / x6 - 2 / (x4**0.71 * x6) - 0.0588 * x8 / x4**1.3)

        # Objective bounds constraint: 1 ≤ f(x) ≤ 4.2
        # Note: pycutest from SIF file only includes the lower bound as a constraint
        f_val = self.objective(y, self.args())
        obj_constraint = f_val - 1.0  # f(x) - 1 ≥ 0

        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4, obj_constraint])
        return None, inequality_constraints
