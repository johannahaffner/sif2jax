import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS112(AbstractConstrainedMinimisation):
    """Problem 112 from the Hock-Schittkowski test collection.

    A 10-variable chemical equilibrium problem with logarithmic objective.

    f(x) = ∑xⱼ(cⱼ + ln(xⱼ/(x₁ + ... + x₁₀)))
           j=1 to 10

    where cⱼ values are from Appendix A

    Subject to:
        Three equality constraints
        1.E-6 ≤ xᵢ, i=1,...,10

    Source: problem 112 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13], Himmelblau [29], White [58]

    Classification: GLR-P1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # cⱼ values from Appendix A (simplified for implementation)
        # In a complete implementation, these would be the actual values from Appendix A
        c = jnp.array(
            [
                -6.089,
                -17.164,
                -34.054,
                -5.914,
                -24.721,
                -14.986,
                -24.100,
                -10.708,
                -26.662,
                -22.179,
            ]
        )

        # Calculate sum of all variables
        x_sum = jnp.sum(y)

        # Calculate objective
        objective_sum = 0.0
        for j in range(10):
            objective_sum += y[j] * (c[j] + jnp.log(y[j] / x_sum))

        return jnp.array(objective_sum)

    def y0(self):
        return jnp.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.01773548,
                0.08200180,
                0.8825646,
                0.7233256e-3,
                0.4907851,
                0.4335469e-3,
                0.01727298,
                0.007765639,
                0.01984929,
                0.05269826,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-47.707579)

    def bounds(self):
        # Bounds from the PDF
        lower = jnp.array([1.0e-6] * 10)
        upper = jnp.array([jnp.inf] * 10)
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y

        # Three equality constraints from the PDF
        eq1 = x1 + 2 * x2 + 2 * x3 + x6 + x10 - 2

        eq2 = x4 + 2 * x5 + x6 + x7 - 1

        eq3 = x3 + x7 + x8 + 2 * x9 + x10 - 1

        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
