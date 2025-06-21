import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS111(AbstractConstrainedMinimisation):
    """Problem 111 from the Hock-Schittkowski test collection.

    A 10-variable problem with exponential objective and equality constraints.

    f(x) = ∑exp(xⱼ)(cⱼ + xⱼ - ln(∑exp(xₖ)))
           j=1 to 10                   k=1 to 10

    where cⱼ values are from Appendix A

    Subject to:
        Three equality constraints involving exponentials
        -100 ≤ xᵢ ≤ 100, i=1,...,10

    Source: problem 111 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13], Himmelblau [29], White [58]

    Classification: GGR-P1-4
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

        # Calculate sum of exponentials
        exp_sum = jnp.sum(jnp.exp(y))
        log_exp_sum = jnp.log(exp_sum)

        # Calculate objective
        objective_sum = 0.0
        for j in range(10):
            objective_sum += jnp.exp(y[j]) * (c[j] + y[j] - log_exp_sum)

        return jnp.array(objective_sum)

    def y0(self):
        return jnp.array(
            [-2.3, -2.3, -2.3, -2.3, -2.3, -2.3, -2.3, -2.3, -2.3, -2.3]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                -3.201212,
                -1.912060,
                -2.444413,
                -6.537489,
                -0.7231524,
                -7.267738,
                -3.596711,
                -4.017769,
                -3.287462,
                -2.335582,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-47.76109026)

    def bounds(self):
        # Bounds from the PDF
        lower = jnp.array([-100.0] * 10)
        upper = jnp.array([100.0] * 10)
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = y

        # Three equality constraints from the PDF
        eq1 = (
            jnp.exp(x1)
            + 2 * jnp.exp(x2)
            + 2 * jnp.exp(x3)
            + jnp.exp(x6)
            + jnp.exp(x10)
            - 2
        )

        eq2 = jnp.exp(x4) + 2 * jnp.exp(x5) + jnp.exp(x6) + jnp.exp(x7) - 1

        eq3 = (
            jnp.exp(x3) + jnp.exp(x7) + jnp.exp(x8) + 2 * jnp.exp(x9) + jnp.exp(x10) - 1
        )

        equality_constraints = jnp.array([eq1, eq2, eq3])
        return equality_constraints, None
