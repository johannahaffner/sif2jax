import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS57(AbstractConstrainedMinimisation):
    """Problem 57 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear objective function with one inequality constraint
    and bounds on variables.

    f(x) = sum_{i=1}^{44} f_i(x)²

    where f_i(x) = b_i - x₁ - (0.49 - x₁)exp(-x₂(a_i - 8))

    Subject to:
        0.49x₂ - x₁x₂ - 0.09 ≥ 0
        0.4 ≤ x₁, -4 ≤ x₂

    The values of a_i and b_i are given in Appendix A of the original reference.

    Source: problem 57 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Gould [27]

    Classification: SQR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y

        # Data from the problem (44 data points)
        a = jnp.array(
            [
                8.0,
                8.0,
                10.0,
                10.0,
                10.0,
                10.0,
                12.0,
                12.0,
                12.0,
                12.0,
                14.0,
                14.0,
                14.0,
                16.0,
                16.0,
                16.0,
                18.0,
                18.0,
                20.0,
                20.0,
                20.0,
                22.0,
                22.0,
                22.0,
                24.0,
                24.0,
                24.0,
                26.0,
                26.0,
                26.0,
                28.0,
                28.0,
                30.0,
                30.0,
                30.0,
                32.0,
                32.0,
                34.0,
                36.0,
                36.0,
                38.0,
                38.0,
                40.0,
                42.0,
            ]
        )

        b = jnp.array(
            [
                0.49,
                0.49,
                0.48,
                0.47,
                0.48,
                0.47,
                0.46,
                0.46,
                0.45,
                0.43,
                0.45,
                0.43,
                0.43,
                0.44,
                0.43,
                0.43,
                0.46,
                0.45,
                0.42,
                0.42,
                0.43,
                0.41,
                0.41,
                0.40,
                0.42,
                0.40,
                0.40,
                0.41,
                0.40,
                0.41,
                0.41,
                0.40,
                0.40,
                0.40,
                0.38,
                0.41,
                0.40,
                0.40,
                0.41,
                0.38,
                0.40,
                0.40,
                0.39,
                0.39,
            ]
        )

        # Compute f_i(x) for all i
        f_i = b - x1 - (0.49 - x1) * jnp.exp(-x2 * (a - 8))

        # Sum of squares
        return jnp.sum(f_i**2)

    def y0(self):
        return jnp.array([0.42, 5.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.419952675, 1.284845629])

    def expected_objective_value(self):
        return jnp.array(0.02845966972)

    def bounds(self):
        lower = jnp.array([0.4, -4.0])
        upper = None
        return lower, upper

    def constraint(self, y):
        x1, x2 = y
        # Inequality constraint: 0.49x₂ - x₁x₂ - 0.09 ≥ 0
        ineq = 0.49 * x2 - x1 * x2 - 0.09
        inequality_constraints = jnp.array([ineq])
        return None, inequality_constraints
