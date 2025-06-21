import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS110(AbstractBoundedMinimisation):
    """Problem 110 from the Hock-Schittkowski test collection.

    A 10-variable problem with logarithmic objective function and simple bounds.

    f(x) = ∑[(ln(xᵢ - 2))² + (ln(10 - xᵢ))²] - (∏xᵢ)^0.2
           i=1 to 10                              i=1 to 10

    Subject to:
        2.001 ≤ xᵢ ≤ 9.999, i=1,...,10

    Source: problem 110 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Himmelblau [29], Paviani [48]

    Classification: GBR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Sum of log terms using vectorized operations
        log_sum = jnp.sum((jnp.log(y - 2)) ** 2 + (jnp.log(10 - y)) ** 2)

        # Product of all elements
        product = jnp.prod(y)

        return log_sum - product**0.2

    def y0(self):
        return jnp.array(
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        )  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF (all variables have the same value)
        return jnp.array(
            [
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
                9.35025655,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-45.77846971)

    def bounds(self):
        # Bounds from the PDF
        lower = jnp.array(
            [2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001]
        )
        upper = jnp.array(
            [9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999]
        )
        return (lower, upper)
