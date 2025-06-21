import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class LIARWHD(AbstractUnconstrainedMinimisation):
    """Simplified version of the NONDIA problem.

    Source:
    G. Li,
    "The secant/finite difference algorithm for solving sparse
    nonlinear systems of equations",
    SIAM Journal on Optimization, (to appear), 1990.

    SIF input: Ph. Toint, Aug 1990.
    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    n: int = 5000  # Other suggested values: 36, 100, 500, 1000, 10000

    def objective(self, y, args):
        """Compute the objective function value.

        The objective consists of two parts from AMPL:
        1. sum {i in 1..N} 4*(-x[1]+x[i]^2)^2
        2. sum {i in 1..N} (x[i]-1.0)^2
        """
        x1 = y[0]

        # First term: 4*(-x[1]+x[i]^2)^2 for all i
        first_terms = 4.0 * (-x1 + y**2) ** 2

        # Second term: (x[i]-1.0)^2 for all i
        second_terms = (y - 1.0) ** 2

        return jnp.sum(first_terms) + jnp.sum(second_terms)

    def y0(self):
        """Initial point with all variables set to 4.0."""
        return jnp.ones(self.n) * 4.0

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is all ones according to the problem definition."""
        return jnp.ones(self.n)

    def expected_objective_value(self):
        """The solution value is 0.0."""
        return jnp.array(0.0)
