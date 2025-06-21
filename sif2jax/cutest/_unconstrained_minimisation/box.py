import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOX(AbstractUnconstrainedMinimisation):
    """A quartic function with a non-trivial sparsity pattern.

    Source: N. Gould, private communication.
    SIF input: N. Gould, Jan 2009

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Other suggested values are 10, 100, 1000, and 100000

    def objective(self, y, args):
        del args
        n = self.n
        n_half = n // 2

        # Sum of (x_i + x_1)^2 terms
        a_terms = jnp.sum((y + y[0]) ** 2)

        # Sum of (x_i + x_n)^2 terms
        b_terms = jnp.sum((y + y[-1]) ** 2)

        # Sum of (x_i + x_{n/2})^2 terms
        c_terms = jnp.sum((y + y[n_half - 1]) ** 2)

        # Linear term: -0.5 * sum(x_i)
        # Note: This is NOT squared, it's a linear group according to the SIF file
        d_term = -0.5 * jnp.sum(y)

        # Sum of x_i^4 terms
        q_terms = jnp.sum(y**4)

        return a_terms + b_terms + c_terms + d_term + q_terms

    def y0(self):
        # Initial values not specified in SIF file
        # Using default all zeros as a reasonable starting point
        return jnp.zeros(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The optimal objective value bound is -(n-1) according to line 74
        return jnp.array(-(self.n - 1))
