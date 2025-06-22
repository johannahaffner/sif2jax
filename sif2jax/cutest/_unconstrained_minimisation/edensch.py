import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class EDENSCH(AbstractUnconstrainedMinimisation):
    """Extended Dennis and Schnabel problem.

    This problem involves a sum of quartic terms for each variable
    and quadratic terms for products of adjacent variables.

    Source: problem 157 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2000  # Default dimension

    def objective(self, y, args):
        del args

        # From AMPL model:
        # sum {i in 1..N-1} ( (x[i]-2)^4 + (x[i]*x[i+1]-2*x[i+1])^2 +
        # (x[i+1]+1)^2 ) + 16
        # Converting to 0-based indexing: i from 0 to n-2

        # Vectorized computation for i = 0 to n-2
        x_i = y[:-1]  # x[0] to x[n-2]
        x_i_plus_1 = y[1:]  # x[1] to x[n-1]

        term1 = (x_i - 2.0) ** 4
        term2 = (x_i * x_i_plus_1 - 2.0 * x_i_plus_1) ** 2
        term3 = (x_i_plus_1 + 1.0) ** 2

        result = jnp.sum(term1 + term2 + term3) + 16.0

        return result

    def y0(self):
        # Starting point from SIF file: all variables = 8.0
        return jnp.full(self.n, 8.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        # Expected minimum value depends on dimension
        if self.n == 36:
            return jnp.array(219.28)
        elif self.n == 2000:
            return jnp.array(1.20032e4)
        return None
