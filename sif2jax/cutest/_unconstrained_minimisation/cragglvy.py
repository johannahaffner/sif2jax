import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CRAGGLVY(AbstractUnconstrainedMinimisation):
    """Extended Cragg and Levy problem.

    This problem is a sum of m sets of 5 groups,
    There are 2m+2 variables. The Hessian matrix is 7-diagonal.

    Source: problem 32 in
    Ph. L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    See also Buckley#18
    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    m: int = 2499  # Number of group sets (default 2499, n=5000)
    # Other suggested values: 1, 4, 24, 49, 249, 499, 2499
    n: int = 0  # Number of variables (will be set in __init__)

    def __init__(self):
        self.n = 2 * self.m + 2  # n = 2m + 2

    def objective(self, y, args):
        del args
        m = self.m

        # Create an array to hold the sum of 5 groups for each set
        terms = jnp.zeros(m)

        # Define function to compute the terms for each set i
        def compute_set_terms(i):
            # Convert from 1-based AMPL indices to 0-based Python indices
            # For i in 1..m (AMPL), we have i in 0..m-1 (Python)
            # x[2*i-1] (AMPL) = y[2*i] (Python) for i starting at 0
            # x[2*i] (AMPL) = y[2*i+1] (Python)
            # x[2*i+1] (AMPL) = y[2*i+2] (Python)
            # x[2*i+2] (AMPL) = y[2*i+3] (Python)
            i2_minus_1 = 2 * i  # x[2*i-1] in AMPL
            i2 = 2 * i + 1  # x[2*i] in AMPL
            i2_plus_1 = 2 * i + 2  # x[2*i+1] in AMPL
            i2_plus_2 = 2 * i + 3  # x[2*i+2] in AMPL

            # Group A(i) = (exp(x_{2i-1}) - x_{2i})^4
            a_i = (jnp.exp(y[i2_minus_1]) - y[i2]) ** 4

            # Group B(i) = 100*(x_{2i} - x_{2i+1})^6
            b_i = 100.0 * (y[i2] - y[i2_plus_1]) ** 6

            # Group C(i) = (tan(x_{2i+1} - x_{2i+2}) + x_{2i+1} - x_{2i+2})^4
            c_arg = y[i2_plus_1] - y[i2_plus_2]
            c_i = (jnp.tan(c_arg) + c_arg) ** 4

            # Group D(i) = (x_{2i-1})^8
            d_i = y[i2_minus_1] ** 8

            # Group F(i) = (x_{2i+2} - 1)^2
            f_i = (y[i2_plus_2] - 1.0) ** 2

            # Sum all terms for this set
            return a_i + b_i + c_i + d_i + f_i

        # Create an array of indices (0 to m-1)
        indices = jnp.arange(m)

        # Compute terms for all sets using vmap
        terms = jax.vmap(compute_set_terms)(indices)

        # Sum all terms
        return jnp.sum(terms)

    def y0(self):
        # Initial values from SIF file (all 2.0 except x1 = 1.0)
        y_init = 2.0 * jnp.ones(self.n)
        y_init = y_init.at[0].set(1.0)
        return inexact_asarray(y_init)

    def args(self):
        return None

    def expected_result(self):
        # The SIF file doesn't specify the optimal solution
        return None

    def expected_objective_value(self):
        # According to the SIF file, optimal objective values depend on the size
        # For m=249 (n=500): 167.45
        if self.m == 1:  # n = 4
            return jnp.array(0.0)
        elif self.m == 4:  # n = 10
            return jnp.array(1.886566)
        elif self.m == 24:  # n = 50
            return jnp.array(15.372)
        elif self.m == 49:  # n = 100
            return jnp.array(32.270)
        elif self.m == 249:  # n = 500
            return jnp.array(167.45)
        elif self.m == 499:  # n = 1000
            return jnp.array(336.42)
        elif self.m == 2499:  # n = 5000
            return jnp.array(1688.2)
        return None
