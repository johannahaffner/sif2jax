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

        # Extract variables using stride-2 slices
        x0 = y[: 2 * m : 2]  # x[2*i-1] in AMPL, i=0..m-1
        x1 = y[1 : 2 * m + 1 : 2]  # x[2*i] in AMPL
        x2 = y[2 : 2 * m + 2 : 2]  # x[2*i+1] in AMPL
        x3 = y[3 : 2 * m + 3 : 2]  # x[2*i+2] in AMPL

        # Group A(i) = (exp(x_{2i-1}) - x_{2i})^4
        a = (jnp.exp(x0) - x1) ** 4

        # Group B(i) = 100*(x_{2i} - x_{2i+1})^6
        b = 100.0 * (x1 - x2) ** 6

        # Group C(i) = (tan(x_{2i+1} - x_{2i+2}) + x_{2i+1} - x_{2i+2})^4
        c_arg = x2 - x3
        c = (jnp.tan(c_arg) + c_arg) ** 4

        # Group D(i) = (x_{2i-1})^8
        d = x0**8

        # Group F(i) = (x_{2i+2} - 1)^2
        f = (x3 - 1.0) ** 2

        # Sum all terms
        return jnp.sum(a + b + c + d + f)

    @property
    def y0(self):
        # Initial values from SIF file (all 2.0 except x1 = 1.0)
        y_init = 2.0 * jnp.ones(self.n)
        y_init = y_init.at[0].set(1.0)
        return inexact_asarray(y_init)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The SIF file doesn't specify the optimal solution
        return None

    @property
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
