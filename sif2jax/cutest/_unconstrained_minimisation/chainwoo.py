import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review
# TODO: This implementation requires verification against another CUTEst interface
class CHAINWOO(AbstractUnconstrainedMinimisation):
    """The chained Woods problem, a variant on Woods function.

    This problem is a sum of n/2 sets of 6 terms, each of which is
    assigned its own group. For a given set i, the groups are
    A(i), B(i), C(i), D(i), E(i) and F(i). Groups A(i) and C(i) contain 1
    nonlinear element each, denoted Y(i) and Z(i).

    This version uses a slightly unorthodox expression of Woods
    function as a sum of squares (see Buckley).

    Source: problem 8 in
    A.R.Conn,N.I.M.Gould and Ph.L.Toint,
    "Testing a class of methods for solving minimization
    problems with simple bounds on their variables,
    Mathematics of Computation 50, pp 399-430, 1988.

    SIF input: Nick Gould and Ph. Toint, Dec 1995.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4000  # Dimension of the problem (2*ns + 2)
    ns: int = 1999  # Number of sets (default 1999, which gives n=4000)

    def objective(self, y, args):
        del args

        # Based on AMPL model in chainwoo.mod
        # sum {i in 1..ns} (
        # 100*(x[2*i]-x[2*i-1]^2)^2 +
        # (1.0-x[2*i-1])^2 +
        # 90*(x[2*i+2]-x[2*i+1]^2)^2 +
        # (1.0-x[2*i+1])^2 +
        # 10*(x[2*i]+x[2*i+2]-2.0)^2 +
        # (x[2*i]-x[2*i+2])^2/10
        # )

        ns = self.ns

        # Vectorized computation
        # For i in 1..ns, we have indices:
        # x[2*i-1] (1-based) = y[2*i-2] (0-based)
        # x[2*i]   (1-based) = y[2*i-1] (0-based)
        # x[2*i+1] (1-based) = y[2*i]   (0-based)
        # x[2*i+2] (1-based) = y[2*i+1] (0-based)

        # Create index arrays
        idx_2i_minus_1 = 2 * jnp.arange(ns)  # 0, 2, 4, ...
        idx_2i = 2 * jnp.arange(ns) + 1  # 1, 3, 5, ...
        idx_2i_plus_1 = 2 * jnp.arange(ns) + 2  # 2, 4, 6, ...
        idx_2i_plus_2 = 2 * jnp.arange(ns) + 3  # 3, 5, 7, ...

        # Get the variable values
        x_2i_minus_1 = y[idx_2i_minus_1]
        x_2i = y[idx_2i]
        x_2i_plus_1 = y[idx_2i_plus_1]
        x_2i_plus_2 = y[idx_2i_plus_2]

        # Compute all terms
        term1 = 100.0 * (x_2i - x_2i_minus_1**2) ** 2
        term2 = (1.0 - x_2i_minus_1) ** 2
        term3 = 90.0 * (x_2i_plus_2 - x_2i_plus_1**2) ** 2
        term4 = (1.0 - x_2i_plus_1) ** 2
        term5 = 10.0 * (x_2i + x_2i_plus_2 - 2.0) ** 2
        term6 = (x_2i - x_2i_plus_2) ** 2 / 10.0

        # Sum all terms plus constant
        return 1.0 + jnp.sum(term1 + term2 + term3 + term4 + term5 + term6)

    def y0(self):
        # Initial values from SIF file
        y_init = jnp.full(self.n, -2.0)

        # Override specific elements
        y_init = y_init.at[0].set(-3.0)  # X1
        y_init = y_init.at[1].set(-1.0)  # X2
        y_init = y_init.at[2].set(-3.0)  # X3
        y_init = y_init.at[3].set(-1.0)  # X4

        return inexact_asarray(y_init)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # SIF file comment (line 154): optimal objective value is 0.0
        return jnp.array(0.0)
