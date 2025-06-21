import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ARGTRIGLS(AbstractUnconstrainedMinimisation):
    """ARGTRIGLS function.

    Variable dimension trigonometric problem in least-squares form.
    This problem is a sum of n least-squares groups, each of
    which has n+1 nonlinear elements. Its Hessian matrix is dense.

    Source: Problem 26 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 200  # SIF file suggests 10, 50, 100, or 200

    def objective(self, y, args):
        del args
        n = self.n

        # Based on AMPL model argtrig.mod, converted to least squares
        # Each residual: i*(cos(x[i])+sin(x[i])) + sum {j in 1..N} cos(x[j]) - (N+i)
        # Note: AMPL uses 1-based indexing

        # Compute sum of cosines for all variables
        sum_cos = jnp.sum(jnp.cos(y))

        # Compute residuals vectorized
        i_values = inexact_asarray(jnp.arange(1, n + 1))  # 1-based indices
        sincos_terms = i_values * (jnp.cos(y) + jnp.sin(y))
        residuals = sincos_terms + sum_cos - (n + i_values)

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1/n as specified in the SIF file
        return inexact_asarray(jnp.ones(self.n) / self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention: *LO SOLTN 0.0
        return jnp.array(0.0)
