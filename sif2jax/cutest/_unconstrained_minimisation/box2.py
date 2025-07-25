import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class BOX2(AbstractUnconstrainedMinimisation):
    """Box problem in 2 variables, obtained by fixing X3 = 1 in BOX3.

    This function is a nonlinear least squares with 10 groups.
    Each group has 2 nonlinear elements of exponential type.

    Source: Problem 11 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification: SXR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables (x3 is fixed at 1.0)

    @property
    def y0(self):
        # Starting point from SIF file (only x1, x2)
        return jnp.array([0.0, 10.0])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        Minimize sum_{i=1}^{10} G(i)^2
        where G(i) = x3 * COEFF * (exp(-0.1*i*x1) - exp(-0.1*i*x2))
        and COEFF = -exp(-0.1*i) + exp(-i)

        Note: x3 is fixed at 1.0
        """
        x1, x2 = y

        # x3 is fixed at 1.0 for BOX2
        x3 = 1.0

        # Vectorized implementation
        i_vals = jnp.arange(1, 11)  # i = 1 to 10
        ri = inexact_asarray(i_vals)
        mti = inexact_asarray(-0.1) * ri

        # Elements with parameter T = -0.1*i
        a_i = jnp.exp(mti * x1)
        b_i = jnp.exp(mti * x2)

        # COEFF = -exp(-0.1*i) + exp(-i) from SIF file
        coeff = -jnp.exp(mti) + jnp.exp(-ri)

        # Group values: G(i) = x3 * coeff + (a_i - b_i)
        g_i = x3 * coeff + a_i - b_i

        # Sum of squares
        return jnp.sum(g_i**2)

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(0.0)
