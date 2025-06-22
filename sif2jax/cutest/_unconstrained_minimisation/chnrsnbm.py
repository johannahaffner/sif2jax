import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class CHNRSNBM(AbstractUnconstrainedMinimisation):
    """A variable dimension version of the chained Rosenbrock function (CHNROSNB)
    by Luksan et al.

    This is a modification of the CHNROSNB function where the alpha values are
    determined by the formula: alpha_i = sin(i) * 1.5

    Source: problem 27 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    that is an extension of that proposed in
    Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#46 (n = 25) (p. 45).
    SIF input: Ph. Toint, Dec 1989.
              this version Nick Gould, June, 2013

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # Same formula as CHNROSNB but with alpha values from sin function
        # Vectorized computation for i in 2..n (0-based: 1..n-1)
        i_indices = jnp.arange(1, self.n)  # 1 to n-1

        # Alpha values are determined by sin(i) + 1.5
        # i_indices is [1, 2, ..., n-1], representing i=2..n in 1-based AMPL notation
        # So we need sin(2), sin(3), ..., sin(n), which is sin(i_indices + 1)
        alpha_vals = jnp.sin(i_indices + 1) + 1.5

        # Get x[i-1] and x[i] values
        x_i_minus_1 = y[i_indices - 1]  # y[0] to y[n-2]
        x_i = y[i_indices]  # y[1] to y[n-1]

        # Compute terms
        # Based on the SIF file, the scaling SCL = 1/(16*alpha^2) is applied
        # The group computes (X(I-1) - X(I)^2) and then the L2 squares it
        # With scaling, we get: (X(I-1) - X(I)^2)^2 / SCL = (X(I-1)
        #  - X(I)^2)^2 * 16*alpha^2
        term1 = (x_i_minus_1 - x_i**2) ** 2 * 16.0 * alpha_vals**2
        term2 = (x_i - 1.0) ** 2

        # Sum all terms
        return jnp.sum(term1 + term2)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return inexact_asarray(jnp.full(self.n, -1.0))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # According to the SIF file comment, the optimal objective value is 0.0
        return jnp.array(0.0)
