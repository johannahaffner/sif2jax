import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLE17(AbstractConstrainedMinimisation):
    """LUKVLE17 - Chained modified HS52 problem.

    Problem 5.17 from Luksan and Vlcek test problems.

    The objective is a chained modified HS52 function:
    f(x) = Σ[i=1 to (n-1)/4] [(4x_{j+1} - x_{j+2})^2 + (x_{j+2} + x_{j+3} - 2)^4 +
                               (x_{j+4} - 1)^2 + (x_{j+5} - 1)^2]
    where j = 4(i-1), l = 4*div(k-1,3)

    Subject to equality constraints:
    c_k(x) = x_{l+1}^2 + 3x_{l+2} = 0, for k ≡ 1 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 + x_{l+4} - 2x_{l+5} = 0, for k ≡ 2 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2}^2 - x_{l+5} = 0, for k ≡ 0 (mod 3), 1 ≤ k ≤ n_C
    where n_C = 3(n-1)/4

    Starting point: x_i = 2 for i = 1, ..., n

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for
    unconstrained and equality constrained optimization",
    Technical Report 767, Inst. Computer Science, Academy of Sciences
    of the Czech Republic, 182 07 Prague, Czech Republic, 1999

    SIF input: Nick Gould, April 2001

    Classification: OOR2-AY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 9997  # Default dimension, (n-1) must be divisible by 4

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained modified HS52 function - vectorized
        num_groups = (n - 1) // 4

        # Extract the relevant indices for vectorized computation
        # For each group i, we need indices j = 4*(i-1) which gives us:
        # j = 0, 4, 8, ... up to 4*(num_groups-1)
        # We need y[j], y[j+1], y[j+2], y[j+3], y[j+4]

        j_indices = jnp.arange(num_groups) * 4

        # Extract slices for vectorized computation
        y_j = y[j_indices]
        y_j1 = y[j_indices + 1]
        y_j2 = y[j_indices + 2]
        y_j3 = y[j_indices + 3]
        y_j4 = y[j_indices + 4]

        # Compute all terms at once
        terms = (
            (4 * y_j - y_j1) ** 2
            + (y_j1 + y_j2 - 2) ** 4
            + (y_j3 - 1) ** 2
            + (y_j4 - 1) ** 2
        )

        return jnp.sum(terms)

    @property
    def y0(self):
        # Starting point: x_i = 2 for all i
        return inexact_asarray(jnp.full(self.n, 2.0))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution pattern based on problem structure
        return None  # Unknown exact solution

    @property
    def expected_objective_value(self):
        return None  # Unknown exact objective value

    @property
    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        n_c = 3 * (n - 1) // 4

        if n_c == 0:
            return jnp.array([]), None

        # n_c is always divisible by 3 when (n-1) % 4 == 0.
        # For group m (m = 0 .. n_c/3 - 1), k = 3m and the three constraints are:
        #   c[3m]   = y[3m]^2   + 3*y[3m+1]
        #   c[3m+1] = y[3m+2]^2 + y[3m+3] - 2*y[3m+4]
        #   c[3m+2] = y[3m+1]^2 - y[3m+4]
        # Using strided slices (stride 3) rather than gathers for AD-friendliness.
        num_groups = n_c // 3
        stop = 3 * num_groups  # last k + 1 = 3*num_groups

        y0_ = y[0:stop:3]        # y[3m]
        y1_ = y[1:stop + 1:3]    # y[3m+1]
        y2_ = y[2:stop + 2:3]    # y[3m+2]
        y3_ = y[3:stop + 3:3]    # y[3m+3]
        y4_ = y[4:stop + 4:3]    # y[3m+4]

        c0 = y0_ ** 2 + 3.0 * y1_
        c1 = y2_ ** 2 + y3_ - 2.0 * y4_
        c2 = y1_ ** 2 - y4_

        # Interleave into a single (3*num_groups,) array.
        constraints = jnp.stack([c0, c1, c2], axis=1).reshape(-1)

        return constraints, None
