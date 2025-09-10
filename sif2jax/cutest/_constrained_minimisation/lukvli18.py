import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLI18(AbstractConstrainedMinimisation):
    """LUKVLI18 - Chained modified HS53 problem.

    Problem 5.18 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS53 function:
    f(x) = Σ[i=1 to (n-1)/4] [(x_{j+1} - x_{j+2})^4 + (x_{j+2} + x_{j+3} - 2)^2 +
                               (x_{j+4} - 1)^2 + (x_{j+5} - 1)^2]
    where j = 4(i-1), l = 4*div(k-1,3)

    Subject to inequality constraints:
    c_k(x) = x_{l+1}^2 + 3x_{l+2} ≤ 0, for k ≡ 1 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 + x_{l+4} - 2x_{l+5} ≤ 0, for k ≡ 2 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2}^2 - x_{l+5} ≤ 0, for k ≡ 0 (mod 3), 1 ≤ k ≤ n_C
    where n_C = 3(n-1)/4

    Starting point: x_i = 2 for i = 1, ..., n

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for
    unconstrained and equality constrained optimization",
    Technical Report 767, Inst. Computer Science, Academy of Sciences
    of the Czech Republic, 182 07 Prague, Czech Republic, 1999


    Equality constraints changed to inequalities

    SIF input: Nick Gould, April 2001

    Classification: OOR2-AY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 9997  # Default dimension, (n-1) must be divisible by 4

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained modified HS53 function - vectorized
        num_groups = (n - 1) // 4
        if num_groups == 0 or n < 5:
            return jnp.array(0.0)

        # For each group i=1..num_groups, we have j = 4*(i-1)
        # We need y[j] through y[j+4]
        i = jnp.arange(num_groups)
        j = 4 * i  # j values in 0-based

        # Extract elements for all groups
        y_j = y[j]  # y[j]
        y_j1 = y[j + 1]  # y[j+1]
        y_j2 = y[j + 2]  # y[j+2]
        y_j3 = y[j + 3]  # y[j+3]
        y_j4 = y[j + 4]  # y[j+4]

        # Compute all terms at once
        terms = (
            (y_j - y_j1) ** 4
            + (y_j1 + y_j2 - 2) ** 2
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
            return None, jnp.array([])

        # Compute constraints based on problem description
        # c_k for k = 1 to n_c
        # l = 4 * ((k-1) // 3) for each k

        # Extend y with zeros to safely access all indices
        extended_length = n + 20
        y_extended = jnp.zeros(extended_length)
        y_extended = y_extended.at[:n].set(y)

        constraints = []

        for k in range(1, n_c + 1):
            l = 4 * ((k - 1) // 3)  # 1-based formula
            k_mod = k % 3

            if k_mod == 1:
                # Type 1: x_{l+1}^2 + 3*x_{l+2}
                # Convert to 0-based: y[l]^2 + 3*y[l+1]
                c = y_extended[l] ** 2 + 3 * y_extended[l + 1]
            elif k_mod == 2:
                # Type 2: x_{l+3}^2 + x_{l+4} - 2*x_{l+5}
                # Convert to 0-based: y[l+2]^2 + y[l+3] - 2*y[l+4]
                c = y_extended[l + 2] ** 2 + y_extended[l + 3] - 2 * y_extended[l + 4]
            else:  # k_mod == 0
                # Type 3: x_{l+2}^2 - x_{l+5}
                # Convert to 0-based: y[l+1]^2 - y[l+4]
                c = y_extended[l + 1] ** 2 - y_extended[l + 4]

            constraints.append(c)

        return None, jnp.array(constraints)
