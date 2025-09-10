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

        # Based on SIF file analysis:
        # K loops as 1, 4, 7, 10, ... (increments by 3)
        # For each K, we generate C(K), C(K+1), C(K+2)
        # But the element definitions show:
        # - E(K) uses X(K)
        # - E(K+1) uses X(K+2)
        # - E(K+2) uses X(K+1)

        # Extend y with zeros to safely access all indices
        extended_length = n + 20
        y_extended = jnp.zeros(extended_length)
        y_extended = y_extended.at[:n].set(y)

        constraints = []

        # Generate constraints in groups of 3
        k = 0  # Start at 0 for 0-based indexing (corresponds to K=1 in SIF)
        while len(constraints) < n_c:
            # C(K): X(K)^2 + 3*X(K+1) using E(K) which is X(K)^2
            c_k = y_extended[k] ** 2 + 3 * y_extended[k + 1]
            constraints.append(c_k)

            if len(constraints) >= n_c:
                break

            # C(K+1): X(K+2)^2 + X(K+3) - 2*X(K+4) using E(K+1) which is X(K+2)^2
            c_k1 = y_extended[k + 2] ** 2 + y_extended[k + 3] - 2 * y_extended[k + 4]
            constraints.append(c_k1)

            if len(constraints) >= n_c:
                break

            # C(K+2): X(K+1)^2 - X(K+4) using E(K+2) which is X(K+1)^2
            c_k2 = y_extended[k + 1] ** 2 - y_extended[k + 4]
            constraints.append(c_k2)

            k += 3  # Move to next group (K increments by 3 in SIF)

        return None, jnp.array(constraints[:n_c])
