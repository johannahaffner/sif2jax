import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLI17(AbstractConstrainedMinimisation):
    """LUKVLI17 - Chained modified HS52 problem.

    Problem 5.17 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS52 function:
    f(x) = Σ[i=1 to (n-1)/4] [(4x_{j+1} - x_{j+2})^2 + (x_{j+2} + x_{j+3} - 2)^4 +
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
        # Chained modified HS52 function - vectorized
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
            (4 * y_j - y_j1) ** 2
            + (y_j1 + y_j2 - 2) ** 4
            + (y_j3 - 1) ** 2
            + (y_j4 - 1) ** 2
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point: x_i = 2 for all i
        return inexact_asarray(jnp.full(self.n, 2.0))

    def args(self):
        return None

    def expected_result(self):
        # Solution pattern based on problem structure
        return None  # Unknown exact solution

    def expected_objective_value(self):
        return None  # Unknown exact objective value

    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        n_c = 3 * (n - 1) // 4

        if n_c == 0:
            return None, jnp.array([])

        # Compute l values for all k
        k_values = jnp.arange(1, n_c + 1)
        l_values = 4 * ((k_values - 1) // 3)

        # Extend y with zeros to safely access all indices
        # Maximum index needed is max(l_values) + 5
        extended_length = n + 20  # Add sufficient padding
        y_extended = jnp.zeros(extended_length)
        y_extended = y_extended.at[:n].set(y)

        # Type 1 constraints: k ≡ 1 (mod 3)
        # c_k = x_{l+1}^2 + 3x_{l+2}
        c1 = y_extended[l_values] ** 2 + 3 * y_extended[l_values + 1]

        # Type 2 constraints: k ≡ 2 (mod 3)
        # c_k = x_{l+3}^2 + x_{l+4} - 2x_{l+5}
        c2 = (
            y_extended[l_values + 2] ** 2
            + y_extended[l_values + 3]
            - 2 * y_extended[l_values + 4]
        )

        # Type 3 constraints: k ≡ 0 (mod 3)
        # c_k = x_{l+2}^2 - x_{l+5}
        c3 = y_extended[l_values + 1] ** 2 - y_extended[l_values + 4]

        # Select constraints based on k modulo 3
        k_mod3 = k_values % 3
        constraints = jnp.where(k_mod3 == 1, c1, jnp.where(k_mod3 == 2, c2, c3))

        return None, constraints
