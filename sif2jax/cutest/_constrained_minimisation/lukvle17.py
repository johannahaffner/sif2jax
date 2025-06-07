import jax.numpy as jnp

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

    def y0(self):
        # Starting point: x_i = 2 for all i
        return jnp.full(self.n, 2.0)

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
        # Equality constraints - vectorized

        if n_c == 0:
            return jnp.array([]), None

        # Pre-allocate constraint array
        constraints = jnp.zeros(n_c)

        # Compute l values for all k
        k_values = jnp.arange(1, n_c + 1)
        l_values = 4 * ((k_values - 1) // 3)

        # Split k values by modulo 3
        mask_mod1 = (k_values % 3) == 1
        mask_mod2 = (k_values % 3) == 2
        mask_mod0 = (k_values % 3) == 0

        # Type 1 constraints: k ≡ 1 (mod 3)
        # c_k = x_{l+1}^2 + 3x_{l+2}
        l1 = l_values[mask_mod1]
        valid1 = l1 + 1 < n
        if jnp.any(valid1):
            l1_valid = l1[valid1]
            constraints = constraints.at[jnp.where(mask_mod1)[0][valid1]].set(
                y[l1_valid] ** 2 + 3 * y[l1_valid + 1]
            )

        # Type 2 constraints: k ≡ 2 (mod 3)
        # c_k = x_{l+3}^2 + x_{l+4} - 2x_{l+5}
        l2 = l_values[mask_mod2]
        valid2 = l2 + 4 < n
        if jnp.any(valid2):
            l2_valid = l2[valid2]
            constraints = constraints.at[jnp.where(mask_mod2)[0][valid2]].set(
                y[l2_valid + 2] ** 2 + y[l2_valid + 3] - 2 * y[l2_valid + 4]
            )

        # Type 3 constraints: k ≡ 0 (mod 3)
        # c_k = x_{l+2}^2 - x_{l+5}
        l0 = l_values[mask_mod0]
        valid0 = l0 + 4 < n
        if jnp.any(valid0):
            l0_valid = l0[valid0]
            constraints = constraints.at[jnp.where(mask_mod0)[0][valid0]].set(
                y[l0_valid + 1] ** 2 - y[l0_valid + 4]
            )

        return constraints, None
