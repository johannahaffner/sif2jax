import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLE16(AbstractConstrainedMinimisation):
    """LUKVLE16 - Chained modified HS51 problem.

    Problem 5.16 from Luksan and Vlcek test problems.

    The objective is a chained modified HS51 function:
    f(x) = Σ[i=1 to (n-1)/4] [(x_{j+1} - x_{j+2})^4 + (x_{j+2} + x_{j+3} - 2)^2 +
                               (x_{j+4} - 1)^2 + (x_{j+5} - 1)^2]
    where j = 4(i-1), l = 4*div(k-1,3)

    Subject to equality constraints:
    c_k(x) = x_{l+1}^2 + 3x_{l+2} - 4 = 0, for k ≡ 1 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 + x_{l+4} - 2x_{l+5} = 0, for k ≡ 2 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2}^2 - x_{l+5} = 0, for k ≡ 0 (mod 3), 1 ≤ k ≤ n_C
    where n_C = 3(n-1)/4

    Starting point:
    x_i = 2.5 for i ≡ 1 (mod 4)
    x_i = 0.5 for i ≡ 2 (mod 4)
    x_i = 2.0 for i ≡ 3 (mod 4)
    x_i = -1.0 for i ≡ 0 (mod 4)

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
        # Chained modified HS51 function - vectorized
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
            (y_j - y_j1) ** 4
            + (y_j1 + y_j2 - 2) ** 2
            + (y_j3 - 1) ** 2
            + (y_j4 - 1) ** 2
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 2.5 for i ≡ 1 (mod 4) -> 0-based: i ≡ 0 (mod 4)
        y = y.at[::4].set(2.5)
        # x_i = 0.5 for i ≡ 2 (mod 4) -> 0-based: i ≡ 1 (mod 4)
        y = y.at[1::4].set(0.5)
        # x_i = 2.0 for i ≡ 3 (mod 4) -> 0-based: i ≡ 2 (mod 4)
        y = y.at[2::4].set(2.0)
        # x_i = -1.0 for i ≡ 0 (mod 4) -> 0-based: i ≡ 3 (mod 4)
        y = y.at[3::4].set(-1.0)
        return y

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
            return jnp.array([]), None

        # From SIF file: DO loop with DI K 3, so K = 1, 4, 7, ...
        # For each K, we create C(K), C(K+1), C(K+2)
        # Vectorized implementation

        # Pad y to ensure we can access all required indices
        max_idx = n_c + 4  # Maximum index we might need
        if max_idx > n:
            padding = max_idx - n
            y = jnp.pad(y, (0, padding), mode="constant", constant_values=0)

        # Create indices for all constraints
        # K = 1, 4, 7, ... (1-based) -> 0, 3, 6, ... (0-based)
        k_indices = jnp.arange(0, n_c, 3)

        # C(K): 3*X(K+1) + E(K) - 4, where E(K) = X(K)^2
        c1_indices = k_indices
        c1 = 3 * y[c1_indices + 1] + y[c1_indices] ** 2 - 4

        # C(K+1): X(K+3) - 2*X(K+4) + E(K+1), where E(K+1) = X(K+2)^2
        c2_indices = k_indices + 1
        c2 = y[c2_indices + 2] - 2 * y[c2_indices + 3] + y[c2_indices + 1] ** 2

        # C(K+2): -X(K+4) + E(K+2), where E(K+2) = X(K+1)^2
        c3_indices = k_indices + 2
        c3 = -y[c3_indices + 2] + y[c3_indices - 1] ** 2

        # Interleave the constraints to maintain the original order
        constraints = jnp.zeros(n_c)
        constraints = constraints.at[::3].set(c1)
        constraints = constraints.at[1::3].set(c2[: len(constraints[1::3])])
        constraints = constraints.at[2::3].set(c3[: len(constraints[2::3])])

        return constraints, None
