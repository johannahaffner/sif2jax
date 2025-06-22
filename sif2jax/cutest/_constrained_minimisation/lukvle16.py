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
        num_triplets = n_c // 3
        constraints = []

        for i in range(num_triplets):
            k = 3 * i + 1  # K = 1, 4, 7, ... in 1-based
            k_idx = k - 1  # Convert to 0-based

            # C(K): 3*X(K+1) + E(K) - 4
            # E(K) uses X(K) with SQR
            if k_idx + 1 < n:
                c1 = 3 * y[k_idx + 1] + y[k_idx] ** 2 - 4
                constraints.append(c1)

            # C(K+1): X(K+3) - 2*X(K+4) + E(K+1)
            # E(K+1) uses X(K+2) with SQR
            if k_idx + 4 < n:
                c2 = y[k_idx + 3] - 2 * y[k_idx + 4] + y[k_idx + 2] ** 2
                constraints.append(c2)

            # C(K+2): -X(K+4) + E(K+2)
            # E(K+2) uses X(K+1) with SQR
            if k_idx + 4 < n:
                c3 = -y[k_idx + 4] + y[k_idx + 1] ** 2
                constraints.append(c3)

        return jnp.array(constraints), None
