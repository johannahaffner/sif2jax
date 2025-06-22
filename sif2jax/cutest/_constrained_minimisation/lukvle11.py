import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLE11(AbstractConstrainedMinimisation):
    """LUKVLE11 - Chained HS46 problem.

    Problem 5.11 from Luksan and Vlcek test problems.

    The objective is a chained HS46 function:
    f(x) = Σ[i=1 to (n-2)/3] [(x_{j+1} - x_{j+2})^2 + (x_{j+3} - 1)^2 +
                               (x_{j+4} - 1)^4 + (x_{j+5} - 1)^6]
    where j = 3(i-1), l = 3*div(k-1,2)

    Subject to equality constraints:
    c_k(x) = x_{l+1}^2 x_{l+4} + sin(x_{l+4} - x_{l+5}) - 1 = 0, for k odd, 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2} + x_{l+3}^4 x_{l+4}^2 - 2 = 0, for k even, 1 ≤ k ≤ n_C
    where n_C = 2(n-2)/3

    Starting point:
    x_i = 2.0 for i ≡ 1 (mod 3)
    x_i = 1.5 for i ≡ 2 (mod 3)
    x_i = 0.5 for i ≡ 0 (mod 3)

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

    n: int = 9998  # Default dimension, must be divisible by 3

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained HS46 function - vectorized

        # We need groups of 5 consecutive elements starting at indices 0, 3, 6, ...
        # Number of complete groups
        num_groups = (n - 2) // 3
        if num_groups == 0:
            return jnp.array(0.0)

        # Create indices for the start of each group
        group_starts = jnp.arange(num_groups) * 3

        # Extract the 5 elements for each group
        # All groups should be valid since n is chosen appropriately
        # For n=9998, we have (9998-2)//3 = 3332 groups
        # Last group starts at 3331*3 = 9993, needs indices up to 9997

        # Extract elements for all groups at once
        x_j1 = y[group_starts]  # First element of each group
        x_j2 = y[group_starts + 1]  # Second element
        x_j3 = y[group_starts + 2]  # Third element
        x_j4 = y[group_starts + 3]  # Fourth element
        x_j5 = y[group_starts + 4]  # Fifth element

        # Compute all terms vectorized
        terms = (x_j1 - x_j2) ** 2 + (x_j3 - 1) ** 2 + (x_j4 - 1) ** 4 + (x_j5 - 1) ** 6

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 2.0 for i ≡ 1 (mod 3) -> 0-based: i ≡ 0 (mod 3)
        y = y.at[::3].set(2.0)
        # x_i = 1.5 for i ≡ 2 (mod 3) -> 0-based: i ≡ 1 (mod 3)
        y = y.at[1::3].set(1.5)
        # x_i = 0.5 for i ≡ 0 (mod 3) -> 0-based: i ≡ 2 (mod 3)
        y = y.at[2::3].set(0.5)
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
        n_c = 2 * (n - 2) // 3

        if n_c == 0:
            return jnp.array([]), None

        # Based on SIF file, constraints use direct indices:
        # For K=1,3,5,... (by 2):
        #   C(K) = EA(K) + EB(K) = X(K)²·X(K+3) + sin(X(K+3)-X(K+4)) - 1
        #   C(K+1) = X(K+1) + E(K+1) = X(K+1) + X(K+2)⁴·X(K+3)² - 2

        # Pad y to ensure we can access all required indices
        max_k = n_c
        max_idx = max_k + 4  # Maximum index needed
        if max_idx > n:
            padding = max_idx - n
            y = jnp.pad(y, (0, padding), mode="constant", constant_values=0)

        # Initialize constraints array
        constraints = jnp.zeros(n_c)

        # Vectorized processing of constraints in pairs
        # K = 1, 3, 5, ... corresponds to 0-based indices 0, 2, 4, ...
        odd_k_indices = jnp.arange(0, n_c, 2)  # 0-based indices for K=1,3,5,...

        # Process C(K) constraints: EA(K) + EB(K)
        # EA(K): X(K)² * X(K+3)
        # EB(K): sin(X(K+3) - X(K+4))
        c_k = (
            y[odd_k_indices] ** 2 * y[odd_k_indices + 3]
            + jnp.sin(y[odd_k_indices + 3] - y[odd_k_indices + 4])
            - 1
        )
        constraints = constraints.at[odd_k_indices].set(c_k)

        # Process C(K+1) constraints: X(K+1) + E(K+1)
        # E(K+1): X(K+2)² * X(K+3)
        even_k_indices = odd_k_indices + 1  # 0-based indices for K+1
        # Only process even indices that are within bounds
        valid_even = even_k_indices < n_c

        # Use jnp.where to avoid boolean indexing issues
        c_k1_all = (
            y[even_k_indices] + y[even_k_indices + 1] ** 2 * y[even_k_indices + 2] - 2
        )
        c_k1 = jnp.where(valid_even, c_k1_all, 0.0)

        # Only set valid indices
        constraints = constraints.at[even_k_indices].set(
            jnp.where(valid_even, c_k1, constraints[even_k_indices])
        )

        return constraints, None
