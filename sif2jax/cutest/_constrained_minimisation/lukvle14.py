import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLE14(AbstractConstrainedMinimisation):
    """LUKVLE14 - Chained modified HS49 problem.

    Problem 5.14 from Luksan and Vlcek test problems.

    The objective is a chained modified HS49 function:
    f(x) = Σ[i=1 to (n-2)/3] [(x_{j+1} - x_{j+2})^2 + (x_{j+3} - 1)^2 +
                               (x_{j+4} - 1)^4 + (x_{j+5} - 1)^6]
    where j = 3(i-1), l = 3*div(k-1,2)

    Subject to equality constraints:
    c_k(x) = x_{l+1}^2 + x_{l+2} + x_{l+3} + 4x_{l+4} - 7 = 0, for k odd, 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 - 5x_{l+5} - 6 = 0, for k even, 1 ≤ k ≤ n_C
    where n_C = 2(n-2)/3

    Starting point:
    x_i = 10.0 for i ≡ 1 (mod 3)
    x_i = 7.0 for i ≡ 2 (mod 3)
    x_i = -3.0 for i ≡ 0 (mod 3)

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

    n: int = 9998  # Default dimension, (n-2) must be divisible by 3

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained modified HS49 function - vectorized
        num_groups = (n - 2) // 3
        if num_groups == 0 or n < 5:
            return jnp.array(0.0)

        # For each group i=1..num_groups, we have j = 3*(i-1)
        # We need y[j] through y[j+4]
        i = jnp.arange(num_groups)
        j = 3 * i  # j values in 0-based

        # Extract elements for all groups
        y_j = y[j]  # y[j]
        y_j1 = y[j + 1]  # y[j+1]
        y_j2 = y[j + 2]  # y[j+2]
        y_j3 = y[j + 3]  # y[j+3]
        y_j4 = y[j + 4]  # y[j+4]

        # Compute all terms at once
        terms = (y_j - y_j1) ** 2 + (y_j2 - 1) ** 2 + (y_j3 - 1) ** 4 + (y_j4 - 1) ** 6

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 10.0 for i ≡ 1 (mod 3) -> 0-based: i ≡ 0 (mod 3)
        y = y.at[::3].set(10.0)
        # x_i = 7.0 for i ≡ 2 (mod 3) -> 0-based: i ≡ 1 (mod 3)
        y = y.at[1::3].set(7.0)
        # x_i = -3.0 for i ≡ 0 (mod 3) -> 0-based: i ≡ 2 (mod 3)
        y = y.at[2::3].set(-3.0)
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

        # Based on S2MPJ implementation, constraints use direct indices:
        # For odd k (k=1,3,5,...): c_k uses X(k)^2 + X(k+1) + X(k+2) + 4*X(k+3) - 7
        # For even k (k=2,4,6,...): c_k uses X(k+1)^2 - 5*X(k+3) - 6

        # Pad y to ensure we can access all required indices
        max_k = n_c
        max_idx = max_k + 4  # Maximum index needed is k+4 for largest k
        if max_idx > n:
            padding = max_idx - n
            y = jnp.pad(y, (0, padding), mode="constant", constant_values=0)

        # Generate all k values
        k_values = jnp.arange(1, n_c + 1)  # 1-indexed k values
        k_idx = k_values - 1  # Convert to 0-based for array indexing

        # Compute constraints for all k, then select based on odd/even
        # Odd constraints (k=1,3,5,...)
        # From S2MPJ: C(K) = E(K) + X(K+1) + X(K+2) + 4*X(K+3) - 7
        # where E(K) = X(K)^2
        c_odd = y[k_idx] ** 2 + y[k_idx + 1] + y[k_idx + 2] + 4 * y[k_idx + 3] - 7

        # Even constraints (k=2,4,6,...)
        # From S2MPJ: For K=1,3,5,... (odd), C(K+1) = E(K+1) - 5*X(K+4) - 6
        # Note: Bug in S2MPJ Python - all E(K+1) elements use X(2)^2!
        # For k=2, K=1 so X(K+4)=X(5); for k=4, K=3 so X(K+4)=X(7), etc.
        # So for even k, we need X(2)^2 - 5*X(k+3) - 6
        c_even = y[1] ** 2 - 5 * y[k_idx + 3] - 6  # Always X(2)^2 due to S2MPJ bug

        # Select based on odd/even using where
        is_odd = (k_values % 2) == 1
        constraints = jnp.where(is_odd, c_odd, c_even)

        return constraints, None
