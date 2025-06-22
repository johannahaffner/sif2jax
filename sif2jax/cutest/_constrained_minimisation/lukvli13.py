import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI13(AbstractConstrainedMinimisation):
    """LUKVLI13 - Chained modified HS48 problem with inequality constraints.

    Problem 5.13 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS48 function:
    f(x) = Σ[i=1 to (n-2)/3] [(x_{j+1} - 1)^2 + (x_{j+2} - x_{j+3})^2 +
                               (x_{j+4} - x_{j+5})^4]
    where j = 3(i-1)

    Subject to inequality constraints:
    c_k(x) = x_{l+1} + x_{l+2}^2 + x_{l+3} + x_{l+4} + x_{l+5} - 5 ≤ 0,
             for k odd, 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 - 2(x_{l+4} + x_{l+5}) - 3 ≤ 0, for k even, 1 ≤ k ≤ n_C
    where n_C = 2(n-2)/3, l = 3*div(k-1,2)

    Starting point:
    x_i = 3.0 for i ≡ 1 (mod 3)
    x_i = 5.0 for i ≡ 2 (mod 3)
    x_i = -3.0 for i ≡ 0 (mod 3)

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

    n: int = 9998  # Default dimension, must be even

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained modified HS48 function - vectorized
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
        terms = (y_j - 1) ** 2 + (y_j1 - y_j2) ** 2 + (y_j3 - y_j4) ** 4

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 3.0 for i ≡ 1 (mod 3) -> 0-based: i ≡ 0 (mod 3)
        y = y.at[::3].set(3.0)
        # x_i = 5.0 for i ≡ 2 (mod 3) -> 0-based: i ≡ 1 (mod 3)
        y = y.at[1::3].set(5.0)
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
            return None, jnp.array([])

        # Based on S2MPJ implementation, constraints use direct indices:
        # For odd k (k=1,3,5,...): c_k uses X(k), X(k+1)^2, X(k+2), X(k+3), 4*X(k+4)
        # For even k (k=2,4,6,...): c_k uses X(k+2)^2 - 2*(X(k+2) + X(k+3))

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
        # From S2MPJ: C(K) = X(K) + E(K) + X(K+2) + X(K+3) + 4*X(K+4) - 5
        # where E(K) = X(K+1)^2
        c_odd = (
            y[k_idx]
            + y[k_idx + 1] ** 2
            + y[k_idx + 2]
            + y[k_idx + 3]
            + 4 * y[k_idx + 4]
            - 5
        )

        # Even constraints (k=2,4,6,...)
        # From S2MPJ: C(K) = E(K) - 2*X(K+2) - 2*X(K+3) - 3
        # where E(K) = X(K+1)^2
        c_even = y[k_idx + 1] ** 2 - 2 * (y[k_idx + 2] + y[k_idx + 3]) - 3

        # Select based on odd/even using where
        is_odd = (k_values % 2) == 1
        constraints = jnp.where(is_odd, c_odd, c_even)

        return None, constraints
