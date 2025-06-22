import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI15(AbstractConstrainedMinimisation):
    """LUKVLI15 - Chained modified HS50 problem with inequality constraints.

    Problem 5.15 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS50 function:
    f(x) = Σ[i=1 to (n-1)/4] [(x_{j+1} - x_{j+2})^2 + (x_{j+2} - x_{j+3})^2 +
                               (x_{j+3} - x_{j+4})^4 + (x_{j+4} - x_{j+5})^4]
    where j = 4(i-1), l = 4*div(k-1,3)

    Subject to inequality constraints:
    c_k(x) = x_{l+1}^2 + 2x_{l+2} + 3x_{l+3} - 6 ≤ 0, for k ≡ 1 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2}^2 + 2x_{l+3} + 3x_{l+4} - 6 ≤ 0, for k ≡ 2 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 + 2x_{l+4} + 3x_{l+5} - 6 ≤ 0, for k ≡ 0 (mod 3), 1 ≤ k ≤ n_C
    where n_C = 3(n-1)/4

    Starting point:
    x_i = 35.0 for i ≡ 1 (mod 4)
    x_i = 11.0 for i ≡ 2 (mod 4)
    x_i = 5.0 for i ≡ 3 (mod 4)
    x_i = -5.0 for i ≡ 0 (mod 4)

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
        # Chained modified HS50 function - vectorized
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
            (y_j - y_j1) ** 2
            + (y_j1 - y_j2) ** 2
            + (y_j2 - y_j3) ** 4
            + (y_j3 - y_j4) ** 4
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 35.0 for i ≡ 1 (mod 4) -> 0-based: i ≡ 0 (mod 4)
        y = y.at[::4].set(35.0)
        # x_i = 11.0 for i ≡ 2 (mod 4) -> 0-based: i ≡ 1 (mod 4)
        y = y.at[1::4].set(11.0)
        # x_i = 5.0 for i ≡ 3 (mod 4) -> 0-based: i ≡ 2 (mod 4)
        y = y.at[2::4].set(5.0)
        # x_i = -5.0 for i ≡ 0 (mod 4) -> 0-based: i ≡ 3 (mod 4)
        y = y.at[3::4].set(-5.0)
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
            return None, jnp.array([])

        # Based on S2MPJ implementation, constraints use direct indices:
        # For K=1,4,7,... (by 3):
        #   C(K) = E(K) + 2*X(K+1) + 3*X(K+2) - 6 <= 0, where E(K) = X(K)^2
        #   C(K+1) = E(K+1) + 2*X(K+2) + 3*X(K+3) - 6 <= 0, where E(K+1) = X(K+1)^2
        #   C(K+2) = E(K+2) + 2*X(K+3) + 3*X(K+4) - 6 >= 0, where E(K+2) = X(K+2)^2
        # Note: C(K+2) is >= so we need to negate it for <= format

        # Pad y to ensure we can access all required indices
        max_k = n_c
        max_idx = max_k + 4  # Maximum index needed
        if max_idx > n:
            padding = max_idx - n
            y = jnp.pad(y, (0, padding), mode="constant", constant_values=0)

        # Vectorized constraint computation
        # Create indices for all constraints
        k_indices = jnp.arange(n_c)

        # All constraints have the same pattern: y[k]^2 + 2*y[k+1] + 3*y[k+2] - 6
        constraints = (
            y[k_indices] ** 2 + 2 * y[k_indices + 1] + 3 * y[k_indices + 2] - 6
        )

        return None, constraints
