import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - constraint function inconsistencies
# There appear to be minor but relevant inconsistencies in the constraint
# function implementation, likely due to an indexing error. The constraint
# values don't match pycutest exactly, suggesting the indices used in the
# constraint formulation may be off by one or have other systematic errors.
class LUKVLE12(AbstractConstrainedMinimisation):
    """LUKVLE12 - Chained HS47 problem.

    Problem 5.12 from Luksan and Vlcek test problems.

    The objective is a chained HS47 function:
    f(x) = Σ[i=1 to (n-1)/4] [(x_{j+1} - x_{j+2})^2 + (x_{j+2} - x_{j+3})^2 +
                               (x_{j+3} - x_{j+4})^4 + (x_{j+4} - x_{j+5})^4]
    where j = 4(i-1)

    Subject to equality constraints (for K = 1, 4, 7, 10, ...):
    c_K(x) = x_K + x_{K+1}^2 + x_{K+2}^2 - 3 = 0
    c_{K+1}(x) = x_{K+1} + x_{K+3} + x_{K+2}^2 - 1 = 0
    c_{K+2}(x) = x_K * x_{K+4} - 1 = 0
    Special case: c_3(x) = -x_1 * x_5 + 1 = 0 (sign flipped)
    where n_C = 3(n-1)/4

    Starting point:
    x_i = 2.0 for i ≡ 1 (mod 4)
    x_i = 1.5 for i ≡ 2 (mod 4)
    x_i = -1.0 for i ≡ 3 (mod 4)
    x_i = 0.5 for i ≡ 0 (mod 4)

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

    n: int = 9997  # Default dimension, must satisfy (n-1) divisible by 4

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained HS47 function - vectorized

        # We need groups of 5 consecutive elements starting at indices 0, 4, 8, ...
        num_groups = (n - 1) // 4
        if num_groups == 0:
            return jnp.array(0.0)

        # Create indices for the start of each group
        # Since we know n >= 5 and num_groups = (n-1)//4,
        # all groups should be valid because group i needs elements from 4*i to 4*i+4
        # and the last group starts at 4*(num_groups-1) = 4*((n-1)//4 - 1) < n-4
        i = jnp.arange(num_groups)
        j = 4 * i  # j values in 0-based

        # Extract elements for all groups
        x_j1 = y[j]  # y[j]
        x_j2 = y[j + 1]  # y[j+1]
        x_j3 = y[j + 2]  # y[j+2]
        x_j4 = y[j + 3]  # y[j+3]
        x_j5 = y[j + 4]  # y[j+4]

        # Compute all terms vectorized
        terms = (
            (x_j1 - x_j2) ** 2
            + (x_j2 - x_j3) ** 2
            + (x_j3 - x_j4) ** 4
            + (x_j4 - x_j5) ** 4
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 2.0 for i ≡ 1 (mod 4) -> 0-based: i ≡ 0 (mod 4)
        y = y.at[::4].set(2.0)
        # x_i = 1.5 for i ≡ 2 (mod 4) -> 0-based: i ≡ 1 (mod 4)
        y = y.at[1::4].set(1.5)
        # x_i = -1.0 for i ≡ 3 (mod 4) -> 0-based: i ≡ 2 (mod 4)
        y = y.at[2::4].set(-1.0)
        # x_i = 0.5 for i ≡ 0 (mod 4) -> 0-based: i ≡ 3 (mod 4)
        y = y.at[3::4].set(0.5)
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

        # Build constraints using K indices from SIF file
        # K takes values 1, 4, 7, 10, ... (in 1-based) with DI K 3
        constraints = []

        # Process each K value
        k = 1
        while len(constraints) < n_c:
            # Convert to 0-based
            k_idx = k - 1

            # Type 1: C(K) = X(K) + X(K+1)² + X(K+2)² - 3 = 0
            if len(constraints) < n_c and k_idx + 2 < n:
                c1 = y[k_idx] + y[k_idx + 1] ** 2 + y[k_idx + 2] ** 2 - 3
                constraints.append(c1)

            # Type 2: C(K+1) = X(K+1) + X(K+3) + X(K+2)² - 1 = 0
            if len(constraints) < n_c and k_idx + 3 < n:
                c2 = y[k_idx + 1] + y[k_idx + 3] + y[k_idx + 2] ** 2 - 1
                constraints.append(c2)

            # Type 3: C(K+2) = X(K) * X(K+4) - 1 = 0
            if len(constraints) < n_c and k_idx + 4 < n:
                c3 = y[k_idx] * y[k_idx + 4] - 1
                # Special case: First type 3 constraint (K=1, C(3)) needs sign flip
                if k == 1:
                    c3 = -c3
                constraints.append(c3)

            # Next K value (K increments by 3)
            k += 3

        # Return constraints as a JAX array
        return jnp.array(constraints), None
