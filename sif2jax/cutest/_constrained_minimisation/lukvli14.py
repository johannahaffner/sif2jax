import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI14(AbstractConstrainedMinimisation):
    """LUKVLI14 - Chained modified HS49 problem with inequality constraints.

    Problem 5.14 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS49 function:
    f(x) = Σ[i=1 to (n-2)/3] [(x_{j+1} - x_{j+2})^2 + (x_{j+3} - 1)^2 +
                               (x_{j+4} - 1)^4 + (x_{j+5} - 1)^6]
    where j = 3(i-1), l = 3*div(k-1,2)

    Subject to inequality constraints:
    c_k(x) = x_{l+1}^2 + x_{l+2} + x_{l+3} + 4x_{l+4} - 7 ≤ 0, for k odd, 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 - 5x_{l+5} - 6 ≤ 0, for k even, 1 ≤ k ≤ n_C
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

    Equality constraints changed to inequalities

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

    @property
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
        n_c = 2 * (n - 2) // 3

        if n_c == 0:
            return None, jnp.array([])

        # From SIF file: K loops as 1, 3, 5, ... (increments by 2)
        # For each K:
        # - C(K) = E(K) + X(K+1) + X(K+2) + 4*X(K+3) - 7
        #   where E(K) = X(K)^2
        # - C(K+1) = E(K+1) - 5*X(K+4) - 6
        #   where E(K+1) = X(K+2)^2
        #
        # So for K=1,3,5,7,... we create constraint pairs:
        # K=1: C(1) = X(1)^2 + X(2) + X(3) + 4*X(4) - 7
        #      C(2) = X(3)^2 - 5*X(5) - 6
        # K=3: C(3) = X(3)^2 + X(4) + X(5) + 4*X(6) - 7
        #      C(4) = X(5)^2 - 5*X(7) - 6
        # K=5: C(5) = X(5)^2 + X(6) + X(7) + 4*X(8) - 7
        #      C(6) = X(7)^2 - 5*X(9) - 6
        # etc.

        # Extend y with zeros to safely access all indices
        extended_length = n + 20
        y_extended = jnp.zeros(extended_length)
        y_extended = y_extended.at[:n].set(y)

        constraints = []

        # Generate constraints in pairs
        k = 0  # Start at 0 for 0-based indexing (corresponds to K=1 in SIF)
        constraint_count = 0

        while constraint_count < n_c:
            # C(K): X(K)^2 + X(K+1) + X(K+2) + 4*X(K+3) - 7
            # Note: K in SIF is 1-indexed, so K=1 means index 0 in our arrays
            c_k = (
                y_extended[k] ** 2
                + y_extended[k + 1]
                + y_extended[k + 2]
                + 4 * y_extended[k + 3]
                - 7
            )
            constraints.append(c_k)
            constraint_count += 1

            if constraint_count >= n_c:
                break

            # C(K+1): X(K+1)^2 - 5*X(K+4) - 6
            # Note: SIF file says E(K+1) uses X(K+2), but pycutest uses X(K+1)
            # Both use X(K+4) for the linear term
            c_k1 = y_extended[k + 1] ** 2 - 5 * y_extended[k + 4] - 6
            constraints.append(c_k1)
            constraint_count += 1

            k += 2  # K increments by 2 in the SIF loop

        return None, jnp.array(constraints)
