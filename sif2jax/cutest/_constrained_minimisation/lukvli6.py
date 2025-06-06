import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI6(AbstractConstrainedMinimisation):
    """LUKVLI6 - Generalized Broyden banded function with exponential inequality
    constraints.

    Problem 5.6 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a generalized Broyden banded function:
    f(x) = Σ[i=1 to n] |(2 + 5x_i^2)x_i + 1 + Σ[j∈J_i] x_j(1 + x_j)|^p
    where p = 7/3, J_i = {j : max(1, i-5) ≤ j ≤ min(n, i+1)}

    Subject to inequality constraints:
    c_k(x) = 4x_{2k} - (x_{2k-1} - x_{2k+1})exp(x_{2k-1} - x_{2k} - x_{2k+1}) - 3 ≤ 0,
    for k = 1, ..., div(n,2)

    Starting point: x_i = 3 for i = 1, ..., n

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for
    unconstrained and equality constrained optimization",
    Technical Report 767, Inst. Computer Science, Academy of Sciences
    of the Czech Republic, 182 07 Prague, Czech Republic, 1999

    Equality constraints changed to inequalities

    SIF input: Nick Gould, April 2001

    Classification: OOR2-AY-V-V
    """

    n: int = 9999  # Default dimension, can be overridden

    def objective(self, y, args):
        del args
        n = len(y)
        p = 7.0 / 3.0
        # Generalized Broyden banded function - vectorized

        # Main terms: (2 + 5x_i^2)x_i + 1
        main_terms = (2 + 5 * y**2) * y + 1

        # For the banded sum, we need to compute a sliding window sum
        # Each element i needs sum of x_j(1 + x_j) for j in [max(0, i-5), min(n-1, i+1)]
        # This is a convolution-like operation

        # Pre-compute x_j(1 + x_j) for all elements
        x_terms = y * (1 + y)

        # Compute window sums directly using vectorized operations
        window_sums = jnp.zeros(n)
        for offset in range(-5, 2):  # -5, -4, -3, -2, -1, 0, 1
            # Determine valid indices for this offset
            if offset < 0:
                # Can add y[i+offset]^2 to position i for i >= -offset
                window_sums = window_sums.at[-offset:].add(x_terms[: n + offset])
            elif offset > 0:
                # Can add y[i+offset]^2 to position i for i < n-offset
                window_sums = window_sums.at[: n - offset].add(x_terms[offset:])
            else:
                # offset == 0, add all elements
                window_sums = window_sums + x_terms

        # Combine terms
        all_terms = main_terms + window_sums

        # Apply the power function
        return jnp.sum(jnp.abs(all_terms) ** p)

    def y0(self):
        # Starting point: x_i = 3 for all i
        return jnp.full(self.n, 3.0)

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
        # Inequality constraints
        constraints = []

        # Constraints from k=1 to div(n,2)
        for k in range(1, n // 2 + 1):
            # Convert 1-based to 0-based indexing
            idx_2k_minus_1 = 2 * k - 2  # x_{2k-1} in 1-based = y[2*k-2] in 0-based
            idx_2k = 2 * k - 1  # x_{2k} in 1-based = y[2*k-1] in 0-based
            idx_2k_plus_1 = 2 * k  # x_{2k+1} in 1-based = y[2*k] in 0-based

            # Handle boundary case
            if idx_2k_plus_1 < n:
                ck = (
                    4 * y[idx_2k]
                    - (y[idx_2k_minus_1] - y[idx_2k_plus_1])
                    * jnp.exp(y[idx_2k_minus_1] - y[idx_2k] - y[idx_2k_plus_1])
                    - 3
                )
                constraints.append(ck)

        inequality_constraints = (
            jnp.array(constraints) if constraints else jnp.array([])
        )
        return None, inequality_constraints
