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

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 9999  # Default dimension, can be overridden

    def objective(self, y, args):
        del args
        n = len(y)
        p = 7.0 / 3.0

        # Main terms: (2 + 5x_i^2)x_i + 1
        main_terms = (2 + 5 * y**2) * y + 1

        # Pre-compute x_j(1 + x_j) for all elements
        x_terms = y * (1 + y)

        # Efficient computation of window sums using cumulative sum
        # Each element i needs sum of x_j(1 + x_j) for j in [max(0, i-5), min(n-1, i+1)]
        # We can use cumulative sum and subtract to get window sums

        # Pad x_terms for easier indexing
        padded_x_terms = jnp.pad(x_terms, (6, 1), mode="constant")

        # Compute cumulative sum
        cumsum = jnp.cumsum(padded_x_terms)

        # For each position i, we want sum from (i-5) to (i+1) inclusive
        # In padded array, position i corresponds to i+6
        # We want sum from (i+6-5) to (i+6+1) = sum from (i+1) to (i+7)
        # This is cumsum[i+7] - cumsum[i]
        window_sums = cumsum[7 : n + 7] - cumsum[:n]

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
        if n < 3:
            return None, jnp.array([])

        # Vectorized constraint computation
        # We need triplets (x_{2k-1}, x_{2k}, x_{2k+1}) for k = 1, ..., n//2
        # But only include if 2k+1 <= n (i.e., 2k < n)
        num_constraints = min(n // 2, (n - 1) // 2)

        if num_constraints == 0:
            return None, jnp.array([])

        # Extract the required elements using slicing
        x_2k_minus_1 = y[0 : 2 * num_constraints : 2]  # x_1, x_3, x_5, ...
        x_2k = y[1 : 2 * num_constraints + 1 : 2]  # x_2, x_4, x_6, ...
        x_2k_plus_1 = y[2 : 2 * num_constraints + 2 : 2]  # x_3, x_5, x_7, ...

        # Compute all constraints at once
        inequality_constraints = (
            4 * x_2k
            - (x_2k_minus_1 - x_2k_plus_1) * jnp.exp(x_2k_minus_1 - x_2k - x_2k_plus_1)
            - 3
        )

        return None, inequality_constraints
