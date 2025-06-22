import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI2(AbstractConstrainedMinimisation):
    """LUKVLI2 - Chained Wood function with Broyden banded inequality constraints.

    Problem 5.2 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained Wood function:
    f(x) = Σ[i=1 to n/2-1] [100(x_{2i-1}^2 - x_{2i})^2 + (x_{2i-1} - 1)^2 +
                             90(x_{2i+1}^2 - x_{2i+2})^2 + (x_{2i+1} - 1)^2 +
                             10(x_{2i} + x_{2i+2} - 2)^2 + (x_{2i} - x_{2i+2})^2/10]

    Subject to inequality constraints:
    c_k(x) = (2 + 5x_{k+5}^2)x_{k+5} + 1 + Σ[i=k-5 to k+1] x_i(1 + x_i) ≤ 0,
    for k = 6, ..., n-2 (with appropriate adjustments for boundary cases)

    Starting point: x_i = -2 for i odd, x_i = 1 for i even

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

    n: int = 10000  # Default dimension, can be overridden

    def objective(self, y, args):
        # TODO: This implementation matches the structure in the SIF file but
        # differs from pycutest by a factor of ~34.5. The SIF file uses
        # group-separable structure where scale factors are applied in the
        # denominator, but even accounting for this there is a discrepancy.
        # This needs further investigation.
        del args
        n = len(y)
        # Chained Wood function - vectorized
        num_terms = n // 2 - 1
        if num_terms <= 0:
            return jnp.array(0.0)

        # Extract all required elements at once
        i = jnp.arange(num_terms)
        x_2i_minus_1 = y[2 * i]  # x_{2i-1} in 1-based
        x_2i = y[2 * i + 1]  # x_{2i} in 1-based
        x_2i_plus_1 = y[2 * i + 2]  # x_{2i+1} in 1-based
        x_2i_plus_2 = y[2 * i + 3]  # x_{2i+2} in 1-based

        # Compute all terms vectorized
        terms = (
            100 * (x_2i_minus_1**2 - x_2i) ** 2
            + (x_2i_minus_1 - 1) ** 2
            + 90 * (x_2i_plus_1**2 - x_2i_plus_2) ** 2
            + (x_2i_plus_1 - 1) ** 2
            + 10 * (x_2i + x_2i_plus_2 - 2) ** 2
            + (x_2i - x_2i_plus_2) ** 2 / 10
        )
        return jnp.sum(terms)

    def y0(self):
        # Starting point: x_i = -2 for i odd, x_i = 1 for i even
        y = jnp.zeros(self.n)
        # JAX uses 0-based indexing, so odd indices in the problem are even in JAX
        y = y.at[::2].set(-2.0)  # i = 1, 3, 5, ... (1-based) -> 0, 2, 4, ... (0-based)
        y = y.at[1::2].set(1.0)  # i = 2, 4, 6, ... (1-based) -> 1, 3, 5, ... (0-based)
        return y

    def args(self):
        return None

    def expected_result(self):
        # Solution is all ones
        return jnp.ones(self.n)

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        if n < 10:  # Need at least 10 elements for meaningful constraints
            return None, jnp.array([])

        # Inequality constraints - vectorized
        # Constraints from k=6 to n-2 (1-based), which is k=5 to n-3 (0-based)
        k_values = jnp.arange(5, n - 2)
        num_constraints = len(k_values)

        if num_constraints == 0:
            return None, jnp.array([])

        # k+5 in 1-based is k+4 in 0-based
        k_plus_5_idx = k_values + 4

        # (2 + 5x_{k+5}^2)x_{k+5} + 1 for all constraints
        main_terms = (2 + 5 * y[k_plus_5_idx] ** 2) * y[k_plus_5_idx] + 1

        # For the sum, we need x_i(1 + x_i) for i in [max(0, k-5), min(n-1, k+1)]
        # This is similar to the window sum in LUKVLI6
        # Pre-compute x_i(1 + x_i) for all elements
        x_terms = y * (1 + y)

        # Use cumulative sum approach for efficient window computation
        # Pad x_terms for easier indexing
        padded_x_terms = jnp.pad(x_terms, (6, 1), mode="constant")
        cumsum = jnp.cumsum(padded_x_terms)

        # For each k, we want sum from max(0, k-5) to min(n-1, k+1)
        # In padded array, position i corresponds to i+6
        # For k in original array, we want sum from (k-5) to (k+1) inclusive
        # In padded: from (k-5+6) to (k+1+6) = from (k+1) to (k+7)
        # This is cumsum[k+7] - cumsum[k]
        # But we need to adjust for the actual k_values which start at 5
        start_positions = k_values + 1
        end_positions = k_values + 7

        window_sums = cumsum[end_positions] - cumsum[start_positions - 1]

        inequality_constraints = main_terms + window_sums
        return None, inequality_constraints
