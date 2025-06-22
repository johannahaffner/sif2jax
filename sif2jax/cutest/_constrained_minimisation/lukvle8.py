import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLE8(AbstractConstrainedMinimisation):
    """LUKVLE8 - Augmented Lagrangian function with discrete boundary value constraints.

    Problem 5.8 from Luksan and Vlcek test problems.

    The objective is an augmented Lagrangian function:
    f(x) = Σ[i=1 to n/5] [exp(∏[j=1 to 5] x_{5i+1-j}) +
                          10(Σ[j=1 to 5] x_{5i+1-j}^2 - 10 - λ_1)^2 +
                          (x_{5i-3}x_{5i-2} - 5x_{5i-1}x_{5i} - λ_2)^2 +
                          (x_{5i-4}^3 + x_{5i-3}^3 + 1 - λ_3)^2]
    where λ_1 = -0.002008, λ_2 = -0.001900, λ_3 = -0.000261

    Subject to equality constraints:
    c_k(x) = 2x_{k+1} + h^2(x_{k+1} + h(k+1) + 1)^3/2 - x_k - x_{k+2} = 0,
    for k = 1, ..., n-2, where h = 1/(n+1)

    Starting point:
    x_i = -1 for i odd
    x_i = 2 for i even

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

    n: int = 10000  # Default dimension, can be overridden

    def objective(self, y, args):
        del args
        # Constants
        lambda1 = -0.002008
        lambda2 = -0.001900
        lambda3 = -0.000261

        # Augmented Lagrangian function - vectorized
        # Work with complete groups of 5
        num_complete_groups = len(y) // 5
        if num_complete_groups == 0:
            return jnp.array(0.0)

        # Reshape into groups of 5 for easier indexing
        usable_length = num_complete_groups * 5
        y_groups = y[:usable_length].reshape(num_complete_groups, 5)

        # For each group, elements are ordered as:
        # [x_{5i-4}, x_{5i-3}, x_{5i-2}, x_{5i-1}, x_{5i}] for i=1,2,...

        # Product term: product of all 5 elements in reverse order
        # x_{5i}, x_{5i-1}, x_{5i-2}, x_{5i-3}, x_{5i-4}
        prod_terms = jnp.prod(y_groups[:, ::-1], axis=1)

        # Sum of squares of all 5 elements
        sum_sq = jnp.sum(y_groups**2, axis=1)

        # Extract specific elements for other terms
        x_5i_minus_4 = y_groups[:, 0]
        x_5i_minus_3 = y_groups[:, 1]
        x_5i_minus_2 = y_groups[:, 2]
        x_5i_minus_1 = y_groups[:, 3]
        x_5i = y_groups[:, 4]

        # Compute all terms
        term1 = jnp.exp(prod_terms)
        term2 = 10 * (sum_sq - 10 - lambda1) ** 2
        term3 = (
            10 * (x_5i_minus_3 * x_5i_minus_2 - 5 * x_5i_minus_1 * x_5i - lambda2) ** 2
        )
        term4 = 10 * (x_5i_minus_4**3 + x_5i_minus_3**3 + 1 - lambda3) ** 2

        return jnp.sum(term1 + term2 + term3 + term4)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = -1 for i odd -> 0-based: i even
        y = y.at[::2].set(-1.0)
        # x_i = 2 for i even -> 0-based: i odd
        y = y.at[1::2].set(2.0)
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
        if n < 3:
            return jnp.array([]), None

        h = 1.0 / (n + 1)

        # Vectorized constraint computation
        # For k = 1 to n-2 (1-based), we need x_k, x_{k+1}, x_{k+2}
        # In 0-based: for k = 0 to n-3, we need y[k], y[k+1], y[k+2]
        k_indices = jnp.arange(n - 2)

        # Extract required slices
        y_k = y[:-2]  # y[0] to y[n-3]
        y_k_plus_1 = y[1:-1]  # y[1] to y[n-2]
        y_k_plus_2 = y[2:]  # y[2] to y[n-1]

        # k+1 in 1-based indexing for the formula
        k_plus_1_1based = k_indices + 1

        # Compute all constraints at once
        h_float = inexact_asarray(h)
        k_plus_1_float = inexact_asarray(k_plus_1_1based)
        constraints = (
            2 * y_k_plus_1
            + h_float**2 * (y_k_plus_1 + h_float * k_plus_1_float + 1) ** 3 / 2
            - y_k
            - y_k_plus_2
        )

        return constraints, None
