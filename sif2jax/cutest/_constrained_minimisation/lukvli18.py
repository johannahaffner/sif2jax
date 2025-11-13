import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLI18(AbstractConstrainedMinimisation):
    """LUKVLI18 - Chained modified HS53 problem.

    Problem 5.18 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained modified HS53 function:
    f(x) = Σ[i=1 to (n-1)/4] [(x_{j+1} - x_{j+2})^4 + (x_{j+2} + x_{j+3} - 2)^2 +
                               (x_{j+4} - 1)^2 + (x_{j+5} - 1)^2]
    where j = 4(i-1), l = 4*div(k-1,3)

    Subject to inequality constraints:
    c_k(x) = x_{l+1}^2 + 3x_{l+2} ≤ 0, for k ≡ 1 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+3}^2 + x_{l+4} - 2x_{l+5} ≤ 0, for k ≡ 2 (mod 3), 1 ≤ k ≤ n_C
    c_k(x) = x_{l+2}^2 - x_{l+5} ≤ 0, for k ≡ 0 (mod 3), 1 ≤ k ≤ n_C
    where n_C = 3(n-1)/4

    Starting point: x_i = 2 for i = 1, ..., n

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
        # Chained modified HS53 function - fully vectorized
        num_groups = (n - 1) // 4

        # For each group i=1..num_groups, we have j = 4*(i-1)
        # We need y[j] through y[j+4]
        j = 4 * jnp.arange(num_groups)  # j values in 0-based

        # Pad y to ensure safe indexing
        y_padded = jnp.pad(y, (0, 5), mode="constant", constant_values=0.0)

        # Extract elements for all groups using advanced indexing
        y_j = y_padded[j]  # y[j]
        y_j1 = y_padded[j + 1]  # y[j+1]
        y_j2 = y_padded[j + 2]  # y[j+2]
        y_j3 = y_padded[j + 3]  # y[j+3]
        y_j4 = y_padded[j + 4]  # y[j+4]

        # Compute all terms at once
        terms = (
            (y_j - y_j1) ** 4
            + (y_j1 + y_j2 - 2) ** 2
            + (y_j3 - 1) ** 2
            + (y_j4 - 1) ** 2
        )

        return jnp.sum(terms)

    @property
    def y0(self):
        # Starting point: x_i = 2 for all i
        return inexact_asarray(jnp.full(self.n, 2.0))

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
        n_c = 3 * (n - 1) // 4

        if n_c == 0:
            return None, jnp.array([])

        # Pad y to ensure safe indexing
        y_padded = jnp.pad(y, (0, 10), mode="constant", constant_values=0.0)

        def constraint_1(i):
            # Type 1: constraint index i where i % 3 == 0
            # Maps to k = i // 3 * 3 in 0-based indexing
            # C(K): X(K)^2 + 3*X(K+1)
            k = (i // 3) * 3
            return y_padded[k] ** 2 + 3 * y_padded[k + 1]

        def constraint_2(i):
            # Type 2: constraint index i where i % 3 == 1
            # Maps to k = (i - 1) // 3 * 3 in 0-based indexing
            # C(K+1): X(K+2)^2 + X(K+3) - 2*X(K+4)
            k = ((i - 1) // 3) * 3
            return y_padded[k + 2] ** 2 + y_padded[k + 3] - 2 * y_padded[k + 4]

        def constraint_0(i):
            # Type 3: constraint index i where i % 3 == 2
            # Maps to k = (i - 2) // 3 * 3 in 0-based indexing
            # C(K+2): X(K+1)^2 - X(K+4)
            k = ((i - 2) // 3) * 3
            return y_padded[k + 1] ** 2 - y_padded[k + 4]

        # Generate all constraint indices
        indices = jnp.arange(n_c)

        # Compute all constraint types using vmap
        c1_vals = jax.vmap(constraint_1)(indices)
        c2_vals = jax.vmap(constraint_2)(indices)
        c0_vals = jax.vmap(constraint_0)(indices)

        # Select appropriate constraint based on index modulo 3
        mod3 = indices % 3
        inequalities = jnp.where(
            mod3 == 0, c1_vals, jnp.where(mod3 == 1, c2_vals, c0_vals)
        )

        return None, inequalities
