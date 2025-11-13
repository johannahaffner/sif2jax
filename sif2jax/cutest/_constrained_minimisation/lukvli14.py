import jax
import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI14(AbstractConstrainedMinimisation):
    """LUKVLI14 - Chained modified HS49 problem with inequality constraints.

    Problem 5.14 from Luksan and Vlcek test problems with inequality constraints.

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for unconstrained and equality
    constrained optimization", Technical Report 767, Inst. Computer Science,
    Academy of Sciences of the Czech Republic, 182 07 Prague, Czech Republic, 1999.

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
        num_groups = (n - 2) // 3

        # Avoid conditional by using where or handling empty case naturally
        # When num_groups is 0, arange returns empty array, sum returns 0.0

        # For each group i=1..num_groups, we have j = 3*(i-1)
        # We need y[j] through y[j+4]
        j = 3 * jnp.arange(num_groups)  # j values in 0-based

        # Pad y to ensure safe indexing (though should always be safe for valid n)
        # This makes the function more robust
        y_padded = jnp.pad(y, (0, 5), mode="constant", constant_values=0.0)

        # Extract elements for all groups using advanced indexing
        y_j = y_padded[j]  # y[j]
        y_j1 = y_padded[j + 1]  # y[j+1]
        y_j2 = y_padded[j + 2]  # y[j+2]
        y_j3 = y_padded[j + 3]  # y[j+3]
        y_j4 = y_padded[j + 4]  # y[j+4]

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
        return None  # Exact solution not specified, only objective value

    @property
    def expected_objective_value(self):
        return jnp.array(1.56415e04)  # From SIF file: LO SOLTN 1.56415E+04

    @property
    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        n_c = 2 * (n - 2) // 3

        def even(k):
            i = 2 * (k // 2)  # Base index for this constraint pair
            return y[i] ** 2 + y[i + 1] + y[i + 2] + 4 * y[i + 3] - 7

        def odd(k):
            i = 2 * (k // 2)  # Base index for this constraint pair
            return y[i + 2] ** 2 - 5 * y[i + 4] - 6

        ks = jnp.arange(0, n_c)
        even_elements = jax.vmap(even)(ks)
        odd_elements = jax.vmap(odd)(ks)

        inequalities = jnp.where(ks % 2 == 0, even_elements, odd_elements)

        return None, inequalities
