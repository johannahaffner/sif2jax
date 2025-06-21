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
        # Chained modified HS53 function - vectorized
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
            (y_j - y_j1) ** 4
            + (y_j1 + y_j2 - 2) ** 2
            + (y_j3 - 1) ** 2
            + (y_j4 - 1) ** 2
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point: x_i = 2 for all i
        return inexact_asarray(jnp.full(self.n, 2.0))

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

        # Vectorized constraint computation
        # We have three types of constraints cycling with period 3
        # Type 1 (k ≡ 1 mod 3): c_k = x_{l+1}^2 + 3x_{l+2}
        # Type 2 (k ≡ 2 mod 3): c_k = x_{l+3}^2 + x_{l+4} - 2x_{l+5}
        # Type 3 (k ≡ 0 mod 3): c_k = x_{l+2}^2 - x_{l+5}
        # where l = 4*((k-1)//3)

        # Number of constraint triplets
        num_triplets = n_c // 3
        remainder = n_c % 3

        # Generate l values for each triplet
        i = jnp.arange(num_triplets + (1 if remainder > 0 else 0))
        l = 4 * i  # l values in 0-based

        # Type 1 constraints (need l+1 < n)
        valid_type1 = l + 1 < n
        l_type1 = l[valid_type1]
        if len(l_type1) > 0:
            y_l = y[l_type1]  # x_{l+1} in 1-based
            y_l1 = y[l_type1 + 1]  # x_{l+2} in 1-based
            c_type1 = y_l**2 + 3 * y_l1
        else:
            c_type1 = jnp.array([])

        # Type 2 constraints (need l+4 < n)
        valid_type2 = l + 4 < n
        l_type2 = l[valid_type2]
        if len(l_type2) > 0:
            y_l2 = y[l_type2 + 2]  # x_{l+3} in 1-based
            y_l3 = y[l_type2 + 3]  # x_{l+4} in 1-based
            y_l4 = y[l_type2 + 4]  # x_{l+5} in 1-based
            c_type2 = y_l2**2 + y_l3 - 2 * y_l4
        else:
            c_type2 = jnp.array([])

        # Type 3 constraints (need l+4 < n)
        l_type3 = l[:num_triplets]  # Only take the first num_triplets
        valid_type3 = l_type3 + 4 < n
        l_type3 = l_type3[valid_type3]
        if len(l_type3) > 0:
            y_l1 = y[l_type3 + 1]  # x_{l+2} in 1-based
            y_l4 = y[l_type3 + 4]  # x_{l+5} in 1-based
            c_type3 = y_l1**2 - y_l4
        else:
            c_type3 = jnp.array([])

        # Interleave constraints in proper order
        min_len = min(len(c_type1), len(c_type2), len(c_type3))
        if min_len > 0:
            constraints = jnp.zeros(3 * min_len)
            constraints = constraints.at[::3].set(c_type1[:min_len])
            constraints = constraints.at[1::3].set(c_type2[:min_len])
            constraints = constraints.at[2::3].set(c_type3[:min_len])

            # Add remaining constraints
            remaining = []
            if len(c_type1) > min_len:
                remaining.extend(c_type1[min_len:])
            if len(c_type2) > min_len:
                remaining.extend(c_type2[min_len:])

            if remaining:
                constraints = jnp.concatenate([constraints, jnp.array(remaining)])
        else:
            # Concatenate whatever we have
            constraints = jnp.concatenate([c_type1, c_type2, c_type3])

        return None, constraints[:n_c]
