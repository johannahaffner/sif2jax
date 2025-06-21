import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI3(AbstractConstrainedMinimisation):
    """LUKVLI3 - Chained Powell singular function with simplified trig-exp
    inequality constraints.

    Problem 5.3 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained Powell singular function:
    f(x) = Σ[i=1 to n/2] [(x_{2i-1} + 10x_{2i})^2 + 5(x_{2i+1} - x_{2i+2})^2 +
                          (x_{2i} - 2x_{2i+1})^4 + 10(x_{2i-1} - x_{2i+2})^4]

    Subject to inequality constraints:
    c_1(x) = 3x_1^3 + 2x_2 - 5 + sin(x_1 - x_2)sin(x_1 + x_2) ≤ 0
    c_2(x) = 4x_{n-1} - x_{n-1} exp(x_{n-1} - x_n) - 3 ≤ 0

    Starting point:
    x_i = 3  for i ≡ 1 (mod 4)
    x_i = -1 for i ≡ 2 (mod 4)
    x_i = 0  for i ≡ 3 (mod 4)
    x_i = 1  for i ≡ 0 (mod 4)

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
        del args
        n = len(y)
        # Chained Powell singular function - vectorized
        # Sum from i=1 to n/2, each term uses x_{2i-1}, x_{2i}, x_{2i+1}, x_{2i+2}
        # This means we need elements from index 0 to 2*n/2+1 = n+1, but we only have n
        # So we sum up to the largest valid i
        num_complete_groups = (n - 2) // 2  # Ensure we have all 4 elements
        if num_complete_groups <= 0 or n < 4:
            return jnp.array(0.0)

        i = jnp.arange(num_complete_groups)
        x_2i_minus_1 = y[2 * i]  # x_{2i-1} in 1-based
        x_2i = y[2 * i + 1]  # x_{2i} in 1-based
        x_2i_plus_1 = y[2 * i + 2]  # x_{2i+1} in 1-based
        x_2i_plus_2 = y[2 * i + 3]  # x_{2i+2} in 1-based

        # Compute all terms vectorized
        terms = (
            (x_2i_minus_1 + 10 * x_2i) ** 2
            + 5 * (x_2i_plus_1 - x_2i_plus_2) ** 2
            + (x_2i - 2 * x_2i_plus_1) ** 4
            + 10 * (x_2i_minus_1 - x_2i_plus_2) ** 4
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 3  for i ≡ 1 (mod 4) -> 0-based: i ≡ 0 (mod 4)
        y = y.at[::4].set(3.0)
        # x_i = -1 for i ≡ 2 (mod 4) -> 0-based: i ≡ 1 (mod 4)
        y = y.at[1::4].set(-1.0)
        # x_i = 0  for i ≡ 3 (mod 4) -> 0-based: i ≡ 2 (mod 4)
        y = y.at[2::4].set(0.0)
        # x_i = 1  for i ≡ 0 (mod 4) -> 0-based: i ≡ 3 (mod 4)
        y = y.at[3::4].set(1.0)
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
        # Two inequality constraints computed directly as JAX array

        # c_1: 3x_1^3 + 2x_2 - 5 + sin(x_1 - x_2)sin(x_1 + x_2) ≤ 0
        c1 = 3 * y[0] ** 3 + 2 * y[1] - 5 + jnp.sin(y[0] - y[1]) * jnp.sin(y[0] + y[1])

        # c_2: 4x_{n-1} - x_{n-1} exp(x_{n-1} - x_n) - 3 ≤ 0
        c2 = 4 * y[n - 2] - y[n - 2] * jnp.exp(y[n - 2] - y[n - 1]) - 3

        inequality_constraints = jnp.array([c1, c2])
        return None, inequality_constraints
