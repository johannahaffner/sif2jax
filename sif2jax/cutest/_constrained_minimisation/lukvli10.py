import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI10(AbstractConstrainedMinimisation):
    """LUKVLI10 - Generalized Brown function with Broyden tridiagonal inequality
    constraints.

    Problem 5.10 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a generalized Brown function:
    f(x) = Σ[i=1 to n/2] [(x_{2i-1}^2)^(x_{2i}^2+1) + (x_{2i}^2)^(x_{2i-1}^2+1)]

    Subject to inequality constraints:
    c_k(x) = (3 - 2x_{k+1})x_{k+1} + 1 - x_k - 2x_{k+2} ≤ 0,
    for k = 1, ..., n-2

    Starting point:
    x_i = -1 for i odd
    x_i = 1 for i even

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
        # Generalized Brown function - vectorized
        num_pairs = n // 2
        if num_pairs == 0:
            return jnp.array(0.0)

        # Extract all odd and even indexed elements
        y_odd = y[::2][:num_pairs]  # x_{2i-1} elements (1-based)
        y_even = y[1::2][:num_pairs]  # x_{2i} elements (1-based)

        # According to the SIF file, the function is:
        # (x^2)^(y^2+1) + (y^2)^(x^2+1)
        # This is different from what was documented!

        # Compute all terms at once
        x_squared = y_odd * y_odd
        y_squared = y_even * y_even

        term1 = jnp.power(x_squared, y_squared + 1)
        term2 = jnp.power(y_squared, x_squared + 1)

        return jnp.sum(term1 + term2)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = -1 for i odd -> 0-based: i even
        y = y.at[::2].set(-1.0)
        # x_i = 1 for i even -> 0-based: i odd
        y = y.at[1::2].set(1.0)
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
            return None, jnp.array([])

        # Vectorized constraint computation
        # c_k: (3 - 2x_{k+1})x_{k+1} + 1 - x_k - 2x_{k+2} ≤ 0
        # for k = 1, ..., n-2

        # For k=1 to n-2 (1-based), we need x_k, x_{k+1}, x_{k+2}
        # In 0-based indexing:
        # k=1: x_1=y[0], x_2=y[1], x_3=y[2]
        # k=2: x_2=y[1], x_3=y[2], x_4=y[3]
        # ...
        # k=n-2: x_{n-2}=y[n-3], x_{n-1}=y[n-2], x_n=y[n-1]

        x_k = y[:-2]  # x_k for k=1 to n-2
        x_k1 = y[1:-1]  # x_{k+1} for k=1 to n-2
        x_k2 = y[2:]  # x_{k+2} for k=1 to n-2

        # Compute all constraints at once
        inequality_constraints = (3 - 2 * x_k1) * x_k1 + 1 - x_k - 2 * x_k2

        return None, inequality_constraints
