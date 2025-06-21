import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLE5(AbstractConstrainedMinimisation):
    """LUKVLE5 - Generalized Broyden tridiagonal function with five diagonal
    constraints.

    Problem 5.5 from Luksan and Vlcek test problems.

    The objective is a generalized Broyden tridiagonal function:
    f(x) = Σ[i=1 to n] |(3 - 2x_i)x_i - x_{i-1} - x_{i+1} + 1|^p
    where p = 7/3, x_0 = x_{n+1} = 0

    Subject to equality constraints:
    c_k(x) = 8x_{k+2}(x_{k+2}^2 - x_{k+1}) - 2(1 - x_{k+2}) + 4(x_{k+2} - x_{k+3}^2) +
             x_{k+1}^2 - x_k + x_{k+3} - x_{k+4}^2 = 0,
    for k = 1, ..., n-4

    Starting point: x_i = -1 for i = 1, ..., n

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
        p = 7.0 / 3.0
        # Generalized Broyden tridiagonal function - vectorized

        # Main diagonal terms: (3 - 2*x_i)*x_i
        main_terms = (3 - 2 * y) * y

        # Handle x_{i-1} terms with boundary x_0 = 0
        x_i_minus_1 = jnp.pad(y[:-1], (1, 0), mode="constant", constant_values=0)

        # Handle x_{i+1} terms with boundary x_{n+1} = 0
        x_i_plus_1 = jnp.pad(y[1:], (0, 1), mode="constant", constant_values=0)

        # Compute all terms at once
        terms = main_terms - x_i_minus_1 - x_i_plus_1 + 1

        # Apply the power function
        return jnp.sum(jnp.abs(terms) ** p)

    def y0(self):
        # Starting point: x_i = -1 for all i
        return inexact_asarray(jnp.full(self.n, -1.0))

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
        if n < 5:
            return jnp.array([]), None

        # Vectorized constraint computation
        # For k = 1 to n-4 (1-based), which is k = 0 to n-5 (0-based)
        # We need x_k, x_{k+1}, x_{k+2}, x_{k+3}, x_{k+4} in 1-based
        # Which is y[k-1], y[k], y[k+1], y[k+2], y[k+3] in 0-based

        num_constraints = n - 4
        k_indices = jnp.arange(num_constraints)  # 0 to n-5

        # For k=0 (which is k=1 in 1-based), x_k = x_1 = y[0]
        # For k=1 (which is k=2 in 1-based), x_k = x_2 = y[1]
        # So x_k in 1-based = y[k] in 0-based when iterating k from 0
        x_k = y[k_indices]  # x_k
        x_k1 = y[k_indices + 1]  # x_{k+1}
        x_k2 = y[k_indices + 2]  # x_{k+2}
        x_k3 = y[k_indices + 3]  # x_{k+3}
        x_k4 = y[k_indices + 4]  # x_{k+4}

        # Compute all constraints at once
        # c_k = 8x_{k+2}(x_{k+2}^2 - x_{k+1}) - 2(1 - x_{k+2}) + 4(x_{k+2} - x_{k+3}^2)
        #       + x_{k+1}^2 - x_k + x_{k+3} - x_{k+4}^2
        constraints = (
            8 * x_k2 * (x_k2**2 - x_k1)
            - 2 * (1 - x_k2)
            + 4 * (x_k2 - x_k3**2)
            + x_k1**2
            - x_k
            + x_k3
            - x_k4**2
        )

        return constraints, None
