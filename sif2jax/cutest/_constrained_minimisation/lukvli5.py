import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLI5(AbstractConstrainedMinimisation):
    """LUKVLI5 - Generalized Broyden tridiagonal function with five diagonal
    inequality constraints.

    Problem 5.5 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a generalized Broyden tridiagonal function:
    f(x) = Σ[i=1 to n] |(3 - 2x_i)x_i - x_{i-1} - x_{i+1} + 1|^p
    where p = 7/3, x_0 = x_{n+1} = 0

    Subject to inequality constraints:
    c_k(x) = 8x_{k+2}(x_{k+2}^2 - x_{k+1}) - 2(1 - x_{k+2}) + 4(x_{k+2} - x_{k+3}^2) +
             x_{k+1}^2 - x_k + x_{k+3} - x_{k+4}^2 ≤ 0,
    for k = 1, ..., n-4

    Starting point: x_i = -1 for i = 1, ..., n

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

    n: int = 10002  # Total dimension including fixed boundary variables

    def objective(self, y, args):
        del args
        p = 7.0 / 3.0
        # Generalized Broyden tridiagonal function - vectorized
        # y includes X(0) to X(N+1), where X(0) and X(N+1) are fixed at 0

        # Sum over i=1 to N (indices 1 to N in y)
        # The function is |(3 - 2x_i)x_i - x_{i-1} - x_{i+1} + 1|^p

        # Get the inner variables (X(1) to X(N))
        x = y[1:-1]  # This is X(1) to X(N)

        # Compute main terms: (3 - 2x_i)x_i + 1
        main_terms = (3 - 2 * x) * x + 1

        # x_{i-1} terms: X(0) to X(N-1), which is y[0:-2]
        x_i_minus_1 = y[:-2]

        # x_{i+1} terms: X(2) to X(N+1), which is y[2:]
        x_i_plus_1 = y[2:]

        # Combine all terms
        terms = main_terms - x_i_minus_1 - x_i_plus_1

        # Apply power function
        return jnp.sum(jnp.abs(terms) ** p)

    @property
    def y0(self):
        # Starting point: X(0) = 0, X(i) = -1 for i=1 to N, X(N+1) = 0
        y = jnp.full(self.n, -1.0)
        y = y.at[0].set(0.0)  # X(0) = 0
        y = y.at[-1].set(0.0)  # X(N+1) = 0
        return inexact_asarray(y)

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
        # X(0) and X(N+1) are fixed at 0
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Fix boundary values
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[-1].set(0.0)
        upper = upper.at[-1].set(0.0)

        return lower, upper

    def constraint(self, y):
        # y contains X(0) to X(N+1)
        # N = len(y) - 2 (since we have N+2 total variables)
        n_inner = len(y) - 2  # This is N

        if n_inner < 4:
            return None, jnp.array([])

        # Vectorized constraint computation
        # For k = 1 to N-4 (1-based)
        # We need X(k), X(k+1), X(k+2), X(k+3), X(k+4)

        k_indices = jnp.arange(1, n_inner - 3)  # k from 1 to N-4 (1-based)

        # In y array: X(k) is at index k
        x_k = y[k_indices]  # X(k)
        x_k1 = y[k_indices + 1]  # X(k+1)
        x_k2 = y[k_indices + 2]  # X(k+2)
        x_k3 = y[k_indices + 3]  # X(k+3)
        x_k4 = y[k_indices + 4]  # X(k+4)

        # Compute all constraints at once
        # c_k = 8x_{k+2}(x_{k+2}^2 - x_{k+1}) - 2(1 - x_{k+2}) + 4(x_{k+2} - x_{k+3}^2)
        #       + x_{k+1}^2 - x_k + x_{k+3} - x_{k+4}^2
        inequality_constraints = (
            8 * x_k2 * (x_k2**2 - x_k1)
            - 2 * (1 - x_k2)
            + 4 * (x_k2 - x_k3**2)
            + x_k1**2
            - x_k
            + x_k3
            - x_k4**2
        )

        return None, inequality_constraints
