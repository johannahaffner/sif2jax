import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class LUKVLE7(AbstractConstrainedMinimisation):
    """LUKVLE7 - Trigonometric tridiagonal function with simplified five-diagonal
    constraints.

    Problem 5.7 from Luksan and Vlcek test problems.

    The objective is a trigonometric tridiagonal function:
    f(x) = Σ[i=1 to n] i[(1 - cos x_i) + sin x_{i-1} - sin x_{i+1}]
    where sin x_0 = sin x_{n+1} = 0

    Subject to equality constraints:
    c_1(x) = 4(x_1 - x_2^2) + x_2 - x_3^2 = 0
    c_2(x) = 8x_2(x_2^2 - x_1) - 2(1 - x_2) + 4(x_2 - x_3^2) + x_3 - x_4^2 = 0
    c_3(x) = 8x_{n-1}(x_{n-1}^2 - x_{n-2}) - 2(1 - x_{n-1}) + 4(x_{n-1} - x_n^2)
                + x_{n-2} - x_{n-3} = 0
    c_4(x) = 8x_n(x_n^2 - x_{n-1}) + 2x_n + x_{n-1}^2 - x_{n-2} = 0

    Starting point: x_i = 1 for i = 1, ..., n

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
        n = len(y)
        # Trigonometric tridiagonal function - vectorized

        # Main terms: (1 - cos x_i)
        cos_terms = 1 - jnp.cos(y)

        # sin x_{i-1} with boundary condition sin x_0 = 0
        sin_x_i_minus_1 = jnp.pad(
            jnp.sin(y[:-1]), (1, 0), mode="constant", constant_values=0
        )

        # sin x_{i+1} with boundary condition sin x_{n+1} = 0
        sin_x_i_plus_1 = jnp.pad(
            jnp.sin(y[1:]), (0, 1), mode="constant", constant_values=0
        )

        # Compute all terms
        terms = cos_terms + sin_x_i_minus_1 - sin_x_i_plus_1

        # Multiply by i (1-based indexing: i = 1, 2, ..., n)
        coefficients = inexact_asarray(jnp.arange(1, n + 1))

        return jnp.sum(coefficients * terms)

    def y0(self):
        # Starting point: x_i = 1 for all i
        return inexact_asarray(jnp.ones(self.n))

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
        # Four equality constraints
        constraints = []

        # c_1: 4(x_1 - x_2^2) + x_2 - x_3^2 = 0
        if n >= 3:
            c1 = 4 * (y[0] - y[1] ** 2) + y[1] - y[2] ** 2
            constraints.append(c1)

        # c_2: 8x_2(x_2^2 - x_1) - 2(1 - x_2) + 4(x_2 - x_3^2) + x_3 - x_4^2 = 0
        if n >= 4:
            c2 = (
                8 * y[1] * (y[1] ** 2 - y[0])
                - 2 * (1 - y[1])
                + 4 * (y[1] - y[2] ** 2)
                + y[2]
                - y[3] ** 2
            )
            constraints.append(c2)

        # c_3: 8x_{n-1}(x_{n-1}^2 - x_{n-2}) - 2(1 - x_{n-1}) + 4(x_{n-1} - x_n^2)
        #       + x_{n-2} - x_{n-3} = 0
        if n >= 4:
            c3 = (
                8 * y[n - 2] * (y[n - 2] ** 2 - y[n - 3])
                - 2 * (1 - y[n - 2])
                + 4 * (y[n - 2] - y[n - 1] ** 2)
                + y[n - 3]
                - (y[n - 4] if n > 4 else y[n - 3])
            )
            constraints.append(c3)

        # c_4: 8x_n(x_n^2 - x_{n-1}) + 2x_n + x_{n-1}^2 - x_{n-2} = 0
        if n >= 3:
            c4 = (
                8 * y[n - 1] * (y[n - 1] ** 2 - y[n - 2])
                + 2 * y[n - 1]
                + y[n - 2] ** 2
                - y[n - 2]
            )
            constraints.append(c4)

        equality_constraints = jnp.array(constraints) if constraints else jnp.array([])
        return equality_constraints, None
