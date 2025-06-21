import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLI1(AbstractConstrainedMinimisation):
    """LUKVLI1 - Chained Rosenbrock function with trigonometric-exponential constraints.

    Problem 5.1 from Luksan and Vlcek test problems with inequality constraints.

    The objective is a chained Rosenbrock function:
    f(x) = Σ[i=1 to n-1] [100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2]

    Subject to inequality constraints:
    c_k(x) = 3x_{k+1}^3 + 2x_{k+2} - 5 + sin(x_{k+1} - x_{k+2})sin(x_{k+1} + x_{k+2})
             + 4x_{k+1} - x_k exp(x_k - x_{k+1}) - 3 ≤ 0, for k = 1, ..., n-2

    Starting point: x_i = -1.2 for i odd, x_i = 1.0 for i even

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
        # Chained Rosenbrock function - vectorized
        y_current = y[:-1]  # y[0] to y[n-2]
        y_next = y[1:]  # y[1] to y[n-1]

        # Compute all terms at once
        terms = 100 * (y_current**2 - y_next) ** 2 + (y_current - 1) ** 2
        return jnp.sum(terms)

    def y0(self):
        # Starting point: x_i = -1.2 for i odd, x_i = 1.0 for i even
        y = jnp.zeros(self.n)
        # JAX uses 0-based indexing, so odd indices in the problem are even in JAX
        y = y.at[::2].set(-1.2)  # i = 1, 3, 5, ... (1-based) -> 0, 2, 4, ... (0-based)
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
        if n < 3:
            return None, jnp.array([])

        # Vectorized constraint computation
        # For k = 0 to n-3, we need y[k], y[k+1], y[k+2]
        y_k = y[:-2]  # y[0] to y[n-3]
        y_k1 = y[1:-1]  # y[1] to y[n-2]
        y_k2 = y[2:]  # y[2] to y[n-1]

        # Compute all constraints at once
        inequality_constraints = (
            3 * y_k1**3
            + 2 * y_k2
            - 5
            + jnp.sin(y_k1 - y_k2) * jnp.sin(y_k1 + y_k2)
            + 4 * y_k1
            - y_k * jnp.exp(y_k - y_k1)
            - 3
        )

        return None, inequality_constraints
