import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class HAGER1(AbstractConstrainedMinimisation):
    """
    A nonlinear optimal control problem, by W. Hager.

    Source: problem P1 in
    W.W. Hager,
    "Multiplier Methods for Nonlinear Optimal Control",
    SIAM J. on Numerical Analysis 27(4): 1061-1080, 1990.

    SIF input: Ph. Toint, March 1991.

    classification SLR2-AN-V-V

    Default N = 5000 (original Hager value)
    """

    n_param: int = 10  # Number of discretized points in [0,1]
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n_param: int = 10):
        self.n_param = n_param

    @property
    def n(self) -> int:
        """Total number of variables: x(0) to x(N) plus u(1) to u(N)."""
        return 2 * self.n_param + 1

    @property
    def m(self) -> int:
        """Number of constraints: N constraints S(i) plus fixed x(0)."""
        return self.n_param + 1

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        y = jnp.zeros(self.n, dtype=jnp.float64)
        # x(0) starts at 1.0
        y = y.at[0].set(1.0)
        return y

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        n = self.n_param
        # h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Objective: 0.5 * x[N]^2 + sum(u[i]^2)/(2*N)
        obj = 0.5 * x[n] ** 2
        obj += jnp.sum(u**2) / (2 * n)

        return obj

    def constraint(self, y: Array):
        """Compute the equality and inequality constraints."""
        n = self.n_param
        h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Equality constraints
        # S(i): (N-0.5)*x(i) + (-N-0.5)*x(i-1) - u(i) = 0
        # Which is: (1/h - 0.5)*x(i) + (-1/h - 0.5)*x(i-1) - u(i) = 0
        eq_constraints = []

        # x(0) = 1.0 constraint
        eq_constraints.append(x[0] - 1.0)

        # S(i) constraints for i = 1 to N
        coeff1 = 1.0 / h - 0.5  # coefficient for x(i)
        coeff2 = -1.0 / h - 0.5  # coefficient for x(i-1)

        for i in range(1, n + 1):
            s_i = coeff1 * x[i] + coeff2 * x[i - 1] - u[i - 1]
            eq_constraints.append(s_i)

        eq_constraints = jnp.array(eq_constraints, dtype=jnp.float64)

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """All variables are free, except x(0) which is fixed by constraint."""
        return None

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the objective and constraint functions."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # Not explicitly given in the SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # Lower bound is 0.0 according to SIF file
        return jnp.array(0.0)
