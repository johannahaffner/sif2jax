import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class HAGER2(AbstractConstrainedMinimisation):
    """
    A nonlinear optimal control problem, by W. Hager.

    Source: problem P2 in
    W.W. Hager,
    "Multiplier Methods for Nonlinear Optimal Control",
    SIAM J. on Numerical Analysis 27(4): 1061-1080, 1990.

    SIF input: Ph. Toint, March 1991.

    classification OLR2-AN-V-V

    Default N = 5000 (original Hager value)
    """

    n_param: int = 5000  # Number of discretized points in [0,1]
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n_param: int = 5000):
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

        # Objective has two parts:
        # 1. Sum of LINSQ elements: (x[i-1]^2 + x[i-1]*x[i] + x[i]^2) scaled by 6/H
        # 2. Sum of u[i]^2 scaled by 4/H

        obj = 0.0

        # LINSQ elements
        for i in range(1, n + 1):
            xa = x[i - 1]
            xb = x[i]
            linsq = xa * xa + xa * xb + xb * xb
            obj += linsq * (6.0 / n)

        # u[i]^2 terms
        obj += jnp.sum(u**2) * (4.0 / n)

        return obj

    def constraint(self, y: Array):
        """Compute the equality and inequality constraints."""
        n = self.n_param
        h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Equality constraints
        eq_constraints = []

        # x(0) = 1.0 constraint
        eq_constraints.append(x[0] - 1.0)

        # S(i) constraints for i = 1 to N
        # S(i): (1/h - 1/4)*x(i) + (-1/h - 1/4)*x(i-1) - u(i) = 0
        coeff1 = 1.0 / h - 0.25  # coefficient for x(i)
        coeff2 = -1.0 / h - 0.25  # coefficient for x(i-1)

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
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given in the SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # From SIF file comments:
        # SOLTN(10) = 0.4325699689
        return jnp.array(0.4325699689)
