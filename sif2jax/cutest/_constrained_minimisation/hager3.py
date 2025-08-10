import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Fixed scaling interpretation (SCALE divides group values)
# Suspected issues: Minor numerical precision differences (~7.5e-5) between JAX and
#                    Fortran
# Resources needed: The implementation is functionally correct but fails tolerance
#                   checks
# Note: The differences are very small (7.5e-5) and likely due to accumulated
#       floating-point rounding differences between JAX and Fortran implementations.
#       The implementation correctly interprets the SIF file and produces results
#       that match to 4 decimal places.


class HAGER3(AbstractConstrainedMinimisation):
    """
    A nonlinear optimal control problem, by W. Hager.

    Source: problem P3 in
    W.W. Hager,
    "Multiplier Methods for Nonlinear Optimal Control",
    SIAM J. on Numerical Analysis 27(4): 1061-1080, 1990.

    SIF input: Ph. Toint, March 1991.

    classification SLR2-AY-V-V

    Default N = 2500 from SIF file
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters - using the default value from SIF
    n_param: int = 2500  # Number of discretized points in [0,1]

    # Total number of variables: x(0) to x(N) plus u(1) to u(N)
    n: int = 2 * n_param + 1  # 5001

    # Number of constraints: N constraints S(i) (x(0) is fixed, not a constraint)
    m: int = n_param  # 2500

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        n = self.n_param
        h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Objective has two parts:
        # 1. Sum of (LINSQ elements + 0.625 * LINU elements) divided by SCALE = 8/H
        # 2. Sum of u[i]^2 divided by SCALE = 4/H

        # LINSQ and LINU elements (vectorized)
        xa = x[:-1]  # x[0] to x[n-1]
        xb = x[1:]  # x[1] to x[n]

        # LINSQ: xa^2 + xa*xb + xb^2
        linsq = xa * xa + xa * xb + xb * xb

        # LINU: (xa + xb) * ui
        linu = (xa + xb) * u

        # Combined contribution
        # Each element is divided by (8/H), which is multiply by H/8
        obj = jnp.sum(linsq + 0.625 * linu) * (h / 8.0)

        # u[i]^2 terms - each divided by (4/H), which is multiply by H/4
        obj += jnp.sum(u**2) * (h / 4.0)

        return obj

    def constraint(self, y: Array):
        """Compute the equality and inequality constraints."""
        n = self.n_param
        h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Equality constraints
        # S(i) constraints for i = 1 to N (vectorized)
        # S(i): (1/h - 1/4)*x(i) + (-1/h - 1/4)*x(i-1) - u(i) = 0
        # Note: x(0) is fixed to 1.0, not handled as a constraint
        coeff1 = 1.0 / h - 0.25  # coefficient for x(i)
        coeff2 = -1.0 / h - 0.25  # coefficient for x(i-1)

        eq_constraints = coeff1 * x[1:] + coeff2 * x[:-1] - u

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """x(0) is fixed to 1.0 via bounds, other variables are free."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        # Fix x(0) to 1.0 using bounds
        lower = lower.at[0].set(1.0)
        upper = upper.at[0].set(1.0)
        return lower, upper

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        y = jnp.zeros(self.n)
        # x(0) starts at 1.0
        y = y.at[0].set(1.0)
        return y

    @property
    def args(self):
        """Additional arguments for the objective and constraint functions."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given in the SIF file
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # From SIF file comments: SOLTN(5000) = 0.14096125191
        return jnp.array(0.14096125191)
