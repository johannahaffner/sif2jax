import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class HAGER4(AbstractConstrainedMinimisation):
    """
    A nonlinear optimal control problem, by W. Hager.

    NOTE: The solution for x given in the article below by Hager has
    a typo. On the interval [1/2, 1], x(t) = (exp(2t) + exp(t))/d. In
    other words, the minus sign in the article should be a plus sign.

    Source: problem P4 in
    W.W. Hager,
    "Multiplier Methods for Nonlinear Optimal Control",
    SIAM J. on Numerical Analysis 27(4): 1061-1080, 1990.

    SIF input: Ph. Toint, April 1991.

    classification OLR2-AN-V-V

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

        # Compute time-dependent quantities
        t = jnp.arange(n + 1, dtype=jnp.float64) * h
        z = jnp.exp(-2.0 * t)

        # Constants for element calculations
        a = -0.5 * z
        b = a * (t + 0.5)
        c = a * (t * t + t + 0.5)

        # Differences for scaling
        da = a[1] - a[0]
        db = b[1] - b[0]
        dc = c[1] - c[0]

        # From SIF: 1/H = N, so SCDB = DB * (1/H) = DB * N
        # and SCDC = DC * (1/2HSQ) = DC * 0.5 * N^2
        scda = 0.5 * da
        scdb = db * n  # DB * (1/H) where 1/H = N
        scdc = dc * (0.5 * n * n)  # DC * (1/2HSQ) where 1/2HSQ = 0.5*N^2

        # ELT elements (vectorized)
        # Element parameters - use z[i-1] for element i
        # For elements 1 to N, we need z[0] to z[n-1]
        d = scda * z[:-1]  # z[0] to z[n-1]
        e = scdb * z[:-1]
        f = scdc * z[:-1]

        # ELT: D*X*X + E*X*(Y-X) + F*(Y-X)**2
        # where X = x(i), Y = x(i-1)
        x_curr = x[1:]  # x[1] to x[n] - this is X in the element
        x_prev = x[:-1]  # x[0] to x[n-1] - this is Y in the element
        diff = x_prev - x_curr  # Y - X

        elt = d * x_curr * x_curr + e * x_curr * diff + f * diff * diff
        obj = jnp.sum(elt)

        # u[i]^2 terms scaled by h/2
        obj += jnp.sum(u**2) * (h / 2.0)

        return obj

    def constraint(self, y: Array):
        """Compute the equality and inequality constraints."""
        n = self.n_param
        h = 1.0 / n

        # Extract variables
        x = y[: n + 1]  # x(0) to x(N)
        u = y[n + 1 :]  # u(1) to u(N)

        # Compute time-dependent quantities
        t = jnp.arange(n + 1, dtype=jnp.float64) * h

        # Equality constraints
        # S(i) constraints for i = 1 to N (vectorized)
        # S(i): (1/h - 1)*x(i) + (-1/h)*x(i-1) - exp(t(i))*u(i) = 0
        # Note: x(0) is fixed to XX0, not handled as a constraint
        eti = jnp.exp(t[1:])  # exp(t(1)) to exp(t(N))
        eq_constraints = (1.0 / h - 1.0) * x[1:] + (-1.0 / h) * x[:-1] - eti * u

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """x(0) is fixed to XX0 via bounds, u(i) bounded above by 1.0."""
        n = self.n_param
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Fix x(0) to XX0 using bounds
        e = jnp.exp(1.0)
        xx0 = (1.0 + 3.0 * e) / (2.0 - 2.0 * e)
        lower = lower.at[0].set(xx0)
        upper = upper.at[0].set(xx0)

        # u(i) <= 1.0 for i = 1 to N
        upper = upper.at[n + 1 :].set(1.0)

        return lower, upper

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        y = jnp.zeros(self.n)
        # x(0) starts at specific value based on constants
        e = jnp.exp(1.0)
        xx0 = (1.0 + 3.0 * e) / (2.0 - 2.0 * e)
        y = y.at[0].set(jnp.array(xx0, dtype=y.dtype))
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
        # From SIF file comments: SOLTN(1000) = 2.794244187
        return jnp.array(2.794244187)
