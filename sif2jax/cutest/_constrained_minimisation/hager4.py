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

        # x(0) starts at specific value based on constants
        e = jnp.exp(1.0)
        xx0 = (1.0 + 3.0 * e) / (2.0 - 2.0 * e)
        y = y.at[0].set(xx0)

        return y

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

        scda = 0.5 * da
        scdb = db / h
        scdc = dc / (0.5 * h * h)

        obj = 0.0

        # ELT elements
        for i in range(1, n + 1):
            # Element parameters
            d = scda * z[i - 1]
            e = scdb * z[i - 1]
            f = scdc * z[i - 1]

            # ELT: D*X*X + E*X*(Y-X) + F*(Y-X)**2
            x_i = x[i]
            y_i = x[i - 1]
            diff = y_i - x_i

            elt = d * x_i * x_i + e * x_i * diff + f * diff * diff
            obj += elt

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
        eq_constraints = []

        # x(0) = XX0 constraint
        e = jnp.exp(1.0)
        xx0 = (1.0 + 3.0 * e) / (2.0 - 2.0 * e)
        eq_constraints.append(x[0] - xx0)

        # S(i) constraints for i = 1 to N
        # S(i): (1/h - 1)*x(i) + (-1/h)*x(i-1) - exp(t(i))*u(i) = 0
        for i in range(1, n + 1):
            eti = jnp.exp(t[i])
            s_i = (1.0 / h - 1.0) * x[i] + (-1.0 / h) * x[i - 1] - eti * u[i - 1]
            eq_constraints.append(s_i)

        eq_constraints = jnp.array(eq_constraints, dtype=jnp.float64)

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """u(i) bounded above by 1.0, x variables are free."""
        n = self.n_param
        lower = jnp.full(self.n, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(self.n, jnp.inf, dtype=jnp.float64)

        # u(i) <= 1.0 for i = 1 to N
        upper = upper.at[n + 1 :].set(1.0)

        return lower, upper

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
        # SOLTN(10) = 2.833914199
        return jnp.array(2.833914199)
