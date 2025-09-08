import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LSQFIT(AbstractConstrainedMinimisation):
    """An elementary constrained linear least-squares fit.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "The LANCELOT User's Manual",
    Dept of Maths, FUNDP, 1991.

    SIF input: Ph. Toint, Jan 1991.

    Classification: SLR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function (sum of squares)."""
        del args

        a, b = y[0], y[1]

        # Data points
        x_data = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_data = jnp.array([0.25, 0.3, 0.625, 0.701, 1.0])

        # Linear model: a*x + b
        predictions = a * x_data + b

        # Sum of squared residuals (halved to match pycutest)
        residuals = predictions - y_data
        return 0.5 * jnp.sum(residuals**2)

    def constraint(self, y):
        """Compute the constraint a + b <= 0.85."""
        a, b = y[0], y[1]

        # Constraint: a + b <= 0.85
        # Return as inequality constraint: a + b - 0.85 <= 0
        return None, jnp.array([a + b - 0.85])

    @property
    def n_var(self):
        """Number of variables."""
        return 2

    @property
    def n_con(self):
        """Number of constraints."""
        return 1

    @property
    def y0s(self):
        """Starting point."""
        # Default starting point for unconstrained variables is 0
        return {0: jnp.zeros(2)}

    @property
    def y0(self):
        """Initial guess for variables."""
        return jnp.zeros(2)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        # From SIF: XR LSQFIT b means b is free
        # Variables not mentioned in BOUNDS have default bounds
        # Default lower bound is 0 for unlisted variables
        lower = jnp.array([0.0, -jnp.inf])  # a >= 0, b free
        upper = jnp.array([jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
