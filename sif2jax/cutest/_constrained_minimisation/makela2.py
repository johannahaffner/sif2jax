import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class MAKELA2(AbstractConstrainedMinimisation):
    """MAKELA2 problem - A nonlinear minmax problem.

    A nonlinear minmax problem in two variables.

    Source:
    M.M. Makela,
    "Nonsmooth optimization",
    Ph.D. thesis, Jyvaskyla University, 1990

    SIF input: Ph. Toint, Nov 1993.

    Classification: LQR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 3  # x1, x2, u

    @property
    def m(self):
        """Number of constraints."""
        return 3  # Three inequality constraints

    def objective(self, y, args):
        """Compute the objective (minimize u)."""
        del args
        u = y[2]
        return u

    def constraint(self, y):
        """Compute the constraints."""
        x1 = y[0]
        x2 = y[1]
        u = y[2]

        # Constraints:
        # F1: -u + x1^2 + x2^2 <= 0
        # F2: -u - 40*x1 - 10*x2 + x1^2 + x2^2 <= -40
        # F3: -u - 10*x1 - 20*x2 + x1^2 + x2^2 <= -60
        c1 = -u + x1**2 + x2**2
        c2 = -u - 40.0 * x1 - 10.0 * x2 + x1**2 + x2**2 + 40.0
        c3 = -u - 10.0 * x1 - 20.0 * x2 + x1**2 + x2**2 + 60.0

        # All are inequality constraints (<=)
        constraints = jnp.array([c1, c2, c3])

        # Return as inequality constraints (pycutest convention: <= 0)
        return None, constraints

    def y0(self):
        """Initial guess."""
        return jnp.array([-1.0, 5.0, 0.0])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(3, -jnp.inf)
        upper = jnp.full(3, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(7.2)
