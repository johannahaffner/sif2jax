import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class MAKELA3(AbstractConstrainedMinimisation):
    """MAKELA3 problem - A nonlinear minmax problem.

    A nonlinear minmax problem in twenty variables.

    Source:
    M.M. Makela,
    "Nonsmooth optimization",
    Ph.D. thesis, Jyvaskyla University, 1990

    SIF input: Ph. Toint, Nov 1993.

    Classification: LQR2-AN-21-20
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 21  # x1, ..., x20, u

    @property
    def m(self):
        """Number of constraints."""
        return 20  # Twenty inequality constraints

    def objective(self, y, args):
        """Compute the objective (minimize u)."""
        del args
        u = y[20]
        return u

    def constraint(self, y):
        """Compute the constraints."""
        x = y[:20]
        u = y[20]

        # Constraints:
        # F(i): -u + x(i)^2 <= 0 for i = 1, ..., 20
        constraints = -u + x**2

        # All are inequality constraints (<=)
        # Return as inequality constraints (pycutest convention: <= 0)
        return None, constraints

    def y0(self):
        """Initial guess."""
        x0 = jnp.zeros(21)
        # Initial values for x1-x10: 1, 2, ..., 10
        x0 = x0.at[:10].set(inexact_asarray(jnp.arange(1, 11)))
        # Initial values for x11-x20: -11, -12, ..., -20
        x0 = x0.at[10:20].set(inexact_asarray(jnp.arange(-11, -21, -1)))
        # u starts at 0
        return inexact_asarray(x0)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(21, -jnp.inf)
        upper = jnp.full(21, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
