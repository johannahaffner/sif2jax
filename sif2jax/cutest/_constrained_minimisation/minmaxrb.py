import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class MINMAXRB(AbstractConstrainedMinimisation):
    """MINMAXRB problem - A Rosenbrock-like nonlinear minmax problem.

    A Rosenbrock-like nonlinear minmax problem.

    Source:
    J. Hald and K. Madsen
    "Combined LP and quasi-Newton methods for minmax optimization"
    Mathematical Programming, vol.20, p. 49-62, 1981.

    SIF input: Ph. Toint, April 1992.

    Classification: LQR2-AN-3-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Minimize U (the minimax variable)."""
        del args
        x1, x2, u = y
        return u

    def constraint(self, y):
        """Minmax constraints: U >= each of the 4 constraint functions."""
        x1, x2, u = y

        # From SIF file - corrected interpretation:
        # GROUPS: C1: U*1.0 + X2*(-10.0), C2: U*1.0 + X2*(10.0),
        #         C3: U*1.0 + X1*(1.0), C4: U*1.0 + X1*(-1.0)
        # CONSTANTS: C3 gets +1.0, C4 gets -1.0 (subtracted in constraint form)
        # GROUP USES: C1 gets +10*X1^2, C2 gets -10*X1^2
        # Final form: group_terms - constants >= 0

        c1 = (
            u + (-10.0) * x2 + 10.0 * x1 * x1
        )  # U - 10*X2 + 10*X1^2 >= 0 (no constants)
        c2 = (
            u + 10.0 * x2 + (-10.0) * x1 * x1
        )  # U + 10*X2 - 10*X1^2 >= 0 (no constants)
        c3 = u + 1.0 * x1 - 1.0  # U + X1 - 1 >= 0 (subtract constant)
        c4 = u + (-1.0) * x1 - (-1.0)  # U - X1 + 1 >= 0 (subtract constant)

        # All constraints are inequalities: c_i >= 0
        constraints = jnp.array([c1, c2, c3, c4])

        return None, constraints

    def equality_constraints(self):
        """All constraints are inequalities."""
        return jnp.zeros(4, dtype=bool)

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.array([-1.2, 1.0, 1.0]))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """No variable bounds."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file: solution is 0.0
        return inexact_asarray(0.0)

    def num_constraints(self):
        """Returns the number of constraints in the problem."""
        return (0, 4, 0)  # 0 equality, 4 inequality, 0 bounds
