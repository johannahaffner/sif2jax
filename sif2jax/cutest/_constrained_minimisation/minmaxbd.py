import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


# Precompute expensive transcendental functions for i=1..20
I_OVER_5 = jnp.arange(1, 21, dtype=jnp.float64) / 5.0  # i/5 for i = 1, 2, ..., 20
EXP_I_OVER_5 = jnp.exp(I_OVER_5)  # exp(i/5)
SIN_I_OVER_5 = jnp.sin(I_OVER_5)  # sin(i/5)
COS_I_OVER_5 = jnp.cos(I_OVER_5)  # cos(i/5)


class MINMAXBD(AbstractConstrainedMinimisation):
    """MINMAXBD problem - The minmax version of the Brown and Dennis problem.

    The minmax version of the Brown and Dennis problem in 4 variables.
    This function involves 20 groups. Each group has 2 nonlinear elements.

    Source: Problem 16 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#30
    SDIF input: Ph. Toint, Dec 1989, modified by Nick Gould, Oct 1992.

    Classification: LOR2-AN-5-20
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Minimize F (the minimax variable)."""
        del args
        x1, x2, x3, x4, f = y
        return f

    def constraint(self, y):
        """Minmax constraints: F >= g_i for i=1..20."""
        x1, x2, x3, x4, f = y

        # Vectorized computation using precomputed values
        # Ensure consistent dtype for the precomputed arrays
        i_over_5 = I_OVER_5.astype(y.dtype)
        exp_i_over_5 = EXP_I_OVER_5.astype(y.dtype)
        sin_i_over_5 = SIN_I_OVER_5.astype(y.dtype)
        cos_i_over_5 = COS_I_OVER_5.astype(y.dtype)

        # A(i) = (X1 + i/5*X2 - exp(i/5))^2
        a_terms = x1 + i_over_5 * x2 - exp_i_over_5
        a_i = a_terms * a_terms

        # B(i) = (X3 + sin(i/5)*X4 - cos(i/5))^2
        b_terms = x3 + sin_i_over_5 * x4 - cos_i_over_5
        b_i = b_terms * b_terms

        # g_i = A(i) + B(i)
        g_i = a_i + b_i

        # From SIF: G(I) = -F + A(I) + B(I) >= 0  =>  -F + g_i >= 0  =>  g_i - F >= 0
        # This means constraints are: g_i - F >= 0 (opposite of what I had)
        constraints = g_i - f

        return None, constraints

    def equality_constraints(self):
        """All constraints are inequalities."""
        return jnp.zeros(20, dtype=bool)

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.array([25.0, 5.0, -5.0, -1.0, 825.559]))

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
        """Expected optimal objective value (not provided in SIF)."""
        return None

    def num_constraints(self):
        """Returns the number of constraints in the problem."""
        return (0, 20, 0)  # 0 equality, 20 inequality, 0 bounds
