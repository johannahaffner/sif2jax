import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class S368(AbstractBoundedMinimisation):
    """S368 problem.

    Wolfe's problem.

    Source:
    P. Wolfe,
    "Explicit solution of an optimization problem",
    Mathematical Programming 2, 258-260, 1972.

    SIF input: Nick Gould, Oct 1992.

    See also Schittkowski #368 (for N = 8)

    Classification: OBR2-MN-V-0

    This is a bounded optimization problem with N variables.
    The default parameter is N = 8, variables bounded between 0 and 1.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(8)

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum over all i,j of:
        M(i,j): -X(i)^2 * X(j)^4
        P(i,j): X(i)^3 * X(j)^3
        """
        del args

        # Vectorized computation using outer products
        # y has shape (n,), we need all pairwise combinations
        x = y  # For clarity

        # Compute powers efficiently
        x_squared = x * x
        x_cubed = x_squared * x
        x_fourth = x_squared * x_squared

        # Create matrices for vectorized computation
        # M(i,j) = -X(i)^2 * X(j)^4 for all i,j pairs
        x2_matrix = x_squared[:, None]  # Shape (n, 1)
        x4_matrix = x_fourth[None, :]  # Shape (1, n)
        m_terms = -x2_matrix * x4_matrix  # Shape (n, n)

        # P(i,j) = X(i)^3 * X(j)^3 for all i,j pairs
        x3_matrix = x_cubed[:, None]  # Shape (n, 1)
        x3_matrix_t = x_cubed[None, :]  # Shape (1, n)
        p_terms = x3_matrix * x3_matrix_t  # Shape (n, n)

        # Sum all terms
        total = jnp.sum(m_terms + p_terms)
        return total

    @property
    def bounds(self):
        """Variable bounds."""
        # From SIF: UP S368 'DEFAULT' 1.0 means upper bound of 1.0
        # Lower bound defaults to 0.0 for bounded variables
        n = self.n
        lower = inexact_asarray(jnp.zeros(n))
        upper = inexact_asarray(jnp.ones(n))
        return lower, upper

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # From SIF: start with X(I) = I/(N+1)
        n = self.n
        indices = jnp.arange(1, n + 1, dtype=float)  # [1, 2, ..., n]
        initial = indices / (n + 1)  # [1/9, 2/9, ..., 8/9] for n=8
        return inexact_asarray(initial)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
