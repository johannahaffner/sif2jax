import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class KSSLS(AbstractUnconstrainedMinimisation):
    """KSS system with a zero root having exponential multiplicity by dimension.

    This is a least-squares version of KSS.

    Source: problem 7.1 (note: originally referenced as 8.1, but no 8.1 exists in the
    paper) in Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics
    University of Notre Dame, Indiana, USA (2012)

    The KSS systems of n × n at zero (1, 1, ..., 1) are defined as:
    x_i² + ∑_{j=1}^n x_j - 2x_i - (n - 1) = 0, for i = 1, 2, ..., n

    SIF input: Nick Gould, Jan 2012.
    Least-squares version of KSS.SIF, Nick Gould, Jan 2020.
    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    n: int = 1000  # Other suggested values: 4, 10, 100

    def objective(self, y, args):
        """Compute the objective function value.

        The system is composed of n equations, where each equation i is:
        x_i² + ∑_{j=1}^n x_j - 2x_i - (n - 1) = 0

        For each equation, we compute the squared residual and then sum them.
        """
        n = self.n

        # Sum of all variables (∑_{j=1}^n x_j)
        sum_all = jnp.sum(y)

        # Define the residual function for a single equation i
        def residual_fn(i):
            # Calculate residual: x_i² + ∑_{j=1}^n x_j - 2x_i - (n - 1)
            return y[i] ** 2 + sum_all - 2.0 * y[i] - (n - 1)

        # Compute residuals for all equations using vmap
        indices = jnp.arange(n)
        residuals = jax.vmap(residual_fn)(indices)

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point with all variables set to 1000 (from SIF file)."""
        return inexact_asarray(jnp.full(self.n, 1000.0))

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is (1, 1, ..., 1) as mentioned in the reference."""
        return jnp.ones(self.n)

    def expected_objective_value(self):
        """The solution value mentioned in the SIF file comment."""
        return jnp.array(0.0)
