import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class KSS(AbstractNonlinearEquations):
    """The KSS system whose root at zero has exponential multiplicity
    as a function of dimension.

    Source: problem 8.1 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.

    classification NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N = 1000

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        For each equation i (1 ≤ i ≤ N):
        ∑_{j=1}^{i-1} X_j + X_i² - 3*X_i + ∑_{j=i+1}^N X_j = N-1

        This simplifies to:
        ∑_{j=1}^N X_j + X_i² - 4*X_i = N-1

        So residual_i = ∑_{j=1}^N X_j + X_i² - 4*X_i - (N-1)
        """
        # Total sum of all variables
        total_sum = jnp.sum(y)

        # For each equation i, compute: total_sum + X_i² - 4*X_i - (N-1)
        residuals = total_sum + y**2 - 4.0 * y - (self.N - 1)

        return residuals

    @property
    def y0(self):
        """Initial guess for the optimization problem."""
        # From SIF: all variables start at 1000
        return jnp.full(self.N, 1000.0)

    @property
    def bounds(self):
        """Variable bounds. All variables are free."""
        # From SIF: FR KSS 'DEFAULT' means all variables are free
        return None

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # The root is at zero (from the problem description)
        return jnp.zeros(self.N)

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # For nonlinear equations, the objective is ||residual||²
        # At the solution, all residuals should be zero
        return jnp.array(0.0)
