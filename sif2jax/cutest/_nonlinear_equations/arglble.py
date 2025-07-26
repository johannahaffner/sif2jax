import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class ARGLBLE(AbstractNonlinearEquations):
    """ARGLBLE problem implementation.

    Variable dimension rank one linear problem

    Source: Problem 33 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#93 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    classification NLR2-AN-V-V

    This is a(n infeasible) linear feasibility problem
    N is the number of free variables
    M is the number of equations ( M .ge. N)
    """

    # Default parameters
    N: int = 200  # Number of variables
    M: int = 400  # Number of equations (M >= N)

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def constraint(self, y):
        """Return the equality constraints (residuals) for the nonlinear system."""
        residuals = self.residual(y)
        # Return (equality_constraints, inequality_constraints)
        # For nonlinear equations, we have only equality constraints
        return residuals, None

    def residual(self, y):
        """Compute the residuals for the system (vectorized)."""
        n = self.N
        m = self.M
        x = y  # Variables

        # Vectorized computation
        # Create coefficient matrix A where A[i,j] = (i+1)*(j+1)
        i_indices = jnp.arange(1, m + 1).reshape(-1, 1)
        j_indices = jnp.arange(1, n + 1).reshape(1, -1)
        A = (i_indices * j_indices).astype(x.dtype)

        # Compute residuals: A @ x - 1
        residuals = A @ x - 1.0

        return residuals

    @property
    def y0(self):
        """Initial guess for variables."""
        return jnp.ones(self.N)

    @property
    def bounds(self):
        """Returns None as this problem has no bounds."""
        return None

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For nonlinear equations, the objective is always zero
        # Note: The SIF file values (SOLTN) refer to least squares objective
        return jnp.array(0.0)
