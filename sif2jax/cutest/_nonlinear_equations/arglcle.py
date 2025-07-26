import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class ARGLCLE(AbstractNonlinearEquations):
    """ARGLCLE problem implementation.

    Variable dimension rank one linear problem, with zero rows and columns

    Source: Problem 34 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#101 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    classification NLR2-AN-V-V

    This is a(n infeasible) linear feasibility problem
    N is the number of free variables
    M is the number of equations ( M.ge.N)

    TODO: Human review needed
    Attempts made: Vectorized implementation, converted to nonlinear equations
    Suspected issues: Empty rows (first/last equations have no variables) may be
                      handled differently by pycutest, causing mismatches
    Additional resources needed: Clarification on how pycutest handles empty constraints
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

        # Create array for all residuals
        residuals = jnp.zeros(m)

        # G(1) - first equation has no variables (empty row)
        residuals = residuals.at[0].set(-1.0)

        # G(i) for i = 2 to M-1 (vectorized)
        # Create coefficient matrix for the middle equations
        # G(i) = sum_{j=2}^{N-1} (i-1) * j * x_j - 1
        i_indices = jnp.arange(2, m)  # i from 2 to M-1
        j_indices = jnp.arange(2, n)  # j from 2 to N-1

        # Create coefficient matrix where A[i-2, j-2] = (i-1) * j
        i_factors = (i_indices - 1).reshape(-1, 1)
        j_factors = j_indices.reshape(1, -1)
        A = (i_factors * j_factors).astype(x.dtype)

        # Extract relevant variables x[1:n-1] (j from 2 to N-1 in 1-based indexing)
        x_subset = x[1 : n - 1]

        # Compute middle residuals: A @ x_subset - 1
        middle_residuals = A @ x_subset - 1.0

        # Set middle residuals
        residuals = residuals.at[1 : m - 1].set(middle_residuals)

        # G(M) - last equation has no variables (empty row)
        residuals = residuals.at[m - 1].set(-1.0)

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
