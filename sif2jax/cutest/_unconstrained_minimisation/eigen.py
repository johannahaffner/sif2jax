import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class EIGEN(AbstractUnconstrainedMinimisation):
    """Base class for EIGEN problems.

    These problems compute eigenvalues and eigenvectors of symmetric matrices
    by solving a nonlinear least squares problem. They are formulated to find
    an orthogonal matrix Q and diagonal matrix D such that A = QᵀDQ where A
    is a specific input matrix.

    Different problems use different matrices A.

    Source: Originating from T.F. Coleman and P.A. Liao
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 50  # Default dimension - individual problems may override

    def _matrix(self):
        """Return the specific matrix A for this eigenvalue problem.

        This method is implemented by each subclass for its specific matrix.
        """
        raise NotImplementedError("Subclasses must implement _matrix")

    def objective(self, y, args):
        del args

        # y contains the variables as ordered in SIF file:
        # For each J from 1 to N:
        #   - D(J) (eigenvalue)
        #   - Q(1,J), Q(2,J), ..., Q(N,J) (J-th column of Q)
        # So the ordering is interleaved: D(1), Q(:,1), D(2), Q(:,2), etc.

        # Efficient extraction using reshape and slicing
        # Create indices for D values and Q columns
        y_reshaped = y.reshape(self.n, self.n + 1)
        d_diag = y_reshaped[:, 0]  # First element of each group is D(j)
        q = y_reshaped[:, 1:].T  # Rest are Q(:,j), need to transpose

        # Get the target matrix A
        a = self._matrix()

        # Eigenvalue equations: QᵀDQ - A
        # More efficient: Q.T @ (d_diag[:, None] * Q)
        qtdq = q.T @ (d_diag[:, None] * q)
        e_residual = qtdq - a

        # Orthogonality equations: QᵀQ - I
        qtq = q.T @ q
        o_residual = qtq - jnp.eye(self.n)

        # Only sum over upper triangular part (I <= J) as per SIF
        # Create upper triangular mask
        triu_mask = jnp.triu(jnp.ones((self.n, self.n), dtype=bool))

        # Compute objective: sum of squared residuals for upper triangular part
        # Use where to avoid boolean multiplication issues
        e_squared = jnp.where(triu_mask, e_residual**2, 0.0)
        o_squared = jnp.where(triu_mask, o_residual**2, 0.0)
        # Cast to ensure pyright understands these are arrays
        total_obj = jnp.sum(jnp.asarray(e_squared)) + jnp.sum(jnp.asarray(o_squared))

        return total_obj

    def y0(self):
        # Starting values as specified in SIF file:
        # - All variables default to 0.0
        # - D(J) eigenvalues are set to 1.0
        # - Q(J,J) diagonal elements are set to 1.0

        # Build the interleaved ordering: D(1), Q(:,1), D(2), Q(:,2), etc.
        # More efficient using reshape
        y_reshaped = jnp.zeros((self.n, self.n + 1))

        # Set D values to 1.0
        y_reshaped = y_reshaped.at[:, 0].set(1.0)

        # Set Q diagonal to 1.0 (identity matrix)
        # Q is stored column-wise, so Q(j,j) is at position j in column j
        # Vectorized approach: create diagonal indices
        diag_indices = jnp.arange(self.n)
        y_reshaped = y_reshaped.at[diag_indices, diag_indices + 1].set(1.0)

        return inexact_asarray(y_reshaped.ravel())

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is not provided in the SIF files
        return None

    def expected_objective_value(self):
        # These problems should have a minimum of 0.0
        return jnp.array(0.0)
