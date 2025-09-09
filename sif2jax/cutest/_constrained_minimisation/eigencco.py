"""Solving symmetric eigenvalue problems as systems of nonlinear equations.

The problem is, given a symmetric matrix A, to find an orthogonal
matrix Q and diagonal matrix D such that A = Q(T) D Q.

Example C: a tridiagonal matrix suggested by J. H. Wilkinson.

Source: An idea by Nick Gould

Constrained optimization version

SIF input: Nick Gould, Nov 1992.

Classification: SQR2-AN-V-V
"""

import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class EIGENCCO(AbstractConstrainedMinimisation):
    """EIGENCCO - Sum-of-squares eigenvalue problem for Wilkinson matrix."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    _M: int = 25  # Default M value from SIF

    def __init__(self, M: int = 25):
        self._M = M

    @property
    def _N(self):
        """The dimension of the matrix is 2M + 1."""
        return 2 * self._M + 1

    @property
    def n(self):
        """Number of variables: N eigenvalues + N² eigenvector components."""
        return self._N + self._N * self._N

    @property
    def m(self):
        """Number of constraints: N(N+1)/2 orthogonality constraints."""
        return self._N * (self._N + 1) // 2

    def _get_matrix_a(self):
        """Create the Wilkinson tridiagonal matrix.

        Diagonal entries: M, M-1, M-2, ..., 1, 0, 1, ..., M-2, M-1, M
        Off-diagonal entries: all 1
        """
        n = self._N
        m = self._M

        # Create diagonal: M, M-1, ..., 0, ..., M-1, M
        # From SIF: A(J,J) = M+1-J (for J from 1 to N in 1-based indexing)
        # Converting to 0-based: A[j,j] = M+1-(j+1) = M-j
        diag = m - jnp.arange(n, dtype=float)

        # Create off-diagonals: all 1.0
        off_diag = jnp.ones(n - 1)

        # Build full matrix
        a = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
        return a

    def _unpack_variables(self, y):
        """Unpack variables from flat array.

        Variables are ordered as: D(1), Q(1,1), Q(2,1), ..., Q(N,1), D(2), Q(1,2), ...
        """
        # Reshape to (N, N+1) where each row is [D(j), Q(1,j), Q(2,j), ..., Q(N,j)]
        y_reshaped = y.reshape(self._N, self._N + 1)

        # Extract D (eigenvalues) and Q (eigenvectors)
        d = y_reshaped[:, 0]  # Shape (N,)
        q = y_reshaped[:, 1:].T  # Shape (N, N) - transpose to get Q as columns

        return d, q

    def objective(self, y, args):
        """Compute the objective function.

        Minimizes the sum of squares of eigen equations: Q(T) D Q - A = 0
        """
        del args

        d, q = self._unpack_variables(y)
        a = self._get_matrix_a()

        # Compute Q^T D Q
        qtdq = q.T @ (d[:, None] * q)

        # Sum of squares for upper triangular part only (I <= J)
        # Vectorized: use triu_indices to get upper triangular indices
        i_indices, j_indices = jnp.triu_indices(self._N)

        # Eigen equation residuals: Q^T D Q - A
        e_residuals = qtdq[i_indices, j_indices] - a[i_indices, j_indices]
        total_obj = jnp.sum(e_residuals**2)

        return total_obj

    def constraint(self, y):
        """Compute the equality constraints.

        Orthogonality constraints: Q(T) Q - I = 0 (upper triangular part only)
        """

        d, q = self._unpack_variables(y)

        # Compute Q^T Q
        qtq = q.T @ q
        identity = jnp.eye(self._N)

        # Extract upper triangular part (including diagonal) as constraints
        # Vectorized approach using triu_indices to get upper triangular indices
        i_indices, j_indices = jnp.triu_indices(self._N)

        # The SIF ordering is column-major: for each j, all i <= j
        # Sort by j first, then by i to match the nested loop order
        order = jnp.lexsort((i_indices, j_indices))
        i_ordered = i_indices[order]
        j_ordered = j_indices[order]

        # Extract constraints in the correct order
        constraints = (qtq - identity)[i_ordered, j_ordered]

        return constraints, None  # Only equality constraints

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # Default all to 0.0
        # D(j) = 1.0 for all j
        # Q(j,j) = 1.0 (diagonal elements)

        # Build in the interleaved format
        y = jnp.zeros(self.n)

        # Use reshape for efficient assignment
        y_reshaped = y.reshape(self._N, self._N + 1)

        # Set D values to 1.0
        y_reshaped = y_reshaped.at[:, 0].set(1.0)

        # Set Q diagonal to 1.0
        diag_indices = jnp.arange(self._N)
        y_reshaped = y_reshaped.at[diag_indices, diag_indices + 1].set(1.0)

        return inexact_asarray(y_reshaped.ravel())

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """The minimum should be 0.0 when the eigenvalue problem is solved exactly."""
        return jnp.array(0.0)

    @property
    def expected_result(self):
        """The expected solution is not provided in the SIF file."""
        return None
