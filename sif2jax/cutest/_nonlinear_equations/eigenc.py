"""Solving symmetric eigenvalue problems as systems of nonlinear equations.

The problem is, given a symmetric matrix A, to find an orthogonal
matrix Q and diagonal matrix D such that A = Q(T) D Q.

Example C: a tridiagonal matrix suggested by J. H. Wilkinson.

Source: An idea by Nick Gould

SIF input: Nick Gould, Nov 1992.

Nonlinear equations version.

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class EIGENC(AbstractNonlinearEquations):
    """EIGENC - Eigenvalue problem for Wilkinson tridiagonal matrix."""

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
        """Number of constraints: N(N+1)/2 eigen equations + N(N+1)/2 orthogonality."""
        return self._N * (self._N + 1)  # Total of both types

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
        diag = jnp.zeros(n)
        for j in range(n):
            diag = diag.at[j].set(float(m - j))

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

    def constraint_residuals(self, y, args):
        """Compute constraint residuals.

        Returns residuals for:
        1. Eigen equations: Q^T D Q - A = 0 (upper triangular part)
        2. Orthogonality: Q^T Q - I = 0 (upper triangular part)
        """
        del args

        d, q = self._unpack_variables(y)
        a = self._get_matrix_a()

        # Compute Q^T D Q
        qtdq = q.T @ (d[:, None] * q)

        # Compute Q^T Q
        qtq = q.T @ q

        # Interleave eigen and orthogonality residuals following SIF structure
        # For each J (column), for each I <= J (row): E(I,J) then O(I,J)
        identity = jnp.eye(self._N)
        all_residuals = []

        for j in range(self._N):  # J from 1 to N (0-indexed)
            for i in range(j + 1):  # I from 1 to J (0-indexed: 0 to j)
                # E(I,J): eigen equation Q^T D Q - A = 0
                e_residual = qtdq[i, j] - a[i, j]
                all_residuals.append(e_residual)

                # O(I,J): orthogonality Q^T Q - I = 0
                o_residual = qtq[i, j] - identity[i, j]
                all_residuals.append(o_residual)

        all_residuals = jnp.array(all_residuals)

        return all_residuals

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
    def bounds(self):
        """No bounds for this problem."""
        return None

    def constraint(self, y):
        """Compute the nonlinear equations (constraint residuals)."""
        return self.constraint_residuals(y, None), None

    @property
    def expected_objective_value(self):
        """The objective should be 0.0 when constraints are satisfied."""
        return jnp.array(0.0)

    @property
    def expected_result(self):
        """The expected solution is not provided in the SIF file."""
        return None
