"""Solving symmetric eigenvalue problems as systems of nonlinear equations.

The problem is, given a symmetric matrix A, to find an orthogonal
matrix Q and diagonal matrix D such that A Q(T) = Q(T) D.

Example B: a tridiagonal matrix with diagonals 2 and off diagonals -1

Source: An idea by Nick Gould

Nonlinear equations version 2.

SIF input: Nick Gould, Nov 1992.

Classification: QQR2-AN-V-V
"""

import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class EIGENB2(AbstractConstrainedMinimisation):
    """EIGENB2 - Quadratic eigenvalue problem for tridiagonal matrix."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    _N: int = 50  # Default dimension from SIF

    def __init__(self, N: int = 50):
        self._N = N

    @property
    def n(self):
        """Number of variables: N eigenvalues + N² eigenvector components."""
        return self._N + self._N * self._N

    @property
    def m(self):
        """Number of constraints: N(N+1)/2 orthogonality constraints."""
        return self._N * (self._N + 1) // 2

    def _get_matrix_a(self):
        """Create the tridiagonal matrix A with diagonal 2 and off-diagonal -1."""
        # Build tridiagonal matrix
        diag = jnp.full(self._N, 2.0)
        off_diag = jnp.full(self._N - 1, -1.0)

        # Create full matrix
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
        """Compute the objective function - only E(I,J) groups (eigen equations).

        From SIF structure analysis: E groups are in the objective with L2 type,
        O groups are constraints.
        """
        del args

        d, q = self._unpack_variables(y)
        a = self._get_matrix_a()

        # E(I,J) groups: eigen equations Q^T * D - A @ Q^T
        # Vectorized: each (i,j) gives residual (Q^T * D)_{ij} - (A @ Q^T)_{ij}
        qt_d = q.T * d[None, :]  # Broadcasting: (N,N) * (1,N) -> (N,N)
        a_qt = a @ q.T  # (N,N) @ (N,N) -> (N,N)
        e_residuals = qt_d - a_qt  # (N,N)

        return jnp.sum(e_residuals**2)

    def constraint(self, y):
        """Compute the orthogonality constraints: Q^T Q - I = 0."""
        d, q = self._unpack_variables(y)

        # Compute Q^T Q
        qtq = q.T @ q
        identity = jnp.eye(self._N)

        # Extract constraints following exact SIF order: for J, for I <= J: O(I,J)
        constraints = []
        for j in range(self._N):  # J from 1 to N (0-indexed)
            for i in range(j + 1):  # I from 1 to J (0-indexed: 0 to j)
                constraints.append(qtq[i, j] - identity[i, j])

        return jnp.array(constraints), None  # Only equality constraints

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
