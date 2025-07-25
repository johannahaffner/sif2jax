import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class SPMSRTLS(AbstractUnconstrainedMinimisation):
    """Liu and Nocedal tridiagonal matrix square root problem.

    This is a least-squares variant of problem SPMSQRT. The problem is to find
    a tridiagonal matrix X such that X^T * X approximates a pentadiagonal matrix A
    in the least-squares sense.

    The matrix dimension is M, and the number of variables is 3*M-2.
    The variables represent the entries of the tridiagonal matrix X.

    Source: problem 151 (p. 93) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-V
    """

    M: int = (
        1667  # Dimension of the matrix (default: 1667, which gives n=4999 variables)
    )
    n: int = 4999  # Number of variables (3*M-2)

    @property
    def y0_iD(self) -> int:
        return 0

    @property
    def provided_y0s(self) -> frozenset:
        return frozenset({0})

    def _compute_B_matrix(self):
        """Compute the tridiagonal matrix B."""
        M = self.M

        # B is an M x M tridiagonal matrix
        # We'll store the diagonals vectorized
        B_diag = jnp.zeros(M)
        B_lower = jnp.zeros(M - 1)  # B[i+1, i] for i = 0, ..., M-2
        B_upper = jnp.zeros(M - 1)  # B[i, i+1] for i = 0, ..., M-2

        # Build k values for all entries following SIF pattern
        # Row 1: k values are not used, we use fixed values
        B_diag = B_diag.at[0].set(jnp.sin(1.0))
        B_upper = B_upper.at[0].set(jnp.sin(4.0))

        # Rows 2 to M generate k values from 3 onwards
        # For each row i (1-indexed, so i=2 to M), we have 3 values
        # Total k values needed: 3*(M-1) starting from k=3
        k_start = 3
        k_values = jnp.arange(k_start, k_start + 3 * (M - 1))
        k_squared = k_values**2
        sin_k_squared = jnp.sin(k_squared)

        # Distribute to the diagonals
        # For row i (0-indexed i = 1 to M-1), we use k values at positions
        # 3*(i-1), 3*(i-1)+1, 3*(i-1)+2
        idx = 0
        for i in range(1, M):
            if i < M - 1:  # Not last row
                B_lower = B_lower.at[i - 1].set(sin_k_squared[idx])
                B_diag = B_diag.at[i].set(sin_k_squared[idx + 1])
                B_upper = B_upper.at[i].set(sin_k_squared[idx + 2])
                idx += 3
            else:  # Last row
                B_lower = B_lower.at[i - 1].set(sin_k_squared[idx])
                B_diag = B_diag.at[i].set(sin_k_squared[idx + 1])
                # No upper diagonal for last row

        return B_diag, B_lower, B_upper

    def objective(self, y, args):
        """Compute the least-squares objective function."""
        del args

        M = self.M

        # Unpack variables into tridiagonal matrix X
        # Variables are ordered as: X(1,1), X(1,2), X(2,1), X(2,2), X(2,3), ...
        # Row 1: 2 elements, Rows 2 to M-1: 3 elements each, Row M: 2 elements
        X_diag = jnp.zeros(M)
        X_lower = jnp.zeros(M - 1)
        X_upper = jnp.zeros(M - 1)

        idx = 0
        # Row 1
        X_diag = X_diag.at[0].set(y[idx])
        X_upper = X_upper.at[0].set(y[idx + 1])
        idx += 2

        # Rows 2 to M-1
        for i in range(1, M - 1):
            X_lower = X_lower.at[i - 1].set(y[idx])
            X_diag = X_diag.at[i].set(y[idx + 1])
            X_upper = X_upper.at[i].set(y[idx + 2])
            idx += 3

        # Row M
        X_lower = X_lower.at[M - 2].set(y[idx])
        X_diag = X_diag.at[M - 1].set(y[idx + 1])

        # Compute B matrix
        B_diag, B_lower, B_upper = self._compute_B_matrix()

        # Compute target matrix A = B^T * B entries
        # Main diagonal of A
        A_diag = jnp.zeros(M)
        A_diag = A_diag.at[0].set(B_diag[0] ** 2 + B_upper[0] * B_lower[0])
        A_diag = A_diag.at[1 : M - 1].set(
            B_diag[1 : M - 1] ** 2 + B_lower[:-1] ** 2 + B_upper[1:] ** 2
        )
        A_diag = A_diag.at[M - 1].set(B_diag[M - 1] ** 2 + B_lower[M - 2] ** 2)

        # First sub/super-diagonals of A
        A_sub1 = B_lower * B_diag[:-1] + B_diag[1:] * B_lower
        A_super1 = B_upper * B_diag[1:] + B_diag[:-1] * B_upper

        # Second sub/super-diagonals of A
        A_sub2 = B_upper[1:] * B_lower[:-1]
        A_super2 = B_upper[:-1] * B_lower[1:]

        # Compute X^T * X entries
        # Main diagonal
        XTX_diag = jnp.zeros(M)
        XTX_diag = XTX_diag.at[0].set(X_diag[0] ** 2 + X_upper[0] * X_lower[0])
        XTX_diag = XTX_diag.at[1 : M - 1].set(
            X_diag[1 : M - 1] ** 2 + X_lower[:-1] ** 2 + X_upper[1:] ** 2
        )
        XTX_diag = XTX_diag.at[M - 1].set(X_diag[M - 1] ** 2 + X_lower[M - 2] ** 2)

        # First sub/super-diagonals
        XTX_sub1 = X_lower * X_diag[:-1] + X_diag[1:] * X_lower
        XTX_super1 = X_upper * X_diag[1:] + X_diag[:-1] * X_upper

        # Second sub/super-diagonals
        XTX_sub2 = X_upper[1:] * X_lower[:-1]
        XTX_super2 = X_upper[:-1] * X_lower[1:]

        # Compute sum of squared residuals
        obj = (
            jnp.sum((XTX_diag - A_diag) ** 2)
            + jnp.sum((XTX_sub1 - A_sub1) ** 2)
            + jnp.sum((XTX_super1 - A_super1) ** 2)
            + jnp.sum((XTX_sub2 - A_sub2) ** 2)
            + jnp.sum((XTX_super2 - A_super2) ** 2)
        )

        return obj

    @property
    def y0(self):
        """Initial point: 0.2 * B."""
        M = self.M
        B_diag, B_lower, B_upper = self._compute_B_matrix()

        # Pack initial values in the same order as variables
        y0_values = []

        # Row 1
        y0_values.append(0.2 * B_diag[0])
        y0_values.append(0.2 * B_upper[0])

        # Rows 2 to M-1
        for i in range(1, M - 1):
            y0_values.append(0.2 * B_lower[i - 1])
            y0_values.append(0.2 * B_diag[i])
            y0_values.append(0.2 * B_upper[i])

        # Row M
        y0_values.append(0.2 * B_lower[M - 2])
        y0_values.append(0.2 * B_diag[M - 1])

        return inexact_asarray(jnp.array(y0_values))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None
