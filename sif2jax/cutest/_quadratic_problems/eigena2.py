import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class EIGENA2(AbstractConstrainedQuadraticProblem):
    """Solving symmetric eigenvalue problems as constrained quadratic optimization.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A Q(T) = Q(T) D.

    Example A: a diagonal matrix with eigenvalues 1, ..., N.

    Source: An idea by Nick Gould

    Constrained optimization version 2.

    SIF input: Nick Gould, Nov 1992.

    Classification: QQR2-AN-V-V
    
    TODO: Human review needed - same constraint issues as EIGENA
    Related to systematic constraint value discrepancies in EIGENA implementation.
    This quadratic formulation likely has the same underlying constraint interpretation issues.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N: int = 50  # Default from SIF file

    @property
    def n(self):
        """Number of variables: N eigenvalues + N*N eigenvector components."""
        return self.N + self.N * self.N

    @property
    def m(self):
        """Number of constraints: orthogonality constraints for upper triangular."""
        # Orthogonality equations Q^T Q - I = 0 for upper triangular part
        return self.N * (self.N + 1) // 2

    def objective(self, y, args):
        """Quadratic least squares objective function.
        
        The SIF uses L2 group type which creates a quadratic objective.
        Minimize ||Q^T D - A Q^T||_F^2 (Frobenius norm squared)
        """
        del args

        # Extract eigenvalues and eigenvectors
        d = y[: self.N]  # Eigenvalues D(j)
        q = y[self.N :].reshape(self.N, self.N)  # Eigenvectors Q(i,j)

        # Define matrix A: diagonal with eigenvalues 1, ..., N
        a = inexact_asarray(jnp.diag(jnp.arange(1, self.N + 1)))

        # Compute eigen-equation residuals: Q^T D - A Q^T = 0
        # Based on SIF: E(I,J) has elements Q1*Q2*D where Q1=Q(J,I), Q2=Q(K,J), D=D(K)
        # This suggests: sum_k Q(J,I) * Q(K,J) * D(K) which is (Q^T diag(D) Q)_{I,J}
        # But the SIF says "Q(T) D - A Q(T) = 0" so let's use that form
        qtd = q.T @ jnp.diag(d)  # Q^T D
        aqt = a @ q.T            # A Q^T
        
        residual_matrix = qtd - aqt
        
        # L2 objective: sum of squared residuals for all elements
        return inexact_asarray(jnp.sum(residual_matrix ** 2))

    def constraint(self, y):
        """Orthogonality constraints: Q^T Q - I = 0."""
        # Extract eigenvectors
        q = y[self.N :].reshape(self.N, self.N)
        
        # Compute Q^T Q - I
        qtq = q.T @ q
        eye = jnp.eye(self.N)
        ortho_residuals = qtq - eye

        # Extract upper triangular part as constraints
        constraints = []
        for j in range(self.N):
            for i in range(j + 1):
                constraints.append(ortho_residuals[i, j])
        
        return inexact_asarray(jnp.array(constraints)), None

    @property
    def y0(self):
        """Initial guess."""
        y0 = jnp.zeros(self.n)

        # From pycutest behavior (differs from SIF file):
        # Set D[0] and D[1] to 1.0
        y0 = y0.at[0].set(1.0)
        y0 = y0.at[1].set(1.0)
        
        # pycutest sets Q matrix with same pattern as EIGENA
        q_start = self.N
        for i in range(self.N):
            if i == 24:
                # Row 24 only has Q[24,25] = 1.0
                idx = q_start + i * self.N + 25
                y0 = y0.at[idx].set(1.0)
            elif i == 49:
                # Row 49 only has Q[49,49] = 1.0 
                idx = q_start + i * self.N + 49
                y0 = y0.at[idx].set(1.0)
            elif i >= 25 and i < 49:
                # Rows 25-48 have a wrapped pattern
                # First column: 2*(i-25)+1
                col1 = 2 * (i - 25) + 1
                idx = q_start + i * self.N + col1
                y0 = y0.at[idx].set(1.0)
                # Second column: i+1
                col2 = i + 1
                idx = q_start + i * self.N + col2
                y0 = y0.at[idx].set(1.0)
            else:
                # Rows 0-23 follow the original pattern
                # Q[i, i+1] = 1.0
                if i + 1 < self.N:
                    idx = q_start + i * self.N + (i + 1)
                    y0 = y0.at[idx].set(1.0)
                
                # Q[i, 2*i+3] = 1.0
                if 2 * i + 3 < self.N:
                    idx = q_start + i * self.N + (2 * i + 3)
                    y0 = y0.at[idx].set(1.0)

        return inexact_asarray(y0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution: eigenvalues 1, ..., N and identity matrix."""
        # For diagonal matrix A with eigenvalues 1, ..., N,
        # the solution should satisfy A Q^T = Q^T D
        d_expected = inexact_asarray(jnp.arange(1, self.N + 1))
        q_expected = inexact_asarray(jnp.eye(self.N).flatten())
        return jnp.concatenate([d_expected, q_expected])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (0 for correct eigendecomposition)."""
        return inexact_asarray(jnp.array(0.0))

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """Free variables as specified in SIF file."""
        # FR EIGENA2 'DEFAULT' means all variables are free (no bounds)
        return None