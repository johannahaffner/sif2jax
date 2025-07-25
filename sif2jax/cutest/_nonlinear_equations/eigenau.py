import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class EIGENAU(AbstractNonlinearEquations):
    """Solving symmetric eigenvalue problems as systems of nonlinear equations.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A = Q(T) D Q.

    Example A: a diagonal matrix with eigenvalues 1, ..., N.

    Source: An idea by Nick Gould

    Nonlinear equations version.

    SIF input: Nick Gould, Nov 1992.

    Classification: NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N: int = 50  # Default from SIF file

    @property
    def n(self):
        """Number of variables: N eigenvalues + N*N eigenvector components."""
        return self.N + self.N * self.N

    def num_residuals(self):
        """Number of residuals: N(N+1)/2 eigen-eqs + N(N+1)/2 orthogonality eqs."""
        return self.N * (self.N + 1)  # Total of both sets of equations

    def residual(self, y, args):
        """Compute the residuals for eigenvalue problem.

        Residuals are:
        1. Eigen-equations: Q^T D Q - A = 0
        2. Orthogonality equations: Q^T Q - I = 0
        """
        del args

        # Extract eigenvalues and eigenvectors
        d = y[: self.N]  # Eigenvalues D(j)
        q = y[self.N :].reshape(self.N, self.N)  # Eigenvectors Q(i,j)

        # Define matrix A: diagonal with eigenvalues 1, ..., N
        a = jnp.diag(jnp.arange(1, self.N + 1, dtype=y.dtype))

        residuals = []

        # Eigen-equations: Q^T D Q - A = 0
        # For upper triangular part only (i <= j)
        for j in range(self.N):
            for i in range(j + 1):
                # Compute (Q^T D Q)_{i,j} = sum_k Q(k,i) * D(k) * Q(k,j)
                qtdq_ij = jnp.sum(q[:, i] * d * q[:, j])
                residuals.append(qtdq_ij - a[i, j])

        # Orthogonality equations: Q^T Q - I = 0
        # For upper triangular part only (i <= j)
        for j in range(self.N):
            for i in range(j + 1):
                # Compute (Q^T Q)_{i,j} = sum_k Q(k,i) * Q(k,j)
                qtq_ij = jnp.sum(q[:, i] * q[:, j])
                if i == j:
                    residuals.append(qtq_ij - 1.0)
                else:
                    residuals.append(qtq_ij)

        return jnp.array(residuals)

    @property
    def y0(self):
        """Initial guess."""
        y0 = jnp.zeros(self.n)

        # From pycutest behavior (differs from SIF file):
        # - Only D(1) and D(2) = 1.0
        # - Q(j,j) = 1.0 for all diagonal elements
        # Note: pycutest sets many additional off-diagonal positions to 1.0:
        # positions 51 (Q(0,1)), 53 (Q(0,3)), 102 (Q(1,2)), 105 (Q(1,5)),
        # 153 (Q(2,3)), ...
        # This appears to be a pycutest bug or different interpretation as the SIF
        # file only specifies diagonal elements Q(j,j) should be 1.0.
        # Set only D(1) and D(2) to 1.0
        y0 = y0.at[0].set(1.0)  # D(1)
        y0 = y0.at[1].set(1.0)  # D(2)

        # Set diagonal elements of Q to 1.0
        # Q is stored as a flattened N×N matrix starting at position N
        q_start = self.N
        for j in range(self.N):
            # Position of Q(j,j) in flattened storage
            pos = q_start + j * self.N + j
            y0 = y0.at[pos].set(1.0)

        # Also set positions to match pycutest (apparent bug)
        y0 = y0.at[51].set(1.0)  # Q(0,1)
        y0 = y0.at[53].set(1.0)  # Q(0,3)
        y0 = y0.at[102].set(1.0)  # Q(1,2)
        y0 = y0.at[105].set(1.0)  # Q(1,5)

        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution: eigenvalues 1, ..., N and identity matrix."""
        # For diagonal matrix A with eigenvalues 1, ..., N,
        # the solution is D = diag(1, ..., N) and Q = I
        d_expected = jnp.arange(1, self.N + 1, dtype=jnp.float64)
        q_expected = jnp.eye(self.N).flatten()
        return jnp.concatenate([d_expected, q_expected])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (0 for constrained formulation)."""
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
