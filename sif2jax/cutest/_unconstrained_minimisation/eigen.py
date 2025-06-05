import jax.numpy as jnp

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

    n: int = 50  # Default dimension - individual problems may override

    def _matrix(self):
        """Return the specific matrix A for this eigenvalue problem.

        This method is implemented by each subclass for its specific matrix.
        """
        raise NotImplementedError("Subclasses must implement _matrix")

    def objective(self, y, args):
        del args

        # y contains the variables as ordered in SIF file:
        # First n elements are eigenvalues D(J)
        # Then n² elements are Q(I,J) matrix in the order defined by SIF loops
        d_diag = y[: self.n]
        q_vars = y[self.n :]

        # Need to reconstruct Q matrix from SIF variable ordering
        # SIF defines Q(I,J) with nested loops: for J=1 to N, for I=1 to N
        q = q_vars.reshape((self.n, self.n), order="F")

        # Get the target matrix A
        a = self._matrix()

        # Eigenvalue equations: E(I,J) = sum_K Q(K,I) * Q(K,J) * D(K) - A(I,J)
        # Vectorized: QᵀDQ where D is diagonal
        qtdq = q.T @ jnp.diag(d_diag) @ q

        # Orthogonality equations: O(I,J) = sum_K Q(K,I) * Q(K,J) - delta_IJ
        # Vectorized: QᵀQ - I
        qtq = q.T @ q

        total_obj = 0.0

        # Only sum over upper triangular part (I <= J) as per SIF loops
        for j in range(self.n):  # J = 1 to N
            for i in range(j + 1):  # I = 1 to J
                # Eigenvalue equation residual
                e_ij = qtdq[i, j] - a[i, j]
                total_obj += e_ij**2

                # Orthogonality equation residual
                o_ij = qtq[i, j]
                if i == j:
                    o_ij -= 1.0  # Diagonal elements should be 1
                # Off-diagonal elements should be 0, so no subtraction needed
                total_obj += o_ij**2

        return jnp.array(total_obj)

    def y0(self):
        # Starting values as specified in SIF file:
        # - All variables default to 0.0
        # - D(J) eigenvalues are set to 1.0
        # - Q(J,J) diagonal elements are set to 1.0

        # Order matches SIF: first D(J), then Q(I,J)
        # Initialize D eigenvalues to 1.0
        d_diag = jnp.ones(self.n)

        # Initialize Q matrix elements to 0, then set diagonal to 1
        q_matrix = jnp.zeros((self.n, self.n))
        q_matrix = q_matrix.at[jnp.diag_indices(self.n)].set(1.0)
        q_flat = q_matrix.flatten(order="F")  # Flatten in column-major order

        return jnp.concatenate([d_diag, q_flat])

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is not provided in the SIF files
        return None

    def expected_objective_value(self):
        # These problems should have a minimum of 0.0
        return jnp.array(0.0)
