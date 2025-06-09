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
        # For each J from 1 to N:
        #   - D(J) (eigenvalue)
        #   - Q(1,J), Q(2,J), ..., Q(N,J) (J-th column of Q)
        # So the ordering is interleaved: D(1), Q(:,1), D(2), Q(:,2), etc.

        # Extract D and Q from the interleaved ordering
        d_diag = jnp.zeros(self.n)
        q = jnp.zeros((self.n, self.n))

        idx = 0
        for j in range(self.n):
            # Extract D(j+1)
            d_diag = d_diag.at[j].set(y[idx])
            idx += 1

            # Extract Q(:,j+1) - the (j+1)-th column of Q
            for i in range(self.n):
                q = q.at[i, j].set(y[idx])
                idx += 1

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

        # Build the interleaved ordering: D(1), Q(:,1), D(2), Q(:,2), etc.
        total_vars = self.n + self.n * self.n  # n eigenvalues + n² matrix elements
        y = jnp.zeros(total_vars)

        idx = 0
        for j in range(self.n):
            # Set D(j+1) = 1.0
            y = y.at[idx].set(1.0)
            idx += 1

            # Set Q(:,j+1) with Q(j+1,j+1) = 1.0 and rest = 0.0
            for i in range(self.n):
                if i == j:
                    y = y.at[idx].set(1.0)  # Q(j+1,j+1) = 1.0
                # else: keep 0.0 (default)
                idx += 1

        return y

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is not provided in the SIF files
        return None

    def expected_objective_value(self):
        # These problems should have a minimum of 0.0
        return jnp.array(0.0)
