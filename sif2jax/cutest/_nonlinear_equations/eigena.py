import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class EIGENA(AbstractNonlinearEquations):
    """Solving symmetric eigenvalue problems as systems of nonlinear equations.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A = Q(T) D Q.

    Example A: a diagonal matrix with eigenvalues 1, ..., N.

    Source: An idea by Nick Gould

    Nonlinear equations version.

    SIF input: Nick Gould, Nov 1992.

    Classification: NOR2-AN-V-V

    TODO: Human review needed

    Attempts made:
    1. Fixed initial values to match pycutest pattern
       (D[0]=D[1]=1.0, rest 0; Q sparse pattern)
    2. Added bounds property (lower=0 for all variables,
       interpreting SIF "nonnegative eigenvalues")
    3. Tested multiple constraint orderings
       (eigen-first vs ortho-first, interleaved, SIF loop order)
    4. Special handling for diagonal constraints with unused eigenvalues

    Current status: ~18/21 tests passing, systematic 50.0 constraint
    value discrepancies remain

    Suspected issues:
    - pycutest may interpret SIF constraint formulation differently
      than mathematical A = Q^T D Q
    - Constraint values differ by exactly eigenvalue magnitudes
      (50.0, 49.0, etc.)
    - Issue appears at diagonal elements of eigen-equation constraints

    Resources needed:
    - Deep investigation into pycutest's SIF constraint parsing
    - Understanding of how pycutest handles GROUPS and CONSTANTS sections
    - Possible consultation with pycutest maintainers
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
        a = inexact_asarray(jnp.diag(jnp.arange(1, self.N + 1)))

        # Vectorized computation
        # Compute Q^T D Q
        qtdq = q.T @ jnp.diag(d) @ q

        # Compute Q^T Q
        qtq = q.T @ q

        # Identity matrix
        eye = jnp.eye(self.N)

        residuals = []

        # Use the most successful ordering: eigen constraints first, ortho second
        # All eigen equations E(I,J) first
        for j in range(self.N):
            for i in range(j + 1):
                residuals.append(qtdq[i, j] - a[i, j])

        # All orthogonality equations O(I,J) second
        for j in range(self.N):
            for i in range(j + 1):
                residuals.append(qtq[i, j] - eye[i, j])

        return inexact_asarray(jnp.array(residuals))

    @property
    def y0(self):
        """Initial guess."""
        y0 = jnp.zeros(self.n)

        # From pycutest behavior (differs from SIF file):
        # Set D[0] and D[1] to 1.0
        y0 = y0.at[0].set(1.0)
        y0 = y0.at[1].set(1.0)

        # pycutest sets Q matrix with a specific pattern:
        # Row i has non-zeros at columns (i+1) mod 50 and (2*i+3) mod 50
        # with special handling for certain rows
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
        # the solution is D = diag(1, ..., N) and Q = I
        d_expected = inexact_asarray(jnp.arange(1, self.N + 1))
        q_expected = inexact_asarray(jnp.eye(self.N).flatten())
        return jnp.concatenate([d_expected, q_expected])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (0 for constrained formulation)."""
        return inexact_asarray(jnp.array(0.0))

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """Nonnegative bounds on all variables as per SIF comment."""
        # The SIF file has comment "nonnegative eigenvalues" which pycutest
        # interprets as lower bounds of 0.0 on ALL variables
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper
