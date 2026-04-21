import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HADAMALS(AbstractBoundedMinimisation):
    """Hadamard matrix search problem with ±1 entries.

    An attempt to find Hadamard matrices of order N.
    The problem is to find an N by N orthonormal matrix Q,
    with column norms N, whose entries are plus or minus one.

    Variables: N×N matrix Q flattened to N² variables
    Bounds: All variables in [-1, 1]

    Objective: Sum of squared deviations from orthogonality constraints
               plus penalties for entries not being ±1

    Source: A suggestion by Alan Edelman (MIT).
    SIF input: Nick Gould, Nov 1993.
    Classification: OBR2-RN-V-V

    Default N = 20 (400 variables).
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 20
    n_squared: int = 400

    def __init__(self, n: int = 20):
        """Initialize HADAMALS problem.

        Args:
            n: Size of the Hadamard matrix (default 20)
        """
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "n_squared", n * n)

    def objective(self, y, args):
        """Compute objective function.

        The objective has two types of groups:
        1. O(I,J): Orthogonality constraints with L2 group type
        2. S(I,J): Entry constraints with LARGEL2 group type (scaled L2)
        """
        n = self.n

        # Reshape flat vector to matrix (column-major for SIF compatibility)
        Q = y.reshape((n, n), order="F")

        # Q^T @ Q orthogonality: want Q^T Q = n*I
        # sum_{i<=j} (QtQ[i,j] - n*delta_{ij})^2
        # Correct diagonal so 0.5*sum(M^2) = triu_sum:
        # need M[i,i] = sqrt(2)*(d[i]-n) so 0.5*M[i,i]^2 = (d[i]-n)^2
        QtQ = Q.T @ Q
        d = jnp.diagonal(QtQ)
        diag_correction = (jnp.sqrt(2.0) - 1.0) * d - jnp.sqrt(2.0) * n
        corrected = QtQ + jnp.diag(diag_correction)
        obj = 0.5 * jnp.sum(corrected**2)

        # S(I,J) groups: entry constraints (Q[i,j]^2 - 1)^2 for i>=2
        Q_slice = Q[1:, :]
        entry_vals = Q_slice**2 - 1.0
        obj = obj + jnp.sum(entry_vals**2)

        return jnp.array(obj)

    @property
    def y0(self):
        """Starting point as specified in SIF file."""
        n = self.n
        n_half = n // 2

        # Initialize matrix
        Q0 = jnp.zeros((n, n))

        # Set initial values for each column as per SIF file
        # Note: The bounds will be enforced separately, so we don't override here
        for j in range(n):
            # First half of rows: 0.9
            Q0 = Q0.at[:n_half, j].set(0.9)
            # Second half of rows: -0.9
            Q0 = Q0.at[n_half:, j].set(-0.9)

        # Flatten in column-major order (Fortran convention)
        return Q0.flatten(order="F")

    @property
    def bounds(self):
        """All variables bounded in [-1, 1], with first column fixed."""
        n = self.n
        n_half = n // 2

        # Default bounds: [-1, 1] for all variables
        lower = -jnp.ones(self.n_squared)
        upper = jnp.ones(self.n_squared)

        # Fix first column values (column-major indexing)
        # Q(i,1) in SIF is at index i in column-major flattened array
        for i in range(n_half):
            lower = lower.at[i].set(1.0)
            upper = upper.at[i].set(1.0)
        for i in range(n_half, n):
            lower = lower.at[i].set(-1.0)
            upper = upper.at[i].set(-1.0)

        return lower, upper

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Optimal solution would be a true Hadamard matrix if one exists
        return None

    @property
    def expected_objective_value(self):
        # Perfect Hadamard matrix would have objective 0
        return jnp.array(0.0)
