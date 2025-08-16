import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HADAMARD(AbstractConstrainedMinimisation):
    """Hadamard matrix problem with minimized maximum entry.

    An attempt to find Hadamard matrices of order N.
    The problem is to find an N by N orthonormal matrix Q,
    with column norms sqrt(N), whose largest entry is as small
    as possible in absolute value.

    Variables: MAXABSQ + N×N matrix Q flattened = 1 + N² variables
    Objective: Minimize MAXABSQ
    Constraints:
      - Orthogonality: Q^T Q = sqrt(N) I (equality)
      - Entry bounds: -MAXABSQ <= Q(i,j) <= MAXABSQ (inequality)

    Source: A suggestion by Alan Edelman (MIT).
    SIF input: Nick Gould, Nov 1993.
    Classification: LQR2-RN-V-V

    Default N = 20 (401 variables).
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 20
    n_squared: int = 400
    total_vars: int = 401
    n_constraints: int = 610  # (n*(n+1))/2 + 2*n*n

    def __init__(self, n: int = 20):
        """Initialize HADAMARD problem.

        Args:
            n: Size of the Hadamard matrix (default 20)
        """
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "n_squared", n * n)
        object.__setattr__(self, "total_vars", 1 + n * n)
        # Number of constraints: orthogonality + entry bounds
        # Orthogonality: n*(n+1)/2 (upper triangle including diagonal)
        # Entry bounds: 2*n*n (lower and upper for each entry)
        n_orth = (n * (n + 1)) // 2
        n_bounds = 2 * n * n
        object.__setattr__(self, "n_constraints", n_orth + n_bounds)

    def objective(self, y, args):
        """Compute objective function.

        The objective is simply MAXABSQ.
        """
        return y[0]  # MAXABSQ is the first variable

    def constraint(self, y):
        """Compute constraint violations.

        Returns:
            Tuple of (equality_constraints, inequality_constraints) where:
            - equality_constraints: orthogonality constraints Q^T Q - sqrt(N) I = 0
            - inequality_constraints: entry bound constraints |Q(i,j)| <= MAXABSQ
        """
        n = self.n
        rn = float(n)  # RN in SIF is N, not sqrt(N)

        maxabsq = y[0]
        Q = y[1:].reshape((n, n), order="F")  # Column-major order

        # Compute Q^T Q
        QtQ = jnp.dot(Q.T, Q)

        # Orthogonality constraints: Q^T Q - RN*I = 0
        # Store upper triangle including diagonal
        orth_constraints = []
        for j in range(n):
            for i in range(j + 1):
                if i == j:
                    # Diagonal: QtQ[i,i] - RN = 0
                    val = QtQ[i, j] - rn
                else:
                    # Off-diagonal: QtQ[i,j] = 0
                    val = QtQ[i, j]
                orth_constraints.append(val)

        equality_constraints = jnp.array(orth_constraints)

        # Entry bound constraints
        # For each Q(i,j): -MAXABSQ <= Q(i,j) <= MAXABSQ
        # In SIF file, these are defined interleaved:
        # For each (I,J): L(I,J), U(I,J)
        # L(I,J): MAXABSQ + Q(I,J) >= 0 (lower bound)
        # U(I,J): MAXABSQ - Q(I,J) >= 0 (upper bound)
        Q_flat = Q.reshape(-1, order="F")

        # Build interleaved constraints
        inequality_constraints = []
        for i in range(self.n_squared):
            # L constraint: MAXABSQ + Q[i] >= 0
            inequality_constraints.append(maxabsq + Q_flat[i])
            # U constraint: MAXABSQ - Q[i] >= 0
            inequality_constraints.append(maxabsq - Q_flat[i])

        inequality_constraints = jnp.array(inequality_constraints)

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Starting point as specified in SIF file."""
        n = self.n

        # MAXABSQ starts at 0.0
        maxabsq_init = 0.0

        # Q starts with all entries at 1.0
        Q_init = jnp.ones((n, n))

        # Flatten in column-major order
        Q_flat = Q_init.reshape(-1, order="F")

        return jnp.concatenate([jnp.array([maxabsq_init]), Q_flat])

    @property
    def bounds(self):
        """Variable bounds.

        MAXABSQ >= 0, Q variables are unbounded.
        """
        total_vars = self.total_vars

        # Lower bounds: MAXABSQ >= 0, others unbounded
        lower = jnp.concatenate([jnp.array([0.0]), -jnp.inf * jnp.ones(self.n_squared)])

        # Upper bounds: all unbounded
        upper = jnp.inf * jnp.ones(total_vars)

        return lower, upper

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution depends on n
        return None

    @property
    def expected_objective_value(self):
        # Optimal MAXABSQ depends on whether a Hadamard matrix exists
        return None
