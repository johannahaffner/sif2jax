import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


# Problem parameters
N = 2500  # Number of variables
M = 700  # Number of constraints
COND = 3.0  # Conditioning parameter

# Compute problem data at module level
RTN = int(jnp.sqrt(N + 0.1))  # sqrt(N)

# Compute diagonal D for Hessian conditioning
i_vals = jnp.arange(1, N + 1, dtype=jnp.float64)
D = jnp.exp((i_vals - 1.0) * COND / (N - 1.0))

# Determine quadratic center XC (alternating -1, 1 pattern)
XC = jnp.zeros(N)
indices = jnp.arange(0, N - 1, 2)
XC = XC.at[indices].set(-1.0)
XC = XC.at[indices + 1].set(1.0)
XC = XC.at[N - 1].set(1.0)

# Y vector parameters (10 nonzeros as in paper)
Y_VALS = jnp.array(
    [
        -0.3569732,
        0.9871576,
        0.5619363,
        -0.1984624,
        0.4653328,
        0.7364367,
        -0.4560378,
        -0.6457813,
        -0.0601357,
        0.1035624,
    ]
)

# Positions of nonzeros (as fractions)
NZ_FRACS = jnp.array(
    [
        0.68971452,
        0.13452678,
        0.51234678,
        0.76591423,
        0.20857854,
        0.85672348,
        0.04356789,
        0.44692743,
        0.30136413,
        0.91367489,
    ]
)

# Convert to integer positions (0-indexed)
K_POSITIONS = jnp.floor(NZ_FRACS * float(N) + 1.1).astype(int) - 1

# Create sparse Y vector
Y = jnp.zeros(N)
Y = Y.at[K_POSITIONS].set(Y_VALS)

# Compute various products
YN2 = jnp.sum(Y_VALS**2)  # ||y||^2
DY = D * Y  # D * y
YDY = jnp.dot(Y, DY)  # y^T * D * y
YXC = jnp.dot(Y, XC)  # y^T * xc
YDXC = jnp.dot(DY, XC)  # y^T * D * xc

# Coefficients for quadratic term construction
MINUS_2_OVER_YN2 = -2.0 / YN2
FOUR_OVER_YN4 = 4.0 / (YN2 * YN2)
AA = MINUS_2_OVER_YN2 * YXC
DD = FOUR_OVER_YN4 * YDY
BB = DD * YXC
CC = MINUS_2_OVER_YN2 * YDXC
BB_PLUS_CC = BB + CC

# Compute gradient at origin C
C = D * XC
C = C.at[K_POSITIONS].add(DY[K_POSITIONS] * AA)
C = C.at[K_POSITIONS].add(Y[K_POSITIONS] * BB_PLUS_CC)

# Precompute the sparse quadratic form matrix H_sparse
Y_AT_K = Y[K_POSITIONS]
DY_AT_K = DY[K_POSITIONS]

# Build the sparse Hessian matrix for the 10x10 nonzero block
DY_OUTER = jnp.outer(DY_AT_K, Y_AT_K)
Y_OUTER = jnp.outer(Y_AT_K, Y_AT_K)
H_SPARSE = (DY_OUTER + DY_OUTER.T) * MINUS_2_OVER_YN2 + Y_OUTER * DD

# Add diagonal corrections
DIAG_CORRECTIONS = DY_AT_K * Y_AT_K * MINUS_2_OVER_YN2 + Y_AT_K * Y_AT_K * (DD / 2)
H_SPARSE = H_SPARSE.at[jnp.diag_indices(len(K_POSITIONS))].add(DIAG_CORRECTIONS)


class MOSARQP1(AbstractConstrainedQuadraticProblem):
    """MOSARQP1 problem - a convex quadratic program.

    A convex quadratic problem with variable dimensions. In this problem,
    half the linear constraints are active at the solution.

    Source:
    J.L. Morales-Perez and R.W.H. Sargent,
    "On the implementation and performance of an interior point method for
    large sparse convex quadratic programming",
    Centre for Process Systems Engineering, Imperial College, London,
    November 1991.

    SIF input: Ph. Toint, August 1993.
               minor correction by Ph. Shott, Jan 1995.

    Classification: QLR2-AN-V-V

    TODO: Human review needed
    Issues:
    - Objective value mismatch: JAX=1988.68 vs pycutest=1989.23 (~0.03% difference)
    - Constraint dimension shape errors in test framework
    - Complex quadratic form from SIF may need detailed verification
    - Performance optimized (runs <40s vs timeout) but accuracy needs review
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return N

    @property
    def m(self):
        """Number of constraints."""
        return M

    @property
    def cond(self):
        """Problem conditioning parameter."""
        return COND

    def objective(self, x, args):
        """Compute the quadratic objective function."""
        del args

        # Linear term: c^T * x
        linear_term = jnp.dot(C, x)

        # Diagonal quadratic term: 0.5 * x^T * D * x
        diag_term = 0.5 * jnp.sum(D * x * x)

        # Y-based quadratic terms (sparse - only 10 nonzeros)
        x_at_k = x[K_POSITIONS]

        # Use precomputed sparse quadratic form
        quad_term = jnp.dot(x_at_k, jnp.dot(H_SPARSE, x_at_k))

        return linear_term + diag_term + quad_term

    def constraint(self, x):
        """Compute the linear constraints (discretized 5-point Laplacian)."""
        n = N
        m = M
        rtn = RTN

        # Vectorized 5-point Laplacian computation
        # Only compute for the first M constraints
        constraint_indices = jnp.arange(m)

        # Compute grid positions
        rows = constraint_indices // rtn
        cols = constraint_indices % rtn

        # Center term (always present)
        constraints = 4.0 * x[constraint_indices]

        # Left neighbors: subtract x[i-1] if col > 0 and i > 0
        left_valid = (cols > 0) & (constraint_indices > 0)
        constraints = constraints - jnp.where(
            left_valid, x[constraint_indices - 1], 0.0
        )

        # Right neighbors: subtract x[i+1] if col < rtn-1 and i < n-1
        right_valid = (cols < rtn - 1) & (constraint_indices < n - 1)
        constraints = constraints - jnp.where(
            right_valid, x[constraint_indices + 1], 0.0
        )

        # Top neighbors: subtract x[i-rtn] if row > 0
        top_valid = rows > 0
        constraints = constraints - jnp.where(
            top_valid, x[constraint_indices - rtn], 0.0
        )

        # Bottom neighbors: subtract x[i+rtn] if row < rtn-1 and i+rtn < n
        bottom_valid = (rows < rtn - 1) & (constraint_indices + rtn < n)
        constraints = constraints - jnp.where(
            bottom_valid, x[constraint_indices + rtn], 0.0
        )

        # RHS values depend on boundary conditions
        rhs = jnp.full(m, -0.5)
        rhs = rhs.at[0].set(0.5)
        rhs = jnp.where(constraint_indices == rtn - 1, 0.5, rhs)

        return constraints - rhs, None

    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(M, dtype=bool)

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.full(N, 0.5))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """No variable bounds."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None

    def num_constraints(self):
        """Returns the number of constraints in the problem."""
        return (M, 0, 0)  # M equality constraints, 0 inequality, 0 bounds
