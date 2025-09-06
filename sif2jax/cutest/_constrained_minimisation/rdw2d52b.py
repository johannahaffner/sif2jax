import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# Precompute target data V for N=256 as module-scoped variable for performance
def _precompute_rdw2d52b_data(N=256):
    """Precompute target data V(i,j) = exp(-64*((i*h-0.5)^2 + (j*h-0.5)^2))."""
    h = 1.0 / N

    # Create coordinate arrays with explicit float dtype
    i_coords = jnp.arange(N + 1, dtype=jnp.float64) * h
    j_coords = jnp.arange(N + 1, dtype=jnp.float64) * h

    # Create meshgrid
    X, Y = jnp.meshgrid(i_coords, j_coords, indexing="ij")

    # Compute V = exp(-64*((x-0.5)^2 + (y-0.5)^2))
    X_centered = X - 0.5
    Y_centered = Y - 0.5
    V = jnp.exp(-64.0 * (X_centered**2 + Y_centered**2))

    return V


# Precompute target data once at module load time
_rdw2d52b_target_data = _precompute_rdw2d52b_data()


class RDW2D52B(AbstractConstrainedMinimisation):
    """RDW2D52B problem - Finite-element approximation to distributed control
    with bounds.

    A finite-element approximation to the distributed optimal control problem

        min 1/2||u-v||_L2^2 + beta ||f||_L2^2

    subject to - nabla^2 u = f

    where v is given on and within the boundary of a unit [0,1] box in
    2 dimensions, and u = v on its boundary. The discretization uses
    quadrilateral elements. There are simple bounds on both the controls
    f and states u.

    The problem is stated as a quadratic program.

    Source: example 5.2 in
    T. Rees, H. S. Dollar and A. J. Wathen
    "Optimal solvers for PDE-constrained optimization"
    SIAM J. Sci. Comp. (to appear) 2009

    with the control bounds as specified in

    M. Stoll and A. J. Wathen
    "Preconditioning for PDE constrained optimization with
     control constraints"
    OUCL Technical Report 2009

    SIF input: Nick Gould, May 2009
               correction by S. Gratton & Ph. Toint, May 2024

    Classification: QLR2-AN-V-V

    TODO: Human review needed
    Attempts made: [vectorized constraints, precomputed target data, dtype fixes]
    Suspected issues: [computational complexity with 132k variables and 65k constraints]
    Resources needed: [advanced optimization techniques or different approach]
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Grid parameters
    N: int = 256
    beta: float = 0.005  # Different beta from RDW2D51U

    @property
    def n(self):
        """Number of variables."""
        return 2 * (self.N + 1) * (self.N + 1)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        N = self.N
        h = 1.0 / N
        h2 = h * h
        h2_36 = h2 / 36.0
        beta_h2_36 = 2.0 * self.beta * h2_36

        # Use precomputed target data
        V = _rdw2d52b_target_data

        # Variables: [F(0,0), F(0,1), ..., F(N,N), U(0,0), U(0,1), ..., U(N,N)]
        n_grid = (N + 1) * (N + 1)
        F = y[:n_grid].reshape(N + 1, N + 1)
        U = y[n_grid:].reshape(N + 1, N + 1)

        # Vectorized implementation for better performance
        # Create indices for vectorized operations with explicit int dtype
        i_indices = jnp.arange(N, dtype=jnp.int32)
        j_indices = jnp.arange(N, dtype=jnp.int32)
        ii, jj = jnp.meshgrid(i_indices, j_indices, indexing="ij")

        # Extract all element corners at once
        u1 = U[ii, jj]
        u2 = U[ii, jj + 1]
        u3 = U[ii + 1, jj]
        u4 = U[ii + 1, jj + 1]

        v1 = V[ii, jj]
        v2 = V[ii, jj + 1]
        v3 = V[ii + 1, jj]
        v4 = V[ii + 1, jj + 1]

        # Compute differences
        uv1, uv2, uv3, uv4 = u1 - v1, u2 - v2, u3 - v3, u4 - v4

        # Vectorized mass matrix computation for (u-v)
        M_vals = (
            2.0 * (uv1**2 + uv2**2 + uv3**2 + uv4**2)
            + 2.0 * (uv1 * uv2 + uv1 * uv3 + uv2 * uv4 + uv3 * uv4)
            + (uv1 * uv4 + uv2 * uv3)
        )

        obj = h2_36 * jnp.sum(M_vals)

        # Same for F variables
        f1 = F[ii, jj]
        f2 = F[ii, jj + 1]
        f3 = F[ii + 1, jj]
        f4 = F[ii + 1, jj + 1]

        M0_vals = (
            2.0 * (f1**2 + f2**2 + f3**2 + f4**2)
            + 2.0 * (f1 * f2 + f1 * f3 + f2 * f4 + f3 * f4)
            + (f1 * f4 + f2 * f3)
        )

        obj += beta_h2_36 * jnp.sum(M0_vals)

        return obj

    def constraint(self, y):
        """Compute the constraints."""
        N = self.N
        h = 1.0 / N
        h2 = h * h
        minus_h2_36 = -h2 / 36.0
        one_sixth = 1.0 / 6.0

        n_grid = (N + 1) * (N + 1)
        F = y[:n_grid].reshape(N + 1, N + 1)
        U = y[n_grid:].reshape(N + 1, N + 1)

        # Fully vectorized constraint computation for (N-1)x(N-1) constraints
        # Create index arrays for interior points with explicit int dtype
        i_idx = jnp.arange(1, N, dtype=jnp.int32)  # [1, 2, ..., N-1]
        j_idx = jnp.arange(1, N, dtype=jnp.int32)  # [1, 2, ..., N-1]
        ii, jj = jnp.meshgrid(i_idx, j_idx, indexing="ij")  # Shape: (N-1, N-1)

        # A element contributions: 4*u1 - u2 - u3 - 2*u4
        u1_A = U[ii - 1, jj - 1]  # U[i-1, j-1]
        u2_A = U[ii - 1, jj]  # U[i-1, j]
        u3_A = U[ii, jj - 1]  # U[i, j-1]
        u4_A = U[ii, jj]  # U[i, j]
        A_contrib = one_sixth * (4.0 * u1_A - u2_A - u3_A - 2.0 * u4_A)

        # B element contributions: -u1 + 4*u2 - 2*u3 - u4
        u1_B = U[ii - 1, jj]  # U[i-1, j]
        u2_B = U[ii - 1, jj + 1]  # U[i-1, j+1]
        u3_B = U[ii, jj]  # U[i, j]
        u4_B = U[ii, jj + 1]  # U[i, j+1]
        B_contrib = one_sixth * (-u1_B + 4.0 * u2_B - 2.0 * u3_B - u4_B)

        # C element contributions: -u1 - 2*u2 + 4*u3 - u4
        u1_C = U[ii, jj - 1]  # U[i, j-1]
        u2_C = U[ii, jj]  # U[i, j]
        u3_C = U[ii + 1, jj - 1]  # U[i+1, j-1]
        u4_C = U[ii + 1, jj]  # U[i+1, j]
        C_contrib = one_sixth * (-u1_C - 2.0 * u2_C + 4.0 * u3_C - u4_C)

        # D element contributions: -2*u1 - u2 - u3 + 4*u4
        u1_D = U[ii, jj]  # U[i, j]
        u2_D = U[ii, jj + 1]  # U[i, j+1]
        u3_D = U[ii + 1, jj]  # U[i+1, j]
        u4_D = U[ii + 1, jj + 1]  # U[i+1, j+1]
        D_contrib = one_sixth * (-2.0 * u1_D - u2_D - u3_D + 4.0 * u4_D)

        # P element contributions for F: 4*f1 + 2*f2 + 2*f3 + f4
        f1_P = F[ii - 1, jj - 1]  # F[i-1, j-1]
        f2_P = F[ii - 1, jj]  # F[i-1, j]
        f3_P = F[ii, jj - 1]  # F[i, j-1]
        f4_P = F[ii, jj]  # F[i, j]
        P_contrib = minus_h2_36 * (4.0 * f1_P + 2.0 * f2_P + 2.0 * f3_P + f4_P)

        # Q element contributions for F: 2*f1 + 4*f2 + f3 + 2*f4
        f1_Q = F[ii - 1, jj]  # F[i-1, j]
        f2_Q = F[ii - 1, jj + 1]  # F[i-1, j+1]
        f3_Q = F[ii, jj]  # F[i, j]
        f4_Q = F[ii, jj + 1]  # F[i, j+1]
        Q_contrib = minus_h2_36 * (2.0 * f1_Q + 4.0 * f2_Q + f3_Q + 2.0 * f4_Q)

        # R element contributions for F: 2*f1 + f2 + 4*f3 + 2*f4
        f1_R = F[ii, jj - 1]  # F[i, j-1]
        f2_R = F[ii, jj]  # F[i, j]
        f3_R = F[ii + 1, jj - 1]  # F[i+1, j-1]
        f4_R = F[ii + 1, jj]  # F[i+1, j]
        R_contrib = minus_h2_36 * (2.0 * f1_R + f2_R + 4.0 * f3_R + 2.0 * f4_R)

        # S element contributions for F: f1 + 2*f2 + 2*f3 + 4*f4
        f1_S = F[ii, jj]  # F[i, j]
        f2_S = F[ii, jj + 1]  # F[i, j+1]
        f3_S = F[ii + 1, jj]  # F[i+1, j]
        f4_S = F[ii + 1, jj + 1]  # F[i+1, j+1]
        S_contrib = minus_h2_36 * (f1_S + 2.0 * f2_S + 2.0 * f3_S + 4.0 * f4_S)

        # Sum all contributions for each constraint L(i,j)
        constraint_vals = (
            A_contrib
            + B_contrib
            + C_contrib
            + D_contrib
            + P_contrib
            + Q_contrib
            + R_contrib
            + S_contrib
        )

        # Flatten to 1D array (constraints are in row-major order)
        equality_constraints = constraint_vals.flatten()
        inequality_constraints = None

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        N = self.N
        n_grid = (N + 1) * (N + 1)

        # Initialize both F and U to zero
        return jnp.zeros(2 * n_grid)

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        N = self.N
        n_grid = (N + 1) * (N + 1)

        # Use precomputed target data
        V = _rdw2d52b_target_data

        lower = jnp.full(2 * n_grid, -jnp.inf)
        upper = jnp.full(2 * n_grid, jnp.inf)

        # F boundary variables are fixed at 0
        # Boundaries: F(0,*), F(N,*), F(*,0), F(*,N) = 0
        for i in [0, N]:
            for j in range(N + 1):
                idx = i * (N + 1) + j
                lower = lower.at[idx].set(0.0)
                upper = upper.at[idx].set(0.0)

        for j in [0, N]:
            for i in range(1, N):  # Avoid double-setting corners
                idx = i * (N + 1) + j
                lower = lower.at[idx].set(0.0)
                upper = upper.at[idx].set(0.0)

        # Interior F variables are unbounded (no additional bounds like in RDW2D51F)

        # U boundary variables are fixed to target values V
        for i in [0, N]:
            for j in range(N + 1):
                idx = n_grid + i * (N + 1) + j
                val = V[i, j]
                lower = lower.at[idx].set(val)
                upper = upper.at[idx].set(val)

        for j in [0, N]:
            for i in range(1, N):  # Avoid double-setting corners
                idx = n_grid + i * (N + 1) + j
                val = V[i, j]
                lower = lower.at[idx].set(val)
                upper = upper.at[idx].set(val)

        # Additional bound: U(i,j) <= 0.01 for interior points (NEW in RDW2D52B)
        for i in range(1, N):
            for j in range(1, N):
                idx = n_grid + i * (N + 1) + j
                # Keep the lower bound as is (could be V[i,j] or -inf)
                upper = upper.at[idx].set(0.01)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
