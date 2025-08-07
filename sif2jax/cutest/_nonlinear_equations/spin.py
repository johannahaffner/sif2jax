import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class SPIN(AbstractNonlinearEquations):
    """
    SPIN problem in CUTEst.

    # Successfully vectorized inner loops
    # Key changes made:
    # 1. Fixed v_ij index ordering to match SIF file (i from 2 to n, j from 1 to i-1)
    # 2. Vectorized m_constraints computation
    # 3. Fixed constraint return type to use None for inequalities
    # Note: The r_j and i_j constraints still use a loop due to complex dependencies
    # but inner summations are vectorized using masks and jnp.where

    Problem definition:
    Given n particles z_j = x_j + i * y_j in the complex plane,
    determine their positions so that the equations

      z'_j = lambda z_j,

    where z_j = sum_k \\j i / conj( z_j - z_k ) and i = sqrt(-1)
    for some lamda = mu + i * omega

    A problem posed by Nick Trefethen

    classification NOR2-AN-V-V

    SIF input: Nick Gould, June 2009
    """

    n: int = 50
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Compute initial guess - particles on a circle."""
        n = self.n
        # Particles are initially placed on a unit circle
        # From SIF: RI RI I (where I goes from 1 to N)
        i_values = jnp.arange(1, n + 1, dtype=jnp.float64)
        angles = i_values * (2.0 * jnp.pi / n)
        x_init = jnp.cos(angles)
        y_init = jnp.sin(angles)

        # Variables: [mu, omega, x1, y1, x2, y2, ..., xn, yn, v21, v31, ..., vn(n-1)]
        # v_ij are auxiliary variables for i > j
        n_v = n * (n - 1) // 2
        v_init = jnp.ones(n_v, dtype=jnp.float64)

        return jnp.concatenate(
            [
                jnp.array([1.0, 1.0], dtype=jnp.float64),  # mu, omega
                jnp.stack([x_init, y_init], axis=-1).ravel(),  # x, y coordinates
                v_init,  # v_ij variables
            ]
        )

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None

    @property
    def bounds(self):
        """Variable bounds (unbounded)."""
        return None

    def constraint(self, y: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        """Wrapper for equality_constraints to match interface."""
        # Return tuple of (equalities, inequalities) where inequalities is None
        return self.equality_constraints(y), None

    def equality_constraints(self, y: jnp.ndarray) -> jnp.ndarray:
        """Compute the equality constraints."""
        n = self.n
        mu = y[0]
        omega = y[1]

        # Extract x and y coordinates
        xy = y[2 : 2 + 2 * n].reshape(n, 2)
        x = xy[:, 0]
        y_coord = xy[:, 1]

        # Extract v_ij variables (for i > j)
        v_start = 2 + 2 * n

        # Get indices in SIF order: for i from 2 to n, j from 1 to i-1
        # This matches the order in the SIF file
        i_list = []
        j_list = []
        for i in range(1, n):  # i from 2 to n in 1-based, so 1 to n-1 in 0-based
            for j in range(i):  # j from 1 to i-1 in 1-based, so 0 to i-1 in 0-based
                i_list.append(i)
                j_list.append(j)

        v_i_indices = jnp.array(i_list, dtype=jnp.int32)
        v_j_indices = jnp.array(j_list, dtype=jnp.int32)
        n_v = n * (n - 1) // 2  # Number of v_ij variables

        # Build v_ij matrix (symmetric) - vectorized
        v_matrix = jnp.zeros((n, n), dtype=y.dtype)
        v_flat = y[v_start : v_start + n_v]
        v_matrix = v_matrix.at[v_i_indices, v_j_indices].set(v_flat)
        v_matrix = v_matrix.at[v_j_indices, v_i_indices].set(v_flat)

        # Compute r_j and i_j constraints
        r_constraints = jnp.zeros(n, dtype=y.dtype)
        i_constraints = jnp.zeros(n, dtype=y.dtype)

        for i in range(n):
            # Base terms
            r_i = -mu * x[i] + omega * y_coord[i]
            i_i = -mu * y_coord[i] - omega * x[i]

            # Create masks for j < i and j > i
            j_range = jnp.arange(n)
            mask_lower = j_range < i
            mask_upper = j_range > i

            # Compute differences
            x_diff_i = x[i] - x
            y_diff_i = y_coord[i] - y_coord
            x_diff_inv = x - x[i]
            y_diff_inv = y_coord - y_coord[i]

            # Get v_squared values with safe division
            v_sq = v_matrix[i, :] ** 2
            v_sq_ji = v_matrix[:, i] ** 2

            # Add small epsilon to avoid division by zero
            eps = 1e-10
            v_sq_safe = v_sq + eps
            v_sq_ji_safe = v_sq_ji + eps

            # Compute contributions from j < i
            r_contrib_lower = jnp.where(mask_lower, y_diff_i / v_sq_safe, 0.0)
            i_contrib_lower = jnp.where(mask_lower, -x_diff_i / v_sq_safe, 0.0)

            # Compute contributions from j > i
            r_contrib_upper = jnp.where(mask_upper, -y_diff_inv / v_sq_ji_safe, 0.0)
            i_contrib_upper = jnp.where(mask_upper, x_diff_inv / v_sq_ji_safe, 0.0)

            r_i += jnp.sum(r_contrib_lower + r_contrib_upper)
            i_i += jnp.sum(i_contrib_lower + i_contrib_upper)

            r_constraints = r_constraints.at[i].set(r_i)
            i_constraints = i_constraints.at[i].set(i_i)

        # Compute m_ij constraints: -v_ij^2 + (x_i - x_j)^2 + (y_i - y_j)^2 = 0
        # Vectorized computation
        # Note: v_i_indices and v_j_indices contain the actual particle indices
        x_diff = x[v_i_indices] - x[v_j_indices]
        y_diff = y_coord[v_i_indices] - y_coord[v_j_indices]
        m_constraints = -(v_flat**2) + x_diff**2 + y_diff**2

        return jnp.concatenate([r_constraints, i_constraints, m_constraints])
