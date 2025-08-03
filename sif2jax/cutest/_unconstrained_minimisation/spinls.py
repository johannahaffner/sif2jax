import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SPINLS(AbstractUnconstrainedMinimisation):
    """SPINLS problem in CUTEst.

    Problem definition:
    Given n particles z_j = x_j + i * y_j in the complex plane,
    determine their positions so that the equations

      z'_j = lambda z_j,

    where z_j = sum_k \\j i / conj( z_j - z_k ) and i = sqrt(-1)
    for some lamda = mu + i * omega

    A problem posed by Nick Trefethen

    Least-squares version of SPIN.SIF, Nick Gould, Jan 2020.

    classification SUR2-AN-V-0

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

    def objective(self, y: jnp.ndarray, args=None) -> jnp.ndarray:
        """Compute the objective function (sum of squares of constraints).

        # TODO: Human review needed - vectorization attempts
        # Attempts made:
        # 1. Full vectorization - failed gradient tests
        # 2. Partial vectorization with jax.lax.scan - better but gradient fails
        # Issues:
        # - Auxiliary variables v_ij create complex gradient dependencies
        # - Constraints use 1/v_ij^2 terms which cause gradient flow issues
        # - Hessian-vector product test: large discrepancy (3937.08 at element 1157)
        # Recommendation:
        # - Consider reformulating without auxiliary variables (like SPIN2LS)
        # - Or investigate numerical stabilization techniques for gradient computation
        """
        n = self.n
        mu = y[0]
        omega = y[1]

        # Extract x and y coordinates
        xy = y[2 : 2 + 2 * n].reshape(n, 2)
        x = xy[:, 0]
        y_coord = xy[:, 1]

        # Extract v_ij variables (for i > j)
        v_start = 2 + 2 * n
        v_flat = y[v_start:]

        # Build v_ij matrix (symmetric) - vectorized
        v_matrix = jnp.zeros((n, n), dtype=y.dtype)
        i_indices, j_indices = jnp.triu_indices(n, k=1)
        v_matrix = v_matrix.at[i_indices, j_indices].set(v_flat)
        v_matrix = v_matrix.at[j_indices, i_indices].set(v_flat)

        # Compute r_j and i_j constraints using scan for the outer loop
        def compute_constraints_for_i(carry, i):
            # Base terms
            r_i = -mu * x[i] + omega * y_coord[i]
            i_i = -mu * y_coord[i] - omega * x[i]

            # Create masks for j < i and j > i
            j_indices = jnp.arange(n)
            mask_lower = j_indices < i
            mask_upper = j_indices > i

            # Compute differences
            x_diff = x[i] - x
            y_diff = y_coord[i] - y_coord
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
            r_contrib_lower = jnp.where(mask_lower, y_diff / v_sq_safe, 0.0)
            i_contrib_lower = jnp.where(mask_lower, -x_diff / v_sq_safe, 0.0)

            # Compute contributions from j > i
            r_contrib_upper = jnp.where(mask_upper, -y_diff_inv / v_sq_ji_safe, 0.0)
            i_contrib_upper = jnp.where(mask_upper, x_diff_inv / v_sq_ji_safe, 0.0)

            r_i += jnp.sum(r_contrib_lower + r_contrib_upper)
            i_i += jnp.sum(i_contrib_lower + i_contrib_upper)

            return carry, (r_i, i_i)

        _, (r_constraints, i_constraints) = jax.lax.scan(
            compute_constraints_for_i, None, jnp.arange(n)
        )

        # Compute m_ij constraints - vectorized
        x_diff = x[i_indices] - x[j_indices]
        y_diff = y_coord[i_indices] - y_coord[j_indices]
        m_constraints = -(v_flat**2) + x_diff**2 + y_diff**2

        # Sum of squares
        return (
            jnp.sum(r_constraints**2)
            + jnp.sum(i_constraints**2)
            + jnp.sum(m_constraints**2)
        )
