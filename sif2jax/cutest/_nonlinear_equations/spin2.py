import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class SPIN2(AbstractNonlinearEquations):
    """
    SPIN2 problem in CUTEst.

    Problem definition:
    Given n particles z_j = x_j + i * y_j in the complex plane,
    determine their positions so that the equations

      z'_j = lambda z_j,

    where z_j = sum_k \\j i / conj( z_j - z_k ) and i = sqrt(-1)
    for some lamda = mu + i * omega

    A problem posed by Nick Trefethen; this is a condensed version of SPIN

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

        # Variables are [mu, omega, x1, y1, x2, y2, ..., xn, yn]
        # No auxiliary v variables in the condensed version
        return jnp.concatenate(
            [
                jnp.array([1.0, 1.0], dtype=jnp.float64),  # mu, omega
                jnp.stack([x_init, y_init], axis=-1).ravel(),  # x, y coordinates
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

    def constraint(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Wrapper for equality_constraints to match interface."""
        # Return tuple of (equalities, inequalities) where inequalities is empty
        return self.equality_constraints(y), jnp.array([], dtype=y.dtype)

    def equality_constraints(self, y: jnp.ndarray) -> jnp.ndarray:
        """Compute the equality constraints."""
        n = self.n
        mu = y[0]
        omega = y[1]

        # Extract x and y coordinates
        xy = y[2 : 2 + 2 * n].reshape(n, 2)
        x = xy[:, 0]
        y_coord = xy[:, 1]

        # Compute r_j and i_j constraints vectorized
        # r_j = - mu * x_j + omega * y_j + sum_k\j (y_j - y_k ) / dist_sq = 0
        # i_j = - mu * y_j - omega * x_j - sum_k\j (x_j - x_k ) / dist_sq = 0

        # Compute pairwise differences
        x_diff = x[:, None] - x[None, :]  # x[i] - x[j], shape (n, n)
        y_diff = y_coord[:, None] - y_coord[None, :]  # y[i] - y[j], shape (n, n)

        # Compute distances squared
        dist_sq = x_diff**2 + y_diff**2

        # Avoid division by zero on diagonal
        dist_sq_safe = jnp.where(jnp.eye(n, dtype=bool), 1.0, dist_sq)

        # Mask for off-diagonal elements
        mask = ~jnp.eye(n, dtype=bool)

        # Compute constraints using vectorized operations
        # r_i = -mu * x[i] + omega * y[i] + sum_{j!=i} (y[i] - y[j]) / dist_sq[i,j]
        r_terms = jnp.where(mask, y_diff / dist_sq_safe, 0.0)
        r_constraints = -mu * x + omega * y_coord + jnp.sum(r_terms, axis=1)

        # i_i = -mu * y[i] - omega * x[i] - sum_{j!=i} (x[i] - x[j]) / dist_sq[i,j]
        i_terms = jnp.where(mask, -x_diff / dist_sq_safe, 0.0)
        i_constraints = -mu * y_coord - omega * x + jnp.sum(i_terms, axis=1)

        return jnp.concatenate([r_constraints, i_constraints])
