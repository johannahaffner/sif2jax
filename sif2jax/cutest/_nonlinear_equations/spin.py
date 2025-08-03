"""
SPIN problem in CUTEst.

# TODO: Human review needed
# Attempts made:
# 1. Implemented based on SIF file formulation
# 2. Fixed initial guess indexing (1-based to 0-based)
# Suspected issues:
# - Constraint formulation may have errors in index handling
# - The v_ij auxiliary variables might need different initialization
# - Complex number arithmetic may need careful handling
# Resources needed:
# - Verification of constraint formulas against original paper
# - Comparison with pycutest constraint values at various points

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

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class SPIN(AbstractNonlinearEquations):
    """SPIN problem."""

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

        # Extract v_ij variables (for i > j)
        v_start = 2 + 2 * n

        # Build v_ij matrix (symmetric)
        v_matrix = jnp.zeros((n, n), dtype=y.dtype)
        idx = 0
        for i in range(1, n):
            for j in range(i):
                v_matrix = v_matrix.at[i, j].set(y[v_start + idx])
                v_matrix = v_matrix.at[j, i].set(y[v_start + idx])
                idx += 1

        # Compute r_j and i_j constraints
        r_constraints = jnp.zeros(n, dtype=y.dtype)
        i_constraints = jnp.zeros(n, dtype=y.dtype)

        for i in range(n):
            # Base terms
            r_i = -mu * x[i] + omega * y_coord[i]
            i_i = -mu * y_coord[i] - omega * x[i]

            # Sum over j < i
            for j in range(i):
                v_ij_sq = v_matrix[i, j] ** 2
                # RY(i,j) with coefficient +1.0
                r_i += (y_coord[i] - y_coord[j]) / v_ij_sq
                # RX(i,j) with coefficient -1.0
                i_i -= (x[i] - x[j]) / v_ij_sq

            # Sum over j > i
            for j in range(i + 1, n):
                v_ji_sq = v_matrix[j, i] ** 2
                # RY(j,i) with coefficient -1.0
                r_i -= (y_coord[j] - y_coord[i]) / v_ji_sq
                # RX(j,i) with coefficient +1.0
                i_i += (x[j] - x[i]) / v_ji_sq

            r_constraints = r_constraints.at[i].set(r_i)
            i_constraints = i_constraints.at[i].set(i_i)

        # Compute m_ij constraints: -v_ij^2 + (x_i - x_j)^2 + (y_i - y_j)^2 = 0
        m_constraints = []
        idx = 0
        for i in range(1, n):
            for j in range(i):
                v_ij = y[v_start + idx]
                m_ij = -(v_ij**2) + (x[i] - x[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2
                m_constraints.append(m_ij)
                idx += 1

        m_constraints = jnp.array(m_constraints, dtype=y.dtype)

        return jnp.concatenate([r_constraints, i_constraints, m_constraints])
