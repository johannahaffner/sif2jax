"""
SPIN2LS problem in CUTEst.

Problem definition:
Given n particles z_j = x_j + i * y_j in the complex plane,
determine their positions so that the equations

  z'_j = lambda z_j,

where z_j = sum_k \\j i / conj( z_j - z_k ) and i = sqrt(-1)
for some lamda = mu + i * omega

A problem posed by Nick Trefethen; this is a condensed version of SPIN

Least-squares version of SPIN2.SIF, Nick Gould, Jan 2020.

classification SUR2-AN-V-0

SIF input: Nick Gould, June 2009
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SPIN2LS(AbstractUnconstrainedMinimisation):
    """SPIN2LS problem - least-squares condensed version of SPIN."""

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

    def objective(self, y: jnp.ndarray, args=None) -> jnp.ndarray:
        """Compute the objective function (sum of squares of constraints)."""
        n = self.n
        mu = y[0]
        omega = y[1]

        # Extract x and y coordinates
        xy = y[2 : 2 + 2 * n].reshape(n, 2)
        x = xy[:, 0]
        y_coord = xy[:, 1]

        # Compute r_j and i_j constraints
        # r_j = - mu * x_j + omega * y_j + sum_k\j (y_j - y_k ) / dist_sq = 0
        # i_j = - mu * y_j - omega * x_j - sum_k\j (x_j - x_k ) / dist_sq = 0

        r_constraints = jnp.zeros(n, dtype=y.dtype)
        i_constraints = jnp.zeros(n, dtype=y.dtype)

        for j in range(n):
            r_j = -mu * x[j] + omega * y_coord[j]
            i_j = -mu * y_coord[j] - omega * x[j]

            # Sum over k != j
            for k in range(n):
                if k != j:
                    dx = x[j] - x[k]
                    dy = y_coord[j] - y_coord[k]
                    dist_sq = dx**2 + dy**2

                    r_j += dy / dist_sq
                    i_j -= dx / dist_sq

            r_constraints = r_constraints.at[j].set(r_j)
            i_constraints = i_constraints.at[j].set(i_j)

        # Sum of squares
        return jnp.sum(r_constraints**2) + jnp.sum(i_constraints**2)
