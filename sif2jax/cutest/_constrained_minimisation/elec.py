import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class ELEC(AbstractConstrainedMinimisation):
    """Minimize Coulomb potential of electrons positioned on a conducting sphere.

    Given np electrons, find the equilibrium state distribution of minimal
    Coulomb potential of the electrons positioned on a conducting sphere.

    This is problem 2 in the COPS (Version 2) collection of
    E. Dolan and J. More'
    see "Benchmarking Optimization Software with COPS"
    Argonne National Labs Technical Report ANL/MCS-246 (2000)

    SIF input: Nick Gould, November 2000

    Classification: OOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters (from SIF file)
    NP: int = 200  # Number of electrons

    @property
    def n(self):
        """Number of variables (3 coordinates per electron)."""
        return 3 * self.NP

    @property
    def m(self):
        """Number of constraints (one per electron)."""
        return self.NP

    def objective(self, y: Array, args) -> Array:
        """Compute the Coulomb potential energy."""
        del args
        # Reshape y into NP electrons with 3 coordinates each
        points = y.reshape(self.NP, 3)

        # Compute pairwise distances and potential energy
        # PE = sum_{i<j} 1 / ||p_i - p_j||

        # Vectorized computation using broadcasting
        # Expand points to shape (NP, 1, 3) and (1, NP, 3)
        points_i = points[:, jnp.newaxis, :]  # Shape: (NP, 1, 3)
        points_j = points[jnp.newaxis, :, :]  # Shape: (1, NP, 3)

        # Compute all pairwise differences
        diffs = points_i - points_j  # Shape: (NP, NP, 3)

        # Compute all pairwise distances with small epsilon for numerical stability
        distances_squared = jnp.sum(diffs**2, axis=2)  # Shape: (NP, NP)
        # Add small epsilon to prevent gradient explosion at small distances
        epsilon = 1e-10
        distances = jnp.sqrt(distances_squared + epsilon)  # Shape: (NP, NP)

        # Create mask for upper triangle (i < j)
        i_indices = jnp.arange(self.NP)[:, jnp.newaxis]
        j_indices = jnp.arange(self.NP)[jnp.newaxis, :]
        upper_triangle_mask = i_indices < j_indices

        # Compute potential energy only for i < j pairs
        # Use where to avoid division by zero on diagonal
        safe_distances = jnp.where(upper_triangle_mask, distances, 1.0)
        potentials = jnp.where(upper_triangle_mask, 1.0 / safe_distances, 0.0)

        return jnp.sum(potentials)

    def constraint(self, y: Array):
        """Constraints: electrons lie on the unit sphere."""
        # Reshape y into NP electrons with 3 coordinates each
        points = y.reshape(self.NP, 3)

        # Each electron must lie on unit sphere: x^2 + y^2 + z^2 = 1
        squared_norms = jnp.sum(points**2, axis=1)
        equality_constraints = squared_norms - 1.0

        return equality_constraints, None

    @property
    def y0(self):
        """Initial guess: quasi-uniform distribution on unit sphere."""
        # From SIF file: quasi-uniform distribution using specific formula
        n = self.NP
        pi = jnp.pi

        # Generate initial points
        points = []
        for i in range(1, n + 1):
            u = i / n
            theta_i = 2 * pi * u
            phi_i = pi * (u - 1 / n)

            # Convert spherical to Cartesian coordinates
            cos_theta = jnp.cos(theta_i)
            sin_theta = jnp.sin(theta_i)
            sin_phi = jnp.sin(phi_i)
            cos_phi = jnp.cos(phi_i)

            x = cos_theta * sin_phi
            y = sin_theta * sin_phi
            z = cos_phi

            # Normalize to ensure exactly on unit sphere
            norm = jnp.sqrt(x**2 + y**2 + z**2)
            x = x / norm
            y = y / norm
            z = z / norm

            points.extend([x, y, z])

        return inexact_asarray(jnp.array(points))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comment for NP=200
        return jnp.array(1.84389e4)
