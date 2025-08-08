import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class CYCLOOCTLS(AbstractBoundedMinimisation):
    """
    The cyclooctane molecule configuration problem (least squares version).

    The cyclooctane molecule is comprised of eight carbon atoms aligned
    in an equally spaced ring. When they take a position of minimum
    potential energy so that next-neighbours are equally spaced.

    Given positions v_1, ..., v_p in R^3 (with p = 8 for cyclooctane),
    and given a spacing c^2 we have that

       ||v_i - v_i+1,mod p||^2 = c^2 for i = 1,..,p, and
       ||v_i - v_i+2,mod p||^2 = 2p/(p-2) c^2

    where (arbitrarily) we have v_1 = 0 and component 1 of v_2 = 0

    Source:
    an extension of the cyclooctane molecule configuration space as
    described in (for example)

     E. Coutsias, S. Martin, A. Thompson & J. Watson
     "Topology of cyclooctane energy landscape"
     J. Chem. Phys. 132-234115 (2010)

    SIF input: Nick Gould, Feb 2020.

    classification  SBR2-MN-V-0
    """

    p: int = 10000  # Number of molecules
    c: float = 1.0  # Radius
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Number of variables is 3*P"""
        return 3 * self.p

    def objective(self, y: Array, args) -> Array:
        """Compute the least-squares objective for cyclooctane configuration"""
        p = self.p
        c2 = self.c**2

        # Calculate 2P/(P-2) * c^2
        sc2 = (2.0 * p / (p - 2)) * c2

        # Reshape into (p, 3) for easier manipulation - each row is a position vector
        positions = y.reshape(p, 3)

        # Residuals for nearest neighbor constraints: ||v_i - v_{i+1}||^2 = c^2
        # Use modular arithmetic for cyclic structure
        next_indices = jnp.arange(p)
        next_indices = (next_indices + 1) % p

        diff_next = positions - positions[next_indices]
        dist_sq_next = jnp.sum(diff_next**2, axis=1)
        residuals_next = dist_sq_next - c2

        # Residuals for next-next neighbor constraints: ||v_i - v_{i+2}||^2 =
        # 2p/(p-2)*c^2
        next2_indices = (jnp.arange(p) + 2) % p

        diff_next2 = positions - positions[next2_indices]
        dist_sq_next2 = jnp.sum(diff_next2**2, axis=1)
        residuals_next2 = dist_sq_next2 - sc2

        # Combine all residuals
        all_residuals = jnp.concatenate([residuals_next, residuals_next2])

        # Return sum of squares
        return 0.5 * jnp.sum(all_residuals**2)

    @property
    def bounds(self):
        """Bounds with first molecule fixed at origin and x-component of second fixed"""
        p = self.p
        lbs = jnp.full(3 * p, -jnp.inf)
        ubs = jnp.full(3 * p, jnp.inf)

        # Fix first molecule at origin (indices 0, 1, 2 for x1, y1, z1)
        lbs = lbs.at[0:3].set(0.0)
        ubs = ubs.at[0:3].set(0.0)

        # Fix x-component of second molecule (index 3 for x2)
        lbs = lbs.at[3].set(0.0)
        ubs = ubs.at[3].set(0.0)

        return lbs, ubs

    @property
    def y0(self) -> Array:
        """Initial point: distributed on unit circle in x-y plane"""
        p = self.p
        angles = jnp.linspace(0, 2 * jnp.pi, p, endpoint=False)

        x_coords = self.c * jnp.cos(angles)
        y_coords = self.c * jnp.sin(angles)
        z_coords = jnp.zeros(p)

        # Stack into single array
        initial = jnp.stack([x_coords, y_coords, z_coords], axis=1).flatten()

        return inexact_asarray(initial)

    @property
    def args(self):
        return None

    @property
    def expected_result(self) -> None:
        """Solution represents minimum energy configuration"""
        return None

    @property
    def expected_objective_value(self) -> Array:
        """Expected optimal objective value: 0"""
        return inexact_asarray(0.0)
