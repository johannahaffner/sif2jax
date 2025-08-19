"""RAYBENDL - Ray bending with piecewise linear curve.

A ray bending problem. A ray across an inhomogeneous 2D medium is
represented by a piecewise linear curve whose knots can be chosen.
The problem is then to optimize the positions of these knots in order
to obtain a ray path corresponding to the minimum travel time from
source to receiver, according to Fermat principle.

The problem becomes harder and harder when the dimension increases
because the knots are getting closer and closer and the objective
has a nondifferentiable kink when two knots coincide.

Source: T.J. Moser, G. Nolet and R. Snieder,
  "Ray bending revisited",
  Bulletin of the Seism. Society of America 21(1).

SIF input: Ph Toint, Dec 1991.

Classification: OXR2-MY-V-0
"""

import jax.numpy as jnp
from equinox import Module

from ..._problem import AbstractBoundedMinimisation


class RAYBENDL(AbstractBoundedMinimisation, Module):
    """RAYBENDL - Ray bending with piecewise linear curve."""

    _nknots: int = 1024  # Number of knots (including source and receiver)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        # 2 coordinates (x, z) for each knot
        return 2 * self._nknots + 2

    @property
    def y0(self):
        """Initial guess - equidistant points on straight line."""
        nknots = self._nknots

        # Source and receiver positions
        xsrc, zsrc = 0.0, 0.0
        xrcv, zrcv = 100.0, 100.0

        # Vectorized computation of equidistant points
        fracs = jnp.linspace(0, 1, nknots + 1)
        x_vals = xsrc + fracs * (xrcv - xsrc)
        z_vals = zsrc + fracs * (zrcv - zsrc)

        # Interleave x and z coordinates using stack and flatten
        return jnp.stack([x_vals, z_vals], axis=1).flatten()

    @property
    def bounds(self):
        """Bounds on variables."""
        nknots = self._nknots

        # Initialize with unbounded arrays
        lower = jnp.full(2 * nknots + 2, -jnp.inf)
        upper = jnp.full(2 * nknots + 2, jnp.inf)

        # Fix source position (x[0], z[0])
        lower = lower.at[0].set(0.0)  # xsrc
        lower = lower.at[1].set(0.0)  # zsrc
        upper = upper.at[0].set(0.0)  # xsrc
        upper = upper.at[1].set(0.0)  # zsrc

        # Fix receiver position (x[nknots], z[nknots])
        lower = lower.at[2 * nknots].set(100.0)  # xrcv
        lower = lower.at[2 * nknots + 1].set(100.0)  # zrcv
        upper = upper.at[2 * nknots].set(100.0)  # xrcv
        upper = upper.at[2 * nknots + 1].set(100.0)  # zrcv

        return lower, upper

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF."""
        # From OBJECT BOUND in SIF: continuous problem solution
        return jnp.array(96.2424)

    def objective(self, y, args=None):
        """Compute the travel time along the ray path."""
        # Extract x and z coordinates (interleaved)
        x = y[0::2]  # Even indices
        z = y[1::2]  # Odd indices

        # Speed parameter
        cz = 0.01

        # Vectorized computation for all segments
        x1 = x[:-1]
        x2 = x[1:]
        z1 = z[:-1]
        z2 = z[1:]

        # Velocities at each knot
        c1 = 1.0 + cz * z1
        c2 = 1.0 + cz * z2

        # Distance for each segment
        dx = x2 - x1
        dz = z2 - z1
        distances = jnp.sqrt(dx * dx + dz * dz)

        # Average slowness for each segment
        slowness = 0.5 * (1.0 / c1 + 1.0 / c2)

        # Travel time for each segment
        segment_times = slowness * distances

        # Total travel time
        return jnp.sum(segment_times)
