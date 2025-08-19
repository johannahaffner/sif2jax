"""RAYBENDS - Ray bending with beta-spline curve.

A ray bending problem. A ray across an inhomogeneous 2D medium is
represented by a beta-spline whose knots can be chosen. The problem
is then to optimize the positions of these knots in order to obtain a
ray path corresponding to the minimum travel time from source to receiver,
according to Fermat principle.

In this version, 10 points are used in every interval of the curve
defining the ray in order to compute more accurate travel times.

Source: T.J. Moser, G. Nolet and R. Snieder,
  "Ray bending revisited",
  Bulletin of the Seism. Society of America 21(1).

SIF input: Ph Toint, Dec 1991.

Classification: OXR2-MY-V-0
"""

import jax
import jax.numpy as jnp
from equinox import Module

from ..._problem import AbstractUnconstrainedMinimisation


class RAYBENDS(AbstractUnconstrainedMinimisation, Module):
    """RAYBENDS - Ray bending with beta-spline curve."""

    _nknots: int = 1024  # Number of knots (NK in SIF)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        # 2 coordinates (x, z) for each knot (0 to NK)
        return 2 * (self._nknots + 1)

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
        # All variables are free (unconstrained)
        return None

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
        # From OBJECT BOUND in SIF: analytical solution to continuous problem
        return jnp.array(96.2424)

    def _beta_spline_basis(self, t):
        """Compute beta-spline basis functions Q1, Q2, Q3, Q4."""
        # Beta-spline basis functions from SIF
        q1 = 1.0 / 6.0 - 0.5 * t + 0.5 * t**2 - t**3 / 6.0
        q2 = 2.0 / 3.0 - t**2 + 0.5 * t**3
        q3 = 1.0 / 6.0 + 0.5 * t + 0.5 * t**2 - 0.5 * t**3
        q4 = t**3 / 6.0
        return q1, q2, q3, q4

    def _compute_segment_time_vectorized(self, x1, x2, x3, x4, z1, z2, z3, z4):
        """Compute travel time for one spline segment using 10 sample points."""
        m = 10  # Number of sample points per segment

        # Sample points within the segment [0, 1]
        t_vals = jnp.linspace(0, 1, m + 1)

        # Vectorized computation of basis functions
        t_k = t_vals[:-1]
        t_kp1 = t_vals[1:]

        # Beta-spline basis at t_k and t_{k+1}
        q1_k, q2_k, q3_k, q4_k = self._beta_spline_basis(t_k)
        q1_kp1, q2_kp1, q3_kp1, q4_kp1 = self._beta_spline_basis(t_kp1)

        # Z-coordinates at sample points
        z_k = z1 * q1_k + z2 * q2_k + z3 * q3_k + z4 * q4_k
        z_kp1 = z1 * q1_kp1 + z2 * q2_kp1 + z3 * q3_kp1 + z4 * q4_kp1

        # Velocities
        c_k = 1.0 + 0.01 * z_k
        c_kp1 = 1.0 + 0.01 * z_kp1

        # Position differences (using relative coordinates)
        dx = (
            (x2 - x1) * (q2_kp1 - q2_k)
            + (x3 - x1) * (q3_kp1 - q3_k)
            + (x4 - x1) * (q4_kp1 - q4_k)
        )
        dz = z_kp1 - z_k

        # Distance between sample points
        distances = jnp.sqrt(dx * dx + dz * dz)

        # Average slowness
        slowness = 0.5 * (1.0 / c_k + 1.0 / c_kp1)

        # Sum segment times
        return jnp.sum(slowness * distances)

    def objective(self, y, args=None):
        """Compute the travel time along the spline-defined ray path."""
        nknots = self._nknots

        # Extract x and z coordinates (interleaved)
        x = y[0::2]  # Even indices
        z = y[1::2]  # Odd indices

        # Build arrays of control points for all segments
        # Special case for segment 0 (uses x[0] three times)
        time_0 = self._compute_segment_time_vectorized(
            x[0], x[0], x[0], x[1], z[0], z[0], z[0], z[1]
        )

        # Special case for segment 1
        time_1 = self._compute_segment_time_vectorized(
            x[0], x[0], x[1], x[2], z[0], z[0], z[1], z[2]
        )

        # Regular segments (2 to NK-1) - use scan for efficiency
        def compute_segment(carry, i):
            x, z = carry
            time_i = self._compute_segment_time_vectorized(
                x[i - 2], x[i - 1], x[i], x[i + 1], z[i - 2], z[i - 1], z[i], z[i + 1]
            )
            return carry, time_i

        _, segment_times = jax.lax.scan(compute_segment, (x, z), jnp.arange(2, nknots))

        # Special case for segment NK
        time_nk = self._compute_segment_time_vectorized(
            x[nknots - 2],
            x[nknots - 1],
            x[nknots],
            x[nknots],
            z[nknots - 2],
            z[nknots - 1],
            z[nknots],
            z[nknots],
        )

        # Special case for segment NK+1
        time_nkp1 = self._compute_segment_time_vectorized(
            x[nknots - 1],
            x[nknots],
            x[nknots],
            x[nknots],
            z[nknots - 1],
            z[nknots],
            z[nknots],
            z[nknots],
        )

        # Sum all segment times
        total_time = time_0 + time_1 + jnp.sum(segment_times) + time_nk + time_nkp1

        # Return total time
        return total_time
