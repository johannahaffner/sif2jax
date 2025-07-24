"""Find the polygon of maximal area with fixed diameter.

Find the polygon of maximal area, among polygons with nv sides and
diameter d <= 1.

This is problem 1 in the COPS (Version 2) collection of
E. Dolan and J. More'
see "Benchmarking Optimization Software with COPS"
Argonne National Labs Technical Report ANL/MCS-246 (2000)

SIF input: Nick Gould, December 2000

Classification: OOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLYGON(AbstractConstrainedMinimisation):
    """Find the polygon of maximal area with diameter <= 1."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    NV: int = 100  # Number of vertices

    @property
    def n(self):
        """Number of variables: 2*NV - 2 (excluding fixed R(NV) and THETA(NV))."""
        return 2 * self.NV - 2

    @property
    def y0(self):
        """Initial guess."""
        nv = self.NV
        nv_plus_1 = nv + 1.0
        nv_plus_1_sq = nv_plus_1 * nv_plus_1
        ratr = nv_plus_1_sq / 4.0
        pi = jnp.pi
        ratt = pi / nv

        y0 = jnp.zeros(self.n)

        # Set initial R and THETA values (interleaved: r1, theta1, r2, theta2, ...)
        # excluding fixed R(NV) and THETA(NV)
        for i in range(nv - 1):  # 1 to NV-1 in 1-based
            ratri = nv_plus_1 - (i + 1)
            ratri = ratri * (i + 1) * ratr
            ratti = ratt * (i + 1)

            y0 = y0.at[2 * i].set(ratri)  # R(i+1)
            y0 = y0.at[2 * i + 1].set(ratti)  # THETA(i+1)

        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the objective function (negative polygon area).

        Area = -0.5 * sum_{i=1}^{NV-1} r[i+1]*r[i]*sin(theta[i+1] - theta[i])
        """
        del args  # Not used

        nv = self.NV

        # Extract interleaved variables
        r = jnp.zeros(nv)
        theta = jnp.zeros(nv)

        # Variables are interleaved: r1, theta1, r2, theta2, ...
        for i in range(nv - 1):
            r = r.at[i].set(y[2 * i])
            theta = theta.at[i].set(y[2 * i + 1])

        # Fixed values
        r = r.at[nv - 1].set(0.0)  # R(NV) = 0.0
        theta = theta.at[nv - 1].set(jnp.pi)  # THETA(NV) = PI

        area = 0.0

        # Sum over i from 1 to NV-1 (0 to NV-2 in 0-based)
        for i in range(nv - 1):
            r1 = r[i]
            r2 = r[i + 1]
            t1 = theta[i + 1]
            t2 = theta[i]

            # SI element: r1 * r2 * sin(t1 - t2)
            area += -0.5 * r1 * r2 * jnp.sin(t1 - t2)

        return jnp.array(area)

    def constraint(self, y):
        """Compute the constraints.

        Inequality constraints:
        - Order constraints: theta[i+1] >= theta[i] for i = 1 to NV-1
        - Distance constraints:
          r[i]^2 + r[j]^2 - 2*r[i]*r[j]*cos(theta[j] - theta[i]) <= 1
          for all i < j
        """
        nv = self.NV

        # Extract interleaved variables
        r = jnp.zeros(nv)
        theta = jnp.zeros(nv)

        # Variables are interleaved: r1, theta1, r2, theta2, ...
        for i in range(nv - 1):
            r = r.at[i].set(y[2 * i])
            theta = theta.at[i].set(y[2 * i + 1])

        # Fixed values
        r = r.at[nv - 1].set(0.0)  # R(NV) = 0.0
        theta = theta.at[nv - 1].set(jnp.pi)  # THETA(NV) = PI

        # Order constraints (NV-1 constraints)
        order_constraints = []
        for i in range(nv - 1):
            # theta[i+1] - theta[i] >= 0
            order_constraints.append(theta[i + 1] - theta[i])

        # Distance constraints (NV*(NV-1)/2 constraints)
        distance_constraints = []
        for i in range(nv - 1):
            for j in range(i + 1, nv):
                # r[i]^2 + r[j]^2 - 2*r[i]*r[j]*cos(theta[j] - theta[i]) <= 1
                dist_sq = (
                    r[i] ** 2
                    + r[j] ** 2
                    - 2.0 * r[i] * r[j] * jnp.cos(theta[j] - theta[i])
                )
                distance_constraints.append(1.0 - dist_sq)

        # No equality constraints
        equalities = None

        # All constraints are inequalities (>= 0)
        inequalities = jnp.concatenate(
            [jnp.array(order_constraints), jnp.array(distance_constraints)]
        )

        return equalities, inequalities

    @property
    def bounds(self):
        """Bounds on variables."""
        nv = self.NV
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)

        # Variables are interleaved: r1, theta1, r2, theta2, ...
        # R(i) in [0, 1] and THETA(i) in [0, PI] for i = 1 to NV-1
        for i in range(nv - 1):
            upper = upper.at[2 * i].set(1.0)  # R(i+1)
            upper = upper.at[2 * i + 1].set(jnp.pi)  # THETA(i+1)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value for NV=100."""
        return jnp.array(-0.77847)
