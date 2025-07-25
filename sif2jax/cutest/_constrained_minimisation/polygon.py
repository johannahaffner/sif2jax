import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLYGON(AbstractConstrainedMinimisation):
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
        for i in range(
            nv - 1
        ):  # 0 to NV-2 in 0-based, corresponds to 1 to NV-1 in 1-based
            i_1based = i + 1
            ratri = nv_plus_1 - i_1based  # RATRI = NV+1 - I
            ratri = ratri * i_1based  # RATRI = RATRI * I
            ratri = ratri / ratr  # RATRI = RATRI / RATR (division, not multiplication)
            ratti = ratt * i_1based  # RATTI = RATT * I

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

        # Extract interleaved variables - vectorized
        indices = jnp.arange(nv - 1)
        r_vars = y[2 * indices]  # R values from variables
        theta_vars = y[2 * indices + 1]  # THETA values from variables

        # Create full r and theta arrays with fixed values
        r = jnp.concatenate([r_vars, jnp.array([0.0])])  # R(NV) = 0.0
        theta = jnp.concatenate([theta_vars, jnp.array([jnp.pi])])  # THETA(NV) = PI

        # Vectorized computation over i from 0 to NV-2
        r1 = r[:-1]  # r[i] for i = 0 to NV-2
        r2 = r[1:]  # r[i+1] for i = 0 to NV-2
        t1 = theta[1:]  # theta[i+1] for i = 0 to NV-2
        t2 = theta[:-1]  # theta[i] for i = 0 to NV-2

        # SI element: r1 * r2 * sin(t1 - t2)
        area_terms = -0.5 * r1 * r2 * jnp.sin(t1 - t2)
        area = jnp.sum(area_terms)

        return area

    def constraint(self, y):
        """Compute the constraints.

        Inequality constraints:
        - Order constraints: theta[i+1] >= theta[i] for i = 1 to NV-1
        - Distance constraints:
          r[i]^2 + r[j]^2 - 2*r[i]*r[j]*cos(theta[j] - theta[i]) <= 1
          for all i < j
        """
        nv = self.NV

        # Extract interleaved variables - vectorized
        indices = jnp.arange(nv - 1)
        r_vars = y[2 * indices]  # R values from variables
        theta_vars = y[2 * indices + 1]  # THETA values from variables

        # Create full r and theta arrays with fixed values
        r = jnp.concatenate([r_vars, jnp.array([0.0])])  # R(NV) = 0.0
        theta = jnp.concatenate([theta_vars, jnp.array([jnp.pi])])  # THETA(NV) = PI

        # Order constraints (NV-1 constraints) - vectorized
        order_constraints = theta[1:] - theta[:-1]  # theta[i+1] - theta[i] >= 0

        # Distance constraints - compute all pairs (i, j) where i < j
        # Use jnp.triu_indices to get upper triangular indices
        i_vals, j_vals = jnp.triu_indices(nv, k=1)

        # Compute distance constraints for all valid pairs
        r_i = r[i_vals]
        r_j = r[j_vals]
        theta_i = theta[i_vals]
        theta_j = theta[j_vals]

        dist_sq = r_i**2 + r_j**2 - 2.0 * r_i * r_j * jnp.cos(theta_j - theta_i)
        distance_constraints = dist_sq - 1.0

        # No equality constraints
        equalities = None

        # All constraints are inequalities (>= 0)
        inequalities = jnp.concatenate([order_constraints, distance_constraints])

        return equalities, inequalities

    @property
    def bounds(self):
        """Bounds on variables."""
        nv = self.NV
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)

        # Variables are interleaved: r1, theta1, r2, theta2, ...
        # R(i) in [0, 1] and THETA(i) in [0, PI] for i = 1 to NV-1
        indices = jnp.arange(nv - 1)
        upper = upper.at[2 * indices].set(1.0)  # R(i+1)
        upper = upper.at[2 * indices + 1].set(jnp.pi)  # THETA(i+1)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value for NV=100."""
        return jnp.array(-0.77847)
