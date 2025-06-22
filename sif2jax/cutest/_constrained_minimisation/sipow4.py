import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class SIPOW4(AbstractConstrainedMinimisation):
    """SIPOW4 problem - Semi-infinite programming example.

    This is a discretization of a one sided approximation problem of
    approximating the function xi * xi * eta by a linear polynomial
    on the boundary of a circle (xi - 0.5)**2 + (eta - 0.5)**2 = 0.5

    Source: problem 4 in
    M. J. D. Powell,
    "Log barrier methods for semi-infinite programming calculations"
    Numerical Analysis Report DAMTP 1992/NA11, U. of Cambridge, UK.

    SIF input: A. R. Conn and Nick Gould, August 1993

    Classification: LLR2-AN-4-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 4  # x1, x2, x3, x4

    @property
    def m(self):
        """Number of constraints."""
        # M constraints (M/2 inequalities in each direction)
        return self.M

    @property
    def M(self):
        """Problem size parameter (must be even)."""
        return 2000  # Default value from SIF file

    def objective(self, y, args):
        """Compute the objective (minimize x4)."""
        del args
        return y[3]  # x4

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2, x3, x4 = y

        M = self.M
        M_half = M // 2

        # Set up uniform spacings on the circle
        theta = inexact_asarray(jnp.arange(1, M_half + 1)) * 2.0 * jnp.pi / M
        pi_4_minus_theta = jnp.pi / 4.0 - theta
        cos_vals = jnp.cos(pi_4_minus_theta)
        sin_vals = jnp.sin(pi_4_minus_theta)

        root_half = jnp.sqrt(0.5)
        xi = 0.5 - root_half * cos_vals
        eta = 0.5 - root_half * sin_vals

        # Compute xi * xi * eta for RHS
        # xixi_eta = xi * xi * eta  # Not used in pycutest's constraint formulation

        # Constraints:
        # C(j): x1 + x2*xi(j) + x3*eta(j) + x4 >= xixi_eta(j) for j=1..M/2
        # C(j+M/2): x1 + x2*xi(j) + x3*eta(j) <= xixi_eta(j) for j=1..M/2
        # Note: Second set doesn't include x4!

        # First M/2 constraints (pycutest returns g(x) not g(x)-b)
        c1 = x1 + x2 * xi + x3 * eta + x4

        # Second M/2 constraints (pycutest returns g(x) not g(x)-b)
        c2 = x1 + x2 * xi + x3 * eta

        # All are inequality constraints
        constraints = jnp.concatenate([c1, c2])

        return None, constraints

    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.array([-0.1, 0.0, 0.0, 1.2]))

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(4, -jnp.inf)
        upper = jnp.full(4, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Depends on M:
        # M = 20: 2.0704432e-1
        # M = 100: 2.6110334e-1
        # M = 500: 2.7060094e-1
        # M = 2000: 2.7236200e-1
        if self.M == 20:
            return jnp.array(2.0704432e-1)
        elif self.M == 100:
            return jnp.array(2.6110334e-1)
        elif self.M == 500:
            return jnp.array(2.7060094e-1)
        elif self.M == 2000:
            return jnp.array(2.7236200e-1)
        else:
            return None
