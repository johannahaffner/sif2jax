import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class SIPOW3(AbstractConstrainedMinimisation):
    """SIPOW3 problem - Semi-infinite programming example.

    This is a discretization of a one sided approximation problem of
    approximating the function xi * xi * eta by a linear polynomial
    on the boundary of the unit square [0,1]x[0,1].

    Source: problem 3 in
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
        M_eighth = M // 8
        M_quarter = M // 4
        M_3eighth = 3 * M // 8

        # Set up uniform spacings on the square
        step = 8.0 / M
        xi = jnp.zeros(M_half)
        eta = jnp.zeros(M_half)

        # Create indices for vectorized assignment
        indices = jnp.arange(M_half)

        # Segment [0,0]x[0,1]
        mask1 = indices < M_eighth
        eta = jnp.where(mask1, indices * step, eta)
        xi = jnp.where(mask1, 0.0, xi)

        # Segment [0,1]x[1,1]
        mask2 = (indices >= M_eighth) & (indices < M_quarter)
        xi = jnp.where(mask2, indices * step - 1.0, xi)
        eta = jnp.where(mask2, 1.0, eta)

        # Segment [1,1]x[1,0]
        mask3 = (indices >= M_quarter) & (indices < M_3eighth)
        xi = jnp.where(mask3, 1.0, xi)
        eta = jnp.where(mask3, 1.0 - (indices * step - 2.0), eta)

        # Segment [1,0]x[0,0]
        mask4 = indices >= M_3eighth
        xi = jnp.where(mask4, 1.0 - (indices * step - 3.0), xi)
        eta = jnp.where(mask4, 0.0, eta)

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
        # M = 20: 3.0315716e-1
        # M = 100: 5.0397238e-1
        # M = 500: 5.3016386e-1
        # M = 2000: 5.3465470e-1
        # M = 10000: 5.3564207e-1
        if self.M == 20:
            return jnp.array(3.0315716e-1)
        elif self.M == 100:
            return jnp.array(5.0397238e-1)
        elif self.M == 500:
            return jnp.array(5.3016386e-1)
        elif self.M == 2000:
            return jnp.array(5.3465470e-1)
        elif self.M == 10000:
            return jnp.array(5.3564207e-1)
        else:
            return None
