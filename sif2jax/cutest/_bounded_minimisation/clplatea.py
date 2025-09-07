import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class CLPLATEA(AbstractBoundedMinimisation):
    """The clamped plate problem (Strang, Nocedal, Dax).

    The problem comes from the discretization the following problem
    in mechanics: a plate is clamped on one edge and loaded on the
    opposite side. The plate is the unit square.

    In this version of the problem, the weight WGHT is entirely put on the
    upper right corner of the plate.

    The plate is clamped on its lower edge, by fixing the
    corresponding variables to zero.

    Source:
    J. Nocedal,
    "Solving large nonlinear systems of equations arising in mechanics",
    Proceedings of the Cocoyoc Numerical Analysis Conference, Mexico,
    pp. 132-141, 1981.

    SIF input: Ph. Toint, Dec 1989.

    classification OXR2-MN-V-0
    """

    n: int = 5041  # Default for P=71, n = P^2 = 71^2 = 5041
    P: int = 71  # Number of discretization points along each axis
    WGHT: float = -0.1  # Total weight on the upper edge
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial point - all zeros."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Lower edge X(1,J) for J=1:P fixed to 0, others unbounded."""
        # The SIF file defines variables in column-major order
        # (outer loop J, inner loop I)
        # So X(i,j) in the SIF file is at position (j-1)*P + (i-1)
        # X(1,J) for J=1:P means positions 0, P, 2P, ..., (P-1)*P

        lower = jnp.full(self.n, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(self.n, jnp.inf, dtype=jnp.float64)

        # Fix X(1,J) to 0 for all J
        # These are at positions (J-1)*P + 0 for J=1:P
        for j in range(self.P):
            idx = j * self.P  # Position of X(1,j+1) in flattened array
            lower = lower.at[idx].set(0.0)
            upper = upper.at[idx].set(0.0)

        return lower, upper

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        # Reshape to grid using Fortran order (column-major)
        # The SIF file defines variables with outer loop J, inner loop I
        X = y.reshape(
            self.P, self.P, order="F"
        )  # X[i,j] corresponds to X(i+1,j+1) in SIF

        # Compute scaling factors
        hp2 = 0.5 * self.P * self.P
        scale_ab = 2.0
        scale_cd = 1.0 / hp2

        # Initialize objective
        obj = 0.0

        # Add contributions from internal nodes
        # (i=2:P, j=2:P in Fortran = i=1:P-1, j=1:P-1 in Python)
        # Group A: (X(i,j) - X(i,j-1))^2 with scale 2.0
        diff_j = X[1:, 1:] - X[1:, :-1]
        obj = obj + scale_ab * jnp.sum(diff_j**2)

        # Group B: (X(i,j) - X(i-1,j))^2 with scale 2.0
        diff_i = X[1:, 1:] - X[:-1, 1:]
        obj = obj + scale_ab * jnp.sum(diff_i**2)

        # Group C: (X(i,j) - X(i,j-1))^4 with scale 1/hp2
        obj = obj + scale_cd * jnp.sum(diff_j**4)

        # Group D: (X(i,j) - X(i-1,j))^4 with scale 1/hp2
        obj = obj + scale_cd * jnp.sum(diff_i**4)

        # Add weight contribution at upper right corner X(P,P)
        obj = obj + self.WGHT * X[-1, -1]

        return obj

    @property
    def expected_result(self):
        # Not provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file SOLTN(71) comment
        return jnp.array(-1.2592e-02)
