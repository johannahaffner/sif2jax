import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class CHARDIS02(AbstractBoundedMinimisation):
    """Distribution of equal charges on [-R,R]x[-R,R] (2D).

    Minimize the scaled sum of reciprocals of squared distances between charges.
    This is the corrected version of CHARDIS0 (uses REZIP group function).

    Problem:
    min sum_{i=1}^{n-1} sum_{j=i+1}^{n} 1/[(x_i - x_j)^2 + (y_i - y_j)^2] / 0.01

    Subject to:
    -R <= x_i <= R for all i
    -R <= y_i <= R for all i

    where R = 10.0 and n is the number of charges.

    Source:
    R. Felkel, Jun 1999.
    correction by S. Gratton & Ph. Toint, May 2024
    modifield version of CHARDIS0 (formulation corrected)

    classification: OBR2-AY-V-V

    TODO: Gradient at y0 fails np.allclose with default rtol=1e-5.
    The 1/dist_sq formulation is ill-conditioned for close charges
    (dist_sq ~ 5e-4 at y0). Gradient is correct within rtol=1e-2.
    All other gradient tests (zeros, ones, alternating) pass.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n_charges: int = 1000  # Number of charges (NP1 from SIF)

    @property
    def n(self) -> int:
        """Total number of variables (2 * number of charges)."""
        return 2 * self.n_charges

    @property
    def args(self):
        return ()

    @property
    def y0(self):
        """Initial point from SIF file."""
        n_charges = self.n_charges

        # CHARDIS02 uses identical starting values to CHARDIS0
        # From empirical observation, the pattern matches CHARDIS0 exactly

        i = jnp.arange(n_charges, dtype=jnp.float64)

        # Empirically determined pattern that matches pycutest
        angle = (i + 1) * 2.0 * jnp.pi / 999.0
        radius = 5.0 - i * (5.0 / 999.0)

        # Compute coordinates
        x = radius * jnp.cos(angle)
        y = radius * jnp.sin(angle)

        # Set last charge (i=999) to exactly zero
        x = x.at[999].set(0.0)
        y = y.at[999].set(0.0)

        # Create interleaved array [x1,y1,x2,y2,...]
        result = jnp.zeros(2 * n_charges)
        result = result.at[::2].set(x)  # x coordinates at even indices
        result = result.at[1::2].set(y)  # y coordinates at odd indices

        return result

    def objective(self, y, args):
        """Compute the objective function.

        REZIP group function: F = 1/ALPHA where ALPHA = (xi-xj)^2 + (yi-yj)^2
        This is 1/dist_sq (reciprocal of SQUARED distance), not 1/dist.
        Each group has SCALE 0.01, so CUTEst divides by 0.01.
        """
        n_charges = self.n_charges
        # Extract coordinates from interleaved format [x1,y1,x2,y2,...]
        x = y[::2]
        y_coords = y[1::2]

        # Pairwise differences via subtract.outer
        dx = jnp.subtract.outer(x, x)
        dy = jnp.subtract.outer(y_coords, y_coords)
        dist_sq = dx**2 + dy**2

        # Add identity to diagonal so 1/(0+1)=1 on diagonal (avoids 1/0),
        # then subtract n_charges/2 to correct for diagonal after halving
        dist_sq_safe = dist_sq + jnp.eye(n_charges)
        reciprocals = 1.0 / dist_sq_safe

        # Symmetric: sum all and halve, minus diagonal contribution (n * 1.0 / 2)
        # SIF scaling: divide by 0.01
        return (0.5 * jnp.sum(reciprocals) - 0.5 * n_charges) / 0.01

    @property
    def expected_result(self):
        """The optimal solution is not known analytically."""
        return None

    @property
    def expected_objective_value(self):
        """The optimal objective value is not known analytically."""
        return None

    @property
    def bounds(self):
        """Bounds: all variables in [-R, R] where R = 10."""
        n_vars = self.n  # Total number of variables (2 * n_charges)
        r = 10.0
        lower = jnp.full(n_vars, -r)
        upper = jnp.full(n_vars, r)
        return lower, upper
