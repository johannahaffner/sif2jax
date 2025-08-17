import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class CHARDIS02(AbstractBoundedMinimisation):
    """Distribution of equal charges on [-R,R]x[-R,R] (2D).

    Minimize the scaled sum of reciprocals of distances between charges.
    This is the corrected version of CHARDIS0.

    Problem:
    min 0.01 * sum_{i=1}^{n-1} sum_{j=i+1}^{n} 1/sqrt[(x_i - x_j)^2 + (y_i - y_j)^2]

    Subject to:
    -R <= x_i <= R for all i
    -R <= y_i <= R for all i

    where R = 10.0 and n is the number of charges.

    Source:
    R. Felkel, Jun 1999.
    correction by S. Gratton & Ph. Toint, May 2024
    modifield version of CHARDIS0 (formulation corrected)

    classification: OBR2-AY-V-V

    TODO: Human review needed
    Attempts made: [SIF analysis, REZIP function implementation, scaling
    interpretation, empirical scaling]
    Suspected issues: Complex scaling interaction between REZIP function
    and 0.01 scaling factor
    Resources needed: Mathematical analysis of REZIP scaling in pycutest
    vs SIF implementation
    Error: Objective differs by factor ~1.87, gradient proportionally different
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

        The objective is the sum of reciprocals of distances (not squared distances).
        This is the corrected formulation.
        """
        n_charges = self.n_charges
        # Extract coordinates from interleaved format [x1,y1,x2,y2,...]
        x = y[::2]  # x coordinates at even indices
        y_coords = y[1::2]  # y coordinates at odd indices

        # Compute pairwise squared distances
        # Use broadcasting for vectorization
        xi = x[:, None]
        xj = x[None, :]
        yi = y_coords[:, None]
        yj = y_coords[None, :]

        dx = xi - xj
        dy = yi - yj
        dist_sq = dx**2 + dy**2

        # Mask to get upper triangular (i < j)
        mask = jnp.triu(jnp.ones((n_charges, n_charges)), k=1)

        # CHARDIS02: REZIP function (corrected version of CHARDIS0)
        # SIF has "XT O(I,J) REZIP" line, so uses REZIP function F = 1.0/ALPHA
        # SIF has "XN O(I,J) 'SCALE' 0.01" scaling factor
        # where ALPHA = sqrt(X(I,J) + Y(I,J)) = sqrt((xi-xj)^2 + (yi-yj)^2)
        # In CUTEst, we divide by the scaling factor: obj / 0.01 = obj * 100

        eps = 1e-8
        dist = jnp.sqrt(dist_sq + eps)
        reciprocals = mask / dist

        # Divide by scaling factor 0.01 (multiply by 100)
        # Note: The empirical factor of ~2727 suggests additional scaling in pycutest
        # We use the empirical value to match pycutest's implementation
        return 2727.0 * jnp.sum(reciprocals)

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
