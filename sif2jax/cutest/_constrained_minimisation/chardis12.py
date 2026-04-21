import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CHARDIS12(AbstractConstrainedMinimisation):
    """Distribution of charges on a round plate (2D).

    Minimize the sum of reciprocals of squared distances between charges,
    with constraints that charges lie on or outside a circle.
    This is the corrected version of CHARDIS1 (uses REZIP group function).

    Problem:
    min sum_{i=1}^{n-1} sum_{j=i+1}^{n} 1/[(x_i - x_j)^2 + (y_i - y_j)^2]

    Subject to:
    x_i^2 + y_i^2 >= R^2 for i = 2, ..., n (charges on or outside circle)
    x_1 = R, y_1 = 0 (first charge fixed at (R, 0))

    where R = 1.0 and n is the number of charges.

    Source:
    R. Felkel, Jun 1999.
    correction by S. Gratton & Ph. Toint, May 2024
    modifield version of CHARDIS1 (formulation corrected)

    classification: OQR2-AY-V-V

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

        # CHARDIS12: exact SIF formulas from START POINT section
        # R = 1.0 for CHARDIS12 (same as CHARDIS1)
        # First charge: X1 = R = 1.0, Y1 = 0.0 (fixed in SIF)
        # For I = 2 to NP1:
        #   I- = I - 1
        #   RealNP1-I = 1000 - I
        #   PHII- = (2π/999) * (I-1)
        #   RI- = (1.0/999) * (1000 - I)  [R = 1.0 in CHARDIS12]
        #   XS = cos(PHII-) * RI-
        #   YS = sin(PHII-) * RI-

        # Initialize arrays
        x = jnp.zeros(n_charges)
        y = jnp.zeros(n_charges)

        # First charge at (R, 0) = (1.0, 0.0) - fixed in SIF
        x = x.at[0].set(1.0)  # R = 1.0 for CHARDIS12
        y = y.at[0].set(0.0)

        # For charges 2 to NP1 (I = 2 to 1000), use exact SIF formulas
        I_values = jnp.arange(
            2, n_charges + 1, dtype=jnp.float64
        )  # I = 2, 3, ..., 1000
        I_minus_1 = I_values - 1.0  # I-1 = 1, 2, ..., 999
        RealNP1_I = 1000.0 - I_values  # 1000 - I = 998, 997, ..., 0

        # SIF calculations
        PHII = (2.0 * jnp.pi / 999.0) * I_minus_1  # (2π/999) * (I-1)
        RI = (1.0 / 999.0) * RealNP1_I  # (R/999) * (1000-I), R=1.0

        XS = jnp.cos(PHII) * RI
        YS = jnp.sin(PHII) * RI

        # Set coordinates for charges 2 to NP1
        x = x.at[1:].set(XS)
        y = y.at[1:].set(YS)

        # Create interleaved array [x1,y1,x2,y2,...] to match pycutest format
        result = jnp.zeros(2 * n_charges)
        result = result.at[::2].set(x)  # x coordinates at even indices
        result = result.at[1::2].set(y)  # y coordinates at odd indices

        return result

    def objective(self, y, args):
        """Compute the objective function.

        REZIP group function: F = 1/ALPHA where ALPHA = (xi-xj)^2 + (yi-yj)^2
        This is 1/dist_sq (reciprocal of SQUARED distance), not 1/dist.
        No scaling factor in CHARDIS12 SIF.
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
        return 0.5 * jnp.sum(reciprocals) - 0.5 * n_charges

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
        """First charge is fixed, others are free."""
        n_charges = self.n_charges
        r = 1.0  # R = 1.0 for CHARDIS12 (from SIF file)

        # For interleaved format [x1,y1,x2,y2,...]
        # First charge at (R,0) is fixed, others are free
        lower = jnp.full(2 * n_charges, -jnp.inf)
        upper = jnp.full(2 * n_charges, jnp.inf)

        # Fix first charge at (R, 0)
        lower = lower.at[0].set(r)  # x1 = R = 1.0
        upper = upper.at[0].set(r)  # x1 = R = 1.0
        lower = lower.at[1].set(0.0)  # y1 = 0
        upper = upper.at[1].set(0.0)  # y1 = 0

        return lower, upper

    def constraint(self, y):
        """Inequality constraints: x_i^2 + y_i^2 >= R^2 for i = 2, ..., n.

        SIF uses XL RES(I) with constant R^2, meaning RES(I) >= R^2,
        i.e. x_i^2 + y_i^2 >= R^2 (charges on or outside the circle).
        Convention: return c(y) where c(y) >= 0.
        """
        r2 = 1.0  # R^2 = 1.0 for CHARDIS12

        # Extract coordinates from interleaved format [x1,y1,x2,y2,...]
        x = y[::2]
        y_coords = y[1::2]

        # x_i^2 + y_i^2 - R^2 >= 0 for charges 2 to n
        inequality_constraints = x[1:] ** 2 + y_coords[1:] ** 2 - r2

        return None, inequality_constraints
