import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CHARDIS1(AbstractConstrainedMinimisation):
    """Distribution of charges on a round plate (2D).

    Minimize the sum of squared distances between charges,
    with constraints that charges must lie on a circle (except the first).
    This is the incorrectly decoded version (see CHARDIS12 for correction).

    Problem:
    min sum_{i=1}^{n-1} sum_{j=i+1}^{n} [(x_i - x_j)^2 + (y_i - y_j)^2]

    Subject to:
    x_i^2 + y_i^2 = R^2 for i = 2, ..., n (charges on circle)
    x_1 is fixed at R, y_1 is fixed at 0 (first charge at (R, 0))

    where R = 1.0 and n is the number of charges.

    Source:
    R. Felkel, Jun 1999.
    incorrectly decoded version (see CHARDIS12 for correction)

    classification: OQR2-AY-V-V

    TODO: Human review needed
    Attempts made: [SIF analysis, starting value correction, constraint
    implementation, linear vs REZIP analysis]
    Suspected issues: Starting value precision, constraint formulation
    vs pycutest implementation
    Resources needed: Detailed comparison of SIF starting value computation
    vs pycutest
    Error: First constraint differs by 0.002, affecting all other computations
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

        # CHARDIS1: exact SIF formulas from START POINT section
        # R = 1.0 for CHARDIS1 (not 10.0 like in bounds)
        # First charge: X1 = R = 1.0, Y1 = 0.0 (fixed in SIF)
        # For I = 2 to NP1:
        #   I- = I - 1
        #   RealNP1-I = 1000 - I
        #   PHII- = (2π/999) * (I-1)
        #   RI- = (1.0/999) * (1000 - I)  [R = 1.0 in CHARDIS1]
        #   XS = cos(PHII-) * RI-
        #   YS = sin(PHII-) * RI-

        # Initialize arrays
        x = jnp.zeros(n_charges)
        y = jnp.zeros(n_charges)

        # First charge at (R, 0) = (1.0, 0.0) - fixed in SIF
        x = x.at[0].set(1.0)  # R = 1.0 for CHARDIS1
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

        The objective is the sum of squared distances between charges.
        This is the "incorrectly decoded" version of CHARDIS1.

        From SIF GROUP USES: CHARDIS1 is missing "XT O(I,J) REZIP" line,
        so it uses default linear function: X(I,J) + Y(I,J)
        where X(I,J), Y(I,J) are squared differences (xi-xj)^2, (yi-yj)^2
        No scaling factor in CHARDIS1 SIF.
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

        # CHARDIS1: sum of squared distances (not reciprocals)
        # This is X(I,J) + Y(I,J) = (xi-xj)^2 + (yi-yj)^2
        masked_dist_sq = mask * dist_sq

        return jnp.sum(masked_dist_sq)

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
        r = 1.0  # R = 1.0 for CHARDIS1 (from SIF file)

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
        """Circle constraints: x_i^2 + y_i^2 = R^2 for i = 2, ..., n as inequalities."""
        r = 1.0  # R = 1.0 for CHARDIS1 (from SIF file)
        r2 = r * r

        # Extract coordinates from interleaved format [x1,y1,x2,y2,...]
        x = y[::2]  # x coordinates at even indices
        y_coords = y[1::2]  # y coordinates at odd indices

        # Based on test evidence: 999 inequality constraints for charges 2 to n
        # These are circle constraints x_i^2 + y_i^2 - R^2 <= 0 (or >= 0)
        inequality_constraints = x[1:] ** 2 + y_coords[1:] ** 2 - r2

        return None, inequality_constraints
