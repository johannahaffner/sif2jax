import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Attempts made: Fixed G3 using INV group type with SQ elements, but values still
# not matching exactly
# Suspected issues: Complex SIF group/element interaction may need more detailed
# analysis
# Additional resources needed: SIF specification documentation for proper INV
# group handling
class BRKMCC(AbstractUnconstrainedMinimisation):
    """BRKMCC function.

    Source: Problem 85 (p.35) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From SIF file structure:
        # G1: L2 group type with X1 coefficient 1.0, constant 2.0 -> (X1 - 2.0)^2
        g1 = (x1 - 2.0) ** 2

        # G2: L2 group type with X2 coefficient 1.0, constant 1.0 -> (X2 - 1.0)^2
        g2 = (x2 - 1.0) ** 2

        # G3: INV group type with scale 25.0, constant -1.0
        # Elements: E1 (SQ type on X1) with coeff -0.25, E2 (SQ type on X2) with
        # coeff -1.0
        # GVAR = -0.25 * X1^2 - 1.0 * X2^2, then INV: 1.0/GVAR, scaled by 25.0,
        # constant -1.0
        gvar_g3 = -0.25 * x1**2 - 1.0 * x2**2
        g3 = 25.0 / gvar_g3 - 1.0

        # G4: L2 group type with X1 coeff 1.0, X2 coeff -2.0, scale 0.2, constant -1.0
        # 0.2 * ((X1 - 2.0*X2) - 1.0)^2
        g4 = 0.2 * ((x1 - 2.0 * x2) - 1.0) ** 2

        return g1 + g2 + g3 + g4

    def y0(self):
        # Initial values from SIF file
        return jnp.array([2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        # According to literature, the minimum is around (0.91, 0.61)
        return jnp.array([0.91, 0.61])

    def expected_objective_value(self):
        # According to the SIF file comment (line 77),
        # the optimal objective value is 0.16904
        return jnp.array(0.16904)
