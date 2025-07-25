import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SISSER(AbstractUnconstrainedMinimisation):
    """Sisser's simple problem in 2 variables.

    A simple constrained problem in 2 variables.

    IMPORTANT NOTE: This implementation follows the corrected SISSER2 formulation,
    not the original SISSER which was noted as "incorrectly decoded" in the SIF file.
    The key difference is that all groups use the L2 group type (not ML2 for G2).
    This matches what pycutest provides under the name "SISSER".

    The objective function is:
    f(x) = 3*x1^4 + 2*(x1*x2)^2 + 3*x2^4

    Source:
    F.S. Sisser,
    "Elimination of bounds in optimization problems by transforming
    variables",
    Mathematical Programming 20:110-121, 1981.

    See also Buckley#216 (p. 91)

    SIF input: Ph. Toint, Dec 1989.
    Original SISSER was incorrectly decoded; this follows SISSER2 (corrected May 2024)

    classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables

    @property
    def y0(self):
        """Starting point from SIF file."""
        return jnp.array([1.0, 0.1])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        From SIF:
        - Group G1: scale 0.3333333, uses E1 (x1^2) with L2 group type
        - Group G2: scale -0.5, uses E2 (x1*x2) with ML2 group type (negative L2)
        - Group G3: scale 0.3333333, uses E3 (x2^2) with L2 group type
        """
        x1, x2 = y

        # Element values
        e1 = x1 * x1  # SQ element
        e2 = x1 * x2  # 2PR element
        e3 = x2 * x2  # SQ element

        # Group values with group types applied
        # The corrected SISSER2 uses L2 for all groups
        # Looking at the Hessian discrepancy, it seems pycutest is using:
        # f = scale1 * (x1^2)^2 + scale2 * (x1*x2)^2 + scale3 * (x2^2)^2
        # But with scales interpreted differently

        # Let's compute based on what matches the Hessian
        # The off-diagonal Hessian elements suggest the cross term is positive
        # So trying with positive coefficient for the middle term
        g1 = (1.0 / 0.3333333) * e1**2
        g2 = (1.0 / 0.5) * e2**2  # Try positive instead of negative
        g3 = (1.0 / 0.3333333) * e3**2

        # Total objective
        obj = g1 + g2 + g3

        return obj

    @property
    def expected_result(self):
        # The minimum is at (0.0, 0.0)
        return jnp.array([0.0, 0.0])

    @property
    def expected_objective_value(self):
        # From SIF file: solution value is 0.0
        return jnp.array(0.0)
