import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ZANGWIL2(AbstractUnconstrainedMinimisation):
    """Zangwill's problem in 2 variables.

    The objective function has the form:
    f(x) = 15.0 * ((16.0 * x1^2 + 16.0 * x2^2 - 8.0 * x1 * x2
                    - 56.0 * x1 - 256.0 * x2 - 991.0))

    Source: problem 7 (p. 102) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification: QUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables

    @property
    def y0(self):
        """Starting point from SIF file."""
        return jnp.array([3.0, 8.0])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        From SIF:
        - Group G1 has SCALE = 15.0
        - Linear terms: -56.0*x1 - 256.0*x2
        - Constant: -991
        - Element uses: X1SQ (16.0), X2SQ (16.0), X1X2 (-8.0)
        """
        x1, x2 = y

        # Element values
        x1sq = x1 * x1  # SQ element
        x2sq = x2 * x2  # SQ element
        x1x2 = x1 * x2  # 2PR element

        # Group G1 value (before scaling)
        # Linear terms: -56.0*x1 - 256.0*x2
        # Element contributions: 16.0*x1sq + 16.0*x2sq - 8.0*x1x2
        # Constant: -991

        # In SIF, the scaling is applied to the entire group including constants
        # Note: In AMPL, it shows division by 15, and the constant sign is flipped
        g1_value = (
            -56.0 * x1 - 256.0 * x2 + 991.0 + 16.0 * x1sq + 16.0 * x2sq - 8.0 * x1x2
        )

        # Apply scale factor (division as per AMPL)
        obj = g1_value / 15.0

        return obj

    @property
    def expected_result(self):
        # The minimum is at (x1, x2) = (2.0, 9.0)
        return jnp.array([2.0, 9.0])

    @property
    def expected_objective_value(self):
        # From SIF file: -18.2
        return jnp.array(-18.2)
