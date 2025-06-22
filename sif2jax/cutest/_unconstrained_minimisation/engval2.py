import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class ENGVAL2(AbstractUnconstrainedMinimisation):
    """The ENGVAL2 problem.

    Source: problem 15 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        x1, x2, x3 = y

        # Precompute common terms to avoid redundant calculations
        x1_sq = x1**2
        x2_sq = x2**2
        x3_sq = x3**2
        x1_plus_x2 = x1 + x2
        x1_sq_plus_x2_sq = x1_sq + x2_sq

        # Precompute more common subexpressions
        x3_minus_2 = x3 - 2
        x1_plus_x2_plus_x3 = x1_plus_x2 + x3
        x1_plus_x2_minus_x3 = x1_plus_x2 - x3
        five_x3_minus_x1_plus_1 = 5 * x3 - x1 + 1

        # From AMPL model:
        g1 = (x1_sq_plus_x2_sq + x3_sq - 1) ** 2
        g2 = (x1_sq_plus_x2_sq + x3_minus_2**2 - 1) ** 2
        g3 = (x1_plus_x2_plus_x3 - 1) ** 2
        g4 = (x1_plus_x2_minus_x3 + 1) ** 2
        g5 = (3 * x2_sq + x1**3 + five_x3_minus_x1_plus_1**2 - 36) ** 2

        return g1 + g2 + g3 + g4 + g5

    def y0(self):
        # Starting point from the file
        return jnp.array([1.0, 2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # From SIF file: *LO SOLTN              0.0
        return jnp.array(0.0)
