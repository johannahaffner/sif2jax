import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MEXHAT(AbstractUnconstrainedMinimisation):
    """
    MEXHAT problem.

    The mexican hat problem with penalty parameter 0.00001

    Source:
    A.A. Brown and M. Bartholomew-Biggs,
    "Some effective methods for unconstrained optimization based on
    the solution of ordinary differential equations",
    Technical Report 178, Numerical Optimization Centre, Hatfield
    Polytechnic, (Hatfield, UK), 1987.

    SIF input: Ph. Toint, June 1990.

    classification OUR2-AN-2-0

    TODO: Human review needed
    Attempts made: Multiple interpretations of SIF scaling and group types
    Suspected issues: Incorrect interpretation of how INVP scaling interacts with
    L2 group type
    Additional resources needed: Clarification on SIF group scaling semantics
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Parameters
        invp = 0.00001
        p = 1.0 / invp  # p = 100000.0

        # Element values
        x1_minus_1 = x1 - 1.0
        x1_minus_1_sq = x1_minus_1 * x1_minus_1

        x2_minus_x1sq = x2 - x1 * x1
        x2_minus_x1sq_sq = x2_minus_x1sq * x2_minus_x1sq

        # Based on AMPL model:
        # minimize f: - 2*(x[1]-1)^2 + p*(   (-0.02+(x[2]-x[1]^2)^2/p+(x[1]-1)^2)^2  )

        # First term: -2*(x1-1)^2
        first_term = -2.0 * x1_minus_1_sq

        # Second term: p * (inner_expression)^2
        # inner_expression = -0.02 + (x2-x1^2)^2/p + (x1-1)^2
        inner_expr = -0.02 + x2_minus_x1sq_sq / p + x1_minus_1_sq
        second_term = p * inner_expr * inner_expr

        return first_term + second_term

    def y0(self):
        return jnp.array([0.86, 0.72])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Two solution values given in SIF file
        # Using the first one
        return jnp.array(-0.0898793)
