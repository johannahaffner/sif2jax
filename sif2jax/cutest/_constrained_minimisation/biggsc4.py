import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BIGGSC4(AbstractConstrainedMinimisation):
    """BIGGSC4 problem.

    A test quadratic test problem (invented starting point).

    Source:
    M. Batholomew-Biggs and F.G. Hernandez,
    "Some improvements to the subroutine OPALQP for dealing with large
    problems",
    Numerical Optimization Centre, Hatfield, 1992.

    SIF input: Ph Toint, April 1992.

    classification QLR2-AN-4-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y
        return -x1 * x3 - x2 * x4

    def y0(self):
        # Default starting point
        return jnp.zeros(4)

    def args(self):
        return None

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(-24.5)

    def bounds(self):
        # Variable bounds: 0 <= x <= 5
        lower = jnp.zeros(4)
        upper = jnp.full(4, 5.0)
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4 = y

        # No equality constraints - all are inequalities
        eq_constraint = None

        # All constraints are G-type (>=) inequalities in the SIF file
        # For ranged constraints (C1-C6), pycutest returns shifted values:
        # g(x) - lower_bound
        # For one-sided constraint (C7), pycutest returns: g(x) - constant
        ineq_constraint = jnp.array(
            [
                (x1 + x2) - 2.5,  # C1: ranged, shifted by lower bound
                (x1 + x3) - 2.5,  # C2: ranged, shifted by lower bound
                (x1 + x4) - 2.5,  # C3: ranged, shifted by lower bound
                (x2 + x3) - 2.0,  # C4: ranged, shifted by lower bound
                (x2 + x4) - 2.0,  # C5: ranged, shifted by lower bound
                (x3 + x4) - 1.5,  # C6: ranged, shifted by lower bound
                (x1 + x2 + x3 + x4) - 5.0,  # C7: one-sided G-type
            ]
        )

        return eq_constraint, ineq_constraint
