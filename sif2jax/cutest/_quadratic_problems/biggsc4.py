import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class BIGGSC4(AbstractConstrainedQuadraticProblem):
    """A test quadratic test problem (invented starting point).

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

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(4)

    @property
    def y0(self):
        """Initial guess - not specified in SIF, use zeros."""
        return jnp.zeros(4, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: -x1*x3 - x2*x4."""
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Product terms from the SIF
        return -x1 * x3 - x2 * x4

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 5."""
        lower = jnp.zeros(4, dtype=jnp.float64)
        upper = jnp.full(4, 5.0, dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Linear constraints.

        From the SIF file:
        C1-C6: Range constraints (treated as single constraints by pycutest)
        C7: One-sided inequality constraint x1 + x2 + x3 + x4 >= 5.0

        For range constraints, pycutest returns them shifted by lower bound.
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # No equality constraints
        eq_constraint = None

        # All constraints are inequalities
        # For ranged constraints (C1-C6), pycutest returns shifted values
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

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(-24.5)
