import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class HATFLDH(AbstractConstrainedQuadraticProblem):
    """A test problem from the OPTIMA user manual.

    This is a nonlinear objective with linear constraints.

    Source:
    "The OPTIMA user manual (issue No.8, p. 91)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    classification QLR2-AN-4-7
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 4

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([1.0, 5.0, 5.0, 1.0])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: -x1*x3 - x2*x4."""
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Two product terms
        return -x1 * x3 - x2 * x4

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 5."""
        lower = jnp.zeros(4)
        upper = jnp.full(4, 5.0)
        return lower, upper

    def constraint(self, y):
        """Linear constraints.

        From the SIF file, we have:
        C1-C6: Range constraints (treated as single constraints by pycutest)
        C7: One-sided inequality constraint x1 + x2 + x3 + x4 >= 5.0

        For range constraints, pycutest returns them shifted by lower bound.
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # No equality constraints
        eq_constraint = None

        # All constraints are inequalities
        inequalities = jnp.array(
            [
                (x1 + x2) - 2.5,  # C1: 2.5 <= x1 + x2 <= 7.5
                (x1 + x3) - 2.5,  # C2: 2.5 <= x1 + x3 <= 7.5
                (x1 + x4) - 2.5,  # C3: 2.5 <= x1 + x4 <= 7.5
                (x2 + x3) - 2.0,  # C4: 2.0 <= x2 + x3 <= 7.0
                (x2 + x4) - 2.0,  # C5: 2.0 <= x2 + x4 <= 7.0
                (x3 + x4) - 1.5,  # C6: 1.5 <= x3 + x4 <= 6.5
                (x1 + x2 + x3 + x4) - 5.0,  # C7: x1 + x2 + x3 + x4 >= 5.0
            ]
        )

        return eq_constraint, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(-24.4999998)
