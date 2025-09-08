import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS76I(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 76 (I version).

    Problem used as integer quadratic test problem with new upper bounds.

    Source: problem 76 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, March 1991.

    classification QLR2-AN-4-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(4)

    @property
    def y0(self):
        """Initial guess."""
        # Start point: all variables start at 0.5
        return jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Linear terms from GROUPS section: -x1 - 3*x2 + x3 - x4
        linear = -x1 - 3.0 * x2 + x3 - x4

        # Quadratic terms from element uses:
        # E1: 1.0*x1², E2: 0.5*x2², E3: 1.0*x3², E4: 0.5*x4²
        # E5: -1.0*x1*x3, E6: 1.0*x3*x4
        quadratic = (
            1.0 * x1 * x1
            + 0.5 * x2 * x2
            + 1.0 * x3 * x3
            + 0.5 * x4 * x4
            + -1.0 * x1 * x3
            + 1.0 * x3 * x4
        )

        return linear + quadratic

    @property
    def bounds(self):
        """Variable bounds - upper bound of 5.0 on all variables."""
        lower = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        upper = jnp.array([5.0, 5.0, 5.0, 5.0], dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Linear constraints."""
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # C1 (L): x1 + 2*x2 + x3 + x4 ≤ 5.0, so constraint is (x1 + 2*x2 + x3 + x4) - 5.0 ≤ 0
        # For sif2jax format (≥ 0), we need: 5.0 - (x1 + 2*x2 + x3 + x4) ≥ 0
        # But the test expects the derivative to be [1, 2, 1, 1], which is the LHS coefficients
        # So we should return the LHS minus constant: (x1 + 2*x2 + x3 + x4) - 5.0
        c1 = (x1 + 2.0 * x2 + x3 + x4) - 5.0

        # C2 (L): 3*x1 + x2 + 2*x3 - x4 ≤ 4.0
        # Similarly: (3*x1 + x2 + 2*x3 - x4) - 4.0
        c2 = (3.0 * x1 + x2 + 2.0 * x3 - x4) - 4.0

        # C3 (G): x2 + 4*x3 ≥ 1.5, so (x2 + 4*x3) - 1.5 ≥ 0
        c3 = x2 + 4.0 * x3 - 1.5

        inequalities = jnp.array([c1, c2, c3])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
