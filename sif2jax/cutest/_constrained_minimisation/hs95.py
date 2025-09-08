import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS95(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 95.

    Source: problem 95 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Ph. Toint, April 1991.

    classification LQR2-AN-6-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(6)

    @property
    def y0(self):
        """Initial guess."""
        # Not specified in SIF, using zeros
        return jnp.zeros(6, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Linear-quadratic objective function."""
        del args
        x1, x2, x3, x4, x5, x6 = y[0], y[1], y[2], y[3], y[4], y[5]

        # Linear terms
        linear = 4.3 * x1 + 31.8 * x2 + 63.3 * x3 + 15.8 * x4 + 68.5 * x5 + 4.7 * x6

        return linear

    @property
    def bounds(self):
        """Upper bounds on variables."""
        lower = jnp.zeros(6, dtype=jnp.float64)
        upper = jnp.array([0.31, 0.046, 0.068, 0.042, 0.028, 0.0134], dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Inequality constraints with bilinear terms."""
        x1, x2, x3, x4, x5, x6 = y[0], y[1], y[2], y[3], y[4], y[5]

        # C1: 17.1*x1 + 38.2*x2 + 204.2*x3 + 212.3*x4 + 623.4*x5 + 1495.5*x6
        #     - 169.0*x1*x3 - 3580.0*x3*x5 - 3810.0*x4*x5 - 18500.0*x4*x6 - 24300.0*x5*x6 >= 4.97
        c1 = (
            17.1 * x1
            + 38.2 * x2
            + 204.2 * x3
            + 212.3 * x4
            + 623.4 * x5
            + 1495.5 * x6
            - 169.0 * x1 * x3
            - 3580.0 * x3 * x5
            - 3810.0 * x4 * x5
            - 18500.0 * x4 * x6
            - 24300.0 * x5 * x6
            - 4.97
        )

        # C2: 17.9*x1 + 36.8*x2 + 113.9*x3 + 169.7*x4 + 337.8*x5 + 1385.2*x6
        #     - 139.0*x1*x3 - 2450.0*x4*x5 - 16600.0*x4*x6 - 17200.0*x5*x6 >= -1.88
        c2 = (
            17.9 * x1
            + 36.8 * x2
            + 113.9 * x3
            + 169.7 * x4
            + 337.8 * x5
            + 1385.2 * x6
            - 139.0 * x1 * x3
            - 2450.0 * x4 * x5
            - 16600.0 * x4 * x6
            - 17200.0 * x5 * x6
            + 1.88
        )

        # C3: -273.0*x2 - 70.0*x4 - 819.0*x5 + 26000.0*x4*x5 >= -29.08
        c3 = -273.0 * x2 - 70.0 * x4 - 819.0 * x5 + 26000.0 * x4 * x5 + 29.08

        # C4: 159.9*x1 - 311.0*x2 + 587.0*x4 + 391.0*x5 + 2198.0*x6 - 14000.0*x1*x6 >= -78.02
        c4 = (
            159.9 * x1
            - 311.0 * x2
            + 587.0 * x4
            + 391.0 * x5
            + 2198.0 * x6
            - 14000.0 * x1 * x6
            + 78.02
        )

        inequalities = jnp.array([c1, c2, c3, c4])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # LO SOLTN = 0.015619514
        return jnp.array(0.015619514)
