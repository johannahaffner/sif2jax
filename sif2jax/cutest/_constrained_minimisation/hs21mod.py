import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS21MOD(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 21 (Modified version).

    Source: problem 21 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, April 1990

    classification SLR2-AN-7-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 7

    @property
    def y0(self):
        """Initial guess."""
        # Start point: x1=-1, x2=-1, rest default to 0
        return jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args
        x1, x2, x3, x4, x5, x6, x7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

        # Objective: -100.0 + 0.01*x1^2 + x2^2 + 0.01*x3^2 + x4^2
        #             + 0.01*x5^2 + 0.01*x6^2 + x7^2
        objective_value = (
            -100.0
            + 0.01 * x1 * x1
            + x2 * x2
            + 0.01 * x3 * x3
            + x4 * x4
            + 0.01 * x5 * x5
            + 0.01 * x6 * x6
            + x7 * x7
        )

        return objective_value

    @property
    def bounds(self):
        """Variable bounds."""
        # x1∈[2,50], x2∈[-50,50], x3∈[-∞,50], x4∈[2,∞], x5∈[-∞,∞], x6∈[-∞,0], x7∈[0,∞]
        lower = jnp.array(
            [2.0, -50.0, -jnp.inf, 2.0, -jnp.inf, -jnp.inf, 0.0], dtype=jnp.float64
        )
        upper = jnp.array(
            [50.0, 50.0, 50.0, jnp.inf, jnp.inf, 0.0, jnp.inf], dtype=jnp.float64
        )
        return lower, upper

    def constraint(self, y):
        """Linear inequality constraint."""
        x1, x2 = y[0], y[1]

        # CON1: 10*x1 - x2 >= 10, so 10*x1 - x2 - 10 >= 0
        c1 = 10.0 * x1 - x2 - 10.0

        inequalities = jnp.array([c1])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # LO SOLTN = -99.96
        return jnp.array(-99.96)
