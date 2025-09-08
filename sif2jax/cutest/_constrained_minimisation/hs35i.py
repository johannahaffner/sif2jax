import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS35I(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 35 (I version).

    MIQP - Problem version with new upper bounds of 5.

    Source: problem 35 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, April 1990

    classification QLR2-AN-3-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 3

    @property
    def y0(self):
        """Initial guess."""
        # Start point: all variables default to 0.5
        return jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args
        x1, x2, x3 = y[0], y[1], y[2]

        # Linear terms from GROUPS section: -8*x1 - 6*x2 - 4*x3
        linear = -8.0 * x1 - 6.0 * x2 - 4.0 * x3

        # Quadratic terms from element uses: 2*x1^2 + 2*x2^2 + x3^2 + 2*x1*x2 + 2*x1*x3
        quadratic = (
            2.0 * x1 * x1
            + 2.0 * x2 * x2
            + 1.0 * x3 * x3
            + 2.0 * x1 * x2
            + 2.0 * x1 * x3
        )

        # Constant from CONSTANTS section: interpret -9.0 as +9.0
        # (based on HS35MOD experience)
        constant = 9.0

        return constant + linear + quadratic

    @property
    def bounds(self):
        """Variable bounds - upper bound of 5.0 on all variables."""
        lower = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        upper = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Linear inequality constraint."""
        x1, x2, x3 = y[0], y[1], y[2]

        # CON1: -x1 - x2 - 2*x3 >= -3.0, so -x1 - x2 - 2*x3 + 3.0 >= 0
        c1 = -x1 - x2 - 2.0 * x3 + 3.0

        inequalities = jnp.array([c1])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # LO SOLTN = 0.1111111111
        return jnp.array(0.1111111111)
