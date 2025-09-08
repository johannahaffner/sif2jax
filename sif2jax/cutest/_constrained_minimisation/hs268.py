import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS268(AbstractConstrainedMinimisation):
    """Schittkowski problem 268.

    A quadratic programming problem.

    Source:
    K. Schittkowski
    "More Test Examples for Nonlinear Programming Codes"
    Springer Verlag, Berlin, Lecture notes in economics and
    mathematical systems, volume 282, 1987

    SIF input: Michel Bierlaire and Annick Sartenaer, October 1992.

    classification QLR2-AN-5-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 5

    @property
    def y0(self):
        """Initial guess."""
        # From SIF: default start point is 1.0 for all variables
        return jnp.ones(5, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args

        # D matrix from SIF
        D = jnp.array(
            [
                [10197.0, -12454.0, -1013.0, 1948.0, 329.0],
                [-12454.0, 20909.0, -1733.0, -4914.0, -186.0],
                [-1013.0, -1733.0, 1755.0, 1089.0, -174.0],
                [1948.0, -4914.0, 1089.0, 1515.0, -22.0],
                [329.0, -186.0, -174.0, -22.0, 27.0],
            ],
            dtype=y.dtype,
        )

        # B vector from SIF
        b = jnp.array([-9170.0, 17099.0, -2271.0, -4336.0, -43.0], dtype=y.dtype)

        # Objective from AMPL: 14463.0 + sum_i,j D[i,j]*x[i]*x[j] - 2*sum_i B[i]*x[i]
        constant = 14463.0
        quadratic = jnp.dot(y, jnp.dot(D, y))
        linear = -2.0 * jnp.dot(b, y)

        return constant + quadratic + linear

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    def constraint(self, y):
        """Linear inequality constraints."""
        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]

        # C1: -x1 - x2 - x3 - x4 - x5 >= -5
        c1 = -x1 - x2 - x3 - x4 - x5 + 5.0

        # C2: 10*x1 + 10*x2 - 3*x3 + 5*x4 + 4*x5 >= 20
        c2 = 10 * x1 + 10 * x2 - 3 * x3 + 5 * x4 + 4 * x5 - 20.0

        # C3: -8*x1 + x2 - 2*x3 - 5*x4 + 3*x5 >= -40
        c3 = -8 * x1 + x2 - 2 * x3 - 5 * x4 + 3 * x5 + 40.0

        # C4: 8*x1 - x2 + 2*x3 + 5*x4 - 3*x5 >= 11
        c4 = 8 * x1 - x2 + 2 * x3 + 5 * x4 - 3 * x5 - 11.0

        # C5: -4*x1 - 2*x2 + 3*x3 - 5*x4 + x5 >= -30
        c5 = -4 * x1 - 2 * x2 + 3 * x3 - 5 * x4 + x5 + 30.0

        inequalities = jnp.array([c1, c2, c3, c4, c5])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
