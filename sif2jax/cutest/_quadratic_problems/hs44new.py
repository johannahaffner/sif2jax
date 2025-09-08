import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class HS44NEW(AbstractConstrainedQuadraticProblem):
    """Hock and Schittkowski problem 44 (new version).

    Source: problem 44 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Ph.L. Toint, October 1990.

    classification QLR2-AN-4-6
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
        return jnp.ones(4, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        f(x) = -x1*x3 + x1*x4 - x2*x4 + x2*x3 + x1 - x2 - x3
        """
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Product terms
        products = -x1 * x3 + x1 * x4 - x2 * x4 + x2 * x3

        # Linear terms
        linear = x1 - x2 - x3

        return products + linear

    @property
    def bounds(self):
        """Lower bounds of 0 on all variables."""
        lower = jnp.zeros(4, dtype=jnp.float64)
        upper = jnp.full(4, jnp.inf, dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Linear inequality constraints.

        From SIF file with negative coefficients:
        CON1: -x1 - 2*x2 >= -8.0  (i.e., x1 + 2*x2 <= 8.0)
        CON2: -4*x1 - x2 >= -12.0  (i.e., 4*x1 + x2 <= 12.0)
        CON3: -3*x1 - 4*x2 >= -12.0  (i.e., 3*x1 + 4*x2 <= 12.0)
        CON4: -2*x3 - x4 >= -8.0  (i.e., 2*x3 + x4 <= 8.0)
        CON5: -x3 - 2*x4 >= -8.0  (i.e., x3 + 2*x4 <= 8.0)
        CON6: -x3 - x4 >= -5.0  (i.e., x3 + x4 <= 5.0)
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Following pycutest convention for >= constraints
        inequalities = jnp.array(
            [
                -x1 - 2 * x2 - (-8.0),  # CON1
                -4 * x1 - x2 - (-12.0),  # CON2
                -3 * x1 - 4 * x2 - (-12.0),  # CON3
                -2 * x3 - x4 - (-8.0),  # CON4
                -x3 - 2 * x4 - (-8.0),  # CON5
                -x3 - x4 - (-5.0),  # CON6
            ]
        )

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file comment."""
        # Note: SIF has two possible solutions: -13.0 or -15.0
        return jnp.array(-15.0)
