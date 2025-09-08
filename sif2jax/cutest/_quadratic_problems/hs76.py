import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class HS76(AbstractConstrainedQuadraticProblem):
    """Hock and Schittkowski problem 76.

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
        return jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        f(x) = x1^2 + 0.5*x2^2 + x3^2 + 0.5*x4^2 - x1*x3 + x3*x4 - x1 - 3*x2 + x3 - x4
        """
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Quadratic terms from elements
        quadratic = x1**2 + 0.5 * x2**2 + x3**2 + 0.5 * x4**2 - x1 * x3 + x3 * x4

        # Linear terms
        linear = -x1 - 3 * x2 + x3 - x4

        return quadratic + linear

    @property
    def bounds(self):
        """Lower bounds: 0 <= x_i for all i."""
        lower = jnp.zeros(4, dtype=jnp.float64)
        upper = jnp.full(4, jnp.inf, dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Linear inequality constraints.

        C1: x1 + 2*x2 + x3 + x4 <= 5.0
        C2: 3*x1 + x2 + 2*x3 - x4 <= 4.0
        C3: x2 + 4*x3 >= 1.5
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Following pycutest convention: LHS - RHS for all constraints
        inequalities = jnp.array(
            [
                # C1: x1 + 2*x2 + x3 + x4 <= 5.0
                x1 + 2 * x2 + x3 + x4 - 5.0,
                # C2: 3*x1 + x2 + 2*x3 - x4 <= 4.0
                3 * x1 + x2 + 2 * x3 - x4 - 4.0,
                # C3: x2 + 4*x3 >= 1.5
                x2 + 4 * x3 - 1.5,
            ]
        )

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
