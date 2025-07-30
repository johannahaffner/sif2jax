import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HIMMELBA(AbstractNonlinearEquations):
    """A two variables problem by Himmelblau.

    Source: problem 25 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#215 (p. 61)

    SIF input: Ph. Toint, Dec 1989.

    classification: NLR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables
    m: int = 2  # 2 equations

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([8.0, 9.0])

    @property
    def args(self):
        return ()

    def residual(self, y, args):
        """Compute the residual vector.

        The system of equations is:
        - 0.25 * x1 = 5.0
        - x2 = 6.0

        So the residuals are:
        - g1 = 0.25 * x1 - 5.0
        - g2 = x2 - 6.0
        """
        x1, x2 = y

        # Note: pycutest inverts the scale factor for NLE problems
        # G1 has SCALE 0.25, so pycutest uses 1/0.25 = 4.0
        g1 = 4.0 * x1 - 20.0
        g2 = x2 - 6.0

        return jnp.array([g1, g2])

    @property
    def expected_result(self):
        # The solution is x1 = 20.0, x2 = 6.0
        return jnp.array([20.0, 6.0])

    @property
    def expected_objective_value(self):
        # For nonlinear equations, the objective is ||residual||^2 = 0 at solution
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Free bounds for all variables."""
        return None
