import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HYPCIR(AbstractNonlinearEquations):
    """Intersection of a circle and an hyperbola.

    Source: problem 214 (p. 68) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification NQR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Starting point."""
        return jnp.array([0.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def m(self):
        """Number of equations."""
        return jnp.array(2)

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    def residual(self, y, args):
        """Compute the residuals (vectorized).

        The equations are:
        1. x1 * x2 = 1  (hyperbola)
        2. x1^2 + x2^2 = 4  (circle)

        So the residuals are:
        1. x1 * x2 - 1
        2. x1^2 + x2^2 - 4
        """
        x1, x2 = y[0], y[1]

        r1 = x1 * x2 - 1.0
        r2 = x1**2 + x2**2 - 4.0

        return jnp.array([r1, r2])

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
