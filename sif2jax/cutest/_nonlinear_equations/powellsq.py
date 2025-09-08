import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class POWELLSQ(AbstractNonlinearEquations):
    """POWELLSQ problem - Powell's singular problem.

    Source:
    M.J.D. Powell,
    "A hybrid method for nonlinear equations",
    In P. Rabinowitz(ed.) "Numerical Methods for Nonlinear Algebraic
    Equations", Gordon and Breach, 1970.

    See also Buckley#217 (p.84.)

    Classification: NOR2-AN-2-2

    SIF input: Ph. Toint, Dec 1989, correction November 2002.
               NIMG corrected July 2005 (thanks to Roger Fletcher)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    def num_residuals(self):
        """Number of residuals."""
        # 2 residual equations
        return jnp.array(2)

    def residual(self, y, args):
        """Compute the residuals."""
        del args
        x1, x2 = y

        # Residual equations:
        # Group G1: x1^2 = 0
        g1 = x1**2

        # Group G2: 10*x1/(x1+0.1) + 2*x2^2 = 0
        g2 = 10.0 * x1 / (x1 + 0.1) + 2.0 * x2**2

        return jnp.array([g1, g2])

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([3.0, 1.0])

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
