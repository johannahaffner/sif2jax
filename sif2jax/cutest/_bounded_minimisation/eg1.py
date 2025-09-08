import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class EG1(AbstractBoundedMinimisation):
    """Simple nonlinear problem from LANCELOT Manual Section 1.2.3.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
    (Release A)"
    Springer Verlag, 1992.

    SIF input: N. Gould and Ph. Toint, June 1994.

    Classification: OBR2-AY-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 3

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        del args
        x1, x2, x3 = y

        # From SIF file structure:
        # GROUP1: x1 with GTYPE1 (alpha^2)
        # GROUP2: x2*x3 with GTYPE2 (alpha^4)
        # GROUP3: x2*sin(x1+x3) + x1*x3

        # GROUP1: alpha = x1, f = alpha^2 = x1^2
        group1 = x1**2

        # GROUP2: alpha = x2*x3, f = alpha^4 = (x2*x3)^4
        group2 = (x2 * x3) ** 4

        # GROUP3: x2*sin(x1+x3) + x1*x3
        group3 = x2 * jnp.sin(x1 + x3) + x1 * x3

        return group1 + group2 + group3

    @property
    def y0(self):
        """Initial guess."""
        # No START POINT specified in SIF file, so default is 0.0 for all variables
        # But we need to satisfy bounds: x2 in [-1,1], x3 in [1,2]
        # Since 0.0 violates x3 >= 1.0, use x3 = 1.0
        return inexact_asarray(jnp.array([0.0, 0.0, 1.0]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        # x1: free (no bounds)
        # x2: -1.0 <= x2 <= 1.0
        # x3: 1.0 <= x3 <= 2.0
        lower_bounds = jnp.array([-jnp.inf, -1.0, 1.0])
        upper_bounds = jnp.array([jnp.inf, 1.0, 2.0])
        return lower_bounds, upper_bounds

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Not provided in SIF file
        return None
