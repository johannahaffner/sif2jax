import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class EXTRASIM(AbstractConstrainedMinimisation):
    """An extremely simple linear program.

    Source:
    A.Conn, N. Gould, Ph. Toint
    "LANCELOT, a Fortran package for large-scale nonlinear optimization"
    Springer Verlag, 1992

    SIF input: Ph. Toint, Dec 1991.

    Classification: LLR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    @property
    def m(self):
        """Number of constraints."""
        return jnp.array(1)

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        del args
        x, y_var = y

        # Objective: x - 1 (minimization of x with constant -1)
        return x - 1.0

    def constraint(self, y: Array):
        """Compute the constraints."""
        x, y_var = y

        # Equality constraint: x + 2*y = 2
        # Rearranged as: x + 2*y - 2 = 0
        equality_constraint = x + 2.0 * y_var - 2.0

        return None, jnp.array([equality_constraint])

    @property
    def y0(self):
        """Initial guess."""
        # No START POINT specified in SIF file, so use default [0, 0]
        # Even though this violates the constraint, it matches what pycutest expects
        return inexact_asarray(jnp.array([0.0, 0.0]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        # x: no bounds specified, defaults to [0, +∞) in SIF
        # y: free (XR means free variable)
        lower_bounds = jnp.array([0.0, -jnp.inf])
        upper_bounds = jnp.array([jnp.inf, jnp.inf])
        return lower_bounds, upper_bounds

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # For linear program min (x - 1) s.t. x + 2*y = 2, x >= 0
        # Minimum x occurs at x = 0, which gives y = 1 from the constraint
        return jnp.array([0.0, 1.0])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # At x = 0, y = 1: objective = 0 - 1 = -1
        return jnp.array(-1.0)
