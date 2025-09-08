import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class PFIT2(AbstractNonlinearEquations):
    """The problem is to fit a model containing a pole, given data
    for values, first and second derivatives at two distinct points.

    The problem is not convex.

    Source: SIF input: Ph. Toint, Nov 1993.
            Lower bound on H added, Nov 2002

    Classification: NOR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(3)

    def num_residuals(self):
        """Number of residual equations."""
        return jnp.array(3)

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals for the three equations."""
        del args
        A, R, H = y

        # Problem constants
        CF = -26.6666666666
        CG = -60.4444444444
        CH = -71.1111111111

        # Common terms involving (1 + H)
        Y = 1.0 + H

        # Element T1: A * R * H
        T1 = A * R * H

        # Element T2: A * R * H * (1 - (1 + H)^(-(A+1)))
        A1 = A + 1.0
        C2 = Y ** (-A1)
        B2 = 1.0 - C2
        T2 = A * R * H * B2

        # Element T3: A * (A + 1) * R * H^2
        T3 = A * (A + 1.0) * R * H * H

        # Element T4: R * (1 - (1 + H)^(-A))
        C4 = Y ** (-A)
        B4 = 1.0 - C4
        T4 = R * B4

        # Element T5: A * (A + 1) * R * H^2 * (1 - (1 + H)^(-(A+2)))
        A2 = A + 2.0
        C5 = Y ** (-A2)
        B5 = 1.0 - C5
        T5 = A * (A + 1.0) * R * H * H * B5

        # Group EF: -0.5 * T3 + 1.0 * T1 - 1.0 * T4
        EF = -0.5 * T3 + 1.0 * T1 - 1.0 * T4

        # Group EG: -1.0 * T3 + 1.0 * T2
        EG = -1.0 * T3 + 1.0 * T2

        # Group EH: -1.0 * T5
        EH = -1.0 * T5

        # The residuals are the differences from target constants
        residual_F = EF - CF
        residual_G = EG - CG
        residual_H = EH - CH

        return jnp.array([residual_F, residual_G, residual_H], dtype=jnp.float64)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([1.0, 0.0, 1.0], dtype=jnp.float64)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds on variables: H >= -0.5, others are free."""
        lower_bounds = jnp.array([-jnp.inf, -jnp.inf, -0.5], dtype=jnp.float64)
        upper_bounds = jnp.array([jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float64)
        return lower_bounds, upper_bounds

    @property
    def expected_result(self):
        """Expected optimal solution: (2.0, 3.0, 2.0)."""
        return jnp.array([2.0, 3.0, 2.0], dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0, dtype=jnp.float64)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None
