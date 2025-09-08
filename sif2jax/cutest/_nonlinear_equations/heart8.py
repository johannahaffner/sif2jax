import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HEART8(AbstractNonlinearEquations):
    """Dipole model of the heart (8 x 8 version).

    Source:
    J. E. Dennis, Jr., D. M. Gay, P. A. Vu,
    "A New Nonlinear Equations Test Problem".
    Tech. Rep. 83-16, Dept. of Math. Sci., Rice Univ., Houston, TX
    June 1983, revised May 1985.

    SIF input: A.R. Conn, May 1993.
               correction by Ph. Shott, January, 1995.

    Classification: NOR2-MN-8-8
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args):
        del args

        # Extract individual variables
        a, b, c, d, t, u, v, w = y

        # Constants from the SIF file (vectorized)
        constants = jnp.array([-1.57, -1.31, -2.65, 2.0, -12.6, 9.48])
        sum_Mx, sum_My = -0.69, -0.044

        # Based on HEART8LS pattern but for NLE formulation
        # G1 = a + b - sum_Mx
        # G2 = c + d - sum_My
        # G3 = t*a + u*b - v*c - w*d - sum_A
        # G4 = v*a + w*b + t*c + u*d - sum_B
        # G5 = a*(t*t - v*v) + b*(u*u - w*w) - 2*c*t*v - 2*d*u*w - sum_C
        # G6 = c*(t*t - v*v) + d*(u*u - w*w) + 2*a*t*v + 2*b*u*w - sum_D
        # G7 = a*t*(t*t - 3*v*v) + b*u*(u*u - 3*w*w) + c*v*(v*v - 3*t*t) +
        #      d*w*(w*w - 3*u*u) - sum_E
        # G8 = c*t*(t*t - 3*v*v) + d*u*(u*u - 3*w*w) - a*v*(v*v - 3*t*t) -
        #      b*w*(w*w - 3*u*u) - sum_F

        # Precompute common terms for efficiency
        t_sq, u_sq, v_sq, w_sq = t**2, u**2, v**2, w**2

        # Vectorized computation of residuals
        residuals = jnp.array(
            [
                a + b - sum_Mx,
                c + d - sum_My,
                t * a + u * b - v * c - w * d - constants[0],
                v * a + w * b + t * c + u * d - constants[1],
                a * (t_sq - v_sq)
                + b * (u_sq - w_sq)
                - 2 * c * t * v
                - 2 * d * u * w
                - constants[2],
                c * (t_sq - v_sq)
                + d * (u_sq - w_sq)
                + 2 * a * t * v
                + 2 * b * u * w
                - constants[3],
                a * t * (t_sq - 3 * v_sq)
                + b * u * (u_sq - 3 * w_sq)
                + c * v * (v_sq - 3 * t_sq)
                + d * w * (w_sq - 3 * u_sq)
                - constants[4],
                c * t * (t_sq - 3 * v_sq)
                + d * u * (u_sq - 3 * w_sq)
                - a * v * (v_sq - 3 * t_sq)
                - b * w * (w_sq - 3 * u_sq)
                - constants[5],
            ]
        )

        return residuals

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def y0(self):
        # Initial values from SIF file: a=0, c=0, others=1
        return jnp.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The solution isn't explicitly provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        """For nonlinear equations, objective is always zero."""
        return jnp.array(0.0)

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def n_var(self):
        return jnp.array(8)

    @property
    def n_con(self):
        return jnp.array(8)
