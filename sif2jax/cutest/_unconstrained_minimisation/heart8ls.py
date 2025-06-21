import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HEART8LS(AbstractUnconstrainedMinimisation):
    """Dipole model of the heart (8 x 8 version, least squares).

    Source:
    J. E. Dennis, Jr., D. M. Gay, P. A. Vu,
    "A New Nonlinear Equations Test Problem".
    Tech. Rep. 83-16, Dept. of Math. Sci., Rice Univ., Houston, TX
    June 1983, revised May 1985.

    SIF input: A.R. Conn, May 1993.
               correction by Ph. Shott, January, 1995.

    Classification: SUR2-MN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Extract individual variables
        a, b, c, d, t, u, v, w = y

        # Constants from the SIF file
        sum_Mx = -0.69
        sum_My = -0.044
        sum_A = -1.57
        sum_B = -1.31
        sum_C = -2.65
        sum_D = 2.0
        sum_E = -12.6
        sum_F = 9.48

        # Equations from the SIF file
        # G1 = a + b - sum_Mx
        r1 = a + b - sum_Mx

        # G2 = c + d - sum_My
        r2 = c + d - sum_My

        # G3 = t*a + u*b - v*c - w*d - sum_A
        r3 = t * a + u * b - v * c - w * d - sum_A

        # G4 = v*a + w*b + t*c + u*d - sum_B
        r4 = v * a + w * b + t * c + u * d - sum_B

        # G5 = a*(t*t - v*v) + b*(u*u - w*w) - 2*c*t*v - 2*d*u*w - sum_C
        r5 = (
            a * (t * t - v * v)
            + b * (u * u - w * w)
            - 2 * c * t * v
            - 2 * d * u * w
            - sum_C
        )

        # G6 = c*(t*t - v*v) + d*(u*u - w*w) + 2*a*t*v + 2*b*u*w - sum_D
        r6 = (
            c * (t * t - v * v)
            + d * (u * u - w * w)
            + 2 * a * t * v
            + 2 * b * u * w
            - sum_D
        )

        # G7 = a*t*(t*t - 3*v*v) + b*u*(u*u - 3*w*w)
        # + c*v*(v*v - 3*t*t) + d*w*(w*w - 3*u*u) - sum_E
        r7 = (
            a * t * (t * t - 3 * v * v)
            + b * u * (u * u - 3 * w * w)
            + c * v * (v * v - 3 * t * t)
            + d * w * (w * w - 3 * u * u)
            - sum_E
        )

        # G8 = c*t*(t*t - 3*v*v) + d*u*(u*u - 3*w*w)
        # - a*v*(v*v - 3*t*t) - b*w*(w*w - 3*u*u) - sum_F
        r8 = (
            c * t * (t * t - 3 * v * v)
            + d * u * (u * u - 3 * w * w)
            - a * v * (v * v - 3 * t * t)
            - b * w * (w * w - 3 * u * u)
            - sum_F
        )

        # Sum of squared residuals (least-squares objective)
        return jnp.sum(jnp.array([r1, r2, r3, r4, r5, r6, r7, r8]) ** 2)

    def y0(self):
        # Initial values from SIF file
        # a=0, c=0, others=1
        return jnp.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The solution isn't provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file, the objective value at the minimum is 0.0
        return jnp.array(0.0)
