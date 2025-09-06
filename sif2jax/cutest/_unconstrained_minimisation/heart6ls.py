import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HEART6LS(AbstractUnconstrainedMinimisation):
    """Dipole model of the heart (6 x 6 version, least squares).

    Source:
    J. E. Dennis, Jr., D. M. Gay, P. A. Vu,
    "A New Nonlinear Equations Test Problem".
    Tech. Rep. 83-16, Dept. of Math. Sci., Rice Univ., Houston, TX
    June 1983, revised May 1985.

    SIF input: A.R. Conn, May 1993.
               correction by Ph. Shott, January, 1995.

    Classification: SUR2-MN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Extract individual variables
        a, c, t, u, v, w = y

        # Constants from the SIF file (vectorized)
        constants = jnp.array([-1.826, -0.754, -4.839, -3.259, -14.023, 15.467])
        sum_Mx, sum_My = -0.816, -0.017

        # Precompute common terms for efficiency
        t_sq, u_sq, v_sq, w_sq = t**2, u**2, v**2, w**2
        tv_diff_sq = t_sq - v_sq
        uw_diff_sq = u_sq - w_sq
        tv_prod = t * v
        uw_prod = u * w
        mx_minus_a = sum_Mx - a
        my_minus_c = sum_My - c

        # Vectorized element computations grouped by type
        # 2PROD elements: [E1, E3, E5, E7]
        prod2 = jnp.array([t * a, v * c, v * a, t * c])

        # VPV elements: [E2, E4, E6, E8]
        vpv = jnp.array(
            [u * mx_minus_a, w * my_minus_c, w * mx_minus_a, u * my_minus_c]
        )

        # ADFSQ/PDFSQ elements: [E9, E11, E13, E15]
        adfsq = jnp.array(
            [
                a * tv_diff_sq,
                mx_minus_a * uw_diff_sq,
                c * tv_diff_sq,
                my_minus_c * uw_diff_sq,
            ]
        )

        # 3PROD elements: [E10, E14]
        prod3 = jnp.array([c * tv_prod, a * tv_prod])

        # P3PRD elements: [E12, E16]
        p3prd = jnp.array([my_minus_c * uw_prod, mx_minus_a * uw_prod])

        # 3DPRD elements: [E17, E18, E21, E22]
        dprd3 = jnp.array(
            [
                a * t * (t_sq - 3 * v_sq),
                c * v * (v_sq - 3 * t_sq),
                c * t * (t_sq - 3 * v_sq),
                a * v * (v_sq - 3 * t_sq),
            ]
        )

        # D3PRD elements: [E19, E20, E23, E24]
        d3prd = jnp.array(
            [
                mx_minus_a * u * (u_sq - 3 * w_sq),
                my_minus_c * w * (w_sq - 3 * u_sq),
                my_minus_c * u * (u_sq - 3 * w_sq),
                mx_minus_a * w * (w_sq - 3 * u_sq),
            ]
        )

        # Reorganize for proper group calculation
        groups = jnp.array(
            [
                prod2[0] + vpv[0] - prod2[1] - vpv[1],  # G1
                prod2[2] + vpv[2] + prod2[3] + vpv[3],  # G2
                adfsq[0] - 2 * prod3[0] + adfsq[1] - 2 * p3prd[0],  # G3
                adfsq[2] + 2 * prod3[1] + adfsq[3] + 2 * p3prd[1],  # G4
                dprd3[0] + dprd3[1] + d3prd[0] + d3prd[1],  # G5
                dprd3[2] - dprd3[3] + d3prd[2] - d3prd[3],  # G6
            ]
        )

        # Apply constants and compute sum of squares in one operation
        residuals = groups - constants
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        # Initial values from SIF file: a=0, c=0, others=1
        return jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The solution isn't explicitly provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file, the objective value at the minimum is 0.0
        return jnp.array(0.0)
