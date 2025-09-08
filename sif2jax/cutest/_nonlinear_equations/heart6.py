import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HEART6(AbstractNonlinearEquations):
    """Dipole model of the heart (6 x 6 version).

    Source:
    J. E. Dennis, Jr., D. M. Gay, P. A. Vu,
    "A New Nonlinear Equations Test Problem".
    Tech. Rep. 83-16, Dept. of Math. Sci., Rice Univ., Houston, TX
    June 1983, revised May 1985.

    SIF input: A.R. Conn, May 1993.

    Classification: NOR2-MN-6-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Pre-compute constants as class attributes for efficiency
    _CONSTANTS = jnp.array([-1.826, -0.754, -4.839, -3.259, -14.023, 15.467])
    _SUM_MX = -0.816
    _SUM_MY = -0.017

    def residual(self, y, args):
        del args

        # Extract individual variables
        a, c, t, u, v, w = y

        # Precompute common terms for efficiency
        t_sq = t * t
        u_sq = u * u
        v_sq = v * v
        w_sq = w * w
        tv_diff_sq = t_sq - v_sq
        uw_diff_sq = u_sq - w_sq
        mx_minus_a = self._SUM_MX - a
        my_minus_c = self._SUM_MY - c
        tv_prod = t * v
        uw_prod = u * w

        # Compute groups directly without intermediate arrays
        # This reduces memory allocation overhead

        # G1 = E1 + E2 - E3 - E4
        G1 = t * a + u * mx_minus_a - v * c - w * my_minus_c

        # G2 = E5 + E6 + E7 + E8
        G2 = v * a + w * mx_minus_a + t * c + u * my_minus_c

        # G3 = E9 - 2*E10 + E11 - 2*E12
        G3 = (
            a * tv_diff_sq
            - 2 * c * tv_prod
            + mx_minus_a * uw_diff_sq
            - 2 * my_minus_c * uw_prod
        )

        # G4 = E13 + 2*E14 + E15 + 2*E16
        G4 = (
            c * tv_diff_sq
            + 2 * a * tv_prod
            + my_minus_c * uw_diff_sq
            + 2 * mx_minus_a * uw_prod
        )

        # G5 = E17 + E18 + E19 + E20
        G5 = (
            a * t * (t_sq - 3 * v_sq)
            + c * v * (v_sq - 3 * t_sq)
            + mx_minus_a * u * (u_sq - 3 * w_sq)
            + my_minus_c * w * (w_sq - 3 * u_sq)
        )

        # G6 = E21 - E22 + E23 - E24
        G6 = (
            c * t * (t_sq - 3 * v_sq)
            - a * v * (v_sq - 3 * t_sq)
            + my_minus_c * u * (u_sq - 3 * w_sq)
            - mx_minus_a * w * (w_sq - 3 * u_sq)
        )

        # Stack results and apply constants in one operation
        return jnp.stack([G1, G2, G3, G4, G5, G6]) - self._CONSTANTS

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

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
        """For nonlinear equations, objective is always zero."""
        return jnp.array(0.0)

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def n_var(self):
        return jnp.array(6)

    @property
    def n_con(self):
        return jnp.array(6)
