import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class BIGGS5(AbstractBoundedMinimisation):
    """Biggs EXP problem in 5 variables.

    Source: Problem 74 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification SXR2-AN-6-0

    This function is a nonlinear least squares with 13 groups. Each
    group has 3 nonlinear elements. It is obtained by fixing
        X6 = 3
    in BIGGS6.

    The number of groups can be varied, but should be larger or equal
    to the number of variables.
    """

    @property
    def name(self) -> str:
        return "BIGGS5"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 6  # 6 variables including fixed X6
    m: int = 13  # Number of groups (residuals)

    @property
    def y0(self):
        # All 6 variables including X6 fixed at 3.0
        return jnp.array([1.0, 2.0, 1.0, 1.0, 4.0, 3.0])

    @property
    def args(self):
        return None

    def _pexp(self, v1, v2, t):
        """Parametric product with exponential element function"""
        return v1 * jnp.exp(t * v2)

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5, x6 = y

        # Vectorized computation for all groups
        i_vals = jnp.arange(1, self.m + 1, dtype=jnp.float64)
        ti_vals = 0.1 * i_vals

        # Compute yi values from the constants - vectorized
        emti_vals = jnp.exp(-ti_vals)
        e2_vals = jnp.exp(-i_vals)
        e3_vals = jnp.exp(-4.0 * ti_vals)
        y_vals = emti_vals - 5.0 * e2_vals + 3.0 * e3_vals

        # Compute the residuals - vectorized
        a_vals = self._pexp(x3, x1, -ti_vals)
        b_vals = self._pexp(x4, x2, -ti_vals)
        c_vals = self._pexp(x6, x5, -ti_vals)

        residuals = a_vals - b_vals + c_vals - y_vals
        obj = jnp.sum(residuals * residuals)

        return obj

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        # This should be verified through optimization
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 137),
        # the optimal objective value is 0.0
        return jnp.array(0.0)

    @property
    def bounds(self):
        # X1 to X5 are free, X6 is fixed at 3.0
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 3.0])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 3.0])
        return lower, upper
