import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class DUALC5(AbstractConstrainedQuadraticProblem):
    """A dual quadratic program from Antonio Frangioni.

    This is the dual of PRIMALC5.SIF

    References:
    - Problem provided by Antonio Frangioni (frangio@DI.UniPi.IT)
    - SIF input: Irv Lustig and Nick Gould, June 1996

    Classification: QLR2-MN-8-278
    - QLR2: Quadratic objective, linear constraints
    - MN: General constraints
    - 8 variables, 278 constraint(s)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 8

    @property
    def m(self):
        """Number of constraints."""
        return 278

    # Linear objective coefficients
    c = jnp.array(
        [
            546.88509078,
            0.0,
            122.12095223,
            616.11938925,
            586.52942284,
            586.52942284,
            953.05364036,
            1585.0340121,
        ]
    )

    # Quadratic matrix in COO format (row, col, value)
    # Total non-zero entries: 64
    Q_row = jnp.array(
        [
            0,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
            5,
            0,
            6,
            0,
            7,
            1,
            1,
            2,
            1,
            3,
            1,
            4,
            1,
            5,
            1,
            6,
            1,
            7,
            2,
            2,
            3,
            2,
            4,
            2,
            5,
            2,
            6,
            2,
            7,
            3,
            3,
            4,
            3,
            5,
            3,
            6,
            3,
            7,
            4,
            4,
            5,
            4,
            6,
            4,
            7,
            5,
            5,
            6,
            5,
            7,
            6,
            6,
            7,
            7,
        ],
        dtype=jnp.int32,
    )
    Q_col = jnp.array(
        [
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
            5,
            0,
            6,
            0,
            7,
            0,
            1,
            2,
            1,
            3,
            1,
            4,
            1,
            5,
            1,
            6,
            1,
            7,
            1,
            2,
            3,
            2,
            4,
            2,
            5,
            2,
            6,
            2,
            7,
            2,
            3,
            4,
            3,
            5,
            3,
            6,
            3,
            7,
            3,
            4,
            5,
            4,
            6,
            4,
            7,
            4,
            5,
            6,
            5,
            7,
            5,
            6,
            7,
            6,
            7,
        ],
        dtype=jnp.int32,
    )
    Q_val = jnp.array(
        [
            13053.0,
            2524.0,
            2524.0,
            869.0,
            869.0,
            2342.0,
            2342.0,
            -4967.0,
            -4967.0,
            2742.0,
            2742.0,
            12580.0,
            12580.0,
            -1574.0,
            -1574.0,
            9564.0,
            4394.0,
            4394.0,
            16968.0,
            16968.0,
            -6110.0,
            -6110.0,
            1727.0,
            1727.0,
            5191.0,
            5191.0,
            -1101.0,
            -1101.0,
            11069.0,
            15583.0,
            15583.0,
            -5984.0,
            -5984.0,
            1344.0,
            1344.0,
            392.0,
            392.0,
            4540.0,
            4540.0,
            54824.0,
            -10447.0,
            -10447.0,
            -3459.0,
            -3459.0,
            7868.0,
            7868.0,
            -6775.0,
            -6775.0,
            7219.0,
            -4378.0,
            -4378.0,
            -7789.0,
            -7789.0,
            -1994.0,
            -1994.0,
            6429.0,
            6769.0,
            6769.0,
            4433.0,
            4433.0,
            44683.0,
            -19349.0,
            -19349.0,
            22577.0,
        ]
    )

    def objective(self, y, args):
        """Quadratic objective: 0.5 * y^T * Q * y + c^T * y"""
        del args

        # Linear term
        linear_term = jnp.dot(self.c, y)

        # Quadratic term using sparse representation
        quad_term = jnp.sum(self.Q_val * y[self.Q_row] * y[self.Q_col])

        return 0.5 * quad_term + linear_term

    def constraint(self, y):
        """Linear equality constraint: sum(y) = 1"""
        constraints = jnp.array([jnp.sum(y) - 1.0])
        return constraints, None

    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)

    @property
    def bounds(self):
        """Bound constraints: 0 <= y <= 1"""
        lbs = jnp.zeros(self.n)
        ubs = jnp.ones(self.n)
        return lbs, ubs

    @property
    def y0(self):
        """Initial point: zeros"""
        return inexact_asarray(jnp.zeros(self.n))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not known analytically)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not known analytically)."""
        return None
