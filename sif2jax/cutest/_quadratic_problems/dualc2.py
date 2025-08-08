import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class DUALC2(AbstractConstrainedQuadraticProblem):
    """A dual quadratic program from Antonio Frangioni.

    This is the dual of PRIMALC2.SIF

    References:
    - Problem provided by Antonio Frangioni (frangio@DI.UniPi.IT)
    - SIF input: Irv Lustig and Nick Gould, June 1996

    Classification: QLR2-MN-7-229
    - QLR2: Quadratic objective, linear constraints
    - MN: General constraints
    - 7 variables, 229 constraint(s)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 7

    @property
    def m(self):
        """Number of constraints."""
        return 229

    # Linear objective coefficients
    c = jnp.array(
        [
            0.0,
            2.7078336115,
            41008.309175,
            3406.2993615,
            11325.133357,
            239354.78723,
            11004.893252,
        ]
    )

    # Quadratic matrix in COO format (row, col, value)
    # Total non-zero entries: 49
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
            2,
            2,
            3,
            2,
            4,
            2,
            5,
            2,
            6,
            3,
            3,
            4,
            3,
            5,
            3,
            6,
            4,
            4,
            5,
            4,
            6,
            5,
            5,
            6,
            6,
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
            2,
            3,
            2,
            4,
            2,
            5,
            2,
            6,
            2,
            3,
            4,
            3,
            5,
            3,
            6,
            3,
            4,
            5,
            4,
            6,
            4,
            5,
            6,
            5,
            6,
        ],
        dtype=jnp.int32,
    )
    Q_val = jnp.array(
        [
            12465.0,
            14562.0,
            14562.0,
            41412.0,
            41412.0,
            -3984.0,
            -3984.0,
            8262.0,
            8262.0,
            62178.0,
            62178.0,
            -20577.0,
            -20577.0,
            17708.0,
            44808.0,
            44808.0,
            -1278.0,
            -1278.0,
            10878.0,
            10878.0,
            74380.0,
            74380.0,
            -22456.0,
            -22456.0,
            164408.0,
            -41278.0,
            -41278.0,
            4578.0,
            4578.0,
            158680.0,
            158680.0,
            -70106.0,
            -70106.0,
            31161.0,
            24199.0,
            24199.0,
            37662.0,
            37662.0,
            6221.0,
            6221.0,
            39937.0,
            120170.0,
            120170.0,
            -23267.0,
            -23267.0,
            492812.0,
            -127852.0,
            -127852.0,
            42338.0,
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
