import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class DUALC8(AbstractConstrainedQuadraticProblem):
    """A dual quadratic program from Antonio Frangioni.

    This is the dual of PRIMALC8.SIF

    References:
    - Problem provided by Antonio Frangioni (frangio@DI.UniPi.IT)
    - SIF input: Irv Lustig and Nick Gould, June 1996

    Classification: QLR2-MN-8-503
    - QLR2: Quadratic objective, linear constraints
    - MN: General constraints
    - 8 variables, 503 constraint(s)
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
        return 503

    # Linear objective coefficients
    c = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.5890148067,
            32.720735483,
            3879.4127355,
            29480.193955,
            3879.4127355,
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
            178836.0,
            178836.0,
            178836.0,
            178836.0,
            178836.0,
            179790.0,
            179790.0,
            149678.0,
            149678.0,
            78662.0,
            78662.0,
            462746.0,
            462746.0,
            -32271.0,
            -32271.0,
            178836.0,
            178836.0,
            178836.0,
            179790.0,
            179790.0,
            149678.0,
            149678.0,
            78662.0,
            78662.0,
            462746.0,
            462746.0,
            -32271.0,
            -32271.0,
            178836.0,
            179790.0,
            179790.0,
            149678.0,
            149678.0,
            78662.0,
            78662.0,
            462746.0,
            462746.0,
            -32271.0,
            -32271.0,
            180848.0,
            151248.0,
            151248.0,
            79164.0,
            79164.0,
            461894.0,
            461894.0,
            -31137.0,
            -31137.0,
            182553.0,
            133438.0,
            133438.0,
            569115.0,
            569115.0,
            -41722.0,
            -41722.0,
            1323220.0,
            2334540.0,
            2334540.0,
            -875455.0,
            -875455.0,
            5703880.0,
            -1509560.0,
            -1509560.0,
            668378.0,
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
