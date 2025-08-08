import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class DUALC1(AbstractConstrainedQuadraticProblem):
    """A dual quadratic program from Antonio Frangioni.

    This is the dual of PRIMALC1.SIF

    References:
    - Problem provided by Antonio Frangioni (frangio@DI.UniPi.IT)
    - SIF input: Irv Lustig and Nick Gould, June 1996

    Classification: QLR2-MN-9-215
    - QLR2: Quadratic objective, linear constraints
    - MN: General constraints
    - 9 variables, 215 constraint(s)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 9

    @property
    def m(self):
        """Number of constraints."""
        return 215

    # Linear objective coefficients
    c = jnp.array(
        [
            0.0,
            5765.7624165,
            3753.0154856,
            3753.4216509,
            11880.124847,
            29548.987048,
            423163.83666,
            3369558.8652,
            439695.6796,
        ]
    )

    # Quadratic matrix in COO format (row, col, value)
    # Total non-zero entries: 81
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
            0,
            8,
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
            1,
            8,
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
            2,
            8,
            3,
            3,
            4,
            3,
            5,
            3,
            6,
            3,
            7,
            3,
            8,
            4,
            4,
            5,
            4,
            6,
            4,
            7,
            4,
            8,
            5,
            5,
            6,
            5,
            7,
            5,
            8,
            6,
            6,
            7,
            6,
            8,
            7,
            7,
            8,
            8,
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
            8,
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
            8,
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
            8,
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
            8,
            3,
            4,
            5,
            4,
            6,
            4,
            7,
            4,
            8,
            4,
            5,
            6,
            5,
            7,
            5,
            8,
            5,
            6,
            7,
            6,
            8,
            6,
            7,
            8,
            7,
            8,
        ],
        dtype=jnp.int32,
    )
    Q_val = jnp.array(
        [
            14882.0,
            4496.0,
            4496.0,
            5258.0,
            5258.0,
            5204.0,
            5204.0,
            8407.0,
            8407.0,
            8092.0,
            8092.0,
            -42247.0,
            -42247.0,
            -116455.0,
            -116455.0,
            51785.0,
            51785.0,
            65963.0,
            -17504.0,
            -17504.0,
            -17864.0,
            -17864.0,
            -15854.0,
            -15854.0,
            -14818.0,
            -14818.0,
            -100219.0,
            -100219.0,
            -101506.0,
            -101506.0,
            25690.0,
            25690.0,
            17582.0,
            17642.0,
            17642.0,
            15837.0,
            15837.0,
            17186.0,
            17186.0,
            27045.0,
            27045.0,
            -53251.0,
            -53251.0,
            26765.0,
            26765.0,
            17738.0,
            15435.0,
            15435.0,
            16898.0,
            16898.0,
            26625.0,
            26625.0,
            -56011.0,
            -56011.0,
            27419.0,
            27419.0,
            35281.0,
            48397.0,
            48397.0,
            48427.0,
            48427.0,
            29317.0,
            29317.0,
            12170.0,
            12170.0,
            93500.0,
            5386.0,
            5386.0,
            -92344.0,
            -92344.0,
            112416.0,
            112416.0,
            1027780.0,
            1744550.0,
            1744550.0,
            -963140.0,
            -963140.0,
            5200790.0,
            -2306625.0,
            -2306625.0,
            1390020.0,
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
