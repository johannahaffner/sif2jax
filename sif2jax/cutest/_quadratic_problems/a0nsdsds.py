"""A0NSDSDS: Quadratic programming reformulation of linear complementarity problem.

Problem provided by Michael Ferris.
Classification: QLR2-AN-6012-2004
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from ..._problem import AbstractConstrainedQuadraticProblem


def _load_problem_data():
    """Load complete problem data from pre-parsed NPZ file."""
    data_file = Path(__file__).parent / "data" / "a0nsdsds_full.npz"
    return np.load(data_file)


# Load data once at module level
_PROBLEM_DATA = _load_problem_data()

# Create sparse constraint matrix
_n_con = int(_PROBLEM_DATA["n_con"])
_n_var = int(_PROBLEM_DATA["n_var"])
_indices = jnp.array(
    np.stack([_PROBLEM_DATA["con_rows"], _PROBLEM_DATA["con_cols"]], axis=1),
    dtype=jnp.int32,
)
_values = jnp.array(_PROBLEM_DATA["con_vals"], dtype=jnp.float64)
_A_SPARSE = sparse.BCOO((_values, _indices), shape=(_n_con, _n_var))

# RHS values for constraints
_RHS = jnp.array(_PROBLEM_DATA["rhs"], dtype=jnp.float64)

# Bounds - already converted from large finite values to infinity during parsing
# SIF files often use large values like 1e21 to represent infinity,
# which we convert to jnp.inf during the SIF parsing stage
_LOWER = jnp.array(_PROBLEM_DATA["lower"], dtype=jnp.float64)
_UPPER = jnp.array(_PROBLEM_DATA["upper"], dtype=jnp.float64)

# Quadratic objective terms (sparse representation)
_QUAD_I = jnp.array(_PROBLEM_DATA["quad_i"], dtype=jnp.int32)
_QUAD_J = jnp.array(_PROBLEM_DATA["quad_j"], dtype=jnp.int32)
_QUAD_VALS = jnp.array(_PROBLEM_DATA["quad_vals"], dtype=jnp.float64)


class A0NSDSDS(AbstractConstrainedQuadraticProblem):
    """A0NSDSDS quadratic programming problem.

    A quadratic programming reformulation of a linear complementarity
    problem provided by Michael Ferris.

    Classification: QLR2-AN-6012-2004
    - QLR2: Quadratic objective, Linear constraints, Regular problem,
      2nd derivatives available
    - AN: Analytical problem
    - 6012 variables
    - 2004 constraints
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 6012

    @property
    def y0(self):
        """Initial guess - all zeros from SIF."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Compute quadratic objective function.

        The quadratic form is stored in upper triangular format (i > j).
        For a symmetric quadratic form Q(x) = 0.5 * x^T * M * x, each
        off-diagonal term x_i * x_j should contribute to both positions
        (i,j) and (j,i) in the matrix.
        """
        del args

        # Since all terms have i > j (upper triangular), we need to
        # account for symmetry by doubling off-diagonal contributions
        # Q(x) = 0.5 * sum_k (2 * x[i_k] * x[j_k] * val_k) for i > j
        obj = jnp.sum(2.0 * y[_QUAD_I] * y[_QUAD_J] * _QUAD_VALS)

        # Apply 0.5 factor: 0.5 * 2.0 = 1.0
        return 0.5 * obj

    def constraint(self, y):
        """Compute linear constraint values.

        Constraints are of the form: Ax = b
        """
        # Compute Ax - b (pycutest returns the residual)
        eq_constraint = _A_SPARSE @ y - _RHS
        ineq_constraint = None

        return eq_constraint, ineq_constraint

    @property
    def bounds(self):
        """Return the variable bounds from parsed SIF data."""
        return _LOWER, _UPPER

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
