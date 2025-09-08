"""A0NNSNSL quadratic programming problem.

A quadratic programming reformulation of a linear complementarity
problem provided by Michael Ferris. Part of the A0N series.

Classification: QLR2-AN-60012-20004
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from sif2jax import AbstractConstrainedQuadraticProblem


# Load parsed data at module level
_DATA_PATH = Path(__file__).parent / "data" / "a0nnsnsl_full.npz"
_PROBLEM_DATA = np.load(_DATA_PATH)

# Constraint matrix (sparse representation)
_con_rows = _PROBLEM_DATA["con_rows"]
_con_cols = _PROBLEM_DATA["con_cols"]
_con_vals = _PROBLEM_DATA["con_vals"]

# Create sparse constraint matrix A
_A_SPARSE = sparse.BCOO(
    (_con_vals, jnp.column_stack([_con_rows, _con_cols])),
    shape=(20004, 60012),
)

# RHS vector
_RHS = jnp.array(_PROBLEM_DATA["rhs"], dtype=jnp.float64)

# Bounds - trim to actual variable count
_LOWER = jnp.array(_PROBLEM_DATA["lower"][:60012], dtype=jnp.float64)
_UPPER = jnp.array(_PROBLEM_DATA["upper"][:60012], dtype=jnp.float64)

# Quadratic objective terms (sparse representation)
_QUAD_I = jnp.array(_PROBLEM_DATA["quad_i"], dtype=jnp.int32)
_QUAD_J = jnp.array(_PROBLEM_DATA["quad_j"], dtype=jnp.int32)
_QUAD_VALS = jnp.array(_PROBLEM_DATA["quad_vals"], dtype=jnp.float64)

# Filter out any indices >= n_var (safety check)
_valid_mask = (_QUAD_I < 60012) & (_QUAD_J < 60012)
_QUAD_I = _QUAD_I[_valid_mask]
_QUAD_J = _QUAD_J[_valid_mask]
_QUAD_VALS = _QUAD_VALS[_valid_mask]


class A0NNSNSL(AbstractConstrainedQuadraticProblem):
    """A0NNSNSL quadratic programming problem.

    A quadratic programming reformulation of a linear complementarity
    problem provided by Michael Ferris. Part of the A0N series.

    Classification: QLR2-AN-60012-20004
    - QLR2: Quadratic objective, Linear constraints, Regular problem,
      2nd derivatives available
    - AN: Analytical problem
    - 60012 variables
    - 20004 constraints
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(60012)

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

    def gradient(self, y, args):
        """Analytical gradient of quadratic objective.

        For Q(x) = 0.5 * x^T * M * x where M is symmetric,
        gradient = M * x

        Since we store upper triangular terms (i > j), each term
        contributes to both gradient[i] and gradient[j].
        """
        del args

        # Initialize gradient
        grad = jnp.zeros_like(y)

        # Each quadratic term val * x[i] * x[j] contributes:
        # - val * x[j] to gradient[i]
        # - val * x[i] to gradient[j]

        # Vectorized accumulation using scatter operations
        grad = grad.at[_QUAD_I].add(_QUAD_VALS * y[_QUAD_J])
        grad = grad.at[_QUAD_J].add(_QUAD_VALS * y[_QUAD_I])

        return grad

    def hessian(self, y, args):
        """Analytical Hessian of quadratic objective.

        For a quadratic function, the Hessian is constant and equals
        the quadratic coefficient matrix M.
        """
        del y, args

        # Build symmetric matrix from upper triangular terms
        # Each term (i, j, val) with i > j contributes to both (i,j) and (j,i)
        rows = jnp.concatenate([_QUAD_I, _QUAD_J])
        cols = jnp.concatenate([_QUAD_J, _QUAD_I])
        vals = jnp.concatenate([_QUAD_VALS, _QUAD_VALS])

        # Create sparse Hessian matrix
        hessian_sparse = sparse.BCOO(
            (vals, jnp.column_stack([rows, cols])), shape=(self.n, self.n)
        )

        return hessian_sparse

    def constraint_jacobian(self, y):
        """Analytical Jacobian of linear constraints.

        For linear constraints Ax = b, the Jacobian is simply A.
        """
        del y  # Jacobian is constant for linear constraints
        return _A_SPARSE, None
