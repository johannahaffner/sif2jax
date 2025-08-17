"""
CMPC10 - Model Predictive Control Problem

A convex quadratic program arising from a model predictive
control application in process control from the OPTEC Workshop
on Large Scale Convex Quadratic Programming (2010).

SIF input: Nick Gould, December 2010

Classification: QLR2-AY-V-C
Variables: 1530
Constraints: 2351
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from ..._problem import AbstractConstrainedQuadraticProblem


def _load_cmpc10_data():
    """Load problem data from NPZ file."""
    data_file = Path(__file__).parent / "data" / "cmpc10.npz"
    return np.load(data_file)


# Load data once at module level
_CMPC10_DATA = _load_cmpc10_data()

# Pre-compute sparse constraint matrices at module level
_n_vars = int(_CMPC10_DATA["n_vars"])
_n_eq = int(_CMPC10_DATA["n_eq"])
_n_ineq = int(_CMPC10_DATA["n_ineq"])

# Equality constraint matrix
if _n_eq > 0:
    _Aeq_indices = jnp.stack(
        [_CMPC10_DATA["Aeq_rows"], _CMPC10_DATA["Aeq_cols"]], axis=1
    )
    _Aeq_sparse = sparse.BCOO(
        (_CMPC10_DATA["Aeq_vals"], _Aeq_indices), shape=(_n_eq, _n_vars)
    )
    _beq = jnp.array(_CMPC10_DATA["beq"])
else:
    _Aeq_sparse = None
    _beq = None

# Inequality constraint matrix
if _n_ineq > 0:
    _Aineq_indices = jnp.stack(
        [_CMPC10_DATA["Aineq_rows"], _CMPC10_DATA["Aineq_cols"]], axis=1
    )
    _Aineq_sparse = sparse.BCOO(
        (_CMPC10_DATA["Aineq_vals"], _Aineq_indices), shape=(_n_ineq, _n_vars)
    )
    _bineq = jnp.array(_CMPC10_DATA["bineq"])
else:
    _Aineq_sparse = None
    _bineq = None


class CMPC10(AbstractConstrainedQuadraticProblem):
    """CMPC10 Model Predictive Control problem.

    A convex quadratic program from process control applications.

    Classification: QLR2-AY-V-C
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return int(_CMPC10_DATA["n_vars"])

    @property
    def m(self):
        """Number of constraints."""
        return int(_CMPC10_DATA["n_cons"])

    def objective(self, y, args):
        """Compute the objective function value.

        Computes: 0.5 * y^T Q y + c^T y
        """
        del args
        data = _CMPC10_DATA

        # Quadratic term using COO format
        Q_vals = data["Q_vals"]
        Q_rows = data["Q_rows"]
        Q_cols = data["Q_cols"]

        quad_vals = Q_vals * y[Q_rows] * y[Q_cols]
        quad_term = 0.5 * jnp.sum(quad_vals)

        # Linear term (sparse)
        c_indices = data["c_indices"]
        c_values = data["c_values"]
        lin_term = jnp.sum(c_values * y[c_indices]) if len(c_indices) > 0 else 0.0

        return quad_term + lin_term

    def constraint(self, y):
        """Compute the constraint values.

        Returns tuple (equalities, inequalities) where:
        - equalities: Aeq*y - beq = 0
        - inequalities: Aineq*y - bineq <= 0
        """
        # Use pre-computed sparse matrices from module level
        equalities = _Aeq_sparse @ y - _beq if _Aeq_sparse is not None else None
        inequalities = _Aineq_sparse @ y - _bineq if _Aineq_sparse is not None else None

        return equalities, inequalities

    def equality_constraints(self):
        """Return boolean mask for equality constraints."""
        data = _CMPC10_DATA
        n_eq = int(data["n_eq"])

        # Create mask: True for equalities, False for inequalities
        # Assuming constraints are ordered as [inequalities, equalities]
        mask = jnp.zeros(self.m, dtype=bool)
        if n_eq > 0:
            # Last n_eq constraints are equalities
            mask = mask.at[-n_eq:].set(True)
        return mask

    @property
    def bounds(self):
        """Return variable bounds."""
        data = _CMPC10_DATA

        # Reconstruct bounds from sparse representation
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        if len(data["lb_indices"]) > 0:
            lower = lower.at[data["lb_indices"]].set(data["lb_values"])
        if len(data["ub_indices"]) > 0:
            upper = upper.at[data["ub_indices"]].set(data["ub_values"])

        return lower, upper

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not available)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not available)."""
        return None
