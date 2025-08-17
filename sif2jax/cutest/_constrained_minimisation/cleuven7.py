"""CLEUVEN7 problem with pre-computed data from NPZ file."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


# Load problem data at module level
_data_file = Path(__file__).parent / "data" / "cleuven7.npz"
_data = np.load(_data_file)

# Convert numpy arrays to JAX arrays for consistency with existing code
_obj_idx = jnp.array(_data["obj_idx"])
_obj_val = jnp.array(_data["obj_val"])
_eq_rows = jnp.array(_data["eq_rows"])
_eq_cols = jnp.array(_data["eq_cols"])
_eq_vals = jnp.array(_data["eq_vals"])
_ineq_rows = jnp.array(_data["ineq_rows"])
_ineq_cols = jnp.array(_data["ineq_cols"])
_ineq_vals = jnp.array(_data["ineq_vals"])
_quad_rows = jnp.array(_data["quad_rows"])
_quad_cols = jnp.array(_data["quad_cols"])
_quad_vals = jnp.array(_data["quad_vals"])
_lower_bounds = jnp.array(_data["lower_bounds"])
_upper_bounds = jnp.array(_data["upper_bounds"])
_eq_rhs = jnp.array(_data["eq_rhs"])
_ineq_rhs = jnp.array(_data["ineq_rhs"])
_n_eq = int(_data["n_eq"])
_n_ineq = int(_data["n_ineq"])

# Build dense inequality matrix (10.5% density is too dense for sparse operations)
_ineq_matrix_dense = jnp.zeros((_n_ineq, 360))
for i in range(len(_ineq_rows)):
    _ineq_matrix_dense = _ineq_matrix_dense.at[_ineq_rows[i], _ineq_cols[i]].add(
        _ineq_vals[i]
    )


class CLEUVEN7(AbstractConstrainedMinimisation):
    """A convex quadratic program from model predictive control.

    Problem from the OPTEC Workshop on Large Scale Convex Quadratic
    Programming - Algorithms, Software, and Applications, Leuven,
    25-26/10/2010.

    References:
        SIF input: Nick Gould, December 2010
        Corrected version: May 2019

    Classification: QLR2-RN-300-946
    """

    n_var: int = eqx.field(default=360, init=False)
    n_con: int = eqx.field(init=False)  # Will be set dynamically
    provided_y0s: frozenset = frozenset({0})
    y0_iD: int = 0

    def __init__(self):
        """Load exact CLEUVEN7 data from NPZ file."""
        self.n_con = _n_eq + _n_ineq

    @property
    def y0(self) -> Float[Array, "360"]:
        """Initial point - zeros for QP problems."""
        return jnp.zeros(self.n_var)

    @property
    def xlb(self) -> Float[Array, "360"]:
        """Lower bounds on variables."""
        return _lower_bounds

    @property
    def xub(self) -> Float[Array, "360"]:
        """Upper bounds on variables."""
        return _upper_bounds

    def objective(self, y: Float[Array, "360"], args=None) -> Float[Array, ""]:
        """Quadratic objective function.

        f(x) = 0.5 * x^T * H * x + c^T * x
        """
        # Linear term
        linear_term = jnp.sum(_obj_val * y[_obj_idx])

        # Quadratic term using sparse representation
        # For (i,j,v) triplets, compute sum of v * x[i] * x[j]
        quad_term = jnp.sum(_quad_vals * y[_quad_rows] * y[_quad_cols])

        # Note: diagonal terms should be multiplied by 0.5
        # Off-diagonal terms appear twice (symmetric matrix)
        # Adjust for proper quadratic form
        diag_mask = (_quad_rows == _quad_cols).astype(jnp.float64)
        diag_adjustment = -0.5 * jnp.sum(
            _quad_vals * diag_mask * y[_quad_rows] * y[_quad_cols]
        )

        return linear_term + quad_term + diag_adjustment

    def constraint(self, y: Float[Array, "360"], args=None):
        """Linear equality and inequality constraints.

        Returns (equalities, inequalities) where:
        - equalities: Ax = b (returned as Ax - b = 0)
        - inequalities: Ax <= b (returned as Ax - b <= 0)
        """
        # No equality constraints in CLEUVEN7
        equalities = None

        # Inequality constraints: Ax <= b
        # Use dense matrix multiplication (5x faster than sparse for 10.5% density)
        inequalities = _ineq_matrix_dense @ y - _ineq_rhs

        return equalities, inequalities

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        return self.xlb, self.xub

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # From SIF file
        return jnp.array(706.5690133723616)
