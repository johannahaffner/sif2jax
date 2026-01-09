"""CLEUVEN3 problem with pre-computed data from NPZ file."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


# Load problem data at module level
_data_file = Path(__file__).parent / "data" / "cleuven3.npz"
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


class CLEUVEN3(AbstractConstrainedMinimisation):
    """A nonconvex quadratic program from model predictive control.

    Problem from the OPTEC Workshop on Large Scale Convex Quadratic Programming
    - Algorithms, Software, and Applications. Leuven, 25-26/10/2010.

    References:
        SIF input: Nick Gould, December 2010
        Corrected version: May 2019

    Classification: QLR2-RN-1200-2973
    """

    n_var: int = eqx.field(default=1200, init=False)
    n_con: int = eqx.field(init=False)  # Will be set dynamically
    provided_y0s: frozenset = frozenset({0})
    y0_iD: int = 0

    def __init__(self):
        """Initialize CLEUVEN3 problem."""
        self.n_con = _n_eq + _n_ineq

    @property
    def y0(self) -> Float[Array, "1200"]:
        """Initial point - zeros for QP problems."""
        return jnp.zeros(self.n_var)

    def objective(self, y: Float[Array, "1200"], args) -> Float[Array, ""]:
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

    def constraint(self, y: Float[Array, "1200"]):
        """Linear equality and inequality constraints.

        Returns (equalities, inequalities) where:
        - equalities: Ax = b (returned as Ax - b = 0)
        - inequalities: Ax <= b (returned as Ax - b <= 0)
        """
        # Equality constraints: Ax = b
        if _n_eq > 0:
            eq_Ax = jnp.zeros(_n_eq)
            eq_Ax = segment_sum(_eq_vals * y[_eq_cols], _eq_rows, num_segments=_n_eq)
            equalities = eq_Ax - _eq_rhs
        else:
            equalities = jnp.array([], dtype=y.dtype)

        # Inequality constraints: Ax <= b
        if _n_ineq > 0:
            ineq_Ax = segment_sum(
                _ineq_vals * y[_ineq_cols], _ineq_rows, num_segments=_n_ineq
            )
            inequalities = ineq_Ax - _ineq_rhs
        else:
            inequalities = jnp.array([], dtype=y.dtype)

        return equalities, inequalities

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        return _lower_bounds, _upper_bounds

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None
