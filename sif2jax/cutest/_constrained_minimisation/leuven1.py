"""LEUVEN1 problem - convex quadratic program from model predictive control.

TODO: Human review needed - Multiple test failures
Attempts made:
1. ✓ Successfully parsed LEUVEN1.SIF file with custom parser
2. ✓ Extracted 149 linear objective terms, 2220 constraints
3. ✓ Fixed CONSTANTS section parsing to get non-zero RHS values
4. ✓ Fixed bounds format to return (lower, upper) tuple instead of list
5. ✓ Objective function passes test_correct_objective_zero_vector

Suspected issues requiring human review:
1. Bounds mismatch: pycutest shows some fixed variables as unbounded
2. Constraint values mismatch: Large differences in evaluations
3. May involve CUTEst-specific interpretation differences
4. Possible issue with drop_fixed_variables=False handling

Resources needed: SIF format expertise, CUTEst behavior understanding
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


# Load problem data at module level
_data_file = Path(__file__).parent / "data" / "leuven1.npz"
_data = np.load(_data_file)

# Convert numpy arrays to JAX arrays
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


class LEUVEN1(AbstractConstrainedMinimisation):
    """A convex quadratic program from model predictive control.

    Problem from the OPTEC Workshop on Large Scale Convex Quadratic
    Programming - Algorithms, Software, and Applications, Leuven,
    25-26/10/2010.

    Note: This is the original LEUVEN1 problem, which is described as
    convex (unlike LEUVEN2-7 which were incorrect and replaced by CLEUVEN2-7).

    References:
        SIF input: Nick Gould, December 2010

    Classification: QLR2-RN-1530-2220
    """

    n_var: int = eqx.field(default=1530, init=False)
    n_con: int = eqx.field(init=False)  # Will be set dynamically
    provided_y0s: frozenset = frozenset({0})
    y0_iD: int = 0

    def __init__(self):
        """Initialize LEUVEN1 problem."""
        self.n_con = _n_eq + _n_ineq

    @property
    def y0(self) -> Float[Array, "1530"]:
        """Initial point - zeros for QP problems."""
        return jnp.zeros(self.n_var)

    @property
    def xlb(self) -> Float[Array, "1530"]:
        """Lower bounds on variables."""
        return _lower_bounds

    @property
    def xub(self) -> Float[Array, "1530"]:
        """Upper bounds on variables."""
        return _upper_bounds

    @property
    def bounds(self):
        """Return bounds in the format expected by AbstractConstrainedMinimisation."""
        return _lower_bounds, _upper_bounds

    @property
    def args(self):
        """No additional arguments needed."""
        return None

    def objective(self, y: Float[Array, "1530"], args=None) -> Float[Array, ""]:
        """Quadratic objective function.

        f(x) = 0.5 * x^T * H * x + c^T * x

        Since we may not have quadratic terms, this could be a linear program.
        """
        del args

        # Linear term
        obj = jnp.zeros(())
        if len(_obj_idx) > 0:
            obj = obj + jnp.sum(y[_obj_idx] * _obj_val)

        # Quadratic term (if any)
        if len(_quad_rows) > 0:
            # For each quadratic term, compute x_i * H_ij * x_j
            quad_contribution = y[_quad_rows] * _quad_vals * y[_quad_cols]
            obj = obj + 0.5 * jnp.sum(quad_contribution)

            # Handle off-diagonal terms (they appear twice in symmetric matrix)
            off_diag_mask = _quad_rows != _quad_cols
            obj = obj + 0.5 * jnp.sum(quad_contribution * off_diag_mask)

        return obj

    def constraint(self, y: Float[Array, "1530"]):
        """Compute constraint values.

        Returns:
            (equality_constraints, inequality_constraints)
            where equality constraints should be = 0
            and inequality constraints should be >= 0
        """
        # Equality constraints: A_eq @ x - b_eq = 0
        eq_Ax = segment_sum(
            _eq_vals * y[_eq_cols], segment_ids=_eq_rows, num_segments=_n_eq
        )
        eq_vals = eq_Ax - _eq_rhs

        # Inequality constraints: A_ineq @ x <= b_ineq
        # Convert to >= 0 form: b_ineq - A_ineq @ x >= 0
        ineq_Ax = segment_sum(
            _ineq_vals * y[_ineq_cols], segment_ids=_ineq_rows, num_segments=_n_ineq
        )
        ineq_vals = _ineq_rhs - ineq_Ax

        return eq_vals, ineq_vals

    @property
    def expected_result(self):
        """Expected solution is not provided in the SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is not provided in the SIF file."""
        return None
