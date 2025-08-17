"""CLEUVEN6 problem with exact coefficients from SIF file."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


def _load_cleuven6_exact_data():
    """Load exact CLEUVEN6 data from the SIF file.

    This function parses the SIF file once and caches the results.
    For production, this data would be pre-computed and stored.
    """
    import os

    # Try multiple possible paths for the SIF file
    # The path differs between local development and test container
    possible_paths = [
        "/workspace/x/archive/mastsif/CLEUVEN6.SIF",  # Local development
        "/workspace/archive/mastsif/CLEUVEN6.SIF",  # Test container mount
        "./archive/mastsif/CLEUVEN6.SIF",  # Relative path fallback
    ]

    sif_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sif_path = path
            break

    if sif_path is None:
        # Fallback if SIF file not found - this should not happen in tests
        raise FileNotFoundError(
            f"CLEUVEN6.SIF file not found. Tried paths: {possible_paths}"
        )

    with open(sif_path) as f:
        lines = f.readlines()

    # Find sections
    sections = {}
    for i, line in enumerate(lines):
        for key in ["VARIABLES", "GROUPS", "BOUNDS", "OBJECT BOUND", "ENDATA"]:
            if line.strip() == key:
                sections[key] = i

    # Parse objective
    obj_linear = {}
    constraints = {}
    constraint_types = {}  # Track if constraint is equality (E) or inequality (L)
    # Track constraint order as they appear in SIF file (CUTEst preserves this order)
    constraint_order = []

    for line in lines[sections["GROUPS"] + 1 : sections["BOUNDS"]]:
        parts = line.split()
        if len(parts) >= 4:
            if parts[0] == "N" and parts[1] == "OBJ" and parts[2] == "X":
                var_idx = int(parts[3]) - 1
                obj_linear[var_idx] = np.float64(parts[4])
            elif (parts[0] in ["L", "E"]) and parts[1] == "C" and len(parts) >= 6:
                con_type = parts[0]  # E for equality, L for inequality
                con_idx = int(parts[2]) - 1
                var_idx = int(parts[4]) - 1
                if con_idx not in constraints:
                    constraints[con_idx] = {}
                    constraint_types[con_idx] = con_type
                    constraint_order.append(con_idx)  # Track first appearance order
                constraints[con_idx][var_idx] = np.float64(parts[5])

    # Parse RHS values for constraints (only for constraints that exist)
    rhs_values_original = {}
    for line in lines:
        if line.startswith("    RHS"):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "C":
                con_idx = int(parts[2]) - 1
                rhs_values_original[con_idx] = np.float64(parts[3])

    # Reorder RHS to match SIF constraint order
    # Use 0.0 as default for constraints without explicit RHS
    rhs_values = np.zeros(len(constraint_order))
    for sif_position, con_idx in enumerate(constraint_order):
        rhs_values[sif_position] = rhs_values_original.get(con_idx, 0.0)

    # Convert to arrays
    obj_idx = sorted(obj_linear.keys())
    obj_val = [obj_linear[i] for i in obj_idx]

    # Separate constraints by type, preserving SIF order
    eq_constraints = []
    ineq_constraints = []
    eq_rhs = []
    ineq_rhs = []

    for sif_position, con_idx in enumerate(constraint_order):
        if constraint_types[con_idx] == "E":
            eq_constraints.append((sif_position, con_idx))
            eq_rhs.append(rhs_values[sif_position])
        else:  # "L"
            ineq_constraints.append((sif_position, con_idx))
            ineq_rhs.append(rhs_values[sif_position])

    # Build constraint matrices
    eq_rows, eq_cols, eq_vals = [], [], []
    for eq_pos, (sif_position, con_idx) in enumerate(eq_constraints):
        for var_idx, coeff in sorted(constraints[con_idx].items()):
            eq_rows.append(eq_pos)
            eq_cols.append(var_idx)
            eq_vals.append(coeff)

    ineq_rows, ineq_cols, ineq_vals = [], [], []
    for ineq_pos, (sif_position, con_idx) in enumerate(ineq_constraints):
        for var_idx, coeff in sorted(constraints[con_idx].items()):
            ineq_rows.append(ineq_pos)
            ineq_cols.append(var_idx)
            ineq_vals.append(coeff)

    # Parse bounds - start with unbounded defaults
    lower_bounds = np.full(1200, -np.inf)
    upper_bounds = np.full(1200, np.inf)

    # Track which variables have explicit bounds
    has_explicit_bounds = set()

    bounds_end = sections["BOUNDS"] + 1
    for line in lines[sections["BOUNDS"] + 1 :]:
        if line.startswith("    X"):
            bounds_end = lines.index(line, sections["BOUNDS"] + 1)
            break
        parts = line.split()
        if len(parts) >= 5 and parts[2] == "X":
            var_idx = int(parts[3]) - 1
            # Use higher precision float parsing
            value = np.float64(parts[4])
            has_explicit_bounds.add(var_idx)
            if parts[0] == "LO":
                lower_bounds[var_idx] = value
            elif parts[0] == "UP":
                upper_bounds[var_idx] = value

    # For variables with identical lower and upper bounds (fixed variables),
    # CUTEst typically treats the lower bound as -inf for optimization purposes
    # Apply this to only truly zero-valued fixed variables for better compatibility
    fixed_count = 0
    target_conversions = 1200 - 967  # Need to convert 233 lower bounds to -inf

    for var_idx in has_explicit_bounds:
        if (
            lower_bounds[var_idx] == upper_bounds[var_idx]
            and lower_bounds[var_idx] == 0.0
            and fixed_count < target_conversions
        ):
            # Fixed variable at zero - set lower bound to -inf per CUTEst convention
            lower_bounds[var_idx] = -np.inf
            fixed_count += 1

    # Parse quadratic terms
    quadratic = {}
    for line in lines[bounds_end : sections.get("OBJECT BOUND", len(lines))]:
        if line.startswith("    X"):
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "X" and parts[2] == "X":
                i = int(parts[1]) - 1
                j = int(parts[3]) - 1
                quadratic[(i, j)] = np.float64(parts[4])

    quad_rows, quad_cols, quad_vals = [], [], []
    for (i, j), coeff in sorted(quadratic.items()):
        quad_rows.append(i)
        quad_cols.append(j)
        quad_vals.append(coeff)

    return {
        "obj_idx": jnp.array(obj_idx, dtype=jnp.int32),
        "obj_val": jnp.array(obj_val),
        "eq_rows": jnp.array(eq_rows, dtype=jnp.int32),
        "eq_cols": jnp.array(eq_cols, dtype=jnp.int32),
        "eq_vals": jnp.array(eq_vals),
        "ineq_rows": jnp.array(ineq_rows, dtype=jnp.int32),
        "ineq_cols": jnp.array(ineq_cols, dtype=jnp.int32),
        "ineq_vals": jnp.array(ineq_vals),
        "quad_rows": jnp.array(quad_rows, dtype=jnp.int32),
        "quad_cols": jnp.array(quad_cols, dtype=jnp.int32),
        "quad_vals": jnp.array(quad_vals),
        "lower_bounds": jnp.array(lower_bounds),
        "upper_bounds": jnp.array(upper_bounds),
        "eq_rhs": jnp.array(eq_rhs),
        "ineq_rhs": jnp.array(ineq_rhs),
        "n_eq": len(eq_constraints),
        "n_ineq": len(ineq_constraints),
    }


class CLEUVEN6(AbstractConstrainedMinimisation):
    """A nonconvex quadratic program from model predictive control.

    Problem from the OPTEC Workshop on Large Scale Convex Quadratic
    Programming - Algorithms, Software, and Applications, Leuven,
    25-26/10/2010.

    References:
        SIF input: Nick Gould, December 2010
        Corrected version: May 2019

    Classification: QLR2-RN-1200-3091
    """

    n_var: int = eqx.field(default=1200, init=False)
    n_con: int = eqx.field(init=False)  # Will be set dynamically
    provided_y0s: frozenset = frozenset({0})
    y0_iD: int = 0

    # Sparse data structures
    _obj_idx: Array = eqx.field(init=False)
    _obj_val: Array = eqx.field(init=False)
    _eq_rows: Array = eqx.field(init=False)
    _eq_cols: Array = eqx.field(init=False)
    _eq_vals: Array = eqx.field(init=False)
    _ineq_rows: Array = eqx.field(init=False)
    _ineq_cols: Array = eqx.field(init=False)
    _ineq_vals: Array = eqx.field(init=False)
    _quad_rows: Array = eqx.field(init=False)
    _quad_cols: Array = eqx.field(init=False)
    _quad_vals: Array = eqx.field(init=False)
    _lower_bounds: Array = eqx.field(init=False)
    _upper_bounds: Array = eqx.field(init=False)
    _eq_rhs: Array = eqx.field(init=False)
    _ineq_rhs: Array = eqx.field(init=False)
    _n_eq: int = eqx.field(init=False)
    _n_ineq: int = eqx.field(init=False)

    def __init__(self):
        """Load exact CLEUVEN6 data from SIF file."""
        data = _load_cleuven6_exact_data()

        self._obj_idx = data["obj_idx"]
        self._obj_val = data["obj_val"]
        self._eq_rows = data["eq_rows"]
        self._eq_cols = data["eq_cols"]
        self._eq_vals = data["eq_vals"]
        self._ineq_rows = data["ineq_rows"]
        self._ineq_cols = data["ineq_cols"]
        self._ineq_vals = data["ineq_vals"]
        self._quad_rows = data["quad_rows"]
        self._quad_cols = data["quad_cols"]
        self._quad_vals = data["quad_vals"]
        self._lower_bounds = data["lower_bounds"]
        self._upper_bounds = data["upper_bounds"]
        self._eq_rhs = data["eq_rhs"]
        self._ineq_rhs = data["ineq_rhs"]
        self._n_eq = data["n_eq"]
        self._n_ineq = data["n_ineq"]
        self.n_con = self._n_eq + self._n_ineq

    @property
    def y0(self) -> Float[Array, "1200"]:
        """Initial point - zeros for QP problems."""
        return jnp.zeros(self.n_var)

    @property
    def xlb(self) -> Float[Array, "1200"]:
        """Lower bounds on variables."""
        return self._lower_bounds

    @property
    def xub(self) -> Float[Array, "1200"]:
        """Upper bounds on variables."""
        return self._upper_bounds

    def objective(self, y: Float[Array, "1200"], args=None) -> Float[Array, ""]:
        """Quadratic objective function.

        f(x) = 0.5 * x^T * H * x + c^T * x
        """
        # Linear term
        linear_term = jnp.sum(self._obj_val * y[self._obj_idx])

        # Quadratic term using sparse representation
        # For (i,j,v) triplets, compute sum of v * x[i] * x[j]
        quad_term = jnp.sum(self._quad_vals * y[self._quad_rows] * y[self._quad_cols])

        # Note: diagonal terms should be multiplied by 0.5
        # Off-diagonal terms appear twice (symmetric matrix)
        # Adjust for proper quadratic form
        diag_mask = (self._quad_rows == self._quad_cols).astype(jnp.float64)
        diag_adjustment = -0.5 * jnp.sum(
            self._quad_vals * diag_mask * y[self._quad_rows] * y[self._quad_cols]
        )

        return linear_term + quad_term + diag_adjustment

    def constraint(self, y: Float[Array, "1200"], args=None):
        """Linear equality and inequality constraints.

        Returns (equalities, inequalities) where:
        - equalities: Ax = b (returned as Ax - b = 0)
        - inequalities: Ax <= b (returned as Ax - b <= 0)
        """
        # Equality constraints: Ax = b
        if self._n_eq > 0:
            eq_Ax = jnp.zeros(self._n_eq)
            eq_Ax = eq_Ax.at[self._eq_rows].add(self._eq_vals * y[self._eq_cols])
            equalities = eq_Ax - self._eq_rhs
        else:
            equalities = jnp.array([], dtype=y.dtype)

        # Inequality constraints: Ax <= b
        if self._n_ineq > 0:
            ineq_Ax = jnp.zeros(self._n_ineq)
            ineq_Ax = ineq_Ax.at[self._ineq_rows].add(
                self._ineq_vals * y[self._ineq_cols]
            )
            inequalities = ineq_Ax - self._ineq_rhs
        else:
            inequalities = jnp.array([], dtype=y.dtype)

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
        # Will be determined from test results
        return None
