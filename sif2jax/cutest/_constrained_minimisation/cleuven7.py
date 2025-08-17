"""CLEUVEN7 problem with exact coefficients from SIF file."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


def _load_cleuven7_exact_data():
    """Load exact CLEUVEN7 data from the SIF file.

    This function parses the SIF file once and caches the results.
    For production, this data would be pre-computed and stored.
    """
    import os

    # Try multiple possible paths for the SIF file
    # The path differs between local development and test container
    possible_paths = [
        "/workspace/x/archive/mastsif/CLEUVEN7.SIF",  # Local development
        "/workspace/archive/mastsif/CLEUVEN7.SIF",  # Test container mount
        "./archive/mastsif/CLEUVEN7.SIF",  # Relative path fallback
    ]

    sif_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sif_path = path
            break

    if sif_path is None:
        # Fallback if SIF file not found - this should not happen in tests
        raise FileNotFoundError(
            f"CLEUVEN7.SIF file not found. Tried paths: {possible_paths}"
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
    # Track constraint order as they appear in SIF file (CUTEst preserves this order)
    constraint_order = []

    for line in lines[sections["GROUPS"] + 1 : sections["BOUNDS"]]:
        parts = line.split()
        if len(parts) >= 4:
            if parts[0] == "N" and parts[1] == "OBJ" and parts[2] == "X":
                var_idx = int(parts[3]) - 1
                obj_linear[var_idx] = float(parts[4])
            elif parts[0] == "L" and parts[1] == "C" and len(parts) >= 6:
                con_idx = int(parts[2]) - 1
                var_idx = int(parts[4]) - 1
                if con_idx not in constraints:
                    constraints[con_idx] = {}
                    constraint_order.append(con_idx)  # Track first appearance order
                constraints[con_idx][var_idx] = float(parts[5])

    # Parse RHS values for constraints (reorder to match SIF constraint order)
    rhs_values_original = np.zeros(946)
    for line in lines:
        if line.startswith("    RHS"):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "C":
                con_idx = int(parts[2]) - 1
                rhs_values_original[con_idx] = float(parts[3])

    # Reorder RHS to match SIF constraint order
    rhs_values = np.zeros(len(constraint_order))
    for sif_position, con_idx in enumerate(constraint_order):
        rhs_values[sif_position] = rhs_values_original[con_idx]

    # Convert to arrays
    obj_idx = sorted(obj_linear.keys())
    obj_val = [obj_linear[i] for i in obj_idx]

    # Use constraint order as they appear in SIF file (CUTEst order)
    con_rows, con_cols, con_vals = [], [], []
    for sif_position, con_idx in enumerate(constraint_order):
        for var_idx, coeff in sorted(constraints[con_idx].items()):
            con_rows.append(sif_position)  # Use SIF position, not original index
            con_cols.append(var_idx)
            con_vals.append(coeff)

    # Parse bounds - start with unbounded defaults
    lower_bounds = np.full(360, -np.inf)
    upper_bounds = np.full(360, np.inf)

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
            value = float(parts[4])
            has_explicit_bounds.add(var_idx)
            if parts[0] == "LO":
                lower_bounds[var_idx] = value
            elif parts[0] == "UP":
                upper_bounds[var_idx] = value

    # For variables with identical lower and upper bounds (fixed variables),
    # CUTEst typically treats the lower bound as -inf for optimization purposes
    for var_idx in has_explicit_bounds:
        if np.abs(lower_bounds[var_idx] - upper_bounds[var_idx]) < 1e-12:
            # Fixed variable - set lower bound to -inf per CUTEst convention
            lower_bounds[var_idx] = -np.inf

    # Parse quadratic terms
    quadratic = {}
    for line in lines[bounds_end : sections.get("OBJECT BOUND", len(lines))]:
        if line.startswith("    X"):
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "X" and parts[2] == "X":
                i = int(parts[1]) - 1
                j = int(parts[3]) - 1
                quadratic[(i, j)] = float(parts[4])

    quad_rows, quad_cols, quad_vals = [], [], []
    for (i, j), coeff in sorted(quadratic.items()):
        quad_rows.append(i)
        quad_cols.append(j)
        quad_vals.append(coeff)

    return {
        "obj_idx": jnp.array(obj_idx, dtype=jnp.int32),
        "obj_val": jnp.array(obj_val),
        "con_rows": jnp.array(con_rows, dtype=jnp.int32),
        "con_cols": jnp.array(con_cols, dtype=jnp.int32),
        "con_vals": jnp.array(con_vals),
        "quad_rows": jnp.array(quad_rows, dtype=jnp.int32),
        "quad_cols": jnp.array(quad_cols, dtype=jnp.int32),
        "quad_vals": jnp.array(quad_vals),
        "lower_bounds": jnp.array(lower_bounds),
        "upper_bounds": jnp.array(upper_bounds),
        "rhs_values": jnp.array(rhs_values),
    }


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
    n_con: int = eqx.field(default=946, init=False)
    provided_y0s: frozenset = frozenset({0})
    y0_iD: int = 0

    # Sparse data structures
    _obj_idx: Array = eqx.field(init=False)
    _obj_val: Array = eqx.field(init=False)
    _con_rows: Array = eqx.field(init=False)
    _con_cols: Array = eqx.field(init=False)
    _con_vals: Array = eqx.field(init=False)
    _quad_rows: Array = eqx.field(init=False)
    _quad_cols: Array = eqx.field(init=False)
    _quad_vals: Array = eqx.field(init=False)
    _lower_bounds: Array = eqx.field(init=False)
    _upper_bounds: Array = eqx.field(init=False)
    _rhs_values: Array = eqx.field(init=False)

    def __init__(self):
        """Load exact CLEUVEN7 data from SIF file."""
        data = _load_cleuven7_exact_data()

        self._obj_idx = data["obj_idx"]
        self._obj_val = data["obj_val"]
        self._con_rows = data["con_rows"]
        self._con_cols = data["con_cols"]
        self._con_vals = data["con_vals"]
        self._quad_rows = data["quad_rows"]
        self._quad_cols = data["quad_cols"]
        self._quad_vals = data["quad_vals"]
        self._lower_bounds = data["lower_bounds"]
        self._upper_bounds = data["upper_bounds"]
        self._rhs_values = data["rhs_values"]

    @property
    def y0(self) -> Float[Array, "360"]:
        """Initial point - zeros for QP problems."""
        return jnp.zeros(self.n_var)

    @property
    def xlb(self) -> Float[Array, "360"]:
        """Lower bounds on variables."""
        return self._lower_bounds

    @property
    def xub(self) -> Float[Array, "360"]:
        """Upper bounds on variables."""
        return self._upper_bounds

    def objective(self, y: Float[Array, "360"], args=None) -> Float[Array, ""]:
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

    def constraint(
        self, y: Float[Array, "360"], args=None
    ) -> tuple[None, Float[Array, "946"]]:
        """Linear inequality constraints Ax <= b.

        Returns (None, inequalities) since all are inequality constraints.
        The constraints are returned in the form Ax - b <= 0.
        """
        # Sparse matrix-vector multiplication
        Ax = jnp.zeros(self.n_con)
        Ax = Ax.at[self._con_rows].add(self._con_vals * y[self._con_cols])

        # All constraints are inequalities (Ax - b <= 0 form)
        return None, Ax - self._rhs_values

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
