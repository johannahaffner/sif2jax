import os

import jax
import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractConstrainedQuadraticProblem


# Load data at module level
_current_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_current_dir, "data")

# Load data directly as JAX arrays
_A_ROWS = jnp.asarray(
    np.load(os.path.join(_data_dir, "table7_A_rows.npy")), dtype=jnp.int32
)
_A_COLS = jnp.asarray(
    np.load(os.path.join(_data_dir, "table7_A_cols.npy")), dtype=jnp.int32
)
_A_VALS = jnp.asarray(np.load(os.path.join(_data_dir, "table7_A_vals.npy")))
_LOWER_BOUNDS = jnp.asarray(np.load(os.path.join(_data_dir, "table7_lower_bounds.npy")))
_UPPER_BOUNDS = jnp.asarray(np.load(os.path.join(_data_dir, "table7_upper_bounds.npy")))
_Q_DIAG_VALS = jnp.asarray(np.load(os.path.join(_data_dir, "table7_Q_diag_vals.npy")))
_M_VAL = int(np.load(os.path.join(_data_dir, "table7_m.npy")))


class TABLE7(AbstractConstrainedQuadraticProblem):
    """A two-norm fitted formulation for tabular data protection.

    Source:
    J. Castro,
    Minimum-distance controlled perturbation methods for
    large-scale tabular data protection,
    European Journal of Operational Research 171 (2006) pp 39-52.

    SIF input: Jordi Castro, 2006 as L2_table7.mps
    see http://www-eio.upc.es/~jcastro/data.html

    classification QLR2-RN-624-230
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 624

    @property
    def m(self):
        """Number of constraints."""
        return 230

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: 0.5 * y^T Q y where Q is diagonal."""
        del args
        # Cast to match y's dtype if needed
        Q_diag_vals = _Q_DIAG_VALS.astype(y.dtype)
        # The QMATRIX values need to be halved for the standard form 0.5 * y^T Q y
        # since the SIF file specifies the full coefficient
        return 0.5 * jnp.sum(Q_diag_vals * y * y)

    @property
    def bounds(self):
        """Variable bounds."""
        return _LOWER_BOUNDS, _UPPER_BOUNDS

    def constraint(self, y):
        """Linear equality constraints: Ay = 0."""
        # Cast A_vals to match y's dtype if needed
        A_vals = _A_VALS.astype(y.dtype)

        # Vectorized sparse matrix-vector multiplication
        selected_y = y[_A_COLS]
        products = A_vals * selected_y

        # Use segment_sum for efficient aggregation
        eq_constraints = jax.ops.segment_sum(
            products, _A_ROWS, num_segments=_M_VAL, indices_are_sorted=False
        )
        return eq_constraints, None

    @property
    def expected_objective_value(self):
        """Expected objective value at y0."""
        return jnp.array(0.0)  # Starting at zero, objective is 0

    @property
    def expected_result(self):
        """Expected result at y0."""
        return jnp.zeros(self.n)  # Optimal is at zero
