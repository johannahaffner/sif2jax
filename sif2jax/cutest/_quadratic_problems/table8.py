import os

import jax
import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractConstrainedQuadraticProblem


# Cache the parsed data at module level
_cached_data = None


def _get_cached_data():
    """Get cached parsed data, loading from numpy files if needed."""
    global _cached_data
    if _cached_data is None:
        # Get the directory containing this file
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(current_dir, "data")

        # Load data directly as JAX arrays
        A_rows = jnp.asarray(
            np.load(os.path.join(data_dir, "table8_A_rows.npy")), dtype=jnp.int32
        )
        A_cols = jnp.asarray(
            np.load(os.path.join(data_dir, "table8_A_cols.npy")), dtype=jnp.int32
        )
        A_vals = jnp.asarray(np.load(os.path.join(data_dir, "table8_A_vals.npy")))
        lower_bounds = jnp.asarray(
            np.load(os.path.join(data_dir, "table8_lower_bounds.npy"))
        )
        upper_bounds = jnp.asarray(
            np.load(os.path.join(data_dir, "table8_upper_bounds.npy"))
        )
        Q_diag_vals = jnp.asarray(
            np.load(os.path.join(data_dir, "table8_Q_diag_vals.npy"))
        )
        m_val = int(np.load(os.path.join(data_dir, "table8_m.npy")))

        _cached_data = (
            A_rows,
            A_cols,
            A_vals,
            lower_bounds,
            upper_bounds,
            Q_diag_vals,
            m_val,
        )
    return _cached_data


class TABLE8(AbstractConstrainedQuadraticProblem):
    """A two-norm fitted formulation for tabular data protection.

    Source:
    J. Castro,
    Minimum-distance controlled perturbation methods for
    large-scale tabular data protection,
    European Journal of Operational Research 171 (2006) pp 39-52.

    SIF input: Jordi Castro, 2006 as L2_table8.mps
    see http://www-eio.upc.es/~jcastro/data.html

    classification QLR2-RN-1271-72
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 1271

    @property
    def m(self):
        """Number of constraints."""
        return 72

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
        # Get the cached problem data
        (
            A_rows,
            A_cols,
            A_vals,
            lower_bounds,
            upper_bounds,
            Q_diag_vals,
            m_val,
        ) = _get_cached_data()

        # Cast to match y's dtype if needed
        Q_diag_vals = Q_diag_vals.astype(y.dtype)
        # The QMATRIX values need to be halved for the standard form 0.5 * y^T Q y
        # since the SIF file specifies the full coefficient
        return 0.5 * jnp.sum(Q_diag_vals * y * y)

    @property
    def bounds(self):
        """Variable bounds."""
        (
            A_rows,
            A_cols,
            A_vals,
            lower_bounds,
            upper_bounds,
            Q_diag_vals,
            m_val,
        ) = _get_cached_data()
        return lower_bounds, upper_bounds

    def constraint(self, y):
        """Linear equality constraints: Ay = 0."""
        # Get the cached problem data
        (
            A_rows,
            A_cols,
            A_vals,
            lower_bounds,
            upper_bounds,
            Q_diag_vals,
            m_val,
        ) = _get_cached_data()

        # Cast A_vals to match y's dtype if needed
        A_vals = A_vals.astype(y.dtype)

        # Vectorized sparse matrix-vector multiplication
        selected_y = y[A_cols]
        products = A_vals * selected_y

        # Use segment_sum for efficient aggregation
        eq_constraints = jax.ops.segment_sum(
            products, A_rows, num_segments=m_val, indices_are_sorted=False
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
