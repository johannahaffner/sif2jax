import jax
import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem
from .table_utils import parse_table_sif


# Cache the parsed data at module level
_cached_data = None


def _get_cached_data():
    """Get cached parsed data, parsing if needed."""
    global _cached_data
    if _cached_data is None:
        _cached_data = parse_table_sif("archive/mastsif/TABLE8.SIF")
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

        Q_diag_vals = jnp.array(Q_diag_vals)
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
        return jnp.array(lower_bounds), jnp.array(upper_bounds)

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

        A_rows = jnp.array(A_rows, dtype=jnp.int32)
        A_cols = jnp.array(A_cols, dtype=jnp.int32)
        A_vals = jnp.array(A_vals)

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
        return 0.0  # Starting at zero, objective is 0

    @property
    def expected_result(self):
        """Expected result at y0."""
        return jnp.zeros(self.n)  # Optimal is at zero
