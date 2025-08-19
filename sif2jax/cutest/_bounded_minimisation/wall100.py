import os

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractBoundedMinimisation


def _parse_wall100_sif():
    """Parse the SIF file to extract Hessian and linear terms."""
    sif_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "archive", "mastsif", "WALL100.SIF"
    )

    # Parse linear terms
    linear_indices = []
    with open(sif_path) as f:
        in_groups = False
        for line in f:
            if line.strip() == "GROUPS":
                in_groups = True
            elif line.strip() == "BOUNDS":
                break
            elif in_groups and line.startswith(" N  OBJ"):
                parts = line.split()
                idx = int(parts[2]) - 1  # Convert to 0-indexed
                linear_indices.append(idx)

    # Parse quadratic terms as triplets for sparse representation
    triplets = []
    with open(sif_path) as f:
        in_quad = False
        for line in f:
            if line.strip() == "QUADRATIC":
                in_quad = True
            elif line.strip() == "ENDATA":
                break
            elif in_quad and line.strip():
                parts = line.split()
                if len(parts) == 3:
                    i = int(parts[0]) - 1  # Convert to 0-indexed
                    j = int(parts[1]) - 1
                    val = float(parts[2])
                    triplets.append((i, j, val))

    # Convert triplets to arrays for vectorized computation
    rows = np.array([t[0] for t in triplets], dtype=np.int32)
    cols = np.array([t[1] for t in triplets], dtype=np.int32)
    data = np.array([t[2] for t in triplets], dtype=np.float64)

    return (rows, cols, data), np.array(linear_indices, dtype=np.int32)


# Parse once at module load time
_SPARSE_H, _LINEAR_INDICES = _parse_wall100_sif()
_ROWS_JAX = jnp.array(_SPARSE_H[0])
_COLS_JAX = jnp.array(_SPARSE_H[1])
_DATA_JAX = jnp.array(_SPARSE_H[2])
_LINEAR_INDICES_JAX = jnp.array(_LINEAR_INDICES)


class WALL100(AbstractBoundedMinimisation):
    """Bound-constrained quadratic program - contact problem involving a wall.

    (using a running bond model)

    Source: Angela Mihai (angela.mihai@strath.ac.uk)

    SIF input: Nick Gould, July 2005

    classification QBR2-RN-149624-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 149624
    ns: int = 44675

    @property
    def y0(self):
        """Initial guess - zeros except for elements 44551-44675 which are 1.0."""
        y = jnp.zeros(self.n)
        # Set elements 44551-44675 (0-indexed: 44550-44674) to 1.0
        y = y.at[44550:44675].set(1.0)
        return y

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Lower bounds: first ns variables >= 0, rest free."""
        lb = jnp.full(self.n, -jnp.inf)
        lb = lb.at[: self.ns].set(0.0)
        ub = jnp.full(self.n, jnp.inf)
        return lb, ub

    def objective(self, y, args):
        """Quadratic objective function with linear and quadratic terms."""
        del args

        # Linear term: -1.0 for each index in linear_indices
        linear_term = -jnp.sum(y[_LINEAR_INDICES_JAX])

        # Quadratic term: 0.5 * y^T H y computed from sparse representation
        # For each triplet (i, j, val), contribute val * y[i] * y[j]
        # If i == j: contributes val * y[i]^2
        # If i != j: contributes 2 * val * y[i] * y[j] (symmetric matrix)

        # Vectorized computation
        yi = y[_ROWS_JAX]
        yj = y[_COLS_JAX]
        products = yi * yj * _DATA_JAX

        # Check which elements are diagonal
        # Convert bool to float to avoid type promotion issues
        diagonal_mask = (_ROWS_JAX == _COLS_JAX).astype(jnp.float64)
        off_diagonal_mask = 1.0 - diagonal_mask

        # Sum contributions: diagonal once, off-diagonal twice (symmetric)
        quad_term = jnp.sum(products * diagonal_mask) + 2.0 * jnp.sum(
            products * off_diagonal_mask
        )

        return 0.5 * quad_term + linear_term

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
