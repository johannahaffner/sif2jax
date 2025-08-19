import os

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractBoundedMinimisation


def _parse_wall20_sif():
    """Parse the SIF file to extract Hessian and linear terms."""
    sif_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "archive", "mastsif", "WALL20.SIF"
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

    # Parse quadratic terms - build full Hessian matrix
    n = 5924
    H = np.zeros((n, n))
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
                    H[i, j] = val
                    if i != j:
                        H[j, i] = val  # Symmetric

    return H, np.array(linear_indices)


# Parse once at module load time
_H, _LINEAR_INDICES = _parse_wall20_sif()
# Convert to JAX arrays
_H_JAX = jnp.array(_H)
_LINEAR_INDICES_JAX = jnp.array(_LINEAR_INDICES, dtype=jnp.int32)


class WALL20(AbstractBoundedMinimisation):
    """Bound-constrained quadratic program - contact problem involving a wall.

    (using a running bond model)

    Source: Angela Mihai (angela.mihai@strath.ac.uk)

    SIF input: Nick Gould, July 2005

    classification QBR2-RN-5924-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5924
    ns: int = 1735

    @property
    def y0(self):
        """Initial guess - zeros except for elements 1711-1735 which are 1.0."""
        y = jnp.zeros(self.n)
        # Set elements 1711-1735 (0-indexed: 1710-1734) to 1.0
        y = y.at[1710:1735].set(1.0)
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

        # Quadratic term: 0.5 * y^T H y
        # Use efficient matrix-vector operations
        quad_term = 0.5 * jnp.dot(y, jnp.dot(_H_JAX, y))

        return quad_term + linear_term

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
