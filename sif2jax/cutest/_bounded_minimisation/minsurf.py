# TODO: Human review needed
# Attempts made: Multiple optimization strategies including:
#   1. Basic vectorization of objective function
#   2. Module-level precomputation of bounds
#   3. Eliminated loops in bounds property
#   4. Used vectorized masks for boundary conditions
# Suspected issues: Test consistently times out after 2 minutes
#   - Problem may be inherently expensive for autodiff
#   - May need sparse matrix representation
#   - Could benefit from custom gradient implementation
# Resources needed: Profile to identify bottleneck in test suite

import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


# Precompute problem constants
P = 7
GRID_SIZE = P + 1
N_VARS = GRID_SIZE * GRID_SIZE


# Precompute bounds at module level
def _precompute_bounds():
    """Precompute the bounds arrays."""
    lower = jnp.full(N_VARS, -jnp.inf)
    upper = jnp.full(N_VARS, jnp.inf)

    # Create mask for boundary points using vectorized operations
    indices = jnp.arange(N_VARS)
    i_coords = indices // GRID_SIZE
    j_coords = indices % GRID_SIZE

    # Boundary mask: edges are fixed at 1.0
    is_boundary = (
        (i_coords == 0)
        | (i_coords == GRID_SIZE - 1)
        | (j_coords == 0)
        | (j_coords == GRID_SIZE - 1)
    )

    # Apply boundary conditions using where
    lower = jnp.where(is_boundary, 1.0, lower)
    upper = jnp.where(is_boundary, 1.0, upper)

    return lower, upper


# Precompute bounds once at module load
LOWER_BOUNDS, UPPER_BOUNDS = _precompute_bounds()


class MINSURF(AbstractBoundedMinimisation):
    """MINSURF problem - A version of the minimum surface problem.

    Variable dimension full rank linear problem.
    A version of the minimum surface problem on the unit square
    with simple boundary conditions.

    SIF input: Ph. Toint, Jan 1991.

    Classification: OXR2-MY-64-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Minimum surface area objective."""
        del args

        # Reshape flat vector to grid for easier indexing
        x_grid = y.reshape(GRID_SIZE, GRID_SIZE)

        # Vectorized computation over all interior grid cells
        weight = (P * P) / 2.0

        # Diagonal differences: X(i,j) - X(i+1,j+1) for i,j = 0..p-1
        a_diffs = x_grid[:P, :P] - x_grid[1:, 1:]

        # Anti-diagonal differences: X(i,j+1) - X(i+1,j) for i,j = 0..p-1
        b_diffs = x_grid[:P, 1:] - x_grid[1:, :P]

        # Sum of squares for each grid cell
        alpha = a_diffs * a_diffs + b_diffs * b_diffs

        # Square root with small regularization for numerical stability
        epsilon = 1e-12
        sqrt_alpha = jnp.sqrt(alpha + epsilon)

        # Sum all contributions
        obj = weight * jnp.sum(sqrt_alpha)

        return obj - P * P * 1.0  # Subtract constants from SIF

    @property
    def y0(self):
        """Initial guess."""
        # Start with boundary conditions satisfied
        x = jnp.ones((GRID_SIZE, GRID_SIZE))
        return inexact_asarray(x.flatten())

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds - boundary variables are fixed at 1.0."""
        # Return precomputed bounds
        return LOWER_BOUNDS, UPPER_BOUNDS

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None

    def num_constraints(self):
        """Returns the number of constraints in the problem."""
        # Count the number of fixed boundary variables
        n_boundary = (
            4 * GRID_SIZE - 4
        )  # Boundary elements (avoiding double-counting corners)
        return (0, 0, n_boundary)  # 0 equality, 0 inequality, n_boundary bounds
