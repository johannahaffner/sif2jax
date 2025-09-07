# TODO: Human review needed
# Attempts made:
#   1. Initial implementation with full vectorization
#   2. Module-level precomputation of bounds and initial point
#   3. Attempted to fix dimension mismatch (pycutest expects 5306, got 5304)
# Suspected issues:
#   - Complex variable indexing in SIF file (references V(NX+2, NY+1) and V(NX+1, NY+2))
#   - Grid dimension calculation needs verification against pycutest
#   - May need different grid layout or extra variables
# Resources needed: Analysis of exact SIF variable indexing pattern

"""MINSURFO problem - Minimal surface with obstacle.

Find the surface with minimal area, given boundary conditions,
and above an obstacle.

This is problem 17 in the COPS (Version 2) collection of
E. Dolan and J. More'
see "Benchmarking Optimization Software with COPS"
Argonne National Labs Technical Report ANL/MCS-246 (2000)

SIF input: Nick Gould, December 2000

Classification: OBR2-AN-V-V
"""

import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


# Problem parameters
NX = 50  # Grid points in x direction
NY = 100  # Grid points in y direction

# Grid dimensions
# Variables V(I,J) where I in 0..NX+1, J in 0..NY+1 gives (NX+2) x (NY+2) = 5304
# Plus element uses reference V(NX+2, NY+1) and V(NX+1, NY+2) for 2 extra variables
NX_PLUS_1 = NX + 1
NY_PLUS_1 = NY + 1
N_VARS = (NX + 2) * (NY + 2) + 2  # 5304 + 2 = 5306

# Grid spacing
HX = 1.0 / (NX + 1)
HY = 1.0 / (NY + 1)
AREA = 0.5 * HX * HY
INV_AREA = 1.0 / AREA
INV_HX2 = 1.0 / (HX * HX)
INV_HY2 = 1.0 / (HY * HY)

# Obstacle bounds
I_LOW = int(0.25 / HX)
I_HIGH = int(0.75 / HX + 0.9999999999)
J_LOW = int(0.25 / HY)
J_HIGH = int(0.75 / HY + 0.9999999999)


def _precompute_bounds():
    """Precompute the bounds arrays."""
    # Initialize with no bounds
    lower = jnp.full(N_VARS, -jnp.inf)
    upper = jnp.full(N_VARS, jnp.inf)

    # Create indices for all variables
    indices = jnp.arange(N_VARS)
    i_coords = indices // (NY + 2)
    j_coords = indices % (NY + 2)

    # Obstacle bounds: v[i,j] >= 1 for i in [I_LOW, I_HIGH], j in [J_LOW, J_HIGH]
    obstacle_mask = (
        (i_coords >= I_LOW)
        & (i_coords <= I_HIGH)
        & (j_coords >= J_LOW)
        & (j_coords <= J_HIGH)
    )
    lower = jnp.where(obstacle_mask, 1.0, lower)

    # Boundary conditions - edges are fixed
    # v(0,j) = 0 for all j
    left_edge = i_coords == 0
    lower = jnp.where(left_edge, 0.0, lower)
    upper = jnp.where(left_edge, 0.0, upper)

    # v(NX+1,j) = 0 for all j
    right_edge = i_coords == NX_PLUS_1
    lower = jnp.where(right_edge, 0.0, lower)
    upper = jnp.where(right_edge, 0.0, upper)

    # v(i,0) = 1 - (2*i*hx - 1)^2 for all i
    bottom_edge = j_coords == 0
    i_values = i_coords.astype(jnp.float32)
    bottom_values = 1.0 - jnp.square(2.0 * i_values * HX - 1.0)
    lower = jnp.where(bottom_edge, bottom_values, lower)
    upper = jnp.where(bottom_edge, bottom_values, upper)

    # v(i,NY+1) = 1 - (2*i*hx - 1)^2 for all i
    top_edge = j_coords == NY_PLUS_1
    lower = jnp.where(top_edge, bottom_values, lower)
    upper = jnp.where(top_edge, bottom_values, upper)

    return lower, upper


def _precompute_initial_point():
    """Precompute the initial point."""
    v = jnp.zeros((NX + 2, NY + 2))

    # Set initial values: v[i,j] = 1 - (2*i*hx - 1)^2
    for i in range(NX + 2):
        val = 1.0 - (2.0 * i * HX - 1.0) ** 2
        v = v.at[i, :].set(val)

    return v.flatten()


# Precompute at module level
LOWER_BOUNDS, UPPER_BOUNDS = _precompute_bounds()
INITIAL_POINT = _precompute_initial_point()


class MINSURFO(AbstractBoundedMinimisation):
    """MINSURFO problem - Minimal surface with obstacle."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Minimal surface area objective."""
        del args

        # Reshape to grid
        v = y.reshape(NX + 2, NY + 2)

        # Part A: Terms for (i,j) in [0,NX] x [0,NY]
        # area * sqrt(1 + ((v[i+1,j]-v[i,j])/hx)^2 + ((v[i,j+1]-v[i,j])/hy)^2)
        v_main = v[:NX_PLUS_1, :NY_PLUS_1]  # (0..NX, 0..NY)
        v_right = v[1 : NX + 2, :NY_PLUS_1]  # (1..NX+1, 0..NY)
        v_up = v[:NX_PLUS_1, 1 : NY + 2]  # (0..NX, 1..NY+1)

        dx_a = (v_right - v_main) * (1.0 / HX)
        dy_a = (v_up - v_main) * (1.0 / HY)
        alpha_a = 1.0 + dx_a * dx_a + dy_a * dy_a

        # Part B: Terms for (i,j) in [1,NX+1] x [1,NY+1]
        # area * sqrt(1 + ((v[i-1,j]-v[i,j])/hx)^2 + ((v[i,j-1]-v[i,j])/hy)^2)
        v_center = v[1:, 1:]  # (1..NX+1, 1..NY+1)
        v_left = v[:-1, 1:]  # (0..NX, 1..NY+1)
        v_down = v[1:, :-1]  # (1..NX+1, 0..NY)

        dx_b = (v_left - v_center) * (1.0 / HX)
        dy_b = (v_down - v_center) * (1.0 / HY)
        alpha_b = 1.0 + dx_b * dx_b + dy_b * dy_b

        # Sum all terms with regularization for numerical stability
        epsilon = 1e-12
        obj_a = AREA * jnp.sum(jnp.sqrt(alpha_a + epsilon))
        obj_b = AREA * jnp.sum(jnp.sqrt(alpha_b + epsilon))

        # Total objective minus constants
        return obj_a + obj_b - 2.0 * AREA * (NX_PLUS_1) * (NY_PLUS_1)

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(INITIAL_POINT)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
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
        # Count bound constraints
        n_obstacle = (I_HIGH - I_LOW + 1) * (J_HIGH - J_LOW + 1)
        n_boundary = 2 * (NY + 2) + 2 * (NX + 2)  # All edges
        return (0, 0, n_obstacle + n_boundary)
