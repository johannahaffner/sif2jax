"""The porous medium equation on the unit square, D = -50.0.

The problem is to solve the porous medium equation on the unit square.
The equation is

    Δ(u²) + d ∂/∂x₁(u³) + f = 0

within the domain. The boundary conditions are that u = 1 on the bottom
and left sides and u = 0 on the top and right sides. Discretization is
using the usual central differences. The function f is a point source of
magnitude 50 at the lower left grid point. The initial approximation
is a discretization of 1 - x₁x₂.

Source: example 3.2.4 in
S. Eisenstat and H. Walker,
"Choosing the forcing terms in an inexact Newton method"
Report 6/94/75, Dept of Maths, Utah State University, 1994.

SIF input: Ph. Toint, July 1994.

Classification: NOR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class POROUS2(AbstractNonlinearEquations):
    """The porous medium equation with D = -50.0."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    P: int = 72  # Number of points in one side of the unit square
    D: float = -50.0  # Diffusion parameter

    @property
    def n(self):
        """Number of variables."""
        return self.P * self.P

    def _get_indices(self):
        """Map 2D indices to 1D indices for all points."""
        idx_map = {}
        idx = 0
        for j in range(1, self.P + 1):  # j from 1 to P
            for i in range(1, self.P + 1):  # i from 1 to P
                idx_map[(i, j)] = idx
                idx += 1
        return idx_map

    @property
    def y0(self):
        """Initial guess: discretization of 1 - x₁x₂."""
        idx_map = self._get_indices()
        h = 1.0 / (self.P - 1)
        y = jnp.zeros(self.n)

        for i in range(1, self.P + 1):
            for j in range(1, self.P + 1):
                x1 = (i - 1) * h
                x2 = (j - 1) * h
                y = y.at[idx_map[(i, j)]].set(1.0 - x1 * x2)

        return y

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables."""
        idx_map = self._get_indices()
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Fix boundary points
        # Bottom and top edges: U(1,j) = 1.0 and U(P,j) = 0.0 for all j
        for j in range(1, self.P + 1):
            lower = lower.at[idx_map[(1, j)]].set(1.0)
            upper = upper.at[idx_map[(1, j)]].set(1.0)
            lower = lower.at[idx_map[(self.P, j)]].set(0.0)
            upper = upper.at[idx_map[(self.P, j)]].set(0.0)

        # Left and right edges: U(i,P) = 1.0 and U(i,1) = 0.0 for i = 2 to P-1
        for i in range(2, self.P):
            lower = lower.at[idx_map[(i, self.P)]].set(1.0)
            upper = upper.at[idx_map[(i, self.P)]].set(1.0)
            lower = lower.at[idx_map[(i, 1)]].set(0.0)
            upper = upper.at[idx_map[(i, 1)]].set(0.0)

        return lower, upper

    def constraint(self, y):
        """Compute the residual of the porous medium equation."""
        h = 1.0 / (self.P - 1)
        h2 = h * h

        # Reshape y to 2D grid (i varies fastest, then j)
        # This matches the SIF ordering: U(1,1), U(2,1), ..., U(P,1), U(1,2), ...
        u = y.reshape((self.P, self.P), order="F")  # Fortran order: column-major

        # Compute u² and u³ for all points
        u2 = u * u
        u3 = u2 * u

        # Extract inner points for Laplacian computations
        # Inner points: i=2 to P-1, j=2 to P-1 (1-indexed)
        u2_inner = u2[1:-1, 1:-1]  # Shape: (P-2, P-2)

        # Compute Laplacian of u² using central differences
        # Δu² = (u²(i+1,j) + u²(i-1,j) + u²(i,j+1) + u²(i,j-1) - 4u²(i,j))/h²
        laplacian_u2 = (
            (
                u2[2:, 1:-1]  # u²(i+1,j)
                + u2[:-2, 1:-1]  # u²(i-1,j)
                + u2[1:-1, 2:]  # u²(i,j+1)
                + u2[1:-1, :-2]  # u²(i,j-1)
                - 4 * u2_inner  # -4u²(i,j)
            )
            / h2
        )

        # Compute ∂u³/∂x₁ using central differences
        # ∂u³/∂x₁ = (u³(i+1,j) - u³(i-1,j))/(2h)
        du3_dx1 = (u3[2:, 1:-1] - u3[:-2, 1:-1]) / (2 * h)

        # Compute residual: Δu² + D * ∂u³/∂x₁
        residual = laplacian_u2 + self.D * du3_dx1

        # Add constant to G(P-1, P-1) from CONSTANTS section
        # The SIF constant is -50.0 on the RHS, which means we add 50.0 to the LHS
        residual = residual.at[-1, -1].add(50.0)

        # Flatten the residual in C order (row-major) to match pycutest ordering
        # The SIF DO loops are DO I ... DO J with I as outer, J as inner
        residual_flat = residual.ravel(order="C")

        return residual_flat, None

    @property
    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
