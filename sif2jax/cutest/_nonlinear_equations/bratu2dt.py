"""The 2D Bratu problem on the unit square, using finite differences.

At the turning point.

Source: Problem 3 in
J.J. More',
"A collection of nonlinear model problems"
Proceedings of the AMS-SIAM Summer seminar on the Computational
Solution of Nonlinear Systems of Equations, Colorado, 1988.
Argonne National Laboratory MCS-P60-0289, 1989.

SIF input: Ph. Toint, Dec 1989.

Classification: NOR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class BRATU2DT(AbstractNonlinearEquations):
    """The 2D Bratu problem at the turning point."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    P: int = 72  # Number of points in one side of the unit square
    LAMBDA: float = 6.80812  # Bratu problem parameter at turning point

    @property
    def n(self):
        """Number of variables."""
        # All P×P points are included as variables (boundary points are fixed)
        return self.P * self.P

    def _get_indices(self):
        """Map 2D indices to 1D indices for all points."""
        # All points (i,j) for i,j = 1 to P (1-indexed)
        # We map them to indices 0 to P^2 - 1
        idx_map = {}
        idx = 0
        for j in range(1, self.P + 1):  # j from 1 to P
            for i in range(1, self.P + 1):  # i from 1 to P
                idx_map[(i, j)] = idx
                idx += 1
        return idx_map

    @property
    def y0(self):
        """Initial guess (all zeros)."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables."""
        # Boundary points are fixed at 0
        idx_map = self._get_indices()
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Fix boundary points to 0
        # Bottom and top edges: U(i,1) and U(i,P) for all i
        for i in range(1, self.P + 1):
            lower = lower.at[idx_map[(i, 1)]].set(0.0)
            upper = upper.at[idx_map[(i, 1)]].set(0.0)
            lower = lower.at[idx_map[(i, self.P)]].set(0.0)
            upper = upper.at[idx_map[(i, self.P)]].set(0.0)

        # Left and right edges: U(1,j) and U(P,j) for j = 2 to P-1
        for j in range(2, self.P):
            lower = lower.at[idx_map[(1, j)]].set(0.0)
            upper = upper.at[idx_map[(1, j)]].set(0.0)
            lower = lower.at[idx_map[(self.P, j)]].set(0.0)
            upper = upper.at[idx_map[(self.P, j)]].set(0.0)

        return lower, upper

    def constraint(self, y, args=None):
        """Compute the system of nonlinear equations."""
        del args  # Not used

        idx_map = self._get_indices()
        h = 1.0 / (self.P - 1)
        h2 = h * h
        c = h2 * self.LAMBDA

        equations = []

        # Loop over interior points only (equations are only for interior points)
        for j in range(2, self.P):  # j from 2 to P-1
            for i in range(2, self.P):  # i from 2 to P-1
                # Get the variable for this point
                u_ij = y[idx_map[(i, j)]]

                # Laplacian operator (5-point stencil)
                # 4*u(i,j) - u(i+1,j) - u(i-1,j) - u(i,j+1) - u(i,j-1)
                laplacian = 4.0 * u_ij

                # All neighbors exist in idx_map now
                laplacian -= y[idx_map[(i + 1, j)]]
                laplacian -= y[idx_map[(i - 1, j)]]
                laplacian -= y[idx_map[(i, j + 1)]]
                laplacian -= y[idx_map[(i, j - 1)]]

                # Nonlinear term: -c * exp(u)
                nonlinear = -c * jnp.exp(u_ij)

                # The equation: Laplacian - c * exp(u) = 0
                equation = laplacian + nonlinear
                equations.append(equation)

        # Return as tuple (equality_constraints, inequality_constraints)
        # For nonlinear equations, all constraints are equality constraints
        return jnp.array(equations), None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value for P=72."""
        return jnp.array(1.30497e-6)
