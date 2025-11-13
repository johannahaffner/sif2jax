"""LIPPERT1 problem from the CUTEst test set.

A discrete approximation to a continuum optimal flow problem
in the unit square. The continuum problem requires that the
divergence of a given flow should be given everywhere in the
region of interest, with the restriction that the capacity of
the flow is bounded. The aim is then to maximize the given flow.

Source: R. A. Lippert
    "Discrete approximations to continuum optimal flow problems"
    Tech. Report, Dept of Maths, M.I.T., 2006
    following a suggestion by Gil Strang

SIF input: Nick Gould, September 2006

Classification: LQR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LIPPERT1(AbstractConstrainedMinimisation):
    """LIPPERT1 problem - primal formulation of discrete optimal flow."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters (matching SIF file defaults)
    nx: int = 100  # Number of nodes in x direction
    ny: int = 100  # Number of nodes in y direction

    @property
    def dx(self):
        return 1.0 / self.nx

    @property
    def dy(self):
        return 1.0 / self.ny

    @property
    def s(self):
        return 1.0  # Source value

    @property
    def n_var(self):
        # Variables: u(0:nx, 1:ny), v(1:nx, 0:ny), t
        return (self.nx + 1) * self.ny + self.nx * (self.ny + 1) + 1

    @property
    def n_con(self):
        # Constraints: nx*ny conservation + 4*nx*ny capacity constraints
        return self.nx * self.ny + 4 * self.nx * self.ny

    def objective(self, y, args):
        """Objective: maximize t => minimize -t."""
        del args
        t = y[0]  # t is the first variable in SIF file
        return -t

    @property
    def y0(self):
        """Get initial point - all zeros (no explicit START POINT in SIF)."""
        # The SIF file has no active START POINT section, so all variables start at 0
        return jnp.zeros(self.n_var)

    @property
    def args(self):
        return None

    def constraint(self, y):
        """Compute all constraints."""
        nx, ny = self.nx, self.ny
        s = self.s

        # Extract variables: t, u, v (in SIF order)
        # SIF uses column-major (Fortran) ordering
        t = y[0]
        u_size = (nx + 1) * ny
        v_size = nx * (ny + 1)
        u = y[1 : 1 + u_size].reshape(ny, nx + 1).T  # Reshape with Fortran ordering
        v = (
            y[1 + u_size : 1 + u_size + v_size].reshape(ny + 1, nx).T
        )  # Reshape with Fortran ordering

        # Conservation constraints: (u_ij - u_i-1,j) + (v_ij - v_i,j-1) = t*s*nx*nx
        # Note: The SIF file formulation includes implicit scaling
        u_diff = u[1:, :] - u[:-1, :]  # shape (nx, ny)
        v_diff = v[:, 1:] - v[:, :-1]  # shape (nx, ny)
        conservation = u_diff + v_diff - t * s * nx * nx  # shape (nx, ny)
        eq_constraints = (
            conservation.T.ravel()
        )  # Flatten in column-major order for SIF compatibility

        # Capacity constraints: 4 per grid cell (inequality constraints)
        # Need to convert to g(x) >= 0 form: 1 - u^2 - v^2 >= 0
        u_curr = u[1:, :]  # u_ij for i=1:nx, j=1:ny, shape (nx, ny)
        u_prev = u[:-1, :]  # u_i-1,j for i=1:nx, j=1:ny, shape (nx, ny)
        v_curr = v[:, 1:]  # v_ij for i=1:nx, j=1:ny, shape (nx, ny)
        v_prev = v[:, :-1]  # v_i,j-1 for i=1:nx, j=1:ny, shape (nx, ny)

        # Compute capacity constraints (as g(x) <= 0 form: u^2 + v^2 - 1 <= 0)
        # Flatten in column-major order for SIF compatibility
        cap_a = (u_curr**2 + v_curr**2 - 1.0).T.ravel()  # nx*ny constraints
        cap_b = (u_prev**2 + v_curr**2 - 1.0).T.ravel()  # nx*ny constraints
        cap_c = (u_curr**2 + v_prev**2 - 1.0).T.ravel()  # nx*ny constraints
        cap_d = (u_prev**2 + v_prev**2 - 1.0).T.ravel()  # nx*ny constraints

        ineq_constraints = jnp.concatenate([cap_a, cap_b, cap_c, cap_d])

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        """Get bounds on variables."""
        # All variables free except t >= 0.01 (t is first variable)
        lower = jnp.full(self.n_var, -jnp.inf)
        lower = lower.at[0].set(0.01)  # t bound
        upper = jnp.full(self.n_var, jnp.inf)
        return (lower, upper)

    @property
    def expected_result(self):
        """Expected result not available from SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Solution from SIF file comment."""
        # From SIF file: -3.77245385 for nx=ny=100
        # No known solution for other sizes
        return None
