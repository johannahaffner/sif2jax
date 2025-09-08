"""LIPPERT2 problem from the CUTEst test set.

A discrete approximation to a continuum optimal flow problem
in the unit square. The continuum problem requires that the
divergence of a given flow should be given everywhere in the
region of interest, with the restriction that the capacity of
the flow is bounded. The aim is then to maximize the given flow.

The discrete problem (dual formulation 2) in the unit square.

Source: R. A. Lippert
    "Discrete approximations to continuum optimal flow problems"
    Tech. Report, Dept of Maths, M.I.T., 2006
    following a suggestion by Gil Strang

SIF input: Nick Gould, September 2006
           correction by S. Gratton & Ph. Toint, May 2024

Classification: LQR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LIPPERT2(AbstractConstrainedMinimisation):
    """LIPPERT2 problem - dual formulation of discrete optimal flow."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters (using smaller default for testing)
    nx: int = 10  # Number of nodes in x direction
    ny: int = 10  # Number of nodes in y direction

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
        # Variables: u(0:nx, 1:ny), v(1:nx, 0:ny), r
        return (self.nx + 1) * self.ny + self.nx * (self.ny + 1) + 1

    @property
    def n_con(self):
        # Constraints: nx*ny conservation + 4*nx*ny capacity constraints
        return self.nx * self.ny + 4 * self.nx * self.ny

    def objective(self, y, args):
        """Objective: minimize r."""
        del args
        r = y[-1]
        return r

    @property
    def y0(self):
        """Get initial point from SIF file."""
        nx, ny = self.nx, self.ny
        dx = self.dx

        x0 = jnp.zeros(self.n_var)

        # Set r (last variable) to 1.0
        x0 = x0.at[-1].set(1.0)

        # Initialize u values with linear gradient
        u_size = (nx + 1) * ny
        u = jnp.zeros((nx + 1, ny))
        for i in range(nx + 1):
            alpha = i * dx / 2.0
            u = u.at[i, :].set(alpha)
        x0 = x0.at[:u_size].set(u.ravel())

        # Initialize v values with linear gradient
        v_size = nx * (ny + 1)
        v = jnp.zeros((nx, ny + 1))
        for j in range(ny + 1):
            alpha = j * dx / 2.0  # Note: using dx as in SIF file
            v = v.at[:, j].set(alpha)
        x0 = x0.at[u_size : u_size + v_size].set(v.ravel())

        return x0

    @property
    def args(self):
        return None

    def constraint(self, y):
        """Compute all constraints."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        s = self.s

        # Extract variables
        u_size = (nx + 1) * ny
        v_size = nx * (ny + 1)
        u = y[:u_size].reshape(nx + 1, ny)
        v = y[u_size : u_size + v_size].reshape(nx, ny + 1)
        r = y[-1]

        # Conservation constraints: dx*(u_ij - u_i-1,j) + dy*(v_ij - v_i,j-1) = s
        # These are equality constraints
        u_diff = (u[1:, :] - u[:-1, :]) * dx  # shape (nx, ny)
        v_diff = (v[:, 1:] - v[:, :-1]) * dy  # shape (nx, ny)
        conservation = u_diff + v_diff - s  # shape (nx, ny)
        eq_constraints = conservation.ravel()  # nx*ny constraints

        # Capacity constraints: 4 per grid cell (inequality constraints)
        # Need to convert to g(x) >= 0 form: r^2 - u^2 - v^2 >= 0
        u_curr = u[1:, :]  # u_ij for i=1:nx, j=1:ny, shape (nx, ny)
        u_prev = u[:-1, :]  # u_i-1,j for i=1:nx, j=1:ny, shape (nx, ny)
        v_curr = v[:, 1:]  # v_ij for i=1:nx, j=1:ny, shape (nx, ny)
        v_prev = v[:, :-1]  # v_i,j-1 for i=1:nx, j=1:ny, shape (nx, ny)

        # Compute capacity constraints (as g(x) >= 0 form)
        r_squared = r**2
        cap_a = (r_squared - u_curr**2 - v_curr**2).ravel()  # nx*ny constraints
        cap_b = (r_squared - u_prev**2 - v_curr**2).ravel()  # nx*ny constraints
        cap_c = (r_squared - u_curr**2 - v_prev**2).ravel()  # nx*ny constraints
        cap_d = (r_squared - u_prev**2 - v_prev**2).ravel()  # nx*ny constraints

        ineq_constraints = jnp.concatenate([cap_a, cap_b, cap_c, cap_d])

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        """Get bounds on variables."""
        # All variables free except r >= 0
        lower = jnp.full(self.n_var, -jnp.inf)
        lower = lower.at[-1].set(0.0)
        upper = jnp.full(self.n_var, jnp.inf)
        return (lower, upper)

    @property
    def expected_result(self):
        """Expected result not available from SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Solution from SIF file comment."""
        # From SIF file: 3.77245385 for nx=ny=100
        # No known solution for other sizes
        return None
