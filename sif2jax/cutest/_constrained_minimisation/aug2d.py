import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# The SIF file includes edge variables X(i,0), X(i,Y+), Y(0,j), Y(X+,j)
# that need to be properly incorporated into the variable structure.
# Current implementation doesn't match pycutest's expected 20200 variables.
class AUG2D(AbstractConstrainedMinimisation):
    """AUG2D problem.

    An expanded system formulation of a 2-D PDE system.

    A five-point discretization of Laplace's equation in a
    rectangular domain may be expressed in the form

          - M v = b,

    where M = sum a_i a_i^T. Letting A = (a_1 .... a_m),
    this system may be expanded as

           ( I   A^T ) (x) = (0),
           ( A    0  ) (v)   (b)

    which is then equivalent to solving the EQP
    minimize 1/2 || x ||_2^2   s.t.    A x = b

    In this variant, we replace the leading I block in the
    above formulation with a zero-one diagonal matrix D.
    This corresponds to certain boundary conditions.
    The resulting QP is thus convex but not strictly convex.

    SIF input: Nick Gould, February 1994

    classification QLR2-AN-V-V
    """

    # Default parameters
    NX: int = 100  # Number of nodes in x direction
    NY: int = 100  # Number of nodes in y direction

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        nx = self.NX
        ny = self.NY

        # Extract x and y variables
        # X variables: X(i,j) for i=1..nx-1, j=1..ny-1 and X(i,ny) for i=1..nx-1
        # Y variables: Y(i,j) for i=1..nx-1, j=1..ny-1 and Y(nx,j) for j=1..ny-1

        # X variables: X(i,j) for i=1..nx-1, j=1..ny-1 and X(i,ny) for i=1..nx-1
        # Total: (nx-1)*(ny-1) + (nx-1) = (nx-1)*ny
        # Y variables: Y(i,j) for i=1..nx, j=1..ny-1
        # Total: nx*(ny-1)

        n_x = (nx - 1) * ny
        n_y = nx * (ny - 1)

        x_vars = y[:n_x]
        y_vars = y[n_x : n_x + n_y]

        # Objective is 1/2 * (||x||^2 + ||y||^2)
        return 0.5 * (jnp.sum(x_vars**2) + jnp.sum(y_vars**2))

    def constraint(self, y):
        """Compute the constraints."""
        nx = self.NX
        ny = self.NY

        n_x = (nx - 1) * ny
        n_y = nx * (ny - 1)

        x_vars = y[:n_x]
        y_vars = y[n_x : n_x + n_y]

        # Reshape variables for easier indexing
        # x_vars correspond to X(i,j) for i=1..nx-1, j=1..ny
        x_reshaped = x_vars.reshape(nx - 1, ny)

        # y_vars correspond to Y(i,j) for i=1..nx, j=1..ny-1
        y_reshaped = y_vars.reshape(nx, ny - 1)

        # Vectorized computation of constraints
        # Initialize constraints with -1.0 (RHS is 1.0)
        constraints = jnp.full((nx, ny), -1.0)

        # Contributions from X variables
        # X(i,j) -> V(i,j) with coeff 1.0 and V(i+1,j) with coeff -1.0
        constraints = constraints.at[:-1, :].add(x_reshaped)
        constraints = constraints.at[1:, :].add(-x_reshaped)

        # Contributions from Y variables
        # Y(i,j) -> V(i,j) with coeff 1.0 and V(i,j+1) with coeff -1.0
        constraints = constraints.at[:, :-1].add(y_reshaped)
        constraints = constraints.at[:, 1:].add(-y_reshaped)

        # Flatten constraints to 1D array
        constraints = constraints.ravel()

        # All constraints are equality constraints
        return constraints, None

    @property
    def bounds(self):
        """Return the bounds on variables."""
        nx = self.NX
        ny = self.NY

        n_x = (nx - 1) * ny
        n_y = nx * (ny - 1)
        n_total = n_x + n_y

        # All variables are free
        y_lwr = jnp.full(n_total, -jnp.inf)
        y_upr = jnp.full(n_total, jnp.inf)

        return y_lwr, y_upr

    @property
    def y0(self):
        """Return the initial point."""
        nx = self.NX
        ny = self.NY

        n_x = (nx - 1) * ny
        n_y = nx * (ny - 1)
        n_total = n_x + n_y

        return jnp.zeros(n_total)

    @property
    def n(self):
        """Number of variables."""
        nx = self.NX
        ny = self.NY

        n_x = (nx - 1) * ny
        n_y = nx * (ny - 1)

        return n_x + n_y

    @property
    def m(self):
        """Number of constraints."""
        return self.NX * self.NY

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
