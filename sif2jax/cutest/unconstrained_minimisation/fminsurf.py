import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This problem currently does not run. It should be debugged.
# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FMINSURF(AbstractUnconstrainedMinimisation):
    """The FMINSURF function.

    The free boundary minimum surface problem.

    The problem comes from the discretization of the minimum surface
    problem on the unit square with "free boundary conditions"
    one must find the minumum surface over the unit square
    (which is clearly 1.0). Furthermore, the average distance of the surface
    from zero is also minimized.

    The Hessian is dense.

    The unit square is discretized into (p-1)**2 little squares. The
    heights of the considered surface above the corners of these little
    squares are the problem variables, There are p**2 of them.
    Given these heights, the area above a little square is
    approximated by the
      S(i,j) = sqrt( 1 + 0.5(p-1)**2 ( a(i,j) + b(i,j) ) ) / (p-1)**2
    where
      a(i,j) = x(i,j) - x(i+1,j+1)
    and
      b(i,j) = x(i+1,j) - x(i,j+1)

    Source: setting the boundary free in
    A Griewank and Ph. Toint,
    "Partitioned variable metric updates for large structured
    optimization problems",
    Numerische Mathematik 39:429-448, 1982.

    SIF input: Ph. Toint, November 1991.
    Classification: OUR2-MY-V-0
    """

    p: int = 75  # Number of points on one side of the unit square
    h00: float = 1.0  # Boundary condition parameters
    slopej: float = 4.0  # Boundary condition parameters
    slopei: float = 8.0  # Boundary condition parameters

    @property
    def n(self):
        # Total number of variables: p^2
        return self.p * self.p

    def objective(self, y, args):
        del args
        p = self.p
        p_1 = p - 1

        # Reshape the variables into a 2D grid
        x = y.reshape((p, p))

        # Calculate the objective function components

        # First part: sum of sqrt(1 + 0.5 * (p-1)^2 * (a_ij + b_ij)) / (p-1)^2
        # for each little square in the grid

        # Vectorized computation of a_ij and b_ij for all little squares
        def compute_area_element(i, j):
            # a_ij = x_ij - x_{i+1,j+1}
            a_ij = x[i, j] - x[i + 1, j + 1]
            # b_ij = x_{i+1,j} - x_{i,j+1}
            b_ij = x[i + 1, j] - x[i, j + 1]

            # Compute square root term
            param = 0.5 * (p_1**2)
            area = jnp.sqrt(1.0 + param * (a_ij**2 + b_ij**2)) / (p_1**2)
            return area

        # Use vmap to compute all area elements
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(p_1), jnp.arange(p_1), indexing="ij"
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()

        compute_area_vectorized = jax.vmap(compute_area_element)
        area_elements = compute_area_vectorized(i_indices, j_indices)
        area_sum = jnp.sum(area_elements)

        # Second part: penalize average distance from zero
        avg_height = jnp.sum(x) / (p**4)

        return area_sum + avg_height**2

    def y0(self):
        # Initialize with zeros, then set boundary values
        p = self.p
        x = jnp.zeros((p, p))

        # Constants from SIF file
        h00 = self.h00
        wtoe = self.slopej / (p - 1)
        ston = self.slopei / (p - 1)

        # Set values on boundaries

        # Function to create the boundary values as specified in the SIF file
        def create_boundary_vals():
            # Initialize with zeros
            vals = jnp.zeros((p, p))

            # Bottom edge (j=0)
            j_vals = jnp.arange(p)
            vals = vals.at[0, :].set(h00 + wtoe * j_vals)

            # Top edge (j=p-1)
            vals = vals.at[p - 1, :].set(h00 + self.slopei + wtoe * j_vals)

            # Left edge (i=0, already set by bottom and top edges)
            i_vals = jnp.arange(1, p - 1)
            vals = vals.at[i_vals, 0].set(h00 + ston * i_vals)

            # Right edge (i=p-1, already set by bottom and top edges)
            vals = vals.at[i_vals, p - 1].set(h00 + self.slopej + ston * i_vals)

            return vals

        # Create and set boundary values
        x = create_boundary_vals()

        # Flatten the 2D grid to 1D vector
        return x.flatten()

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FMINSRF2(AbstractUnconstrainedMinimisation):
    """The FMINSRF2 function.

    The free boundary minimum surface problem.

    The problem comes from the discretization of the minimum surface
    problem on the unit square with "free boundary conditions"
    one must find the minumum surface over the unit square
    (which is clearly 1.0). Furthermore, the distance of the surface
    from zero at the centre of the unit square is also minimized.

    The unit square is discretized into (p-1)**2 little squares. The
    heights of the considered surface above the corners of these little
    squares are the problem variables, There are p**2 of them.
    Given these heights, the area above a little square is
    approximated by the
      S(i,j) = sqrt( 1 + 0.5(p-1)**2 ( a(i,j) + b(i,j) ) ) / (p-1)**2
    where
      a(i,j) = x(i,j) - x(i+1,j+1)
    and
      b(i,j) = x(i+1,j) - x(i,j+1)

    Source: setting the boundary free in
    A Griewank and Ph. Toint,
    "Partitioned variable metric updates for large structured
    optimization problems",
    Numerische Mathematik 39:429-448, 1982.

    SIF input: Ph. Toint, November 1991.
    Classification: OUR2-MY-V-0
    """

    p: int = 75  # Number of points on one side of the unit square
    h00: float = 1.0  # Boundary condition parameters
    slopej: float = 4.0  # Boundary condition parameters
    slopei: float = 8.0  # Boundary condition parameters

    @property
    def n(self):
        # Total number of variables: p^2
        return self.p * self.p

    def objective(self, y, args):
        del args
        p = self.p
        p_1 = p - 1

        # Reshape the variables into a 2D grid
        x = y.reshape((p, p))

        # Calculate the objective function components

        # First part: sum of sqrt(1 + 0.5 * (p-1)^2 * (a_ij + b_ij)) / (p-1)^2
        # for each little square in the grid

        # Vectorized computation of a_ij and b_ij for all little squares
        def compute_area_element(i, j):
            # a_ij = x_ij - x_{i+1,j+1}
            a_ij = x[i, j] - x[i + 1, j + 1]
            # b_ij = x_{i+1,j} - x_{i,j+1}
            b_ij = x[i + 1, j] - x[i, j + 1]

            # Compute square root term
            param = 0.5 * (p_1**2)
            area = jnp.sqrt(1.0 + param * (a_ij**2 + b_ij**2)) / (p_1**2)
            return area

        # Use vmap to compute all area elements
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(p_1), jnp.arange(p_1), indexing="ij"
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()

        compute_area_vectorized = jax.vmap(compute_area_element)
        area_elements = compute_area_vectorized(i_indices, j_indices)
        area_sum = jnp.sum(area_elements)

        # Second part: penalize value at center point
        mid = p // 2  # Integer division to find the middle point
        center_val = x[mid, mid]

        # Scale by p^2 as in the SIF file
        center_penalty = (center_val**2) * (p**2)

        return area_sum + center_penalty

    def y0(self):
        # Initialize with zeros, then set boundary values
        p = self.p
        x = jnp.zeros((p, p))

        # Constants from SIF file
        h00 = self.h00
        wtoe = self.slopej / (p - 1)
        ston = self.slopei / (p - 1)

        # Set values on boundaries

        # Function to create the boundary values as specified in the SIF file
        def create_boundary_vals():
            # Initialize with zeros
            vals = jnp.zeros((p, p))

            # Bottom edge (j=0)
            j_vals = jnp.arange(p)
            vals = vals.at[0, :].set(h00 + wtoe * j_vals)

            # Top edge (j=p-1)
            vals = vals.at[p - 1, :].set(h00 + self.slopei + wtoe * j_vals)

            # Left edge (i=0, already set by bottom and top edges)
            i_vals = jnp.arange(1, p - 1)
            vals = vals.at[i_vals, 0].set(h00 + ston * i_vals)

            # Right edge (i=p-1, already set by bottom and top edges)
            vals = vals.at[i_vals, p - 1].set(h00 + self.slopej + ston * i_vals)

            return vals

        # Create and set boundary values
        x = create_boundary_vals()

        # Flatten the 2D grid to 1D vector
        return x.flatten()

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
