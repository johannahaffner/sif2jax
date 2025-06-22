import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
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

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

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

        # Reshape the variables into a 2D grid
        x = y.reshape((p, p))

        # From AMPL file - the objective is:
        # sum {i in 1..p-1, j in 1..p-1}
        # sqrt(0.5*(p-1)^2*((x[i,j]-x[i+1,j+1])^2+(x[i+1,j]-x[i,j+1])^2)+1.0)/scale +
        # (sum {j in 1..p, i in 1..p} x[i,j])^2/p^4
        # where scale = (p-1)^2

        # First part: sum of area elements (vectorized)
        scale = (p - 1) ** 2

        # Vectorized computation of all a_ij and b_ij
        a_vals = x[:-1, :-1] - x[1:, 1:]  # x[i,j] - x[i+1,j+1]
        b_vals = x[1:, :-1] - x[:-1, 1:]  # x[i+1,j] - x[i,j+1]

        # Compute all area elements at once
        area_elements = jnp.sqrt(0.5 * scale * (a_vals**2 + b_vals**2) + 1.0) / scale
        area_sum = jnp.sum(area_elements)

        # Second part: (sum of all x values)^2 / p^4
        total_sum = jnp.sum(x)
        penalty = (total_sum**2) / (p**4)

        return area_sum + penalty

    def y0(self):
        # Initialize with zeros, then set boundary values
        p = self.p
        x = jnp.zeros((p, p))

        # Constants from AMPL file
        h00 = self.h00
        wtoe = self.slopej / (p - 1)
        ston = self.slopei / (p - 1)
        h01 = h00 + self.slopej
        h10 = h00 + self.slopei

        # From AMPL file (converting 1-based to 0-based indexing correctly):
        # let {j in 1..p} x[1,j] := (j-1)*wtoe+h00;
        # -> x[0,j-1] := (j-1)*wtoe+h00
        # let {j in 1..p} x[p,j] := (j-1)*wtoe+h10;
        # -> x[p-1,j-1] := (j-1)*wtoe+h10
        # let {i in 2..p-1} x[i,p] := (i-1)*ston+h00;
        # -> x[i-1,p-1] := (i-1)*ston+h00
        # let {i in 2..p-1} x[i,1] := (i-1)*ston+h01;
        # -> x[i-1,0] := (i-1)*ston+h01

        # Bottom edge (i=1 in AMPL = i=0 in 0-based): x[0,j] := (j-1)*wtoe+h00
        j_vals = jnp.arange(p)
        x = x.at[0, :].set((j_vals) * wtoe + h00)

        # Top edge (i=p in AMPL = i=p-1 in 0-based): x[p-1,j] := (j-1)*wtoe+h10
        x = x.at[p - 1, :].set((j_vals) * wtoe + h10)

        # Left edge (j=1 in AMPL = j=0 in 0-based):
        # x[i,0] := (i-1)*ston+h01 for i=2..p-1
        i_vals = jnp.arange(1, p - 1)
        x = x.at[i_vals, 0].set((i_vals) * ston + h01)

        # Right edge (j=p in AMPL = j=p-1 in 0-based):
        # x[i,p-1] := (i-1)*ston+h00 for i=2..p-1
        x = x.at[i_vals, p - 1].set((i_vals) * ston + h00)

        # Interior points: 0.0 (already initialized)

        # Flatten the 2D grid to 1D vector
        return x.flatten()

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: Human review needed
# Attempts made: [fixed boundary condition setup from SIF file analysis]
# Suspected issues: [starting point indexing/flattening still doesn't match
# PyCUTEst exactly]
# Additional resources needed: [detailed comparison of boundary setup with
# reference implementation]
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

        # Set values on boundaries following SIF file exactly

        # Function to create the boundary values as specified in the SIF file
        def create_boundary_vals():
            # Initialize with zeros
            vals = jnp.zeros((p, p))

            # Following SIF convention: X(I,J) where I=1..P, J=1..P
            # Converting to 0-based: X[I-1,J-1]

            # Lower and upper edges (SIF lines 122-130)
            j_vals = jnp.arange(p)  # J=1..P becomes j=0..p-1
            # X(1,J) = TL = (J-1)*WTOE + H00
            vals = vals.at[0, :].set((j_vals) * wtoe + h00)
            # X(P,J) = TU = (J-1)*WTOE + H10
            h10 = h00 + self.slopei
            vals = vals.at[p - 1, :].set((j_vals) * wtoe + h10)

            # Left and right edges (SIF lines 134-142)
            i_vals = jnp.arange(1, p - 1)  # I=2..P-1 becomes i=1..p-2
            # X(I,1) = TR = (I-1)*STON + H00
            vals = vals.at[i_vals, 0].set((i_vals) * ston + h00)
            # X(I,P) = TL = (I-1)*STON + H01
            h01 = h00 + self.slopej
            vals = vals.at[i_vals, p - 1].set((i_vals) * ston + h01)

            return inexact_asarray(vals)

        # Create and set boundary values
        x = create_boundary_vals()

        # Flatten the 2D grid to 1D vector
        return inexact_asarray(x.flatten())

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
