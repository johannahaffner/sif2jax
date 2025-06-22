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
