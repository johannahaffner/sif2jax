import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Initial implementation with sphere packing formulation
# Suspected issues: Complex constraint structure with n(n-1)/2 normalization
#                  constraints,
#                  starting point calculation mismatch, constraint dimension issues
# Resources needed: Detailed analysis of SIF GROUP USES section and constraint structure


class KISSING2(AbstractConstrainedMinimisation):
    """A second formulation of the KISSING NUMBER PROBLEM.

    This problem is associated to the family of Hard-Spheres problem. It belongs
    to the family of sphere packing problems, a class of challenging problems
    dating from the beginning of the 17th century which is related to practical
    problems in Chemistry, Biology and Physics. Given a fixed unit sphere at the
    origin in R^n, the problem consists of arranging a further m unit spheres so
    that sum of the distances to these spheres is as small as possible.

    After some algebraic manipulations, we can formulate this problem as:

    Minimize sum_{i=1}^m <p_i, p_i> - m*n

    subject to:
    <p_i - p_j, p_i - p_j> >= 4 for all different pair of indices i, j
    and
    <p_i, p_i> >= 4 for all indices i

    as well as n(n-1)/2 normalization constraints fixing components.

    The goal is to find an objective value equal to 0.

    Reference: "Sphere Packings, Lattices and Groups", J. H. Conway and
    N. J. C. Sloane, Springer-Verlag, NY, 1988.

    SIF input: Nick Gould, September 2000
    classification QQR2-RN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters (from SIF file)
    m = 25  # Number of points
    n = 4  # Dimension of sphere

    def objective(self, y: Array, args) -> Array:
        """Objective: sum_{i=1}^m <p_i, p_i> - m*n"""
        # Reshape variables into m points of n dimensions each
        points = y.reshape(self.m, self.n)

        # Sum of squared norms minus m*n
        squared_norms = jnp.sum(points**2, axis=1)
        total_norm = jnp.sum(squared_norms)

        return total_norm - self.m * self.n

    def constraint(self, y: Array):
        """Constraints for sphere packing problem."""
        # Reshape variables into m points of n dimensions each
        points = y.reshape(self.m, self.n)

        equality_constraints = []
        inequality_constraints = []

        # Normalization constraints: fix components according to SIF
        # From SIF: DO I 2 n, DO J I n: XX KISSING2 P(I,J) 0.0
        # This fixes P(2,2)=0, P(2,3)=0, P(2,4)=0, P(3,3)=0, P(3,4)=0, P(4,4)=0
        # These are n(n-1)/2 = 6 constraints for n=4
        eq_constraints = []
        for i in range(2, self.n + 1):  # i = 2, 3, 4 (1-indexed)
            for j in range(i, self.n + 1):  # j = i, ..., n
                # P(i,j) = 0 constraint (converting to 0-indexed)
                eq_constraints.append(points[i - 1, j - 1])

        equality_constraints = jnp.array(eq_constraints)

        # Inequality constraints: distances between points
        ineq_constraints = []

        # For all i,j pairs: <p_i - p_j, p_i - p_j> >= 4
        # Convert to form <= 0: 4 - ||p_i - p_j||^2 <= 0
        for i in range(self.m):
            for j in range(self.m):
                if i != j:
                    diff = points[i] - points[j]
                    squared_distance = jnp.sum(diff**2)
                    # 4 - squared_distance <= 0, so constraint is 4 - squared_distance
                    ineq_constraints.append(4.0 - squared_distance)

        # For all i: <p_i, p_i> >= 4
        # Convert to form <= 0: 4 - ||p_i||^2 <= 0
        for i in range(self.m):
            squared_norm = jnp.sum(points[i] ** 2)
            # 4 - squared_norm <= 0, so constraint is 4 - squared_norm
            ineq_constraints.append(4.0 - squared_norm)

        inequality_constraints = jnp.array(ineq_constraints)

        return equality_constraints, inequality_constraints

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        # From SIF START POINT section:
        # For each point i: cos = cos(2*pi*i/m), sin = sin(2*pi*i/m)
        # P(i,1) = cos, P(i,j) = sin for j=2,...,n-1, P(i,n) = cos

        points = jnp.zeros((self.m, self.n))

        for i in range(self.m):
            angle = 2 * jnp.pi * (i + 1) / self.m  # i+1 for 1-indexed
            cos_val = jnp.cos(angle)
            sin_val = jnp.sin(angle)

            # P(i,1) = cos
            points = points.at[i, 0].set(cos_val)
            # P(i,j) = sin for j=2,...,n-1
            for j in range(1, self.n - 1):
                points = points.at[i, j].set(sin_val)
            # P(i,n) = cos
            points = points.at[i, self.n - 1].set(cos_val)

        return points.flatten()

    @property
    def bounds(self):
        """Variable bounds. All variables are free (XR bounds in SIF)."""
        return None

    @property
    def args(self):
        """Additional arguments for the objective function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        return None

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # From SIF comment: "The goal is to find an objective value equal to 0."
        return jnp.array(0.0)
