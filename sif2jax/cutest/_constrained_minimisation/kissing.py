import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class KISSING(AbstractConstrainedMinimisation):
    """KISSING NUMBER PROBLEM

    This problem is associated to the family of Hard-Spheres problem. It
    belongs to the family of sphere packing problems, a class of challenging
    problems dating from the beginning of the 17th century which is related
    to practical problems in Chemistry, Biology and Physics. It consists on
    maximizing the minimum pairwise distance between NP points on a sphere
    in R^{MDIM}.

    This problem may be reduced to a nonconvex nonlinear optimization problem
    with a potentially large number of (nonoptimal) points satisfying
    optimality conditions.

    After some algebraic manipulations, we can formulate this problem as:

                            Minimize z
                            subject to

      z >= <x_i, x_j> for all different pair of indices i, j

                            ||x_i||^2 = 1    for all i = 1,...,NP

     The goal is to find an objective value less than 0.5 (This means
     that the NP points stored belong to the sphere and every distance
     between two of them is greater than 1.0).

    References:
    [1] "Validation of an Augmented Lagrangian algorithm with a
         Gauss-Newton Hessian approximation using a set of
         Hard-Spheres problems", N. Krejic, J. M. Martinez, M. Mello
         and E. A. Pilotta, Tech. Report RP 29/98, IMECC-UNICAMP,
         Campinas, 1998.
    [2] "Inexact-Restoration Algorithm for Constrained Optimization",
         J. M. Martinez and E. A. Pilotta, Tech. Report, IMECC-UNICAMP,
         Campinas, 1998.
    [3]  "Sphere Packings, Lattices and Groups", J. H. Conway and
          N. J. C. Sloane, Springer-Verlag, NY, 1988.

    SIF input: September 29, 1998
               Jose Mario Martinez
               Elvio Angel Pilotta

    classification LQR2-RN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    NP = 42  # Number of points
    MDIM = 3  # Dimension

    def objective(self, y, args):
        """Objective function: minimize z"""
        # z is the last variable
        z = y[-1]
        return z

    @property
    def y0(self):
        """Starting point from SIF file"""
        # Initialize points with the provided starting values for first 12 points
        x = jnp.zeros((self.NP, self.MDIM))

        # First 12 points from SIF file
        x = x.at[0, :].set(jnp.array([-0.10890604, 0.85395078, -0.45461680]))
        x = x.at[1, :].set(jnp.array([0.49883922, -0.18439316, -0.04798594]))
        x = x.at[2, :].set(jnp.array([0.28262888, -0.48054070, 0.46715332]))
        x = x.at[3, :].set(jnp.array([-0.00580106, -0.49987584, -0.44130302]))
        x = x.at[4, :].set(jnp.array([0.81712540, -0.36874258, -0.68321896]))
        x = x.at[5, :].set(jnp.array([0.29642426, 0.82315508, 0.35938150]))
        x = x.at[6, :].set(jnp.array([0.09215152, -0.53564686, 0.00191436]))
        x = x.at[7, :].set(jnp.array([0.11700318, 0.96722760, -0.14916438]))
        x = x.at[8, :].set(jnp.array([0.01791524, 0.17759446, -0.61875872]))
        x = x.at[9, :].set(jnp.array([-0.63833630, 0.80830972, 0.45846734]))
        x = x.at[10, :].set(jnp.array([0.28446456, 0.45686938, 0.16368980]))
        x = x.at[11, :].set(jnp.array([0.76557382, 0.16700944, -0.31647534]))

        # Remaining points default to zero (as in SIF file)
        # They are already initialized to zero above

        # Flatten x and append initial z = 0.0 (default value)
        y_flat = jnp.concatenate([x.flatten(), jnp.array([0.0])])

        return y_flat

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The target is to achieve z < 0.5
        return None

    @property
    def expected_objective_value(self):
        # Should be less than 0.5
        return jnp.array(0.447214)  # Approximately sqrt(2) - 1

    @property
    def bounds(self):
        # All variables are free (XR in SIF file means free/unconstrained)
        return None

    def constraint(self, y):
        """Constraints: equality (sphere) and inequality (pairwise distances)"""
        # Extract variables
        x_flat = y[:-1]  # All x variables
        z = y[-1]  # z variable

        # Reshape x into (NP, MDIM)
        x = x_flat.reshape((self.NP, self.MDIM))

        # Equality constraints: ||x_i||^2 = 1 for all i
        equality_constraints = jnp.sum(x**2, axis=1) - 1.0

        # Inequality constraints: z >= <x_i, x_j> for all different pairs i, j
        # From SIF: XL IC(I,J) Z -1.0 means IC(I,J) + Z >= 0
        # where IC(I,J) is the dot product <x_i, x_j>
        inequality_constraints = []
        for i in range(self.NP - 1):
            for j in range(i + 1, self.NP):
                # Inner product <x_i, x_j>
                dot_product = jnp.dot(x[i], x[j])
                # Constraint: dot_product - z >= 0 (from SIF formulation)
                inequality_constraints.append(dot_product - z)

        inequality_constraints = jnp.array(inequality_constraints)

        return equality_constraints, inequality_constraints
