import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GILBERT(AbstractConstrainedMinimisation):
    """
    GILBERT - A simple constrained problem.

    # TODO: Human review needed
    # Attempts made: [
    #   1. Initial implementation with basic quadratic objective and sphere constraint
    #   2. Applied SCALE 2.0 factor from SIF SPHERE group to constraint
    #   3. Analyzed L2 group type scaling for objective function
    #   4. Fixed constraint to use 0.5 scaling factor
    # ]
    # Suspected issues: [
    #   - Objective function scaling still incorrect (expected ~85K, getting 250K)
    #   - Constraint scaling slightly off (0.5 difference suggests wrong factor)
    #   - SIF GROUP TYPE and SCALE interpretations need refinement
    #   - May need deeper analysis of ELEMENT/GROUP interactions in this problem
    # ]
    # Resources needed: [
    #   - Detailed SIF GROUP and ELEMENT type documentation
    #   - Reference implementation for comparison
    #   - Understanding of how SCALE factors interact with L2 group types
    # ]

    A diagonal convex quadratic objective is minimized on the unit sphere.

    Source:
    J.Ch. Gilbert,
    "On the Realization of the Wolfe Conditions in Reduced Quasi-Newton
    Methods for Equality Constrained Optimization",
    RR-2127, INRIA (F), 1993.

    Classification: QQR2-AN-V-1

    SIF input: Ph. Toint, April 1994
    """

    # Problem parameters from SIF file
    n: int = 5000  # Problem size (default parameter)

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # From SIF: alternating +10/-10 pattern
        y0 = jnp.zeros(self.n)
        s = 10.0
        for i in range(self.n):
            y0 = y0.at[i].set(s)
            s = -s  # Alternate sign
        return y0

    @property
    def bounds(self):
        """Variable bounds."""
        # X(1) >= 0.0, others are free
        lower = jnp.full(self.n, -jnp.inf)
        lower = lower.at[0].set(0.0)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def objective(self, y, args):
        """
        Objective function: weighted sum of squares.

        From SIF: sum_i (N+1-i)/N * x_i^2
        The L2 group type in SIF provides f = GVAR^2, so no additional scaling
        """
        del args
        n = self.n

        # Compute weights: (N+1-i)/N for i=1..N (1-indexed)
        weights = jnp.arange(n, 0, -1, dtype=y.dtype) / n  # (N, N-1, ..., 1)/N

        # Each O(I) group has coefficient AI and is squared via L2 group type
        return jnp.sum(weights * y**2)

    def constraint(self, y):
        """
        Constraint: unit sphere constraint sum(x_i^2) = 1.

        From SIF: SPHERE constraint with 'SCALE' 2.0
        This means the constraint is actually 0.5 * sum(x_i^2) - 1 = 0
        """
        # The SPHERE group uses XSQ(I) elements and 'SCALE' 2.0
        # This results in constraint: 0.5 * sum(x_i^2) - 1 = 0
        sphere_constraint = 0.5 * jnp.sum(y**2) - 1.0

        # Return (equality_constraints, None) since we have only equality constraint
        return jnp.array([sphere_constraint]), None

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # For N=5000: approximately 482.027
        return None  # Not specified exactly in SIF for this size

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
