import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class NOBNDTOR(AbstractBoundedMinimisation):
    """The quadratic elastic torsion problem.

    The problem comes from the obstacle problem on a square.

    The square is discretized into (px-1)(py-1) little squares. The
    heights of the considered surface above the corners of these little
    squares are the problem variables. There are px**2 of them.

    The dimension of the problem is specified by Q, which is half the
    number discretization points along one of the coordinate
    direction. Since the number of variables is P**2, it is given by 4Q**2.

    Source: problem 1 (c=5, starting point U = upper bound) in
    J.J. More',
    "A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: Ph. Toint, Dec 1989.

    A variant of TORSION1 in which some of the variables are unconstrained.

    classification QBR2-AY-V-0

    TODO: Human review needed - complex bounds logic
    The SIF file has complex region-specific bounds with some variables
    marked as unconstrained (1.0D+21) in specific patterns.
    The current implementation doesn't match the exact bound pattern.
    """

    n: int = 5476  # Default for Q=37, n = P^2 = (2*Q)^2 = 74^2 = 5476
    Q: int = 37  # Half the number of discretization points
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def P(self):
        """Number of discretization points along each axis."""
        return 2 * self.Q

    @property
    def y0(self):
        """Initial point - upper bounds for interior points, 0 on boundary."""
        P = self.P
        h = 1.0 / (P - 1)
        y = jnp.zeros((P, P))

        # Set interior points to upper bounds
        for i in range(1, P - 1):
            for j in range(1, P - 1):
                # Distance to boundary
                dist_to_boundary = min(i, j, P - 1 - i, P - 1 - j)
                upper_bound = dist_to_boundary * h
                y = y.at[i, j].set(upper_bound)

        return y.flatten()

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Compute bounds for the problem.

        Variables on the boundary are fixed to 0.
        Interior variables have bounds based on distance to boundary.
        SIF file uses 1.0D+21 for unconstrained, which we treat as inf.
        """
        P = self.P
        Q = self.Q
        h = 1.0 / (P - 1)

        lower = jnp.zeros((P, P))
        upper = jnp.zeros((P, P))

        # Fix boundary variables to 0 (already initialized to 0)

        # Set bounds for interior points
        # The SIF file has complex logic for bounds based on position in the square
        # Variables bounded by distance * h, except those marked with 1.0D+21

        for i in range(1, P - 1):
            for j in range(1, P - 1):
                # Default to unconstrained for interior points
                lower = lower.at[i, j].set(-jnp.inf)
                upper = upper.at[i, j].set(jnp.inf)

                # Apply specific bounds based on position
                # Lower half of square (i <= Q)
                if i <= Q:
                    if j < i:
                        # Region with (j-1)*h bounds
                        bound = (j - 1) * h
                        if bound < 1e20:  # Not marked as unconstrained
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)
                    elif j >= i and j <= P - i:
                        # Region with (i-1)*h bounds
                        bound = (i - 1) * h
                        if bound < 1e20:
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)
                    elif j > P - i:
                        # Region with (P-j)*h bounds
                        bound = (P - j) * h
                        if bound < 1e20:
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)
                # Upper half of square (i > Q)
                else:
                    if j <= P - i:
                        # Region with (j-1)*h bounds
                        bound = (j - 1) * h
                        if bound < 1e20:
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)
                    elif j > P - i and j < i:
                        # Region with (P-i)*h bounds
                        bound = (P - i - 1) * h
                        if bound < 1e20:
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)
                    elif j >= i:
                        # Region with (P-j)*h bounds
                        bound = (P - j) * h
                        if bound < 1e20:
                            lower = lower.at[i, j].set(-bound)
                            upper = upper.at[i, j].set(bound)

        return lower.flatten(), upper.flatten()

    def objective(self, y, args):
        """Quadratic elastic torsion objective - vectorized.

        The objective is:
        sum_{i,j interior} c0 * x[i,j] + 0.25 * sum of squared differences
        where c0 = -h^2 * c
        """
        del args
        P = self.P
        h = 1.0 / (P - 1)
        h2 = h * h
        c = 5.0
        c0 = -h2 * c

        # Reshape to 2D grid
        x = y.reshape((P, P))

        obj = 0.0

        # Linear terms for interior nodes
        interior_mask = jnp.zeros((P, P))
        interior_mask = interior_mask.at[1 : P - 1, 1 : P - 1].set(1.0)
        obj += c0 * jnp.sum(x * interior_mask)

        # Quadratic terms - squared differences with neighbors
        # Each interior node (i,j) has 4 elements:
        # A: (x[i+1,j] - x[i,j])^2
        # B: (x[i,j+1] - x[i,j])^2
        # C: (x[i-1,j] - x[i,j])^2
        # D: (x[i,j-1] - x[i,j])^2

        # Use slicing to compute all differences at once
        # Right neighbors (A elements)
        diff_right = x[2:P, 1 : P - 1] - x[1 : P - 1, 1 : P - 1]
        obj += 0.25 * jnp.sum(diff_right**2)

        # Top neighbors (B elements)
        diff_top = x[1 : P - 1, 2:P] - x[1 : P - 1, 1 : P - 1]
        obj += 0.25 * jnp.sum(diff_top**2)

        # Left neighbors (C elements)
        diff_left = x[0 : P - 2, 1 : P - 1] - x[1 : P - 1, 1 : P - 1]
        obj += 0.25 * jnp.sum(diff_left**2)

        # Bottom neighbors (D elements)
        diff_bottom = x[1 : P - 1, 0 : P - 2] - x[1 : P - 1, 1 : P - 1]
        obj += 0.25 * jnp.sum(diff_bottom**2)

        return obj

    @property
    def expected_result(self):
        """Solution not explicitly provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value varies with Q."""
        # From SIF file comments:
        # Q=2: -5.1851852D-1
        # Q=5: -4.9234185D-1
        # Q=11: -4.5608771D-1
        if self.Q == 2:
            return jnp.array(-0.51851852)
        elif self.Q == 5:
            return jnp.array(-0.49234185)
        elif self.Q == 11:
            return jnp.array(-0.45608771)
        else:
            return None
