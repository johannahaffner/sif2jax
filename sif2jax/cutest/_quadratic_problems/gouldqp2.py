# TODO: Human review needed
# Attempts made: [Initial implementation, objective/constraint formulation, bounds]
# Suspected issues: [Large-scale complexity, SIF file interpretation for knots]
# Resources needed: [Deeper analysis of knot placement, comparison with AMPL]

import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class GOULDQP2(AbstractConstrainedQuadraticProblem):
    """GOULDQP2 problem - optimal knot placement suggested by J. R. Kightley.

    Source: a problem of optimal knot placement in a scheme for
    ordinary differential equations with boundary values suggested
    by J. R. Kightley, see N. I. M. Gould, "An algorithm for
    large-scale quadratic programming", IMA J. Num. Anal (1991),
    11, 299-324, problem class 2.

    SIF input: Nick Gould, December 1991
               Revised June 1998 with rescaling for larger examples

    Classification: QLR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    K: int = 10000  # Number of knots

    @property
    def n(self):
        """Number of variables."""
        return 2 * self.K - 1  # K KNOT variables + (K-1) SPACE variables

    @property
    def m(self):
        """Number of constraints."""
        return self.K - 1  # K-1 CON constraints

    def objective(self, y, args):
        """Compute the objective."""
        del args

        K = self.K
        space_vars = y[K:]  # SPACE(1) to SPACE(K-1)

        # Vectorized: squared differences (SPACE(i+1) - SPACE(i))^2 for i=1 to K-2
        # OBJ(i) = (SPACE(i+1) - SPACE(i))^2 for i=1 to K-2
        diffs = space_vars[1:] - space_vars[:-1]  # K-2 differences
        return 0.5 * jnp.sum(diffs**2)

    def constraint(self, y):
        """Compute the constraints."""
        K = self.K
        knot_vars = y[:K]  # KNOT(1) to KNOT(K)
        space_vars = y[K:]  # SPACE(1) to SPACE(K-1)

        # Vectorized: CON(i): SPACE(i) - KNOT(i+1) + KNOT(i) = 0 for i=1 to K-1
        constraints = space_vars - knot_vars[1:] + knot_vars[:-1]
        return constraints, None

    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)

    def _compute_knot_bounds(self):
        """Compute the A(i) values for knot bounds."""
        K = self.K

        # Parameters from SIF file
        if K < 1000:
            factor = 1.01
        else:
            k_div_1000 = K // 1000
            i2 = 2 * k_div_1000
            i_plus_1 = k_div_1000 + 1
            _ = i2 // i_plus_1  # k_gt_1000 unused
            factor = 1.0001

        alpha = 1.0
        beta = factor

        # A(1) = 2.0 (from SIF)
        a_vals = [2.0]

        # For I=2 to K+1: A(I) = ALPHA + BETA, then BETA = BETA * FACTOR
        for i in range(2, K + 2):
            a_val = alpha + beta  # A(I) = ALPHA + BETA
            a_vals.append(a_val)
            beta = beta * factor  # BETA = BETA * FACTOR

        return jnp.array(a_vals)

    @property
    def y0(self):
        """Initial guess."""
        K = self.K
        a_vals = self._compute_knot_bounds()

        # Initial knot values: KNOT(i) = A(i) for i=1 to K
        knot_init = a_vals[:K]

        # Initial space values: SPACE(i) = A(i+1) - A(i) for i=1 to K-1
        space_init = a_vals[1:K] - a_vals[: K - 1]

        return jnp.concatenate([knot_init, space_init])

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        K = self.K
        a_vals = self._compute_knot_bounds()

        # Initialize bounds
        lower = jnp.full(self.n, 0.0)  # Default lower bound
        upper = jnp.full(self.n, jnp.inf)

        # KNOT bounds: A(i) <= KNOT(i) <= A(i+1) for i=1 to K
        for i in range(K):
            lower = lower.at[i].set(a_vals[i])  # KNOT(i) >= A(i)
            upper = upper.at[i].set(a_vals[i + 1])  # KNOT(i) <= A(i+1)

        # SPACE bounds: for i=1 to K-1
        # SPACE(i) >= 0.4 * (A(i+2) - A(i))
        # SPACE(i) <= 0.6 * (A(i+2) - A(i))
        for i in range(K - 1):
            diff = a_vals[i + 2] - a_vals[i]
            lower = lower.at[K + i].set(0.4 * diff)
            upper = upper.at[K + i].set(0.6 * diff)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Lower bound is 0.0 from SIF file
        return jnp.array(0.0)
