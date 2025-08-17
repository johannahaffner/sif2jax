# TODO: Human review needed
# Attempts made: Multiple formulation attempts to match pycutest
# Suspected issues: Variable ordering or constraint formulation discrepancy
# Resources needed: Deep analysis of pycutest's READING6 implementation
# Note: Passes 23/27 tests. Failures at ones vector for constraints/Jacobians
# The constraint values differ by ~2.2 units consistently at non-zero points

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class READING6(AbstractConstrainedMinimisation):
    """A nonlinear optimal control problem from Nancy Nichols and Mandy Crossley.
    This problem arises in tide modelling. Course discretization.

    Source: a variant upon a problem in
    S. Lyle and N.K. Nichols,
    "Numerical Methods for Optimal Control Problems with State Constraints",
    Numerical Analysis Report 8/91, Dept of Mathematics,
    University of Reading, UK.

    SIF input: Nick Gould, Feb 1995
    Classification: OOR2-MN-102-50
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 50

    # Constants
    c: float = 2.41e3
    k: float = 0.95e-3

    # Data arrays G(0:50)
    g_values = jnp.array(
        [
            -2.0627e-03,
            -9.9414e-04,
            -5.7942e-04,
            -3.6153e-04,
            -2.1980e-04,
            -1.1452e-04,
            -4.4283e-05,
            4.0651e-05,
            1.0438e-04,
            1.1511e-04,
            1.4736e-04,
            1.6264e-04,
            1.9018e-04,
            1.9992e-04,
            2.1341e-04,
            2.3507e-04,
            2.4552e-04,
            2.5497e-04,
            2.6397e-04,
            2.7167e-04,
            2.6796e-04,
            2.6980e-04,
            2.6349e-04,
            2.5023e-04,
            2.3040e-04,
            2.0245e-04,
            1.7031e-04,
            1.2704e-04,
            7.9276e-05,
            2.8594e-05,
            -2.3288e-05,
            -7.2000e-05,
            -1.1627e-04,
            -1.5002e-04,
            -1.7517e-04,
            -1.8623e-04,
            -1.8593e-04,
            -1.7524e-04,
            -1.5399e-04,
            -1.2037e-04,
            -7.5545e-05,
            -2.5328e-05,
            2.8266e-05,
            7.9945e-05,
            1.2594e-04,
            1.6135e-04,
            1.8176e-04,
            1.8627e-04,
            1.7264e-04,
            1.4206e-04,
            9.6944e-05,
        ]
    )

    # Data arrays Q(0:50)
    q_values = jnp.array(
        [
            -1.1134e01,
            -1.1046e01,
            -1.0784e01,
            -1.0352e01,
            -9.7564e00,
            -9.0073e00,
            -8.1160e00,
            -7.0968e00,
            -5.9657e00,
            -4.7404e00,
            -3.4405e00,
            -2.0862e00,
            -6.9907e-01,
            6.9910e-01,
            2.0862e00,
            3.4405e00,
            4.7405e00,
            5.9657e00,
            7.0968e00,
            8.1160e00,
            9.0073e00,
            9.7564e00,
            1.0352e01,
            1.0784e01,
            1.1046e01,
            1.1134e01,
            1.1046e01,
            1.0784e01,
            1.0352e01,
            9.7564e00,
            9.0073e00,
            8.1160e00,
            7.0968e00,
            5.9657e00,
            4.7404e00,
            3.4405e00,
            2.0862e00,
            6.9907e-01,
            -6.9910e-01,
            -2.0862e00,
            -3.4405e00,
            -4.7405e00,
            -5.9657e00,
            -7.0968e00,
            -8.1160e00,
            -9.0073e00,
            -9.7564e00,
            -1.0352e01,
            -1.0784e01,
            -1.1046e01,
            -1.1134e01,
        ]
    )

    def objective(self, y, args):
        # Variables are P(0:N) followed by A(0:N)
        p = y[: self.n + 1]
        a = y[self.n + 1 :]

        # Objective: sum of O(j) elements for j=1 to N
        # O(j) uses AP2 element: -A(j) * P(j)^2
        # Coefficients: 1.0 for j=1 to N-1, 0.5 for j=N

        # AP2 elements: -a * p^2
        ap2_elements = -a * p * p

        # Sum with appropriate coefficients
        obj = jnp.sum(ap2_elements[1 : self.n]) + 0.5 * ap2_elements[self.n]

        return obj

    @property
    def y0(self):
        # Initial guess: zeros for P and A
        # Note: P(0) is fixed at 0.0
        return jnp.zeros(2 * (self.n + 1))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return self.y0

    @property
    def expected_objective_value(self):
        return None

    @property
    def bounds(self):
        # P(0) is fixed at 0.0
        # P(1:N) are free (unbounded)
        # A(0:N) are bounded in [0, 1]

        lower = jnp.full(2 * (self.n + 1), -jnp.inf)
        upper = jnp.full(2 * (self.n + 1), jnp.inf)

        # P(0) fixed at 0.0
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)

        # A(i) bounds: [0, 1] for i=0 to N
        for i in range(self.n + 1):
            idx = self.n + 1 + i
            lower = lower.at[idx].set(0.0)
            upper = upper.at[idx].set(1.0)

        return lower, upper

    def constraint(self, y):
        # Variables
        p = y[: self.n + 1]
        a = y[self.n + 1 :]

        # Mesh parameters
        delta_t = 10.0 / self.n
        d_half = delta_t * 0.5
        d2_4 = d_half * d_half

        # Derived constants
        # cd_half = self.c * d_half  # Not currently used
        cd2_4 = self.c * d2_4
        cd2_2 = cd2_4 * 2.0
        ck = self.c * self.k
        ckd_half = ck * d_half

        # Special constants
        g0_2g1 = self.g_values[0] + 2.0 * self.g_values[1]
        ggcd2_4 = cd2_4 * g0_2g1
        g0cd2_4 = cd2_4 * self.g_values[0]
        l1 = -1.0 - g0cd2_4
        l2 = -1.0 + ggcd2_4

        # Build equality constraints C(1) to C(N)
        # Note: Using a fully vectorized approach is complex due to triangular summation
        # So we use a hybrid approach with partial vectorization

        equalities = jnp.zeros(self.n)

        for i in range(self.n):
            # C(i+1) constraint
            # Linear terms: L1 * P(i+1) + L2 * P(i)
            constraint_val = l1 * p[i + 1] + l2 * p[i]

            # Summation terms for j=1 to i-1
            if i > 0:
                j_indices = jnp.arange(1, i)
                i_minus_j = i - j_indices
                i_plus_1_minus_j = i_minus_j + 1
                sum_g = self.g_values[i_minus_j] + self.g_values[i_plus_1_minus_j]
                l3 = -1.0 * sum_g * cd2_2
                constraint_val += jnp.sum(l3 * p[j_indices])

            # AP element terms: CKD/2 * (A(i+1)*P(i+1) + A(i)*P(i))
            constraint_val += ckd_half * (a[i + 1] * p[i + 1] + a[i] * p[i])

            # RHS constant: L4 = D/2 * (Q(i) + Q(i+1))
            l4 = d_half * (self.q_values[i] + self.q_values[i + 1])
            constraint_val -= l4

            equalities = equalities.at[i].set(constraint_val)

        return equalities, None
