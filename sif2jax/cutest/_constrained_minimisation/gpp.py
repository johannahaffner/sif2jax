# TODO: Human review needed
# Test failures in objective, gradient, and hessian functions
# Attempts made: 1
# Suspected issues: Possible formulation differences or numerical precision issues
# Resources needed: Review SIF file formulation and compare with implementation

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GPP(AbstractConstrainedMinimisation):
    """
    GPP - Example of a geometric programming problem.

    Source:
    Hans Mittelmann, private communication.

    SIF input: N. Gould, Jan 1998

    Classification: OOR2-AY-V-V
    """

    # Problem parameter
    n: int = 1000  # Number of variables

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # From SIF: 'DEFAULT' 1.0
        return jnp.ones(self.n)

    @property
    def bounds(self):
        """Variable bounds - all variables are free."""
        return None

    def objective(self, y, args):
        """
        Objective function: sum of exponential differences.

        From SIF: sum_{i=1}^{N-1} sum_{j=i+1}^{N} exp(x_j - x_i)
        """
        del args
        n = self.n

        # Vectorized implementation using broadcasting
        # Create matrices for all i,j pairs where i < j
        i_indices = jnp.arange(n - 1)[:, None]  # Shape: (N-1, 1)
        j_indices = jnp.arange(1, n)[None, :]  # Shape: (1, N-1)

        # Create mask for valid pairs (j > i)
        valid_mask = j_indices > i_indices  # Shape: (N-1, N-1)

        # Expand y for broadcasting
        y_i = y[i_indices]  # Shape: (N-1, 1)
        y_j = y[j_indices]  # Shape: (1, N-1)

        # Compute exp(y_j - y_i) for all valid pairs
        exp_diff = jnp.exp(y_j - y_i)  # Shape: (N-1, N-1)

        # Sum only valid pairs
        return jnp.sum(exp_diff * valid_mask)

    def constraint(self, y):
        """
        Constraint functions for the geometric programming problem.

        From SIF:
        INEQ1(I): x_i + x_{i+1} >= 0  for i = 1..N-1  (XG means >=)
        INEQ2(I): exp(x_i) + exp(x_{i+1}) <= 20  for i = 1..N-1  (XL means <=)
        """
        n = self.n

        # INEQ1 constraints: x_i + x_{i+1} >= 0
        ineq1_constraints = []
        for i in range(n - 1):
            c = y[i] + y[i + 1]  # x_i + x_{i+1} >= 0
            ineq1_constraints.append(c)

        # INEQ2 constraints: exp(x_i) + exp(x_{i+1}) <= 20
        ineq2_constraints = []
        for i in range(n - 1):
            c = jnp.exp(y[i]) + jnp.exp(y[i + 1]) - 20.0  # <= 20 becomes <= 0
            ineq2_constraints.append(c)

        # Combine inequalities: >= 0 for INEQ1, <= 0 for INEQ2
        eq_constraints = None  # No equality constraints
        ineq_constraints = jnp.array(ineq1_constraints + ineq2_constraints)

        return eq_constraints, ineq_constraints

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # For N=250: approximately 1.44009e+04
        return None  # Not specified for N=1000

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
