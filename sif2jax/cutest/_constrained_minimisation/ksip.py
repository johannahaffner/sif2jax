import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class KSIP(AbstractConstrainedMinimisation):
    """Source: problem from Kortanek and No
    The problem is really a semi-infinite QP
    to appear in SIAM J. Optimization.

    The structure is the following:
    min  "Sum"{ Xj^2/(2j) + Xj/j  ;  j=1,...,n }   subject to
    "Sum"{ t^(j-1)*Xj } ; j=1,...,n  >=  b(t) for all t in [0 1].

    Four examples are considered for n = 20, corresponding to the RHS
    function, b(t) : sin(t), 1/(2-t), exp(t), and tan(t).

    The interval [0 1] is dicretized via steps of 1/1000

    SIF input: A.R. Conn, May 1993

    classification QLR2-AN-20-1001
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    N = 20  # Number of variables
    M = 1000  # Number of discretization points

    def objective(self, y: Array, args) -> Array:
        """Quadratic objective: sum of Xj^2/(2j) + Xj/j for j=1,...,N"""
        j_values = jnp.arange(1, self.N + 1)
        terms = y**2 / (2.0 * j_values) + y / j_values
        return jnp.sum(terms)

    def constraint(self, y: Array):
        """Semi-infinite constraints discretized at 1000 points.

        For each t in [0, 1] (discretized), we have:
        sum_{j=1}^N t^(j-1) * X_j >= sin(t)

        Converted to inequality constraints <= 0:
        sin(t) - sum_{j=1}^N t^(j-1) * X_j <= 0
        """
        # Generate discretization points: t = i/M for i = 0, 1, ..., M
        t_values = jnp.arange(self.M + 1) / self.M

        constraints = []
        for t in t_values:
            # Compute sum_{j=1}^N t^(j-1) * X_j
            constraint_sum = 0.0
            for j in range(1, self.N + 1):
                xj = y[j - 1]
                constraint_sum += (t ** (j - 1)) * xj

            # sin(t) - sum_{j=1}^N t^(j-1) * X_j <= 0
            constraint_val = jnp.sin(t) - constraint_sum
            constraints.append(constraint_val)

        # No equality constraints
        equality_constraints = jnp.array([])

        # All constraints are inequalities
        inequality_constraints = jnp.array(constraints)

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess for the optimization problem."""
        # From SIF: all variables start at 2.0
        return jnp.full(self.N, 2.0)

    @property
    def bounds(self):
        """Variable bounds. All variables are free."""
        # From SIF: FR KSIP 'DEFAULT' means all variables are free
        return None

    @property
    def args(self):
        """Additional arguments for the objective function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # This is a complex semi-infinite programming problem
        # The exact solution is not straightforward to compute
        return jnp.zeros(self.N)

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # From SIF comments, multiple solutions are mentioned
        # Using the first one as a reference
        return jnp.array(0.57579024357147)
