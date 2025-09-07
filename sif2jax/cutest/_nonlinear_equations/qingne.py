import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class QINGNE(AbstractNonlinearEquations):
    """SCIPY global optimization benchmark example Qing - Nonlinear equations version.

    This is a system of nonlinear equations:
    x_i^2 - i = 0 for i=1 to n

    The solution is x_i = sqrt(i) for all i.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/
    benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of QING.SIF

    SIF input: Nick Gould, Jan 2020

    classification: NOR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 100  # Default number of variables

    @property
    def m(self):
        """Number of constraints equals number of variables."""
        return self.n

    @property
    def y0(self):
        """Starting point from SIF file - all variables initialized to 1.0."""
        return jnp.ones(self.n)

    @property
    def bounds(self):
        """No explicit bounds for the nonlinear equations formulation."""
        return None

    @property
    def args(self):
        return ()

    def constraint(self, y):
        """Compute the nonlinear equations.

        The equations are x_i^2 - i = 0 for i=1 to n.
        Returns a tuple (equality_constraints, inequality_constraints).
        """
        # Create index array [1, 2, ..., n]
        indices = jnp.arange(1, self.n + 1, dtype=y.dtype)

        # Compute x_i^2 - i for all i
        equality_constraints = y * y - indices

        # No inequality constraints
        return equality_constraints, None

    @property
    def expected_result(self):
        """The solution is x_i = sqrt(i)."""
        indices = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        return jnp.sqrt(indices)

    @property
    def expected_constraint_value(self):
        """At the solution, all constraints should be zero."""
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        """For feasibility problems, objective is typically constant (zero or one)."""
        return jnp.array(0.0)
