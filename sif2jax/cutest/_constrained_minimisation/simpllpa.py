import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class SIMPLLPA(AbstractConstrainedMinimisation):
    """A simple linear programming problem in 2 variables.

    Source:
    N. Gould, private communication.

    SIF input: N. Gould, Dec 1989.

    classification: LLR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables
    m_eq: int = 0  # no equality constraints
    m_ineq: int = 2  # 2 inequality constraints

    @property
    def y0(self):
        # Both variables start at 0.1
        return jnp.array([0.1, 0.1])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Linear objective function: 2*x1 + x2."""
        x1, x2 = y
        return 2.0 * x1 + x2

    @property
    def bounds(self):
        """Standard LP non-negativity bounds: x >= 0."""
        # Both variables are non-negative
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def constraint(self, y):
        """Returns the constraints on the variable y.

        Inequality constraints (g(x) >= 0):
        - x1 + x2 >= 1.0  =>  x1 + x2 - 1.0 >= 0
        - x1 + 2*x2 >= 1.5  =>  x1 + 2*x2 - 1.5 >= 0
        """
        x1, x2 = y

        # No equality constraints
        eq_constraints = None

        # Inequality constraints (must be >= 0)
        g1 = x1 + x2 - 1.0
        g2 = x1 + 2.0 * x2 - 1.5
        ineq_constraints = jnp.array([g1, g2])

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(1.0)
