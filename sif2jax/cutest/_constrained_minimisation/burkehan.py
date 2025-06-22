import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BURKEHAN(AbstractConstrainedMinimisation):
    """
    BURKEHAN problem.

    An infeasible problem.

    Source:
    J. V. Burke and S. P. Han,
    "A robust SQP method",
    Mathematical Programming 43 1989:277-303

    SIF input: Nick Gould, May 2008

    classification QOR2-AN-1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        return y[0]

    def y0(self):
        # Starting point
        return jnp.array([10.0])

    def args(self):
        return None

    def expected_result(self):
        # Infeasible problem - no feasible solution
        return None

    def expected_objective_value(self):
        # Infeasible problem - no feasible solution
        return None

    def bounds(self):
        # Variable bounds: -inf <= x <= 0
        lower = jnp.array([-jnp.inf])
        upper = jnp.array([0.0])
        return (lower, upper)

    def constraint(self, y):
        # No equality constraints
        eq_constraint = None

        # Inequality constraint: x^2 >= 1 (from x^2 - 1 >= 0)
        # Written in SIF as L (less than) constraint: x^2 <= -1
        # pycutest returns the raw value: x^2 - (-1) = x^2 + 1
        ineq_constraint = jnp.array([y[0] ** 2 + 1.0])

        return eq_constraint, ineq_constraint
