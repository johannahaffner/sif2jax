import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class MARATOS(AbstractConstrainedMinimisation):
    """
    MARATOS problem.

    The Maratos problem with penalty parameter = 0.000001

    Source:
    A.A. Brown and M. Bartholomew-Biggs,
    "Some effective methods for unconstrained optimization based on
    the solution of ordinary differential equations",
    Technical Report 178, Numerical Optimization Centre, Hatfield
    Polytechnic, (Hatfield, UK), 1987.

    SIF input: Nick Gould, June 1990.

    classification QQR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y
        tau = 0.000001
        return -x1 + tau * (x1**2 + x2**2)

    def y0(self):
        # Starting point
        return jnp.array([1.1, 0.1])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution value provided as 1.0 but commented out
        return None

    def num_variables(self):
        return 2

    def bounds(self):
        # All variables are free (unbounded)
        return jnp.array([-jnp.inf, -jnp.inf]), jnp.array([jnp.inf, jnp.inf])

    def constraint(self, y):
        x1, x2 = y

        # Equality constraints:
        # C: x1^2 + x2^2 = 1.0
        eq_constraint = jnp.array([x1**2 + x2**2 - 1.0])

        # Inequality constraints: none
        ineq_constraint = None

        return eq_constraint, ineq_constraint
