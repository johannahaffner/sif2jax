import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class S308NE(AbstractNonlinearEquations):
    """S308NE problem.

    Nonlinear-equations version of problem 308 in
    K. Schittkowski, "More Test Problems for Nonlinear Programming Codes",
    Springer Verlag, Berlin, 1987.

    SIF input: Ph. Toint, April 1991.
    Nonlinear-equations version of S308.SIF, Nick Gould, Jan 2020.

    Classification: NOR2-AN-2-3

    This is a system of 3 nonlinear equations in 2 variables.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def m(self):
        """Number of equations."""
        return 3

    def constraint(self, y):
        """Compute the system of nonlinear equations."""
        x1, x2 = y[0], y[1]

        # From the SIF file:
        # O1: QUAD element = x1*x1 + x1*x2 + x2*x2
        # O2: SINE element = sin(x1)
        # O3: COSN element = cos(x2)
        eq1 = x1 * x1 + x1 * x2 + x2 * x2
        eq2 = jnp.sin(x1)
        eq3 = jnp.cos(x2)

        equations = inexact_asarray(jnp.array([eq1, eq2, eq3]))

        return equations, None

    def equality_constraints(self):
        """All equations are equalities."""
        return jnp.ones(self.m, dtype=bool)

    @property
    def y0(self):
        """Initial guess from SIF file."""
        return inexact_asarray(jnp.array([3.0, 0.1]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No bounds specified (FR = free)."""
        return None

    @property
    def expected_result(self):
        """Expected result not explicitly provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # For nonlinear equations, the objective is typically ||F(x)||^2
        # The SIF file mentions solution value 0.773199 (commented out)
        return inexact_asarray(jnp.array(0.773199))
