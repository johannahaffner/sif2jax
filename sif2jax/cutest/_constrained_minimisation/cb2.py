import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CB2(AbstractConstrainedMinimisation):
    """
    CB2 problem.

    A nonlinear minmax problem.

    Source:
    R. Wommersley and R. Fletcher,
    "An algorithm for composite nonsmooth optimization problems"
    JOTA, vol.48, pp.493-523, 1986

    SIF input: Ph. Toint, April 1992.

    classification LOR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, u = y
        # Linear objective: minimize u
        return u

    def y0(self):
        # Starting point
        return jnp.array([2.0, 2.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # No expected result given in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(0.0)

    def bounds(self):
        # No variable bounds
        return None

    def constraint(self, y):
        x1, x2, u = y

        # No equality constraints
        eq_constraint = None

        # Three inequality constraints (G-type: >= 0)
        # C1: u - x1^2 - x2^4 >= 0
        # C2: u - (2-x1)^2 - (2-x2)^2 >= 0
        # C3: u - 2*exp(x2-x1) >= 0
        # In pycutest format (raw values for G-type)
        ineq_constraint = jnp.array(
            [
                u - x1**2 - x2**4,  # C1
                u - (2.0 - x1) ** 2 - (2.0 - x2) ** 2,  # C2
                u - 2.0 * jnp.exp(x2 - x1),  # C3
            ]
        )

        return eq_constraint, ineq_constraint
