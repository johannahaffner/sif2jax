import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ZECEVIC2(AbstractConstrainedMinimisation):
    """ZECEVIC2 problem - Problem 2 from Zecevic.

    Source: problem 2 in A. Zecevic, "Contribution to methods
    of external penalty functions - algorithm MNBS"
    Advanced Business School, Belgrade, (whatever is left of) Yugoslavia.

    SIF input: Nick Gould, April 1993.

    Classification: QLR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x1, x2 = y

        # OBJ = -2*x1 - 3*x2 + 2*x2^2
        return -2.0 * x1 - 3.0 * x2 + 2.0 * x2**2

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2 = y

        # No equality constraints
        equality_constraints = None

        # Inequality constraints (pycutest convention: <= 0)
        # CON1: x1 + x2 <= 2  =>  x1 + x2 - 2 <= 0
        con1 = x1 + x2 - 2.0

        # CON2: x1 + 4*x2 <= 4  =>  x1 + 4*x2 - 4 <= 0
        con2 = x1 + 4.0 * x2 - 4.0

        inequality_constraints = jnp.array([con1, con2])

        return equality_constraints, inequality_constraints

    def y0(self):
        """Initial guess."""
        return jnp.array([0.1, -0.1])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Bounds on variables."""
        return jnp.array([[0.0, 10.0], [0.0, 10.0]])

    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(-4.125)
