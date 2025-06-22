import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ZECEVIC3(AbstractConstrainedMinimisation):
    """ZECEVIC3 problem - Problem 3 from Zecevic.

    Source: problem 3 in A. Zecevic, "Contribution to methods
    of external penalty functions - algorithm MNBS"
    Advanced Business School, Belgrade, (whatever is left of) Yugoslavia.

    SIF input: Nick Gould, April 1993.

    Classification: QQR2-AN-2-2
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

        # OBJ = -84*x1 - 24*x2 + 7*x1^2 + 3*x2^2 + 300
        # Note: pycutest seems to add the constant instead of subtracting
        return -84.0 * x1 - 24.0 * x2 + 7.0 * x1**2 + 3.0 * x2**2 + 300.0

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2 = y

        # No equality constraints
        equality_constraints = None

        # Inequality constraints (pycutest convention: <= 0)
        # CON1: -x1*x2 <= -1  =>  -x1*x2 + 1 <= 0
        con1 = -x1 * x2 + 1.0

        # CON2: x1^2 + x2^2 <= 9  =>  x1^2 + x2^2 - 9 <= 0
        con2 = x1**2 + x2**2 - 9.0

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
        return jnp.array(-97.30952)
