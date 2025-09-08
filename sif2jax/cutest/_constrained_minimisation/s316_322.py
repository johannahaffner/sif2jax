import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class S316_322(AbstractConstrainedMinimisation):
    """S316-322 problem.

    Problems 316 to 322 in K. Schittkowski,
    "More Test Problems for Nonlinear Programming Codes",
    Springer Verlag, Berlin, 1987.

    SIF input: Ph. Toint, April 1991.

    Classification: QQR2-AN-2-1

    This is a constrained optimization problem with 2 variables and 1 constraint.
    The problem can be parameterized by different denominator values:
    - Problem 316: DEN=100.0
    - Problem 317: DEN=64.0
    - Problem 318: DEN=36.0
    - Problem 319: DEN=16.0
    - Problem 320: DEN=4.0
    - Problem 321: DEN=1.0
    - Problem 322: DEN=0.01
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Default to problem 316
    denominator: float = 100.0

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    @property
    def m(self):
        """Number of constraints."""
        return jnp.array(1)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x1, x2 = y[0], y[1]

        # From SIF file:
        # OBJ1: L2 group type with x1 coefficient 1.0 and constant 20.0
        # OBJ2: L2 group type with x2 coefficient 1.0 and constant -20.0
        # L2 group type: GVAR * GVAR where GVAR is the linear combination
        # Looking at the gradient signs, it seems the objective should be:
        # (x1 - 20)^2 + (x2 + 20)^2

        obj1 = (x1 - 20.0) * (x1 - 20.0)  # L2 group: (x1 - 20)^2
        obj2 = (x2 + 20.0) * (x2 + 20.0)  # L2 group: (x2 + 20)^2

        return obj1 + obj2

    def constraint(self, y):
        """Compute the equality constraint."""
        x1, x2 = y[0], y[1]

        # From SIF file:
        # CON: 0.01*x1^2 + (1/DEN)*x2^2 = 1.0
        scal = 1.0 / self.denominator
        eq_constraint = 0.01 * x1 * x1 + scal * x2 * x2 - 1.0

        return inexact_asarray(jnp.array([eq_constraint])), None

    def equality_constraints(self):
        """The constraint is an equality."""
        return jnp.ones(self.m, dtype=bool)

    @property
    def y0(self):
        """Initial guess - not specified in SIF, use zeros."""
        return inexact_asarray(jnp.zeros(self.n))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No bounds specified (FR = free)."""
        return None

    @property
    def name(self):
        """Problem name for pycutest."""
        return "S316-322"

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file (for problem 316)."""
        if self.denominator == 100.0:
            return inexact_asarray(jnp.array(334.315))
        elif self.denominator == 64.0:
            return inexact_asarray(jnp.array(372.467))
        elif self.denominator == 36.0:
            return inexact_asarray(jnp.array(412.750))
        elif self.denominator == 16.0:
            return inexact_asarray(jnp.array(452.404))
        elif self.denominator == 4.0:
            return inexact_asarray(jnp.array(485.531))
        elif self.denominator == 1.0:
            return inexact_asarray(jnp.array(496.112))
        elif self.denominator == 0.01:
            return inexact_asarray(jnp.array(499.960))
        else:
            return None
