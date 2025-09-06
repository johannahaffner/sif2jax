import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GIGOMEZ3(AbstractConstrainedMinimisation):
    """
    GIGOMEZ3 - A nonlinear minmax problem.

    Source:
    C. Gigola and S. Gomez,
    "A Regularization Method for Solving the Finite Convex Min-Max Problem",
    SINUM 27(6), pp. 1621-1634, 1990.

    Classification: LOR2-AY-3-3

    SIF input: Ph. Toint, August 1993.
    """

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess from SIF file."""
        return jnp.array([2.0, 2.0, 2.0])

    @property
    def bounds(self):
        """Variable bounds - all variables are free."""
        return None

    def objective(self, y, args):
        """
        Objective function: minimize Z (third variable).
        From SIF: XN OBJ Z 1.0
        """
        del args
        x1, x2, z = y
        return z

    def constraint(self, y):
        """
        Constraint functions for the minmax problem.

        From SIF file:
        C1: Z >= -X1^4 - X2^2
        C2: Z >= -(2-X1)^2 - (2-X2)^2
        C3: Z >= -2*exp(X2-X1)
        """
        x1, x2, z = y

        # From SIF file constraint formulation:
        # XG C1: Z >= 0, E C1: X1FR (-1) + X2SQ (-1) → Z - X1^4 - X2^2 >= 0
        # XG C2: Z >= 0, E C2: SX1 (-1) + SX2 (-1) → Z - (2-X1)^2 - (2-X2)^2 >= 0
        # XG C3: Z >= 0, E C3: EEXP (-2) → Z - 2*exp(X2-X1) >= 0
        c1 = z - x1**4 - x2**2  # C1
        c2 = z - (2.0 - x1) ** 2 - (2.0 - x2) ** 2  # C2
        c3 = z - 2.0 * jnp.exp(x2 - x1)  # C3

        # Return (None, inequality_constraints) like other GIGOMEZ problems
        return None, jnp.array([c1, c2, c3])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(2.0)

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
