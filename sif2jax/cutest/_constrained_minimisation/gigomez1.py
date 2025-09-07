import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GIGOMEZ1(AbstractConstrainedMinimisation):
    """
    GIGOMEZ1 - A nonlinear minmax problem.

    Source:
    C. Gigola and S. Gomez,
    "A Regularization Method for Solving the Finite Convex Min-Max Problem",
    SINUM 27(6), pp. 1621-1634, 1990.

    Classification: LQR2-AN-3-3

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
        C1: Z >= 5*X1 - X2
        C2: Z >= -X1^2 - X2^2 - 4*X2
        C3: Z >= -5*X1 - X2
        """
        x1, x2, z = y

        # From SIF file constraint formulation:
        # XG C1: Z + 5*X1 - X2 >= 0 → c1 = Z + 5*X1 - X2
        # XG C2: Z - 4*X2 - X1^2 - X2^2 >= 0 → c2 = Z - 4*X2 - X1^2 - X2^2
        # XG C3: Z - 5*X1 - X2 >= 0 → c3 = Z - 5*X1 - X2
        c1 = z + 5.0 * x1 - x2  # C1
        c2 = z - 4.0 * x2 - x1**2 - x2**2  # C2 with quadratic elements
        c3 = z - 5.0 * x1 - x2  # C3

        # Return (None, inequality_constraints) like GIGOMEZ2
        return None, jnp.array([c1, c2, c3])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(-3.0)

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
